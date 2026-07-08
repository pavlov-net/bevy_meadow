//! GPU-driven cull + compact compute pass for the meadow renderer.
//!
//! Per frame, for every (active patch, view) pair, the compute kernel
//! `cull_and_compact` (in `meadow_compute.wgsl`) derives each blade,
//! samples the heightfield once, applies the per-view LOD / density /
//! frustum gates, and atomically appends survivors into that view's
//! contiguous region of a per-variant `blades` buffer. A tiny
//! `write_instance_counts` kernel copies each view's survivor count
//! into its `DrawIndexedIndirectArgs.instance_count`. The raster side
//! (`render.rs::DrawMeadowPatch`) then issues exactly one
//! `draw_indexed_indirect` per view, reading `indirect[slot]`.
//!
//! Streaming-correctness: patches are spawned per-loaded-chunk, but the
//! `patches` storage buffer holds *all* placements. The compute reads
//! `active_patches[wg.x]` — a per-variant list of the `patch_index`
//! values of currently-live `MeadowPatch` entities — so off-chunk
//! patches are never processed. `active_count` drives the dispatch X
//! dim.
//!
//! All resources here live in the render world. Pipeline init mirrors
//! `examples/shader/compute_shader_game_of_life.rs` (RenderStartup +
//! `RenderGraph`-schedule node `.before(camera_driver)`).

use bevy::camera::primitives::Frustum;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::ecs::resource::Resource;
use bevy::light::{CascadeShadowConfig, DirectionalLight};
use bevy::pbr::LightEntity;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
    uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
    BufferDescriptor, BufferId, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
    ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType, StorageBuffer,
    TextureSampleType, TextureViewId, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::sync_world::RenderEntity;
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, RetainedViewEntity};
use bevy::render::{Extract, ExtractSchedule, Render, RenderStartup, RenderSystems};
use std::borrow::Cow;

use crate::material::{MeadowMaterial, VariantParams};
use crate::mesh::{
    BLADE_RECORD_SIZE, INDIRECT_ARGS_SIZE, MEADOW_MAX_BANDS, MEADOW_MAX_VIEWS, MESH_TASK_BLADES,
    SHADOW_MAX_DIST,
};
use crate::plugin::{MeadowPatch, MeadowVariantId, MeadowVariantRegistry, MeadowViewer};
use crate::render::{MeadowRenderDriver, RenderMeadowMeshIds};

const MEADOW_COMPUTE_SHADER: &str = "embedded://bevy_meadow/meadow_compute.wgsl";

/// Flag bit 0 in `MeadowViewCull.params.x` marking a shadow (cascade)
/// view. Mirror of `MEADOW_VIEW_FLAG_SHADOW` in `meadow_compute.wgsl`.
pub(crate) const MEADOW_VIEW_FLAG_SHADOW: u32 = 1;

/// Blade-capacity rounding granularity (512Ki blades). Active footprints
/// are rounded up to this and buffers are grown-only, so chunk crossings
/// don't thrash buffer reallocation.
const CAP_GRAN: u32 = 1 << 19;

fn round_cap(n: u32) -> u32 {
    n.max(1).div_ceil(CAP_GRAN) * CAP_GRAN
}

// ---------- GPU-uploaded per-view culling data ----------

/// Rust mirror of `MeadowViewCull` (`meadow_compute.wgsl`). `frustum`
/// holds the 6 world-space half-space planes (normal.xyz, d).
/// `params  = (flags, lod_max, base0, cap0)` — band 0 (near blade) region;
/// flags bit 0 = shadow view.
/// `params2 = (base1, cap1, shadow_near, _)` — band 1 (far tuft) region,
/// plus the per-cascade near radial clip for shadow views.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct MeadowViewCull {
    pub frustum: [Vec4; 6],
    pub params: Vec4,
    pub params2: Vec4,
}

/// Rust mirror of `MeadowViewCullData`. `count` = number of active
/// views; `views[slot]` is per-view. `base_offset`/`capacity` in each
/// view's `params` are per-(variant, view), so this whole struct is
/// rebuilt per variant (frusta + flags + lod_max are shared, base/cap
/// are filled per variant).
#[derive(ShaderType, Clone, Default)]
pub struct MeadowViewCullData {
    pub count: u32,
    pub views: [MeadowViewCull; MEADOW_MAX_VIEWS],
}

/// Mesh-shader path state. Always present (and always false) when the
/// `mesh-shaders` feature is off — the compute path then behaves
/// byte-identically to before. Read by the compute node, the buffer prep
/// (per-view cap collapse + task work-list upload), and
/// `DrawMeadowPatch` (view skip).
#[derive(Resource, Default, Clone, Copy)]
pub struct MeadowMeshPathActive {
    /// The mesh path draws every meadow view this frame (all-or-nothing;
    /// the compute cull/compact pass has nothing to produce). Written per
    /// frame by `mesh_path::decide_meadow_mesh_path`.
    pub active: bool,
    /// The device supports the mesh path (set once at startup). Gates
    /// work the compute path prepares solely for the mesh path (the task
    /// work list), which must stay warm across path switches.
    pub available: bool,
}

// ---------- view-slot mapping ----------

/// Frame-stable mapping `RetainedViewEntity -> dense slot [0, N)`.
/// Slot 0 = main 3D camera; slots 1.. = directional cascades in
/// `(light_entity, cascade_index)` order. Read by `DrawMeadowPatch`
/// (to pick the per-view indirect offset) and by the per-variant
/// view-cull fill. Rebuilt each frame.
#[derive(Resource, Default)]
pub struct MeadowViewSlots {
    /// Shared per-view cull data WITHOUT per-variant base/cap filled.
    /// Holds the frusta + flags + lod_max for every active view.
    pub shared: MeadowViewCullData,
    /// `RetainedViewEntity -> slot`.
    pub by_retained: HashMap<RetainedViewEntity, u32>,
    pub count: u32,
}

/// Per-cascade radial bounds `(near, far)` in metres from the camera,
/// indexed by `cascade_index`, derived from the DOMINANT directional
/// light's `CascadeShadowConfig` (highest-illuminance shadow caster; see
/// `extract_meadow_cascade_bounds`). `build_meadow_view_slots` uses
/// these to clip each shadow cascade's grass cull to its own distance
/// slice — so a blade rasterizes into ~1 cascade instead of all 4 —
/// while their union still covers `[0, SHADOW_MAX_DIST]`, so no receiver
/// loses its grass shadow.
#[derive(Resource, Default)]
pub struct MeadowCascadeBounds {
    pub per_cascade: Vec<(f32, f32)>,
    /// Render-world entity bits of the DOMINANT shadow-casting
    /// directional light — the single light whose cascades grass serves
    /// (see `extract_meadow_cascade_bounds`). `build_meadow_view_slots`
    /// filters cascade views against this, so the light choice and the
    /// cascade schedule can't disagree.
    pub dominant_light_bits: Option<u64>,
}

// ---------- per-variant GPU buffer set ----------

/// Per-variant GPU buffers driving the compute + indirect-draw path.
/// Sized off a LOD-weighted survivor estimate of the active footprint,
/// rounded up to `CAP_GRAN` and grown-only so chunk crossings rarely
/// reallocate.
pub struct VariantGpuBuffers {
    /// Compacted survivor records, one contiguous region per (view, band)
    /// in `slot * MEADOW_MAX_BANDS + band` order, each sized + located by
    /// `base_offsets`. The main view holds a near (blade) region followed
    /// by a far (tuft) region; shadow slots hold only a near region.
    pub blades: Buffer,
    /// Group-4 (raster) bind group binding `blades`. Rebuilt only on
    /// realloc (a bind group references the buffer, not its contents, so
    /// `queue.write_buffer` doesn't invalidate it).
    pub draw_bind_group: BindGroup,
    /// One `DrawIndexedIndirectArgs` (20 B) per view.
    pub indirect: Buffer,
    /// One atomic append cursor (u32) per view. Zeroed each frame.
    pub cursors: Buffer,
    /// Per-variant list of live patch indices (dispatch X dim source).
    pub active_patches: Buffer,
    /// Mesh-shader path task work list (`MeadowTaskSlices`: a count
    /// header then `(patch_index, slice_base)` entries): one entry per
    /// 128-blade slice of each active patch, so the mesh path's folded
    /// `draw_mesh_tasks` grid (see `mesh.rs::MESH_TASK_DISPATCH_STRIDE`)
    /// launches exactly the needed task workgroups instead of a grid
    /// sized for the 65536-blade maximum (mostly immediate early-outs on
    /// typical patches). A stub when the mesh path is unavailable.
    pub task_slices: Buffer,
    /// Entries in `task_slices` this frame.
    pub num_task_slices: u32,
    /// Allocated capacity of `task_slices` (entries). Tracks resizes.
    pub task_slice_capacity: u32,
    /// CPU copy of the last-uploaded work list — the list only changes
    /// when patches stream in/out, so steady frames skip the (up to
    /// ~1 MB) re-upload.
    pub uploaded_task_slices: Vec<[u32; 2]>,
    /// Per-variant `MeadowViewCullData` (frusta + per-view base/cap).
    /// Storage (not uniform) to match the compute shader's
    /// `var<storage, read> view_cull` at group-0 binding 4.
    pub view_cull: StorageBuffer<MeadowViewCullData>,
    /// Number of live patches uploaded into `active_patches`.
    pub num_active: u32,
    /// Main-view near-band (blade) region capacity (records).
    pub cap_main_near: u32,
    /// Main-view far-band (tuft) region capacity (records).
    pub cap_main_far: u32,
    /// Per-cascade shadow region capacity (records). Band 0 only — tufts
    /// never cast, so shadow slots' band-1 capacity is zero.
    pub cap_shadow: u32,
    /// Allocated capacity of `active_patches` (in u32s). Tracks resizes.
    pub active_capacity: u32,
    /// Per-(view, band) base offset into `blades` (records), indexed
    /// `slot * MEADOW_MAX_BANDS + band`.
    pub base_offsets: [u32; MEADOW_MAX_VIEWS * MEADOW_MAX_BANDS],
    /// Whether the static indirect args have been written with all LOD
    /// meshes resident. Reset on realloc; the per-frame indirect write runs
    /// only until this latches (meshes load async at startup), then stops —
    /// the static fields change only on realloc, while `instance_count` is
    /// GPU-written each frame.
    pub indirect_ready: bool,
}

#[derive(Resource, Default)]
pub struct MeadowGpuBuffers {
    pub by_variant: HashMap<MeadowVariantId, VariantGpuBuffers>,
}

/// Per-variant extracted data the prepare/compute steps need.
///
/// `Assets<MeadowMaterial>` doesn't exist in the render world (the PBR
/// material plugin only mirrors materials as `PreparedMaterial`/render
/// assets, not raw `Assets`), so the three GPU-backed handles are
/// snapshotted here main-side and looked up in the render world via
/// `RenderAssets<GpuShaderBuffer>` / `RenderAssets<GpuImage>`.
pub struct ExtractedVariant {
    pub patches: AssetId<bevy::render::storage::ShaderBuffer>,
    pub trunk_slots: AssetId<bevy::render::storage::ShaderBuffer>,
    pub heightfield: AssetId<Image>,
    pub variant_params: VariantParams,
    pub active_patches: Vec<u32>,
    /// Mesh-path task work list: `(patch_index, slice_base)` per
    /// 128-blade slice of each active patch.
    pub task_slices: Vec<[u32; 2]>,
    /// LOD-weighted estimate of main-view NEAR-band (blade) survivors —
    /// bounds the main near region. Sized off what renders, not the active
    /// set, let alone the full placement list (tens of millions of blades).
    pub est_main_near: u32,
    /// Estimate of main-view FAR-band (tuft) survivors — bounds the main
    /// far region (sparse: blade_count × tuft_density_near).
    pub est_main_far: u32,
    /// LOD-weighted estimate of shadow survivors (band 0; only blades
    /// within `SHADOW_MAX_DIST` of the viewer cast) — bounds each cascade
    /// region.
    pub est_shadow_records: u32,
}

#[derive(Resource, Default)]
pub struct MeadowExtractedVariants {
    pub by_variant: HashMap<MeadowVariantId, ExtractedVariant>,
}

/// Render-world map `driver main-entity -> variant`, so
/// `DrawMeadowPatch` resolves the variant from the phase item.
#[derive(Resource, Default)]
pub struct RenderMeadowDriver {
    pub by_entity: bevy::ecs::entity::hash_map::EntityHashMap<MeadowVariantId>,
}

// ---------- compute pipeline + bind groups ----------

#[derive(Resource)]
pub struct MeadowComputePipeline {
    pub layout: BindGroupLayoutDescriptor,
    pub cull_and_compact: CachedComputePipelineId,
    pub write_instance_counts: CachedComputePipelineId,
}

/// Identity fingerprint of the resources a variant's compute bind group
/// binds: the 8 storage/uniform buffers (vp uniform, patches, trunk_slots,
/// view_cull, active_patches, blades, cursors, indirect) plus the
/// heightfield texture view. A bind group references resources by handle,
/// not contents, so `queue.write_buffer` doesn't invalidate it — only a
/// realloc/grow of one of our buffers, a heightfield fallback→resident
/// flip, or a trunk-slots re-upload changes an id here. Rebuild only when
/// this differs from the cached value.
type ComputeBindGroupFingerprint = ([BufferId; 8], TextureViewId);

#[derive(Resource, Default)]
pub struct MeadowComputeBindGroups {
    pub by_variant: HashMap<MeadowVariantId, (BindGroup, ComputeBindGroupFingerprint)>,
}

/// Group-4 (raster) layout descriptor — pushed onto the material
/// pipeline in `MeadowExt::specialize` AND used to build the per-variant
/// draw bind group, so the two are byte-identical.
#[derive(Resource)]
pub struct MeadowDrawBindGroupLayout(pub BindGroupLayoutDescriptor);

/// Per-variant `VariantParams` uniform, extracted off the material so
/// compute can read it at compute group-0 binding 0.
#[derive(Resource, Default)]
pub struct MeadowVariantParamsBuffers {
    pub by_variant: HashMap<MeadowVariantId, UniformBuffer<VariantParams>>,
}

// ---------- plugin wiring ----------

/// Register all render-world compute resources + systems on the render
/// sub-app. Called from `MeadowRenderPlugin::build`.
pub fn build_meadow_compute(render_app: &mut SubApp) {
    render_app
        .init_resource::<MeadowGpuBuffers>()
        .init_resource::<MeadowExtractedVariants>()
        .init_resource::<RenderMeadowDriver>()
        .init_resource::<MeadowComputeBindGroups>()
        .init_resource::<MeadowVariantParamsBuffers>()
        .init_resource::<MeadowViewSlots>()
        .init_resource::<MeadowCascadeBounds>()
        .init_resource::<MeadowMeshPathActive>()
        .add_systems(
            RenderStartup,
            (init_meadow_compute_pipeline, init_meadow_draw_bgl),
        )
        .add_systems(
            ExtractSchedule,
            (
                extract_meadow_variants,
                extract_meadow_driver,
                extract_meadow_cascade_bounds,
            ),
        )
        .add_systems(
            Render,
            (
                // All resource prep runs in PrepareResources (chained
                // strictly before PrepareBindGroups), so the per-variant
                // buffers + view slots exist before the bind groups bind
                // them. The two view-cull producers (`build_meadow_view_slots`
                // shared frusta + `prepare_meadow_gpu_buffers` per-variant
                // base/cap upload) are ordered.
                build_meadow_view_slots.in_set(RenderSystems::PrepareResources),
                prepare_meadow_variant_params.in_set(RenderSystems::PrepareResources),
                prepare_meadow_gpu_buffers
                    .in_set(RenderSystems::PrepareResources)
                    .after(build_meadow_view_slots),
                prepare_meadow_compute_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            ),
        )
        .add_systems(RenderGraph, meadow_compute_node.before(camera_driver));
}

// ---------- RenderStartup: pipeline + layouts ----------

fn compute_bind_group_layout() -> BindGroupLayoutDescriptor {
    // Compute group 0 — order + types match meadow_compute.wgsl exactly.
    // The runtime-array storage buffers use `*_sized(.., None)` since the
    // element type is a WGSL struct, not a Rust `ShaderType`.
    BindGroupLayoutDescriptor::new(
        "meadow_compute_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<VariantParams>(false), // 0 variant_params
                storage_buffer_read_only_sized(false, None), // 1 patches
                storage_buffer_read_only_sized(false, None), // 2 trunk_slots
                texture_2d(TextureSampleType::Float { filterable: false }), // 3 heightfield
                storage_buffer_read_only::<MeadowViewCullData>(false), // 4 view_cull
                storage_buffer_read_only_sized(false, None), // 5 active_patches
                storage_buffer_sized(false, None),      // 6 out_blades (rw)
                storage_buffer_sized(false, None),      // 7 cursors (rw atomic)
                storage_buffer_sized(false, None),      // 8 indirect (rw)
            ),
        ),
    )
}

fn init_meadow_compute_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = compute_bind_group_layout();
    let shader = asset_server.load(MEADOW_COMPUTE_SHADER);
    let cull_and_compact = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("meadow_cull_and_compact".into()),
        layout: vec![layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("cull_and_compact")),
        ..default()
    });
    let write_instance_counts = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("meadow_write_instance_counts".into()),
        layout: vec![layout.clone()],
        shader,
        entry_point: Some(Cow::from("write_instance_counts")),
        ..default()
    });
    commands.insert_resource(MeadowComputePipeline {
        layout,
        cull_and_compact,
        write_instance_counts,
    });
}

/// Group-4 (raster) layout descriptor. Built identically here and in
/// `MeadowExt::specialize` so the descriptor pushed onto the material
/// pipeline layout matches the one the draw bind group is built from.
///
/// Binding 0: `var<storage, read> blades: array<CompactedBladeRecord>`
/// (VERTEX). Mirror of `meadow.wgsl`'s `@group(4) @binding(0)`.
pub fn meadow_draw_bind_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "meadow_draw_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX,
            storage_buffer_read_only_sized(false, None),
        ),
    )
}

fn init_meadow_draw_bgl(mut commands: Commands) {
    commands.insert_resource(MeadowDrawBindGroupLayout(meadow_draw_bind_group_layout()));
}

// ---------- ExtractSchedule ----------

/// Extract per-variant data: material handle + `variant_params` from
/// the registry, the live patch-index list from `MeadowPatch` entities,
/// and the total blade sum (over all placements) for buffer sizing.
fn extract_meadow_variants(
    registry: Extract<Option<Res<MeadowVariantRegistry>>>,
    materials: Extract<Res<Assets<MeadowMaterial>>>,
    patches: Extract<Query<&MeadowPatch>>,
    viewer: Extract<Option<Res<MeadowViewer>>>,
    mut out: ResMut<MeadowExtractedVariants>,
) {
    out.by_variant.clear();
    let Some(registry) = registry.as_deref() else {
        return;
    };
    let viewer_xz = viewer.as_deref().map(|v| v.eye_xz).unwrap_or(Vec2::ZERO);

    // Per-variant LOD curve (full / max view distance). Used to weight the
    // buffer-capacity estimate by the fraction of each patch's blades that
    // actually survive the LOD gate at the viewer distance — instead of
    // sizing for every active blade (the LOD culls most far ones).
    let lod_params: HashMap<MeadowVariantId, (f32, f32, f32)> = registry
        .iter()
        .map(|(id, e)| {
            (
                *id,
                (
                    e.variant.lod.full_distance,
                    e.variant.lod.tuft_start,
                    e.variant.lod.tuft_density_near,
                ),
            )
        })
        .collect();

    // Per-variant active footprint from live entities (streaming correctness
    // — only loaded-chunk patches have entities). We accumulate SURVIVOR
    // ESTIMATES, not raw blade counts: `main_est` weights each patch's blades
    // by its LOD survival fraction at the viewer distance; `shadow_est` uses
    // the tighter shadow view distance (only near grass casts). Sizing the
    // buffers off what actually renders — not the whole active set, let alone
    // the full placement list — is what keeps the storage binding far under
    // the 2 GB limit. The kernel's `slot >= cap` guard covers any under-est.
    #[derive(Default)]
    struct ActiveAcc {
        indices: Vec<u32>,
        task_slices: Vec<[u32; 2]>,
        main_near: f32,
        main_far: f32,
        shadow_est: f32,
    }
    let mut active: HashMap<MeadowVariantId, ActiveAcc> = HashMap::default();
    for patch in patches.iter() {
        if patch.blade_count == 0 {
            continue;
        }
        let (lod_full, tuft_start, tuft_density_near) = lod_params
            .get(&patch.variant)
            .copied()
            .unwrap_or((40.0, 55.0, 0.12));
        let dist = patch.centre.distance(viewer_xz);
        let n = patch.blade_count as f32;

        let acc = active.entry(patch.variant).or_default();
        acc.indices.push(patch.patch_index);
        // Mesh-path task work list: exact 128-blade slices for this
        // patch (see `VariantGpuBuffers::task_slices`). Compiled out with
        // the feature — nothing reads it.
        if cfg!(feature = "mesh-shaders") {
            let mut slice_base = 0u32;
            while slice_base < patch.blade_count {
                acc.task_slices.push([patch.patch_index, slice_base]);
                slice_base += MESH_TASK_BLADES;
            }
        }
        // Band is per-patch (centre distance vs `tuft_start`), matching the
        // kernel. Band 0 (near blade) inside `tuft_start`, tapering to none
        // by then; band 1 (far tuft) beyond, sparse — bounded by
        // `tuft_density_near`. Frustum culling only reduces further, so both
        // are safe upper bounds.
        if dist < tuft_start {
            let near_survive =
                1.0 - ((dist - lod_full) / (tuft_start - lod_full).max(1e-3)).clamp(0.0, 1.0);
            acc.main_near += n * near_survive;
        } else {
            acc.main_far += n * tuft_density_near;
        }
        if dist <= SHADOW_MAX_DIST + patch.radius {
            // Shadow casters are band 0 (SHADOW_MAX_DIST < tuft_start). Sized
            // for the full 0..SHADOW_MAX_DIST estimate per shadow slot; the
            // per-cascade radial clip thins each further, so this over-
            // estimates (safe).
            let shadow_survive =
                1.0 - ((dist - lod_full) / (SHADOW_MAX_DIST - lod_full).max(1e-3)).clamp(0.0, 1.0);
            acc.shadow_est += n * shadow_survive;
        }
    }

    for (id, entry) in registry.iter() {
        let Some(material) = materials.get(&entry.material) else {
            continue;
        };
        let acc = active.remove(id).unwrap_or_default();
        out.by_variant.insert(
            *id,
            ExtractedVariant {
                patches: material.extension.patches.id(),
                trunk_slots: material.extension.trunk_slots.id(),
                heightfield: material.extension.heightfield.id(),
                variant_params: material.extension.variant_params,
                active_patches: acc.indices,
                task_slices: acc.task_slices,
                est_main_near: acc.main_near.ceil() as u32,
                est_main_far: acc.main_far.ceil() as u32,
                est_shadow_records: acc.shadow_est.ceil() as u32,
            },
        );
    }
}

/// Mirror `MeadowRenderDriver -> variant` into the render world so the
/// draw command can resolve the variant from the phase item's entity.
fn extract_meadow_driver(
    drivers: Extract<Query<(Entity, &MeadowRenderDriver)>>,
    mut out: ResMut<RenderMeadowDriver>,
) {
    out.by_entity.clear();
    for (entity, driver) in drivers.iter() {
        out.by_entity.insert(entity, driver.variant);
    }
}

/// Snapshot the DOMINANT directional light's cascade schedule so
/// `build_meadow_view_slots` can map `cascade_index -> (near, far)`.
/// Near of cascade `i` = `minimum_distance` (i==0) else
/// `bounds[i-1] * (1 - overlap_proportion)`; far = `bounds[i]` — mirrors
/// `bevy_light::cascade::calculate_cascade_bounds`.
///
/// Dominant = the shadow-casting directional light with the highest
/// illuminance (ties broken by entity for frame stability). Grass only
/// ever serves ONE light's cascades: apps running two shadow-casting
/// directional lights (e.g. a sun + moon day/night rig with both enabled
/// around twilight) would otherwise overflow the view slots and split
/// cascades across lights — and the dim body's grass shadows are
/// sub-visible anyway. The winner's render-world identity is published
/// in `dominant_light_bits` so `build_meadow_view_slots` filters against
/// the SAME light this schedule came from; selection lives only here.
fn extract_meadow_cascade_bounds(
    configs: Extract<
        Query<(
            Entity,
            &CascadeShadowConfig,
            &DirectionalLight,
            &RenderEntity,
        )>,
    >,
    mut out: ResMut<MeadowCascadeBounds>,
) {
    out.per_cascade.clear();
    out.dominant_light_bits = None;
    let Some((_, config, _, render_entity)) = configs
        .iter()
        .filter(|(_, _, light, _)| light.shadow_maps_enabled)
        .max_by(|a, b| {
            a.2.illuminance
                .total_cmp(&b.2.illuminance)
                .then_with(|| b.0.cmp(&a.0))
        })
    else {
        return;
    };
    out.dominant_light_bits = Some(render_entity.id().to_bits());
    let overlap = config.overlap_proportion;
    for (i, &far) in config.bounds.iter().enumerate() {
        let near = if i == 0 {
            config.minimum_distance
        } else {
            config.bounds[i - 1] * (1.0 - overlap)
        };
        out.per_cascade.push((near, far));
    }
}

// ---------- Prepare ----------

/// Build the frame-stable view-slot map + shared per-view cull data
/// (frusta + flags + lod_max). Slot 0 = main 3D camera (an
/// `ExtractedView` with a `Frustum` and NO `LightEntity`); slots 1.. =
/// directional cascades, ordered for determinism.
pub(crate) fn build_meadow_view_slots(
    main_views: Query<(&ExtractedView, &Frustum), Without<LightEntity>>,
    cascades: Query<(&ExtractedView, &Frustum, &LightEntity)>,
    cascade_bounds: Res<MeadowCascadeBounds>,
    mut slots: ResMut<MeadowViewSlots>,
) {
    slots.by_retained.clear();
    slots.shared = MeadowViewCullData::default();
    let mut next_slot: u32 = 0;

    let push = |slots: &mut MeadowViewSlots,
                next_slot: &mut u32,
                view: &ExtractedView,
                frustum: &Frustum,
                is_shadow: bool,
                lod_max: f32,
                shadow_near: f32| {
        let slot = *next_slot as usize;
        if slot >= MEADOW_MAX_VIEWS {
            return;
        }
        let planes = frustum.0.half_spaces.map(|h| h.normal_d());
        let flags = if is_shadow {
            MEADOW_VIEW_FLAG_SHADOW
        } else {
            0
        };
        // params.x = flags (bit 0 = shadow). params.y = lod_max (shadow: the
        // per-cascade far clip; main: filled per variant). params.zw +
        // params2.xy (base/cap per band) are filled in
        // `prepare_meadow_gpu_buffers`; params2.z carries the per-cascade near
        // radial clip. The cascade index itself isn't needed on the GPU — the
        // slice it implies is already baked into lod_max + shadow_near.
        slots.shared.views[slot] = MeadowViewCull {
            frustum: planes,
            params: Vec4::new(flags as f32, lod_max, 0.0, 0.0),
            params2: Vec4::new(0.0, 0.0, shadow_near, 0.0),
        };
        slots
            .by_retained
            .insert(view.retained_view_entity, *next_slot);
        *next_slot += 1;
    };

    // Slot 0: the main camera view. There can be several non-light
    // views (e.g. UI / 2D); pick the first with a Frustum that isn't a
    // light. Deterministic enough — there is one player 3D camera.
    if let Some((view, frustum)) = main_views.iter().next() {
        push(&mut slots, &mut next_slot, view, frustum, false, 0.0, 0.0);
    }

    // Slots 1..: the DOMINANT directional light's cascades, sorted by
    // cascade index. Each cascade clips grass to its own radial slice
    // `[near, min(far, SHADOW_MAX_DIST)]` (from the cascade schedule) so
    // a blade casts into ~1 cascade instead of all 4, while the slices'
    // union still covers `[0, SHADOW_MAX_DIST]`.
    //
    // The dominant light is chosen ONCE, in
    // `extract_meadow_cascade_bounds` (highest-illuminance shadow
    // caster), which publishes the winner's identity alongside its
    // cascade schedule — filtering against it here guarantees the
    // schedule and the cascade views belong to the same light. Serving
    // one light matters: with two shadow-casting directional lights
    // (a sun + moon rig around twilight) there are more cascade views
    // than slots, and naive slotting gives one body all its cascades and
    // the other ONLY the nearest slice — grass shadows then exist only
    // right around the camera whenever the brighter body draws the
    // short straw.
    let mut casc: Vec<(&ExtractedView, &Frustum, usize)> = Vec::new();
    for (view, frustum, light) in cascades.iter() {
        if let LightEntity::Directional {
            light_entity,
            cascade_index,
        } = light
            && cascade_bounds.dominant_light_bits == Some(light_entity.to_bits())
        {
            casc.push((view, frustum, *cascade_index));
        }
    }
    casc.sort_by_key(|&(_, _, cascade_index)| cascade_index);
    for (view, frustum, cascade_index) in casc {
        let (near, far) = cascade_bounds
            .per_cascade
            .get(cascade_index)
            .copied()
            .unwrap_or((0.0, SHADOW_MAX_DIST));
        let lod_max = far.min(SHADOW_MAX_DIST);
        push(
            &mut slots,
            &mut next_slot,
            view,
            frustum,
            true,
            lod_max,
            near,
        );
    }

    slots.count = next_slot;
    slots.shared.count = next_slot;
}

/// Copy each variant's `variant_params` into its compute uniform buffer.
fn prepare_meadow_variant_params(
    extracted: Res<MeadowExtractedVariants>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut params: ResMut<MeadowVariantParamsBuffers>,
) {
    params
        .by_variant
        .retain(|id, _| extracted.by_variant.contains_key(id));
    for (id, ev) in extracted.by_variant.iter() {
        let buf = params.by_variant.entry(*id).or_default();
        buf.set(ev.variant_params);
        buf.write_buffer(&render_device, &render_queue);
    }
}

/// Allocate/resize per-variant buffers, compute per-slot base offsets,
/// fill per-variant view-cull base/cap, zero cursors, write the static
/// indirect fields, and upload the active-patch list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_meadow_gpu_buffers(
    extracted: Res<MeadowExtractedVariants>,
    slots: Res<MeadowViewSlots>,
    mesh_ids: Res<RenderMeadowMeshIds>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    pipeline_cache: Res<PipelineCache>,
    draw_bgl: Res<MeadowDrawBindGroupLayout>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mesh_path: Res<MeadowMeshPathActive>,
    mut buffers: ResMut<MeadowGpuBuffers>,
) {
    buffers
        .by_variant
        .retain(|id, _| extracted.by_variant.contains_key(id));

    // Per-band static indirect fields (index_count, first_index,
    // base_vertex) from each LOD mesh's slices in the shared allocator:
    // band 0 = blade (27 indices), band 1 = tuft (21). A band whose mesh
    // isn't resident yet stays `None` and writes a zero (no-op) record.
    let band_slices: [Option<(u32, u32, u32)>; MEADOW_MAX_BANDS] = core::array::from_fn(|band| {
        let mesh_id = mesh_ids.lod[band]?;
        let gpu_mesh = render_meshes.get(mesh_id)?;
        let RenderMeshBufferInfo::Indexed { count, .. } = &gpu_mesh.buffer_info else {
            return None;
        };
        let vertex_slice = mesh_allocator.mesh_vertex_slice(&mesh_id)?;
        let index_slice = mesh_allocator.mesh_index_slice(&mesh_id)?;
        Some((*count, index_slice.range.start, vertex_slice.range.start))
    });

    let n_views = slots.count.max(1) as usize;

    let make_buf = |label: &str, size: u64, usage: BufferUsages| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };

    for (id, ev) in extracted.by_variant.iter() {
        // Size off the active footprint (rounded up, grown-only): per-band
        // main caps (near blade + sparse far tuft) and a per-cascade shadow
        // cap (band 0 only). Never sized off the full placement list — that
        // blows the 2 GB storage-binding limit. One lookup for the previous
        // caps + the grow decision.
        //
        // When the mesh-shader path serves every view, the compute path
        // produces nothing: collapse all caps to zero so the blades buffer
        // shrinks to a stub (the VRAM saving is the point of that path).
        // Caps are grown-only WITHIN a mode; a path toggle changes them in
        // either direction, which forces one realloc (and re-latches the
        // static indirect args via `indirect_ready`).
        let prev = buffers.by_variant.get(id);
        let (cap_main_near, cap_main_far, cap_shadow) = if mesh_path.active {
            (0, 0, 0)
        } else {
            (
                round_cap(ev.est_main_near).max(prev.map_or(0, |vb| vb.cap_main_near)),
                round_cap(ev.est_main_far).max(prev.map_or(0, |vb| vb.cap_main_far)),
                round_cap(ev.est_shadow_records).max(prev.map_or(0, |vb| vb.cap_shadow)),
            )
        };
        let needs_realloc = prev.is_none_or(|vb| {
            (cap_main_near, cap_main_far, cap_shadow)
                != (vb.cap_main_near, vb.cap_main_far, vb.cap_shadow)
        });

        // Capacity of a (slot, band) region: the main view holds a near
        // (blade) + far (tuft) region; shadow slots hold only band 0
        // (tufts never cast).
        let cap_for = |slot: usize, band: usize| -> u32 {
            if slot == 0 {
                if band == 0 {
                    cap_main_near
                } else {
                    cap_main_far
                }
            } else if band == 0 {
                cap_shadow
            } else {
                0 // shadow band 1 = tufts, which never cast → empty region
            }
        };

        // Per-(slot, band) base offsets + total record capacity, laid out
        // `slot * MEADOW_MAX_BANDS + band`.
        let mut base_offsets = [0u32; MEADOW_MAX_VIEWS * MEADOW_MAX_BANDS];
        let mut total_records: u32 = 0;
        for slot in 0..MEADOW_MAX_VIEWS {
            for band in 0..MEADOW_MAX_BANDS {
                base_offsets[slot * MEADOW_MAX_BANDS + band] = total_records;
                total_records = total_records.saturating_add(cap_for(slot, band));
            }
        }

        let active_len = ev.active_patches.len().max(1) as u32;
        // Work-list buffer sizing: a stub when the mesh path can never
        // run on this device (nothing binds it).
        let slice_len = if mesh_path.available {
            ev.task_slices.len().max(1) as u32
        } else {
            1
        };
        // `MeadowTaskSlices` layout: 8 B count header + 8 B entries.
        let task_slices_size = |len: u32| 8 + u64::from(len) * 8;

        if needs_realloc {
            debug!(
                "meadow variant {:?}: blades buffer {} MiB ({} records = main near {} + far {} + {}×shadow {})",
                id,
                u64::from(total_records) * BLADE_RECORD_SIZE / (1 << 20),
                total_records,
                cap_main_near,
                cap_main_far,
                MEADOW_MAX_VIEWS - 1,
                cap_shadow,
            );
            let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
            // Blades + its group-4 draw bind group are co-located: the BG
            // binds only `blades`, so it need only be rebuilt here, on
            // realloc, alongside the buffer it references.
            // `max(1)`: with the mesh path serving every view the caps are
            // all zero, but wgpu rejects zero-size buffers (and the bind
            // group still binds it).
            let blades = make_buf(
                "meadow_blades",
                u64::from(total_records.max(1)) * BLADE_RECORD_SIZE,
                BufferUsages::STORAGE,
            );
            let draw_bind_group = render_device.create_bind_group(
                Some("meadow_draw_bind_group"),
                &pipeline_cache.get_bind_group_layout(&draw_bgl.0),
                &BindGroupEntries::single(blades.as_entire_binding()),
            );
            // Indirect + cursor arrays carry one record per (view, band).
            let band_views = (MEADOW_MAX_VIEWS * MEADOW_MAX_BANDS) as u64;
            buffers.by_variant.insert(
                *id,
                VariantGpuBuffers {
                    blades,
                    draw_bind_group,
                    indirect: make_buf(
                        "meadow_indirect",
                        band_views * INDIRECT_ARGS_SIZE,
                        storage | BufferUsages::INDIRECT,
                    ),
                    cursors: make_buf("meadow_cursors", band_views * 4, storage),
                    active_patches: make_buf(
                        "meadow_active_patches",
                        u64::from(active_len) * 4,
                        storage,
                    ),
                    task_slices: make_buf(
                        "meadow_task_slices",
                        task_slices_size(slice_len),
                        storage,
                    ),
                    num_task_slices: 0,
                    task_slice_capacity: slice_len,
                    uploaded_task_slices: Vec::new(),
                    view_cull: StorageBuffer::default(),
                    num_active: 0,
                    cap_main_near,
                    cap_main_far,
                    cap_shadow,
                    active_capacity: active_len,
                    base_offsets,
                    indirect_ready: false,
                },
            );
        }

        // Single mutable handle for the rest of this variant's updates.
        let vb = buffers.by_variant.get_mut(id).unwrap();

        // Grow active_patches / task_slices if the live set outgrew the
        // allocations.
        if active_len > vb.active_capacity {
            vb.active_patches = make_buf(
                "meadow_active_patches",
                u64::from(active_len) * 4,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );
            vb.active_capacity = active_len;
        }
        if slice_len > vb.task_slice_capacity {
            vb.task_slices = make_buf(
                "meadow_task_slices",
                task_slices_size(slice_len),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );
            vb.task_slice_capacity = slice_len;
            vb.uploaded_task_slices.clear();
        }
        vb.base_offsets = base_offsets;
        vb.num_active = ev.active_patches.len() as u32;
        vb.num_task_slices = ev.task_slices.len() as u32;

        // Build this variant's view-cull data: shared frusta/flags/lod +
        // per-(variant, view, band) base offsets + caps.
        let mut vc = slots.shared.clone();
        let main_lod_max = ev.variant_params.lod.y;
        for (slot, view) in vc.views[..n_views].iter_mut().enumerate() {
            // Main view: lod_max = this variant's max_view_distance. Shadow
            // views keep the per-cascade clip from build_meadow_view_slots.
            if (view.params.x as u32 & MEADOW_VIEW_FLAG_SHADOW) == 0 {
                view.params.y = main_lod_max;
            }
            // params.zw = band 0 (near) base/cap; params2.xy = band 1 (far)
            // base/cap. params2.z (shadow_near) is already set.
            view.params.z = base_offsets[slot * MEADOW_MAX_BANDS] as f32;
            view.params.w = cap_for(slot, 0) as f32;
            view.params2.x = base_offsets[slot * MEADOW_MAX_BANDS + 1] as f32;
            view.params2.y = cap_for(slot, 1) as f32;
        }
        vb.view_cull.set(vc);
        vb.view_cull.write_buffer(&render_device, &render_queue);

        // Zero the per-(view, band) cursors each frame.
        render_queue.write_buffer(
            &vb.cursors,
            0,
            bytemuck::cast_slice(&[0u32; MEADOW_MAX_VIEWS * MEADOW_MAX_BANDS]),
        );

        // Upload the active patch indices.
        if !ev.active_patches.is_empty() {
            render_queue.write_buffer(
                &vb.active_patches,
                0,
                bytemuck::cast_slice(&ev.active_patches),
            );
        }
        // Upload the mesh-path task work list — header (count) + entries,
        // `MeadowTaskSlices` in the WGSL — only when the device can run
        // the mesh path AND the list changed (patch streaming), so steady
        // frames upload nothing.
        if mesh_path.available && vb.uploaded_task_slices != ev.task_slices {
            render_queue.write_buffer(
                &vb.task_slices,
                0,
                bytemuck::cast_slice(&[ev.task_slices.len() as u32, 0u32]),
            );
            if !ev.task_slices.is_empty() {
                render_queue.write_buffer(
                    &vb.task_slices,
                    8,
                    bytemuck::cast_slice(&ev.task_slices),
                );
            }
            vb.uploaded_task_slices.clone_from(&ev.task_slices);
        }

        // Static indirect fields per (view, band): each band's own
        // index_count / first_index / base_vertex (from its LOD mesh) and
        // first_instance = the (view, band) base offset. These change only on
        // realloc (which clears `indirect_ready`), so write them once meshes
        // are resident rather than every frame; `instance_count` (offset 1)
        // is GPU-written by `write_instance_counts`. A non-resident band's
        // record stays zero (index_count 0 → draws nothing) and we retry next
        // frame until both meshes resolve, then latch.
        if !vb.indirect_ready {
            let mut indirect_data = [0u32; MEADOW_MAX_VIEWS * MEADOW_MAX_BANDS * 5];
            for slot in 0..MEADOW_MAX_VIEWS {
                for (band, slice) in band_slices.iter().enumerate() {
                    let Some((count, first_index, base_vertex)) = *slice else {
                        continue;
                    };
                    let idx = slot * MEADOW_MAX_BANDS + band;
                    let args = &mut indirect_data[idx * 5..idx * 5 + 5];
                    args[0] = count; // index_count
                    args[1] = 0; // instance_count (GPU-written)
                    args[2] = first_index; // first_index
                    args[3] = base_vertex; // base_vertex (i32 bits)
                    args[4] = base_offsets[idx]; // first_instance
                }
            }
            render_queue.write_buffer(&vb.indirect, 0, bytemuck::cast_slice(&indirect_data));
            vb.indirect_ready = band_slices.iter().all(Option::is_some);
        }
    }
}

/// Build the compute bind group per variant. Skips a variant whose
/// input GPU assets (`patches`/`trunk_slots` shader buffers, heightfield
/// image) aren't resident yet — the node then skips its dispatch.
#[allow(clippy::too_many_arguments)]
fn prepare_meadow_compute_bind_groups(
    extracted: Res<MeadowExtractedVariants>,
    shader_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<bevy::render::texture::FallbackImage>,
    pipeline: Res<MeadowComputePipeline>,
    pipeline_cache: Res<PipelineCache>,
    params: Res<MeadowVariantParamsBuffers>,
    buffers: Res<MeadowGpuBuffers>,
    render_device: Res<RenderDevice>,
    mut bind_groups: ResMut<MeadowComputeBindGroups>,
) {
    bind_groups
        .by_variant
        .retain(|id, _| extracted.by_variant.contains_key(id));

    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);

    for (id, ev) in extracted.by_variant.iter() {
        // Reach the GPU-resident input assets via the handle ids
        // snapshotted main-side (the render world has no
        // `Assets<MeadowMaterial>`).
        let Some(patches) = shader_buffers.get(ev.patches) else {
            continue;
        };
        let Some(trunk_slots) = shader_buffers.get(ev.trunk_slots) else {
            continue;
        };
        // Until the consumer supplies an atlas the handle is default /
        // not yet resident; fall back to the 1×1 zeros texture (the
        // shader's `sample_heightfield` short-circuits a <2px texture to
        // ground_y = 0), matching the prior `FallbackImage` path the
        // `AsBindGroup` derive gave the raster shader.
        let heightfield = gpu_images.get(ev.heightfield).unwrap_or(&fallback_image.d2);
        let Some(params_buf) = params.by_variant.get(id).and_then(|b| b.buffer()) else {
            continue;
        };
        let Some(vb) = buffers.by_variant.get(id) else {
            continue;
        };
        let Some(view_cull) = vb.view_cull.buffer() else {
            continue;
        };

        // Identity of every bound resource — see `ComputeBindGroupFingerprint`
        // for why the per-frame `write_buffer` uploads don't change these ids.
        let fingerprint: ComputeBindGroupFingerprint = (
            [
                params_buf.id(),
                patches.buffer.id(),
                trunk_slots.buffer.id(),
                view_cull.id(),
                vb.active_patches.id(),
                vb.blades.id(),
                vb.cursors.id(),
                vb.indirect.id(),
            ],
            heightfield.texture_view.id(),
        );
        // Skip the (non-trivial) bind-group rebuild when nothing changed.
        if let Some((_, cached)) = bind_groups.by_variant.get(id)
            && *cached == fingerprint
        {
            continue;
        }

        let bg = render_device.create_bind_group(
            Some("meadow_compute_bind_group"),
            &layout,
            &BindGroupEntries::sequential((
                params_buf.as_entire_binding(),
                patches.buffer.as_entire_binding(),
                trunk_slots.buffer.as_entire_binding(),
                &heightfield.texture_view,
                view_cull.as_entire_binding(),
                vb.active_patches.as_entire_binding(),
                vb.blades.as_entire_binding(),
                vb.cursors.as_entire_binding(),
                vb.indirect.as_entire_binding(),
            )),
        );
        bind_groups.by_variant.insert(*id, (bg, fingerprint));
    }
}

// ---------- RenderGraph node ----------

fn meadow_compute_node(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<MeadowComputePipeline>,
    bind_groups: Res<MeadowComputeBindGroups>,
    buffers: Res<MeadowGpuBuffers>,
    slots: Res<MeadowViewSlots>,
    mesh_path: Res<MeadowMeshPathActive>,
) {
    // The mesh-shader path derives + culls inline in its own raster
    // passes; nothing reads the compacted buffers, so skip the dispatch
    // entirely (this is the compute-side work the mesh path saves).
    if mesh_path.active {
        return;
    }
    let (Some(cull), Some(finalize)) = (
        pipeline_cache.get_compute_pipeline(pipeline.cull_and_compact),
        pipeline_cache.get_compute_pipeline(pipeline.write_instance_counts),
    ) else {
        return;
    };
    if slots.count == 0 || bind_groups.by_variant.is_empty() {
        return;
    }
    let n_views = slots.count;

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("meadow_compute"),
            timestamp_writes: None,
        });

    for (id, (bg, _)) in bind_groups.by_variant.iter() {
        let Some(vb) = buffers.by_variant.get(id) else {
            continue;
        };
        if vb.num_active == 0 {
            continue;
        }
        // One workgroup per (active patch, view); workgroup size lives
        // in the WGSL (`@workgroup_size(256)`).
        pass.set_bind_group(0, bg, &[]);
        pass.set_pipeline(cull);
        pass.dispatch_workgroups(vb.num_active, n_views, 1);
        pass.set_pipeline(finalize);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
