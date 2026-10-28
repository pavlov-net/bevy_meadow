//! GPU-driven cull + compact compute pass for the meadow renderer.
//!
//! Per frame, for every (active patch, view) pair, the compute kernel
//! `cull_and_compact` (in `meadow_compute.wesl`) derives each blade,
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

use bevy::camera::Camera3d;
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
use std::sync::Arc;

use crate::field::MeadowCoverField;
use crate::material::{MAX_MEADOW_PALETTES, MeadowMaterial, PaletteParams, VariantParams};
use crate::mesh::{
    BLADE_RECORD_SIZE, INDIRECT_ARGS_SIZE, MEADOW_MAX_BANDS, MEADOW_MAX_VIEWS, MESH_TASK_BLADES,
    SHADOW_MAX_DIST,
};
use crate::plugin::MeadowHeightfield;
use crate::plugin::{
    MeadowLodCurve, MeadowPalettes, MeadowPatch, MeadowVariant, MeadowVariantId,
    MeadowVariantRegistry, MeadowViewer, palette_table,
};
use crate::render::{MeadowRenderDriver, RenderMeadowMeshIds};
use crate::rt_selection::{LivePlacements, RT_NEAR_DISTANCE, RtInputs, RtPose};

const MEADOW_COMPUTE_SHADER: &str = "embedded://bevy_meadow/meadow_compute.wesl";

/// Fixed per-variant capacity for persistent RT slots, least view-weighted
/// distance first. The near band shares the exact visible ribbon, with no
/// width inflation. 32,768 nine-triangle blades are a 294,912-triangle
/// budget.
pub const RT_NEAR_MAX_BLADES: u32 = 32_768;
/// Far persistent slots, selected with gradually reduced density out to 50 m.
pub const RT_FAR_MAX_BLADES: u32 = 262_144;
/// Share the visible ribbon topology so raster and RT cannot drift.
const RT_NEAR_VERTS_PER_BLADE: u32 = crate::mesh::BLADE_VERTS_PER_BLADE;
const RT_NEAR_INDICES_PER_BLADE: u32 = crate::mesh::BLADE_INDICES_PER_BLADE;
/// Vertices / indices per far-band proxy (one triangle).
const RT_FAR_VERTS_PER_BLADE: u32 = 3;
const RT_FAR_INDICES_PER_BLADE: u32 = 3;
/// Bytes per solari `PackedVertex` (3 × vec4<f32>). Asserted against solari's
/// own stride below so the two can't drift; kept as a local literal because
/// the `bevy_solari` dep is optional (`solari` feature).
const RT_VERTEX_SIZE: u64 = 48;
#[cfg(all(feature = "solari", not(target_family = "wasm")))]
const _: () =
    assert!(RT_VERTEX_SIZE == bevy_solari::scene::RaytracingGeometryBuffers::VERTEX_STRIDE);
/// The same triangles as the visible ribbon, at `slot * 11`.
const RT_NEAR_BLADE_INDICES: [u32; 27] = crate::mesh::BLADE_INDICES;
/// Static far-band index pattern (single triangle) at `slot * 3`.
const RT_FAR_BLADE_INDICES: [u32; 3] = [0, 1, 2];

/// Flag bit 0 in `MeadowViewCull.params.x` marking a shadow (cascade)
/// view. Mirror of `MEADOW_VIEW_FLAG_SHADOW` in `meadow_compute.wesl`.
pub(crate) const MEADOW_VIEW_FLAG_SHADOW: u32 = 1;

/// Blade-capacity rounding granularity (512Ki blades). Active footprints
/// are rounded up to this and buffers are grown-only, so chunk crossings
/// don't thrash buffer reallocation.
const CAP_GRAN: u32 = 1 << 19;

fn round_cap(n: u32) -> u32 {
    n.max(1).div_ceil(CAP_GRAN) * CAP_GRAN
}

/// Region caps `(main near, main far, shadow)` shrunk in proportion until
/// the blades buffer, which binds whole, fits a binding of
/// `max_binding_bytes`. The survivors' estimate ignores the frustum, so it
/// runs high; past the cap the kernels drop survivors rather than write
/// out of their region.
fn fit_caps((near, far, shadow): (u32, u32, u32), max_binding_bytes: u64) -> (u32, u32, u32) {
    let shadow_slots = (MEADOW_MAX_VIEWS - 1) as u64;
    let total = u64::from(near) + u64::from(far) + shadow_slots * u64::from(shadow);
    let max_records = max_binding_bytes / BLADE_RECORD_SIZE;
    if total <= max_records {
        return (near, far, shadow);
    }
    let fit = |cap: u32| (u64::from(cap) * max_records / total) as u32;
    (fit(near), fit(far), fit(shadow))
}

// ---------- GPU-uploaded per-view culling data ----------

/// Rust mirror of `MeadowViewCull` (`meadow_compute.wesl`). `frustum`
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
    /// work list), which stays warm across automatic path switches;
    /// `MeadowForceComputePath` additionally pauses the list build, and
    /// extract order guarantees it's rebuilt the frame the force unflips.
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
    /// One atomic append cursor (u32) per view. Zeroed by
    /// `meadow_compute_node` (`clear_buffer`) ahead of each frame's
    /// dispatches, so idle frames pay nothing.
    pub cursors: Buffer,
    /// Per-variant list of live patch indices (dispatch X dim source).
    pub active_patches: Buffer,
    /// Active-patch list last uploaded into `active_patches`. The list only
    /// changes when patches stream in/out, so steady frames skip the
    /// re-upload. Holding the `Arc` keeps its pointer from being reused.
    pub uploaded_active_patches: Option<Arc<Vec<u32>>>,
    /// Mesh-shader path task work list (`MeadowTaskSlices`: a count
    /// header then `(patch_index, slice_base)` entries): one entry per
    /// 128-blade slice of the blades each active patch's LOD gate can pass
    /// (`task_slice_bound`), so the mesh path's folded `draw_mesh_tasks`
    /// grid (see `mesh.rs::MESH_TASK_DISPATCH_STRIDE`) launches only task
    /// workgroups that can hold survivors. A stub when the mesh path is
    /// unavailable.
    pub task_slices: Buffer,
    /// Entries in `task_slices` this frame.
    pub num_task_slices: u32,
    /// Allocated capacity of `task_slices` (entries). Tracks resizes.
    pub task_slice_capacity: u32,
    /// Work list last uploaded into `task_slices`. The list changes when
    /// patches stream in/out or the viewer moves a couple of metres, so
    /// steady frames skip the re-upload.
    pub uploaded_task_slices: Option<Arc<Vec<[u32; 2]>>>,
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

/// Swap in `next` only when the contents differ, so consumers can treat an
/// unchanged `Arc` pointer as unchanged contents.
fn replace_if_changed<T: PartialEq>(list: &mut Arc<Vec<T>>, next: Vec<T>) {
    if **list != next {
        *list = Arc::new(next);
    }
}

/// One variant's lists derived from its live `MeadowPatch` entities, shared
/// with the render world. Each `Arc` is replaced only when its contents
/// change (see `replace_if_changed`).
#[derive(Default)]
struct VariantPatchLists {
    active_patches: Arc<Vec<u32>>,
    task_slices: Arc<Vec<[u32; 2]>>,
    /// `(patch_index, placement)` per active patch, for RT selection.
    rt_live: LivePlacements,
}

/// Patch-derived lists for every registered variant. They depend only on
/// the `MeadowPatch` set, the registry (placements, membership) and the
/// task-list key, so `extract_meadow_variants` rebuilds them only when one
/// of those changes.
#[derive(Default)]
struct MeadowPatchLists {
    by_variant: HashMap<MeadowVariantId, VariantPatchLists>,
    /// What the task list was last built for, `None` before the first
    /// rebuild: whether it is built at all, the viewer position and whether
    /// shadow views exist (`task_slice_bound`).
    task_key: Option<(bool, Vec2, bool)>,
}

/// Viewer travel, in metres, that rebuilds the mesh-path task list.
const TASK_LIST_REBUILD_DISTANCE: f32 = 2.0;

/// Viewer distance, in metres, from the position a task list was built for
/// within which its slice bounds hold: the travel before a rebuild plus the
/// lag of the viewer position the kernels read (`VariantParams`).
const TASK_LIST_VIEWER_STEP: f32 = 2.0 * TASK_LIST_REBUILD_DISTANCE;

/// CPU twin of `meadow_shared.wesl::meadow_gate_count`: the blades of a
/// patch that pass a survival `threshold`, a prefix of its blade indices.
pub(crate) fn gate_count(threshold: f32, blade_count: u32) -> u32 {
    ((threshold.clamp(0.0, 1.0) * blade_count as f32).ceil() as u32).min(blade_count)
}

/// Shadow-caster survival threshold at planar player distance `dist`: the
/// shadow-view arm of `meadow_shared.wesl::meadow_band_threshold`.
fn shadow_survival(lod: &MeadowLodCurve, dist: f32) -> f32 {
    let ramp = ((dist - lod.shadow_full_dist) / (SHADOW_MAX_DIST - lod.shadow_full_dist).max(1e-3))
        .clamp(0.0, 1.0);
    1.0 + (lod.shadow_far_density - 1.0) * ramp
}

/// Upper bound on the blades of a patch that the LOD gate passes in any
/// view while the viewer stays within `TASK_LIST_VIEWER_STEP` of
/// `viewer_xz`. Survivors are a prefix of the blade indices, so the task
/// list covers only this many. Every threshold falls with distance, so
/// each band's is taken at the nearest the viewer can come; `shadows`
/// adds the shadow casters' threshold to band 0. `gate_slack` is how much
/// nearer than `centre` the patch gates its prefix: a tile's outer radius
/// (`gate_dist` in the kernels), 0 for a disc.
fn task_slice_bound(
    variant: &MeadowVariant,
    centre: Vec2,
    gate_slack: f32,
    blade_count: u32,
    viewer_xz: Vec2,
    shadows: bool,
) -> u32 {
    let lod = &variant.lod;
    let dist = centre.distance(viewer_xz);
    let nearest = (dist - gate_slack - TASK_LIST_VIEWER_STEP).max(0.0);
    let farthest = dist + TASK_LIST_VIEWER_STEP;
    let mut threshold = 0.0_f32;
    // Band 0 (near blades, shadow casters) inside `tuft_start`.
    if nearest < lod.tuft_start {
        threshold = lod.near_survival(nearest);
        if shadows {
            threshold = threshold.max(shadow_survival(lod, nearest));
        }
    }
    // Band 1 (far tufts) beyond it.
    if farthest >= lod.tuft_start {
        let (tuft_near, tuft_far) = variant.tuft_survival();
        threshold = threshold.max(tuft_near.max(tuft_far));
    }
    gate_count(threshold, blade_count)
}

/// How much nearer than its centre patch `index` of `placements` gates its
/// LOD prefix: a tile's outer radius, 0 for a disc.
fn gate_slack(placements: &[crate::placement::PatchPlacement], index: u32) -> f32 {
    placements
        .get(index as usize)
        .filter(|p| p.is_tile())
        .map_or(0.0, crate::placement::PatchPlacement::outer_radius)
}

impl MeadowPatchLists {
    /// Whether the task list holds for these inputs: built (or not) the same
    /// way, with the same shadow views, and for a viewer within
    /// `TASK_LIST_REBUILD_DISTANCE` when it is built.
    fn task_list_current(&self, build_task_slices: bool, viewer_xz: Vec2, shadows: bool) -> bool {
        self.task_key.is_some_and(|(built, viewer, with_shadows)| {
            built == build_task_slices
                && with_shadows == shadows
                && (!built || viewer.distance(viewer_xz) <= TASK_LIST_REBUILD_DISTANCE)
        })
    }

    fn rebuild<'a>(
        &mut self,
        registry: &MeadowVariantRegistry,
        patches: impl Iterator<Item = &'a MeadowPatch>,
        build_task_slices: bool,
        viewer_xz: Vec2,
        shadows: bool,
    ) {
        // Non-empty patches, by variant.
        let mut members: HashMap<MeadowVariantId, Vec<&MeadowPatch>> = HashMap::default();
        for patch in patches {
            if patch.blade_count > 0 {
                members.entry(patch.variant).or_default().push(patch);
            }
        }
        self.by_variant.retain(|id, _| registry.get(*id).is_some());
        for (id, entry) in registry.iter() {
            let members = members.remove(id).unwrap_or_default();
            let lists = self.by_variant.entry(*id).or_default();
            replace_if_changed(
                &mut lists.active_patches,
                members.iter().map(|p| p.patch_index).collect(),
            );
            // Mesh-path task work list: 128-blade slices over each patch's
            // gate bound (see `VariantGpuBuffers::task_slices`).
            let mut task_slices = Vec::new();
            if build_task_slices {
                for p in &members {
                    let bound = task_slice_bound(
                        &entry.variant,
                        p.centre,
                        gate_slack(&entry.placements, p.patch_index),
                        p.blade_count,
                        viewer_xz,
                        shadows,
                    );
                    task_slices.extend(
                        (0..bound)
                            .step_by(MESH_TASK_BLADES as usize)
                            .map(|slice_base| [p.patch_index, slice_base]),
                    );
                }
            }
            replace_if_changed(&mut lists.task_slices, task_slices);
            replace_if_changed(
                &mut lists.rt_live,
                members
                    .iter()
                    .filter_map(|p| {
                        entry
                            .placements
                            .get(p.patch_index as usize)
                            .copied()
                            .map(|placement| (p.patch_index, placement))
                    })
                    .collect(),
            );
        }
        self.task_key = Some((build_task_slices, viewer_xz, shadows));
    }
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
    /// The variant's clearing-profile table
    /// (`RegisteredVariant::clearing_profile_image`).
    pub clearing_profiles: AssetId<Image>,
    /// The cover field atlas (`MeadowCoverField::atlas`).
    pub cover: AssetId<Image>,
    /// The material's params with the cover field's lanes filled in.
    pub variant_params: VariantParams,
    /// The variant's palette table. Replaced only when its contents change.
    pub palettes: PaletteTable,
    /// Replaced only when its contents change.
    pub active_patches: Arc<Vec<u32>>,
    /// Mesh-path task work list: `(patch_index, slice_base)` per
    /// 128-blade slice of each active patch. Replaced only when its
    /// contents change.
    pub task_slices: Arc<Vec<[u32; 2]>>,
    /// LOD-weighted estimate of main-view NEAR-band (blade) survivors —
    /// bounds the main near region. Sized off what renders, not the active
    /// set, let alone the full placement list (tens of millions of blades).
    pub est_main_near: u32,
    /// Estimate of main-view FAR-band (tuft) survivors — bounds the main
    /// far region (sparse: blade_count × the near tuft survival, `MeadowVariant::tuft_survival`).
    pub est_main_far: u32,
    /// LOD-weighted estimate of shadow survivors (band 0; only blades
    /// within `SHADOW_MAX_DIST` of the viewer cast) — bounds each cascade
    /// region.
    pub est_shadow_records: u32,
    /// The raytracing casters' frame, while they are enabled.
    pub(crate) rt: Option<RtExtract>,
}

impl ExtractedVariant {
    /// The heightfield, cover field and clearing-profile views the kernels
    /// bind, in that order, each falling back to the 1×1 zeros texture until
    /// it is resident: the kernels read a field narrower than 2 texels as
    /// ground at 0, full cover on neutral ground and no clearing profile.
    pub(crate) fn textures<'a>(
        &self,
        gpu_images: &'a RenderAssets<GpuImage>,
        fallback: &'a bevy::render::texture::FallbackImage,
    ) -> [&'a GpuImage; 3] {
        [self.heightfield, self.cover, self.clearing_profiles]
            .map(|id| gpu_images.get(id).unwrap_or(&fallback.d2))
    }
}

#[derive(Resource, Default)]
pub struct MeadowExtractedVariants {
    pub by_variant: HashMap<MeadowVariantId, ExtractedVariant>,
    /// Where the casters' selection publishes its telemetry.
    pub(crate) telemetry: Arc<Mutex<Vec<MeadowRtVariantStats>>>,
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
/// binds: the 9 storage/uniform buffers (vp uniform, patches, trunk_slots,
/// view_cull, active_patches, blades, cursors, indirect, palettes) plus the
/// heightfield, cover field and clearing-profile texture views. A bind group
/// references resources by handle, not contents, so `queue.write_buffer` doesn't invalidate it — only a
/// realloc/grow of one of our buffers, a heightfield fallback→resident
/// flip, or a trunk-slots re-upload changes an id here. Rebuild only when
/// this differs from the cached value.
type ComputeBindGroupFingerprint = ([BufferId; 9], [TextureViewId; 3]);

#[derive(Resource, Default)]
pub struct MeadowComputeBindGroups {
    pub by_variant: HashMap<MeadowVariantId, (BindGroup, ComputeBindGroupFingerprint)>,
}

/// Group-4 (raster) layout descriptor — pushed onto the material
/// pipeline in `MeadowExt::specialize` AND used to build the per-variant
/// draw bind group, so the two are byte-identical.
#[derive(Resource)]
pub struct MeadowDrawBindGroupLayout(pub BindGroupLayoutDescriptor);

/// A variant's shader palette table, shared between frames until it
/// changes.
pub type PaletteTable = Arc<[PaletteParams; MAX_MEADOW_PALETTES]>;

/// Per-variant `VariantParams` uniform, extracted off the material so
/// compute can read it at compute group-0 binding 0, and the variant's
/// palette table, bound beside it by the kernels that derive blades.
#[derive(Resource, Default)]
pub struct MeadowVariantParamsBuffers {
    pub by_variant: HashMap<MeadowVariantId, VariantParamsGpu>,
}

/// One variant's [`MeadowVariantParamsBuffers`] entry.
pub struct VariantParamsGpu {
    pub params: UniformBuffer<VariantParams>,
    /// `array<PaletteParams, MAX_MEADOW_PALETTES>`, rewritten only when the
    /// extracted table changes.
    pub palettes: Buffer,
    /// Table last written into `palettes`. Holding the `Arc` keeps its
    /// pointer from being reused.
    pub uploaded_palettes: Option<PaletteTable>,
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
                extract_meadow_variants.after(extract_meadow_cascade_bounds),
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
                prepare_meadow_variant_params
                    .in_set(RenderSystems::PrepareResources)
                    .after(build_meadow_view_slots),
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
    // Compute group 0 — order + types match meadow_compute.wesl exactly.
    // The runtime-array storage buffers use `*_sized(.., None)` since the
    // element type is a WGSL struct, not a Rust `ShaderType`.
    // Bindings 14 (palettes), 15 (cover field) and 16 (clearing profiles)
    // sit past the RT expansion's 9..=13.
    BindGroupLayoutDescriptor::new(
        "meadow_compute_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, uniform_buffer::<VariantParams>(false)),
                (1, storage_buffer_read_only_sized(false, None)), // patches
                (2, storage_buffer_read_only_sized(false, None)), // trunk_slots
                (
                    3,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ), // heightfield
                (4, storage_buffer_read_only::<MeadowViewCullData>(false)), // view_cull
                (5, storage_buffer_read_only_sized(false, None)), // active_patches
                (6, storage_buffer_sized(false, None)),           // out_blades (rw)
                (7, storage_buffer_sized(false, None)),           // cursors (rw atomic)
                (8, storage_buffer_sized(false, None)),           // indirect (rw)
                (14, storage_buffer_read_only_sized(false, None)), // palettes
                (
                    15,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ), // cover_field
                (
                    16,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ), // clearing_profiles
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
/// (VERTEX). Mirror of `meadow.wesl`'s `@group(4) @binding(0)`.
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

/// One variant's LOD/shadow gate parameters plus the survivor estimates
/// `extract_meadow_variants` accumulates against them each frame.
struct VariantEstimate<'a> {
    id: MeadowVariantId,
    placements: &'a [crate::placement::PatchPlacement],
    lod: MeadowLodCurve,
    /// Near tuft survival fraction (`MeadowVariant::tuft_survival`).
    tuft_near: f32,
    main_near: f32,
    main_far: f32,
    shadow: f32,
}

/// Extract per-variant data: material handle + `variant_params` from
/// the registry, the live patch-index list from `MeadowPatch` entities,
/// and viewer-dependent survivor estimates for buffer sizing.
fn extract_meadow_variants(
    registry: Extract<Option<Res<MeadowVariantRegistry>>>,
    materials: Extract<Res<Assets<MeadowMaterial>>>,
    patches: Extract<Query<Ref<MeadowPatch>>>,
    mut removed_patches: Extract<RemovedComponents<MeadowPatch>>,
    (viewer, cascade_bounds): (Extract<Option<Res<MeadowViewer>>>, Res<MeadowCascadeBounds>),
    (cover_field, heightfield): (
        Extract<Option<Res<MeadowCoverField>>>,
        Extract<Option<Res<MeadowHeightfield>>>,
    ),
    (rt_config, time): (
        Extract<Option<Res<MeadowRaytracingConfig>>>,
        Extract<Res<Time>>,
    ),
    diagnostics: Extract<Res<MeadowRtDiagnostics>>,
    telemetry: Extract<Res<MeadowRtTelemetry>>,
    #[cfg(feature = "mesh-shaders")] force_compute: Extract<
        Res<crate::mesh_path::MeadowForceComputePath>,
    >,
    #[cfg(feature = "mesh-shaders")] mesh_path: Res<MeadowMeshPathActive>,
    meadow_palettes: Extract<Res<MeadowPalettes>>,
    mut out: ResMut<MeadowExtractedVariants>,
    mut lists: Local<MeadowPatchLists>,
    mut palette_tables: Local<HashMap<MeadowVariantId, PaletteTable>>,
) {
    let patches_removed = !removed_patches.is_empty();
    removed_patches.clear();
    out.by_variant.clear();
    out.telemetry = Arc::clone(&telemetry.0);
    let diagnostics = MeadowRtDiagnostics {
        exact_only: diagnostics.exact_only,
        exact_radius_m: if diagnostics.exact_radius_m.is_finite() {
            diagnostics.exact_radius_m.clamp(0.5, RT_NEAR_DISTANCE)
        } else {
            4.0
        },
    };
    let registry_changed = (*registry).as_ref().is_some_and(DetectChanges::is_changed);
    let Some(registry) = registry.as_deref() else {
        return;
    };
    let eye = viewer.as_deref().map_or(Vec3::ZERO, |v| v.eye);
    let viewer_xz = eye.xz();
    let rt_pose = viewer.as_deref().map_or_else(RtPose::default, |v| RtPose {
        eye: v.eye,
        forward: v.depth_plane.xyz().normalize_or_zero(),
        cone: v.view_cone.max(0.0),
    });
    let margins = viewer
        .as_deref()
        .map_or_else(Default::default, |v| v.caster_margins);
    let cover_field = cover_field.as_deref().cloned().unwrap_or_default();
    let (cover_extents, cover_params) = cover_field.params();
    let heights = heightfield.as_deref().and_then(|h| h.grid.clone());
    let rt_enabled = rt_config.as_deref().is_some_and(|c| c.enabled);
    let exact_radius = diagnostics.exact_only.then_some(diagnostics.exact_radius_m);
    let ease = 1.0 - (-time.delta_secs() / RT_DRAWN_EASE_SECONDS).exp();
    // The task work list exists solely for the mesh path — build it only
    // while that path can consume it (device support, not force-compute).
    // Extract runs before the prepare systems, so the list is rebuilt and
    // re-uploaded the same frame the force toggle unflips.
    #[cfg(feature = "mesh-shaders")]
    let build_task_slices = mesh_path.available && !force_compute.0;
    #[cfg(not(feature = "mesh-shaders"))]
    let build_task_slices = false;

    // Per-variant LOD curve (full / max view distance). Used to weight the
    // buffer-capacity estimate by the fraction of each patch's blades that
    // actually survive the LOD gate at the viewer distance — instead of
    // sizing for every active blade (the LOD culls most far ones).
    let mut estimates: Vec<VariantEstimate> = registry
        .iter()
        .map(|(id, e)| VariantEstimate {
            id: *id,
            placements: &e.placements,
            lod: e.variant.lod,
            tuft_near: e.variant.tuft_survival().0,
            main_near: 0.0,
            main_far: 0.0,
            shadow: 0.0,
        })
        .collect();

    // Per-variant active footprint from live entities (streaming correctness
    // — only loaded-chunk patches have entities). We accumulate SURVIVOR
    // ESTIMATES, not raw blade counts: `main_near`/`main_far` weight each
    // patch's blades by its LOD survival fraction at the viewer distance;
    // `shadow` uses the tighter shadow view distance (only near grass casts).
    // Sizing the buffers off what actually renders — not the whole active
    // set, let alone the full placement list — is what keeps the storage
    // binding far under the 2 GB limit. The kernel's `slot >= cap` guard
    // covers any under-est. The same pass detects patch-set changes.
    let mut patches_changed = patches_removed;
    for patch in patches.iter() {
        patches_changed |= patch.is_changed();
        if patch.blade_count == 0 {
            continue;
        }
        // Variants are few; a linear scan beats hashing per patch.
        let Some(est) = estimates.iter_mut().find(|e| e.id == patch.variant) else {
            continue;
        };
        let dist = (patch.centre.distance(viewer_xz)
            - gate_slack(est.placements, patch.patch_index))
        .max(0.0);
        // Band is per-patch (centre distance vs `tuft_start`), matching the
        // kernel: band 0 (near blades, shadow casters) inside `tuft_start`,
        // band 1 (far tufts) beyond. The kernels' survival thresholds
        // (`meadow_band_threshold`) at the patch centre; frustum and
        // cascade culling only reduce further, so these are safe upper
        // bounds.
        if dist < est.lod.tuft_start {
            est.main_near += gate_count(est.lod.near_survival(dist), patch.blade_count) as f32;
            // Sized per shadow slot for every caster; each cascade's
            // radial clip thins it further.
            est.shadow += gate_count(shadow_survival(&est.lod, dist), patch.blade_count) as f32;
        } else {
            est.main_far += gate_count(est.tuft_near, patch.blade_count) as f32;
        }
    }

    // One rule decides shadow views: grass serves the dominant
    // shadow-mapped directional light's cascades.
    let shadows = cascade_bounds.dominant_light_bits.is_some();
    if registry_changed
        || patches_changed
        || !lists.task_list_current(build_task_slices, viewer_xz, shadows)
    {
        lists.rebuild(
            registry,
            patches.iter().map(Ref::into_inner),
            build_task_slices,
            viewer_xz,
            shadows,
        );
    }

    // A table write replaces only the tables whose contents changed, so the
    // render side re-uploads only those.
    let palettes_changed = meadow_palettes.is_changed() || registry_changed;
    palette_tables.retain(|id, _| registry.get(*id).is_some());
    // `estimates` was built from this same registry iteration order.
    for ((id, entry), est) in registry.iter().zip(&estimates) {
        let current = palette_tables.get(id);
        if palettes_changed || current.is_none() {
            let table = palette_table(meadow_palettes.current(*id, &entry.variant));
            if current.is_none_or(|current| **current != table) {
                palette_tables.insert(*id, Arc::new(table));
            }
        }
        let Some(material) = materials.get(&entry.material) else {
            continue;
        };
        let Some(variant_lists) = lists.by_variant.get(id) else {
            continue;
        };
        let rt = (rt_enabled && !variant_lists.active_patches.is_empty()).then(|| RtExtract {
            inputs: RtInputs {
                placements: Arc::clone(&variant_lists.rt_live),
                trunks: Arc::clone(&entry.trunk_slots),
                profiles: Arc::clone(&entry.clearing_profiles),
                cover: cover_field.grid.clone(),
                survival: cover_field.survival,
                heights: heights.clone(),
            },
            viewer: rt_pose,
            margins,
            exact_radius,
            sunward_widen: entry.variant.raytraced_sunward_widen,
            ease,
        });
        let mut variant_params = material.extension.variant_params;
        variant_params.cover_extents = cover_extents;
        variant_params.cover = cover_params;
        out.by_variant.insert(
            *id,
            ExtractedVariant {
                patches: material.extension.patches.id(),
                trunk_slots: material.extension.trunk_slots.id(),
                heightfield: material.extension.heightfield.id(),
                clearing_profiles: entry.clearing_profile_image.id(),
                cover: cover_field.atlas.id(),
                variant_params,
                palettes: Arc::clone(&palette_tables[id]),
                active_patches: Arc::clone(&variant_lists.active_patches),
                task_slices: Arc::clone(&variant_lists.task_slices),
                est_main_near: est.main_near.ceil() as u32,
                est_main_far: est.main_far.ceil() as u32,
                est_shadow_records: est.shadow.ceil() as u32,
                rt,
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
/// `ExtractedView` with a `Frustum` and a `Camera3d`, NO `LightEntity`);
/// slots 1.. = directional cascades, ordered for determinism.
pub(crate) fn build_meadow_view_slots(
    main_views: Query<(&ExtractedView, &Frustum), (With<Camera3d>, Without<LightEntity>)>,
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

    // Slot 0: the main camera view. `With<Camera3d>` keeps non-3D views
    // (a menu's 2D UI camera) from claiming the slot and dragging the
    // whole per-frame meadow pipeline along; with no 3D camera the slot
    // map stays empty and the prepare systems idle. Deterministic
    // enough — there is one player 3D camera.
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

/// Copy each variant's `variant_params` into its compute uniform buffer,
/// stamping the wind-time fields (`wind.zw`) in on the way, and write its
/// palette table when the extracted table changed.
fn prepare_meadow_variant_params(
    extracted: Res<MeadowExtractedVariants>,
    slots: Res<MeadowViewSlots>,
    time: Res<Time>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut params: ResMut<MeadowVariantParamsBuffers>,
) {
    params
        .by_variant
        .retain(|id, _| extracted.by_variant.contains_key(id));
    // No meadow views (e.g. a menu with only a 2D camera) — nothing
    // consumes the uniforms this frame, so skip the uploads.
    if slots.count == 0 {
        return;
    }
    // Wind time: the same values `prepare_globals_buffer` writes into the
    // `globals` uniform the raster VS reads (same render-world `Time`, same
    // frame), so the compute / mesh / RT kernels — which read `wind.zw`
    // from this uniform — sway bit-identically to the raster grass. The
    // wrap gives a one-frame sway-phase pop + wrong grass motion vectors
    // once per wrap period (~1 h default) — accepted: parity demands the
    // raster path's wrapped clock, and a phase-continuous fix isn't worth
    // the complexity.
    let now = time.elapsed_secs_wrapped();
    let prev = now - time.delta_secs();
    for (id, ev) in extracted.by_variant.iter() {
        let gpu = params
            .by_variant
            .entry(*id)
            .or_insert_with(|| VariantParamsGpu {
                params: UniformBuffer::default(),
                palettes: render_device.create_buffer(&BufferDescriptor {
                    label: Some("meadow_palettes"),
                    size: size_of::<[PaletteParams; MAX_MEADOW_PALETTES]>() as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                uploaded_palettes: None,
            });
        let mut variant_params = ev.variant_params;
        variant_params.wind.z = now;
        variant_params.wind.w = prev;
        gpu.params.set(variant_params);
        gpu.params.write_buffer(&render_device, &render_queue);
        if !gpu
            .uploaded_palettes
            .as_ref()
            .is_some_and(|uploaded| Arc::ptr_eq(uploaded, &ev.palettes))
        {
            render_queue.write_buffer(
                &gpu.palettes,
                0,
                bytemuck::cast_slice(ev.palettes.as_slice()),
            );
            gpu.uploaded_palettes = Some(Arc::clone(&ev.palettes));
        }
    }
}

/// Allocate/resize per-variant buffers, compute per-slot base offsets,
/// fill per-variant view-cull base/cap, write the static indirect
/// fields, and upload the active-patch list when it changed.
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

    // No meadow views this frame (a menu with only a 2D camera): nothing
    // dispatches or draws, so skip the per-frame view-cull / patch-list
    // uploads. Buffers are left as-is (grown-only) and refresh the frame
    // a 3D camera reappears — this runs before the bind-group prepares.
    if slots.count == 0 {
        return;
    }

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
            fit_caps(
                (
                    round_cap(ev.est_main_near).max(prev.map_or(0, |vb| vb.cap_main_near)),
                    round_cap(ev.est_main_far).max(prev.map_or(0, |vb| vb.cap_main_far)),
                    round_cap(ev.est_shadow_records).max(prev.map_or(0, |vb| vb.cap_shadow)),
                ),
                render_device.limits().max_storage_buffer_binding_size,
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
                    uploaded_active_patches: None,
                    task_slices: make_buf(
                        "meadow_task_slices",
                        task_slices_size(slice_len),
                        storage,
                    ),
                    num_task_slices: 0,
                    task_slice_capacity: slice_len,
                    uploaded_task_slices: None,
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
            vb.uploaded_active_patches = None;
        }
        if slice_len > vb.task_slice_capacity {
            vb.task_slices = make_buf(
                "meadow_task_slices",
                task_slices_size(slice_len),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );
            vb.task_slice_capacity = slice_len;
            vb.uploaded_task_slices = None;
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

        // Upload the active patch indices — only when the list changed
        // (patch streaming); steady frames upload nothing.
        if !ev.active_patches.is_empty()
            && !vb
                .uploaded_active_patches
                .as_ref()
                .is_some_and(|uploaded| Arc::ptr_eq(uploaded, &ev.active_patches))
        {
            render_queue.write_buffer(
                &vb.active_patches,
                0,
                bytemuck::cast_slice(ev.active_patches.as_slice()),
            );
            vb.uploaded_active_patches = Some(Arc::clone(&ev.active_patches));
        }
        // Upload the mesh-path task work list — header (count) + entries,
        // `MeadowTaskSlices` in the WGSL — only when the device can run
        // the mesh path AND the list changed (patch streaming), so steady
        // frames upload nothing.
        if mesh_path.available
            && !vb
                .uploaded_task_slices
                .as_ref()
                .is_some_and(|uploaded| Arc::ptr_eq(uploaded, &ev.task_slices))
        {
            render_queue.write_buffer(
                &vb.task_slices,
                0,
                bytemuck::cast_slice(&[ev.task_slices.len() as u32, 0u32]),
            );
            if !ev.task_slices.is_empty() {
                render_queue.write_buffer(
                    &vb.task_slices,
                    8,
                    bytemuck::cast_slice(ev.task_slices.as_slice()),
                );
            }
            vb.uploaded_task_slices = Some(Arc::clone(&ev.task_slices));
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
        let textures = ev.textures(&gpu_images, &fallback_image);
        let [heightfield, cover, profiles] = textures;
        let Some(gpu) = params.by_variant.get(id) else {
            continue;
        };
        let Some(params_buf) = gpu.params.buffer() else {
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
                gpu.palettes.id(),
            ],
            textures.map(|image| image.texture_view.id()),
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
            &BindGroupEntries::with_indices((
                (0, params_buf.as_entire_binding()),
                (1, patches.buffer.as_entire_binding()),
                (2, trunk_slots.buffer.as_entire_binding()),
                (3, &heightfield.texture_view),
                (4, view_cull.as_entire_binding()),
                (5, vb.active_patches.as_entire_binding()),
                (6, vb.blades.as_entire_binding()),
                (7, vb.cursors.as_entire_binding()),
                (8, vb.indirect.as_entire_binding()),
                (14, gpu.palettes.as_entire_binding()),
                (15, &cover.texture_view),
                (16, &profiles.texture_view),
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

    // Zero the per-(view, band) append cursors of every variant about to
    // dispatch — recorded ahead of the pass in the same encoder, so it's
    // free vs a queue write and absent whenever the node early-outs.
    for id in bind_groups.by_variant.keys() {
        let Some(vb) = buffers.by_variant.get(id) else {
            continue;
        };
        if vb.num_active == 0 {
            continue;
        }
        render_context
            .command_encoder()
            .clear_buffer(&vb.cursors, 0, None);
    }

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

// ================= raytracing blade expansion =================
//
// A standalone compute (independent of the raster path — compute OR
// mesh-shader) that bakes the shadow-caster blade set into solari-layout
// triangle buffers per variant, so a raytracing consumer (`crate::solari`)
// registers them as `RaytracingGeometry` and grass casts RT shadows the same
// way it casts cascade shadows in the non-RT path. Gated by
// `MeadowRaytracingConfig::enabled` (set by the consumer) so it costs nothing
// when no raytracer is present.
//
// The GPU selection (`rt_select_*`) fills persistent blade slots the frame
// the CPU finds one due (`rt_selection.rs`), nearest blades as exact
// ribbons, distant ones as cheaper chords, and a blade keeps its slot while
// selected or drawn. The expansion updates wind in place every frame and
// keeps each slot's previous vertices. Identity generations tell consumers
// when temporal anchors must be dropped.

use bevy::render::diagnostic::{DiagnosticsRecorder, RecordDiagnostics};
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{
    BindGroupEntry, BindGroupLayout, BufferInitDescriptor, CommandEncoder,
    CommandEncoderDescriptor, ComputePipeline, IntoBinding, MapMode, PollType, TextureView,
};
#[cfg(all(feature = "solari", not(target_family = "wasm")))]
use bevy_solari::scene::RaytracingProducerEncoder;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::rt_selection::{MeadowRtMargins, RtSelect, RtSelectState, RtSelector};

/// Consumer-set switch enabling the per-variant RT blade expansion. Off by
/// default (no cost; the pipelines aren't even compiled until first enabled);
/// a raytracing consumer sets `enabled = true` while it wants grass in its
/// acceleration structure. Extracted to the render world.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct MeadowRaytracingConfig {
    pub enabled: bool,
}

/// Opt-in bounded reference mode: exact local ribbons, no distant proxies.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct MeadowRtDiagnostics {
    pub exact_only: bool,
    pub exact_radius_m: f32,
}
impl Default for MeadowRtDiagnostics {
    fn default() -> Self {
        Self {
            exact_only: false,
            exact_radius_m: 4.0,
        }
    }
}

/// Caster selection telemetry, read back from the GPU a few frames late.
/// Counts are of live blades; coverage is the near band's reach,
/// view-weighted outside the exact-only mode, not a visibility or GPU
/// timing metric.
#[derive(Clone, Debug)]
pub struct MeadowRtVariantStats {
    pub variant: MeadowVariantId,
    pub exact_only: bool,
    pub requested_radius_m: f32,
    pub coverage_radius_m: f32,
    /// The near reach the expansion draws with, easing toward the coverage.
    pub drawn_radius_m: f32,
    pub near_selected: usize,
    pub near_capacity: usize,
    pub far_selected: usize,
    pub far_capacity: usize,
    pub candidate_count: usize,
    /// Slots kept for blades drawn past the selection's limits.
    pub pinned: usize,
    /// Selected blades no slot was free for.
    pub dropped: usize,
    /// How many times a slot could pass from a drawn blade to another one,
    /// which drops the consumer's temporal anchors on the casters.
    pub topology_changes: u64,
    /// Whether the latest selection's outcome is still on its way back.
    pub pending: bool,
    pub selection_center: Vec3,
}

/// Shared snapshot published by the render world, readable from main-world
/// UI.
#[derive(Resource, Clone, Default)]
pub struct MeadowRtTelemetry(pub Arc<Mutex<Vec<MeadowRtVariantStats>>>);

/// MUST equal `RT_SELECT_BINS` in `meadow_compute.wesl`: bins per
/// selection key histogram.
pub(crate) const RT_SELECT_BINS: u32 = 16_384;
/// MUST equal `RT_SCAN_ROW` in `meadow_compute.wesl`: workgroups per
/// dispatch row of a scan.
pub(crate) const RT_SCAN_ROW: u32 = 65_535;
/// Slots of both bands, near band first.
const RT_SLOTS: u32 = RT_NEAR_MAX_BLADES + RT_FAR_MAX_BLADES;
/// Seconds the drawn reach and budget term take to cover all but 1/e of
/// the way to a selection's.
const RT_DRAWN_EASE_SECONDS: f32 = 0.15;

/// A variant's casters for one frame, from the extract.
pub(crate) struct RtExtract {
    pub(crate) inputs: RtInputs,
    pub(crate) viewer: RtPose,
    /// The narrowest margins the selection holds its casters for
    /// ([`MeadowViewer::caster_margins`]).
    pub(crate) margins: MeadowRtMargins,
    /// The exact-only reference's radius.
    pub(crate) exact_radius: Option<f32>,
    /// [`MeadowVariant::raytraced_sunward_widen`].
    pub(crate) sunward_widen: f32,
    /// Share of the way the drawn limits ease toward the selection's this
    /// frame.
    pub(crate) ease: f32,
}

/// `RtParams` in `meadow_compute.wesl`.
#[repr(C)]
#[derive(ShaderType, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RtParams {
    /// The drawn limits, which only `rt_ease_drawn` writes.
    drawn: Vec4,
    eye: Vec4,
    forward: Vec4,
    ease: Vec4,
}

/// Where the CPU's share of [`RtParams`] starts.
const RT_PARAMS_CPU: usize = std::mem::offset_of!(RtParams, eye);

impl RtExtract {
    /// The expansion's parameters but the drawn limits: the pose it draws
    /// from (the exact-only reference's ranks by eye distance alone) and
    /// the ease.
    pub(crate) fn params(&self) -> RtParams {
        let pose = match self.exact_radius {
            Some(_) => self.viewer.by_distance(),
            None => self.viewer,
        };
        RtParams {
            drawn: Vec4::ZERO,
            eye: pose.eye.extend(pose.cone_cos()),
            forward: pose.ranked_forward().extend(self.sunward_widen),
            ease: Vec4::new(self.ease, 0.0, 0.0, 0.0),
        }
    }
}

/// `RtSelectParams` in `meadow_compute.wesl`.
#[derive(ShaderType, Clone, Copy, Default)]
pub(crate) struct RtSelectParams {
    eye: Vec4,
    forward: Vec4,
    margin: Vec4,
    counts: UVec4,
    run: UVec4,
}

impl RtSelectParams {
    fn new(select: &RtSelect) -> Self {
        let (sin_m, cos_m) = select.margins.angle.sin_cos();
        Self {
            eye: select.pose.eye.extend(select.near_radius()),
            forward: select.pose.ranked_forward().extend(select.pose.cone_cos()),
            margin: Vec4::new(select.margins.distance, sin_m, cos_m, 0.0),
            counts: UVec4::new(
                select.scan.near_chunks,
                select.scan.coverage.len() as u32,
                select.purge.len() as u32,
                u32::from(select.exact_radius.is_none()) | u32::from(select.clear) << 1,
            ),
            run: UVec4::new(select.scan.chunks.len() as u32, select.serial, 0, 0),
        }
    }
}

/// `RtSelectState` in `meadow_compute.wesl`; pairs run near band first.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RtSelectStateGpu {
    pub(crate) serial: u32,
    pub(crate) flags: u32,
    pub(crate) limit: [u32; 2],
    pub(crate) boundary: [u32; 2],
    pub(crate) selected: [u32; 2],
    pub(crate) drawn_other: [u32; 2],
    pub(crate) free: [u32; 2],
    pub(crate) claimed: [u32; 2],
    pub(crate) near_candidates: u32,
    pub(crate) pinned: u32,
    pub(crate) dropped: u32,
    pub(crate) drawn: [u32; 2],
    pub(crate) drawn_valid: u32,
    pub(crate) bound: [u32; 2],
}

impl RtSelectStateGpu {
    pub(crate) fn decode(&self) -> RtSelectState {
        RtSelectState {
            serial: self.serial,
            near_reach: f32::from_bits(self.limit[0]),
            far_budget: f32::from_bits(self.limit[1]),
            near_candidates: self.near_candidates,
            near_selected: self.selected[0],
            far_selected: self.selected[1],
            pinned: self.pinned,
            dropped: self.dropped,
            drawn_near_reach: f32::from_bits(self.drawn[0]),
            drawn_far_budget: f32::from_bits(self.drawn[1]),
        }
    }
}

/// Byte layout of the selection's scratch buffer (`RtSelectScratch`): the
/// state, then the four key histograms, the free slots and the owner
/// tables.
pub(crate) const RT_STATE_SIZE: u64 = size_of::<RtSelectStateGpu>() as u64;
/// The state each selection resets: all but what persists from `drawn` on.
const RT_STATE_RESET: u64 = std::mem::offset_of!(RtSelectStateGpu, drawn) as u64;
const RT_HISTS_SIZE: u64 = 4 * 4 * RT_SELECT_BINS as u64;
const RT_TABLES_OFFSET: u64 = RT_STATE_SIZE + RT_HISTS_SIZE + 4 * RT_SLOTS as u64;
const RT_TABLES_SIZE: u64 = 4 * 2 * RT_SLOTS as u64;
const RT_SCRATCH_SIZE: u64 = RT_TABLES_OFFSET + RT_TABLES_SIZE;
/// Bytes of a slot (`RtSlot`).
const RT_SLOT_SIZE: u64 = 16;

/// Where a readback stands: free, copied into this frame, being mapped, or
/// mapped and ready to read.
const READBACK_IDLE: u8 = 0;
const READBACK_COPIED: u8 = 1;
const READBACK_MAPPING: u8 = 2;
const READBACK_MAPPED: u8 = 3;

/// A selection state copy for the CPU to read without stalling.
struct RtReadback {
    buffer: Buffer,
    stage: Arc<AtomicU8>,
}

fn rt_buffer(render_device: &RenderDevice, label: &str, size: u64, usage: BufferUsages) -> Buffer {
    render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// A variant's GPU selection: its slots, near band first, the selection's
/// parameters and scan, and its scratch.
pub(crate) struct RtSelectBuffers {
    params: UniformBuffer<RtSelectParams>,
    /// The purged patch indices, the coverage and the scan chunks
    /// (`rt_scan`).
    scan: Buffer,
    pub(crate) slots: Buffer,
    pub(crate) scratch: Buffer,
    readbacks: [RtReadback; 2],
}

impl RtSelectBuffers {
    fn new(render_device: &RenderDevice, render_queue: &RenderQueue) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let mut params = UniformBuffer::<RtSelectParams>::default();
        params.set_label(Some("meadow_rt_select_params"));
        params.write_buffer(render_device, render_queue);
        Self {
            params,
            scan: rt_buffer(render_device, "meadow_rt_select_scan", 16, storage),
            slots: rt_buffer(
                render_device,
                "meadow_rt_slots",
                RT_SLOT_SIZE * u64::from(RT_SLOTS),
                storage,
            ),
            scratch: rt_buffer(
                render_device,
                "meadow_rt_select_scratch",
                RT_SCRATCH_SIZE,
                storage,
            ),
            readbacks: std::array::from_fn(|_| RtReadback {
                buffer: rt_buffer(
                    render_device,
                    "meadow_rt_select_readback",
                    RT_STATE_SIZE,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                ),
                stage: Arc::default(),
            }),
        }
    }

    /// Upload `select`'s parameters and scan, growing the scan buffer to
    /// fit it.
    fn upload(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        select: &RtSelect,
    ) {
        self.params.set(RtSelectParams::new(select));
        self.params.write_buffer(render_device, render_queue);
        let mut words = select.purge.clone();
        words.extend(select.scan.coverage.iter().flatten());
        words.extend(select.scan.chunks.iter().flatten());
        let bytes = 4 * words.len() as u64;
        if bytes > self.scan.size() {
            self.scan = rt_buffer(
                render_device,
                "meadow_rt_select_scan",
                bytes.next_power_of_two(),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );
        }
        if !words.is_empty() {
            render_queue.write_buffer(&self.scan, 0, bytemuck::cast_slice(&words));
        }
    }

    fn mapping(&self) -> bool {
        self.readbacks
            .iter()
            .any(|r| r.stage.load(Ordering::Acquire) == READBACK_MAPPING)
    }

    /// The state a mapped readback holds; and start mapping the readbacks
    /// copied into the last frame.
    fn poll_readbacks(&self) -> Option<RtSelectState> {
        let mut read = None;
        for readback in &self.readbacks {
            match readback.stage.load(Ordering::Acquire) {
                READBACK_MAPPED => {
                    read = readback
                        .buffer
                        .slice(..)
                        .get_mapped_range()
                        .ok()
                        .map(|view| {
                            bytemuck::pod_read_unaligned::<RtSelectStateGpu>(&view[..]).decode()
                        });
                    readback.buffer.unmap();
                    readback.stage.store(READBACK_IDLE, Ordering::Release);
                }
                READBACK_COPIED => {
                    readback.stage.store(READBACK_MAPPING, Ordering::Release);
                    let stage = Arc::clone(&readback.stage);
                    readback
                        .buffer
                        .slice(..)
                        .map_async(MapMode::Read, move |result| {
                            let next = if result.is_ok() {
                                READBACK_MAPPED
                            } else {
                                READBACK_IDLE
                            };
                            stage.store(next, Ordering::Release);
                        });
                }
                _ => {}
            }
        }
        read
    }

    /// Copy the selection state into a free readback, if one is.
    fn copy_readback(&self, encoder: &mut CommandEncoder) {
        if let Some(readback) = self
            .readbacks
            .iter()
            .find(|r| r.stage.load(Ordering::Acquire) == READBACK_IDLE)
        {
            encoder.copy_buffer_to_buffer(&self.scratch, 0, &readback.buffer, 0, RT_STATE_SIZE);
            readback.stage.store(READBACK_COPIED, Ordering::Release);
        }
    }
}

/// The RT kernels' entry points, in [`RtPass`] order.
pub(crate) const RT_ENTRIES: [&str; 12] = [
    "rt_select_purge",
    "rt_select_index",
    "rt_select_count_near",
    "rt_select_fit_near",
    "rt_select_drawn_far",
    "rt_select_count_far",
    "rt_select_fit_far",
    "rt_select_evict",
    "rt_select_place",
    "rt_ease_drawn",
    "expand_rt_blades",
    "rt_pad_unused",
];

/// The RT kernels, by their place in [`RT_ENTRIES`].
#[derive(Clone, Copy)]
pub(crate) enum RtPass {
    Purge,
    Index,
    CountNear,
    FitNear,
    DrawnFar,
    CountFar,
    FitFar,
    Evict,
    Place,
    Ease,
    Expand,
    Pad,
}

/// The RT kernels' compute pipelines, in [`RT_ENTRIES`] order.
pub(crate) struct RtPasses<'a>(pub(crate) [&'a ComputePipeline; 12]);

impl RtPasses<'_> {
    /// Record `kernel` over `workgroups`, in rows of `RT_SCAN_ROW`.
    fn dispatch(
        &self,
        pass: &mut bevy::render::render_resource::ComputePass,
        kernel: RtPass,
        workgroups: u32,
    ) {
        if workgroups > 0 {
            pass.set_pipeline(self.0[kernel as usize]);
            pass.dispatch_workgroups(
                workgroups.min(RT_SCAN_ROW),
                workgroups.div_ceil(RT_SCAN_ROW),
                1,
            );
        }
    }

    /// Record `kernels` with `bind_group`.
    fn record(
        &self,
        encoder: &mut CommandEncoder,
        label: &str,
        bind_group: &BindGroup,
        kernels: &[(RtPass, u32)],
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, bind_group, &[]);
        for &(kernel, workgroups) in kernels {
            self.dispatch(&mut pass, kernel, workgroups);
        }
    }

    /// Record `select`, uploaded with [`RtSelectBuffers::upload`], with
    /// `bind_group` binding `buffers`.
    fn record_select(
        &self,
        encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        buffers: &RtSelectBuffers,
        select: &RtSelect,
    ) {
        let reset = if select.clear {
            RT_STATE_SIZE
        } else {
            RT_STATE_RESET
        };
        encoder.clear_buffer(&buffers.scratch, 0, Some(reset));
        encoder.clear_buffer(&buffers.scratch, RT_STATE_SIZE, Some(RT_HISTS_SIZE));
        encoder.clear_buffer(&buffers.scratch, RT_TABLES_OFFSET, Some(RT_TABLES_SIZE));
        let slots = RT_SLOTS.div_ceil(256);
        let chunks = select.scan.chunks.len() as u32;
        let far = u32::from(select.exact_radius.is_none());
        let purge = u32::from(select.clear || !select.purge.is_empty());
        self.record(
            encoder,
            "meadow_rt_select",
            bind_group,
            &[
                (RtPass::Purge, purge * slots),
                (RtPass::Index, slots),
                (RtPass::CountNear, select.scan.near_chunks),
                (RtPass::FitNear, 1),
                (RtPass::DrawnFar, far * RT_FAR_MAX_BLADES.div_ceil(256)),
                (RtPass::CountFar, far * chunks),
                (RtPass::FitFar, 1),
                (RtPass::Evict, slots),
                (RtPass::Place, chunks),
            ],
        );
    }
}

/// One band's RT geometry (solari's 48-byte vertex layout + u32 indices),
/// sized to its fixed blade capacity. Stable handles (never realloc): the
/// vertices are rewritten each frame and the indices are static, so a
/// consumer registers them once and rebuilds its BLAS per frame.
pub struct RtBandBuffers {
    /// `array<PackedVertex>`. `STORAGE | BLAS_INPUT | COPY_SRC`:
    /// compute-written and BLAS-read. Prefilled with NaN (inactive
    /// primitives) at creation.
    pub vertices: Buffer,
    /// `array<u32>` static per-slot index pattern. `STORAGE | BLAS_INPUT`.
    /// Shared between variants (contents are a pure function of the band's
    /// capacity).
    pub indices: Buffer,
    /// Full vertex capacity of `vertices` (cap × verts-per-blade).
    pub vertex_count: u32,
    /// Full index capacity of `indices` (cap × indices-per-blade).
    pub index_count: u32,
    /// Previous-frame positions in the same persistent blade slots, which
    /// the expansion writes beside `vertices`.
    pub previous_vertices: Buffer,
    /// Changes only when a slot may pass from a drawn blade to another one,
    /// not when blades sway or come and go.
    pub topology_generation: u64,
}

/// The resources the kernels read a variant's blades from.
pub(crate) struct RtBlades<'a> {
    pub(crate) params: &'a Buffer,
    pub(crate) patches: &'a Buffer,
    pub(crate) trunks: &'a Buffer,
    pub(crate) palettes: &'a Buffer,
    /// Heightfield, cover field and clearing profiles.
    pub(crate) textures: [&'a TextureView; 3],
}

/// Identity of every resource [`RtBlades`] binds; a bind group references
/// resources, not their contents.
type RtBladesKey = ([BufferId; 4], [TextureViewId; 3]);

impl<'a> RtBlades<'a> {
    fn key(&self) -> RtBladesKey {
        (
            [self.params, self.patches, self.trunks, self.palettes].map(Buffer::id),
            self.textures.map(TextureView::id),
        )
    }

    /// The bindings the selection and the expansion share, with the slots.
    fn entries(&self, slots: &'a Buffer) -> Vec<BindGroupEntry<'a>> {
        let [heightfield, cover, profiles] = self.textures;
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self.patches.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: self.trunks.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: heightfield.into_binding(),
            },
            BindGroupEntry {
                binding: 10,
                resource: slots.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 14,
                resource: self.palettes.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 15,
                resource: cover.into_binding(),
            },
            BindGroupEntry {
                binding: 16,
                resource: profiles.into_binding(),
            },
        ]
    }
}

/// What a variant's casters do this frame.
enum RtFrameInputs<'a> {
    /// Select and expand from its blades.
    Draw(RtBlades<'a>, &'a RtExtract),
    /// Stand still: it has no live patches for a while.
    Hold,
    /// Clear: its blades' inputs are missing.
    Pad,
}

impl ExtractedVariant {
    /// The resources the RT kernels read this variant's blades from, once
    /// all are resident.
    fn blades<'a>(
        &'a self,
        id: &MeadowVariantId,
        buffers: &MeadowGpuBuffers,
        params: &'a MeadowVariantParamsBuffers,
        shader_buffers: &'a RenderAssets<GpuShaderBuffer>,
        gpu_images: &'a RenderAssets<GpuImage>,
        fallback: &'a bevy::render::texture::FallbackImage,
    ) -> Option<RtBlades<'a>> {
        if buffers.by_variant.get(id)?.num_active == 0 {
            return None;
        }
        let gpu = params.by_variant.get(id)?;
        Some(RtBlades {
            params: gpu.params.buffer()?,
            patches: &shader_buffers.get(self.patches)?.buffer,
            trunks: &shader_buffers.get(self.trunk_slots)?.buffer,
            palettes: &gpu.palettes,
            textures: self
                .textures(gpu_images, fallback)
                .map(|image| &image.texture_view),
        })
    }
}

/// The RT kernels' bind group layouts.
pub(crate) struct RtLayouts {
    pub(crate) select: BindGroupLayout,
    pub(crate) expand: BindGroupLayout,
    pub(crate) pad: BindGroupLayout,
}

/// Per-variant RT geometry, its slots and their selection, and the
/// expansion's parameters.
pub struct RtVariantBuffers {
    /// Near band: [`RT_NEAR_MAX_BLADES`] × 11-vert / 9-tri ribbons.
    pub near: RtBandBuffers,
    /// Far band: [`RT_FAR_MAX_BLADES`] × 3-vert / 1-tri proxies.
    pub far: RtBandBuffers,
    /// `rt_params` in `meadow_compute.wesl`.
    params: Buffer,
    pub(crate) select: RtSelectBuffers,
    pub(crate) selector: RtSelector,
    select_group: Option<((RtBladesKey, BufferId), BindGroup)>,
    expand_group: Option<(RtBladesKey, BindGroup)>,
    pad_group: Option<BindGroup>,
    /// Whether the buffers are fully NaN-padded from a frame whose inputs
    /// were missing — skips re-padding them every frame while a variant
    /// idles.
    empty_padded: bool,
    /// The slot identity the bands' topology generations stand for.
    identity: Option<u64>,
    /// Frames in a row the variant had no live patches, or was missing from
    /// the extract.
    inactive_frames: u32,
}

impl RtVariantBuffers {
    pub(crate) fn new(
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        (near_indices, far_indices): (Buffer, Buffer),
    ) -> Self {
        Self {
            near: make_rt_band(
                render_device,
                "meadow_rt_near",
                RT_NEAR_MAX_BLADES,
                RT_NEAR_VERTS_PER_BLADE,
                RT_NEAR_INDICES_PER_BLADE,
                near_indices,
            ),
            far: make_rt_band(
                render_device,
                "meadow_rt_far",
                RT_FAR_MAX_BLADES,
                RT_FAR_VERTS_PER_BLADE,
                RT_FAR_INDICES_PER_BLADE,
                far_indices,
            ),
            params: rt_buffer(
                render_device,
                "meadow_rt_params",
                size_of::<RtParams>() as u64,
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            select: RtSelectBuffers::new(render_device, render_queue),
            selector: RtSelector::default(),
            select_group: None,
            expand_group: None,
            pad_group: None,
            empty_padded: false,
            identity: None,
            inactive_frames: 0,
        }
    }

    /// Read back the selection's state, once a copy is mapped.
    pub(crate) fn poll_readbacks(&mut self) {
        if let Some(state) = self.select.poll_readbacks() {
            self.selector.state = Some(state);
        }
    }

    /// Select if a selection is due, timed as `meadow_rt_select`, ease the
    /// drawn limits toward the selection's, and copy the state for the CPU
    /// while it changes. Returns the selection made.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        encoder: &mut CommandEncoder,
        passes: &RtPasses,
        layout: &BindGroupLayout,
        blades: &RtBlades,
        rt: &RtExtract,
        diagnostics: Option<&DiagnosticsRecorder>,
    ) -> Option<RtSelect> {
        render_queue.write_buffer(
            &self.params,
            RT_PARAMS_CPU as u64,
            &bytemuck::bytes_of(&rt.params())[RT_PARAMS_CPU..],
        );
        let select = self
            .selector
            .update(&rt.inputs, rt.viewer, rt.exact_radius, &rt.margins);
        if let Some(select) = &select {
            self.select.upload(render_device, render_queue, select);
        }
        let key = (blades.key(), self.select.scan.id());
        if self
            .select_group
            .as_ref()
            .is_none_or(|(cached, _)| *cached != key)
        {
            let mut entries = blades.entries(&self.select.slots);
            entries.extend([
                BindGroupEntry {
                    binding: 17,
                    resource: self.select.params.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 18,
                    resource: self.select.scan.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 19,
                    resource: self.select.scratch.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 22,
                    resource: self.params.as_entire_binding(),
                },
            ]);
            let group =
                render_device.create_bind_group("meadow_rt_select_bind_group", layout, &entries);
            self.select_group = Some((key, group));
        }
        let group = &self.select_group.as_ref().unwrap().1;
        if let Some(select) = &select {
            let span = diagnostics.time_span(encoder, "meadow_rt_select");
            passes.record_select(encoder, group, &self.select, select);
            span.end(encoder);
        }
        passes.record(encoder, "meadow_rt_ease", group, &[(RtPass::Ease, 1)]);
        let easing = !self
            .selector
            .state
            .is_some_and(|state| self.selector.current() && state.settled());
        if select.is_some() || easing {
            self.select.copy_readback(encoder);
        }
        select
    }

    /// Expand the slots' blades from this frame's pose.
    pub(crate) fn expand(
        &mut self,
        render_device: &RenderDevice,
        encoder: &mut CommandEncoder,
        passes: &RtPasses,
        layout: &BindGroupLayout,
        blades: &RtBlades,
    ) {
        if self.identity != Some(self.selector.identity) {
            self.identity = Some(self.selector.identity);
            for band in [&mut self.near, &mut self.far] {
                band.topology_generation = band.topology_generation.wrapping_add(1);
            }
        }
        self.empty_padded = false;
        let key = blades.key();
        if self
            .expand_group
            .as_ref()
            .is_none_or(|(cached, _)| *cached != key)
        {
            let mut entries = blades.entries(&self.select.slots);
            entries.extend([
                BindGroupEntry {
                    binding: 9,
                    resource: self.params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: self.near.vertices.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: self.near.previous_vertices.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 13,
                    resource: self.far.vertices.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 23,
                    resource: self.far.previous_vertices.as_entire_binding(),
                },
            ]);
            let group =
                render_device.create_bind_group("meadow_rt_expand_bind_group", layout, &entries);
            self.expand_group = Some((key, group));
        }
        let workgroups = RT_NEAR_MAX_BLADES.max(RT_FAR_MAX_BLADES).div_ceil(256);
        passes.record(
            encoder,
            "meadow_rt_expand",
            &self.expand_group.as_ref().unwrap().1,
            &[(RtPass::Expand, workgroups)],
        );
    }

    /// Clear both bands for a frame whose inputs are missing, so streaming
    /// cannot leave stale casters, and start the selection over.
    fn pad(
        &mut self,
        render_device: &RenderDevice,
        encoder: &mut CommandEncoder,
        passes: &RtPasses,
        layout: &BindGroupLayout,
    ) {
        if self.empty_padded {
            return;
        }
        self.empty_padded = true;
        self.selector = RtSelector::default();
        let group = self.pad_group.get_or_insert_with(|| {
            render_device.create_bind_group(
                "meadow_rt_pad_bind_group",
                layout,
                &BindGroupEntries::with_indices((
                    (11, self.near.vertices.as_entire_binding()),
                    (12, self.near.previous_vertices.as_entire_binding()),
                    (13, self.far.vertices.as_entire_binding()),
                    (23, self.far.previous_vertices.as_entire_binding()),
                )),
            )
        });
        let workgroups = RT_NEAR_MAX_BLADES.max(RT_FAR_MAX_BLADES).div_ceil(256);
        passes.record(
            encoder,
            "meadow_rt_clear",
            group,
            &[(RtPass::Pad, workgroups)],
        );
        for band in [&mut self.near, &mut self.far] {
            band.topology_generation = band.topology_generation.wrapping_add(1);
        }
    }

    /// Keep last frame's casters standing still while the variant has no
    /// live patches: the expansion would read no patch.
    fn hold(&self, encoder: &mut CommandEncoder) {
        if self.empty_padded {
            return;
        }
        for band in [&self.near, &self.far] {
            encoder.copy_buffer_to_buffer(
                &band.vertices,
                0,
                &band.previous_vertices,
                0,
                u64::from(band.vertex_count) * RT_VERTEX_SIZE,
            );
        }
    }

    fn stats(&self, variant: MeadowVariantId, rt: &RtExtract) -> MeadowRtVariantStats {
        let state = self.selector.state.unwrap_or_default();
        MeadowRtVariantStats {
            variant,
            exact_only: rt.exact_radius.is_some(),
            requested_radius_m: rt.exact_radius.unwrap_or(RT_NEAR_DISTANCE),
            coverage_radius_m: state.near_reach,
            drawn_radius_m: state.drawn_near_reach,
            near_selected: state.near_selected as usize,
            near_capacity: RT_NEAR_MAX_BLADES as usize,
            far_selected: state.far_selected as usize,
            far_capacity: RT_FAR_MAX_BLADES as usize,
            candidate_count: state.near_candidates as usize,
            pinned: state.pinned as usize,
            dropped: state.dropped as usize,
            topology_changes: self.selector.identity_changes,
            pending: !self.selector.current(),
            selection_center: self.selector.pose().map_or(Vec3::ZERO, |pose| pose.eye),
        }
    }
}

/// Frames a variant may stay inactive — no live patches, or missing from
/// the extract entirely — before its RT buffers and selection are freed (and
/// the consumer's BLAS rebuild stops). Generous enough to ride out streaming
/// gaps; a variant left behind by a world transition stops costing
/// anything.
const RT_INACTIVE_FREE_FRAMES: u32 = 300;

/// Public per-variant RT geometry, for a raytracing consumer.
#[derive(Resource, Default)]
pub struct MeadowRtBuffers {
    pub by_variant: HashMap<MeadowVariantId, RtVariantBuffers>,
    /// The static (near, far) index buffers, shared by every variant.
    shared_indices: Option<(Buffer, Buffer)>,
}

impl MeadowRtBuffers {
    /// Free the buffers and selection of every variant inactive for
    /// [`RT_INACTIVE_FREE_FRAMES`]. A variant missing from `extracted` (no
    /// registry, material churn, a world transition) counts the same as one
    /// with no live patches: only live patches reset its streak.
    pub(crate) fn free_inactive(&mut self, extracted: &MeadowExtractedVariants) {
        self.by_variant.retain(|id, rtv| {
            let active = extracted
                .by_variant
                .get(id)
                .is_some_and(|ev| !ev.active_patches.is_empty());
            rtv.inactive_frames = if active { 0 } else { rtv.inactive_frames + 1 };
            rtv.inactive_frames < RT_INACTIVE_FREE_FRAMES
        });
    }
}

/// Consumers bind geometry after slot generations and wind positions update.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct MeadowRtExpanded;

/// The RT kernels' layouts and pipelines, in [`RT_ENTRIES`] order.
#[derive(Resource)]
pub struct MeadowRtPipelines {
    select_layout: BindGroupLayoutDescriptor,
    expand_layout: BindGroupLayoutDescriptor,
    pad_layout: BindGroupLayoutDescriptor,
    ids: [CachedComputePipelineId; 12],
}

impl MeadowRtPipelines {
    /// The pipelines, once all are compiled, and their layouts.
    fn ready<'a>(&self, pipeline_cache: &'a PipelineCache) -> Option<(RtPasses<'a>, RtLayouts)> {
        let mut pipelines = [None; 12];
        for (pipeline, id) in pipelines.iter_mut().zip(self.ids) {
            *pipeline = Some(pipeline_cache.get_compute_pipeline(id)?);
        }
        Some((
            RtPasses(pipelines.map(Option::unwrap)),
            RtLayouts {
                select: pipeline_cache.get_bind_group_layout(&self.select_layout),
                expand: pipeline_cache.get_bind_group_layout(&self.expand_layout),
                pad: pipeline_cache.get_bind_group_layout(&self.pad_layout),
            },
        ))
    }
}

/// Register the RT expansion resources + systems. Called from
/// `MeadowRenderPlugin`; the `MeadowRaytracingConfig` extract plugin is
/// added on the main app by the caller.
pub fn build_meadow_raytracing(render_app: &mut SubApp) {
    render_app.init_resource::<MeadowRtBuffers>().add_systems(
        Render,
        (
            prepare_meadow_rt_buffers
                .in_set(RenderSystems::PrepareResources)
                .after(prepare_meadow_gpu_buffers),
            dispatch_meadow_rt
                .in_set(RenderSystems::PrepareResources)
                .in_set(MeadowRtExpanded)
                .after(prepare_meadow_rt_buffers)
                // The kernels read this frame's `VariantParams` uniform.
                .after(prepare_meadow_variant_params),
        ),
    );
}

/// The bindings every RT kernel reads blades through: the read-only inputs,
/// the palette table, the cover field and the clearing profiles of
/// `meadow_compute.wesl`'s group 0, and the slots.
fn rt_blade_layout_entries() -> [(
    u32,
    bevy::render::render_resource::BindGroupLayoutEntryBuilder,
); 8] {
    let texture = || texture_2d(TextureSampleType::Float { filterable: false });
    [
        (0, uniform_buffer::<VariantParams>(false)),
        (1, storage_buffer_read_only_sized(false, None)), // patches
        (2, storage_buffer_read_only_sized(false, None)), // trunk_slots
        (3, texture()),                                   // heightfield
        (10, storage_buffer_sized(false, None)),          // rt_slots
        (14, storage_buffer_read_only_sized(false, None)), // palettes
        (15, texture()),                                  // cover_field
        (16, texture()),                                  // clearing_profiles
    ]
}

fn rt_layout(
    label: &'static str,
    extra: &[(
        u32,
        bevy::render::render_resource::BindGroupLayoutEntryBuilder,
    )],
) -> BindGroupLayoutDescriptor {
    let entries = rt_blade_layout_entries()
        .into_iter()
        .chain(extra.iter().copied())
        .map(|(binding, entry)| entry.build(binding, ShaderStages::COMPUTE))
        .collect::<Vec<_>>();
    BindGroupLayoutDescriptor::new(label, &entries)
}

/// The expansion's group 0: the blade bindings and the bands' geometry.
pub(crate) fn rt_expand_layout() -> BindGroupLayoutDescriptor {
    rt_layout(
        "meadow_rt_expand_layout",
        &[
            (9, uniform_buffer::<RtParams>(false)),  // rt_params
            (11, storage_buffer_sized(false, None)), // rt_verts_near
            (12, storage_buffer_sized(false, None)), // rt_previous_near
            (13, storage_buffer_sized(false, None)), // rt_verts_far
            (23, storage_buffer_sized(false, None)), // rt_previous_far
        ],
    )
}

/// The selection's group 0: the blade bindings and the selection's.
pub(crate) fn rt_select_layout() -> BindGroupLayoutDescriptor {
    rt_layout(
        "meadow_rt_select_layout",
        &[
            (17, uniform_buffer::<RtSelectParams>(false)), // rt_select
            (18, storage_buffer_read_only_sized(false, None)), // rt_scan
            (19, storage_buffer_sized(false, None)),       // rt_scratch
            (22, storage_buffer_sized(false, None)),       // rt_drawn_params
        ],
    )
}

/// The pad pass's group 0: the bands' geometry.
pub(crate) fn rt_pad_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "meadow_rt_pad_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (11, storage_buffer_sized(false, None)),
                (12, storage_buffer_sized(false, None)),
                (13, storage_buffer_sized(false, None)),
                (23, storage_buffer_sized(false, None)),
            ),
        ),
    )
}

fn init_meadow_rt_pipelines(
    asset_server: &AssetServer,
    pipeline_cache: &PipelineCache,
) -> MeadowRtPipelines {
    let select_layout = rt_select_layout();
    let expand_layout = rt_expand_layout();
    let pad_layout = rt_pad_layout();
    let shader = asset_server.load(MEADOW_COMPUTE_SHADER);
    let ids = std::array::from_fn(|i| {
        let layout = match i {
            i if i == RtPass::Expand as usize => &expand_layout,
            i if i == RtPass::Pad as usize => &pad_layout,
            _ => &select_layout,
        };
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("meadow_{}", RT_ENTRIES[i]).into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from(RT_ENTRIES[i])),
            ..default()
        })
    });
    MeadowRtPipelines {
        select_layout,
        expand_layout,
        pad_layout,
        ids,
    }
}

/// 64 KiB of quiet-NaN f32s, the prefill pattern for RT vertex buffers.
static NAN_CHUNK: [u32; 16384] = [0x7FC0_0000; 16384];

/// Build one band's static index buffer: slot `s` -> verts
/// `s * verts_per_blade` + the band's triangle pattern. Contents are a pure
/// function of the capacity, so one buffer serves every variant.
pub(crate) fn make_rt_indices(
    render_device: &RenderDevice,
    label: &str,
    blades: u32,
    verts_per_blade: u32,
    index_pattern: &[u32],
) -> Buffer {
    let mut indices = Vec::with_capacity((blades as usize) * index_pattern.len());
    for slot in 0..blades {
        let base = slot * verts_per_blade;
        indices.extend(index_pattern.iter().map(|i| base + i));
    }
    render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&indices),
        usage: BufferUsages::STORAGE | BufferUsages::BLAS_INPUT,
    })
}

/// The static (near, far) index buffers.
pub(crate) fn make_rt_band_indices(render_device: &RenderDevice) -> (Buffer, Buffer) {
    (
        make_rt_indices(
            render_device,
            "meadow_rt_near_indices",
            RT_NEAR_MAX_BLADES,
            RT_NEAR_VERTS_PER_BLADE,
            &RT_NEAR_BLADE_INDICES,
        ),
        make_rt_indices(
            render_device,
            "meadow_rt_far_indices",
            RT_FAR_MAX_BLADES,
            RT_FAR_VERTS_PER_BLADE,
            &RT_FAR_BLADE_INDICES,
        ),
    )
}

fn make_rt_band(
    render_device: &RenderDevice,
    label: &str,
    blades: u32,
    verts_per_blade: u32,
    indices_per_blade: u32,
    indices: Buffer,
) -> RtBandBuffers {
    let vertex_count = blades * verts_per_blade;
    let vertices = render_device.create_buffer(&BufferDescriptor {
        label: Some(&format!("{label}_vertices")),
        size: u64::from(vertex_count) * RT_VERTEX_SIZE,
        usage: BufferUsages::STORAGE | BufferUsages::BLAS_INPUT | BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        // Prefill every f32 with NaN: all slots start as inactive primitives,
        // so the geometry is empty (not a degenerate blob at the origin)
        // until the first expansion writes real blades.
        let mut view = vertices
            .slice(..)
            .get_mapped_range_mut()
            .expect("meadow RT vertex buffer was created mapped");
        let pattern: &[u8] = bytemuck::cast_slice(&NAN_CHUNK);
        let total = view.len();
        let mut off = 0;
        while off < total {
            let len = pattern.len().min(total - off);
            view.slice(off..off + len).copy_from_slice(&pattern[..len]);
            off += len;
        }
    }
    vertices.unmap();
    RtBandBuffers {
        vertices,
        previous_vertices: rt_buffer(
            render_device,
            &format!("{label}_previous_vertices"),
            u64::from(vertex_count) * RT_VERTEX_SIZE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ),
        indices,
        vertex_count,
        index_count: blades * indices_per_blade,
        topology_generation: 0,
    }
}

/// Allocate the fixed-cap RT buffers for every variant with live patches
/// (once), free them for variants that stay inactive past
/// [`RT_INACTIVE_FREE_FRAMES`], and lazily compile the RT pipelines the
/// first frame RT is enabled. No-op when RT is off.
#[allow(clippy::too_many_arguments)]
fn prepare_meadow_rt_buffers(
    mut commands: Commands,
    config: Res<MeadowRaytracingConfig>,
    extracted: Res<MeadowExtractedVariants>,
    slots: Res<MeadowViewSlots>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Option<Res<MeadowRtPipelines>>,
    mut rt: ResMut<MeadowRtBuffers>,
) {
    if !config.enabled {
        rt.by_variant.clear();
        rt.shared_indices = None;
        return;
    }
    if pipelines.is_none() {
        commands.insert_resource(init_meadow_rt_pipelines(&asset_server, &pipeline_cache));
    }
    // No meadow views this frame (a menu with only a 2D camera): nothing
    // expands, so allocate nothing and let the inactivity streaks freeze:
    // buffers ride out the menu and resume the frame a 3D camera reappears.
    if slots.count == 0 {
        return;
    }

    rt.free_inactive(&extracted);
    for (id, ev) in extracted.by_variant.iter() {
        if rt.by_variant.contains_key(id) || ev.active_patches.is_empty() {
            continue;
        }
        let indices = rt
            .shared_indices
            .get_or_insert_with(|| make_rt_band_indices(&render_device))
            .clone();
        rt.by_variant.insert(
            *id,
            RtVariantBuffers::new(&render_device, &render_queue, indices),
        );
    }
}

/// Per variant: read back the selection's state, select when due, ease the
/// drawn limits, and expand the slots' blades — or hold them while the
/// variant has no live patches, or clear them while its inputs are missing.
/// Records into solari's shared `RaytracingProducerEncoder` when its scene
/// plugin is present — one submit covers every raytracing geometry producer,
/// ahead of the BLAS builds — and otherwise into a private encoder it
/// submits, so the passes work without a raytracer. Either way, queue
/// submission order guarantees a consumer's BLAS build (a later render set)
/// sees the finished buffers.
#[allow(clippy::too_many_arguments)]
fn dispatch_meadow_rt(
    config: Res<MeadowRaytracingConfig>,
    pipelines: Option<Res<MeadowRtPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    extracted: Res<MeadowExtractedVariants>,
    slots: Res<MeadowViewSlots>,
    params: Res<MeadowVariantParamsBuffers>,
    buffers: Res<MeadowGpuBuffers>,
    mut rt: ResMut<MeadowRtBuffers>,
    shader_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<bevy::render::texture::FallbackImage>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    #[cfg(all(feature = "solari", not(target_family = "wasm")))] mut producer_encoder: Option<
        ResMut<RaytracingProducerEncoder>,
    >,
    diagnostics: Option<Res<DiagnosticsRecorder>>,
) {
    // No meadow views this frame (a menu with only a 2D camera): the
    // sibling prepares idle too, leaving the params uniform frozen and the
    // GPU active-patch list stale. Idle instead; the consumer tolerates
    // frames with no dispatch.
    if !config.enabled || rt.by_variant.is_empty() || slots.count == 0 {
        return;
    }
    let Some(pipelines) = pipelines else {
        return;
    };
    let Some((passes, layouts)) = pipelines.ready(&pipeline_cache) else {
        return; // pipelines still compiling; buffers stay NaN (empty geometry)
    };
    if rt.by_variant.values().any(|rtv| rtv.select.mapping()) {
        let _ = render_device.poll(PollType::Poll);
    }
    for rtv in rt.by_variant.values_mut() {
        rtv.poll_readbacks();
    }
    let frames: Vec<(MeadowVariantId, RtFrameInputs)> = rt
        .by_variant
        .keys()
        .map(|id| {
            let frame = match extracted.by_variant.get(id) {
                Some(ev) if ev.active_patches.is_empty() => RtFrameInputs::Hold,
                Some(ev) => ev
                    .rt
                    .as_ref()
                    .zip(ev.blades(
                        id,
                        &buffers,
                        &params,
                        &shader_buffers,
                        &gpu_images,
                        &fallback_image,
                    ))
                    .map_or(RtFrameInputs::Pad, |(extract, blades)| {
                        RtFrameInputs::Draw(blades, extract)
                    }),
                None => RtFrameInputs::Pad,
            };
            (*id, frame)
        })
        .collect();
    // Fully idle frames encode nothing.
    if frames.iter().all(|(id, frame)| {
        !matches!(frame, RtFrameInputs::Draw(..)) && rt.by_variant[id].empty_padded
    }) {
        return;
    }

    let mut own_encoder: Option<CommandEncoder> = None;
    let new_encoder = || {
        render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("meadow_rt_encoder"),
        })
    };
    #[cfg(all(feature = "solari", not(target_family = "wasm")))]
    let encoder = match producer_encoder.as_mut() {
        Some(shared) => shared.encoder(&render_device),
        None => own_encoder.insert(new_encoder()),
    };
    #[cfg(not(all(feature = "solari", not(target_family = "wasm"))))]
    let encoder = own_encoder.insert(new_encoder());

    let diagnostics = diagnostics.as_deref();
    for (id, frame) in &frames {
        if let RtFrameInputs::Draw(blades, rt_frame) = frame {
            let rtv = rt.by_variant.get_mut(id).unwrap();
            rtv.select(
                &render_device,
                &render_queue,
                encoder,
                &passes,
                &layouts.select,
                blades,
                rt_frame,
                diagnostics,
            );
        }
    }
    let span = diagnostics.time_span(encoder, "meadow_rt_expand");
    let mut stats = Vec::new();
    for (id, frame) in &frames {
        let rtv = rt.by_variant.get_mut(id).unwrap();
        match frame {
            RtFrameInputs::Draw(blades, rt_frame) => {
                rtv.expand(&render_device, encoder, &passes, &layouts.expand, blades);
                stats.push(rtv.stats(*id, rt_frame));
            }
            RtFrameInputs::Hold => rtv.hold(encoder),
            RtFrameInputs::Pad => rtv.pad(&render_device, encoder, &passes, &layouts.pad),
        }
    }
    span.end(encoder);
    if let Ok(mut published) = extracted.telemetry.lock() {
        *published = stats;
    }

    // Standalone fallback only — the shared producer encoder is submitted
    // by solari, once, ahead of the BLAS builds.
    if let Some(own) = own_encoder {
        render_queue.submit([own.finish()]);
    }
}

#[cfg(test)]
mod rt_gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_if_changed_keeps_pointer_for_equal_contents() {
        let mut list = Arc::<Vec<u32>>::default();
        let empty = Arc::clone(&list);
        replace_if_changed(&mut list, Vec::new());
        assert!(Arc::ptr_eq(&list, &empty));
        replace_if_changed(&mut list, vec![1, 2]);
        assert!(!Arc::ptr_eq(&list, &empty));
        let filled = Arc::clone(&list);
        replace_if_changed(&mut list, vec![1, 2]);
        assert!(Arc::ptr_eq(&list, &filled));
    }

    /// The task list must cover every blade the kernels' prefix gate passes
    /// in any view, for any viewer within the step of the build position.
    #[test]
    fn task_slice_bound_covers_the_gate_within_the_viewer_step() {
        let variant = MeadowVariant {
            blades_per_m2: 500.0,
            lod: MeadowLodCurve {
                full_distance: 15.0,
                ..MeadowLodCurve::default()
            },
            ..MeadowVariant::default()
        };
        let lod = &variant.lod;
        let (tuft_near, tuft_far) = variant.tuft_survival();
        let blade_count = 40_000;
        // Mirror of `meadow_band_threshold`: main views use the near or tuft
        // arm, shadow views the shadow arm of band 0 (tufts never cast). The
        // band follows the centre's distance `band_dist`, the threshold
        // the gate distance `dist`.
        let gate = |band_dist: f32, dist: f32, shadow: bool| -> u32 {
            let threshold = if band_dist >= lod.tuft_start {
                if shadow {
                    return 0;
                }
                let t = ((dist - lod.tuft_start) / (lod.max_view_distance - lod.tuft_start))
                    .clamp(0.0, 1.0);
                tuft_near + (tuft_far - tuft_near) * t
            } else if shadow {
                shadow_survival(lod, dist)
            } else {
                lod.near_survival(dist)
            };
            gate_count(threshold, blade_count)
        };
        let centre = Vec2::new(3.0, -2.0);
        // A disc gates at its centre, a tile at its nearest corner.
        for (slack, step) in [(0.0, 0..260), (4.0 * std::f32::consts::SQRT_2, 0..260)]
            .into_iter()
            .flat_map(|(slack, steps)| steps.map(move |step| (slack, step)))
        {
            let built = centre + Vec2::new(step as f32 * 0.5, 0.0);
            for shadows in [false, true] {
                let bound = task_slice_bound(&variant, centre, slack, blade_count, built, shadows);
                for k in 0..16 {
                    let angle = k as f32 * std::f32::consts::TAU / 16.0;
                    let viewer = built + Vec2::from_angle(angle) * TASK_LIST_VIEWER_STEP * 0.999;
                    let band_dist = centre.distance(viewer);
                    let dist = (band_dist - slack).max(0.0);
                    let needed = gate(band_dist, dist, false).max(if shadows {
                        gate(band_dist, dist, true)
                    } else {
                        0
                    });
                    assert!(
                        bound >= needed,
                        "built at {built:?}, viewer {viewer:?}: bound {bound} < {needed}"
                    );
                }
            }
        }
        // Shadow casters thin with player distance, so the band-0 scan
        // shrinks before `tuft_start` with shadow views too.
        let with_shadows = task_slice_bound(
            &variant,
            centre,
            0.0,
            blade_count,
            centre + Vec2::X * 50.0,
            true,
        );
        assert!(with_shadows < blade_count / 2, "{with_shadows}");
        // Far patches scan only their tuft share.
        let far = task_slice_bound(
            &variant,
            centre,
            0.0,
            blade_count,
            centre + Vec2::X * 90.0,
            false,
        );
        assert_eq!(far, gate_count(tuft_near, blade_count));
    }

    /// The kernels' near arm thins like `MeadowLodCurve::near_survival`.
    #[test]
    fn near_survival_matches_the_shader() {
        let shared = include_str!("meadow_shared.wesl");
        for needle in [
            "let ramp = clamp((dist - lod_full) / max(tuft_start - lod_full, 1e-3), 0.0, 1.0);",
            "return mix(1.0, 1.0 / params.lod.w, ramp);",
        ] {
            assert!(shared.contains(needle), "`{needle}` missing");
        }
        let lod = MeadowLodCurve::default();
        assert_eq!(lod.near_survival(lod.full_distance), 1.0);
        assert!((lod.near_survival(lod.tuft_start) - 1.0 / lod.max_widen).abs() < 1e-6);
    }

    /// Caps past the binding limit shrink to fit it, in proportion.
    #[test]
    fn caps_fit_the_binding_limit() {
        let limit = 2_147_483_644;
        let roomy = (CAP_GRAN, CAP_GRAN, CAP_GRAN);
        assert_eq!(fit_caps(roomy, limit), roomy);
        let caps = (24 * CAP_GRAN, 2 * CAP_GRAN, 8 * CAP_GRAN);
        let (near, far, shadow) = fit_caps(caps, limit);
        let records =
            u64::from(near) + u64::from(far) + (MEADOW_MAX_VIEWS as u64 - 1) * u64::from(shadow);
        assert!(records * BLADE_RECORD_SIZE <= limit, "{records}");
        // In proportion: 12 near per far, 3 per shadow slot, to rounding.
        assert_eq!((near / far, near / shadow), (12, 3));
    }

    /// Tufts keep their count per area at any blade density.
    #[test]
    fn tuft_survival_scales_with_blade_density() {
        let reference = MeadowVariant {
            blades_per_m2: crate::plugin::TUFT_REFERENCE_BLADES_PER_M2,
            ..MeadowVariant::default()
        };
        let dense = MeadowVariant {
            blades_per_m2: 600.0,
            ..MeadowVariant::default()
        };
        let (near, far) = reference.tuft_survival();
        assert_eq!(
            (near, far),
            (
                reference.lod.tuft_density_near,
                reference.lod.tuft_density_far
            )
        );
        let (dense_near, dense_far) = dense.tuft_survival();
        assert!((dense_near * 600.0 - near * reference.blades_per_m2).abs() < 1e-4);
        assert!((dense_far * 600.0 - far * reference.blades_per_m2).abs() < 1e-4);
    }
}
