//! Mesh-shader render path (`mesh-shaders` cargo feature) — the faster
//! path (about 2x) where the GPU supports it.
//!
//! A full port of the meadow renderer to task/mesh pipelines: the
//! 128-wide task stage does the cull / gate / derive / compact work the
//! compute kernel does today and hands fully-derived survivors to
//! small pure-expander mesh workgroups through the task payload — no
//! intermediate `out_blades` buffer, no cursors, no indirect args
//! (see `meadow_mesh.wesl` for the stage-split rationale). It serves
//! BOTH the main camera view — forward (full PBR fragment + motion
//! vectors) or deferred (G-buffer fragment drawn straight into bevy's
//! deferred prepass attachments, shaded by the deferred lighting pass /
//! solari like every other deferred surface) — and the directional
//! shadow cascades (depth-only proxy-silhouette pipelines drawn into
//! each cascade after bevy's shadow pass).
//!
//! The compute path stays intact behind [`MeadowForceComputePath`] (force
//! it at runtime to compare) and is the automatic fallback when
//! `EXPERIMENTAL_MESH_SHADER` is absent (pre-Turing/pre-RDNA2,
//! non-Vulkan): [`MeadowMeshPathActive`] flips
//! per frame, `DrawMeadowPatch` skips the views the mesh path serves, and
//! `prepare_meadow_gpu_buffers` collapses the per-view blade regions to
//! zero so the VRAM saving is real.
//!
//! The pipelines are bevy mesh pipelines ([`MeshPipelineDescriptor`]
//! queued on the [`PipelineCache`]), built against Bevy's own frame data
//! rather than a rebuilt copy of it:
//!
//! - The **PBR fragments** (`meadow_mesh_pbr_fragment.wesl`,
//!   `meadow_mesh_deferred_fragment.wesl`) compile with the shader defs
//!   lifted verbatim from the meadow material's own specialized pipelines
//!   ([`SpecializedMaterialPipelineCache`] /
//!   [`SpecializedPrepassMaterialPipelineCache`]), so their view-binding
//!   declarations match the real view bind group layouts by construction.
//! - Bind groups 0/1 of the main pipeline are Bevy's own
//!   [`MeshViewBindGroup`] (view uniforms, lights, shadow maps, clusters,
//!   probes), giving full lighting/shadow-receive parity with the compute
//!   path's `ExtendedMaterial` fragment.
//! - The per-view cull data is the same `view_cull` storage buffer the
//!   compute path uploads.

use std::hash::{Hash, Hasher};

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::camera::{Camera3d, MainPassResolutionOverride, Viewport};
use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::core_3d::{
    CORE_3D_DEPTH_FORMAT, main_opaque_pass_3d, main_transparent_pass_3d,
};
use bevy::core_pipeline::deferred::copy_lighting_id::copy_deferred_lighting_id;
use bevy::core_pipeline::deferred::node::late_deferred_prepass;
use bevy::core_pipeline::prepass::{
    DeferredPrepass, MOTION_VECTOR_PREPASS_FORMAT, MotionVectorPrepass, PreviousViewData,
    PreviousViewUniformOffset, ViewPrepassTextures,
};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::resource::Resource;
use bevy::material::descriptor::{MeshPipelineDescriptor, MeshState, TaskState};
use bevy::pbr::{
    LATE_SHADOW_PASS, LightEntity, MeshViewBindGroup, PrepassViewBindGroup, ShadowView,
    SpecializedMaterialPipelineCache, SpecializedPrepassMaterialPipelineCache, ViewLightEntities,
    per_view_shadow_pass,
};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::camera::{ExtractedCamera, TemporalJitter};
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only, storage_buffer_read_only_sized, texture_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntries, BufferBinding, BufferId, CachedPipelineState, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, DynamicUniformBuffer,
    Extent3d, FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderStages, ShaderType, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewId, WgpuFeatures,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::sync_world::MainEntity;
use bevy::render::texture::GpuImage;
use bevy::render::view::{
    ExtractedView, Msaa, ViewDepthStencilTexture, ViewTarget, ViewUniformOffset,
};
use bevy::render::{Render, RenderStartup, RenderSystems};
use bevy::shader::{Shader, ShaderDefVal, load_shader_library};

use crate::compute::{
    MeadowExtractedVariants, MeadowGpuBuffers, MeadowMeshPathActive, MeadowVariantParamsBuffers,
    MeadowViewCullData, MeadowViewSlots, RenderMeadowDriver, build_meadow_view_slots,
    prepare_meadow_gpu_buffers,
};
use crate::material::VariantParams;
use crate::mesh::{
    MEADOW_MAX_VIEWS, MESH_OUT_PRIMS, MESH_OUT_VERTS, MESH_SURVIVOR_BLADE_BYTES, MESH_TASK_BLADES,
    MESH_TASK_DISPATCH_STRIDE, MESH_WG_TUFTS,
};
use crate::plugin::MeadowVariantId;

/// Main-world toggle: force the compute path even when the mesh path is
/// available. Flip at runtime to compare paths (the switch takes one
/// frame; blade placement is hash-identical between paths, so the field
/// should not visibly change beyond the shading-model delta).
#[derive(Resource, Default, Clone, Copy, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct MeadowForceComputePath(pub bool);

/// Bind-group index of the meadow resources in every pipeline built from
/// the geometry module (the WGSL hardcodes `@group(3)`): the main
/// pipeline has bevy_pbr's view groups at 0/1 and an empty group 2; the
/// shadow pipeline fills slots 0-2 with empty bind groups so one module
/// serves both layouts.
const MEADOW_GROUP: u32 = 3;

/// Format of the meadow-owned motion target (xy = motion vector, z =
/// valid flag): color target 1 of the forward main pipeline and the
/// texture `prepare_meadow_mesh_mv_target` allocates for it.
const MEADOW_MV_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

// ---------- per-view uniform ----------

/// Rust mirror of `MeadowMeshView` in `meadow_mesh_bindings.wesl`. One
/// entry per meadow view slot, each bound by its own bind group (see
/// [`meadow_mesh_bgl_desc`] for why not a dynamic offset).
#[derive(ShaderType, Clone, Copy)]
pub struct MeadowMeshViewUniform {
    /// Rasterization matrix (includes temporal jitter when active).
    pub clip_from_world: Mat4,
    /// Jitter-free current matrix (motion-vector numerator).
    pub unjittered_clip_from_world: Mat4,
    /// Previous frame's jitter-free matrix (motion-vector denominator).
    pub prev_clip_from_world: Mat4,
    /// x = view slot (indexes `view_cull.views`); yzw reserved.
    pub params: UVec4,
}

impl Default for MeadowMeshViewUniform {
    fn default() -> Self {
        Self {
            clip_from_world: Mat4::IDENTITY,
            unjittered_clip_from_world: Mat4::IDENTITY,
            prev_clip_from_world: Mat4::IDENTITY,
            params: UVec4::ZERO,
        }
    }
}

/// Per-frame per-view-slot uniforms + their byte offsets in `buffer`,
/// slot-indexed to match [`MeadowViewSlots`].
#[derive(Resource, Default)]
pub struct MeadowMeshViewUniforms {
    pub buffer: DynamicUniformBuffer<MeadowMeshViewUniform>,
    pub offsets: [u32; MEADOW_MAX_VIEWS],
}

// ---------- pipelines resource ----------

/// Key for the main-view mesh pipeline. `color_format`/`samples` come
/// from the meadow material's specialized forward pipeline descriptor
/// (so they match the pass bevy renders opaque with).
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct MeadowMainPipelineKey {
    pub color_format: TextureFormat,
    pub samples: u32,
    /// Adds the meadow-owned motion target as color target 1 (MV prepass
    /// active + single-sample). The composite pass then copies it into
    /// bevy's MV texture.
    pub mv: bool,
    /// Hash of the specialized meadow pipeline's shader defs. The defs
    /// determine the view bind group LAYOUT (prepass texture entries
    /// appear when DLSS/TAA enables the prepasses), so a def change must
    /// produce a new pipeline — binding the new `MeshViewBindGroup` with
    /// a pipeline built against the old layout fails validation and
    /// silently drops every draw.
    pub defs_hash: u64,
}

/// Key for the deferred-view mesh pipeline. Derived from the meadow
/// material's specialized deferred prepass pipeline: `defs_hash` covers
/// everything the descriptor varies on (target list, view layout), while
/// `mv`/`normal` are lifted out for the pass to select the matching
/// prepass view bind group and attachment slots.
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct MeadowDeferredPipelineKey {
    /// Hash of the specialized deferred pipeline's shader defs (same
    /// layout-mismatch rationale as [`MeadowMainPipelineKey::defs_hash`]).
    pub defs_hash: u64,
    /// MOTION_VECTOR_PREPASS: the pipeline was built against the
    /// motion-vectors prepass view layout and writes color target 1.
    pub mv: bool,
    /// NORMAL_PREPASS: the pipeline writes color target 0.
    pub normal: bool,
}

/// The mesh path's shader handles and the lazily-queued pipelines
/// (ids into bevy's [`PipelineCache`]).
#[derive(Resource, Default)]
pub struct MeadowMeshPipelines {
    /// Adapter supports `EXPERIMENTAL_MESH_SHADER` + our output budgets.
    pub supported: bool,
    /// Task + mesh module (`meadow_mesh.wesl`; meadow group at
    /// [`MEADOW_GROUP`]) — one module serves every meadow mesh pipeline.
    pub geom_shader: Handle<Shader>,
    /// Forward fragment (entries `fragment`/`fragment_mv`), used on
    /// non-deferred main views. A compile failure (bevy logs the shader
    /// error) leaves the main view to the compute path.
    pub forward_fragment_shader: Handle<Shader>,
    /// Deferred G-buffer fragment (entry `fragment_deferred`), used on
    /// deferred main views; same failure handling.
    pub deferred_fragment_shader: Handle<Shader>,
    /// Empty bind group for the shadow pipeline's unused slots 0-2 (and
    /// layout-compatible with the main pipeline's empty slot 2).
    pub empty_bind_group: Option<BindGroup>,
    /// The meadow bind group layout ([`meadow_mesh_bgl_desc`]), resolved
    /// once so the per-frame bind group prepare doesn't rebuild and
    /// re-hash the descriptor.
    pub meadow_layout: Option<BindGroupLayout>,
    /// Pipelines by key. Entries for def sets a view no longer uses stay
    /// (the cache keeps them alive anyway), like bevy's own specialized
    /// pipeline maps.
    pub main_pipelines: HashMap<MeadowMainPipelineKey, CachedRenderPipelineId>,
    pub deferred_pipelines: HashMap<MeadowDeferredPipelineKey, CachedRenderPipelineId>,
    /// Key the current frame's main view resolves to when it renders
    /// forward (None until known, and always None on deferred views —
    /// exactly one of `current_main_key`/`current_deferred_key` is set
    /// per frame, which is what routes the pass systems).
    /// `decide_meadow_mesh_path` requires the matching pipeline to exist.
    pub current_main_key: Option<MeadowMainPipelineKey>,
    /// Deferred-view counterpart of `current_main_key`.
    pub current_deferred_key: Option<MeadowDeferredPipelineKey>,
    pub shadow_pipeline: Option<CachedRenderPipelineId>,
    /// Fullscreen composite copying the meadow-owned motion target's
    /// valid texels into bevy's MV prepass texture (which can't be an
    /// attachment of the main pass — it rides inside the mesh-view bind
    /// group as a sampled resource). Queued once at startup.
    pub composite_pipeline: Option<CachedRenderPipelineId>,
}

/// Meadow-owned motion target: color target 1 of the main pass
/// (Rgba16Float; xy = motion vector, z = valid flag), sized to the main
/// view's physical target and composited into bevy's Rg16Float MV
/// prepass texture right after. Exists only in single-sample MV configs
/// (DLSS/TAA).
#[derive(Resource, Default)]
pub struct MeadowMeshMvTarget {
    pub view: Option<TextureView>,
    pub size: UVec2,
    /// Bind group for the composite pass (references `view`; the view
    /// keeps the underlying texture alive).
    pub composite_bind_group: Option<BindGroup>,
}

/// Identity fingerprint of a variant's mesh-path bind groups (same
/// rationale as the compute path's `ComputeBindGroupFingerprint`), plus
/// the view-slot count the per-slot groups were built for.
type MeshBindGroupFingerprint = ([BufferId; 6], TextureViewId, u32);

/// Per variant, one meadow bind group per view slot (`[slot]`), differing
/// only in which `MeadowMeshViewUniform` entry binding 6 points at.
#[derive(Resource, Default)]
pub struct MeadowMeshBindGroups {
    pub by_variant: HashMap<MeadowVariantId, (Vec<BindGroup>, MeshBindGroupFingerprint)>,
}

// ---------- plugin wiring ----------

/// Register the mesh-path shaders as embedded assets on the main app.
/// Called from `MeadowRenderPlugin::build` under
/// `cfg(feature = "mesh-shaders")`. The bindings and PBR-common modules
/// are libraries the WESL roots import; the rest are pipeline roots
/// loaded by handle in [`init_meadow_mesh_path`].
pub fn register_meadow_mesh_shaders(app: &mut App) {
    load_shader_library!(app, "meadow_mesh_bindings.wesl");
    load_shader_library!(app, "meadow_mesh_pbr_common.wesl");
    embedded_asset!(app, "meadow_mesh.wesl");
    embedded_asset!(app, "meadow_mesh_pbr_fragment.wesl");
    embedded_asset!(app, "meadow_mesh_deferred_fragment.wesl");
    embedded_asset!(app, "meadow_mv_composite.wgsl");
}

/// Register the mesh-path resources + systems on the render sub-app.
/// Called from `MeadowRenderPlugin::build` under
/// `cfg(feature = "mesh-shaders")`; the main-world
/// [`MeadowForceComputePath`] toggle is initialized there too.
pub fn build_meadow_mesh_path(render_app: &mut SubApp) {
    render_app
        .init_resource::<MeadowForceComputePath>()
        .init_resource::<MeadowMeshViewUniforms>()
        .init_resource::<MeadowMeshPipelines>()
        .init_resource::<MeadowMeshBindGroups>()
        .init_resource::<MeadowMeshMvTarget>()
        .add_systems(RenderStartup, init_meadow_mesh_path)
        .add_systems(
            Render,
            (
                prepare_meadow_mesh_view_uniforms
                    .in_set(RenderSystems::PrepareResources)
                    .after(build_meadow_view_slots),
                prepare_meadow_mesh_pipelines.in_set(RenderSystems::PrepareResources),
                prepare_meadow_mesh_mv_target
                    .in_set(RenderSystems::PrepareResources)
                    .after(prepare_meadow_mesh_pipelines),
                // The path decision gates the compute path's buffer prep
                // (cap zeroing), so it must land in between.
                decide_meadow_mesh_path
                    .in_set(RenderSystems::PrepareResources)
                    .after(prepare_meadow_mesh_mv_target)
                    .before(prepare_meadow_gpu_buffers),
                prepare_meadow_mesh_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            ),
        )
        // Pass systems live in the per-camera Core3d schedule.
        .add_systems(
            Core3d,
            (
                meadow_mesh_shadow_pass
                    .after(per_view_shadow_pass::<LATE_SHADOW_PASS>)
                    .before(Core3dSystems::MainPass),
                meadow_mesh_main_pass
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d),
                meadow_mesh_mv_composite_pass
                    .after(meadow_mesh_main_pass)
                    .before(main_transparent_pass_3d),
                // G-buffer contribution: after every deferred mesh draw
                // (late prepass included), before the lighting-id copy —
                // and thereby before bevy's deferred lighting and
                // solari's G-buffer consumption (both in the MainPass
                // set).
                meadow_mesh_deferred_pass
                    .after(late_deferred_prepass)
                    .before(copy_deferred_lighting_id),
            ),
        );
}

// ---------- RenderStartup: detection + static pipelines ----------

/// Meadow bind group layout. All entries visible to TASK|MESH|FRAGMENT —
/// the fragment only reads bindings 0 (palette) and 6 (MV matrices), but
/// a superset visibility is valid and keeps this a single layout.
///
/// Binding 6 selects the view by BIND GROUP (one group per view slot,
/// see [`prepare_meadow_mesh_bind_groups`]), not by dynamic offset. With
/// a dynamic offset here, the task stage reads the wrong data whenever an
/// earlier group in the pipeline layout holds dynamic uniform buffers
/// that are not visible to the task stage — exactly bevy's view layout at
/// group 0 (vertex/fragment visibility). Observed on NVIDIA/Vulkan
/// (driver 616.92, wgpu 30): the camera-view task stage saw another
/// dynamic buffer's contents as `mesh_view`, culled everything, and the
/// shadow pipeline (empty groups 0-2) was unaffected. Non-dynamic
/// uniform and storage bindings in this group read correctly.
fn meadow_mesh_bgl_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "meadow_mesh_path_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::TASK | ShaderStages::MESH | ShaderStages::FRAGMENT,
            (
                uniform_buffer::<VariantParams>(false), // 0 variant_params
                storage_buffer_read_only_sized(false, None), // 1 patches
                storage_buffer_read_only_sized(false, None), // 2 trunk_slots
                texture_2d(TextureSampleType::Float { filterable: false }), // 3 heightfield
                storage_buffer_read_only::<MeadowViewCullData>(false), // 4 view_cull
                storage_buffer_read_only_sized(false, None), // 5 task_slices
                uniform_buffer::<MeadowMeshViewUniform>(false), // 6 mesh_view (per-slot group)
            ),
        ),
    )
}

/// Empty layout for the pipeline slots the meadow groups don't use.
fn empty_bgl_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new("meadow_mesh_empty_layout", &[])
}

/// Layout of the MV composite pass: the meadow-owned motion target.
fn composite_bgl_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "meadow_mv_composite_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

fn init_meadow_mesh_path(
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<MeadowMeshPipelines>,
    mut path_state: ResMut<MeadowMeshPathActive>,
) {
    let features = render_device.features();
    let limits = render_device.limits();
    // Budgets from `meadow_mesh.wesl`, derived from the shared consts:
    // MESH_TASK_BLADES-wide task workgroups passing a payload of
    // MESH_TASK_BLADES fully-derived survivors, mesh workgroups emitting
    // ≤MESH_OUT_VERTS/≤MESH_OUT_PRIMS, and at most
    // ceil(MESH_TASK_BLADES / MESH_WG_TUFTS) mesh workgroups per task
    // workgroup (tufts pack fewest blades).
    let supported = features.contains(WgpuFeatures::EXPERIMENTAL_MESH_SHADER)
        && limits.max_task_invocations_per_workgroup >= MESH_TASK_BLADES
        && limits.max_task_payload_size >= 16 + MESH_TASK_BLADES * MESH_SURVIVOR_BLADE_BYTES
        && limits.max_mesh_output_vertices >= MESH_OUT_VERTS
        && limits.max_mesh_output_primitives >= MESH_OUT_PRIMS
        && limits.max_mesh_workgroup_total_count >= MESH_TASK_BLADES.div_ceil(MESH_WG_TUFTS);

    if !supported {
        info!(
            "meadow mesh-shader path unavailable (EXPERIMENTAL_MESH_SHADER: {}); using compute path",
            features.contains(WgpuFeatures::EXPERIMENTAL_MESH_SHADER)
        );
        pipelines.supported = false;
        return;
    }
    // The takeover itself is logged by `decide_meadow_mesh_path` when the
    // pipelines land (a shader that fails to compile keeps the compute
    // path, with bevy's error in the log) — don't promise it here.
    info!("meadow mesh-shader path available");

    let asset_server = asset_server.as_ref();
    pipelines.geom_shader = load_embedded_asset!(asset_server, "meadow_mesh.wesl");
    pipelines.forward_fragment_shader =
        load_embedded_asset!(asset_server, "meadow_mesh_pbr_fragment.wesl");
    pipelines.deferred_fragment_shader =
        load_embedded_asset!(asset_server, "meadow_mesh_deferred_fragment.wesl");
    let composite_shader = load_embedded_asset!(asset_server, "meadow_mv_composite.wgsl");

    pipelines.meadow_layout = Some(pipeline_cache.get_bind_group_layout(&meadow_mesh_bgl_desc()));
    // One empty bind group covers the main pipeline's slot 2 and the
    // shadow pipeline's slots 0-2 (wgpu dedups identical layouts, so it's
    // compatible anywhere an empty layout appears).
    let empty_layout = pipeline_cache.get_bind_group_layout(&empty_bgl_desc());
    pipelines.empty_bind_group =
        Some(render_device.create_bind_group(Some("meadow_mesh_empty"), &empty_layout, &[]));

    // Shadow pipeline: the geometry module hardcodes the meadow group at
    // [`MEADOW_GROUP`]; slots 0-2 are empty layouts here (the pass binds
    // the cached empty bind group), so one module serves both the main
    // and shadow pipeline layouts.
    //
    // `unclipped_depth` matters: bevy's directional-shadow pipelines
    // depth-CLAMP ortho casters that sit between the light and the
    // cascade volume, rather than clipping them away. Without it, grass
    // outside a cascade's depth range silently drops out — observed as
    // sparser, flatter grass shadows vs the compute path (which rides
    // bevy's own prepass pipelines).
    let unclipped_depth = features.contains(WgpuFeatures::DEPTH_CLIP_CONTROL);
    pipelines.shadow_pipeline = Some(pipeline_cache.queue_mesh_pipeline(
        meadow_mesh_pipeline_descriptor(
            "meadow_mesh_shadow_pipeline",
            vec![
                empty_bgl_desc(),
                empty_bgl_desc(),
                empty_bgl_desc(),
                meadow_mesh_bgl_desc(),
            ],
            &pipelines.geom_shader,
            "meadow_mesh_shadow",
            None,
            1,
            unclipped_depth,
        ),
    ));

    // MV composite: plain fullscreen pipeline, everything about it is
    // static — queue it once here.
    pipelines.composite_pipeline = Some(pipeline_cache.queue_render_pipeline(
        RenderPipelineDescriptor {
            label: Some("meadow_mv_composite_pipeline".into()),
            layout: vec![composite_bgl_desc()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: composite_shader,
                entry_point: Some("fs".into()),
                targets: vec![Some(ColorTargetState {
                    format: MOTION_VECTOR_PREPASS_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        },
    ));

    pipelines.supported = true;
    // Lets the compute path's prepare (always compiled) keep the task
    // work list warm for this device.
    path_state.available = true;
}

/// Create/resize the meadow-owned motion target for the main view, in
/// single-sample MV-prepass configs (DLSS/TAA).
fn prepare_meadow_mesh_mv_target(
    views: Query<
        (
            &ExtractedCamera,
            &Msaa,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        (With<Camera3d>, Without<LightEntity>),
    >,
    pipelines: Res<MeadowMeshPipelines>,
    force: Res<MeadowForceComputePath>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut target: ResMut<MeadowMeshMvTarget>,
) {
    if !pipelines.supported {
        return;
    }
    // Forced-compute: the meadow-owned MV target would sit unused — drop
    // it (a full-resolution Rgba16Float texture) and recreate it the
    // frame the force flag unflips.
    if force.0 {
        if target.view.is_some() {
            *target = MeadowMeshMvTarget::default();
        }
        return;
    }
    // Deferred views don't route through this target: the deferred pass
    // writes motion vectors straight into bevy's MV prepass attachment
    // (its fragment binds no bind group that samples it, so there is no
    // bound-vs-attached conflict to composite around).
    let wanted = views
        .iter()
        .next()
        .and_then(|(camera, msaa, has_mv, deferred)| {
            (has_mv && msaa.samples() == 1 && !deferred)
                .then_some(camera.physical_target_size)
                .flatten()
        });
    let Some(size) = wanted else {
        if target.view.is_some() {
            *target = MeadowMeshMvTarget::default();
        }
        return;
    };
    if target.view.is_some() && target.size == size {
        return;
    }
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("meadow_mesh_motion_target"),
        size: Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: MEADOW_MV_TARGET_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let composite_bind_group = render_device.create_bind_group(
        Some("meadow_mv_composite_bind_group"),
        &pipeline_cache.get_bind_group_layout(&composite_bgl_desc()),
        &BindGroupEntries::single(&view),
    );
    *target = MeadowMeshMvTarget {
        view: Some(view),
        size,
        composite_bind_group: Some(composite_bind_group),
    };
}

// ---------- Prepare: view uniforms ----------

fn prepare_meadow_mesh_view_uniforms(
    slots: Res<MeadowViewSlots>,
    views: Query<(
        &ExtractedView,
        Option<&TemporalJitter>,
        Option<&PreviousViewData>,
        Option<&MainPassResolutionOverride>,
    )>,
    pipelines: Res<MeadowMeshPipelines>,
    force: Res<MeadowForceComputePath>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut out: ResMut<MeadowMeshViewUniforms>,
) {
    // The matrices feed only the mesh-path pipelines — skip the assembly
    // + upload while the compute path is pinned to serve every view.
    if !pipelines.supported || force.0 {
        return;
    }
    let mut entries = [MeadowMeshViewUniform::default(); MEADOW_MAX_VIEWS];
    for (view, jitter, prev, resolution_override) in views.iter() {
        let Some(&slot) = slots.by_retained.get(&view.retained_view_entity) else {
            continue;
        };
        // Mirror of bevy's `prepare_view_uniforms` matrix assembly
        // (bevy_render/src/view/mod.rs) so the mesh path rasterizes with
        // exactly the matrices the rest of the frame uses — including the
        // DLSS render-resolution override, which scales the jitter.
        let unjittered_projection = view.clip_from_view;
        let mut clip_from_view = unjittered_projection;
        if let Some(jitter) = jitter {
            let viewport = resolution_override.map_or_else(
                || Vec2::new(view.viewport.z as f32, view.viewport.w as f32),
                |o| o.0.as_vec2(),
            );
            jitter.jitter_projection(&mut clip_from_view, viewport);
        }
        let world_from_view = view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();
        let clip_from_world = if jitter.is_some() {
            clip_from_view * view_from_world
        } else {
            view.clip_from_world
                .unwrap_or(clip_from_view * view_from_world)
        };
        let unjittered_clip_from_world = unjittered_projection * view_from_world;
        let prev_clip_from_world = prev
            .map(|p| p.unjittered_clip_from_world)
            .unwrap_or(unjittered_clip_from_world);
        entries[slot as usize] = MeadowMeshViewUniform {
            clip_from_world,
            unjittered_clip_from_world,
            prev_clip_from_world,
            params: UVec4::new(slot, 0, 0, 0),
        };
    }

    out.buffer.clear();
    for (slot, entry) in entries.iter().enumerate().take(slots.count as usize) {
        out.offsets[slot] = out.buffer.push(entry);
    }
    if slots.count > 0 {
        out.buffer.write_buffer(&render_device, &render_queue);
    }
}

// ---------- Prepare: pipelines ----------

fn hash_defs(defs: &[ShaderDefVal]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    defs.hash(&mut hasher);
    hasher.finish()
}

/// Whether the cached pipeline failed to build (shader compile error or
/// pipeline creation error — bevy logs the details itself).
fn pipeline_failed(pipeline_cache: &PipelineCache, id: CachedRenderPipelineId) -> bool {
    matches!(
        pipeline_cache.get_render_pipeline_state(id),
        CachedPipelineState::Err(_)
    )
}

/// The compiled GPU pipeline behind a cached id, once the cache has
/// built it.
fn compiled_pipeline(
    pipeline_cache: &PipelineCache,
    id: Option<CachedRenderPipelineId>,
) -> Option<&RenderPipeline> {
    pipeline_cache.get_render_pipeline(id?)
}

/// The main-view pipeline for `key`, queued on first use: bevy's view
/// main + binding-array layouts exactly as the specialized material
/// pipeline uses them (the pass binds bevy's `MeshViewBindGroup` against
/// them), an empty slot 2 (the material pipeline has mesh/material groups
/// there; we bind an empty group), and the meadow group at
/// [`MEADOW_GROUP`]. The PBR fragment compiles with the material
/// pipeline's own shader defs.
fn ensure_main_pipeline(
    pipelines: &mut MeadowMeshPipelines,
    pipeline_cache: &PipelineCache,
    key: MeadowMainPipelineKey,
    view_layouts: &[BindGroupLayoutDescriptor],
    shader_defs: &[ShaderDefVal],
    color_target: &ColorTargetState,
) -> CachedRenderPipelineId {
    if let Some(&id) = pipelines.main_pipelines.get(&key) {
        return id;
    }
    let mut targets = vec![Some(color_target.clone())];
    if key.mv {
        targets.push(Some(ColorTargetState {
            format: MEADOW_MV_TARGET_FORMAT,
            blend: None,
            write_mask: ColorWrites::ALL,
        }));
    }
    let id = pipeline_cache.queue_mesh_pipeline(meadow_mesh_pipeline_descriptor(
        "meadow_mesh_main_pipeline",
        vec![
            view_layouts[0].clone(),
            view_layouts[1].clone(),
            empty_bgl_desc(),
            meadow_mesh_bgl_desc(),
        ],
        &pipelines.geom_shader,
        "meadow_mesh",
        Some(FragmentState {
            shader: pipelines.forward_fragment_shader.clone(),
            shader_defs: shader_defs.to_vec(),
            entry_point: Some(if key.mv { "fragment_mv" } else { "fragment" }.into()),
            targets,
            ..default()
        }),
        key.samples,
        false,
    ));
    info!("meadow mesh-shader main pipeline queued ({key:?})");
    pipelines.main_pipelines.insert(key, id);
    id
}

#[allow(clippy::too_many_arguments)]
fn prepare_meadow_mesh_pipelines(
    views: Query<
        (
            &ExtractedView,
            &Msaa,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        (With<Camera3d>, Without<LightEntity>),
    >,
    drivers: Res<RenderMeadowDriver>,
    specialized: Res<SpecializedMaterialPipelineCache>,
    prepass_specialized: Res<SpecializedPrepassMaterialPipelineCache>,
    pipeline_cache: Res<PipelineCache>,
    force: Res<MeadowForceComputePath>,
    mut pipelines: ResMut<MeadowMeshPipelines>,
) {
    pipelines.current_main_key = None;
    pipelines.current_deferred_key = None;
    // Forced-compute leaves both keys `None`, so `decide_meadow_mesh_path`
    // can never activate against a stale key — the frame the force flag
    // unflips, the key (and any missing pipeline) is recomputed here
    // before the decision runs.
    if !pipelines.supported || force.0 {
        return;
    }
    let Some((view, msaa, has_mv, has_deferred)) = views.iter().next() else {
        return;
    };

    if has_deferred {
        // The meadow material's specialized DEFERRED PREPASS pipeline for
        // this view is the ground truth: its shader defs determine the
        // prepass view bind group layout AND the G-buffer target list
        // (both must match the fragment's declarations / output
        // locations by construction).
        let Some(view_cache) = prepass_specialized.get(&view.retained_view_entity) else {
            return;
        };
        let Some(&(_, pipeline_id, _)) = drivers
            .by_entity
            .keys()
            .find_map(|e| view_cache.get(&MainEntity::from(*e)))
        else {
            return;
        };
        // The specialized cache hands out ids at queue time, but
        // `get_render_pipeline_descriptor` PANICS for ids the pipeline
        // cache hasn't processed yet (freshly specialized this frame).
        // Waiting for the compiled pipeline is bounds-safe and also
        // guarantees we mirror a descriptor that actually built.
        if pipeline_cache.get_render_pipeline(pipeline_id).is_none() {
            return;
        }
        let descriptor = pipeline_cache.get_render_pipeline_descriptor(pipeline_id);
        let Some(frag) = descriptor.fragment.as_ref() else {
            return;
        };
        // The driver's prepass-cache entry can transiently be a plain
        // depth/normal prepass pipeline (renderer method still
        // settling) — only the deferred permutation carries the
        // G-buffer targets this pipeline mirrors.
        if !frag.shader_defs.contains(&"DEFERRED_PREPASS".into()) {
            return;
        }
        if descriptor.layout.is_empty() {
            return;
        }

        let key = MeadowDeferredPipelineKey {
            defs_hash: hash_defs(&frag.shader_defs),
            mv: frag.shader_defs.contains(&"MOTION_VECTOR_PREPASS".into()),
            normal: frag.shader_defs.contains(&"NORMAL_PREPASS".into()),
        };
        let id = match pipelines.deferred_pipelines.get(&key) {
            Some(&id) => id,
            None => {
                // Groups: the prepass view layout exactly as the
                // specialized deferred pipeline uses it (the pass binds
                // bevy's `PrepassViewBindGroup` against it), empty slots
                // 1-2 (bevy has empty + mesh/material groups there), and
                // the meadow group at [`MEADOW_GROUP`].
                let id = pipeline_cache.queue_mesh_pipeline(meadow_mesh_pipeline_descriptor(
                    "meadow_mesh_deferred_pipeline",
                    vec![
                        descriptor.layout[0].clone(),
                        empty_bgl_desc(),
                        empty_bgl_desc(),
                        meadow_mesh_bgl_desc(),
                    ],
                    &pipelines.geom_shader,
                    "meadow_mesh",
                    Some(FragmentState {
                        shader: pipelines.deferred_fragment_shader.clone(),
                        shader_defs: frag.shader_defs.clone(),
                        entry_point: Some("fragment_deferred".into()),
                        // Bevy's own prepass target list ([normal?, motion?,
                        // gbuffer, lighting-id]) — `None` holes preserved so
                        // target indices match `FragmentOutput`'s locations.
                        targets: frag.targets.clone(),
                        ..default()
                    }),
                    descriptor.multisample.count,
                    false,
                ));
                info!("meadow mesh-shader deferred pipeline queued ({key:?})");
                pipelines.deferred_pipelines.insert(key, id);
                id
            }
        };
        // A failed pipeline (bevy logged the shader error) leaves the key
        // `None` and the compute path keeps serving the view.
        if !pipeline_failed(&pipeline_cache, id) {
            pipelines.current_deferred_key = Some(key);
        }
        return;
    }

    // The meadow material's own specialized forward pipeline for this
    // view: its shader defs + view bind group layouts + color target are
    // the ground truth we build the mesh pipeline against.
    let Some(view_cache) = specialized.get(&view.retained_view_entity) else {
        return;
    };
    let Some(&pipeline_id) = drivers
        .by_entity
        .keys()
        .find_map(|e| view_cache.get(&MainEntity::from(*e)))
    else {
        return;
    };
    // Same bounds-safety rationale as the deferred branch above.
    if pipeline_cache.get_render_pipeline(pipeline_id).is_none() {
        return;
    }
    let descriptor = pipeline_cache.get_render_pipeline_descriptor(pipeline_id);
    let Some(frag) = descriptor.fragment.as_ref() else {
        return;
    };
    let Some(Some(color_target)) = frag.targets.first().cloned() else {
        return;
    };
    if descriptor.layout.len() < 2 {
        return;
    }

    let key = MeadowMainPipelineKey {
        color_format: color_target.format,
        samples: descriptor.multisample.count,
        mv: has_mv && msaa.samples() == 1 && descriptor.multisample.count == 1,
        defs_hash: hash_defs(&frag.shader_defs),
    };
    let id = ensure_main_pipeline(
        &mut pipelines,
        &pipeline_cache,
        key,
        &descriptor.layout,
        &frag.shader_defs,
        &color_target,
    );
    // Same failure handling as the deferred arm.
    if !pipeline_failed(&pipeline_cache, id) {
        pipelines.current_main_key = Some(key);
    }
}

/// Describe a meadow mesh pipeline: shared `meadow_task` stage and the
/// depth state every meadow raster pass uses — Depth32Float,
/// `GreaterEqual` (reverse-Z), write on, no stencil/bias (shadow bias is
/// applied receiver-side in bevy's shadow sampling), two-sided
/// primitives.
#[allow(clippy::too_many_arguments)]
fn meadow_mesh_pipeline_descriptor(
    label: &'static str,
    layout: Vec<BindGroupLayoutDescriptor>,
    geom_shader: &Handle<Shader>,
    mesh_entry: &'static str,
    fragment: Option<FragmentState>,
    samples: u32,
    unclipped_depth: bool,
) -> MeshPipelineDescriptor {
    MeshPipelineDescriptor {
        label: Some(label.into()),
        layout,
        immediate_size: 0,
        task: Some(TaskState {
            shader: geom_shader.clone(),
            entry_point: Some("meadow_task".into()),
            ..default()
        }),
        mesh: MeshState {
            shader: geom_shader.clone(),
            entry_point: Some(mesh_entry.into()),
            ..default()
        },
        primitive: PrimitiveState {
            cull_mode: None, // blades are two-sided
            unclipped_depth,
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: default(),
            bias: default(),
        }),
        multisample: MultisampleState {
            count: samples,
            ..default()
        },
        fragment,
        // wgpu's default compilation options, as the pipelines were
        // profiled with.
        zero_initialize_workgroup_memory: true,
    }
}

// ---------- Prepare: path decision ----------

/// Flip [`MeadowMeshPathActive`] for this frame. All-or-nothing: the mesh
/// path takes over main + shadow views together, or the compute path
/// serves everything (mixing per-view paths within a frame is valid, but
/// coupling them keeps path selection and the buffer-cap logic
/// simple).
fn decide_meadow_mesh_path(
    pipelines: Res<MeadowMeshPipelines>,
    pipeline_cache: Res<PipelineCache>,
    force: Res<MeadowForceComputePath>,
    views: Query<Has<DeferredPrepass>, (With<Camera3d>, With<ExtractedView>)>,
    mv_target: Res<MeadowMeshMvTarget>,
    mut active: ResMut<MeadowMeshPathActive>,
) {
    let compiled = |id: Option<&CachedRenderPipelineId>| {
        compiled_pipeline(&pipeline_cache, id.copied()).is_some()
    };
    // `prepare_meadow_mesh_pipelines` set exactly one of the two keys
    // for the main view's current mode (forward color pass vs deferred
    // G-buffer pass) — require that mode's pipeline to have compiled.
    let deferred = views.iter().any(|d| d);
    let main_view_ready = if deferred {
        pipelines
            .current_deferred_key
            .is_some_and(|key| compiled(pipelines.deferred_pipelines.get(&key)))
    } else {
        pipelines.current_main_key.is_some_and(|key| {
            compiled(pipelines.main_pipelines.get(&key)) && (!key.mv || mv_target.view.is_some())
        })
    };
    let ready = pipelines.supported
        && !force.0
        && compiled(pipelines.shadow_pipeline.as_ref())
        && main_view_ready;
    if ready != active.active {
        info!(
            "meadow render path: {}",
            if ready {
                "mesh shaders"
            } else {
                "compute + indirect"
            }
        );
    }
    active.active = ready;
}

// ---------- Prepare: bind groups ----------

#[allow(clippy::too_many_arguments)]
fn prepare_meadow_mesh_bind_groups(
    extracted: Res<MeadowExtractedVariants>,
    shader_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<bevy::render::texture::FallbackImage>,
    params: Res<MeadowVariantParamsBuffers>,
    buffers: Res<MeadowGpuBuffers>,
    view_uniforms: Res<MeadowMeshViewUniforms>,
    slots: Res<MeadowViewSlots>,
    pipelines: Res<MeadowMeshPipelines>,
    force: Res<MeadowForceComputePath>,
    render_device: Res<RenderDevice>,
    mut bind_groups: ResMut<MeadowMeshBindGroups>,
) {
    if !pipelines.supported || force.0 {
        return;
    }
    bind_groups
        .by_variant
        .retain(|id, _| extracted.by_variant.contains_key(id));

    let Some(view_uniform_buffer) = view_uniforms.buffer.buffer() else {
        return;
    };
    let Some(layout) = &pipelines.meadow_layout else {
        return;
    };
    let slot_count = slots.count;
    if slot_count == 0 {
        return;
    }

    for (id, ev) in extracted.by_variant.iter() {
        let Some(patches) = shader_buffers.get(ev.patches) else {
            continue;
        };
        let Some(trunk_slots) = shader_buffers.get(ev.trunk_slots) else {
            continue;
        };
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

        let fingerprint: MeshBindGroupFingerprint = (
            [
                params_buf.id(),
                patches.buffer.id(),
                trunk_slots.buffer.id(),
                view_cull.id(),
                vb.task_slices.id(),
                view_uniform_buffer.id(),
            ],
            heightfield.texture_view.id(),
            slot_count,
        );
        if let Some((_, cached)) = bind_groups.by_variant.get(id)
            && *cached == fingerprint
        {
            continue;
        }

        let per_slot = (0..slot_count as usize)
            .map(|slot| {
                render_device.create_bind_group(
                    Some("meadow_mesh_bind_group"),
                    layout,
                    &BindGroupEntries::sequential((
                        params_buf.as_entire_binding(),
                        patches.buffer.as_entire_binding(),
                        trunk_slots.buffer.as_entire_binding(),
                        &heightfield.texture_view,
                        view_cull.as_entire_binding(),
                        vb.task_slices.as_entire_binding(),
                        BufferBinding {
                            buffer: view_uniform_buffer,
                            offset: u64::from(view_uniforms.offsets[slot]),
                            size: Some(MeadowMeshViewUniform::min_size()),
                        },
                    )),
                )
            })
            .collect();
        bind_groups.by_variant.insert(*id, (per_slot, fingerprint));
    }
}

// ---------- Core3d: main-view pass ----------

/// Draw the meadow into the main view: one `draw_mesh_tasks` per variant,
/// writing color (PBR), depth, and — when the view has a motion-vector
/// prepass — motion vectors, all in a single pass between the opaque and
/// transparent passes.
#[allow(clippy::too_many_arguments)]
pub fn meadow_mesh_main_pass(
    view: ViewQuery<
        (
            &ExtractedView,
            &ExtractedCamera,
            &ViewTarget,
            &ViewDepthStencilTexture,
            Option<&MeshViewBindGroup>,
            Option<&MainPassResolutionOverride>,
        ),
        With<Camera3d>,
    >,
    active: Res<MeadowMeshPathActive>,
    pipelines: Res<MeadowMeshPipelines>,
    pipeline_cache: Res<PipelineCache>,
    bind_groups: Res<MeadowMeshBindGroups>,
    slots: Res<MeadowViewSlots>,
    buffers: Res<MeadowGpuBuffers>,
    mv_target: Res<MeadowMeshMvTarget>,
    mut ctx: RenderContext,
) {
    if !active.active || bind_groups.by_variant.is_empty() {
        return;
    }
    let (extracted_view, camera, target, depth, mesh_view_bind_group, resolution_override) =
        view.into_inner();
    // Only the view the meadow cull tracks as slot 0 (the main camera).
    let Some(&slot) = slots.by_retained.get(&extracted_view.retained_view_entity) else {
        return;
    };
    if slot != 0 {
        return;
    }
    let Some(mesh_view_bind_group) = mesh_view_bind_group else {
        return;
    };
    // The key was computed this frame in `prepare_meadow_mesh_pipelines`
    // from the same specialized descriptor the pipeline was built from —
    // reconstructing it here from live view state risks divergence.
    let Some(key) = pipelines.current_main_key else {
        return;
    };
    let Some(pipeline) =
        compiled_pipeline(&pipeline_cache, pipelines.main_pipelines.get(&key).copied())
    else {
        // Config changed this frame (e.g. DLSS/MSAA toggle); the pipeline
        // for the new key compiles and the path re-activates.
        return;
    };

    let mut color_attachments = vec![Some(target.get_color_attachment())];
    if key.mv {
        // The meadow-owned motion target (composited into bevy's MV
        // texture right after this pass). Cleared here — this is its
        // only producer.
        let Some(mv_view) = &mv_target.view else {
            return;
        };
        color_attachments.push(Some(RenderPassColorAttachment {
            view: mv_view,
            resolve_target: None,
            ops: Operations {
                // wgpu's default color is transparent black.
                load: LoadOp::Clear(default()),
                store: StoreOp::Store,
            },
            depth_slice: None,
        }));
    }
    let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("meadow_mesh_main_pass"),
        color_attachments: &color_attachments,
        depth_stencil_attachment,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, "meadow_mesh_main_pass");
    // Match the geometry passes' viewport — under DLSS the scene renders
    // at a reduced resolution into a sub-viewport of the full-size
    // targets (`MainPassResolutionOverride`); rasterizing full-texture
    // here would scale/misplace the grass relative to everything else.
    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        render_pass.set_camera_viewport(&viewport);
    }
    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(
        0,
        &mesh_view_bind_group.main,
        &mesh_view_bind_group.main_offsets,
    );
    render_pass.set_bind_group(1, &mesh_view_bind_group.binding_array, &[]);
    render_pass.set_bind_group(2, &mesh_view_bind_group.empty, &[]);
    draw_meadow_task_lists(&mut render_pass, 0, &bind_groups, &buffers);
    pass_span.end(&mut render_pass);
}

/// Per-variant meadow dispatch: bind the variant's meadow group for
/// `view_slot` at [`MEADOW_GROUP`] and issue the folded 2D
/// `draw_mesh_tasks` grid. The fold is a contract with the WGSL's
/// `flat = wg.y * MESH_TASK_DISPATCH_STRIDE + wg.x` reconstruction —
/// keep it in this one place.
fn draw_meadow_task_lists<'a>(
    pass: &mut TrackedRenderPass<'a>,
    view_slot: usize,
    bind_groups: &'a MeadowMeshBindGroups,
    buffers: &MeadowGpuBuffers,
) {
    for (id, (per_slot, _)) in bind_groups.by_variant.iter() {
        let Some(vb) = buffers.by_variant.get(id) else {
            continue;
        };
        if vb.num_task_slices == 0 {
            continue;
        }
        // A slot the groups weren't built for (view count grew this
        // frame) skips a frame; the prepare step rebuilds next frame.
        let Some(bg) = per_slot.get(view_slot) else {
            continue;
        };
        pass.set_bind_group(MEADOW_GROUP as usize, bg, &[]);
        let x = vb.num_task_slices.min(MESH_TASK_DISPATCH_STRIDE);
        let y = vb.num_task_slices.div_ceil(MESH_TASK_DISPATCH_STRIDE);
        pass.draw_mesh_tasks(x, y, 1);
    }
}

// ---------- Core3d: deferred-view G-buffer pass ----------

/// Draw the meadow into bevy's deferred prepass attachments on deferred
/// main views: one pass writing the packed G-buffer, the lighting-pass
/// id, motion vectors, and depth, ordered after the deferred mesh draws
/// and before the lighting-id copy (so the deferred lighting pass and
/// solari shade the grass like any other deferred surface).
///
/// The real attachments can be render targets here — unlike PTR's
/// terrain (whose fragment binds the full mesh-view group, which on
/// deferred views contains the current-frame deferred texture, forcing
/// private targets + a composite), this fragment's only view binding is
/// the `View` uniform, served by bevy's prepass view bind group; nothing
/// bound by the pass is attached to it.
///
/// The pass tail restages the scene depth (grass included) into the
/// prepass depth texture, mirroring the copy at the end of bevy's
/// deferred prepass — the deferred lighting pass and solari reconstruct
/// world positions from that texture, and bevy's own copy ran before the
/// grass drew.
#[allow(clippy::too_many_arguments)]
pub fn meadow_mesh_deferred_pass(
    view: ViewQuery<
        (
            &ExtractedView,
            &ExtractedCamera,
            &ViewDepthStencilTexture,
            &ViewPrepassTextures,
            &ViewUniformOffset,
            Option<&PreviousViewUniformOffset>,
            Option<&MainPassResolutionOverride>,
        ),
        (With<Camera3d>, With<DeferredPrepass>),
    >,
    active: Res<MeadowMeshPathActive>,
    pipelines: Res<MeadowMeshPipelines>,
    pipeline_cache: Res<PipelineCache>,
    prepass_view_bind_group: Res<PrepassViewBindGroup>,
    bind_groups: Res<MeadowMeshBindGroups>,
    slots: Res<MeadowViewSlots>,
    buffers: Res<MeadowGpuBuffers>,
    mut ctx: RenderContext,
) {
    if !active.active || bind_groups.by_variant.is_empty() {
        return;
    }
    let (
        extracted_view,
        camera,
        depth,
        prepass_textures,
        view_offset,
        prev_view_offset,
        resolution_override,
    ) = view.into_inner();
    // Only the view the meadow cull tracks as slot 0 (the main camera).
    if slots.by_retained.get(&extracted_view.retained_view_entity) != Some(&0) {
        return;
    }
    let Some(key) = pipelines.current_deferred_key else {
        return;
    };
    let Some(pipeline) = compiled_pipeline(
        &pipeline_cache,
        pipelines.deferred_pipelines.get(&key).copied(),
    ) else {
        return;
    };
    let Some(empty_bind_group) = &pipelines.empty_bind_group else {
        return;
    };

    // The view bind group variant must match the prepass view layout the
    // pipeline was built against (bindings 0/2 carry dynamic offsets).
    let offsets = [
        view_offset.offset,
        prev_view_offset.map_or(0, |prev| prev.offset),
    ];
    let (view_bind_group, view_offsets): (&BindGroup, &[u32]) = if key.mv {
        let (Some(bind_group), Some(_)) = (
            prepass_view_bind_group.motion_vectors.as_ref(),
            prev_view_offset,
        ) else {
            return;
        };
        (bind_group, &offsets)
    } else {
        let Some(bind_group) = prepass_view_bind_group.no_motion_vectors.as_ref() else {
            return;
        };
        (bind_group, &offsets[..1])
    };

    // Same attachment arrangement as bevy's deferred prepass ([normal?,
    // motion?, gbuffer, lighting-id], `None` holes preserved) — the
    // pipeline's target list was mirrored from the same def set. All
    // `get_attachment()` calls load here: bevy's prepass performed the
    // frame's clears earlier in the chain.
    let (Some(deferred_texture), Some(lighting_pass_id)) = (
        &prepass_textures.deferred,
        &prepass_textures.deferred_lighting_pass_id,
    ) else {
        return;
    };
    let normal_attachment = if key.normal {
        match &prepass_textures.normal {
            Some(texture) => Some(texture.get_attachment()),
            None => return,
        }
    } else {
        None
    };
    let motion_attachment = if key.mv {
        match &prepass_textures.motion_vectors {
            Some(texture) => Some(texture.get_attachment()),
            None => return,
        }
    } else {
        None
    };
    let color_attachments = [
        normal_attachment,
        motion_attachment,
        Some(deferred_texture.get_attachment()),
        Some(lighting_pass_id.get_attachment()),
    ];
    let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    {
        let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("meadow_mesh_deferred_pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let pass_span = diagnostics.pass_span(&mut render_pass, "meadow_mesh_deferred_pass");
        // Same DLSS sub-viewport handling as the forward main pass.
        if let Some(viewport) =
            Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
        {
            render_pass.set_camera_viewport(&viewport);
        }
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, view_bind_group, view_offsets);
        render_pass.set_bind_group(1, empty_bind_group, &[]);
        render_pass.set_bind_group(2, empty_bind_group, &[]);
        draw_meadow_task_lists(&mut render_pass, 0, &bind_groups, &buffers);
        pass_span.end(&mut render_pass);
    }

    // Restage the scene depth (now including grass) into the prepass
    // depth texture, exactly like the tail of bevy's deferred prepass.
    if let Some(prepass_depth_texture) = &prepass_textures.depth {
        ctx.command_encoder().copy_texture_to_texture(
            depth.texture().as_image_copy(),
            prepass_depth_texture.texture.texture.as_image_copy(),
            prepass_textures.size,
        );
    }
}

// ---------- Core3d: motion-vector composite ----------

/// Copy the grass motion vectors from the meadow-owned motion target
/// (written by the main pass as color target 1) into bevy's MV prepass
/// texture. A fullscreen triangle with per-texel discard on the valid
/// flag — sub-0.1ms, and it exists only because the MV prepass texture
/// rides inside the mesh-view bind group as a sampled resource, so it
/// can't be an attachment of the pass that binds that group.
#[allow(clippy::too_many_arguments)]
pub fn meadow_mesh_mv_composite_pass(
    view: ViewQuery<
        (
            &ExtractedView,
            &ExtractedCamera,
            Option<&ViewPrepassTextures>,
            Option<&MainPassResolutionOverride>,
        ),
        With<Camera3d>,
    >,
    active: Res<MeadowMeshPathActive>,
    pipelines: Res<MeadowMeshPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mv_target: Res<MeadowMeshMvTarget>,
    slots: Res<MeadowViewSlots>,
    mut ctx: RenderContext,
) {
    if !active.active {
        return;
    }
    let (Some(pipeline), Some(bind_group)) = (
        compiled_pipeline(&pipeline_cache, pipelines.composite_pipeline),
        &mv_target.composite_bind_group,
    ) else {
        return;
    };
    if !pipelines.current_main_key.is_some_and(|k| k.mv) {
        return;
    }
    let (extracted_view, camera, prepass, resolution_override) = view.into_inner();
    if slots.by_retained.get(&extracted_view.retained_view_entity) != Some(&0) {
        return;
    }
    let Some(mv) = prepass.and_then(|p| p.motion_vectors.as_ref()) else {
        return;
    };

    let color_attachments = [Some(mv.get_attachment())];
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("meadow_mesh_mv_composite"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, "meadow_mesh_mv_composite");
    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        render_pass.set_camera_viewport(&viewport);
    }
    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
    pass_span.end(&mut render_pass);
}

// ---------- Core3d: shadow-cascade pass ----------

/// Draw grass depth into each directional shadow cascade, after bevy's
/// shadow pass (the cascade's depth attachment re-opens with
/// `LoadOp::Load`). The task stage's per-cascade view-depth slice +
/// distance-ramped density mirror the compute kernel exactly.
#[allow(clippy::too_many_arguments)]
pub fn meadow_mesh_shadow_pass(
    view: ViewQuery<&ViewLightEntities>,
    light_views: Query<(&ShadowView, &ExtractedView, &LightEntity)>,
    active: Res<MeadowMeshPathActive>,
    pipelines: Res<MeadowMeshPipelines>,
    pipeline_cache: Res<PipelineCache>,
    bind_groups: Res<MeadowMeshBindGroups>,
    slots: Res<MeadowViewSlots>,
    buffers: Res<MeadowGpuBuffers>,
    mut ctx: RenderContext,
) {
    if !active.active || bind_groups.by_variant.is_empty() {
        return;
    }
    let (Some(pipeline), Some(empty_bind_group)) = (
        compiled_pipeline(&pipeline_cache, pipelines.shadow_pipeline),
        &pipelines.empty_bind_group,
    ) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    for &light_entity in view.into_inner().lights.iter() {
        let Ok((shadow_view, extracted_view, light)) = light_views.get(light_entity) else {
            continue;
        };
        if !matches!(light, LightEntity::Directional { .. }) {
            continue;
        }
        let Some(&slot) = slots.by_retained.get(&extracted_view.retained_view_entity) else {
            continue;
        };
        let depth_stencil_attachment =
            Some(shadow_view.depth_attachment.get_attachment(StoreOp::Store));
        let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("meadow_mesh_shadow_pass"),
            color_attachments: &[],
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let pass_span = diagnostics.pass_span(&mut render_pass, "meadow_mesh_shadow_pass");
        render_pass.set_render_pipeline(pipeline);
        // The shadow layout's slots 0-2 are empty (the geometry module
        // hardcodes the meadow group at MEADOW_GROUP).
        for slot_index in 0..MEADOW_GROUP as usize {
            render_pass.set_bind_group(slot_index, empty_bind_group, &[]);
        }
        draw_meadow_task_lists(&mut render_pass, slot as usize, &bind_groups, &buffers);
        pass_span.end(&mut render_pass);
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use bevy::asset::{AssetId, uuid::Uuid};
    use bevy::shader::{
        Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ShaderDefVal, ValidateShader,
    };

    /// The bevy checkout the tests read the real shader library from:
    /// `BEVY_CHECKOUT` if set, else the sibling `../bevy-main` or `../bevy`
    /// checkout. `None` (test skips) when none holds a bevy source tree.
    fn bevy_checkout() -> Option<PathBuf> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let candidates = match std::env::var_os("BEVY_CHECKOUT") {
            Some(root) => vec![PathBuf::from(root)],
            None => vec![manifest.join("../bevy-main"), manifest.join("../bevy")],
        };
        candidates.into_iter().find(|root| {
            root.join("crates/bevy_pbr/src/render/pbr_functions.wesl")
                .exists()
        })
    }

    /// Bevy's own shader cache, with naga standing in for the GPU device:
    /// the composed WGSL is parsed and validated (mesh-shader
    /// capabilities included) and kept as the "module" for assertions.
    struct Library {
        cache: ShaderCache<String, ()>,
    }

    fn validate_wgsl(
        _device: &(),
        source: ShaderCacheSource,
        _validate: &ValidateShader,
    ) -> Result<String, ShaderCacheError> {
        let ShaderCacheSource::Wgsl(wgsl) = source else {
            panic!("wesl compiles to WGSL");
        };
        let module = naga::front::wgsl::parse_str(&wgsl)
            .map_err(|e| ShaderCacheError::ProcessShaderError(e.emit_to_string(&wgsl)))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| ShaderCacheError::ProcessShaderError(format!("{e:?}")))?;
        Ok(wgsl)
    }

    impl Library {
        /// The meadow shaders alone — enough for the geometry module.
        fn meadow() -> Self {
            let mut lib = Self {
                cache: ShaderCache::new((), validate_wgsl),
            };
            for (source, path) in [
                (
                    include_str!("meadow_shared.wesl"),
                    "embedded://bevy_meadow/meadow_shared.wesl",
                ),
                (
                    include_str!("meadow_mesh_bindings.wesl"),
                    "embedded://bevy_meadow/meadow_mesh_bindings.wesl",
                ),
            ] {
                lib.add(Shader::from_wesl(source, path));
            }
            lib
        }

        /// The meadow shaders (PBR-common module included) plus bevy's
        /// shader crates from the checkout, with the loader-settings defs
        /// bevy registers (`mesh_view_types.wesl` in
        /// `MeshRenderPlugin::build`) carried on the Shader asset for the
        /// def closure. `None` (tests skip) without a bevy checkout.
        fn with_bevy() -> Option<Self> {
            let root = bevy_checkout()?;
            let mut lib = Self::meadow();
            lib.add(Shader::from_wesl(
                include_str!("meadow_mesh_pbr_common.wesl"),
                "embedded://bevy_meadow/meadow_mesh_pbr_common.wesl",
            ));
            for crate_name in ["bevy_pbr", "bevy_render", "bevy_core_pipeline"] {
                let src = root.join("crates").join(crate_name).join("src");
                lib.add_wesl_dir(crate_name, &src, &src);
            }
            Some(lib)
        }

        fn add(&mut self, shader: Shader) -> AssetId<Shader> {
            let id = AssetId::Uuid {
                uuid: Uuid::new_v4(),
            };
            self.cache.set_shader(id, shader);
            id
        }

        /// Register every `.wesl` under `dir` the way `load_shader_library!`
        /// does at runtime: `embedded://<crate>/<path-under-src>` paths, so
        /// module names and import scanning match the real asset registry.
        fn add_wesl_dir(&mut self, crate_name: &str, src_root: &Path, dir: &Path) {
            for entry in std::fs::read_dir(dir).expect("readable bevy source dir") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    self.add_wesl_dir(crate_name, src_root, &path);
                } else if path.extension().is_some_and(|e| e == "wesl") {
                    let rel = path
                        .strip_prefix(src_root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    let source = std::fs::read_to_string(&path).expect("readable wesl file");
                    let mut shader =
                        Shader::from_wesl(source, format!("embedded://{crate_name}/{rel}"));
                    if rel == "render/mesh_view_types.wesl" {
                        shader.shader_defs = vec![
                            ShaderDefVal::UInt("MAX_DIRECTIONAL_LIGHTS".into(), 10),
                            ShaderDefVal::UInt("MAX_CASCADES_PER_LIGHT".into(), 4),
                            ShaderDefVal::UInt("MAX_RECT_LIGHTS".into(), 8),
                        ];
                    }
                    self.add(shader);
                }
            }
        }

        /// Compile `shader` with `defs` (plus the device-global defs bevy's
        /// `PipelineCache` splices onto every shader), panicking on any
        /// error, and return the validated WGSL.
        fn compile(&mut self, shader: AssetId<Shader>, defs: &[ShaderDefVal]) -> Arc<String> {
            let mut defs = defs.to_vec();
            defs.extend([
                ShaderDefVal::UInt("AVAILABLE_STORAGE_BUFFER_BINDINGS".into(), 12),
                ShaderDefVal::Bool("AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3".into(), true),
                ShaderDefVal::Bool("AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_6".into(), true),
            ]);
            match self.cache.get(0, shader, &defs) {
                Ok(wgsl) => wgsl,
                Err(ShaderCacheError::ShaderImportNotYetAvailable) => {
                    panic!("an import is missing from a full checkout (see the warning above)")
                }
                Err(err) => panic!("{err}"),
            }
        }
    }

    /// Run bevy's shader cache offline against the REAL bevy shader
    /// library (skipped when no bevy checkout is present) with a
    /// representative forward def set: the forward fragment must compile,
    /// both entry points must survive under their unmangled names, and
    /// the output must be WGSL that naga parses + validates.
    #[test]
    fn pbr_fragment_compiles_against_bevy_library() {
        let Some(mut lib) = Library::with_bevy() else {
            eprintln!("skipping: no bevy checkout at ../bevy-main (set BEVY_CHECKOUT)");
            return;
        };
        let fragment = lib.add(Shader::from_wesl(
            include_str!("meadow_mesh_pbr_fragment.wesl"),
            "embedded://bevy_meadow/meadow_mesh_pbr_fragment.wesl",
        ));

        // Representative forward-view def set (a subset of what the
        // meadow material's specialized pipeline hands over at runtime).
        let defs: Vec<ShaderDefVal> = vec![
            "VERTEX_POSITIONS".into(),
            "VERTEX_NORMALS".into(),
            "TONEMAP_IN_SHADER".into(),
            "TONEMAP_METHOD_TONY_MC_MAPFACE".into(),
            ShaderDefVal::UInt("TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(), 18),
            ShaderDefVal::UInt("TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(), 19),
        ];
        let wgsl = lib.compile(fragment, &defs);

        // The pipeline descriptors reference these entry names directly.
        assert!(wgsl.contains("fn fragment("), "entry `fragment` lost");
        assert!(wgsl.contains("fn fragment_mv("), "entry `fragment_mv` lost");
    }

    /// Deferred counterpart: compiles the G-buffer fragment with the def
    /// set `PrepassPipeline::specialize` produces for a deferred, depth,
    /// and motion-vector view (a solari session's main view) — the exact
    /// shape `prepare_meadow_mesh_pipelines` lifts from the meadow
    /// material's specialized deferred prepass pipeline at runtime.
    #[test]
    fn deferred_fragment_compiles_against_bevy_library() {
        let Some(mut lib) = Library::with_bevy() else {
            eprintln!("skipping: no bevy checkout at ../bevy-main (set BEVY_CHECKOUT)");
            return;
        };
        let fragment = lib.add(Shader::from_wesl(
            include_str!("meadow_mesh_deferred_fragment.wesl"),
            "embedded://bevy_meadow/meadow_mesh_deferred_fragment.wesl",
        ));

        let defs: Vec<ShaderDefVal> = vec![
            "PREPASS_PIPELINE".into(),
            ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 3),
            "VERTEX_OUTPUT_INSTANCE_INDEX".into(),
            "DEPTH_PREPASS".into(),
            "VERTEX_POSITIONS".into(),
            "VERTEX_UVS".into(),
            "VERTEX_UVS_A".into(),
            "NORMAL_PREPASS_OR_DEFERRED_PREPASS".into(),
            "VERTEX_NORMALS".into(),
            "MOTION_VECTOR_PREPASS_OR_DEFERRED_PREPASS".into(),
            "DEFERRED_PREPASS".into(),
            "MOTION_VECTOR_PREPASS".into(),
            "PREPASS_FRAGMENT".into(),
        ];
        let wgsl = lib.compile(fragment, &defs);

        assert!(
            wgsl.contains("fn fragment_deferred("),
            "entry `fragment_deferred` lost"
        );
        // The G-buffer write must survive to the entry point's output —
        // guards against the packing path getting condcomp'd away.
        assert!(
            wgsl.contains("deferred_lighting_pass_id"),
            "lighting-pass id output missing"
        );
    }

    #[test]
    fn compute_raster_fragments_compile_against_bevy_library() {
        let Some(mut lib) = Library::with_bevy() else {
            eprintln!("skipping: set BEVY_CHECKOUT to validate raster shaders");
            return;
        };
        let shader = lib.add(Shader::from_wesl(
            include_str!("meadow.wesl"),
            "embedded://bevy_meadow/meadow.wesl",
        ));
        let base: Vec<ShaderDefVal> = vec![
            ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 3),
            "VERTEX_OUTPUT_INSTANCE_INDEX".into(),
            "VERTEX_POSITIONS".into(),
            "VERTEX_NORMALS".into(),
            "VERTEX_UVS".into(),
            "VERTEX_UVS_A".into(),
            "VERTEX_TANGENTS".into(),
        ];
        lib.compile(shader, &base);
        let mut deferred = base;
        deferred.extend([
            "PREPASS_PIPELINE".into(),
            "DEPTH_PREPASS".into(),
            "NORMAL_PREPASS_OR_DEFERRED_PREPASS".into(),
            "MOTION_VECTOR_PREPASS_OR_DEFERRED_PREPASS".into(),
            "DEFERRED_PREPASS".into(),
            "MOTION_VECTOR_PREPASS".into(),
            "PREPASS_FRAGMENT".into(),
        ]);
        lib.compile(shader, &deferred);
        // WebGL packs depth into the material-properties word: the desktop
        // Solari extension must be absent even if its runtime marker is set.
        deferred.push("WEBGL2".into());
        let webgl = lib.compile(shader, &deferred);
        assert!(!webgl.contains("meadow_pack_geometric_normal"));
    }

    #[test]
    fn triangle_normal_reconstruction_respects_jitter_viewport_and_large_world_origin() {
        use bevy::math::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
        // Three samples on one small triangle; finite differences of the
        // reconstructed plane must preserve its normal, including reversed-Z,
        // jitter, a viewport origin, and reduced render resolution.
        let points = [
            Vec3::new(-0.01, 0.0, -3.0),
            Vec3::new(0.0, 0.013, -3.004),
            Vec3::new(0.012, 0.006, -3.02),
        ];
        let normal = (points[1] - points[0])
            .cross(points[2] - points[0])
            .normalize();
        let viewport = Vec4::new(17.0, 23.0, 960.0, 540.0);
        for mut projection in [
            Mat4::perspective_infinite_reverse_rh(1.0, 16.0 / 9.0, 0.1),
            Mat4::orthographic_rh(-2.0, 2.0, -1.0, 1.0, 100.0, 0.1),
        ] {
            projection.z_axis.x += 0.37 / viewport.z;
            projection.z_axis.y -= 0.21 / viewport.w;
            let inverse = projection.inverse();
            let reconstructed = points.map(|p| {
                let ndc = projection.project_point3(p);
                let uv = Vec2::new(ndc.x, ndc.y) * Vec2::new(0.5, -0.5) + Vec2::splat(0.5);
                let pixel =
                    uv * Vec2::new(viewport.z, viewport.w) + Vec2::new(viewport.x, viewport.y);
                let recovered_uv =
                    (pixel - Vec2::new(viewport.x, viewport.y)) / Vec2::new(viewport.z, viewport.w);
                let recovered_ndc = Vec3::new(
                    recovered_uv.x * 2.0 - 1.0,
                    1.0 - recovered_uv.y * 2.0,
                    ndc.z,
                );
                inverse.project_point3(recovered_ndc)
            });
            let reconstructed_normal = (reconstructed[1] - reconstructed[0])
                .cross(reconstructed[2] - reconstructed[0])
                .normalize();
            for origin in [1442.0, 1_000_000.0] {
                let world_from_view = Mat4::from_rotation_translation(
                    Quat::from_rotation_y(0.7),
                    Vec3::new(origin, 100.0, -origin),
                );
                let normal_to_world = Mat3::from_mat4(world_from_view.inverse()).transpose();
                let actual = (normal_to_world * reconstructed_normal).normalize();
                let expected = (normal_to_world * normal).normalize();
                assert!(actual.dot(expected) > 0.99999, "{actual:?} vs {expected:?}");
            }
        }
    }

    #[test]
    fn geometric_normal_payload_preserves_material_and_has_safe_fallback() {
        use bevy::math::{Vec2, Vec3};
        const BIT: u32 = 1 << 28;
        // CPU reference of `meadow_pack_geometric_normal` and
        // `meadow_unpack_geometric_normal_oct`: the payload leaves
        // emissive/base color, reflectance/metallic and every other A bit
        // intact.
        fn pack(mut pixel: [u32; 4], normal: Vec3, enabled: bool) -> [u32; 4] {
            let magnitude = normal.abs();
            let scale = magnitude.max_element();
            if !enabled || !(scale > 1e-20) || !magnitude.cmplt(Vec3::splat(1e30)).all() {
                return pixel;
            }
            let normal = normal / scale;
            let n = normal / normal.abs().element_sum();
            let sign_x = if n.x >= 0.0 { 1.0 } else { -1.0 };
            let sign_y = if n.y >= 0.0 { 1.0 } else { -1.0 };
            let oct = if n.z >= 0.0 {
                Vec2::new(n.x, n.y)
            } else {
                Vec2::new((1.0 - n.y.abs()) * sign_x, (1.0 - n.x.abs()) * sign_y)
            } * 0.5
                + Vec2::splat(0.5);
            let encoded = (oct.clamp(Vec2::ZERO, Vec2::ONE) * 255.0).round();
            pixel[2] =
                (pixel[2] & 0xFFFF) | ((encoded.x as u32) << 16) | ((encoded.y as u32) << 24);
            pixel[3] |= BIT;
            pixel
        }
        fn decode(pixel: [u32; 4]) -> Vec3 {
            let x = ((pixel[2] >> 16) & 255) as f32 / 255.0 * 2.0 - 1.0;
            let y = (pixel[2] >> 24) as f32 / 255.0 * 2.0 - 1.0;
            let mut n = Vec3::new(x, y, 1.0 - x.abs() - y.abs());
            let t = (-n.z).clamp(0.0, 1.0);
            n.x += if n.x >= 0.0 { -t } else { t };
            n.y += if n.y >= 0.0 { -t } else { t };
            n.normalize()
        }
        let original = [0x12345678, 0x89ABCDEF, 0xFEDCBA98, 0x0F123456];
        for normal in [
            Vec3::X,
            -Vec3::X,
            Vec3::Y,
            -Vec3::Y,
            Vec3::Z,
            -Vec3::Z,
            Vec3::new(0.3, -0.7, -0.2),
            Vec3::new(-0.2, 0.5, 0.8),
        ] {
            assert_eq!(pack(original, normal, false), original);
            let packed = pack(original, normal, true);
            assert_eq!(packed[0..2], original[0..2]);
            assert_eq!(packed[2] & 0xFFFF, original[2] & 0xFFFF);
            assert_eq!(packed[3] & !BIT, original[3]);
            assert_ne!(packed[3] & BIT, 0);
            assert!(decode(packed).dot(normal.normalize()) > 0.9998);
        }
        for invalid in [
            Vec3::ZERO,
            Vec3::splat(1e-30),
            Vec3::splat(f32::NAN),
            Vec3::new(f32::INFINITY, 1.0, 0.0),
            Vec3::splat(1e31),
        ] {
            assert_eq!(pack(original, invalid, true), original);
        }
    }

    /// `PACKED_NORMAL_TANGENT_GUARD` in `meadow_receivers.wesl`.
    const PACKED_NORMAL_TANGENT_GUARD: f32 = 0.02;

    /// A unit normal after the payload's two-unorm8 octahedral round trip.
    fn oct8x2(n: bevy::math::Vec3) -> bevy::math::Vec3 {
        use bevy::math::{Vec2, Vec3};
        let v = n / n.abs().element_sum();
        let sign = |x: f32| if x >= 0.0 { 1.0 } else { -1.0 };
        let e = if v.z >= 0.0 {
            Vec2::new(v.x, v.y)
        } else {
            Vec2::new((1.0 - v.y.abs()) * sign(v.x), (1.0 - v.x.abs()) * sign(v.y))
        };
        let e = ((e * 0.5 + Vec2::splat(0.5)) * 255.0).round() / 255.0;
        let f = e * 2.0 - Vec2::ONE;
        let mut n = Vec3::new(f.x, f.y, 1.0 - f.x.abs() - f.y.abs());
        let t = (-n.z).clamp(0.0, 1.0);
        n.x -= sign(n.x) * t;
        n.y -= sign(n.y) * t;
        n.normalize()
    }

    /// CPU model of how Solari consumes a packed normal: the ray origin moves
    /// along ±n toward the ray's side, and only when |n·d| exceeds the guard
    /// the receiver pass hands over with the normal. The negated comparison
    /// mirrors the shader: a NaN cosine takes the no-offset branch.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn packed_normal_offset(
        p: bevy::math::Vec3,
        n: bevy::math::Vec3,
        d: bevy::math::Vec3,
    ) -> bevy::math::Vec3 {
        let cosine = n.dot(d);
        if !(cosine.abs() > PACKED_NORMAL_TANGENT_GUARD) {
            return p;
        }
        let n = if cosine > 0.0 { n } else { -n };
        let distance = (4.0 * f32::EPSILON * n.abs().dot(p.abs())).max(0.00001);
        p + n * distance
    }

    #[test]
    fn oct8x2_quantization_sign_reversal_does_not_introduce_a_grazing_self_hit() {
        use bevy::math::Vec3;
        let n = Vec3::new(0.52699474, 0.847_684_3, 0.06089206).normalize();
        let d = Vec3::new(-0.833_263_8, 0.49988333, 0.23619491).normalize();
        let q = oct8x2(n);
        // The quantized normal lands on the other side of this grazing ray.
        assert!(n.dot(d) < 0.0 && q.dot(d) > 0.0);
        let p = Vec3::ZERO;
        // Intersection distance along `d` with the true blade plane through `p`.
        let plane_hit = |origin: Vec3| n.dot(p - origin) / n.dot(d);
        // Unguarded, even the 10 micron minimum offset invents a ~1 cm self hit.
        assert!(plane_hit(p + q * 0.00001) > 0.001);
        assert_eq!(packed_normal_offset(p, q, d), p);
        assert_eq!(packed_normal_offset(p, q, -d), p);
    }

    #[test]
    fn oct8x2_round_trip_normals_never_choose_wrong_side_outside_guard() {
        use bevy::math::Vec3;
        for x in -16..=16 {
            for y in -16..=16 {
                for z in [-16, -3, 3, 16] {
                    let n = Vec3::new(x as f32, y as f32, z as f32).normalize();
                    let q = oct8x2(n);
                    let axis = if n.x.abs() < 0.8 { Vec3::X } else { Vec3::Y };
                    let tangent = n.cross(axis).normalize();
                    for cosine in [-0.1, -0.03, -0.001, 0.001, 0.03, 0.1] {
                        let d = (tangent + n * cosine).normalize();
                        if q.dot(d).abs() > PACKED_NORMAL_TANGENT_GUARD {
                            assert!(q.dot(d) * n.dot(d) > 0.0);
                        } else {
                            assert_eq!(
                                packed_normal_offset(Vec3::splat(1442.0), q, d),
                                Vec3::splat(1442.0)
                            );
                        }
                    }
                }
            }
        }
    }

    /// The Solari receiver pass compiles against the real bevy and
    /// bevy_solari shader libraries, and binds nothing in group 1 beyond its
    /// own five resources: importing Solari's helpers must not drag Solari's
    /// lighting bindings into meadow's pipeline layout.
    #[test]
    fn receiver_shader_compiles_against_bevy_and_solari_libraries() {
        let (Some(mut lib), Some(root)) = (Library::with_bevy(), bevy_checkout()) else {
            eprintln!("skipping: set BEVY_CHECKOUT to validate the receiver shader");
            return;
        };
        let solari = root.join("crates/bevy_solari/src");
        if !solari.join("realtime/receiver_override.wesl").exists()
            || !solari.join("scene/triangle_anchor.wesl").exists()
        {
            eprintln!("skipping: the bevy checkout has no Solari receiver-override interface");
            return;
        }
        lib.add_wesl_dir("bevy_solari", &solari, &solari);
        let shader = lib.add(Shader::from_wesl(
            include_str!("meadow_receivers.wesl"),
            "embedded://bevy_meadow/meadow_receivers.wesl",
        ));
        let wgsl = lib.compile(shader, &[]);
        assert!(
            wgsl.contains("fn resolve_meadow_receivers("),
            "entry `resolve_meadow_receivers` lost"
        );
        assert_eq!(
            wgsl.matches("@group(1)").count(),
            5,
            "group 1 must hold exactly meadow's five receiver bindings:\n{wgsl}"
        );

        // bevy_pbr's deferred flags share the G-buffer A byte with meadow's
        // bits 27 and 28 (flag bits 3 and 4).
        let types = std::fs::read_to_string(root.join("crates/bevy_pbr/src/deferred/types.wesl"))
            .expect("readable deferred types");
        for line in types.lines().filter(|l| l.starts_with("const DEFERRED_")) {
            let Some(shift) = line.split("<<").nth(1).and_then(|s| {
                s.trim()
                    .trim_end_matches(';')
                    .trim_end_matches('u')
                    .parse::<u32>()
                    .ok()
            }) else {
                continue;
            };
            assert!(
                shift != 3 && shift != 4,
                "bevy_pbr deferred flag collides with a meadow G-buffer bit: {line}"
            );
        }
    }

    #[test]
    fn grass_shading_normal_stays_in_view_hemisphere() {
        use bevy::math::Vec3;
        // CPU reference for meadow_shared.wesl::meadow_shading_normal.
        // Exercise hillside angles on both sides of camera elevation,
        // grazing views, and degenerate camera/blade coincidence.
        fn normal(direction: Vec3) -> Vec3 {
            if direction.length_squared() < 1e-12 {
                return Vec3::Y;
            }
            let v = direction.normalize();
            if v.y >= 0.1 {
                return Vec3::Y;
            }
            let tangent = Vec3::Y - v * v.y;
            if tangent.length_squared() < 1e-12 {
                return v;
            }
            tangent.normalize() * (1.0_f32 - 0.1 * 0.1).sqrt() + v * 0.1
        }
        assert_eq!(normal(Vec3::ZERO), Vec3::Y);
        for elevation in -100..=100 {
            for azimuth in 0..16 {
                let angle = azimuth as f32 * std::f32::consts::TAU / 16.0;
                let v = Vec3::new(angle.cos(), elevation as f32 / 10.0, angle.sin()).normalize();
                let n = normal(v);
                assert!(n.is_finite());
                assert!((n.length() - 1.0).abs() < 1e-5);
                assert!(n.dot(v) >= 0.0999, "view {v:?}, normal {n:?}");
                if v.y >= 0.1 {
                    assert_eq!(n, Vec3::Y);
                }
            }
        }
        assert_eq!(normal(Vec3::Y), Vec3::Y);
        assert_eq!(normal(Vec3::NEG_Y), Vec3::NEG_Y);
    }

    /// Compile the task/mesh module through bevy's shader cache.
    fn compile_geom_module() -> Arc<String> {
        let mut lib = Library::meadow();
        let geom = lib.add(Shader::from_wesl(
            include_str!("meadow_mesh.wesl"),
            "embedded://bevy_meadow/meadow_mesh.wesl",
        ));
        lib.compile(geom, &[])
    }

    /// The task/mesh module must compile and validate with the same naga
    /// wgpu uses at runtime — the WGSL mesh shader frontend is new, so pin
    /// it in `cargo test` rather than discovering breakage at pipeline
    /// creation on the GPU box. Every entry point the pipeline descriptors
    /// name must survive unmangled.
    #[test]
    fn geom_module_compiles_and_validates() {
        let wgsl = compile_geom_module();
        for entry in ["meadow_task", "meadow_mesh", "meadow_mesh_shadow"] {
            assert!(
                wgsl.contains(&format!("fn {entry}(")),
                "entry `{entry}` lost"
            );
        }
    }

    /// Validate RT expansion and padding together: their storage layouts
    /// must track the CPU allocation when the near ribbon topology changes.
    #[test]
    fn rt_compute_module_compiles_and_validates() {
        let mut lib = Library::meadow();
        let shader = lib.add(Shader::from_wesl(
            include_str!("meadow_compute.wesl"),
            "embedded://bevy_meadow/meadow_compute.wesl",
        ));
        let wgsl = lib.compile(shader, &[]);
        for entry in ["expand_rt_blades", "rt_pad_unused"] {
            assert!(
                wgsl.contains(&format!("fn {entry}(")),
                "entry `{entry}` lost"
            );
        }
        assert_eq!(
            crate::compute::RT_NEAR_MAX_BLADES * crate::mesh::BLADE_INDICES_PER_BLADE / 3,
            294_912,
            "near fidelity must not increase the triangle budget",
        );
    }

    /// Run the validated module through naga's SPIR-V backend with
    /// mesh-shading enabled (SPIR-V 1.6, like wgpu-hal picks on Vulkan
    /// 1.3+) — pins the mesh-shader codegen path end-to-end minus the
    /// driver, so backend regressions surface in `cargo test`.
    #[test]
    fn geom_module_compiles_to_spirv() {
        use naga::back::spv;
        let source = compile_geom_module();
        let module = naga::front::wgsl::parse_str(&source).expect("parses");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
        let options = spv::Options {
            lang_version: (1, 6),
            ..Default::default()
        };
        let words = spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|e| panic!("meadow_mesh.wesl failed SPIR-V codegen: {e:?}"));
        assert!(!words.is_empty());
    }

    /// The derivation helpers are duplicated between the compute kernel
    /// and the mesh-shader module, and the wind helper between the raster
    /// material and the mesh path (see the MIRROR comments in each). The
    /// two paths must derive bit-identical blades — pin the shared
    /// function bodies to each other so drift fails the build.
    #[test]
    fn mirrored_helpers_match_compute_kernel() {
        let compute = include_str!("meadow_compute.wesl");
        let mesh = include_str!("meadow_mesh.wesl");
        let raster = include_str!("meadow.wesl");

        /// Function body with comments stripped and whitespace collapsed —
        /// the mirrors must match structurally; comments may differ.
        fn body_of(src: &str, name: &str) -> String {
            let start = src
                .find(&format!("fn {name}("))
                .unwrap_or_else(|| panic!("`{name}` missing"));
            let open = start + src[start..].find('{').unwrap();
            let mut depth = 0usize;
            let mut end = None;
            for (i, c) in src[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(open + i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let end = end.unwrap_or_else(|| panic!("`{name}` unterminated"));
            src[open..=end]
                .lines()
                .map(|l| l.split("//").next().unwrap_or("").trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        }

        for f in [
            "hash_u32",
            "hash01_u32",
            "hash01_pair",
            "derive_blade",
            "blade_visibility",
            "sample_heightfield",
            "patch_sphere_culled",
        ] {
            assert_eq!(
                body_of(compute, f),
                body_of(mesh, f),
                "`{f}` drifted between meadow_compute.wesl and meadow_mesh.wesl"
            );
        }
        assert_eq!(
            body_of(raster, "wind_displacement"),
            body_of(mesh, "wind_displacement"),
            "`wind_displacement` drifted between meadow.wesl and meadow_mesh.wesl"
        );
    }
}
