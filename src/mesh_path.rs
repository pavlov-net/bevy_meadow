//! Mesh-shader render path (`mesh-shaders` cargo feature) — the faster
//! path (about 2x) where the GPU supports it.
//!
//! A full port of the meadow renderer to task/mesh pipelines: the
//! 128-wide task stage does the cull / gate / derive / compact work the
//! compute kernel does today and hands fully-derived survivors to
//! small pure-expander mesh workgroups through the task payload — no
//! intermediate `out_blades` buffer, no cursors, no indirect args
//! (see `meadow_mesh.wgsl` for the stage-split rationale). It serves
//! BOTH the main camera view (full PBR fragment + motion vectors) and
//! the directional shadow cascades (depth-only proxy-silhouette
//! pipelines drawn into each cascade after bevy's shadow pass).
//!
//! The compute path stays intact behind [`MeadowForceComputePath`] (force
//! it at runtime to compare) and is the automatic fallback when
//! `EXPERIMENTAL_MESH_SHADER` is absent (pre-Turing/pre-RDNA2, non-Vulkan)
//! or the view renders deferred: [`MeadowMeshPathActive`] flips
//! per frame, `DrawMeadowPatch` skips the views the mesh path serves, and
//! `prepare_meadow_gpu_buffers` collapses the per-view blade regions to
//! zero so the VRAM saving is real.
//!
//! Because Bevy's material/pipeline machinery cannot express mesh
//! pipelines, everything here is raw wgpu — but it deliberately *reuses*
//! Bevy's frame data rather than rebuilding it:
//!
//! - The **PBR fragment** is composed by driving the wesl compiler (the
//!   same one bevy_shader uses) over the loaded `Shader` assets, with
//!   the shader defs lifted verbatim
//!   from the meadow material's own specialized forward pipeline
//!   ([`SpecializedMaterialPipelineCache`]), so the composed module's
//!   view-binding declarations match the real view bind group layout by
//!   construction.
//! - Bind groups 0/1 of the main pipeline are Bevy's own
//!   [`MeshViewBindGroup`] (view uniforms, lights, shadow maps, clusters,
//!   probes), giving full lighting/shadow-receive parity with the compute
//!   path's `ExtendedMaterial` fragment.
//! - The per-view cull data is the same `view_cull` storage buffer the
//!   compute path uploads.

use std::borrow::Cow;
use std::collections::HashMap as StdHashMap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use bevy::camera::{Camera3d, MainPassResolutionOverride, Viewport};
use bevy::core_pipeline::core_3d::{
    CORE_3D_DEPTH_FORMAT, main_opaque_pass_3d, main_transparent_pass_3d,
};
use bevy::core_pipeline::prepass::{
    DeferredPrepass, MOTION_VECTOR_PREPASS_FORMAT, MotionVectorPrepass, PreviousViewData,
    ViewPrepassTextures,
};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::resource::Resource;
use bevy::pbr::{
    LATE_SHADOW_PASS, LightEntity, MeshViewBindGroup, ShadowView, SpecializedMaterialPipelineCache,
    ViewLightEntities, per_view_shadow_pass,
};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::camera::{ExtractedCamera, TemporalJitter};
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only, storage_buffer_read_only_sized, texture_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferId,
    DynamicUniformBuffer, PipelineCache, RenderPassDescriptor, ShaderStages, ShaderType, StoreOp,
    TextureSampleType, TextureViewId,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::sync_world::MainEntity;
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, Msaa, ViewDepthStencilTexture, ViewTarget};
use bevy::render::{Extract, ExtractSchedule, Render, RenderStartup, RenderSystems};
use bevy::shader::{Shader, ShaderDefVal, ShaderImport};

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

// ---------- per-view uniform ----------

/// Rust mirror of `MeadowMeshView` in `meadow_mesh.wgsl`. One entry per
/// meadow view slot in a dynamic-offset uniform buffer.
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

/// Per-frame per-view-slot uniforms + their dynamic offsets, slot-indexed
/// to match [`MeadowViewSlots`].
#[derive(Resource, Default)]
pub struct MeadowMeshViewUniforms {
    pub buffer: DynamicUniformBuffer<MeadowMeshViewUniform>,
    pub offsets: [u32; MEADOW_MAX_VIEWS],
}

// ---------- pipelines resource ----------

/// Key for the main-view mesh pipeline. `color_format`/`samples` come
/// from the meadow material's specialized forward pipeline descriptor
/// (so they match the pass bevy renders opaque with); `pbr` selects the
/// composed bevy_pbr fragment over the flat-lit fallback.
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct MeadowMainPipelineKey {
    pub color_format: wgpu::TextureFormat,
    pub samples: u32,
    pub pbr: bool,
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

/// Everything raw-wgpu the mesh path owns: shader modules, the meadow
/// bind group layout, and the lazily-created pipelines.
#[derive(Resource, Default)]
pub struct MeadowMeshPipelines {
    /// Adapter supports `EXPERIMENTAL_MESH_SHADER` + our output budgets.
    pub supported: bool,
    /// Task + mesh + flat-fragment module (meadow group at
    /// [`MEADOW_GROUP`]) — one module serves every meadow mesh pipeline.
    pub geom_module: Option<wgpu::ShaderModule>,
    /// The meadow bind group layout (descriptor kept for cache identity).
    pub meadow_bgl_desc: Option<BindGroupLayoutDescriptor>,
    /// Empty bind group for the shadow pipeline's unused slots 0-2 (and
    /// layout-compatible with the main pipeline's empty slot 2).
    pub empty_bind_group: Option<BindGroup>,
    /// wesl-composed PBR fragment (entries `fragment`/`fragment_mv`).
    pub pbr_fragment: Option<wgpu::ShaderModule>,
    /// Hash of the shader-def set the PBR fragment was composed with;
    /// recompose when the specialized pipeline's defs change.
    pub pbr_defs_hash: Option<u64>,
    /// (def-set hash, source-snapshot generation) a composition attempt
    /// permanently failed for (fall back to the flat fragment). A def
    /// change — e.g. DLSS toggling the prepasses — or a refreshed shader
    /// collection (hot reload) retries.
    pub pbr_failed: Option<(u64, u64)>,
    /// Module name already logged as missing (waiting-to-stream); keeps
    /// the retry loop visible in the log without spamming it.
    pub pbr_missing_logged: Option<String>,
    pub main_pipelines: HashMap<MeadowMainPipelineKey, wgpu::RenderPipeline>,
    /// Key the current frame's main view resolves to (None until known).
    /// `decide_meadow_mesh_path` requires the matching pipeline to exist.
    pub current_main_key: Option<MeadowMainPipelineKey>,
    pub shadow_pipeline: Option<wgpu::RenderPipeline>,
    /// Fullscreen composite copying the meadow-owned motion target's
    /// valid texels into bevy's MV prepass texture (which can't be an
    /// attachment of the main pass — it rides inside the mesh-view bind
    /// group as a sampled resource). Built once at startup.
    pub composite_pipeline: Option<wgpu::RenderPipeline>,
    pub composite_bgl: Option<wgpu::BindGroupLayout>,
}

/// Meadow-owned motion target: color target 1 of the main pass
/// (Rgba16Float; xy = motion vector, z = valid flag), sized to the main
/// view's physical target and composited into bevy's Rg16Float MV
/// prepass texture right after. Exists only in single-sample MV configs
/// (DLSS/TAA).
#[derive(Resource, Default)]
pub struct MeadowMeshMvTarget {
    pub view: Option<wgpu::TextureView>,
    pub size: UVec2,
    /// Bind group for the composite pass (references `view`; the view
    /// keeps the underlying texture alive).
    pub composite_bind_group: Option<wgpu::BindGroup>,
}

/// Identity fingerprint of a variant's mesh-path bind group (same
/// rationale as the compute path's `ComputeBindGroupFingerprint`).
type MeshBindGroupFingerprint = ([BufferId; 6], TextureViewId);

#[derive(Resource, Default)]
pub struct MeadowMeshBindGroups {
    pub by_variant: HashMap<MeadowVariantId, (BindGroup, MeshBindGroupFingerprint)>,
}

/// Render-world snapshot of the composable `Shader` assets the PBR
/// fragment needs (the `bevy_pbr::`/`bevy_render::`/
/// `bevy_core_pipeline::` libraries + `bevy_meadow::meadow_shared`),
/// keyed by wesl module path. Whole `Shader` clones — the composer
/// needs each dep's own `shader_defs` for the def closure. Re-cloned on
/// shader-asset changes until composition settles, then frozen.
#[derive(Resource, Default)]
pub struct MeadowMeshShaderSources {
    pub by_module: StdHashMap<wesl::syntax::ModulePath, Shader>,
    /// Bumped on every re-collection; a permanent composition failure is
    /// keyed to it so a refreshed snapshot retries.
    pub generation: u64,
    pub frozen: bool,
}

// ---------- plugin wiring ----------

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
        .init_resource::<MeadowMeshShaderSources>()
        .init_resource::<MeadowMeshMvTarget>()
        .add_systems(RenderStartup, init_meadow_mesh_path)
        .add_systems(ExtractSchedule, extract_meadow_shader_sources)
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
            ),
        );
}

// ---------- RenderStartup: detection + static modules ----------

/// Meadow bind group layout. All entries visible to TASK|MESH|FRAGMENT —
/// the fragment only reads bindings 0 (palette) and 6 (MV matrices), but
/// a superset visibility is valid and keeps this a single layout.
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
                uniform_buffer::<MeadowMeshViewUniform>(true), // 6 mesh_view (dynamic)
            ),
        ),
    )
}

/// Assemble the task/mesh module source: `enable` directive + the shared
/// struct/geometry library (plain WGSL) + the mesh shader body.
fn assemble_geom_source() -> String {
    let shared = include_str!("meadow_shared.wesl");
    let body = include_str!("meadow_mesh.wgsl");
    format!("enable wgpu_mesh_shader;\n{shared}\n{body}")
}

fn init_meadow_mesh_path(
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<MeadowMeshPipelines>,
    mut path_state: ResMut<MeadowMeshPathActive>,
) {
    let features = render_device.features();
    let limits = render_device.limits();
    // Budgets from `meadow_mesh.wgsl`, derived from the shared consts:
    // MESH_TASK_BLADES-wide task workgroups passing a payload of
    // MESH_TASK_BLADES fully-derived survivors, mesh workgroups emitting
    // ≤MESH_OUT_VERTS/≤MESH_OUT_PRIMS, and at most
    // ceil(MESH_TASK_BLADES / MESH_WG_TUFTS) mesh workgroups per task
    // workgroup (tufts pack fewest blades).
    let supported = features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
        && limits.max_task_invocations_per_workgroup >= MESH_TASK_BLADES
        && limits.max_task_payload_size >= 16 + MESH_TASK_BLADES * MESH_SURVIVOR_BLADE_BYTES
        && limits.max_mesh_output_vertices >= MESH_OUT_VERTS
        && limits.max_mesh_output_primitives >= MESH_OUT_PRIMS
        && limits.max_mesh_workgroup_total_count >= MESH_TASK_BLADES.div_ceil(MESH_WG_TUFTS);

    if !supported {
        info!(
            "meadow mesh-shader path unavailable (EXPERIMENTAL_MESH_SHADER: {}); using compute path",
            features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
        );
        pipelines.supported = false;
        return;
    }
    // The takeover itself is logged by `decide_meadow_mesh_path` when the
    // pipelines land (and composition failures downgrade to the flat
    // fragment with their own warning) — don't promise it here.
    info!("meadow mesh-shader path available");

    let device = render_device.wgpu_device();
    pipelines.geom_module = Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("meadow_mesh_geom"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(assemble_geom_source())),
    }));
    let meadow_bgl_desc = meadow_mesh_bgl_desc();
    // One empty bind group covers the main pipeline's slot 2 and the
    // shadow pipeline's slots 0-2 (wgpu dedups identical layouts, so it's
    // compatible anywhere an empty layout appears).
    let empty_layout = pipeline_cache.get_bind_group_layout(&empty_bgl_desc());
    pipelines.empty_bind_group =
        Some(render_device.create_bind_group(Some("meadow_mesh_empty"), &empty_layout, &[]));
    pipelines.meadow_bgl_desc = Some(meadow_bgl_desc);

    // MV composite: plain fullscreen pipeline, everything about it is
    // static — build it once here.
    let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("meadow_mv_composite"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MV_COMPOSITE_SOURCE)),
    });
    let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("meadow_mv_composite_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("meadow_mv_composite_pipeline_layout"),
        bind_group_layouts: &[Some(&composite_bgl)],
        immediate_size: 0,
    });
    pipelines.composite_pipeline = Some(device.create_render_pipeline(
        &wgpu::RenderPipelineDescriptor {
            label: Some("meadow_mv_composite_pipeline"),
            layout: Some(&composite_layout),
            vertex: wgpu::VertexState {
                module: &composite_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: MOTION_VECTOR_PREPASS_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        },
    ));
    pipelines.composite_bgl = Some(composite_bgl);
    pipelines.supported = true;
    // Lets the compute path's prepare (always compiled) keep the task
    // work list warm for this device.
    path_state.available = true;
}

/// Empty layout for the pipeline slots the meadow groups don't use.
fn empty_bgl_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new("meadow_mesh_empty_layout", &[])
}

/// Create/resize the meadow-owned motion target for the main view, in
/// single-sample MV-prepass configs (DLSS/TAA).
fn prepare_meadow_mesh_mv_target(
    views: Query<
        (&ExtractedCamera, &Msaa, Has<MotionVectorPrepass>),
        (With<Camera3d>, Without<LightEntity>),
    >,
    pipelines: Res<MeadowMeshPipelines>,
    force: Res<MeadowForceComputePath>,
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
    let wanted = views.iter().next().and_then(|(camera, msaa, has_mv)| {
        (has_mv && msaa.samples() == 1)
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
    let Some(composite_bgl) = &pipelines.composite_bgl else {
        return;
    };
    let device = render_device.wgpu_device();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("meadow_mesh_motion_target"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("meadow_mv_composite_bind_group"),
        layout: composite_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    *target = MeadowMeshMvTarget {
        view: Some(view),
        size,
        composite_bind_group: Some(composite_bind_group),
    };
}

// ---------- ExtractSchedule ----------

/// Snapshot the composable shader libraries for the PBR fragment
/// composition. Re-runs only while unsettled AND when a shader asset
/// actually loaded/changed — the full-library clone is ~1 MB of strings,
/// so an every-frame refresh would be wasteful in configs where
/// composition never gets to run (e.g. no meadow spawned yet).
fn extract_meadow_shader_sources(
    shaders: Extract<Res<Assets<Shader>>>,
    pipelines: Res<MeadowMeshPipelines>,
    mut out: ResMut<MeadowMeshShaderSources>,
) {
    if out.frozen || !pipelines.supported || (!out.by_module.is_empty() && !shaders.is_changed()) {
        return;
    }
    out.by_module.clear();
    for (_, shader) in shaders.iter() {
        let ShaderImport::Custom(name) = &shader.import_path else {
            continue;
        };
        if !is_collectable_module(name) {
            continue;
        }
        let Some(path) = custom_module_path(name) else {
            continue;
        };
        out.by_module.insert(path, shader.clone());
    }
    out.generation += 1;
}

/// Module names the collection above keeps: bevy's shader library plus
/// the meadow shared library. A `ModuleNotFound` for one of these can
/// still resolve on a later frame (the asset streams in); anything else
/// can never appear and is a permanent composition failure.
fn is_collectable_module(name: &str) -> bool {
    name.starts_with("bevy_pbr::")
        || name.starts_with("bevy_render::")
        || name.starts_with("bevy_core_pipeline::")
        || name == "bevy_meadow::meadow_shared"
}

/// The wesl module path of a `ShaderImport::Custom` name (bevy derives
/// these from `embedded://` paths, e.g.
/// `bevy_pbr::render::pbr_functions`).
fn custom_module_path(name: &str) -> Option<wesl::syntax::ModulePath> {
    let mut segments = name.split("::");
    let package = segments.next().filter(|s| !s.is_empty())?;
    let components = segments
        .map(|s| (!s.is_empty()).then(|| s.to_string()))
        .collect::<Option<Vec<_>>>()?;
    Some(wesl::syntax::ModulePath {
        origin: wesl::syntax::PathOrigin::Package(package.to_string()),
        components,
    })
}

// ---------- PBR fragment composition ----------

/// The composed fragment source. Mirrors what the compute path's
/// `meadow.wesl` forward fragment produces (season palette × height shade
/// × clump luminance through `apply_pbr_lighting`, shadow-receiver bit
/// forced, tip emissive), but builds the `PbrInput` by hand — there is no
/// material or mesh bind group on this pipeline, so
/// `pbr_input_from_standard_material` is not usable. The hardcoded
/// material values mirror the meadow `StandardMaterial`
/// (`plugin.rs::register_variant`: white base, perceptual_roughness 0.85,
/// double-sided). Varyings MUST match `MeadowVertexOut` in
/// `meadow_mesh.wgsl` by location.
const PBR_FRAGMENT_SOURCE: &str = r#"
import bevy_pbr::render::{
    mesh_view_bindings::view,
    pbr_types::pbr_input_new,
    pbr_functions::{
        apply_pbr_lighting, main_pass_post_lighting_processing, calculate_view,
        prepare_world_normal,
    },
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
};
import bevy_meadow::meadow_shared::VariantParams;

struct MeadowMeshView {
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    prev_clip_from_world: mat4x4<f32>,
    params: vec4<u32>,
}

@group(3) @binding(0) var<uniform> variant_params: VariantParams;
@group(3) @binding(6) var<uniform> mesh_view: MeadowMeshView;

struct MeadowVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) prev_world_position: vec4<f32>,
    @location(2) misc: vec2<f32>,
}

// MIRROR: meadow.wesl::season_palette.
fn season_palette() -> vec3<f32> {
    let a_idx = u32(variant_params.season_blend.x);
    let b_idx = u32(variant_params.season_blend.y);
    let t = variant_params.season_blend.z;
    var a = variant_params.palette_summer.rgb;
    var b = variant_params.palette_summer.rgb;
    if (a_idx == 0u) { a = variant_params.palette_spring.rgb; }
    else if (a_idx == 1u) { a = variant_params.palette_summer.rgb; }
    else if (a_idx == 2u) { a = variant_params.palette_autumn.rgb; }
    else { a = variant_params.palette_winter.rgb; }
    if (b_idx == 0u) { b = variant_params.palette_spring.rgb; }
    else if (b_idx == 1u) { b = variant_params.palette_summer.rgb; }
    else if (b_idx == 2u) { b = variant_params.palette_autumn.rgb; }
    else { b = variant_params.palette_winter.rgb; }
    return mix(a, b, clamp(t, 0.0, 1.0));
}

fn shade_meadow_fragment(in: MeadowVertexOut, is_front: bool) -> vec4<f32> {
    var pbr_input = pbr_input_new();
    let palette = season_palette();
    let shade = mix(0.70, 1.00, in.misc.x);
    let lum = in.misc.y;
    // Mirror of meadow.wesl::apply_blade_palette with the meadow
    // StandardMaterial's values baked in (white base → blade_lum = lum,
    // roughness 0.85).
    pbr_input.material.base_color = vec4<f32>(palette * shade * lum, 1.0);
    pbr_input.material.perceptual_roughness = 0.85;
    let tip_glow = smoothstep(0.7, 1.0, in.misc.x) * 0.06;
    pbr_input.material.emissive = vec4<f32>(palette * tip_glow, 1.0);
    pbr_input.frag_coord = in.position;
    pbr_input.world_position = in.world_position;
    // Same hardcoded upright normal as the compute path, including its
    // double-sided back-face flip (the meadow StandardMaterial sets
    // double_sided, so pbr_input_from_standard_material flips there).
    pbr_input.world_normal = prepare_world_normal(vec3<f32>(0.0, 1.0, 0.0), true, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr_input.V = calculate_view(in.world_position, pbr_input.is_orthographic);
    // No real MeshUniform row exists for mesh-emitted geometry; force the
    // shadow-receiver bit the same way meadow.wesl does.
    pbr_input.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;

    var color = apply_pbr_lighting(pbr_input);
    color = main_pass_post_lighting_processing(pbr_input, color);
    return color;
}

@fragment
fn fragment(in: MeadowVertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    return shade_meadow_fragment(in, is_front);
}

// MV variant: also writes the motion vector + valid flag to the
// meadow-owned motion target (color target 1) — composited into bevy's
// MV prepass texture by `meadow_mesh_composite_pass`.
struct MeadowPbrMvOut {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec4<f32>,
}

// MIRROR: meadow_mesh.wgsl::meadow_motion_vector (pinned by the mirror
// test) — same jitter-free NDC delta the flat fallback writes.
fn meadow_motion_vector(world: vec4<f32>, prev_world: vec4<f32>) -> vec2<f32> {
    let clip = mesh_view.unjittered_clip_from_world * vec4<f32>(world.xyz, 1.0);
    let prev_clip = mesh_view.prev_clip_from_world * vec4<f32>(prev_world.xyz, 1.0);
    let ndc = clip.xy / clip.w;
    let prev_ndc = prev_clip.xy / prev_clip.w;
    return (ndc - prev_ndc) * vec2<f32>(0.5, -0.5);
}

@fragment
fn fragment_mv(in: MeadowVertexOut, @builtin(front_facing) is_front: bool) -> MeadowPbrMvOut {
    var out: MeadowPbrMvOut;
    out.color = shade_meadow_fragment(in, is_front);
    out.motion = vec4<f32>(
        meadow_motion_vector(in.world_position, in.prev_world_position),
        1.0,
        0.0,
    );
    return out;
}
"#;

/// Fullscreen composite: copy valid texels from the meadow-owned motion
/// target into bevy's Rg16Float MV prepass texture. Standalone WGSL —
/// a plain (vertex) render pipeline, nothing mesh-shader about it.
const MV_COMPOSITE_SOURCE: &str = r#"
@group(0) @binding(0) var meadow_motion: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle.
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let v = textureLoad(meadow_motion, vec2<i32>(pos.xy), 0);
    if (v.z == 0.0) {
        discard;
    }
    return v.xy;
}
"#;

/// Which PBR fragment to compose. `Forward` is the only variant today;
/// a deferred G-buffer fragment slots in as a second arm carrying its
/// own root source + import roots.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeadowPbrVariant {
    Forward,
}

impl MeadowPbrVariant {
    fn root_source(self) -> &'static str {
        match self {
            Self::Forward => PBR_FRAGMENT_SOURCE,
        }
    }

    /// Module-level roots of the variant's `import` statements — the def
    /// closure walks the shader-library import graph from these. Must
    /// stay in sync with the imports in [`Self::root_source`].
    fn import_roots(self) -> &'static [&'static str] {
        match self {
            Self::Forward => &[
                "bevy_pbr::render::mesh_view_bindings",
                "bevy_pbr::render::pbr_types",
                "bevy_pbr::render::pbr_functions",
                "bevy_pbr::render::mesh_types",
                "bevy_meadow::meadow_shared",
            ],
        }
    }
}

/// Composition failure, classified for the caller's retry policy.
enum PbrComposeError {
    /// An imported module isn't in the collected sources yet but matches
    /// the collected prefixes, so it can still stream in — retry when
    /// the collection refreshes. Carries the module name.
    MissingModule(String),
    /// Anything else (syntax/validation errors, a module outside the
    /// collected set) — permanent for this def set + source snapshot.
    Permanent(String),
}

/// The innermost `ModuleNotFound` module path in a wesl error, if any.
fn missing_module_path(error: &wesl::Error) -> Option<&wesl::syntax::ModulePath> {
    match error {
        wesl::Error::ResolveError(wesl::ResolveError::ModuleNotFound(path, _))
        | wesl::Error::ImportError(wesl::ImportError::ResolveError(
            wesl::ResolveError::ModuleNotFound(path, _),
        )) => Some(path),
        wesl::Error::Error(diagnostic) => missing_module_path(&diagnostic.error),
        _ => None,
    }
}

/// `pkg::a::b` name of a package-origin module path (the inverse of
/// [`custom_module_path`]); other origins render via `Display`.
fn module_path_name(path: &wesl::syntax::ModulePath) -> String {
    match &path.origin {
        wesl::syntax::PathOrigin::Package(pkg) => std::iter::once(pkg.as_str())
            .chain(path.components.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("::"),
        _ => path.to_string(),
    }
}

/// Wesl import resolver over the collected shader-library snapshot, plus
/// bevy's virtual `constants` module (shader defs with values) and the
/// virtual root module holding the inline fragment source. Mirrors
/// bevy_shader's `ShaderResolver`.
struct MeadowShaderResolver<'a> {
    sources: &'a StdHashMap<wesl::syntax::ModulePath, Shader>,
    constants_source: &'a str,
    root_path: &'a wesl::syntax::ModulePath,
    root_source: &'a str,
}

impl wesl::Resolver for MeadowShaderResolver<'_> {
    fn resolve_source(
        &self,
        module_path: &wesl::syntax::ModulePath,
    ) -> Result<Cow<'_, str>, wesl::ResolveError> {
        let module_path = self.canonical_path(module_path);
        if module_path == *self.root_path {
            return Ok(Cow::Borrowed(self.root_source));
        }
        if module_path.origin == wesl::syntax::PathOrigin::Package("constants".to_string())
            && module_path.components.is_empty()
        {
            return Ok(Cow::Borrowed(self.constants_source));
        }
        self.sources
            .get(&module_path)
            .map(|shader| Cow::Borrowed(shader.source.as_str()))
            .ok_or_else(|| {
                wesl::ResolveError::ModuleNotFound(
                    module_path.clone(),
                    "module not collected from the shader library".to_string(),
                )
            })
    }

    fn canonical_path(&self, module_path: &wesl::syntax::ModulePath) -> wesl::syntax::ModulePath {
        // Collapse a '/'-bearing package to its last segment, like
        // bevy_shader's resolver.
        match &module_path.origin {
            wesl::syntax::PathOrigin::Package(pkg) if pkg.contains('/') => {
                wesl::syntax::ModulePath {
                    origin: wesl::syntax::PathOrigin::Package(
                        pkg.rsplit('/').next().unwrap().to_string(),
                    ),
                    components: module_path.components.clone(),
                }
            }
            _ => module_path.clone(),
        }
    }

    fn display_name(&self, module_path: &wesl::syntax::ModulePath) -> Option<String> {
        let module_path = self.canonical_path(module_path);
        Some(self.sources.get(&module_path)?.path.clone())
    }
}

/// Compose the PBR fragment to WGSL text with bevy_shader's exact wesl
/// recipe: feature flags + a virtual `constants` module built from the
/// def list, the import-closure defs of the library modules, and the
/// device-global defs; compiled with `EscapeMangler` (root declarations
/// — the entry points — keep their names).
fn compose_pbr_fragment_wgsl(
    sources: &MeadowMeshShaderSources,
    defs: &[ShaderDefVal],
    max_storage_buffers_per_shader_stage: u32,
    rec2020: bool,
    variant: MeadowPbrVariant,
) -> Result<String, PbrComposeError> {
    let root_path = wesl::syntax::ModulePath {
        origin: wesl::syntax::PathOrigin::Package("bevy_meadow".to_string()),
        components: vec!["mesh_pbr_fragment".to_string()],
    };

    let mut compiler_options = wesl::CompileOptions {
        imports: true,
        condcomp: true,
        ..Default::default()
    };

    // Def closure: the shader_defs of every library module transitively
    // imported by the root. Library defs (e.g. mesh_view_types' MAX_*
    // constants) live on the Shader assets, not in the pipeline's def
    // list — bevy's shader cache does the same walk.
    let mut closure_defs: Vec<&ShaderDefVal> = Vec::new();
    let mut visited: std::collections::HashSet<wesl::syntax::ModulePath> =
        std::collections::HashSet::new();
    let mut to_visit: Vec<wesl::syntax::ModulePath> = variant
        .import_roots()
        .iter()
        .filter_map(|name| custom_module_path(name))
        .collect();
    while let Some(path) = to_visit.pop() {
        if !visited.insert(path.clone()) {
            continue;
        }
        let Some(shader) = sources.by_module.get(&path) else {
            // Item-level import candidates and not-yet-collected modules;
            // the compile itself reports genuinely missing modules.
            continue;
        };
        closure_defs.extend(shader.shader_defs.iter());
        for import in &shader.imports {
            if let ShaderImport::Custom(name) = import
                && let Some(import_path) = custom_module_path(name)
            {
                to_visit.push(import_path);
            }
        }
    }

    // Device-global defs bevy's PipelineCache registers outside the
    // specialize def list (it splices them into every shader inside its
    // private cache, so they never appear on the `Assets<Shader>` clones
    // we collect). Applied last, like the root shader's own defs in
    // bevy's fold.
    let mut global_defs: Vec<ShaderDefVal> = vec![
        ShaderDefVal::UInt(
            "AVAILABLE_STORAGE_BUFFER_BINDINGS".into(),
            max_storage_buffers_per_shader_stage,
        ),
        ShaderDefVal::Bool(
            "AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3".into(),
            max_storage_buffers_per_shader_stage >= 3,
        ),
        ShaderDefVal::Bool(
            "AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_6".into(),
            max_storage_buffers_per_shader_stage >= 6,
        ),
    ];
    if rec2020 {
        global_defs.push(ShaderDefVal::Bool("WORKING_COLOR_SPACE_REC2020".into(), true));
    }

    let mut constants = std::collections::BTreeMap::new();
    for shader_def in closure_defs
        .into_iter()
        .chain(defs.iter())
        .chain(global_defs.iter())
    {
        match shader_def {
            ShaderDefVal::Bool(key, value) => {
                compiler_options
                    .features
                    .flags
                    .insert(key.to_string(), (*value).into());
            }
            ShaderDefVal::Int(key, value) => {
                compiler_options
                    .features
                    .flags
                    .insert(key.to_string(), true.into());
                constants.insert(key.as_ref(), value.to_string());
            }
            ShaderDefVal::UInt(key, value) => {
                compiler_options
                    .features
                    .flags
                    .insert(key.to_string(), true.into());
                constants.insert(key.as_ref(), value.to_string());
            }
        }
    }
    let constants_source: String = constants
        .iter()
        .map(|(name, value)| format!("const {name} = {value};\n"))
        .collect();

    let resolver = MeadowShaderResolver {
        sources: &sources.by_module,
        constants_source: &constants_source,
        root_path: &root_path,
        root_source: variant.root_source(),
    };

    let compiled = wesl::compile_sourcemap(
        &root_path,
        &resolver,
        &wesl::EscapeMangler,
        &compiler_options,
    )
    .map_err(|error| {
        if let Some(path) = missing_module_path(&error) {
            let name = module_path_name(path);
            if is_collectable_module(&name) {
                return PbrComposeError::MissingModule(name);
            }
        }
        PbrComposeError::Permanent(format!("composing meadow PBR fragment: {error}"))
    })?;

    Ok(compiled.to_string())
}

/// Compose the PBR fragment and create the raw shader module. The WGSL
/// output is handed to wgpu, which validates it against the real device
/// (a genuine capability miss fails there and the caller falls back to
/// the flat fragment).
fn compose_pbr_fragment(
    sources: &MeadowMeshShaderSources,
    defs: &[ShaderDefVal],
    render_device: &RenderDevice,
    rec2020: bool,
    variant: MeadowPbrVariant,
) -> Result<wgpu::ShaderModule, PbrComposeError> {
    let wgsl = compose_pbr_fragment_wgsl(
        sources,
        defs,
        render_device.limits().max_storage_buffers_per_shader_stage,
        rec2020,
        variant,
    )?;
    Ok(render_device
        .wgpu_device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meadow_mesh_pbr_fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl)),
        }))
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

#[allow(clippy::too_many_arguments)]
fn prepare_meadow_mesh_pipelines(
    views: Query<
        (&ExtractedView, &Msaa, Has<MotionVectorPrepass>),
        (With<Camera3d>, Without<LightEntity>),
    >,
    drivers: Res<RenderMeadowDriver>,
    specialized: Res<SpecializedMaterialPipelineCache>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    working_color_space: Res<bevy::render::WorkingColorSpace>,
    force: Res<MeadowForceComputePath>,
    mut pipelines: ResMut<MeadowMeshPipelines>,
    mut shader_sources: ResMut<MeadowMeshShaderSources>,
) {
    pipelines.current_main_key = None;
    // Forced-compute leaves the key `None`, so `decide_meadow_mesh_path`
    // can never activate against a stale key — the frame the force flag
    // unflips, the key (and any missing pipeline) is recomputed here
    // before the decision runs.
    if !pipelines.supported || force.0 {
        return;
    }
    let Some((view, msaa, has_mv)) = views.iter().next() else {
        return;
    };
    // The meadow material's own specialized forward pipeline for this
    // view: its shader defs + view bind group layouts + color target are
    // the ground truth we build the raw mesh pipeline against.
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
    // The specialized cache hands out ids at queue time, but
    // `get_render_pipeline_descriptor` PANICS for ids the pipeline cache
    // hasn't processed yet (freshly specialized this frame). Waiting for
    // the compiled pipeline is bounds-safe and also guarantees we mirror
    // a descriptor that actually built.
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

    // Compose (or re-compose on def change) the PBR fragment. The def
    // set changes with the view config (DLSS/TAA toggling prepasses,
    // MSAA), and it must match the view bind group layout, so both the
    // fragment module and the pipelines are keyed by its hash.
    let mut hasher = std::hash::DefaultHasher::new();
    frag.shader_defs.hash(&mut hasher);
    let defs_hash = hasher.finish();
    if pipelines.pbr_defs_hash != Some(defs_hash)
        && pipelines.pbr_failed != Some((defs_hash, shader_sources.generation))
    {
        match compose_pbr_fragment(
            &shader_sources,
            &frag.shader_defs,
            &render_device,
            working_color_space.is_rec2020(),
            MeadowPbrVariant::Forward,
        ) {
            Ok(module) => {
                info!("meadow mesh-shader PBR fragment composed");
                pipelines.pbr_fragment = Some(module);
                pipelines.pbr_defs_hash = Some(defs_hash);
                pipelines.pbr_failed = None;
                pipelines.pbr_missing_logged = None;
                // Stale-def pipelines are unreachable via the key; clear
                // to bound the map.
                pipelines.main_pipelines.clear();
                shader_sources.frozen = true;
            }
            Err(PbrComposeError::MissingModule(module)) => {
                // A library shader still streaming in — retry as assets
                // load. Logged once per module name so a renamed/removed
                // upstream module can't stall the mesh path silently.
                if pipelines.pbr_missing_logged.as_deref() != Some(module.as_str()) {
                    warn!(
                        "meadow PBR fragment waiting on shader module `{module}` \
                        (retries as shader assets load)"
                    );
                    pipelines.pbr_missing_logged = Some(module);
                }
                return;
            }
            Err(PbrComposeError::Permanent(err)) => {
                warn!("{err}; meadow mesh path falls back to flat-lit fragment");
                pipelines.pbr_failed = Some((defs_hash, shader_sources.generation));
                // The old module's view-binding declarations match the
                // OLD layout — unusable with pipelines built for the new
                // defs. Flat until a compose succeeds.
                pipelines.pbr_fragment = None;
                pipelines.pbr_defs_hash = None;
            }
        }
    }

    let key = MeadowMainPipelineKey {
        color_format: color_target.format,
        samples: descriptor.multisample.count,
        pbr: pipelines.pbr_fragment.is_some(),
        mv: has_mv && msaa.samples() == 1 && descriptor.multisample.count == 1,
        defs_hash,
    };
    pipelines.current_main_key = Some(key);

    let device = render_device.wgpu_device();
    // Clones are cheap handle bumps; owning them here frees `pipelines`
    // for the mutable pipeline inserts below.
    let (Some(geom_module), Some(meadow_bgl_desc)) = (
        pipelines.geom_module.clone(),
        pipelines.meadow_bgl_desc.clone(),
    ) else {
        return;
    };
    let geom_module = &geom_module;
    let meadow = pipeline_cache.get_bind_group_layout(&meadow_bgl_desc);
    let empty = pipeline_cache.get_bind_group_layout(&empty_bgl_desc());

    if !pipelines.main_pipelines.contains_key(&key) {
        // Bind group layouts: bevy's view main + binding-array layouts
        // exactly as the specialized pipeline uses them, an empty slot 2
        // (the material pipeline has mesh/material groups there; we bind
        // an empty group), and the meadow group at [`MEADOW_GROUP`].
        let view_main = pipeline_cache.get_bind_group_layout(&descriptor.layout[0]);
        let view_arrays = pipeline_cache.get_bind_group_layout(&descriptor.layout[1]);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meadow_mesh_main_layout"),
            bind_group_layouts: &[
                Some(view_main.deref()),
                Some(view_arrays.deref()),
                Some(empty.deref()),
                Some(meadow.deref()),
            ],
            immediate_size: 0,
        });

        let (frag_module, entry): (&wgpu::ShaderModule, &str) =
            match (&pipelines.pbr_fragment, key.mv) {
                (Some(m), false) => (m, "fragment"),
                (Some(m), true) => (m, "fragment_mv"),
                (None, false) => (geom_module, "meadow_frag_flat"),
                (None, true) => (geom_module, "meadow_frag_flat_mv"),
            };
        let mut targets: Vec<Option<wgpu::ColorTargetState>> = vec![Some(color_target)];
        if key.mv {
            targets.push(Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }
        let pipeline = create_meadow_mesh_pipeline(
            device,
            "meadow_mesh_main_pipeline",
            &layout,
            geom_module,
            "meadow_mesh",
            Some(wgpu::FragmentState {
                module: frag_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            key.samples,
            false,
        );
        info!("meadow mesh-shader main pipeline created ({key:?})");
        pipelines.main_pipelines.insert(key, pipeline);
    }

    if pipelines.shadow_pipeline.is_none() {
        // The geometry module hardcodes the meadow group at
        // [`MEADOW_GROUP`]; slots 0-2 are empty layouts here (the pass
        // binds the cached empty bind group), so one module serves both
        // the main and shadow pipeline layouts.
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meadow_mesh_shadow_layout"),
            bind_group_layouts: &[
                Some(empty.deref()),
                Some(empty.deref()),
                Some(empty.deref()),
                Some(meadow.deref()),
            ],
            immediate_size: 0,
        });
        // `unclipped_depth` matters: bevy's directional-shadow pipelines
        // depth-CLAMP ortho casters that sit between the light and the
        // cascade volume, rather than clipping them away. Without it,
        // grass outside a cascade's depth range silently drops out —
        // observed as sparser, flatter grass shadows vs the compute path
        // (which rides bevy's own prepass pipelines).
        let unclipped_depth = render_device
            .features()
            .contains(wgpu::Features::DEPTH_CLIP_CONTROL);
        let pipeline = create_meadow_mesh_pipeline(
            device,
            "meadow_mesh_shadow_pipeline",
            &layout,
            geom_module,
            "meadow_mesh_shadow",
            None,
            1,
            unclipped_depth,
        );
        info!("meadow mesh-shader shadow pipeline created");
        pipelines.shadow_pipeline = Some(pipeline);
    }
}

/// Create a meadow mesh pipeline: shared `meadow_task` stage and the
/// depth state every meadow raster pass uses — Depth32Float,
/// `GreaterEqual` (reverse-Z), write on, no stencil/bias (shadow bias is
/// applied receiver-side in bevy's shadow sampling), two-sided
/// primitives.
#[allow(clippy::too_many_arguments)]
fn create_meadow_mesh_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    geom_module: &wgpu::ShaderModule,
    mesh_entry: &str,
    fragment: Option<wgpu::FragmentState>,
    samples: u32,
    unclipped_depth: bool,
) -> wgpu::RenderPipeline {
    device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        task: Some(wgpu::TaskState {
            module: geom_module,
            entry_point: Some("meadow_task"),
            compilation_options: Default::default(),
        }),
        mesh: wgpu::MeshState {
            module: geom_module,
            entry_point: Some(mesh_entry),
            compilation_options: Default::default(),
        },
        fragment,
        primitive: wgpu::PrimitiveState {
            cull_mode: None, // blades are two-sided
            unclipped_depth,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: samples,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    })
}

// ---------- Prepare: path decision ----------

/// Flip [`MeadowMeshPathActive`] for this frame. All-or-nothing: the mesh
/// path takes over main + shadow views together, or the compute path
/// serves everything (mixing per-view paths within a frame is valid, but
/// coupling them keeps path selection and the buffer-cap logic
/// simple).
fn decide_meadow_mesh_path(
    pipelines: Res<MeadowMeshPipelines>,
    force: Res<MeadowForceComputePath>,
    views: Query<Has<DeferredPrepass>, (With<Camera3d>, With<ExtractedView>)>,
    mv_target: Res<MeadowMeshMvTarget>,
    mut active: ResMut<MeadowMeshPathActive>,
) {
    let deferred = views.iter().any(|d| d);
    let ready = pipelines.supported
        && !force.0
        && !deferred
        && pipelines.shadow_pipeline.is_some()
        && pipelines.current_main_key.is_some_and(|key| {
            pipelines.main_pipelines.contains_key(&key) && (!key.mv || mv_target.view.is_some())
        });
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
    pipelines: Res<MeadowMeshPipelines>,
    force: Res<MeadowForceComputePath>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut bind_groups: ResMut<MeadowMeshBindGroups>,
) {
    if !pipelines.supported || force.0 {
        return;
    }
    let Some(meadow_bgl_desc) = &pipelines.meadow_bgl_desc else {
        return;
    };
    bind_groups
        .by_variant
        .retain(|id, _| extracted.by_variant.contains_key(id));

    let Some(view_uniform_binding) = view_uniforms.buffer.binding() else {
        return;
    };
    let Some(view_uniform_buffer) = view_uniforms.buffer.buffer() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(meadow_bgl_desc);

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
        );
        if let Some((_, cached)) = bind_groups.by_variant.get(id)
            && *cached == fingerprint
        {
            continue;
        }

        let bg = render_device.create_bind_group(
            Some("meadow_mesh_bind_group"),
            &layout,
            &BindGroupEntries::sequential((
                params_buf.as_entire_binding(),
                patches.buffer.as_entire_binding(),
                trunk_slots.buffer.as_entire_binding(),
                &heightfield.texture_view,
                view_cull.as_entire_binding(),
                vb.task_slices.as_entire_binding(),
                view_uniform_binding.clone(),
            )),
        );
        bind_groups.by_variant.insert(*id, (bg, fingerprint));
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
    bind_groups: Res<MeadowMeshBindGroups>,
    view_uniforms: Res<MeadowMeshViewUniforms>,
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
    let Some(pipeline) = pipelines.main_pipelines.get(&key) else {
        // Config changed this frame (e.g. DLSS/MSAA toggle); the pipeline
        // for the new key is created next frame and the path re-activates.
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
        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
            view: mv_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
    {
        let raw = render_pass.wgpu_pass();
        raw.set_pipeline(pipeline);
        raw.set_bind_group(
            0,
            &*mesh_view_bind_group.main,
            &mesh_view_bind_group.main_offsets,
        );
        raw.set_bind_group(1, &*mesh_view_bind_group.binding_array, &[]);
        raw.set_bind_group(2, &*mesh_view_bind_group.empty, &[]);
        draw_meadow_task_lists(raw, view_uniforms.offsets[0], &bind_groups, &buffers);
    }
    pass_span.end(&mut render_pass);
}

/// Per-variant meadow dispatch: bind the variant's meadow group at
/// [`MEADOW_GROUP`] with the view's dynamic offset and issue the folded
/// 2D `draw_mesh_tasks` grid. The fold is a contract with the WGSL's
/// `flat = wg.y * MESH_TASK_DISPATCH_STRIDE + wg.x` reconstruction —
/// keep it in this one place.
fn draw_meadow_task_lists(
    raw: &mut wgpu::RenderPass,
    view_offset: u32,
    bind_groups: &MeadowMeshBindGroups,
    buffers: &MeadowGpuBuffers,
) {
    for (id, (bg, _)) in bind_groups.by_variant.iter() {
        let Some(vb) = buffers.by_variant.get(id) else {
            continue;
        };
        if vb.num_task_slices == 0 {
            continue;
        }
        raw.set_bind_group(MEADOW_GROUP, &**bg, &[view_offset]);
        let x = vb.num_task_slices.min(MESH_TASK_DISPATCH_STRIDE);
        let y = vb.num_task_slices.div_ceil(MESH_TASK_DISPATCH_STRIDE);
        raw.draw_mesh_tasks(x, y, 1);
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
    mv_target: Res<MeadowMeshMvTarget>,
    slots: Res<MeadowViewSlots>,
    mut ctx: RenderContext,
) {
    if !active.active {
        return;
    }
    let (Some(pipeline), Some(bind_group)) = (
        &pipelines.composite_pipeline,
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
    {
        let raw = render_pass.wgpu_pass();
        raw.set_pipeline(pipeline);
        raw.set_bind_group(0, bind_group, &[]);
        raw.draw(0..3, 0..1);
    }
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
    bind_groups: Res<MeadowMeshBindGroups>,
    view_uniforms: Res<MeadowMeshViewUniforms>,
    slots: Res<MeadowViewSlots>,
    buffers: Res<MeadowGpuBuffers>,
    mut ctx: RenderContext,
) {
    if !active.active || bind_groups.by_variant.is_empty() {
        return;
    }
    let (Some(pipeline), Some(empty_bind_group)) =
        (&pipelines.shadow_pipeline, &pipelines.empty_bind_group)
    else {
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
        {
            let raw = render_pass.wgpu_pass();
            raw.set_pipeline(pipeline);
            // The shadow layout's slots 0-2 are empty (the geometry
            // module hardcodes the meadow group at MEADOW_GROUP).
            for slot_index in 0..MEADOW_GROUP {
                raw.set_bind_group(slot_index, &**empty_bind_group, &[]);
            }
            draw_meadow_task_lists(
                raw,
                view_uniforms.offsets[slot as usize],
                &bind_groups,
                &buffers,
            );
        }
        pass_span.end(&mut render_pass);
    }
}

#[cfg(test)]
mod tests {
    use bevy::shader::{Shader, ShaderDefVal};
    use std::path::{Path, PathBuf};

    /// The bevy checkout the composer test reads the real shader library
    /// from: `BEVY_CHECKOUT` if set, else the sibling `../bevy` checkout.
    /// `None` (test skips) when neither holds a bevy source tree.
    fn bevy_checkout() -> Option<PathBuf> {
        let root = std::env::var_os("BEVY_CHECKOUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../bevy"));
        root.join("crates/bevy_pbr/src/render/pbr_functions.wesl")
            .exists()
            .then_some(root)
    }

    /// Register every `.wesl` under `dir` the way `load_shader_library!`
    /// does at runtime: `embedded://<crate>/<path-under-src>` paths, so
    /// module names and import scanning match the real asset registry.
    fn collect_wesl_dir(
        crate_name: &str,
        src_root: &Path,
        dir: &Path,
        out: &mut super::MeadowMeshShaderSources,
    ) {
        for entry in std::fs::read_dir(dir).expect("readable bevy source dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                collect_wesl_dir(crate_name, src_root, &path, out);
            } else if path.extension().is_some_and(|e| e == "wesl") {
                let rel = path
                    .strip_prefix(src_root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let source = std::fs::read_to_string(&path).expect("readable wesl file");
                let shader = Shader::from_wesl(source, format!("embedded://{crate_name}/{rel}"));
                let bevy::shader::ShaderImport::Custom(name) = &shader.import_path else {
                    panic!("embedded wesl path must derive a custom import");
                };
                let module_path = super::custom_module_path(name).expect("module path");
                out.by_module.insert(module_path, shader);
            }
        }
    }

    /// Run the composer offline against the REAL bevy shader library
    /// (skipped when no bevy checkout is present) with a representative
    /// forward def set: the composition must succeed, both entry points
    /// must survive under their unmangled names, and the output must be
    /// WGSL that naga parses + validates.
    #[test]
    fn pbr_fragment_composes_against_bevy_library() {
        let Some(root) = bevy_checkout() else {
            eprintln!("skipping: no bevy checkout at ../bevy (set BEVY_CHECKOUT)");
            return;
        };
        let mut sources = super::MeadowMeshShaderSources::default();
        for crate_name in ["bevy_pbr", "bevy_render", "bevy_core_pipeline"] {
            let src = root.join("crates").join(crate_name).join("src");
            collect_wesl_dir(crate_name, &src, &src, &mut sources);
        }
        // Library defs bevy registers through loader settings
        // (`mesh_view_types.wesl` in `MeshRenderPlugin::build`) — carried
        // on the Shader asset, folded in by the composer's def closure.
        let mesh_view_types = super::custom_module_path("bevy_pbr::render::mesh_view_types")
            .expect("module path");
        sources
            .by_module
            .get_mut(&mesh_view_types)
            .expect("mesh_view_types collected")
            .shader_defs = vec![
            ShaderDefVal::UInt("MAX_DIRECTIONAL_LIGHTS".into(), 10),
            ShaderDefVal::UInt("MAX_CASCADES_PER_LIGHT".into(), 4),
            ShaderDefVal::UInt("MAX_RECT_LIGHTS".into(), 8),
        ];
        // The meadow shared library, as `MeadowPlugin` registers it.
        sources.by_module.insert(
            super::custom_module_path("bevy_meadow::meadow_shared").expect("module path"),
            Shader::from_wesl(
                include_str!("meadow_shared.wesl"),
                "embedded://bevy_meadow/meadow_shared.wesl",
            ),
        );

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

        let wgsl = match super::compose_pbr_fragment_wgsl(
            &sources,
            &defs,
            12,
            false,
            super::MeadowPbrVariant::Forward,
        ) {
            Ok(wgsl) => wgsl,
            Err(super::PbrComposeError::MissingModule(module)) => {
                panic!("composer reported module `{module}` missing from a full checkout")
            }
            Err(super::PbrComposeError::Permanent(err)) => panic!("{err}"),
        };

        // The raw-wgpu pipelines reference these entry names directly.
        assert!(wgsl.contains("fn fragment("), "entry `fragment` lost");
        assert!(wgsl.contains("fn fragment_mv("), "entry `fragment_mv` lost");

        let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| {
            panic!(
                "composed PBR fragment failed to parse:\n{}",
                e.emit_to_string(&wgsl)
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("composed PBR fragment failed validation: {e:?}"));
    }

    /// The assembled task/mesh module must parse + validate with the same
    /// (workspace-patched) naga wgpu uses at runtime — the WGSL mesh
    /// shader frontend is new, so pin it in `cargo test` rather than
    /// discovering breakage at pipeline creation on the GPU box.
    #[test]
    fn geom_module_parses_and_validates() {
        let source = super::assemble_geom_source();
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
            panic!(
                "meadow_mesh.wgsl failed to parse:\n{}",
                e.emit_to_string(&source)
            )
        });
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("meadow_mesh.wgsl failed validation: {e:?}"));
    }

    /// Run the validated module through naga's SPIR-V backend with
    /// mesh-shading enabled (SPIR-V 1.6, like wgpu-hal picks on Vulkan
    /// 1.3+) — pins the mesh-shader codegen path end-to-end minus the
    /// driver, so backend regressions surface in `cargo test`.
    #[test]
    fn geom_module_compiles_to_spirv() {
        use naga::back::spv;
        let source = super::assemble_geom_source();
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
            .unwrap_or_else(|e| panic!("meadow_mesh.wgsl failed SPIR-V codegen: {e:?}"));
        assert!(!words.is_empty());
    }

    /// The derivation helpers are duplicated between the compute kernel
    /// and the mesh-shader module (see the MIRROR comments in both). The
    /// two paths must derive bit-identical blades — pin the shared
    /// function bodies to each other so drift fails the build.
    #[test]
    fn mirrored_helpers_match_compute_kernel() {
        let compute = include_str!("meadow_compute.wesl");
        let mesh = include_str!("meadow_mesh.wgsl");
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
            let body = &src[open..=end.unwrap_or_else(|| panic!("`{name}` body unterminated"))];
            body.lines()
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
                "`{f}` drifted between meadow_compute.wesl and meadow_mesh.wgsl"
            );
        }
        for f in ["wind_displacement", "season_palette"] {
            assert_eq!(
                body_of(raster, f),
                body_of(mesh, f),
                "`{f}` drifted between meadow.wesl and meadow_mesh.wgsl"
            );
        }

        // The composed PBR fragment (a Rust string, invisible to the
        // shader mirrors above) carries its own copies of the palette,
        // the motion-vector math, and the interface structs that must
        // match the mesh stage by location — pin them too. Struct bodies
        // reuse `body_of` via the `struct Name {` prefix.
        let pbr = super::PBR_FRAGMENT_SOURCE;
        assert_eq!(
            body_of(mesh, "season_palette"),
            body_of(pbr, "season_palette"),
            "`season_palette` drifted between meadow_mesh.wgsl and PBR_FRAGMENT_SOURCE"
        );
        assert_eq!(
            body_of(mesh, "meadow_motion_vector"),
            body_of(pbr, "meadow_motion_vector"),
            "`meadow_motion_vector` drifted between meadow_mesh.wgsl and PBR_FRAGMENT_SOURCE"
        );
        fn struct_of(src: &str, name: &str) -> String {
            let start = src
                .find(&format!("struct {name} {{"))
                .unwrap_or_else(|| panic!("`struct {name}` missing"));
            let end = start + src[start..].find('}').unwrap();
            src[start..=end]
                .lines()
                .map(|l| l.split("//").next().unwrap_or("").trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        }
        for s in ["MeadowVertexOut", "MeadowMeshView"] {
            assert_eq!(
                struct_of(mesh, s),
                struct_of(pbr, s),
                "`{s}` drifted between meadow_mesh.wgsl and PBR_FRAGMENT_SOURCE \
                 (fragment inputs are matched by location)"
            );
        }
    }
}
