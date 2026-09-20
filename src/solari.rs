//! `bevy_solari` integration: meadow blades as raytraced shadow casters and as
//! receivers whose rays start on the blade.
//!
//! Casters: one proxy entity per (live variant, band) registers the band's
//! [`MeadowRtBuffers`] output as [`RaytracingGeometry`], rebuilt every frame
//! because the blades sway. The near band is the exact 9-triangle ribbon, the
//! far band a 1-triangle chord. Requires [`MeadowRaytracingConfig::enabled`].
//!
//! Receivers: meadow's upright shading normal is not a blade's geometric
//! normal, so Solari's ordinary `position + normal * t_min` origin lands
//! inside or behind neighbouring blades. `meadow_receivers.wesl` fills Solari's
//! per-view receiver-override texture from the G-buffer bits both meadow raster
//! paths write (see `meadow_shared.wesl`). It runs for every view with
//! `SolariLighting::receiver_overrides` set, whether or not the casters are
//! enabled. The geometric-normal payload it reads is enabled by
//! [`MeadowSolariGeometryNormals`](crate::MeadowSolariGeometryNormals).
//!
//! Add [`MeadowSolariPlugin`] after [`MeadowPlugin`](crate::MeadowPlugin) in
//! sessions that light with Solari.

use std::borrow::Cow;

use bevy::asset::embedded_asset;
use bevy::core_pipeline::Core3d;
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::binding_types::{
    texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
    CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
    ShaderStages, ShaderType, StorageTextureAccess, TextureSampleType, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::{ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::ShaderCacheError;
use bevy_solari::realtime::{SolariLightingSystems, SolariReceiverOverrides};
use bevy_solari::scene::{
    RaytracingGeometry, RaytracingGeometryBuffers, RaytracingGeometryPreviousVertices,
    RaytracingGeometryTopologyGeneration, RaytracingGeometryUpdateMode, RaytracingInstanceTag,
    RaytracingSceneBindings,
};

use crate::compute::{
    MeadowRaytracingConfig, MeadowRtBuffers, MeadowRtDiagnostics, MeadowRtExpanded,
};
use crate::plugin::{MeadowPatch, MeadowVariantId, MeadowVariantRegistry};

/// [`RaytracingInstanceTag`] of near-band casters, the only geometry that
/// reproduces the rasterised blade and may anchor a receiver pixel. Far chords
/// carry no tag.
pub const MEADOW_RT_NEAR_TAG: u32 = 0x4D44_5701;

const MEADOW_RECEIVERS_SHADER: &str = "embedded://bevy_meadow/meadow_receivers.wesl";

/// Registers the caster proxies and the receiver-override pass.
pub struct MeadowSolariPlugin;

impl Plugin for MeadowSolariPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "meadow_receivers.wesl");

        app.init_resource::<MeadowReceiverOrigins>()
            .add_plugins((
                ExtractComponentPlugin::<MeadowRtProxy>::default(),
                ExtractResourcePlugin::<MeadowReceiverOrigins>::default(),
            ))
            .add_systems(Update, manage_meadow_rt_proxies);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                Render,
                (
                    // The inserts flush at the end of `PrepareResources`, ahead
                    // of Solari's `PrepareBindGroups` binder, so a slot
                    // reassignment and its generation reach Solari together.
                    bind_meadow_rt_buffers
                        .in_set(RenderSystems::PrepareResources)
                        .after(MeadowRtExpanded),
                    prepare_meadow_receiver_pipeline.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_systems(
                Core3d,
                write_meadow_receiver_overrides
                    .in_set(SolariLightingSystems::WriteReceiverOverrides),
            );
    }
}

/// Which ray-origin tiers the receiver pass writes for meadow pixels. All on
/// by default; each later tier refines the one before it. A consumer that
/// changes a field at runtime must also reset Solari's lighting history,
/// because reservoirs gathered under another origin policy stay confident.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, ExtractResource)]
#[extract_app(RenderApp)]
pub struct MeadowReceiverOrigins {
    /// Rays start at the depth-reconstructed position, without the offset along
    /// the upright shading normal. Applies to pixels that have no geometric
    /// normal (payload disabled, or `geometric_normals` off).
    pub unshifted: bool,
    /// Rays start at the depth-reconstructed position and are offset per ray
    /// along the packed geometric normal, toward the ray's own side.
    pub geometric_normals: bool,
    /// Rays start on the raytracing triangle under the pixel when a camera ray
    /// matches it to a near-band caster. Requires `geometric_normals` and live
    /// casters; unmatched pixels keep the `geometric_normals` tier.
    pub exact_anchors: bool,
}

impl Default for MeadowReceiverOrigins {
    fn default() -> Self {
        Self {
            unshifted: true,
            geometric_normals: true,
            exact_anchors: true,
        }
    }
}

// ---------- casters ----------

/// Which of meadow's two caster bands a proxy entity carries.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MeadowRtBand {
    Near,
    Far,
}

/// A main-world proxy entity carrying one (variant, band) of the expanded
/// blades into the TLAS. Removal is by despawn: Solari ignores `Visibility`.
#[derive(Component, Clone, Copy, ExtractComponent)]
#[extract_app(RenderApp)]
struct MeadowRtProxy {
    variant: MeadowVariantId,
    band: MeadowRtBand,
}

/// Keep one proxy entity per (live variant, band) while the casters are
/// enabled. Proxies carry a black-emissive `StandardMaterial` (only occlusion
/// matters for shadows) and [`RaytracingGeometry`].
fn manage_meadow_rt_proxies(
    mut commands: Commands,
    config: Option<Res<MeadowRaytracingConfig>>,
    registry: Option<Res<MeadowVariantRegistry>>,
    diagnostics: Option<Res<MeadowRtDiagnostics>>,
    patches: Query<&MeadowPatch>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    proxies: Query<(Entity, &MeadowRtProxy)>,
    mut material: Local<Option<Handle<StandardMaterial>>>,
) {
    let enabled = config.is_some_and(|c| c.enabled);
    if !enabled && proxies.is_empty() {
        return; // the common non-RT frame: no set-building, no registry walk
    }

    // Only variants with live patch entities get proxies: the registry
    // accumulates across world transitions and never shrinks, and a proxy for
    // a patch-less variant rebuilds an empty full-capacity BLAS every frame.
    // Meadow's own buffer allocation is gated on the same activity signal,
    // with a grace window.
    let live_variants: HashSet<MeadowVariantId> = patches.iter().map(|p| p.variant).collect();

    let mut want: HashSet<(MeadowVariantId, MeadowRtBand)> = HashSet::default();
    if enabled && let Some(registry) = registry.as_deref() {
        for (id, _) in registry.iter() {
            if live_variants.contains(id) {
                want.insert((*id, MeadowRtBand::Near));
                // The exact-only reference mode has no far band.
                if !diagnostics.as_ref().is_some_and(|d| d.exact_only) {
                    want.insert((*id, MeadowRtBand::Far));
                }
            }
        }
    }

    for (entity, proxy) in &proxies {
        if !want.remove(&(proxy.variant, proxy.band)) {
            commands.entity(entity).despawn();
        }
    }
    for (variant, band) in want {
        let mat = material
            .get_or_insert_with(|| {
                materials.add(StandardMaterial {
                    base_color: Color::srgb(0.3, 0.5, 0.2),
                    perceptual_roughness: 1.0,
                    metallic: 0.0,
                    emissive: LinearRgba::BLACK,
                    ..default()
                })
            })
            .clone();
        commands.spawn((
            MeadowRtProxy { variant, band },
            RaytracingGeometry,
            MeshMaterial3d(mat),
            // Blade vertices are in absolute world metres.
            Transform::IDENTITY,
        ));
    }
}

/// Render world: bind each proxy's band buffers so Solari rebuilds its BLAS
/// from them every frame. Checked every frame by buffer identity, so a proxy
/// re-binds when meadow recreates a variant's buffers. The topology generation
/// is synced independently of buffer identity: it changes whenever meadow
/// reassigns blade slots, which invalidates Solari's previous-frame anchors.
fn bind_meadow_rt_buffers(
    mut commands: Commands,
    rt: Res<MeadowRtBuffers>,
    proxies: Query<(
        Entity,
        &MeadowRtProxy,
        Option<&RaytracingGeometryBuffers>,
        Option<&RaytracingGeometryTopologyGeneration>,
    )>,
) {
    for (entity, proxy, existing, generation) in &proxies {
        let Some(rtv) = rt.by_variant.get(&proxy.variant) else {
            // Buffers gone (casters disabled mid-frame or variant dropped):
            // unbind so no BLAS rebuilds from buffers nothing writes. The
            // main-world reconciler despawns the proxy shortly after.
            if existing.is_some() {
                commands.entity(entity).remove::<(
                    RaytracingGeometryBuffers,
                    RaytracingGeometryPreviousVertices,
                    RaytracingGeometryTopologyGeneration,
                    RaytracingInstanceTag,
                )>();
            }
            continue;
        };
        let band = match proxy.band {
            MeadowRtBand::Near => &rtv.near,
            MeadowRtBand::Far => &rtv.far,
        };
        let up_to_date = existing.is_some_and(|b| {
            b.vertex_buffer.id() == band.vertices.id() && b.index_buffer.id() == band.indices.id()
        });
        if generation.is_none_or(|g| g.0 != band.topology_generation) {
            commands
                .entity(entity)
                .insert(RaytracingGeometryTopologyGeneration(
                    band.topology_generation,
                ));
        }
        if !up_to_date {
            let mut proxy_commands = commands.entity(entity);
            proxy_commands.insert((
                RaytracingGeometryPreviousVertices(band.previous_vertices.clone()),
                RaytracingGeometryBuffers {
                    vertex_buffer: band.vertices.clone(),
                    index_buffer: band.indices.clone(),
                    vertex_count: band.vertex_count,
                    index_count: band.index_count,
                    update_mode: RaytracingGeometryUpdateMode::RebuildEveryFrame,
                },
            ));
            if proxy.band == MeadowRtBand::Near {
                proxy_commands.insert(RaytracingInstanceTag(MEADOW_RT_NEAR_TAG));
            }
        }
    }
}

// ---------- receivers ----------

/// Mirror of `MeadowReceiverParams` in `meadow_receivers.wesl`.
#[derive(ShaderType, Clone, Copy)]
struct MeadowReceiverParams {
    unshifted: u32,
    geometric_normals: u32,
    exact_anchors: u32,
    near_tag: u32,
}

impl From<MeadowReceiverOrigins> for MeadowReceiverParams {
    fn from(origins: MeadowReceiverOrigins) -> Self {
        Self {
            unshifted: u32::from(origins.unshifted),
            geometric_normals: u32::from(origins.geometric_normals),
            exact_anchors: u32::from(origins.exact_anchors),
            near_tag: MEADOW_RT_NEAR_TAG,
        }
    }
}

#[derive(Resource)]
struct MeadowReceiverPipeline {
    /// Group 1; group 0 is Solari's scene bind group.
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    params: UniformBuffer<MeadowReceiverParams>,
    failure_logged: bool,
}

/// Queue the receiver pipeline the first frame Solari's scene bindings exist,
/// so it compiles alongside Solari's own pipelines, and upload the params
/// every frame. Until the pipeline is ready meadow pixels keep Solari's
/// ordinary origins.
fn prepare_meadow_receiver_pipeline(
    mut commands: Commands,
    scene_bindings: Option<Res<RaytracingSceneBindings>>,
    origins: Option<Res<MeadowReceiverOrigins>>,
    pipeline: Option<ResMut<MeadowReceiverPipeline>>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let params = MeadowReceiverParams::from(origins.as_deref().copied().unwrap_or_default());
    let Some(mut pipeline) = pipeline else {
        let Some(scene_bindings) = scene_bindings else {
            return;
        };
        let layout = BindGroupLayoutDescriptor::new(
            "meadow_receiver_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<ViewUniform>(true),
                    texture_2d(TextureSampleType::Uint), // G-buffer
                    texture_depth_2d(),
                    texture_storage_2d(
                        SolariReceiverOverrides::FORMAT,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<MeadowReceiverParams>(false),
                ),
            ),
        );
        let id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("meadow_receiver_overrides".into()),
            layout: vec![scene_bindings.bind_group_layout.clone(), layout.clone()],
            shader: asset_server.load(MEADOW_RECEIVERS_SHADER),
            entry_point: Some(Cow::from("resolve_meadow_receivers")),
            ..default()
        });
        let mut params = UniformBuffer::from(params);
        params.set_label(Some("meadow_receiver_params"));
        params.write_buffer(&render_device, &render_queue);
        commands.insert_resource(MeadowReceiverPipeline {
            layout,
            pipeline: id,
            params,
            failure_logged: false,
        });
        return;
    };

    pipeline.params.set(params);
    pipeline.params.write_buffer(&render_device, &render_queue);

    // Retries while the shader or an import is still loading are not failures.
    if !pipeline.failure_logged
        && let CachedPipelineState::Err(err) =
            pipeline_cache.get_compute_pipeline_state(pipeline.pipeline)
        && !matches!(
            err,
            ShaderCacheError::ShaderNotLoaded(_) | ShaderCacheError::ShaderImportNotYetAvailable
        )
    {
        warn!(
            "meadow receiver-override pipeline failed to build; grass keeps Solari's ordinary \
             ray origins"
        );
        pipeline.failure_logged = true;
    }
}

/// Write this view's receiver overrides, between Solari's clear of the texture
/// and its lighting passes. Independent of [`MeadowRaytracingConfig`]: receiver
/// pixels need their origin tiers with the casters off too.
fn write_meadow_receiver_overrides(
    view: ViewQuery<(
        &SolariReceiverOverrides,
        &ViewPrepassTextures,
        &ViewUniformOffset,
    )>,
    pipeline: Option<Res<MeadowReceiverPipeline>>,
    scene_bindings: Option<Res<RaytracingSceneBindings>>,
    pipeline_cache: Res<PipelineCache>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (overrides, prepass_textures, view_uniform_offset) = view.into_inner();
    let (Some(pipeline), Some(scene_bindings)) = (pipeline, scene_bindings) else {
        return;
    };
    let (
        Some(compute_pipeline),
        Some(scene_bind_group),
        Some(gbuffer),
        Some(depth_buffer),
        Some(view_binding),
        Some(params_binding),
    ) = (
        pipeline_cache.get_compute_pipeline(pipeline.pipeline),
        &scene_bindings.bind_group,
        prepass_textures.deferred_view(),
        prepass_textures.depth_only_view(),
        view_uniforms.uniforms.binding(),
        pipeline.params.binding(),
    )
    else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        Some("meadow_receiver_bind_group"),
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            view_binding,
            gbuffer,
            depth_buffer,
            overrides.write_view(),
            params_binding,
        )),
    );

    let size = overrides.size();
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("meadow_receiver_overrides"),
            timestamp_writes: None,
        });
    let span = diagnostics.time_span(&mut pass, "meadow_receiver_overrides");
    pass.set_pipeline(compute_pipeline);
    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_bind_group(1, &bind_group, &[view_uniform_offset.offset]);
    pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
    span.end(&mut pass);
}
