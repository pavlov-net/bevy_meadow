//! Custom render plumbing for `MeadowMaterial`.
//!
//! Bevy 0.19's auto-instancing batches at most one instance per entity
//! per draw — it cannot, on its own, produce the per-band
//! `instance_count` the compute pass computes from a single per-variant
//! render-driver entity. The standard `MaterialPlugin::<MeadowMaterial>`
//! chain ends in `DrawMesh`, which uses `item.batch_range()` (clobbered to
//! `output_index..output_index+1` by `batch_and_prepare_*` for
//! unbatchable items). To get one `draw_indexed_indirect` per (view, band)
//! with a GPU-written `instance_count`, we install a parallel chain
//! `DrawMeadow*` for the deferred / prepass / shadow phases and hijack the
//! meadow `PreparedMaterial`'s `draw_functions` map to point at our chain
//! instead of the default `DrawMaterial` / `DrawPrepass`.
//!
//! Per-instance identity is delivered via the `first_instance` field of
//! each indirect record: the compute pass front-packs a (view, band)'s
//! survivors into a contiguous `blades[]` region whose base is that record's
//! `first_instance`, so `@builtin(instance_index)` indexes `blades[]`
//! directly. The vertex layout stays identical to `DrawMaterial`'s chain —
//! so `MaterialExtension`'s standard specialization (deferred / prepass /
//! shadow shader compilation) keeps working.
//!
//! Hooks:
//! - `extract_meadow_mesh_ids` (ExtractSchedule): mirror the LOD mesh asset
//!   ids ([0]=blade, [1]=tuft) into the render world.
//! - `extract_meadow_material_ids` (ExtractSchedule): list of
//!   `MeadowMaterial` asset ids so the override system can filter
//!   `ErasedRenderAssets<PreparedMaterial>` to just meadow entries.
//! - `override_meadow_draw_functions` (Render::PrepareAssets, after
//!   `prepare_assets::<MeshMaterial3d<MeadowMaterial>>` and before
//!   `queue_material_meshes`): swap each meadow material's draw-fn
//!   ids so `queue_material_meshes` keys phase items into bins
//!   pointing at `DrawMeadow*` instead of `DrawMaterial` / `DrawPrepass`.
//! - `DrawMeadowPatch` (RenderCommand): the only command in the chain that
//!   differs from `DrawMaterial` / `DrawPrepass`. Resolves the driver's
//!   variant + the view's slot, then issues one `draw_indexed_indirect` per
//!   LOD band (skipping the tuft band for shadow views), each binding its
//!   band's mesh slices and indirect record.

use std::sync::Arc;

use bevy::asset::{AssetId, Assets, UntypedAssetId};
use bevy::core_pipeline::core_3d::Opaque3d;
use bevy::core_pipeline::deferred::Opaque3dDeferred;
use bevy::core_pipeline::prepass::Opaque3dPrepass;
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::resource::Resource;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::material::labels::DrawFunctionLabel;
use bevy::mesh::{BaseMeshPipelineKey, Mesh};
use bevy::pbr::MeshMaterial3d;
use bevy::pbr::{
    DeferredOpaqueDrawFunction, MainPassOpaqueDrawFunction, MeshPipelineKey, PreparedMaterial,
    PrepassOpaqueDepthOnlyDrawFunction, PrepassOpaqueDrawFunction, SetMaterialBindGroup,
    SetMeshBindGroup, SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup,
    SetPrepassEmptyMaterialBindGroup, SetPrepassViewBindGroup, SetPrepassViewEmptyBindGroup,
    Shadow, ShadowsDepthOnlyDrawFunction, ShadowsDrawFunction,
};
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::render::erased_render_asset::{ErasedRenderAssets, prepare_erased_assets};
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::{RenderAssets, prepare_assets};
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
    SetItemPipeline, TrackedRenderPass,
};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::ExtractedView;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};

use crate::compute::{
    MEADOW_VIEW_FLAG_SHADOW, MeadowMeshPathActive, MeadowViewSlots, RenderMeadowDriver,
    build_meadow_compute,
};
use crate::material::MeadowMaterial;
use crate::mesh::{INDIRECT_ARGS_SIZE, MEADOW_MAX_BANDS, MeadowMeshes};
use crate::plugin::MeadowVariantId;

/// Render-world list of `MeadowMaterial` asset ids. Used by the
/// override system to filter `ErasedRenderAssets<PreparedMaterial>`
/// to meadow-only entries; only meadow materials get their draw
/// functions swapped.
#[derive(Resource, Default)]
pub struct RenderMeadowMaterialIds {
    pub ids: HashSet<UntypedAssetId>,
}

/// One per registered variant. The single renderable entity that
/// produces a meadow phase item per view; it carries the blade-template
/// mesh + the variant's material (so the override + prepass-reads-material
/// machinery still keys on it) and a world-spanning `Aabb` +
/// `NoFrustumCulling` so it lands in every view's `VisibleEntities`
/// (main + every shadow cascade). `DrawMeadowPatch` reads
/// `variant` (via `RenderMeadowDriver`) to pick the per-variant blade /
/// indirect buffers, and `ExtractedView.retained_view_entity` to pick
/// the per-view slot.
#[derive(Component, Clone, Copy)]
pub struct MeadowRenderDriver {
    pub variant: MeadowVariantId,
}

/// Render plugin. Registered from `MeadowPlugin::build` after the
/// stock `MaterialPlugin::<MeadowMaterial>` so the standard
/// preparation pipeline is in place before our override hooks fire.
pub struct MeadowRenderPlugin;

impl Plugin for MeadowRenderPlugin {
    fn build(&self, app: &mut App) {
        // Main-world toggle to force the compute path (game code /
        // debug UI flips it; mirrored to the render world on change).
        #[cfg(feature = "mesh-shaders")]
        {
            app.init_resource::<crate::mesh_path::MeadowForceComputePath>();
            app.add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<
                crate::mesh_path::MeadowForceComputePath,
            >::default());
        }

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<RenderMeadowMaterialIds>()
            .init_resource::<RenderMeadowMeshIds>()
            // Main pass (deferred materials never enter Opaque3d but
            // we register here for safety so a forward-render fallback
            // works without re-plumbing).
            .add_render_command::<Opaque3d, DrawMeadowOpaque>()
            // Deferred opaque uses the prepass-shape chain.
            .add_render_command::<Opaque3dDeferred, DrawMeadowPrepass>()
            // Prepass (motion-vector / normal prepass).
            .add_render_command::<Opaque3dPrepass, DrawMeadowPrepass>()
            .add_render_command::<Opaque3dPrepass, DrawMeadowDepthOnlyPrepass>()
            // Shadow casters.
            .add_render_command::<Shadow, DrawMeadowPrepass>()
            .add_render_command::<Shadow, DrawMeadowDepthOnlyPrepass>()
            .add_systems(
                ExtractSchedule,
                (extract_meadow_material_ids, extract_meadow_mesh_ids),
            )
            .add_systems(
                Render,
                (
                    // Force ordering AFTER `prepare_erased_assets::<MeshMaterial3d<MeadowMaterial>>`
                    // so the `ErasedRenderAssets<PreparedMaterial>` map has
                    // meadow entries to override. Without this, the systems
                    // run in arbitrary order within `PrepareAssets`; if our
                    // override runs first, the map is still empty for
                    // meadow ids and we silently skip every entry — then
                    // `queue_material_meshes` keys phase items into bins
                    // pointing at the stock `DrawMaterial` chain (which
                    // calls `DrawMesh` and reads `batch_range = 0..1`
                    // because of `NoAutomaticBatching`), so each patch
                    // renders one degenerate blade and visually nothing.
                    override_meadow_draw_functions
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_erased_assets::<MeshMaterial3d<MeadowMaterial>>),
                    force_meadow_mesh_prepass_reads_material
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_assets::<RenderMesh>),
                ),
            );

        // GPU-driven cull/compact compute + per-view indirect machinery.
        build_meadow_compute(render_app);

        // Task/mesh-shader path (raw wgpu pipelines; runtime-gated on
        // EXPERIMENTAL_MESH_SHADER, force-compute toggle via
        // `MeadowForceComputePath`).
        #[cfg(feature = "mesh-shaders")]
        crate::mesh_path::build_meadow_mesh_path(render_app);
    }
}

/// Render-world snapshot of the LOD mesh asset ids, indexed by band
/// (`[0]` = blade, `[1]` = tuft). Populated by `extract_meadow_mesh_ids`
/// each frame so the compute prepare step, `DrawMeadowPatch`, and
/// `force_meadow_mesh_prepass_reads_material` can find them in
/// `RenderAssets<RenderMesh>` / `MeshAllocator`.
#[derive(Resource, Default)]
pub struct RenderMeadowMeshIds {
    pub lod: [Option<AssetId<Mesh>>; 2],
}

/// Snapshot the set of `MeadowMaterial` asset ids each frame so
/// `override_meadow_draw_functions` can filter the render-world's
/// global `ErasedRenderAssets<PreparedMaterial>` to just meadow
/// entries.
fn extract_meadow_material_ids(
    materials: Extract<Res<Assets<MeadowMaterial>>>,
    mut ids: ResMut<RenderMeadowMaterialIds>,
) {
    ids.ids.clear();
    for (id, _) in materials.iter() {
        ids.ids.insert(id.into());
    }
}

/// Mirror the LOD meshes' `AssetId`s from main world so the compute
/// prepare + draw + prepass-reads systems can locate them in the render
/// world. The handles are created once at plugin startup and stable for
/// the app's lifetime, so re-extracting each frame is cheap.
fn extract_meadow_mesh_ids(
    meadow_meshes: Extract<Option<Res<MeadowMeshes>>>,
    mut ids: ResMut<RenderMeadowMeshIds>,
) {
    ids.lod = meadow_meshes
        .as_deref()
        .map(|m| [Some(m.lod_meshes[0].id()), Some(m.lod_meshes[1].id())])
        .unwrap_or([None, None]);
}

/// Set `MeshPipelineKey::PREPASS_READS_MATERIAL` on BOTH LOD meshes'
/// render-world `key_bits`. Without this, the directional-light shadow
/// pass and any depth-only main-camera prepass build their pipeline
/// layouts against an empty material bind group at slot 3 (Bevy's
/// `is_depth_only_opaque_prepass` returns `true` for any opaque material
/// in those phases), but the meadow vertex shader reads `variant_params`
/// at `@group(MATERIAL_BIND_GROUP) @binding(100)` to compute every blade's
/// height / lod / heightfield-extents — so pipeline creation fails with
/// `Shader global ResourceBinding ... not available in the pipeline
/// layout`. Both meshes render in the main camera's prepass (the tuft for
/// depth + motion vectors), so both need the bit.
///
/// Bevy includes the `PREPASS_READS_MATERIAL` mesh-key bit *for* materials
/// whose vertex shader needs material data (it's listed in
/// `ALL_PREPASS_BITS`, so setting it makes `is_depth_only_opaque_prepass`
/// fall through to the full-material-layout branch in both
/// `PrepassPipeline::specialize` and `specialize_shadow_material_meshes`),
/// but doesn't expose a clean per-material setter. Both consumers OR
/// `mesh.key_bits` into the final `mesh_key`, so setting the bit on the
/// mesh propagates to both paths and the meadow renders correctly in
/// shadows even when ray tracing is off (RT-on hides the issue because
/// raytraced-lighting setups typically zero `shadow_maps_enabled` on
/// directional lights).
fn force_meadow_mesh_prepass_reads_material(
    mesh_ids: Res<RenderMeadowMeshIds>,
    mut render_meshes: ResMut<RenderAssets<RenderMesh>>,
) {
    let bit = MeshPipelineKey::PREPASS_READS_MATERIAL.bits();
    let mut changed = false;
    for asset_id in mesh_ids.lod.into_iter().flatten() {
        // `bypass_change_detection` for the read so the steady-state
        // early-return doesn't trip `RenderAssets<RenderMesh>::is_changed`
        // every frame on every consumer.
        let Some(render_mesh) = render_meshes.bypass_change_detection().get_mut(asset_id) else {
            continue;
        };
        let bits = render_mesh.key_bits.bits();
        if bits & bit != 0 {
            continue;
        }
        render_mesh.key_bits = BaseMeshPipelineKey::from_bits_retain(bits | bit);
        changed = true;
    }
    if changed {
        render_meshes.set_changed();
    }
}

/// Hijack each meadow `PreparedMaterial`'s draw-function map: replace
/// the default `DrawMaterial` / `DrawPrepass` ids with our parallel
/// `DrawMeadow*` chain ids. Runs in `RenderSystems::PrepareAssets`,
/// which lands after `prepare_assets::<MeshMaterial3d<MeadowMaterial>>`
/// and before `queue_material_meshes` reads the ids into
/// `Opaque3dBatchSetKey` / phase bins.
///
/// `MaterialProperties` is non-`Clone`, so we use `Arc::get_mut` and
/// rely on the asset map being the unique owner at this point in the
/// schedule — `specialize_material_meshes` (which clones the Arc)
/// runs in the *later* `RenderSystems::Specialize` set. If a future
/// schedule reorder ever shares the Arc here, `get_mut` returns
/// `None` and the override is silently skipped — visually
/// indistinguishable from "stock chain ran" for one frame.
fn override_meadow_draw_functions(
    mut materials: ResMut<ErasedRenderAssets<PreparedMaterial>>,
    meadow_ids: Res<RenderMeadowMaterialIds>,
    opaque_draw: Res<DrawFunctions<Opaque3d>>,
    deferred_draw: Res<DrawFunctions<Opaque3dDeferred>>,
    prepass_draw: Res<DrawFunctions<Opaque3dPrepass>>,
    shadow_draw: Res<DrawFunctions<Shadow>>,
) {
    if meadow_ids.ids.is_empty() {
        return;
    }
    let opaque_id = opaque_draw.read().id::<DrawMeadowOpaque>();
    let prepass_id = prepass_draw.read().id::<DrawMeadowPrepass>();
    let prepass_depth_only_id = prepass_draw.read().id::<DrawMeadowDepthOnlyPrepass>();
    let deferred_id = deferred_draw.read().id::<DrawMeadowPrepass>();
    let shadow_id = shadow_draw.read().id::<DrawMeadowPrepass>();
    let shadow_depth_only_id = shadow_draw.read().id::<DrawMeadowDepthOnlyPrepass>();

    let main_label = MainPassOpaqueDrawFunction.intern();
    let prepass_label = PrepassOpaqueDrawFunction.intern();
    let prepass_depth_only_label = PrepassOpaqueDepthOnlyDrawFunction.intern();
    let deferred_label = DeferredOpaqueDrawFunction.intern();
    let shadow_label = ShadowsDrawFunction.intern();
    let shadow_depth_only_label = ShadowsDepthOnlyDrawFunction.intern();

    for (id, prepared) in materials.iter_mut() {
        if !meadow_ids.ids.contains(&id) {
            continue;
        }
        let Some(props) = Arc::get_mut(&mut prepared.properties) else {
            // Arc was cloned elsewhere — skip; next frame the new
            // PreparedMaterial Arc will be unique again.
            continue;
        };
        for (label, draw_id) in props.draw_functions.iter_mut() {
            if *label == main_label {
                *draw_id = opaque_id;
            } else if *label == prepass_label {
                *draw_id = prepass_id;
            } else if *label == prepass_depth_only_label {
                *draw_id = prepass_depth_only_id;
            } else if *label == deferred_label {
                *draw_id = deferred_id;
            } else if *label == shadow_label {
                *draw_id = shadow_id;
            } else if *label == shadow_depth_only_label {
                *draw_id = shadow_depth_only_id;
            }
        }
    }
}

// ---------- Custom RenderCommand chains ----------

/// Mirrors `bevy_pbr::DrawMaterial`, but with `DrawMeadowPatch`
/// swapping in for `DrawMesh`.
pub type DrawMeadowOpaque = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    SetMaterialBindGroup<3>,
    DrawMeadowPatch,
);

/// Mirrors `bevy_pbr::DrawPrepass`. Used for `Opaque3dPrepass`,
/// `Opaque3dDeferred`, and main-path `Shadow` registrations.
pub type DrawMeadowPrepass = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetPrepassViewEmptyBindGroup<1>,
    SetMeshBindGroup<2>,
    SetMaterialBindGroup<3>,
    DrawMeadowPatch,
);

/// Mirrors `bevy_pbr::DrawDepthOnlyPrepass`. Used for depth-only
/// permutations of prepass / shadow.
pub type DrawMeadowDepthOnlyPrepass = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetPrepassViewEmptyBindGroup<1>,
    SetMeshBindGroup<2>,
    SetPrepassEmptyMaterialBindGroup<3>,
    DrawMeadowPatch,
);

/// Replaces `bevy_pbr::DrawMesh`'s final draw call. Resolves the driver
/// entity's variant + the current view's slot, binds the variant's blade
/// buffer at group 4, and issues one `draw_indexed_indirect` per LOD band
/// over that (view, band) region. Each band binds its own LOD mesh
/// (band 0 = blade, band 1 = tuft) and indirect record `slot*BANDS+band`,
/// whose `first_instance` is the (view, band) base offset, so
/// `@builtin(instance_index)` indexes `blades[]` directly. Shadow views
/// draw only band 0 — tufts never cast (and the compute pass leaves their
/// band-1 count at 0 anyway).
pub struct DrawMeadowPatch;

impl<P: PhaseItem> RenderCommand<P> for DrawMeadowPatch {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<MeshAllocator>,
        SRes<RenderMeadowDriver>,
        SRes<MeadowViewSlots>,
        SRes<crate::compute::MeadowGpuBuffers>,
        SRes<RenderMeadowMeshIds>,
        SRes<MeadowMeshPathActive>,
    );
    type ViewQuery = bevy::ecs::system::lifetimeless::Read<ExtractedView>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        view: ROQueryItem<'_, '_, Self::ViewQuery>,
        _item_query: Option<()>,
        (meshes, mesh_allocator, drivers, view_slots, gpu_buffers, mesh_ids, mesh_path): SystemParamItem<
            'w,
            '_,
            Self::Param,
        >,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // `into_inner()` lifts these to `&'w` so the references borrowed
        // out of the maps below outlive the command (the `pass` calls
        // require `'w`).
        let gpu_buffers = gpu_buffers.into_inner();
        let meshes = meshes.into_inner();
        let mesh_allocator = mesh_allocator.into_inner();
        let view_slots = view_slots.into_inner();
        let mesh_ids = mesh_ids.into_inner();

        let main_entity: MainEntity = item.main_entity();
        let Some(&variant) = drivers.by_entity.get(&main_entity.id()) else {
            return RenderCommandResult::Skip;
        };
        let Some(&slot) = view_slots.by_retained.get(&view.retained_view_entity) else {
            return RenderCommandResult::Skip;
        };
        let Some(vb) = gpu_buffers.by_variant.get(&variant) else {
            return RenderCommandResult::Skip;
        };

        // Shadow views cast only band 0 (near blade); tufts never cast.
        let is_shadow = view_slots
            .shared
            .views
            .get(slot as usize)
            .is_some_and(|v| (v.params.x as u32 & MEADOW_VIEW_FLAG_SHADOW) != 0);

        // When the mesh-shader path is active it draws every meadow view
        // via its own pass systems (`mesh_path.rs`); skipping here
        // removes the meadow from the stock opaque/prepass/deferred/
        // shadow phases in one place. Always false without the
        // `mesh-shaders` feature.
        if mesh_path.active {
            return RenderCommandResult::Skip;
        }

        // Group 4: the variant's blade buffer (shared by both bands). The
        // per-(view, band) base offset rides in via each indirect record's
        // `first_instance`.
        pass.set_bind_group(4, &vb.draw_bind_group, &[]);

        for band in 0..MEADOW_MAX_BANDS {
            if is_shadow && band == 1 {
                continue;
            }
            let Some(mesh_id) = mesh_ids.lod[band] else {
                continue;
            };
            let Some(gpu_mesh) = meshes.get(mesh_id) else {
                continue;
            };
            let Some(vertex_slice) = mesh_allocator.mesh_vertex_slice(&mesh_id) else {
                continue;
            };
            let RenderMeshBufferInfo::Indexed { index_format, .. } = &gpu_mesh.buffer_info else {
                continue; // Meadow always uses indexed meshes.
            };
            let Some(index_slice) = mesh_allocator.mesh_index_slice(&mesh_id) else {
                continue;
            };

            pass.set_vertex_buffer(0, vertex_slice.buffer.slice(..));
            pass.set_index_buffer(index_slice.buffer.slice(..), *index_format);
            let record = (slot as usize * MEADOW_MAX_BANDS + band) as u64;
            pass.draw_indexed_indirect(&vb.indirect, record * INDIRECT_ARGS_SIZE);
        }
        RenderCommandResult::Success
    }
}
