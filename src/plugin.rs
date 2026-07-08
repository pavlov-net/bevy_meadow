//! `MeadowPlugin` — registers the `MeadowMaterial` plugin, the
//! variant registry, the shared LOD meshes resource (blade + tuft
//! templates), and the broadcast systems that push wind direction /
//! season blend / heightfield atlas into every variant's material on
//! resource change.

use std::collections::HashMap;

use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::{NoFrustumCulling, VisibilityRange};
use bevy::ecs::component::Component;
use bevy::image::Image;
use bevy::pbr::{MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::render::batching::NoAutomaticBatching;
use bevy::shader::load_shader_library;

use crate::render::MeadowRenderDriver;

use bevy::render::storage::ShaderBuffer;

use crate::material::{
    MAX_TRUNK_DISCS_PER_PATCH, MeadowExt, MeadowMaterial, PatchData, PatchTrunkSlot, TrunkDisc,
    VariantParams,
};
use crate::mesh::{MeadowMeshes, build_blade_template_mesh, build_tuft_template_mesh};
use crate::placement::PatchOverride;

/// Per-variant scalar tuning. Defaults target a dense temperate-forest
/// meadow; other biomes override what they care about.
#[derive(Debug, Clone)]
pub struct MeadowVariant {
    pub patches_per_m2: f32,
    pub patch_radius_range: (f32, f32),
    pub min_patch_spacing: f32,
    pub patch_edge_noise_amp: f32,
    pub canopy_soft_threshold: f32,
    pub canopy_threshold: f32,
    pub blades_per_m2: f32,
    pub height_range: (f32, f32),
    pub width_range: (f32, f32),
    pub rim_falloff_fraction: f32,
    pub biome_weight_threshold: f32,
    pub wind: WindParams,
    pub palette: SeasonalPalette,
    pub lod: MeadowLodCurve,
    pub overrides: Vec<PatchOverride>,
    pub seed: u64,
    pub audio_tag: Option<&'static str>,
}

impl Default for MeadowVariant {
    fn default() -> Self {
        Self {
            patches_per_m2: 0.0008,
            patch_radius_range: (8.0, 20.0),
            min_patch_spacing: 4.0,
            patch_edge_noise_amp: 0.18,
            canopy_soft_threshold: 0.35,
            canopy_threshold: 0.6,
            blades_per_m2: 150.0,
            height_range: (0.18, 0.38),
            width_range: (0.02, 0.035),
            rim_falloff_fraction: 0.25,
            biome_weight_threshold: 0.6,
            wind: WindParams::default(),
            palette: SeasonalPalette::default(),
            lod: MeadowLodCurve::default(),
            overrides: Vec::new(),
            seed: 0x6D45_5F4D_4541_4400,
            audio_tag: None,
        }
    }
}

/// Per-variant wind parameters. Direction is *not* per-variant — it
/// lives on `WindDirection` so meadow + tree foliage agree on the
/// gust direction.
#[derive(Debug, Clone, Copy)]
pub struct WindParams {
    /// Peak XZ displacement of a blade tip in metres.
    pub amplitude: f32,
    /// Seconds per gust cycle.
    pub period: f32,
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            amplitude: 0.18,
            period: 4.5,
        }
    }
}

/// Four-corner seasonal albedo. The palette system on
/// `Changed<WorldClock>` lerps between adjacent corners and writes
/// the result into every variant's `VariantParams.season_blend`.
#[derive(Debug, Clone, Copy)]
pub struct SeasonalPalette {
    pub spring: Vec3,
    pub summer: Vec3,
    pub autumn: Vec3,
    pub winter: Vec3,
}

impl Default for SeasonalPalette {
    fn default() -> Self {
        Self {
            spring: Vec3::new(0.16, 0.30, 0.09),
            summer: Vec3::new(0.18, 0.32, 0.11),
            autumn: Vec3::new(0.36, 0.28, 0.10),
            winter: Vec3::new(0.55, 0.55, 0.55),
        }
    }
}

/// Distance LOD thresholds (metres). The per-blade hash gate in the
/// vertex shader linearly tapers blade survival probability across
/// `full_distance..max_view_distance` — at `full_distance` every
/// blade survives, at `max_view_distance` none do. Beyond
/// `max_view_distance` the patch as a whole is culled by
/// `VisibilityRange` with a small soft band so the patch fades
/// gracefully rather than popping out.
#[derive(Debug, Clone, Copy)]
pub struct MeadowLodCurve {
    /// Every blade survives at this player distance or closer.
    pub full_distance: f32,
    /// No grass survives at this distance or beyond — the far tuft band
    /// has faded to nothing here.
    pub max_view_distance: f32,
    /// Player distance beyond which a patch renders as far-LOD tufts
    /// (band 1) instead of blades (band 0). Near blades taper to nothing
    /// by here; sparse tufts ramp in.
    pub tuft_start: f32,
    /// Tuft survival fraction at `tuft_start` (densest).
    pub tuft_density_near: f32,
    /// Tuft survival fraction at `max_view_distance` (sparsest).
    pub tuft_density_far: f32,
    /// Player distance beyond which tuft height fades toward zero,
    /// reaching 0 at `max_view_distance`, so the band dissolves into the
    /// ground rather than ending at a hard edge.
    pub height_fade_start: f32,
    /// Shadow-caster density is full within this player distance, then
    /// ramps toward `shadow_far_density` at the shadow cull.
    pub shadow_full_dist: f32,
    /// Shadow-caster density at the shadow cull distance (`SHADOW_MAX_DIST`).
    pub shadow_far_density: f32,
}

impl Default for MeadowLodCurve {
    fn default() -> Self {
        Self {
            full_distance: 40.0,
            max_view_distance: 110.0,
            tuft_start: 55.0,
            tuft_density_near: 0.12,
            tuft_density_far: 0.02,
            height_fade_start: 98.0,
            shadow_full_dist: 18.0,
            shadow_far_density: 0.25,
        }
    }
}

/// Newtype id returned from `MeadowVariantRegistry::register`.
/// Stable across the variant's lifetime; consumers store it (e.g. in a
/// per-biome newtype `Resource`) so per-chunk dispatch can find the
/// variant's material.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeadowVariantId(pub u32);

/// Runtime catalogue of registered variants. One entry per variant
/// (typically one per biome). Survives world transitions; consumers
/// re-register when the world's biome makeup changes.
#[derive(Resource, Default)]
pub struct MeadowVariantRegistry {
    next_id: u32,
    entries: HashMap<MeadowVariantId, RegisteredVariant>,
}

pub struct RegisteredVariant {
    pub variant: MeadowVariant,
    pub material: Handle<MeadowMaterial>,
    /// CPU-side placement list, written by `upload_placements`.
    /// Consumers (per-chunk spawn) read this to spawn `MeadowPatch`
    /// entities; the GPU side reads the same data via the
    /// `patches` storage buffer.
    pub placements: Vec<crate::placement::PatchPlacement>,
}

impl MeadowVariantRegistry {
    /// Register a new variant. Allocates the per-variant
    /// `MeadowMaterial` and its `patches` / `trunk_slots` storage
    /// buffers.
    pub fn register(
        &mut self,
        variant: MeadowVariant,
        meadow_materials: &mut Assets<MeadowMaterial>,
        shader_buffers: &mut Assets<ShaderBuffer>,
    ) -> MeadowVariantId {
        let id = MeadowVariantId(self.next_id);
        self.next_id += 1;

        // Initial buffer size = 1 row each so the bind group is
        // well-formed before `upload_placements` runs; `set_data`
        // resizes as the placement list grows.
        let mut patches_buf = ShaderBuffer::default();
        patches_buf.set_data(vec![PatchData::default()]);
        let patches_handle = shader_buffers.add(patches_buf);

        let mut trunks_buf = ShaderBuffer::default();
        trunks_buf.set_data(vec![PatchTrunkSlot::default()]);
        let trunks_handle = shader_buffers.add(trunks_buf);

        let variant_params = variant_params_from(&variant);
        let material_handle = meadow_materials.add(MeadowMaterial {
            base: StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.85,
                double_sided: true,
                cull_mode: None,
                ..default()
            },
            extension: MeadowExt {
                variant_params,
                patches: patches_handle,
                trunk_slots: trunks_handle,
                heightfield: Handle::default(),
            },
        });

        self.entries.insert(
            id,
            RegisteredVariant {
                variant,
                material: material_handle,
                placements: Vec::new(),
            },
        );
        id
    }

    pub fn get(&self, id: MeadowVariantId) -> Option<&RegisteredVariant> {
        self.entries.get(&id)
    }

    pub fn get_mut(&mut self, id: MeadowVariantId) -> Option<&mut RegisteredVariant> {
        self.entries.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&MeadowVariantId, &RegisteredVariant)> {
        self.entries.iter()
    }
}

fn variant_params_from(variant: &MeadowVariant) -> VariantParams {
    // `wind_direction` / `wind_state` come from `VariantParams::default()`
    // so the freshly-registered variant has sane values before the
    // broadcast systems run for the first time.
    VariantParams {
        height_range: Vec4::new(variant.height_range.0, variant.height_range.1, 0.0, 0.0),
        width_range: Vec4::new(variant.width_range.0, variant.width_range.1, 0.0, 0.0),
        wind: Vec4::new(variant.wind.amplitude, variant.wind.period, 0.0, 0.0),
        palette_spring: Vec4::new(
            variant.palette.spring.x,
            variant.palette.spring.y,
            variant.palette.spring.z,
            0.0,
        ),
        palette_summer: Vec4::new(
            variant.palette.summer.x,
            variant.palette.summer.y,
            variant.palette.summer.z,
            0.0,
        ),
        palette_autumn: Vec4::new(
            variant.palette.autumn.x,
            variant.palette.autumn.y,
            variant.palette.autumn.z,
            0.0,
        ),
        palette_winter: Vec4::new(
            variant.palette.winter.x,
            variant.palette.winter.y,
            variant.palette.winter.z,
            0.0,
        ),
        season_blend: Vec4::new(0.0, 1.0, 0.0, 0.0),
        // `x = full_distance`, `y = max_view_distance`,
        // `z = rim_falloff_fraction`, `w = unused`. The shader's
        // per-blade hash gate uses these to taper density linearly
        // across `full..max_view`.
        lod: Vec4::new(
            variant.lod.full_distance,
            variant.lod.max_view_distance,
            variant.rim_falloff_fraction,
            0.0,
        ),
        density: Vec4::new(
            variant.blades_per_m2,
            variant.patch_edge_noise_amp,
            variant.lod.shadow_full_dist,
            variant.lod.shadow_far_density,
        ),
        tuft: Vec4::new(
            variant.lod.tuft_start,
            variant.lod.tuft_density_near,
            variant.lod.tuft_density_far,
            variant.lod.height_fade_start,
        ),
        // Overwritten by `broadcast_heightfield` when the consumer
        // sets `MeadowHeightfield`; the default 1×1 extent makes the
        // shader's bilerp resolve to texel (0,0) of the fallback
        // image (which Bevy fills with all-zero) so blades sit at
        // Y=0 until the real atlas is supplied.
        heightfield_extents: Vec4::new(0.0, 0.0, 1.0, 1.0),
        ..VariantParams::default()
    }
}

/// World-level R32Float heightfield atlas. The consumer (typically a
/// world/biome loader) builds an atlas covering every active chunk's
/// terrain Y values and writes this resource on world enter. The `broadcast_heightfield` system below picks up
/// the change and patches it into every registered variant's
/// material — both the `Handle<Image>` and the
/// `VariantParams.heightfield_extents`. The vertex shader maps a
/// blade's world XZ into texel coords via
/// `(world_xz - extents.xy) / (extents.zw - extents.xy) * dims`
/// and does manual bilinear filtering via four `textureLoad`s.
///
/// `Default` produces a degenerate (`min == max`) extent + empty
/// handle so the broadcast is a no-op until the consumer sets a
/// real atlas; the fallback `FallbackImage` (a 1×1 zeros texture)
/// keeps the variant's bind group well-formed in the meantime.
#[derive(Resource, Debug, Clone, Default)]
pub struct MeadowHeightfield {
    pub atlas: Handle<Image>,
    pub world_min: Vec2,
    pub world_max: Vec2,
}

/// World-level camera viewpoint for meadow LOD. Consumers write this each
/// frame from their primary camera; `broadcast_viewer` propagates both
/// fields into every variant's `VariantParams` (see those fields'
/// docstrings for the shader-side rationale).
///
/// `eye_xz` is the camera position — the omnidirectional LOD/density pivot.
/// `depth_plane` is the camera view-depth plane `(forward.xyz,
/// -dot(forward, eye))`, used for shadow cascade assignment so casters land
/// in the same cascade the receiver samples.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
pub struct MeadowViewer {
    pub eye_xz: Vec2,
    pub depth_plane: Vec4,
}

/// World-level shared wind direction. `bevy_meadow` and `bevy_tree`
/// both read from this resource so foliage agrees on the gust
/// direction. The broadcast system below copies its value into every
/// variant's `VariantParams.wind_direction` on change.
#[derive(Resource, Debug, Clone, Copy)]
pub struct WindDirection(pub Vec2);

impl Default for WindDirection {
    fn default() -> Self {
        // East-ish; the actual direction is content-tunable. A
        // unit vector keeps shader normalize() cheap and stable.
        Self(Vec2::new(0.866, 0.5))
    }
}

/// World-level shared wind dynamics. The atmosphere consumer writes
/// `speed` (multiplier on per-variant amplitude) and `gustiness`
/// (0 = steady sway, 1 = fully modulated) on weather change;
/// `crest_wavenumber` is meadow-side tuning for how far apart gust
/// crests sit along wind direction.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct MeadowWindState {
    pub speed: f32,
    pub gustiness: f32,
    pub crest_wavenumber: f32,
}

impl Default for MeadowWindState {
    fn default() -> Self {
        Self {
            speed: 1.0,
            gustiness: 0.6,
            // 0.6 rad/m → ~10 m wavelength, so a full crest fits
            // inside one meadow patch and reads as a clear band.
            crest_wavenumber: 0.6,
        }
    }
}

/// Required-components struct for a placed patch. Spawning is one
/// `commands.spawn((MeadowPatch { ... }, BelongsToWorld(world),
/// ChildOf(chunk_root)))`; the `#[require(...)]` list pulls in
/// every component the meadow render path needs.
///
/// Patch entities are gameplay/streaming-only: they carry no `Mesh3d`,
/// so they emit no phase items — the per-variant render driver (see
/// `crate::render`) issues every draw from the compute-compacted blade
/// buffer. `NoAutomaticBatching` is a harmless holdover from the earlier
/// per-patch draw model.
#[derive(Component, Clone, Copy)]
#[require(
    Transform,
    Visibility,
    VisibilityRange,
    Aabb,
    PatchAudioTag,
    NoAutomaticBatching
)]
pub struct MeadowPatch {
    pub variant: MeadowVariantId,
    /// Index of this patch in the variant's placement list — also
    /// the row this patch occupies in the material's `patches` and
    /// `trunk_slots` storage buffers. Ranges over the placement list.
    pub patch_index: u32,
    /// Number of blade instances `DrawMeadowPatch` will draw for
    /// this patch. Set at spawn from the placement; subject to the
    /// `MAX_BLADES_PER_PATCH` cap. Zero means the patch is empty
    /// (e.g. fully suppressed by overrides) — `DrawMeadowPatch`
    /// short-circuits in that case.
    pub blade_count: u32,
    pub centre: Vec2,
    pub radius: f32,
}

/// Marker emitted on patch entities so the audio system can swap
/// footstep / ambience tags when the player overlaps a patch.
/// `Some(tag)` if the variant declared one; `None` otherwise.
/// Required-component default produces an empty tag, which the
/// audio system treats as a no-op overlay.
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct PatchAudioTag(pub Option<&'static str>);

/// World-level seasonal blend state. The consumer (which owns its
/// own `WorldClock` / season concept) writes to this resource; the
/// meadow plugin's `broadcast_season` system reads it on change
/// and writes the blend into every variant's
/// `VariantParams.season_blend`.
///
/// `from_idx` / `to_idx`: 0 = spring, 1 = summer, 2 = autumn, 3 =
/// winter. `blend_t` ∈ [0, 1] from the `from` corner toward `to`.
#[derive(Resource, Debug, Clone, Copy)]
pub struct MeadowSeasonState {
    pub from_idx: u32,
    pub to_idx: u32,
    pub blend_t: f32,
}

impl Default for MeadowSeasonState {
    fn default() -> Self {
        // Summer-locked default — pleasant baseline if no consumer
        // is feeding seasonal state in.
        Self {
            from_idx: 1,
            to_idx: 1,
            blend_t: 0.0,
        }
    }
}

/// Plug everything in.
pub struct MeadowPlugin;

impl Plugin for MeadowPlugin {
    fn build(&self, app: &mut App) {
        // Shared struct/helper library imported by both the compute
        // kernel and the raster passthrough shader.
        load_shader_library!(app, "meadow_shared.wgsl");
        load_shader_library!(app, "meadow.wgsl");
        // The compute kernel is loaded by handle (not a `#define_import_path`
        // library), so embed it for `asset_server.load(embedded://…)`.
        bevy::asset::embedded_asset!(app, "meadow_compute.wgsl");

        app.add_plugins(MaterialPlugin::<MeadowMaterial>::default());
        // Custom RenderCommand chain that issues
        // `draw_indexed(0..27, 0, 0..blade_count)` per patch entity
        // — required to break out of `DrawMaterial`'s
        // one-instance-per-entity model. See `crate::render`.
        app.add_plugins(crate::render::MeadowRenderPlugin);

        app.init_resource::<MeadowVariantRegistry>();
        app.init_resource::<WindDirection>();
        app.init_resource::<MeadowWindState>();
        app.init_resource::<MeadowSeasonState>();
        app.init_resource::<MeadowHeightfield>();
        app.init_resource::<MeadowViewer>();

        app.add_systems(Startup, setup_meadow_meshes);
        app.add_systems(
            Update,
            (
                broadcast_wind_direction.run_if(resource_changed::<WindDirection>),
                broadcast_wind_state.run_if(resource_changed::<MeadowWindState>),
                broadcast_season.run_if(resource_changed::<MeadowSeasonState>),
                broadcast_heightfield.run_if(resource_changed::<MeadowHeightfield>),
                // `wind.zw` carries (current_time, previous_time);
                // motion-vector TAA reads `previous_time` to cancel
                // per-vertex sway delta, so the reupload is required
                // for correctness rather than waste.
                tick_meadow_time,
                broadcast_viewer.run_if(resource_changed::<MeadowViewer>),
                spawn_meadow_render_drivers,
            ),
        );
    }
}

fn setup_meadow_meshes(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let blade = meshes.add(build_blade_template_mesh());
    let tuft = meshes.add(build_tuft_template_mesh());
    commands.insert_resource(MeadowMeshes {
        lod_meshes: [blade, tuft],
    });
}

/// One renderable driver entity per registered variant. The patches
/// themselves no longer carry a `Mesh3d`/`MeshMaterial3d` (so they emit
/// no phase items); instead this single per-variant entity produces one
/// meadow phase item per view, and `DrawMeadowPatch` issues the
/// per-view indirect draw over the compute-compacted blade buffer. The
/// world-spanning `Aabb` + `NoFrustumCulling` guarantee it lands in
/// every view's `VisibleEntities` (main camera + every shadow cascade).
///
/// Spawned lazily (rather than in `register`, which has no `Commands`
/// or blade-mesh handle) once both the registry and the shared meshes
/// resource exist; each variant gets exactly one driver.
fn spawn_meadow_render_drivers(
    mut commands: Commands,
    registry: Res<MeadowVariantRegistry>,
    meadow_meshes: Option<Res<MeadowMeshes>>,
    existing: Query<&MeadowRenderDriver>,
) {
    let Some(meadow_meshes) = meadow_meshes else {
        return;
    };
    if !registry.is_changed() && !existing.is_empty() {
        // Steady state: nothing new to spawn.
        if existing.iter().count() == registry.iter().count() {
            return;
        }
    }
    let have: std::collections::HashSet<MeadowVariantId> =
        existing.iter().map(|d| d.variant).collect();

    for (id, entry) in registry.iter() {
        if have.contains(id) {
            continue;
        }
        commands.spawn((
            MeadowRenderDriver { variant: *id },
            Mesh3d(meadow_meshes.lod_meshes[0].clone()),
            MeshMaterial3d(entry.material.clone()),
            Transform::default(),
            Visibility::Visible,
            // World-spanning bound so the driver is visible to every view.
            Aabb {
                center: Vec3A::ZERO,
                half_extents: Vec3A::splat(1.0e6),
            },
            NoFrustumCulling,
            NoAutomaticBatching,
        ));
    }
}

/// Push `MeadowSeasonState` into every registered variant's
/// material. Runs only on `Changed<MeadowSeasonState>` — the
/// consumer's clock-bridge system writes the resource at most
/// once per virtual hour.
fn broadcast_season(
    season: Res<MeadowSeasonState>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
) {
    let blend = Vec4::new(
        season.from_idx as f32,
        season.to_idx as f32,
        season.blend_t.clamp(0.0, 1.0),
        0.0,
    );
    for (_, entry) in registry.iter() {
        // Compare before writing — a same-value `get_mut` deref still
        // fires `AssetEvent::Modified` and forces a needless re-extract.
        if let Some(mut mat) = materials.get_mut(&entry.material)
            && mat.extension.variant_params.season_blend != blend
        {
            mat.extension.variant_params.season_blend = blend;
        }
    }
}

/// Per-frame time update. The shader can't read `globals.time`
/// from a prepass-compatible pipeline (the prepass pipeline layout
/// at `bevy_pbr/src/prepass/mod.rs` doesn't include the globals
/// bind group at binding 11), so we route time through the
/// material's `wind.zw` field instead. Writes elapsed_secs +
/// previous_secs each frame on every registered variant.
fn tick_meadow_time(
    time: Res<Time>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
) {
    let now = time.elapsed_secs();
    for (_, entry) in registry.iter() {
        if let Some(mut mat) = materials.get_mut(&entry.material) {
            let prev = mat.extension.variant_params.wind.z;
            mat.extension.variant_params.wind.z = now;
            mat.extension.variant_params.wind.w = prev;
        }
    }
}

/// Push the `WindDirection` resource into every registered variant's
/// material. Runs only on `Changed<WindDirection>`.
fn broadcast_wind_direction(
    direction: Res<WindDirection>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
) {
    let dir = direction.0.normalize_or_zero();
    let dir = if dir.length_squared() < 0.5 {
        Vec2::new(1.0, 0.0)
    } else {
        dir
    };
    let packed = Vec4::new(dir.x, dir.y, 0.0, 0.0);
    for (_, entry) in registry.iter() {
        if let Some(mut mat) = materials.get_mut(&entry.material)
            && mat.extension.variant_params.wind_direction != packed
        {
            mat.extension.variant_params.wind_direction = packed;
        }
    }
}

fn broadcast_wind_state(
    state: Res<MeadowWindState>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
) {
    let packed = Vec4::new(
        state.speed.max(0.0),
        state.gustiness.clamp(0.0, 1.0),
        state.crest_wavenumber.max(0.0),
        0.0,
    );
    for (_, entry) in registry.iter() {
        // Compare before writing — `Assets::get_mut` fires
        // `AssetEvent::Modified` on every deref-mut; a same-value
        // write would force every consumer to re-extract for nothing.
        if let Some(mut mat) = materials.get_mut(&entry.material)
            && mat.extension.variant_params.wind_state != packed
        {
            mat.extension.variant_params.wind_state = packed;
        }
    }
}

/// Push `MeadowViewer` into every registered variant's
/// `viewer_world_xz`. Gated at the registration site by
/// `resource_changed::<MeadowViewer>`, with an inner equality guard so
/// `AssetEvent::Modified` only fires on a real change.
fn broadcast_viewer(
    viewer: Res<MeadowViewer>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
) {
    let eye = Vec4::new(viewer.eye_xz.x, viewer.eye_xz.y, 0.0, 0.0);
    let fwd = viewer.depth_plane;
    for (_, entry) in registry.iter() {
        if let Some(mut mat) = materials.get_mut(&entry.material)
            && (mat.extension.variant_params.viewer_world_xz != eye
                || mat.extension.variant_params.viewer_forward != fwd)
        {
            mat.extension.variant_params.viewer_world_xz = eye;
            mat.extension.variant_params.viewer_forward = fwd;
        }
    }
}

/// Push the `MeadowHeightfield` resource into every registered
/// variant's material. Runs only on `Changed<MeadowHeightfield>` —
/// typically once per world enter, when the consumer's chunk-loader
/// finishes building the atlas.
fn broadcast_heightfield(
    heightfield: Res<MeadowHeightfield>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
) {
    let extents = Vec4::new(
        heightfield.world_min.x,
        heightfield.world_min.y,
        heightfield.world_max.x,
        heightfield.world_max.y,
    );
    for (_, entry) in registry.iter() {
        if let Some(mut mat) = materials.get_mut(&entry.material)
            && (mat.extension.heightfield != heightfield.atlas
                || mat.extension.variant_params.heightfield_extents != extents)
        {
            mat.extension.heightfield = heightfield.atlas.clone();
            mat.extension.variant_params.heightfield_extents = extents;
        }
    }
}

/// Helper: write a placement list into a variant's `patches` storage
/// buffer. Called from the consumer's `enter_world` after
/// `place_patches` returns. The consumer is expected to spawn
/// `MeadowPatch` entities with `patch_index` matching each
/// placement's position in the list.
///
/// Storage buffer (not uniform array) — no fixed cap; the buffer
/// resizes to fit the placement list exactly.
pub fn upload_placements(
    variant_id: MeadowVariantId,
    placements: &[crate::placement::PatchPlacement],
    registry: &mut MeadowVariantRegistry,
    materials: &Assets<MeadowMaterial>,
    shader_buffers: &mut Assets<ShaderBuffer>,
) {
    let Some(entry) = registry.get_mut(variant_id) else {
        return;
    };
    // Cache the CPU-side placement list on the variant so the
    // consumer's per-chunk patch-spawn system can read placements
    // without round-tripping through the GPU storage buffer.
    entry.placements = placements.to_vec();
    let Some(mat) = materials.get(&entry.material) else {
        return;
    };
    let mut rows: Vec<PatchData> = Vec::with_capacity(placements.len());
    for p in placements {
        // Pack seed (u32) into f32-bits so the shader can `bitcast<u32>`
        // it back. This dance is needed because `PatchData` carries
        // four floats per row.
        let seed_f = f32::from_bits(p.seed);
        let flags = 0u32;
        rows.push(PatchData {
            centre_xz_radius_seed: Vec4::new(p.centre.x, p.centre.y, p.radius, seed_f),
            blade_count_noise_canopy_flags: Vec4::new(
                p.blade_count as f32,
                p.edge_noise_amp,
                p.canopy_density_at_centre,
                f32::from_bits(flags),
            ),
        });
    }
    if rows.is_empty() {
        rows.push(PatchData::default());
    }
    let Some(mut buf) = shader_buffers.get_mut(&mat.extension.patches) else {
        return;
    };
    buf.set_data(rows);
}

/// Helper: write a per-patch trunk-slot list into a variant's
/// `trunk_slots` storage buffer. The consumer's tree/trunk sync system
/// rebuilds dirty slots and calls this with the full updated list.
///
/// `slots` is sparse — only patches with at least one trunk disc are
/// included. Patches not in the list get a default (count=0) slot.
/// The output buffer is sized to `max(patch_idx) + 1` rows.
pub fn upload_trunk_slots(
    variant_id: MeadowVariantId,
    slots: &[(u32, Vec<TrunkDisc>)],
    registry: &MeadowVariantRegistry,
    materials: &Assets<MeadowMaterial>,
    shader_buffers: &mut Assets<ShaderBuffer>,
) {
    let Some(entry) = registry.get(variant_id) else {
        return;
    };
    let Some(mat) = materials.get(&entry.material) else {
        return;
    };
    // Size the output to match `entry.placements.len()` so the
    // shader's `trunk_slots[patch_idx]` is in-bounds for every
    // placed patch — even patches not in `slots` (i.e. patches
    // with no nearby trunks) need a row, default-filled with
    // `count = 0`.
    let row_count = entry.placements.len().max(1);
    let mut rows: Vec<PatchTrunkSlot> = vec![PatchTrunkSlot::default(); row_count];
    for (patch_idx, discs) in slots {
        let i = *patch_idx as usize;
        if i >= rows.len() {
            continue;
        }
        let count = discs.len().min(MAX_TRUNK_DISCS_PER_PATCH);
        let slot = &mut rows[i];
        slot.count = count as u32;
        for (j, d) in discs.iter().take(count).enumerate() {
            slot.discs[j] = *d;
        }
    }
    let Some(mut buf) = shader_buffers.get_mut(&mat.extension.trunk_slots) else {
        return;
    };
    buf.set_data(rows);
}
