//! `MeadowPlugin` — registers the `MeadowMaterial` plugin, the
//! variant registry, the shared LOD meshes resource (blade + tuft
//! templates), and the broadcast systems that push wind direction /
//! viewer / heightfield atlas into every variant's material on resource
//! change.

use std::collections::HashMap;
use std::sync::Arc;

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

use crate::compute::MeadowRaytracingConfig;
use crate::field::{MeadowCoverField, MeadowGrid};
use crate::material::{
    ClearingProfileId, ClearingProfiles, MAX_MEADOW_PALETTES, MeadowClearingProfile, MeadowExt,
    MeadowMaterial, PaletteParams, PatchData, PatchTrunkSlot, TrunkDisc, VariantParams,
};
use crate::mesh::{MeadowMeshes, build_blade_template_mesh, build_tuft_template_mesh};
use crate::placement::{PatchOverride, PatchShape};

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
    /// The palette table at registration, 1..=[`MAX_MEADOW_PALETTES`]
    /// entries. Each patch selects one by
    /// [`PatchPlacement::palette`](crate::PatchPlacement); the consumer
    /// decides what the entries mean and rewrites the table in
    /// [`MeadowPalettes`].
    pub palettes: Vec<MeadowPalette>,
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
            palettes: vec![MeadowPalette::default()],
            lod: MeadowLodCurve::default(),
            overrides: Vec::new(),
            seed: 0x6D45_5F4D_4541_4400,
            audio_tag: None,
        }
    }
}

impl MeadowVariant {
    /// The tuft survival fractions `(near, far)` the kernels apply to this
    /// variant's blades: the LOD curve's, rescaled from
    /// [`TUFT_REFERENCE_BLADES_PER_M2`] to `blades_per_m2`.
    pub fn tuft_survival(&self) -> (f32, f32) {
        let scale = TUFT_REFERENCE_BLADES_PER_M2 / self.blades_per_m2.max(1e-3);
        (
            (self.lod.tuft_density_near * scale).min(1.0),
            (self.lod.tuft_density_far * scale).min(1.0),
        )
    }

    /// Mean one-sided area of a blade at its unscaled height, m²: blade
    /// widths and heights are independent uniform draws over their ranges.
    pub fn mean_blade_area(&self) -> f32 {
        let mean = |(lo, hi): (f32, f32)| 0.5 * (lo + hi);
        BLADE_AREA_PER_WIDTH_HEIGHT * mean(self.width_range) * mean(self.height_range)
    }

    /// Width factor of a survivor of a band that keeps share `keep` of its
    /// patch's blades: the inverse of the area share the survivors draw, up
    /// to [`MeadowLodCurve::max_widen`]. A tuft draws
    /// [`TUFT_BLADE_AREA_RATIO`] blades' area. CPU twin of
    /// `meadow_shared.wesl::meadow_lod_widen`.
    pub fn lod_widen(&self, keep: f32, tuft: bool) -> f32 {
        let drawn = keep * if tuft { TUFT_BLADE_AREA_RATIO } else { 1.0 };
        (1.0 / drawn.max(1e-4)).min(self.lod.widen_cap())
    }

    /// The tuft band's widened area share of the full blade area at
    /// `tuft_start` and at `max_view_distance`, before
    /// [`Self::drawn_area_share`] caps it at 1 and fades it with the tufts'
    /// height. Between, the share is linear in distance.
    pub fn tuft_area_share(&self) -> (f32, f32) {
        let (near, far) = self.tuft_survival();
        let scale = TUFT_BLADE_AREA_RATIO * self.lod.widen_cap();
        (near * scale, far * scale)
    }

    /// Share of the full blade area the main view draws at player distance
    /// `dist`: 1 where widening keeps up with the thinning, falling past
    /// `max_widen` and through the tufts' height fade to 0 at
    /// `max_view_distance`. What the meadow leaves undrawn is the consumer's
    /// to show some other way, such as on the ground.
    pub fn drawn_area_share(&self, dist: f32) -> f32 {
        let lod = &self.lod;
        if dist < lod.tuft_start {
            let keep = lod.near_survival(dist);
            return (keep * self.lod_widen(keep, false)).min(1.0);
        }
        let (near, far) = self.tuft_area_share();
        let ramp = ((dist - lod.tuft_start) / (lod.max_view_distance - lod.tuft_start).max(1e-3))
            .clamp(0.0, 1.0);
        let fade = ((lod.max_view_distance - dist)
            / (lod.max_view_distance - lod.height_fade_start).max(1e-3))
        .clamp(0.0, 1.0);
        (near + (far - near) * ramp).min(1.0) * fade
    }
}

/// One-sided area of a blade over its width times its height: the tapered,
/// bent ribbon of `meadow_shared.wesl::local_blade_position`, mean over the
/// blade leans.
pub const BLADE_AREA_PER_WIDTH_HEIGHT: f32 = 0.655;

/// Area of a far-band tuft over that of a blade of the same width and
/// height: seven arm triangles against the bent, tapered ribbon (mean over
/// the blade leans). MUST equal `MEADOW_TUFT_BLADE_AREA` in
/// `meadow_shared.wesl`.
pub const TUFT_BLADE_AREA_RATIO: f32 = 7.03;

/// Blade density, per m², that [`MeadowLodCurve`]'s tuft survival
/// fractions are stated at.
pub const TUFT_REFERENCE_BLADES_PER_M2: f32 = 150.0;

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

/// Mean of the blade height shade, `mix(0.70, 1.00, y_norm)` in
/// `meadow_shared.wesl::meadow_blade_base_color`, over the blade. A
/// patch's mean shaded albedo is its entry's mean albedo times this.
pub const MEAN_BLADE_SHADE: f32 = 0.85;

/// Share of the dead coverage carried by dead patches when the entry has
/// them; single dead blades carry the rest.
const DEAD_PATCH_SPLIT: f32 = 0.5;

/// Widest soft dead-patch edge, in the uniform patch noise's `[0, 1]`
/// range; the shaders narrow it to twice the patch share or its complement.
const DEAD_PATCH_EDGE: f32 = 0.1;

/// Largest height variance: keeps every blade at least 10 % of its height.
/// MUST equal `MEADOW_MAX_HEIGHT_VARIANCE` in `meadow_shared.wesl`.
const MAX_HEIGHT_VARIANCE: f32 = 0.9;

/// One entry of a variant's palette table: the blade colours and the
/// variance amounts for the patches that select it. Engine-generic: the
/// consumer decides what an entry stands for and rewrites the table as its
/// own state (calendar, climate) moves.
///
/// Blades vary around the entry with zero mean (`meadow_shared.wesl`), so
/// a patch's coverage-weighted mean albedo is [`Self::mean_albedo`] and its
/// mean blade height the variant's unvaried `height_range`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeadowPalette {
    /// Live blade albedo, linear.
    pub live: Vec3,
    /// Dry or dead blade albedo, linear.
    pub dry: Vec3,
    /// Expected share of dead blades in `[0, 1]`: half in dead patches (when
    /// `dead_patch_size > 0`), the rest as single dead blades. The cover
    /// field's ground dryness scales the patch half ([`CoverTexel`](crate::CoverTexel)).
    pub dead_share: f32,
    /// Per-blade hue and value jitter: a value scale of `1 ± color_jitter`
    /// and a hue rotation of up to `color_jitter` radians.
    pub color_jitter: f32,
    /// Per-blade and per-patch height variance: blade heights scale by
    /// `1 ± height_variance`, clamped to `[0, 0.9]`.
    pub height_variance: f32,
    /// Size in metres of the dead-patch noise's base octave; 0 leaves only
    /// single dead blades.
    pub dead_patch_size: f32,
}

impl MeadowPalette {
    /// The coverage-weighted mean albedo on ground of neutral dryness:
    /// `mix(live, dry, dead_share)`.
    pub fn mean_albedo(&self) -> Vec3 {
        self.live.lerp(self.dry, self.dead_share.clamp(0.0, 1.0))
    }

    /// The mean albedo under the blade height shade ([`MEAN_BLADE_SHADE`]):
    /// the colour a patch averages to on screen.
    pub fn shaded_mean_albedo(&self) -> Vec3 {
        self.mean_albedo() * MEAN_BLADE_SHADE
    }

    /// The entry as the shaders read it. Dead patches take
    /// `DEAD_PATCH_SPLIT` of the dead share and single blades the rest, at
    /// the threshold that makes the expected dead weight equal `dead_share`
    /// on ground of neutral dryness; the shaders scale the patch share by
    /// the ground's dryness (`meadow_patch_dead`).
    pub fn params(&self) -> PaletteParams {
        let share = self.dead_share.clamp(0.0, 1.0);
        let (frequency, patch_share) = if self.dead_patch_size > 0.0 {
            (self.dead_patch_size.recip(), share * DEAD_PATCH_SPLIT)
        } else {
            (0.0, 0.0)
        };
        // Expected weight: patch + (1 - patch) * blade = share.
        let blade_share = (share - patch_share) / (1.0 - patch_share).max(1e-4);
        PaletteParams {
            live_jitter: self.live.extend(self.color_jitter.max(0.0)),
            dry_blade_share: self.dry.extend(blade_share),
            variance: Vec4::new(
                self.height_variance.clamp(0.0, MAX_HEIGHT_VARIANCE),
                frequency,
                patch_share,
                DEAD_PATCH_EDGE,
            ),
        }
    }
}

impl Default for MeadowPalette {
    /// A uniform summer green: no dead blades, no jitter, no height variance.
    fn default() -> Self {
        let live = Vec3::new(0.18, 0.32, 0.11);
        Self {
            live,
            dry: live,
            dead_share: 0.0,
            color_jitter: 0.0,
            height_variance: 0.0,
            dead_patch_size: 0.0,
        }
    }
}

/// Panics unless a palette table of `len` entries fits the shader table.
fn assert_table_len(len: usize) {
    assert!(
        (1..=MAX_MEADOW_PALETTES).contains(&len),
        "a meadow palette table holds 1..={MAX_MEADOW_PALETTES} entries, got {len}"
    );
}

/// The shader table for `palettes`: entry `i` at slot `i`, the last entry
/// repeated through the remaining slots so any index the CPU clamps into
/// the table reads a written entry.
///
/// # Panics
///
/// If `palettes` is empty or longer than [`MAX_MEADOW_PALETTES`].
pub fn palette_table(palettes: &[MeadowPalette]) -> [PaletteParams; MAX_MEADOW_PALETTES] {
    assert_table_len(palettes.len());
    let last = palettes[palettes.len() - 1];
    std::array::from_fn(|i| palettes.get(i).unwrap_or(&last).params())
}

/// The current palette table of every variant whose consumer rewrites it;
/// a variant without an entry keeps the table it registered with
/// ([`MeadowVariant::palettes`]). Separate from [`MeadowVariantRegistry`]
/// so a table write (a calendar hour, a look slider) leaves the
/// registry's dependants — patch lists, broadcasts, render drivers — idle;
/// only the variant's palette buffer and its Solari caster colours update.
#[derive(Resource, Default)]
pub struct MeadowPalettes {
    tables: HashMap<MeadowVariantId, Vec<MeadowPalette>>,
}

impl MeadowPalettes {
    /// Replace `id`'s table.
    ///
    /// # Panics
    ///
    /// If `palettes` is empty or longer than [`MAX_MEADOW_PALETTES`].
    pub fn set(&mut self, id: MeadowVariantId, palettes: &[MeadowPalette]) {
        assert_table_len(palettes.len());
        self.tables.insert(id, palettes.to_vec());
    }

    /// `id`'s current table: the last one [`Self::set`], else `variant`'s
    /// registered table.
    pub fn current<'a>(
        &'a self,
        id: MeadowVariantId,
        variant: &'a MeadowVariant,
    ) -> &'a [MeadowPalette] {
        self.tables
            .get(&id)
            .map_or(&variant.palettes, Vec::as_slice)
    }
}

/// Distance LOD thresholds (metres). Every blade survives within
/// `full_distance`; near blades then thin to `1 / max_widen` by
/// `tuft_start`, where patches switch to sparse tufts that thin out to
/// `max_view_distance`. Each survivor of a thinned band widens to keep the
/// band's blade area ([`MeadowVariant::lod_widen`]) up to `max_widen`;
/// past that cap, and through the tufts' height fade, the drawn area falls
/// ([`MeadowVariant::drawn_area_share`]). Beyond `max_view_distance` the
/// patch as a whole is culled by `VisibilityRange` with a small soft band
/// so the patch fades gracefully rather than popping out.
#[derive(Debug, Clone, Copy)]
pub struct MeadowLodCurve {
    /// Every blade survives at this player distance or closer.
    pub full_distance: f32,
    /// No grass survives at this distance or beyond — the far tuft band
    /// has faded to nothing here.
    pub max_view_distance: f32,
    /// Player distance beyond which a patch renders as far-LOD tufts
    /// (band 1) instead of blades (band 0). Near blades thin to
    /// `1 / max_widen` by here; sparse tufts take over.
    pub tuft_start: f32,
    /// Tuft survival fraction at `tuft_start` (densest), stated for a
    /// variant of [`TUFT_REFERENCE_BLADES_PER_M2`] blades/m². Tufts sample a
    /// patch's blades, so the variant scales it by that reference over its
    /// own `blades_per_m2` ([`MeadowVariant::tuft_survival`]): the far band
    /// keeps its tufts per area at any blade density.
    pub tuft_density_near: f32,
    /// Tuft survival fraction at `max_view_distance` (sparsest), stated and
    /// scaled like `tuft_density_near`.
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
    /// Largest factor a thinned blade or tuft widens by to keep its band's
    /// blade area. Near blades thin no further than its inverse.
    pub max_widen: f32,
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
            max_widen: 4.0,
        }
    }
}

impl MeadowLodCurve {
    /// `max_widen`, at least 1.
    pub fn widen_cap(&self) -> f32 {
        self.max_widen.max(1.0)
    }

    /// Near-blade survival at player distance `dist`: 1 within
    /// `full_distance`, thinning linearly to `1 / max_widen` at
    /// `tuft_start`. The main-view band-0 arm of
    /// `meadow_shared.wesl::meadow_band_threshold`.
    pub fn near_survival(&self, dist: f32) -> f32 {
        let ramp = ((dist - self.full_distance) / (self.tuft_start - self.full_distance).max(1e-3))
            .clamp(0.0, 1.0);
        1.0 + (1.0 / self.widen_cap() - 1.0) * ramp
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
    /// CPU copy of the `trunk_slots` rows, one per placement, written by
    /// `upload_trunk_slots` / `upload_trunk_rows`.
    pub trunk_slots: Arc<Vec<PatchTrunkSlot>>,
    /// The clearing-profile table the discs reference, written by
    /// `upload_clearing_profiles`, and the texture the kernels read it from
    /// (the default image, too narrow to hold a profile, until the first
    /// upload).
    pub clearing_profiles: Arc<ClearingProfiles>,
    pub clearing_profile_image: Handle<Image>,
}

impl MeadowVariantRegistry {
    /// Register a new variant. Allocates the per-variant
    /// `MeadowMaterial` and its `patches` / `trunk_slots` storage
    /// buffers.
    ///
    /// # Panics
    ///
    /// If `variant.palettes` is empty or longer than
    /// [`MAX_MEADOW_PALETTES`].
    pub fn register(
        &mut self,
        variant: MeadowVariant,
        meadow_materials: &mut Assets<MeadowMaterial>,
        shader_buffers: &mut Assets<ShaderBuffer>,
    ) -> MeadowVariantId {
        assert_table_len(variant.palettes.len());
        let id = MeadowVariantId(self.next_id);
        self.next_id += 1;

        // Initial buffer size = 1 row each so the bind group is
        // well-formed before `upload_placements` runs; the upload
        // resizes them as the placement list grows.
        let patches_buf = ShaderBuffer::new(vec![PatchData::default()], default());
        let patches_handle = shader_buffers.add(patches_buf);

        let trunks_buf = ShaderBuffer::new(vec![PatchTrunkSlot::default()], default());
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
                trunk_slots: Arc::default(),
                clearing_profiles: Arc::default(),
                clearing_profile_image: Handle::default(),
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
    let (tuft_near, tuft_far) = variant.tuft_survival();
    VariantParams {
        height_range: Vec4::new(variant.height_range.0, variant.height_range.1, 0.0, 0.0),
        width_range: Vec4::new(variant.width_range.0, variant.width_range.1, 0.0, 0.0),
        wind: Vec4::new(variant.wind.amplitude, variant.wind.period, 0.0, 0.0),
        lod: Vec4::new(
            variant.lod.full_distance,
            variant.lod.max_view_distance,
            variant.rim_falloff_fraction,
            variant.lod.widen_cap(),
        ),
        density: Vec4::new(
            variant.blades_per_m2,
            variant.patch_edge_noise_amp,
            variant.lod.shadow_full_dist,
            variant.lod.shadow_far_density,
        ),
        tuft: Vec4::new(
            variant.lod.tuft_start,
            tuft_near,
            tuft_far,
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
///
/// [`Self::set_grid`] builds the atlas from a [`MeadowGrid`] and keeps the
/// grid for raytracing caster selection, which then measures blade
/// distances in 3D; without a grid it measures them in the plane of the
/// eye.
#[derive(Resource, Debug, Clone, Default)]
pub struct MeadowHeightfield {
    pub atlas: Handle<Image>,
    pub world_min: Vec2,
    pub world_max: Vec2,
    /// The CPU copy of `atlas`, when set through [`Self::set_grid`].
    pub grid: Option<Arc<MeadowGrid<f32>>>,
}

impl MeadowHeightfield {
    /// Replace the heightfield with `grid`, uploading it as a new atlas.
    pub fn set_grid(&mut self, grid: MeadowGrid<f32>, images: &mut Assets<Image>) {
        self.atlas = images.add(grid.to_image());
        self.world_min = grid.world_min;
        self.world_max = grid.world_max;
        self.grid = Some(Arc::new(grid));
    }
}

/// World-level camera viewpoint for meadow LOD. Consumers write this each
/// frame from their primary camera; `broadcast_viewer` propagates both
/// fields into every variant's `VariantParams` (see those fields'
/// docstrings for the shader-side rationale).
///
/// `eye` is the camera position — the omnidirectional LOD/density pivot.
/// Disc patches gate LOD by its planar distance; tile blades and
/// raytracing casters by its 3D distance. `depth_plane` is the camera
/// view-depth plane `(forward.xyz, -dot(forward, eye))`, used for shadow
/// cascade assignment so casters land in the same cascade the receiver
/// samples.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
pub struct MeadowViewer {
    pub eye: Vec3,
    pub depth_plane: Vec4,
}

/// World-level sun: past their blade normals, blades shade as a canopy lit
/// by it and the sky (`meadow_shared.wesl::meadow_canopy_shade`), their mean
/// the near blades' under any sun, looking into it or away. The consumer
/// writes it as its sun and sky change; `broadcast_sun` copies it into
/// every variant.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
pub struct MeadowSun {
    /// Unit direction to the sun, or zero without one.
    pub to_sun: Vec3,
    /// The sun's beam (direct normal) illuminance over the sky's diffuse
    /// illuminance on level ground: how the canopy weighs their colours and
    /// its sheen.
    pub sun_over_sky: f32,
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

/// Plug everything in.
pub struct MeadowPlugin;

impl Plugin for MeadowPlugin {
    fn build(&self, app: &mut App) {
        // Shared struct/helper library imported by both the compute
        // kernel and the raster passthrough shader.
        load_shader_library!(app, "meadow_shared.wesl");
        load_shader_library!(app, "meadow.wesl");
        // The compute kernel is loaded by handle, so embed it for
        // `asset_server.load(embedded://…)`.
        bevy::asset::embedded_asset!(app, "meadow_compute.wesl");

        app.add_plugins(MaterialPlugin::<MeadowMaterial>::default());
        // Custom RenderCommand chain that issues
        // `draw_indexed(0..27, 0, 0..blade_count)` per patch entity
        // — required to break out of `DrawMaterial`'s
        // one-instance-per-entity model. See `crate::render`.
        app.add_plugins(crate::render::MeadowRenderPlugin);

        app.init_resource::<MeadowVariantRegistry>();
        app.init_resource::<MeadowPalettes>();
        app.init_resource::<WindDirection>();
        app.init_resource::<MeadowWindFreeze>();
        app.init_resource::<MeadowSolariGeometryNormals>();
        // Each broadcast also runs when the registry changes: a variant
        // registered after a resource's last write otherwise keeps its
        // `VariantParams` defaults until that resource next changes.
        let registered = resource_changed::<MeadowVariantRegistry>;
        app.add_systems(
            PostUpdate,
            (
                broadcast_raytracing_flags.run_if(
                    resource_changed::<MeadowSolariGeometryNormals>
                        .or_eager(resource_exists_and_changed::<MeadowRaytracingConfig>)
                        .or_eager(resource_removed::<MeadowRaytracingConfig>)
                        .or_eager(registered),
                ),
                apply_wind_freeze,
            )
                .before(bevy::asset::AssetEventSystems),
        );
        app.init_resource::<MeadowWindState>();
        app.init_resource::<MeadowHeightfield>();
        app.init_resource::<MeadowCoverField>();
        app.init_resource::<MeadowViewer>();
        app.init_resource::<MeadowSun>();

        app.add_systems(Startup, setup_meadow_meshes);
        app.add_systems(
            Update,
            (
                broadcast_wind_direction
                    .run_if(resource_changed::<WindDirection>.or_eager(registered)),
                broadcast_wind_state
                    .run_if(resource_changed::<MeadowWindState>.or_eager(registered)),
                broadcast_heightfield
                    .run_if(resource_changed::<MeadowHeightfield>.or_eager(registered)),
                broadcast_viewer.run_if(resource_changed::<MeadowViewer>.or_eager(registered)),
                broadcast_sun.run_if(resource_changed::<MeadowSun>.or_eager(registered)),
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
    let eye = Vec4::new(viewer.eye.x, viewer.eye.z, viewer.eye.y, 0.0);
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

/// Push `MeadowSun` into every registered variant's `sun`.
fn broadcast_sun(
    sun: Res<MeadowSun>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
) {
    let packed = sun.to_sun.extend(sun.sun_over_sky.max(0.0));
    for (_, entry) in registry.iter() {
        if let Some(mut mat) = materials.get_mut(&entry.material)
            && mat.extension.variant_params.sun != packed
        {
            mat.extension.variant_params.sun = packed;
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

/// `PatchData` flag bits carrying a patch's palette index (bits 8..15),
/// read back by `meadow_patch_palette` in `meadow_shared.wesl`.
const PALETTE_FLAG_SHIFT: u32 = 8;

/// `PatchData` flag bit marking a tile patch, read back by
/// `meadow_patch_is_tile` in `meadow_shared.wesl`.
const TILE_FLAG: u32 = 1 << 1;

/// The flags word for a patch selecting `palette`, clamped into the table.
fn palette_flags(palette: u32) -> u32 {
    palette.min(MAX_MEADOW_PALETTES as u32 - 1) << PALETTE_FLAG_SHIFT
}

/// A placement's `PatchData` row: a tile carries its cover ceiling in the
/// edge-noise lane and the tile flag.
fn patch_row(p: &crate::placement::PatchPlacement) -> PatchData {
    // Pack seed (u32) into f32-bits so the shader can `bitcast<u32>` it
    // back: `PatchData` carries four floats per row.
    let seed_f = f32::from_bits(p.seed);
    let (lane_y, flags) = match p.shape {
        PatchShape::Disc => (p.edge_noise_amp, palette_flags(p.palette)),
        PatchShape::Tile { cover_ceiling } => (
            cover_ceiling.clamp(1e-4, 1.0),
            palette_flags(p.palette) | TILE_FLAG,
        ),
    };
    PatchData {
        centre_xz_radius_seed: Vec4::new(p.centre.x, p.centre.y, p.radius, seed_f),
        blade_count_noise_canopy_flags: Vec4::new(
            p.blade_count as f32,
            lane_y,
            p.canopy_density_at_centre,
            f32::from_bits(flags),
        ),
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
    let mut rows: Vec<PatchData> = placements.iter().map(patch_row).collect();
    if rows.is_empty() {
        rows.push(PatchData::default());
    }
    let Some(mut buf) = shader_buffers.get_mut(&mat.extension.patches) else {
        return;
    };
    buf.clear();
    buf.extend_from_slice(&rows);
}

/// Helper: write a per-patch trunk-slot list into a variant's
/// `trunk_slots` storage buffer. The consumer's tree/trunk sync system
/// rebuilds dirty slots and calls this with the full updated list.
///
/// `slots` is sparse — only patches with at least one trunk disc are
/// included. Patches not in the list get a default (count=0) slot.
/// The output buffer holds one row per placement.
pub fn upload_trunk_slots(
    variant_id: MeadowVariantId,
    slots: &[(u32, Vec<TrunkDisc>)],
    registry: &mut MeadowVariantRegistry,
    materials: &Assets<MeadowMaterial>,
    shader_buffers: &mut Assets<ShaderBuffer>,
) {
    let Some(entry) = registry.get(variant_id) else {
        return;
    };
    // Size the output to match `entry.placements.len()` so the
    // shader's `trunk_slots[patch_idx]` is in-bounds for every
    // placed patch — even patches not in `slots` (i.e. patches
    // with no nearby trunks) need a row, default-filled with
    // `count = 0`.
    let mut rows = vec![PatchTrunkSlot::default(); entry.placements.len()];
    for (patch_idx, discs) in slots {
        if let Some(slot) = rows.get_mut(*patch_idx as usize) {
            *slot = PatchTrunkSlot::from_discs(discs);
        }
    }
    upload_trunk_rows(variant_id, rows, registry, materials, shader_buffers);
}

/// Helper: replace a variant's whole `trunk_slots` buffer with `rows`, one
/// per placement (row `i` holds patch `i`'s discs). For consumers that keep
/// the rows themselves and rewrite a few at a time.
pub fn upload_trunk_rows(
    variant_id: MeadowVariantId,
    rows: Vec<PatchTrunkSlot>,
    registry: &mut MeadowVariantRegistry,
    materials: &Assets<MeadowMaterial>,
    shader_buffers: &mut Assets<ShaderBuffer>,
) {
    let Some(entry) = registry.get_mut(variant_id) else {
        return;
    };
    let rows = if rows.is_empty() {
        vec![PatchTrunkSlot::default()]
    } else {
        rows
    };
    if let Some(mat) = materials.get(&entry.material)
        && let Some(mut buf) = shader_buffers.get_mut(&mat.extension.trunk_slots)
    {
        buf.clear();
        buf.extend_from_slice(&rows);
    }
    entry.trunk_slots = Arc::new(rows);
}

/// Helper: replace a variant's clearing-profile table with `profiles`,
/// returning their ids in order for [`TrunkDisc::with_profile`]. Ids from
/// an earlier upload are stale; rebuild the discs that hold them. Empty
/// without the variant.
///
/// # Panics
///
/// As [`ClearingProfiles::new`].
pub fn upload_clearing_profiles(
    variant_id: MeadowVariantId,
    profiles: &[MeadowClearingProfile],
    registry: &mut MeadowVariantRegistry,
    images: &mut Assets<Image>,
) -> Vec<ClearingProfileId> {
    let Some(entry) = registry.get_mut(variant_id) else {
        return Vec::new();
    };
    let (table, ids) = ClearingProfiles::new(profiles);
    entry.clearing_profile_image = images.add(table.to_image());
    entry.clearing_profiles = Arc::new(table);
    ids
}

/// Diagnostic freeze shared by visible geometry, motion vectors, and RT casters.
#[derive(Resource, Default)]
pub struct MeadowWindFreeze {
    pub enabled: bool,
}

fn apply_wind_freeze(
    freeze: Res<MeadowWindFreeze>,
    time: Res<Time>,
    direction: Res<WindDirection>,
    state: Res<MeadowWindState>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
    mut frozen: Local<Option<(Vec4, Vec4)>>,
) {
    if !freeze.enabled && frozen.is_none() {
        return;
    }
    let dir = direction.0.normalize_or_zero();
    let dir = if dir.length_squared() < 0.5 {
        Vec2::X
    } else {
        dir
    };
    let current = (
        Vec4::new(dir.x, dir.y, 0.0, 0.0),
        Vec4::new(
            state.speed.max(0.0),
            state.gustiness.clamp(0.0, 1.0),
            state.crest_wavenumber.max(0.0),
            0.0,
        ),
    );
    let packed = if freeze.enabled {
        *frozen.get_or_insert_with(|| {
            let mut snapshot = current;
            // Positive encoding distinguishes a frozen phase of zero from live.
            snapshot.1.w = time.elapsed_secs_wrapped() + 1.0;
            snapshot
        })
    } else {
        *frozen = None;
        current
    };
    for (_, entry) in registry.iter() {
        if let Some(mut mat) = materials.get_mut(&entry.material) {
            let params = &mat.extension.variant_params;
            if params.wind_direction != packed.0 || params.wind_state != packed.1 {
                mat.extension.variant_params.wind_direction = packed.0;
                mat.extension.variant_params.wind_state = packed.1;
            }
        }
    }
}

/// Enable the geometric-normal G-buffer payload (`meadow_shared.wesl`), which
/// the receiver pass of `solari::MeadowSolariPlugin` consumes. The payload
/// overwrites the AO and clearcoat bytes, so it must be disabled whenever
/// ordinary deferred lighting consumes the G-buffer.
#[derive(Resource, Default)]
pub struct MeadowSolariGeometryNormals {
    pub enabled: bool,
}

/// Mark in every variant's `width_range` whether the geometric-normal
/// payload is on (`z`, [`MeadowSolariGeometryNormals`]) and whether
/// raytraced casters shade the grass (`w`, [`MeadowRaytracingConfig`]):
/// past the casters grass shades as a canopy, across the raytraced casters'
/// fade or past the shadow maps' cut (`meadow_blade_shade`).
fn broadcast_raytracing_flags(
    geometry_normals: Res<MeadowSolariGeometryNormals>,
    raytracing: Option<Res<MeadowRaytracingConfig>>,
    registry: Res<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
) {
    let flag = |on: bool| if on { 1.0 } else { 0.0 };
    let normals = flag(geometry_normals.enabled);
    let casters = flag(raytracing.is_some_and(|config| config.enabled));
    for (_, entry) in registry.iter() {
        if let Some(mut mat) = materials.get_mut(&entry.material) {
            let width_range = mat.extension.variant_params.width_range;
            if width_range.z != normals || width_range.w != casters {
                mat.extension.variant_params.width_range.z = normals;
                mat.extension.variant_params.width_range.w = casters;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::render::render_resource::ShaderType;

    use super::*;

    /// The Rust layouts match the WGSL mirrors in `meadow_shared.wesl`, and
    /// both derivation kernels bind a table of `MAX_MEADOW_PALETTES`.
    #[test]
    fn layouts_match_the_shaders() {
        assert_eq!(PaletteParams::min_size().get(), 48);
        assert_eq!(size_of::<PaletteParams>(), 48);
        assert_eq!(VariantParams::min_size().get(), 14 * 16);
        let shared = include_str!("meadow_shared.wesl");
        assert!(
            shared.contains(&format!(
                "const MEADOW_MAX_PALETTES: u32 = {MAX_MEADOW_PALETTES}u;"
            )),
            "MEADOW_MAX_PALETTES drifted from MAX_MEADOW_PALETTES"
        );
        let table = format!("array<PaletteParams, {MAX_MEADOW_PALETTES}>");
        for (name, source) in [
            ("meadow_compute.wesl", include_str!("meadow_compute.wesl")),
            (
                "meadow_mesh_bindings.wesl",
                include_str!("meadow_mesh_bindings.wesl"),
            ),
        ] {
            assert!(source.contains(&table), "{name}'s table length drifted");
        }
    }

    #[test]
    fn palette_entry_packs_into_its_lanes() {
        let palette = MeadowPalette {
            live: Vec3::new(0.06, 0.11, 0.03),
            dry: Vec3::new(0.21, 0.17, 0.09),
            dead_share: 0.25,
            color_jitter: 0.08,
            height_variance: 1.5,
            dead_patch_size: 4.0,
        };
        let params = palette.params();
        assert_eq!(params.live_jitter, Vec4::new(0.06, 0.11, 0.03, 0.08));
        assert_eq!(params.dry_blade_share.truncate(), palette.dry);
        // Height variance clamps; the noise frequency is 1 / patch size.
        assert_eq!(params.variance.x, 0.9);
        assert_eq!(params.variance.y, 0.25);
        // Half the share in patches (edge up to 0.1), the rest in single
        // blades: the expected dead weight is the share.
        assert_eq!(params.variance.z, 0.125);
        assert_eq!(params.variance.w, DEAD_PATCH_EDGE);
        let blade_share = params.dry_blade_share.w;
        assert!((0.125 + 0.875 * blade_share - 0.25).abs() < 1e-6);
    }

    /// `meadow_patch_share`.
    fn patch_share(params: &PaletteParams, dryness: f32) -> f32 {
        (params.variance.z * 2.0 * dryness).clamp(0.0, 1.0)
    }

    /// `meadow_patch_dead`.
    fn patch_dead(params: &PaletteParams, noise: f32, share: f32) -> f32 {
        let edge = params
            .variance
            .w
            .min(2.0 * share.min(1.0 - share))
            .max(1e-4);
        if share > 0.0 {
            ((noise - (1.0 - share)) / edge + 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// `meadow_patch_noise`.
    fn patch_noise(params: &PaletteParams, root: Vec2, dryness: f32) -> f32 {
        if params.variance.y > 0.0 && patch_share(params, dryness) > 0.0 {
            crate::field::patch_noise(root * params.variance.y)
        } else {
            0.5
        }
    }

    /// `meadow_dryness_height`.
    fn dryness_height(dryness: f32) -> f32 {
        let d = dryness.clamp(0.0, 1.0) - crate::CoverTexel::NEUTRAL_DRYNESS;
        1.0 - d * if d > 0.0 { 0.4 } else { 0.2 }
    }

    /// `meadow_blade_height_scale`.
    fn height_scale(params: &PaletteParams, root: Vec2, dryness: f32, seed: u32) -> f32 {
        let blade_term = 2.0 * crate::field::hash01_u32(seed.wrapping_add(19)) - 1.0;
        let share = patch_share(params, dryness);
        let chance = patch_dead(params, patch_noise(params, root, dryness), share);
        let patch_term = (share - chance) / share.max(1.0 - share);
        dryness_height(dryness) * (1.0 + params.variance.x * 0.5 * (blade_term + patch_term))
    }

    /// Whether `meadow_blade_albedo` kills the blade with seed `seed` at
    /// `root`.
    fn blade_is_dead(params: &PaletteParams, root: Vec2, dryness: f32, seed: u32) -> bool {
        use crate::field::hash01_u32;
        let chance = patch_dead(
            params,
            patch_noise(params, root, dryness),
            patch_share(params, dryness),
        );
        hash01_u32(seed.wrapping_add(43)) < chance
            || hash01_u32(seed.wrapping_add(19)) < params.dry_blade_share.w
    }

    /// `n` blades' seeds and roots, spread over 20 km.
    fn blades(n: u32) -> impl Iterator<Item = (u32, Vec2)> {
        use crate::field::{hash_u32, hash01_u32};
        (0..n).map(|i| {
            let seed = hash_u32(i ^ 0x2545_f491);
            let unit = Vec2::new(hash01_u32(seed), hash01_u32(seed.wrapping_add(1)));
            (seed, (unit - 0.5) * 20_000.0)
        })
    }

    /// `meadow_stand_albedo`'s dead weight at a root is the chance
    /// `meadow_blade_albedo` kills a blade there: the patch's, then a single
    /// blade's; a footprint past a patch takes the patch share, the
    /// expectation of the patch chance over the uniform noise.
    #[test]
    fn stand_dead_weight_is_the_blades_expectation() {
        let shared = include_str!("meadow_shared.wesl");
        for needle in [
            "let dead = patch_dead + (1.0 - patch_dead) * palette.dry_blade_share.w;",
            "patch_dead = mix(share, meadow_patch_dead(palette, noise, share), resolved);",
        ] {
            assert!(shared.contains(needle), "`{needle}` missing");
        }
        let params = MeadowPalette {
            dead_share: 0.3,
            dead_patch_size: 6.0,
            ..MeadowPalette::default()
        }
        .params();
        let n = 100_000u32;
        for root in [
            Vec2::new(3.0, 7.0),
            Vec2::new(-40.0, 12.5),
            Vec2::new(900.0, -3.0),
        ] {
            for dryness in [0.2, 0.5, 0.9] {
                let share = patch_share(&params, dryness);
                let chance = patch_dead(&params, patch_noise(&params, root, dryness), share);
                let expected = chance + (1.0 - chance) * params.dry_blade_share.w;
                let dead = (0..n)
                    .filter(|i| {
                        blade_is_dead(&params, root, dryness, crate::field::hash_u32(i ^ 0x9e37))
                    })
                    .count() as f32
                    / n as f32;
                assert!(
                    (dead - expected).abs() < 0.01,
                    "{root} {dryness}: {dead} vs {expected}"
                );
            }
        }
        // Over many roots the patch chance averages to its share.
        let mean = blades(n)
            .map(|(_, root)| {
                patch_dead(
                    &params,
                    patch_noise(&params, root, 0.5),
                    patch_share(&params, 0.5),
                )
            })
            .sum::<f32>()
            / n as f32;
        assert!((mean - patch_share(&params, 0.5)).abs() < 0.01, "{mean}");
    }

    /// The coverage-weighted mean dead share over many blades is the dead
    /// share on neutral ground, so a meadow's mean albedo is
    /// `mix(live, dry, dead_share)`; elsewhere the patch share scales by
    /// `2 * dryness` and single blades keep theirs.
    #[test]
    fn mean_dead_weight_follows_the_share_and_the_dryness() {
        let n = 400_000u32;
        for (dead_share, size) in [
            (0.04, 6.0),
            (0.06, 6.0),
            (0.2, 8.0),
            (0.35, 6.0),
            (0.7, 4.0),
        ] {
            let params = MeadowPalette {
                dead_share,
                dead_patch_size: size,
                ..MeadowPalette::default()
            }
            .params();
            let blade = params.dry_blade_share.w;
            for dryness in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let dead = blades(n)
                    .filter(|&(seed, root)| blade_is_dead(&params, root, dryness, seed))
                    .count();
                let mean = dead as f32 / n as f32;
                let scaled = patch_share(&params, dryness);
                let expected = scaled + (1.0 - scaled) * blade;
                assert!(
                    (mean - expected).abs() < 0.002 + 0.01 * expected,
                    "share {dead_share}, dryness {dryness}: mean {mean}, expected {expected}"
                );
                if dryness == crate::CoverTexel::NEUTRAL_DRYNESS {
                    assert!((expected - dead_share).abs() < 1e-6);
                }
            }
        }
    }

    /// Blade heights average the dryness factor: the unvaried height on
    /// neutral ground, 10 % taller on the wettest and 20 % shorter on the
    /// driest. Dead-patch blades run short where the ground has patches;
    /// ground without them has no patch term.
    #[test]
    fn mean_height_follows_the_dryness() {
        let n = 200_000u32;
        let params = MeadowPalette {
            dead_share: 0.2,
            height_variance: 0.3,
            dead_patch_size: 6.0,
            ..MeadowPalette::default()
        }
        .params();
        for (dryness, factor) in [
            (0.0, 1.1),
            (0.25, 1.05),
            (0.5, 1.0),
            (0.75, 0.9),
            (1.0, 0.8),
        ] {
            assert!((dryness_height(dryness) - factor).abs() < 1e-6);
            let share = patch_share(&params, dryness);
            let (mut sum, mut dead_sum, mut dead_count) = (0.0f64, 0.0f64, 0u32);
            for (seed, root) in blades(n) {
                let h = height_scale(&params, root, dryness, seed);
                sum += f64::from(h);
                if patch_dead(&params, patch_noise(&params, root, dryness), share) >= 1.0 {
                    dead_sum += f64::from(h);
                    dead_count += 1;
                }
            }
            let mean = (sum / f64::from(n)) as f32;
            assert!(
                (mean - factor).abs() < 0.003,
                "dryness {dryness}: mean height {mean}, expected {factor}"
            );
            if dead_count > 0 {
                let dead_mean = (dead_sum / f64::from(dead_count)) as f32;
                assert!(
                    dead_mean < 0.9 * mean,
                    "dryness {dryness}: patch blades average {dead_mean}"
                );
            }
        }
        // Without a patch share only the blade term varies the height.
        for (seed, root) in blades(1000) {
            let blade_term = 2.0 * crate::field::hash01_u32(seed.wrapping_add(19)) - 1.0;
            let expected = 1.1 * (1.0 + 0.15 * blade_term);
            assert!((height_scale(&params, root, 0.0, seed) - expected).abs() < 1e-5);
        }
    }

    /// The CPU twins carry the shader's dead-patch weight and height.
    #[test]
    fn patch_twins_match_the_shader() {
        let shared = include_str!("meadow_shared.wesl");
        for needle in [
            "return clamp(palette.variance.z * 2.0 * dryness, 0.0, 1.0);",
            "let edge = max(min(palette.variance.w, 2.0 * min(share, 1.0 - share)), 1e-4);",
            "let ramp = clamp((noise - (1.0 - share)) / edge + 0.5, 0.0, 1.0);",
            "return select(0.0, ramp, share > 0.0);",
            "return meadow_patch_fbm(root_xz * palette.variance.y);",
            "if (palette.variance.y <= 0.0 || meadow_patch_share(palette, dryness) <= 0.0) {",
            "const MEADOW_MAX_DRYNESS_HEIGHT: f32 = 1.1;",
            "let d = clamp(dryness, 0.0, 1.0) - MEADOW_NEUTRAL_DRYNESS;",
            "return 1.0 - d * select(0.2, 0.4, d > 0.0);",
            "let blade_term = 2.0 * hash01_u32(seed + 19u) - 1.0;",
            "let patch_term = (share - meadow_patch_dead(palette, noise, share)) / max(share, 1.0 - share);",
            "return meadow_dryness_height(dryness) * (1.0 + palette.variance.x * 0.5 * (blade_term + patch_term));",
            "let patch_chance = meadow_patch_dead(palette, noise, meadow_patch_share(palette, dryness));",
            "let dead = hash01_u32(seed + 43u) < patch_chance || hash01_u32(seed + 19u) < palette.dry_blade_share.w;",
        ] {
            assert!(
                shared.contains(needle),
                "`{needle}` missing from meadow_shared.wesl"
            );
        }
    }

    /// The bytes the palette buffer uploads hold each lane in WGSL field
    /// order, `r, g, b` first: the shader reads a channel-swapped or
    /// shuffled table as a wrong colour, not an error.
    #[test]
    fn palette_bytes_follow_the_wgsl_lane_order() {
        let palette = MeadowPalette {
            live: Vec3::new(0.19, 0.155, 0.085),
            dry: Vec3::new(0.15, 0.128, 0.08),
            dead_share: 0.0,
            color_jitter: 0.04,
            height_variance: 0.1,
            dead_patch_size: 0.0,
        };
        let table = palette_table(&[palette]);
        let lanes: &[f32] = bytemuck::cast_slice(&table[..1]);
        assert_eq!(
            lanes[..8],
            [0.19, 0.155, 0.085, 0.04, 0.15, 0.128, 0.08, 0.0]
        );
        assert_eq!(lanes[8], 0.1);
        assert_eq!(size_of_val(&table), 48 * MAX_MEADOW_PALETTES);
    }

    /// Without patches every dead blade is single and no root crosses the
    /// patch threshold.
    #[test]
    fn palette_without_patches_kills_single_blades_only() {
        let params = MeadowPalette {
            dead_share: 0.3,
            dead_patch_size: 0.0,
            ..MeadowPalette::default()
        }
        .params();
        assert_eq!(params.variance.y, 0.0);
        assert_eq!(params.variance.z, 0.0);
        assert!((params.dry_blade_share.w - 0.3).abs() < 1e-6);
    }

    #[test]
    fn mean_albedo_weights_dry_by_the_dead_share() {
        let palette = MeadowPalette {
            live: Vec3::new(0.1, 0.2, 0.0),
            dry: Vec3::new(0.3, 0.2, 0.2),
            dead_share: 0.25,
            ..MeadowPalette::default()
        };
        assert!(
            palette
                .mean_albedo()
                .abs_diff_eq(Vec3::new(0.15, 0.2, 0.05), 1e-6)
        );
        assert!(
            palette
                .shaded_mean_albedo()
                .abs_diff_eq(palette.mean_albedo() * MEAN_BLADE_SHADE, 1e-6)
        );
    }

    /// Slots past the table repeat its last entry, so an index the CPU
    /// clamps into the table never reads an unwritten entry.
    #[test]
    fn palette_table_repeats_the_last_entry() {
        let a = MeadowPalette::default();
        let b = MeadowPalette {
            dead_share: 0.5,
            ..a
        };
        let table = palette_table(&[a, b]);
        assert_eq!(table[0], a.params());
        assert!(table[1..].iter().all(|p| *p == b.params()));
    }

    #[test]
    #[should_panic(expected = "1..=64 entries")]
    fn palette_table_rejects_an_empty_table() {
        palette_table(&[]);
    }

    /// A variant reads its registered table until the consumer sets one.
    #[test]
    fn current_palettes_fall_back_to_the_registered_table() {
        let variant = MeadowVariant::default();
        let id = MeadowVariantId(3);
        let mut palettes = MeadowPalettes::default();
        assert_eq!(palettes.current(id, &variant), variant.palettes.as_slice());
        let dry = MeadowPalette {
            dead_share: 1.0,
            ..MeadowPalette::default()
        };
        palettes.set(id, &[dry, dry]);
        assert_eq!(palettes.current(id, &variant), [dry, dry].as_slice());
        assert_eq!(
            palettes.current(MeadowVariantId(4), &variant),
            variant.palettes.as_slice()
        );
    }

    /// A tile row carries the tile flag beside its palette bits and its
    /// cover ceiling in the edge-noise lane; a disc row neither.
    #[test]
    fn tile_rows_pack_their_flag_and_ceiling() {
        let tile =
            crate::placement::PatchPlacement::tile(Vec2::new(4.0, -4.0), 4.0, 900, 7, 0.25, 3);
        let row = patch_row(&tile);
        let flags = row.blade_count_noise_canopy_flags.w.to_bits();
        assert_eq!(flags, TILE_FLAG | (3 << PALETTE_FLAG_SHIFT));
        assert_eq!(row.blade_count_noise_canopy_flags.y, 0.25);
        assert_eq!(row.centre_xz_radius_seed.z, 4.0);
        let disc = crate::placement::PatchPlacement {
            shape: PatchShape::Disc,
            edge_noise_amp: 0.1,
            ..tile
        };
        let row = patch_row(&disc);
        assert_eq!(
            row.blade_count_noise_canopy_flags.w.to_bits() & TILE_FLAG,
            0
        );
        assert_eq!(row.blade_count_noise_canopy_flags.y, 0.1);
        assert!((tile.outer_radius() - 4.0 * std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    /// A variant's clearing profiles get their own texture, reused by later
    /// uploads, and leave the default image alone.
    #[test]
    fn clearing_profiles_upload_to_their_own_image() {
        let mut images = Assets::<Image>::default();
        let mut registry = MeadowVariantRegistry::default();
        let id = registry.register(
            MeadowVariant::default(),
            &mut Assets::default(),
            &mut Assets::default(),
        );
        let profile = MeadowClearingProfile {
            radius: 1.0,
            rings: 2,
            sectors: 3,
            heights: vec![0.5; 6],
        };
        let ids =
            upload_clearing_profiles(id, &[profile.clone(), profile], &mut registry, &mut images);
        assert_eq!(ids.len(), 2);
        let handle = &registry.get(id).unwrap().clearing_profile_image;
        assert_ne!(*handle, Handle::default());
        assert_eq!(images.get(handle).unwrap().height(), 2);
    }

    /// The shaders carry their own copies of the tile flag, the height
    /// variance clamp and the clearing-disc constants, and read discs at
    /// their size.
    #[test]
    fn disc_and_tile_constants_match_the_shaders() {
        use crate::material::{FLAT_HEIGHT_CAP, MAX_DISC_EDGE_NOISE};
        use crate::rt_selection::BLADE_GROUND_LIFT;
        let shared = include_str!("meadow_shared.wesl");
        for line in [
            format!("const MEADOW_PATCH_FLAG_TILE: u32 = {TILE_FLAG}u;"),
            format!("const MEADOW_MAX_HEIGHT_VARIANCE: f32 = {MAX_HEIGHT_VARIANCE:?};"),
            format!("const MEADOW_BLADE_GROUND_LIFT: f32 = {BLADE_GROUND_LIFT:?};"),
            format!("const MEADOW_FLAT_HEIGHT_CAP: f32 = {FLAT_HEIGHT_CAP:?};"),
            format!("const MEADOW_MAX_DISC_EDGE_NOISE: f32 = {MAX_DISC_EDGE_NOISE:?};"),
        ] {
            assert!(
                shared.contains(&line),
                "`{line}` missing from meadow_shared.wesl"
            );
        }
        assert_eq!(size_of::<TrunkDisc>(), 48);
        assert_eq!(TrunkDisc::min_size().get(), 48);
        assert_eq!(
            size_of::<PatchTrunkSlot>(),
            16 + 48 * crate::MAX_TRUNK_DISCS_PER_PATCH
        );
    }

    /// The palette index sits in bits 8..15, clear of the audio bit, and is
    /// clamped into the table.
    #[test]
    fn palette_index_packs_into_the_flag_bits() {
        assert_eq!(palette_flags(0), 0);
        assert_eq!(palette_flags(5), 5 << 8);
        assert_eq!(palette_flags(1000), (MAX_MEADOW_PALETTES as u32 - 1) << 8);
        assert_eq!(palette_flags(MAX_MEADOW_PALETTES as u32 - 1) & 1, 0);
    }

    /// Thinning keeps the drawn blade area through the near band and into
    /// the tufts while widening keeps up, and the drawn share only falls
    /// with distance.
    #[test]
    fn lod_keeps_the_blade_area() {
        let variant = MeadowVariant {
            blades_per_m2: 1500.0,
            lod: MeadowLodCurve {
                full_distance: 40.0,
                tuft_start: 55.0,
                max_view_distance: 250.0,
                height_fade_start: 190.0,
                ..MeadowLodCurve::default()
            },
            ..MeadowVariant::default()
        };
        let lod = &variant.lod;
        for d in [0.0, 40.0, 47.5, 54.9] {
            let keep = lod.near_survival(d);
            assert!(
                (keep * variant.lod_widen(keep, false) - 1.0).abs() < 1e-5,
                "{d}"
            );
            assert_eq!(variant.drawn_area_share(d), 1.0);
        }
        assert!((lod.near_survival(55.0) - 1.0 / lod.max_widen).abs() < 1e-6);
        // A tuft at its band's keep share draws its blades' area, capped.
        let (tuft_near, _) = variant.tuft_survival();
        let tuft = tuft_near * TUFT_BLADE_AREA_RATIO * variant.lod_widen(tuft_near, true);
        assert!((tuft - variant.tuft_area_share().0.min(1.0)).abs() < 1e-5);
        let mut last = 1.0;
        for i in 0..=260 {
            let share = variant.drawn_area_share(i as f32);
            assert!(share <= last + 1e-6, "{i}: {share} > {last}");
            last = share;
        }
        assert_eq!(variant.drawn_area_share(250.0), 0.0);
        // No widening keeps every near blade and draws them all.
        let plain = MeadowVariant {
            lod: MeadowLodCurve {
                max_widen: 1.0,
                ..variant.lod
            },
            ..variant.clone()
        };
        assert_eq!(plain.lod.near_survival(54.0), 1.0);
        assert_eq!(plain.lod_widen(0.25, false), 1.0);
    }
}
