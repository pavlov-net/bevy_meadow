//! `MeadowMaterial` — `ExtendedMaterial<StandardMaterial, MeadowExt>`,
//! one asset per registered variant. Carries the per-variant scalar
//! params (wind, palette, LOD curve), the dynamic-sized `PatchData`
//! and `PatchTrunkSlot` storage buffers, and the world heightfield
//! atlas the vertex shader samples per blade.
//!
//! Going through `MaterialExtension` instead of a pure custom
//! pipeline lets the fragment write via
//! `pbr_input_from_standard_material` + `deferred_output`, picking up
//! PBR, shadow receiving, and the engine's GI integrations for free.

use bevy::asset::Asset;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use crate::compute::meadow_draw_bind_group_layout;

pub(crate) const MEADOW_SHADER: &str = "embedded://bevy_meadow/meadow.wesl";

/// Maximum trunk discs collapsed per patch. The plan estimates ~10
/// per patch given typical tree spacing; 16 leaves headroom. This is
/// per-slot, not a global cap — each patch's slot is fixed-size, but
/// the variant's `trunk_slots` storage buffer holds an arbitrary
/// number of slots (one per patch).
pub const MAX_TRUNK_DISCS_PER_PATCH: usize = 16;

pub type MeadowMaterial = ExtendedMaterial<StandardMaterial, MeadowExt>;

/// Capacity of a variant's palette table: 64 entries × 48 B in its own
/// 3 KiB storage buffer, bound only by the kernels that derive blades.
/// MUST equal `MEADOW_MAX_PALETTES` in `meadow_shared.wesl`; a patch's
/// palette index travels in 8 bits of its flags lane, so this stays at or
/// below 256.
pub const MAX_MEADOW_PALETTES: usize = 64;

/// One palette-table entry as the shaders read it (`PaletteParams` in
/// `meadow_shared.wesl`, 3 × vec4 = 48 B). Packed by
/// [`MeadowPalette::params`](crate::MeadowPalette::params) with the
/// amounts clamped and the per-blade thresholds precomputed.
#[repr(C)]
#[derive(ShaderType, Debug, Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PaletteParams {
    /// `rgb` = live blade albedo (linear), `w` = per-blade colour jitter.
    pub live_jitter: Vec4,
    /// `rgb` = dry blade albedo (linear), `w` = single-dead-blade
    /// threshold on the blade's random.
    pub dry_blade_share: Vec4,
    /// `x` = height variance, `y` = dead-patch noise frequency (`1 /
    /// dead_patch_size`, 0 = no patches), `z` = dead-patch share at neutral
    /// ground dryness, `w` = widest soft patch edge in the uniform patch
    /// noise.
    pub variance: Vec4,
}

/// Per-variant scalar parameters. Bound at uniform slot 100. All the
/// per-frame churn (wind direction, viewer) lands here and only here, so
/// per-frame uploads of a variant material are a single uniform write
/// rather than a full bind-group rebuild.
#[derive(ShaderType, Debug, Clone, Copy)]
pub struct VariantParams {
    /// Min/max blade height in metres, packed `(min, max, _, _)`.
    pub height_range: Vec4,
    /// Min/max blade width in metres, packed `(min, max,
    /// solari_geometry_normal, raytraced_casters)`. `z` enables a
    /// Solari-only G-buffer extension; zero for vanilla deferred. `w` = 1
    /// where raytraced casters shade the grass, from
    /// `MeadowRaytracingConfig`.
    pub width_range: Vec4,
    /// `x = amplitude` (peak XZ displacement of a blade tip in
    /// metres), `y = period` (seconds per gust cycle),
    /// `z = current_time`, `w = previous_time` (for motion-vector
    /// TAA; both in `globals`' wrapped seconds). The time fields are
    /// stamped render-side by `prepare_meadow_variant_params` for the
    /// compute / mesh / RT kernels; the raster VS reads the same clock
    /// straight from the `globals` binding. They stay zero on the
    /// asset — a per-frame asset write would fire
    /// `AssetEvent::Modified` and re-extract every variant material.
    pub wind: Vec4,
    /// Globally-shared wind direction, normalised. The
    /// `WindDirection` resource broadcasts to every variant on
    /// resource change.
    pub wind_direction: Vec4,
    /// Wind dynamics packed `(speed_mul, gustiness, crest_wavenumber, freeze_time_plus_one)`.
    /// A positive `w` freezes shared wind evaluation at `w - 1` seconds.
    /// Broadcast from the `MeadowWindState` resource; see that type's
    /// doc for per-field semantics.
    pub wind_state: Vec4,
    /// `x = full_distance` (every blade survives at this range or
    /// closer), `y = max_view_distance` (no blade survives at this
    /// range or beyond — `VisibilityRange` then culls the patch
    /// entity itself in a small soft band), `z = rim_falloff_fraction`
    /// (smoothstep on the outer edge of the noisy patch radius),
    /// `w = max_widen` (the widening cap thinned blades keep their area
    /// by, at least 1; [`MeadowLodCurve`](crate::MeadowLodCurve)).
    pub lod: Vec4,
    /// `x = blades_per_m²`, `y = patch_edge_noise_amp`,
    /// `z = shadow_full_dist` (shadow-caster density is full within this
    /// player distance), `w = shadow_far_density` (caster density at the
    /// shadow cull, ramped down from 1.0). The shadow density ramp lives
    /// here so it hot-reloads without a Rust rebuild.
    pub density: Vec4,
    /// World-space extent of the bound `heightfield` texture: `xy =
    /// world_min_xz`, `zw = world_max_xz`. The shader maps a blade's
    /// world XZ into texel coords via `(world_xz - xy) / (zw - xy) *
    /// dims`. Set by the consumer's atlas builder via
    /// `MeadowHeightfield::set` on world enter.
    pub heightfield_extents: Vec4,
    /// Player camera world position, packed `(x, z, y, _)`. Broadcast
    /// from `MeadowViewer` every frame. The shader's per-blade
    /// distance LOD gate reads this rather than `view.world_position`,
    /// because `view.world_position` is the projection origin of the
    /// current view — for shadow cascade passes that's the
    /// directional light, not the player, so reading it there causes
    /// every blade to survive the LOD gate out to the cascade limit
    /// (~12× the visible blade count, dominating cascade depth time).
    /// A uniform decouples the LOD math from which view is rendering.
    pub viewer_world_xz: Vec4,
    /// Far-LOD tuft band, packed `(tuft_start, density_near, density_far,
    /// height_fade_start)`. Patches whose player distance exceeds
    /// `tuft_start` render as sparse fanned tufts (band 1) instead of
    /// blades; per-tuft survival ramps `density_near → density_far` across
    /// `tuft_start..max_view_distance` (`lod.y`), and tuft height fades to
    /// zero over `height_fade_start..max_view_distance` so the band
    /// dissolves into the ground rather than ending at a hard edge.
    pub tuft: Vec4,
    /// Main camera view-depth plane, packed `(forward.xyz, -dot(forward,
    /// eye))`. A world point `P`'s signed distance from the camera along its
    /// forward axis is `dot(forward, P) + w`. Shadow cascade assignment uses
    /// this (broadcast from `MeadowViewer`, like `viewer_world_xz`) so a
    /// caster lands in the same cascade the receiver under it samples — Bevy
    /// picks a receiver's cascade by view-space depth, so clipping casters by
    /// planar distance instead makes the two drift apart as the camera
    /// orbits and pops near shadows on and off.
    pub viewer_forward: Vec4,
    /// World-space extent of the cover field atlas: `xy = world_min_xz`,
    /// `zw = world_max_xz`. Filled from [`MeadowCoverField`](crate::MeadowCoverField)
    /// for the deriving kernels.
    pub cover_extents: Vec4,
    /// Cover survival, packed `(1 / noise_scale_m, clumpiness,
    /// max_height_scale, _)` (see [`CoverSurvival`](crate::CoverSurvival)).
    pub cover: Vec4,
    /// `xyz` = unit direction to the sun, zero without one; `w` = its beam
    /// illuminance over the sky's on level ground ([`MeadowSun`](crate::MeadowSun)).
    /// Far grass shades as a canopy lit from it
    /// (`meadow_shared.wesl::meadow_blade_shade`).
    pub sun: Vec4,
}

impl Default for VariantParams {
    fn default() -> Self {
        let (cover_extents, cover) = crate::field::MeadowCoverField::default().params();
        Self {
            height_range: Vec4::new(0.18, 0.38, 0.0, 0.0),
            width_range: Vec4::new(0.04, 0.07, 0.0, 0.0),
            wind: Vec4::new(0.18, 4.5, 0.0, 0.0),
            wind_direction: Vec4::new(1.0, 0.0, 0.0, 0.0),
            wind_state: Vec4::new(1.0, 0.6, 0.6, 0.0),
            lod: Vec4::new(40.0, 110.0, 0.25, 1.0),
            density: Vec4::new(150.0, 0.18, 18.0, 0.25),
            // Default to a 1×1 metre extent at the origin — degenerate
            // but well-formed; the shader's bilerp short-circuits when
            // the bound heightfield is the 1×1 fallback so blades
            // ride at Y=0 until the consumer sets a real atlas.
            heightfield_extents: Vec4::new(0.0, 0.0, 1.0, 1.0),
            // Origin — overwritten every frame by `broadcast_viewer`
            // from the consumer-written `MeadowViewer` resource.
            viewer_world_xz: Vec4::ZERO,
            // (tuft_start, density_near, density_far, height_fade_start).
            tuft: Vec4::new(55.0, 0.12, 0.02, 98.0),
            // Overwritten every frame by `broadcast_viewer`; ZERO until then
            // means view_depth = 0, so no grass shadows render pre-camera.
            viewer_forward: Vec4::ZERO,
            cover_extents,
            cover,
            sun: Vec4::ZERO,
        }
    }
}

/// Per-patch row in the variant's `patches` uniform array, indexed
/// by the patch entity's `MeshTag`. Packed for cheap shader fetch:
/// the WGSL side reads one `vec4` for centre/radius/seed and a
/// second `vec4` for blade_count/edge_noise_amp/canopy_density/flags.
#[repr(C)]
#[derive(ShaderType, Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PatchData {
    /// `xy = world centre`, `z = radius (metres)`, `w = seed (f32 from u32 bits)`.
    pub centre_xz_radius_seed: Vec4,
    /// `x = blade_count (f32)`, `y = edge_noise_amp` (a tile's cover
    /// ceiling), `z = canopy_density_at_centre`, `w = flags` (u32 bits:
    /// bit 0 = audio_enabled, bit 1 = tile, bits 8..15 = palette index).
    pub blade_count_noise_canopy_flags: Vec4,
}

/// A clearing-profile sample above any blade: it caps nothing.
pub const NO_HEIGHT_CAP: f32 = 1e30;

/// Height, metres, at or below which a clearing profile holds grass flat:
/// within a profile cell whose four samples are at or below it the disc
/// clears the grass instead of keeping invisible blades. MUST equal
/// `MEADOW_FLAT_HEIGHT_CAP` in `meadow_shared.wesl`.
pub const FLAT_HEIGHT_CAP: f32 = 1e-3;

/// Largest [`TrunkDisc::with_edge_noise`] and
/// [`TrunkDisc::with_profile_noise`] amount.
pub const MAX_DISC_EDGE_NOISE: f32 = 0.5;

/// Most texels in a row of a variant's clearing-profile table, and most
/// rows: WebGPU's minimum texture size. A profile holds at most this many
/// samples less its two-texel header.
pub const CLEARING_PROFILE_TABLE_SIZE: usize = 8192;

/// Grass height caps on a polar grid about a clearing disc's centre, such
/// as the underside of a shrub's foliage. Upload a variant's profiles with
/// [`upload_clearing_profiles`](crate::upload_clearing_profiles) and lay one
/// about a disc with [`TrunkDisc::with_profile`].
///
/// Ring `i` lies `i / (rings - 1)` of `radius` from the centre and sector
/// `j` at `j / sectors` of a turn from the profile's local +x toward its
/// local +z. A sample is the height above the disc's profile origin that
/// grass tops may reach there; a blade's cap interpolates bilinearly between
/// the four samples around it, the angle wrapping, less the blade's own
/// ground above the origin, and rises to the blade's height across one ring
/// step past the outer ring; beyond that nothing is capped. A sample at or
/// above the blade (such as [`NO_HEIGHT_CAP`]) leaves it its height; in a
/// cell whose four samples lie at or below [`FLAT_HEIGHT_CAP`] above the
/// blade's ground the disc clears the grass.
#[derive(Clone, Debug, PartialEq)]
pub struct MeadowClearingProfile {
    /// Distance of the outer ring from the centre, in the profile's units.
    pub radius: f32,
    /// At least 2.
    pub rings: u32,
    /// At least 1; 1 makes the profile radial.
    pub sectors: u32,
    /// `heights[ring * sectors + sector]`, in the profile's units.
    pub heights: Vec<f32>,
}

/// A profile in a variant's clearing-profile table, from
/// [`upload_clearing_profiles`](crate::upload_clearing_profiles); stale once
/// the variant's table is uploaded again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearingProfileId {
    row: u32,
    /// Where the profile has relaxed to full height: one ring step past its
    /// outer ring.
    relaxed: f32,
}

/// A variant's clearing profiles, one per row of an `R32Float` table:
/// texel 0 holds the row's ring count, texel 1 its sector count, then its
/// samples ring by ring.
#[derive(Clone, Debug, PartialEq)]
pub struct ClearingProfiles {
    width: u32,
    texels: Vec<f32>,
}

impl Default for ClearingProfiles {
    /// No profiles: one row whose zero ring count reads as none.
    fn default() -> Self {
        Self {
            width: 1,
            texels: vec![0.0],
        }
    }
}

impl ClearingProfiles {
    /// The table of `profiles`, one per row in order, and their ids.
    ///
    /// # Panics
    ///
    /// If a profile has fewer than 2 rings or 1 sector, a sample count
    /// other than `rings * sectors` or more samples than
    /// [`CLEARING_PROFILE_TABLE_SIZE`] allows, or `profiles` holds more than
    /// [`CLEARING_PROFILE_TABLE_SIZE`].
    pub fn new(profiles: &[MeadowClearingProfile]) -> (Self, Vec<ClearingProfileId>) {
        assert!(
            profiles.len() <= CLEARING_PROFILE_TABLE_SIZE,
            "{} clearing profiles, at most {CLEARING_PROFILE_TABLE_SIZE}",
            profiles.len()
        );
        if profiles.is_empty() {
            return (Self::default(), Vec::new());
        }
        for p in profiles {
            assert!(
                p.rings >= 2 && p.sectors >= 1,
                "a clearing profile needs 2 rings and 1 sector"
            );
            assert_eq!(
                p.heights.len(),
                p.rings as usize * p.sectors as usize,
                "a clearing profile holds rings * sectors samples"
            );
            assert!(
                p.heights.len() + 2 <= CLEARING_PROFILE_TABLE_SIZE,
                "a clearing profile holds at most {} samples",
                CLEARING_PROFILE_TABLE_SIZE - 2
            );
        }
        let width = 2 + profiles.iter().map(|p| p.heights.len()).max().unwrap_or(0);
        let mut texels = vec![0.0; width * profiles.len()];
        let ids = profiles
            .iter()
            .zip(texels.chunks_exact_mut(width))
            .enumerate()
            .map(|(row, (p, texels))| {
                texels[0] = p.rings as f32;
                texels[1] = p.sectors as f32;
                texels[2..2 + p.heights.len()].copy_from_slice(&p.heights);
                ClearingProfileId {
                    row: row as u32,
                    relaxed: p.radius.max(0.0) * p.rings as f32 / (p.rings - 1) as f32,
                }
            })
            .collect();
        (
            Self {
                width: width as u32,
                texels,
            },
            ids,
        )
    }

    /// The table as the kernels read it.
    pub(crate) fn to_image(&self) -> Image {
        crate::field::grid_image(
            UVec2::new(self.width, self.rows()),
            bevy::render::render_resource::TextureFormat::R32Float,
            bytemuck::cast_slice(&self.texels),
        )
    }

    fn rows(&self) -> u32 {
        self.texels.len() as u32 / self.width
    }

    fn texel(&self, row: u32, x: u32) -> f32 {
        self.texels[(row * self.width + x) as usize]
    }

    /// `meadow_clearing_profile`: the height of a blade `height` tall under
    /// row `row` at `u` of the distance where the profile has relaxed and
    /// `turns` of a turn from its local +x, the row's samples times
    /// `scale_lift.x` plus `scale_lift.y` (the profile origin above the
    /// blade's ground), and whether the row holds grass flat there. A row
    /// the table lacks or cannot hold caps nothing.
    fn cap(&self, row: u32, u: f32, turns: f32, scale_lift: Vec2, height: f32) -> (f32, bool) {
        if row >= self.rows() || self.width < 2 {
            return (height, false);
        }
        let rings = self.texel(row, 0) as u32;
        let sectors = self.texel(row, 1) as u32;
        if rings < 2 || sectors < 1 || 2 + rings * sectors > self.width {
            return (height, false);
        }
        let last = rings - 1;
        let q = (u * rings as f32).clamp(0.0, rings as f32);
        let i = (q as u32).min(last);
        let f = q - i as f32;
        let s = (turns - turns.floor()) * sectors as f32;
        let j0 = (s as u32).min(sectors - 1);
        let g = s - j0 as f32;
        let j1 = (j0 + 1) % sectors;
        let sample = |ring: u32, j: u32| {
            self.texel(row, 2 + ring * sectors + j) * scale_lift.x + scale_lift.y
        };
        let (h00, h01) = (sample(i, j0), sample(i, j1));
        let blend =
            |a: f32, b: f32| a.max(0.0).min(height) * (1.0 - g) + b.max(0.0).min(height) * g;
        let mut next = height;
        let mut flat = false;
        if i < last {
            let (h10, h11) = (sample(i + 1, j0), sample(i + 1, j1));
            next = blend(h10, h11);
            flat = h00.max(h01).max(h10.max(h11)) <= FLAT_HEIGHT_CAP;
        }
        (blend(h00, h01) * (1.0 - f) + next * f, flat)
    }
}

/// One clearing disc in a patch's disc list.
///
/// The disc clears grass from its core, letting it back in across its fade
/// band. On a tile patch it scales the cover density, so the band thins
/// blades through the cover's own survival noise; on a disc patch blades
/// collapse toward their root. A clearing profile (`with_profile`) caps
/// grass height about the centre by direction and distance, such as under
/// a shrub's skirt. Edge noise (`with_edge_noise`, `with_profile_noise`)
/// wobbles each outline so neither reads as a circle. `meadow_disc_effect`
/// in `meadow_shared.wesl` evaluates a disc; [`TrunkDisc::effect`] mirrors
/// it. The builders keep `noise_profile.w`, the disc's reach, current.
#[repr(C)]
#[derive(ShaderType, Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TrunkDisc {
    /// `xy = world centre`, `z = core radius`, `w = fade band`.
    pub center_radius_fade: Vec4,
    /// `x` = turns from world +X toward world +Z of the profile's local +x,
    /// `y` = world height of its origin, `z` = the radius in metres where it
    /// has relaxed to full height (0 for no profile), `w` = factor on its
    /// samples.
    pub profile_frame: Vec4,
    /// `x` = the clearing's edge noise amount, `y` = the profile's, `z` =
    /// the profile's row in the variant's table, `w` = [`Self::outer_radius`].
    pub noise_profile: Vec4,
}

impl TrunkDisc {
    /// A disc about `centre` with a `radius` core and a `fade` band, capping
    /// no height.
    pub fn new(centre: Vec2, radius: f32, fade: f32) -> Self {
        Self {
            center_radius_fade: Vec4::new(centre.x, centre.y, radius, fade),
            profile_frame: Vec4::new(0.0, 0.0, 0.0, 1.0),
            noise_profile: Vec4::ZERO,
        }
        .with_reach()
    }

    /// The disc, capping grass height by clearing profile `profile` laid
    /// about its centre: its samples heights above world height `origin_y`
    /// (such as the ground at a shrub's stem) times `height_scale` (at least
    /// 1e-6), its local +x along world XZ direction `x_axis` and its local
    /// +z a quarter turn on (where a rotation about +Y carries +z), its radii
    /// times `radius_scale`.
    pub fn with_profile(
        mut self,
        profile: ClearingProfileId,
        origin_y: f32,
        x_axis: Vec2,
        radius_scale: f32,
        height_scale: f32,
    ) -> Self {
        let axis = x_axis.try_normalize().unwrap_or(Vec2::X);
        self.profile_frame = Vec4::new(
            axis.y.atan2(axis.x) / std::f32::consts::TAU,
            origin_y,
            profile.relaxed * radius_scale.max(0.0),
            height_scale.max(1e-6),
        );
        self.noise_profile.z = profile.row as f32;
        self.with_reach()
    }

    /// The disc, its clearing's outline wobbling by up to `amount` of its
    /// distance from the centre (clamped to [`MAX_DISC_EDGE_NOISE`]).
    pub fn with_edge_noise(mut self, amount: f32) -> Self {
        self.noise_profile.x = amount.clamp(0.0, MAX_DISC_EDGE_NOISE);
        self.with_reach()
    }

    /// The disc, its profile's outline wobbling by up to `amount` of its
    /// distance from the centre (clamped to [`MAX_DISC_EDGE_NOISE`]).
    pub fn with_profile_noise(mut self, amount: f32) -> Self {
        self.noise_profile.y = amount.clamp(0.0, MAX_DISC_EDGE_NOISE);
        self.with_reach()
    }

    /// The disc with its reach recomputed: the farther of its clearing and
    /// its profile's relaxed end, each widened so the distance its edge
    /// noise shrinks still ends inside.
    fn with_reach(mut self) -> Self {
        let c = self.center_radius_fade;
        let n = self.noise_profile;
        let widen = |r: f32, noise: f32| r / (1.0 - noise.clamp(0.0, MAX_DISC_EDGE_NOISE));
        self.noise_profile.w =
            widen(c.z + c.w.max(1e-4), n.x).max(widen(self.profile_frame.z, n.y));
        self
    }

    /// World centre.
    pub fn centre(&self) -> Vec2 {
        self.center_radius_fade.truncate().truncate()
    }

    /// Core radius.
    pub fn radius(&self) -> f32 {
        self.center_radius_fade.z
    }

    /// The radius past which the disc has no effect.
    pub fn outer_radius(&self) -> f32 {
        self.noise_profile.w
    }

    /// The disc at `root`, its profile read from the variant's `profiles`,
    /// for a blade `height` tall growing from world height `ground_y`: the
    /// share of grass it lets stand (0 in its core, 1 past its fade, 0 where
    /// its profile holds grass flat) and the blade's height under its
    /// profile. Mirrors `meadow_disc_effect`.
    pub fn effect(
        &self,
        profiles: &ClearingProfiles,
        root: Vec2,
        ground_y: f32,
        height: f32,
    ) -> (f32, f32) {
        let c = self.center_radius_fade;
        let a = self.profile_frame;
        let n = self.noise_profile;
        let centre = self.centre();
        let offset = root - centre;
        let clear_outer = c.z + c.w.max(1e-4);
        let wavelength = (0.5 * clear_outer.max(a.z)).max(0.25);
        let noise = 2.0 * crate::field::uniform_noise(offset / wavelength + centre * 0.37) - 1.0;
        let dist = offset.length();
        let wobbled = |amount: f32| dist * (1.0 + amount.clamp(0.0, MAX_DISC_EDGE_NOISE) * noise);
        let mut stand = smoothstep(c.z, clear_outer, wobbled(n.x));
        let mut capped = height;
        if a.z > 0.0 {
            let mut turns = -a.x;
            if offset != Vec2::ZERO {
                turns += offset.y.atan2(offset.x) / std::f32::consts::TAU;
            }
            let scale_lift = Vec2::new(a.w, a.y - ground_y);
            let flat;
            (capped, flat) =
                profiles.cap(n.z as u32, wobbled(n.y) / a.z, turns, scale_lift, height);
            if flat {
                stand = 0.0;
            }
        }
        (stand, capped)
    }
}

/// WGSL `smoothstep`.
fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Default for TrunkDisc {
    fn default() -> Self {
        Self::new(Vec2::ZERO, 0.0, 0.0)
    }
}

/// Per-patch slot of trunk discs. Padded so the slot is
/// `std140`-aligned to 16 B; the explicit padding keeps the layout
/// stable across any future field reordering.
#[repr(C)]
#[derive(ShaderType, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PatchTrunkSlot {
    pub count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub discs: [TrunkDisc; MAX_TRUNK_DISCS_PER_PATCH],
}

impl PatchTrunkSlot {
    /// A slot holding the first [`MAX_TRUNK_DISCS_PER_PATCH`] of `discs`.
    pub fn from_discs(discs: &[TrunkDisc]) -> Self {
        let mut slot = Self::default();
        let count = discs.len().min(MAX_TRUNK_DISCS_PER_PATCH);
        slot.count = count as u32;
        slot.discs[..count].copy_from_slice(&discs[..count]);
        slot
    }

    /// The slot's discs.
    pub fn discs(&self) -> &[TrunkDisc] {
        &self.discs[..(self.count as usize).min(MAX_TRUNK_DISCS_PER_PATCH)]
    }
}

impl Default for PatchTrunkSlot {
    fn default() -> Self {
        Self {
            count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            discs: [TrunkDisc::default(); MAX_TRUNK_DISCS_PER_PATCH],
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Default, Debug, Clone)]
pub struct MeadowExt {
    #[uniform(100)]
    pub variant_params: VariantParams,
    /// Per-patch rows. Updated via `upload_placements` after
    /// `place_patches` returns.
    #[storage(101, read_only)]
    pub patches: Handle<ShaderBuffer>,
    /// Per-patch trunk-disc slot rows. Same dynamic-sized story as
    /// `patches`. The shader indexes `trunk_slots[patch_idx]` to
    /// fade blades inside any tree's trunk disc.
    #[storage(102, read_only)]
    pub trunk_slots: Handle<ShaderBuffer>,
    /// World-XZ → ground Y heightfield atlas, R32Float. Sampled
    /// per-blade in the vertex shader to plant blades at the actual
    /// terrain height; replaces the per-patch tilted-plane
    /// approximation that floated grass on slopes. The texture is
    /// declared `unfilterable_float` (R32Float can't be linearly
    /// sampled without the `float32-filterable` wgpu feature) and
    /// the shader does manual bilinear via `textureLoad`. Set by
    /// the consumer's atlas builder via `MeadowHeightfield`; until
    /// then, Bevy's `FallbackImage` provides a 1×1 zeros texture
    /// so the bind group is well-formed and blades ride at Y=0.
    #[texture(105, sample_type = "float", filterable = false)]
    pub heightfield: Handle<Image>,
}

impl MaterialExtension for MeadowExt {
    fn vertex_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }
    fn deferred_vertex_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }
    fn fragment_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }

    /// Inject the group-4 bind-group layout (the per-variant compacted
    /// blade buffer the passthrough VS reads) into every pipeline
    /// permutation (main / prepass / deferred / shadow). The draw bind
    /// group is built from the *same* descriptor in
    /// `compute::prepare_meadow_gpu_buffers` (the realloc branch), so
    /// layout and bind group stay compatible.
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.layout.push(meadow_draw_bind_group_layout());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-sector profile: the same heights in every direction.
    fn radial(radius: f32, heights: &[f32]) -> MeadowClearingProfile {
        MeadowClearingProfile {
            radius,
            rings: heights.len() as u32,
            sectors: 1,
            heights: heights.to_vec(),
        }
    }

    /// The disc's effect on a blade 1 m tall at 64 points around a circle
    /// of radius `r`.
    fn ring(
        disc: TrunkDisc,
        profiles: &ClearingProfiles,
        r: f32,
    ) -> impl Iterator<Item = (f32, f32)> + '_ {
        (0..64).map(move |i| {
            let a = i as f32 / 64.0 * std::f32::consts::TAU;
            disc.effect(profiles, disc.centre() + Vec2::from_angle(a) * r, 0.0, 1.0)
        })
    }

    /// How far `values` spread.
    fn spread(values: impl Iterator<Item = f32>) -> f32 {
        let (lo, hi) = values.fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        hi - lo
    }

    #[test]
    fn disc_clears_its_core() {
        let none = ClearingProfiles::default();
        let disc = TrunkDisc::new(Vec2::new(3.0, -2.0), 0.5, 1.0);
        let at = |r: f32| disc.effect(&none, disc.centre() + Vec2::new(r, 0.0), 0.0, 0.8);
        assert_eq!(at(0.25), (0.0, 0.8));
        assert!(at(1.0).0 > 0.0 && at(1.0).0 < 1.0);
        assert_eq!(at(1.6), (1.0, 0.8));
        assert!((disc.outer_radius() - 1.5).abs() < 1e-6);
    }

    /// A profile holds each ring's sample at its radius, interpolates
    /// between rings, rises from the outer ring to the blade's height one
    /// ring step past its radius and caps nothing beyond.
    #[test]
    fn profile_interpolates_its_rings() {
        let heights = [0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
        let (table, ids) =
            ClearingProfiles::new(&[radial(3.5, &heights), radial(1.0, &[NO_HEIGHT_CAP, 0.2])]);
        let disc = TrunkDisc::new(Vec2::new(-1.0, 2.0), 0.1, 0.1).with_profile(
            ids[0],
            0.0,
            Vec2::X,
            1.0,
            1.0,
        );
        let cap = |r: f32| {
            disc.effect(&table, disc.centre() + Vec2::new(0.0, r), 0.0, 0.8)
                .1
        };
        // Ring 0 is clamped to the blade's height.
        assert_eq!(cap(0.0), 0.8);
        for (i, h) in heights.iter().enumerate().skip(1) {
            assert!((cap(i as f32 * 0.5) - h).abs() < 1e-6, "ring {i}");
        }
        assert!((cap(1.25) - 0.45).abs() < 1e-6);
        assert!((cap(3.75) - 0.425).abs() < 1e-6);
        assert_eq!(cap(4.0), 0.8);
        assert_eq!(cap(4.5), 0.8);
        assert!((disc.outer_radius() - 4.0).abs() < 1e-6);
        // A sample of `NO_HEIGHT_CAP` leaves the blade its height.
        let open =
            TrunkDisc::new(Vec2::ZERO, 0.1, 0.1).with_profile(ids[1], 0.0, Vec2::X, 1.0, 1.0);
        assert_eq!(open.effect(&table, Vec2::ZERO, 0.0, 0.5).1, 0.5);
        assert!((open.effect(&table, Vec2::new(0.5, 0.0), 0.0, 0.5).1 - 0.35).abs() < 1e-6);
        // A row the table lacks caps nothing.
        let stale =
            TrunkDisc::new(Vec2::ZERO, 0.1, 0.1).with_profile(ids[0], 0.0, Vec2::X, 1.0, 1.0);
        let none = ClearingProfiles::default();
        assert_eq!(
            stale.effect(&none, Vec2::new(1.0, 0.0), 0.0, 0.5),
            (1.0, 0.5)
        );
    }

    /// Sectors turn with the disc's axis, interpolate between their
    /// directions and wrap; the radius and height scales stretch the
    /// profile.
    #[test]
    fn profile_follows_its_direction_and_scale() {
        let (table, ids) = ClearingProfiles::new(&[MeadowClearingProfile {
            radius: 1.0,
            rings: 2,
            sectors: 4,
            heights: vec![0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        }]);
        // Local +x along world +z, so local +z along world -x.
        let disc =
            TrunkDisc::new(Vec2::ZERO, 0.0, 0.0).with_profile(ids[0], 0.0, Vec2::Y, 2.0, 2.0);
        let cap = |dir: Vec2| disc.effect(&table, dir * 1.0, 0.0, 1.0).1;
        assert!((cap(Vec2::Y) - 0.2).abs() < 1e-6);
        assert!((cap(-Vec2::X) - 0.4).abs() < 1e-6);
        assert!((cap(Vec2::new(-1.0, 1.0).normalize()) - 0.3).abs() < 1e-5);
        assert!((cap(Vec2::new(1.0, 1.0).normalize()) - 0.5).abs() < 1e-5);
        assert!((disc.outer_radius() - 4.0).abs() < 1e-6);
    }

    /// In a cell of flat samples the disc clears grass, whatever the
    /// blade's height; a cell rising out of flat keeps it.
    #[test]
    fn flat_profile_clears_its_grass() {
        let (table, ids) =
            ClearingProfiles::new(&[radial(7.0, &[0.5, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0])]);
        let disc =
            TrunkDisc::new(Vec2::ZERO, 0.1, 0.1).with_profile(ids[0], 0.0, Vec2::X, 1.0, 1.0);
        let at = |r: f32, height: f32| disc.effect(&table, Vec2::new(r, 0.0), 0.0, height);
        assert_eq!(at(0.5, 1.0), (1.0, 0.25));
        assert_eq!(at(1.5, 1.0).0, 0.0);
        assert_eq!(at(1.5, 0.0).0, 0.0);
        assert_eq!(at(2.5, 0.0).0, 1.0);
        assert!((at(2.5, 1.0).1 - 0.15).abs() < 1e-6);
        assert_eq!(at(6.5, 1.0).0, 0.0);
        assert_eq!(at(7.5, 1.0), (1.0, 0.5));
    }

    /// Edge noise wobbles the clearing's band and the profile, each by its
    /// own amount, and the disc still ends inside its outer radius.
    #[test]
    fn edge_noise_wobbles_inside_the_outer_radius() {
        let (table, ids) = ClearingProfiles::new(&[radial(1.75, &[0.1; 8])]);
        let disc = TrunkDisc::new(Vec2::new(10.0, 4.0), 0.5, 1.5)
            .with_profile(ids[0], 0.0, Vec2::X, 1.0, 1.0)
            .with_edge_noise(0.4)
            .with_profile_noise(0.4);
        assert!(spread(ring(disc, &table, 1.2).map(|e| e.0)) > 0.3);
        assert!(spread(ring(disc, &table, 1.9).map(|e| e.1)) > 0.3);
        assert!(ring(disc, &table, disc.outer_radius()).all(|e| e == (1.0, 1.0)));
        // The clearing's noise alone leaves the profile round.
        let round = disc.with_profile_noise(0.0);
        assert!(spread(ring(round, &table, 1.2).map(|e| e.0)) > 0.3);
        assert!(spread(ring(round, &table, 1.9).map(|e| e.1)) < 1e-4);
    }

    /// The profile stands on its origin: a blade on higher ground keeps
    /// less height under it, and where the ground meets the profile the disc
    /// clears the grass.
    #[test]
    fn profile_caps_against_the_blade_ground() {
        let (table, ids) = ClearingProfiles::new(&[radial(2.0, &[0.4, 0.4, 0.4])]);
        let disc =
            TrunkDisc::new(Vec2::ZERO, 0.0, 0.0).with_profile(ids[0], 10.0, Vec2::X, 1.0, 1.0);
        let at = |ground_y: f32| disc.effect(&table, Vec2::new(0.5, 0.0), ground_y, 1.0);
        assert!((at(10.0).1 - 0.4).abs() < 1e-5);
        assert!((at(10.3).1 - 0.1).abs() < 1e-5);
        assert!((at(9.8).1 - 0.6).abs() < 1e-5);
        assert_eq!(at(10.4).0, 0.0);
        assert_eq!(at(10.3).0, 1.0);
    }

    #[test]
    #[should_panic(expected = "rings * sectors")]
    fn a_profile_holds_its_grid() {
        ClearingProfiles::new(&[MeadowClearingProfile {
            radius: 1.0,
            rings: 2,
            sectors: 2,
            heights: vec![0.0; 3],
        }]);
    }
}
