//! Patch placement: noisy-blob meadow patches scattered across a
//! biome AABB, biased away from tree canopy density, with optional
//! authored `PatchOverride`s. Pure CPU; deterministic from
//! `(variant.seed, biome AABB, tree positions)`.
//!
//! `place_patches` is the single entry point. It runs once per biome
//! at world load (typically from the biome's `enter_world`), returns
//! a `Vec<PatchPlacement>`, and the caller spawns one `MeadowPatch`
//! entity per placement that lands in a chunk.

use bevy::math::{Rect, Vec2};
use wyrand::WyRand;

/// Authored override consulted before procedural placement. Force
/// guarantees a patch at the given centre/radius (story location,
/// set-piece clearing); Suppress carves a hole (cave entrance, plaza).
/// Typically authored inline in the consumer's level/biome data and
/// passed through [`PlacementParams`].
#[derive(Clone, Copy, Debug)]
pub struct PatchOverride {
    pub centre: Vec2,
    pub radius: f32,
    pub mode: PatchOverrideMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchOverrideMode {
    Force,
    Suppress,
}

/// One placed patch returned from `place_patches`. The consumer
/// uploads each into a `MeadowPatch` entity at chunk dispatch.
#[derive(Clone, Copy, Debug)]
pub struct PatchPlacement {
    pub centre: Vec2,
    pub radius: f32,
    pub blade_count: u32,
    /// Per-patch hash seed. Derived from `variant.seed + index` so
    /// re-ordering placements doesn't change individual patches.
    pub seed: u32,
    pub edge_noise_amp: f32,
    pub canopy_density_at_centre: f32,
}

impl PatchPlacement {
    /// Conservative bounding radius for culling / trunk-disc
    /// inclusion: max blade height isn't included here (the entity's
    /// `Aabb` carries that), only the noisy-blob radial growth.
    pub fn outer_radius(&self) -> f32 {
        self.radius * (1.0 + self.edge_noise_amp)
    }
}

/// Tree-density grid used to bias patches away from thickets and
/// toward clearings. Built once at biome enter from the live tree
/// positions. Each cell holds the sum of inverse-square contributions
/// from trees within `INFLUENCE_RADIUS`. The result is a coarse
/// canopy-coverage proxy.
///
/// Cell size = 4 m. Influence radius = 16 m. Cost is
/// `O(trees × cells_per_tree)` once per enter.
pub struct TreeDensityField {
    rect: Rect,
    cells_xz: (u32, u32),
    /// Row-major `(cells.0 * cells.1)` cells.
    cells: Vec<f32>,
    cell_size: f32,
}

const TREE_DENSITY_CELL_SIZE: f32 = 4.0;
const TREE_DENSITY_INFLUENCE_RADIUS: f32 = 16.0;

impl TreeDensityField {
    /// Compute the field. `tree_positions_xz` is borrowed; the field
    /// stores only the rasterised summary. The output is normalised
    /// to roughly `[0, 1]` for typical forest densities
    /// (`canopy_threshold` is tuned against that range), but the upper
    /// bound is open — extremely dense thickets exceed 1.0.
    pub fn new(rect: Rect, tree_positions_xz: impl IntoIterator<Item = Vec2>) -> Self {
        let cells_x = ((rect.width() / TREE_DENSITY_CELL_SIZE).ceil() as u32).max(1);
        let cells_z = ((rect.height() / TREE_DENSITY_CELL_SIZE).ceil() as u32).max(1);
        let mut cells = vec![0.0_f32; (cells_x * cells_z) as usize];

        let inv_r2 = 1.0 / (TREE_DENSITY_INFLUENCE_RADIUS * TREE_DENSITY_INFLUENCE_RADIUS);
        let r = TREE_DENSITY_INFLUENCE_RADIUS;

        for tree_pos in tree_positions_xz {
            // Only walk cells inside this tree's influence square —
            // the outer rect-vs-inflated-tree clip keeps the loop
            // O(trees × ~64) for a 4 m grid + 16 m radius.
            let lo = tree_pos - Vec2::splat(r);
            let hi = tree_pos + Vec2::splat(r);
            let lo_cell = world_to_cell(lo, rect, cells_x, cells_z);
            let hi_cell = world_to_cell(hi, rect, cells_x, cells_z);

            for cz in lo_cell.1..=hi_cell.1 {
                for cx in lo_cell.0..=hi_cell.0 {
                    let cell_centre = cell_to_world(cx, cz, rect);
                    let d_sq = cell_centre.distance_squared(tree_pos);
                    let r_sq = TREE_DENSITY_INFLUENCE_RADIUS * TREE_DENSITY_INFLUENCE_RADIUS;
                    if d_sq < r_sq {
                        // Inverse-square contribution, normalised so a
                        // tree directly under a cell adds 1.0.
                        let contribution = (1.0 - d_sq * inv_r2).max(0.0);
                        cells[(cz * cells_x + cx) as usize] += contribution;
                    }
                }
            }
        }

        Self {
            rect,
            cells_xz: (cells_x, cells_z),
            cells,
            cell_size: TREE_DENSITY_CELL_SIZE,
        }
    }

    /// Sample density at a world XZ. Bilinear interpolation between
    /// the four surrounding cells. Outside the field's rect returns
    /// 0.0 — patches there fall under the open-canopy decision
    /// regardless.
    pub fn sample(&self, world_xz: Vec2) -> f32 {
        if !self.rect.contains(world_xz) {
            return 0.0;
        }
        let local = world_xz - self.rect.min;
        let fx = (local.x / self.cell_size).clamp(0.0, (self.cells_xz.0 - 1) as f32);
        let fz = (local.y / self.cell_size).clamp(0.0, (self.cells_xz.1 - 1) as f32);
        let x0 = fx.floor() as u32;
        let z0 = fz.floor() as u32;
        let x1 = (x0 + 1).min(self.cells_xz.0 - 1);
        let z1 = (z0 + 1).min(self.cells_xz.1 - 1);
        let tx = fx - x0 as f32;
        let tz = fz - z0 as f32;
        let c = |cx: u32, cz: u32| self.cells[(cz * self.cells_xz.0 + cx) as usize];
        let a = c(x0, z0);
        let b = c(x1, z0);
        let cc = c(x0, z1);
        let d = c(x1, z1);
        let top = a + (b - a) * tx;
        let bot = cc + (d - cc) * tx;
        top + (bot - top) * tz
    }
}

fn world_to_cell(world_xz: Vec2, rect: Rect, cells_x: u32, cells_z: u32) -> (u32, u32) {
    let local = world_xz - rect.min;
    let cx = (local.x / TREE_DENSITY_CELL_SIZE)
        .floor()
        .clamp(0.0, (cells_x - 1) as f32) as u32;
    let cz = (local.y / TREE_DENSITY_CELL_SIZE)
        .floor()
        .clamp(0.0, (cells_z - 1) as f32) as u32;
    (cx, cz)
}

fn cell_to_world(cx: u32, cz: u32, rect: Rect) -> Vec2 {
    rect.min
        + Vec2::new(
            (cx as f32 + 0.5) * TREE_DENSITY_CELL_SIZE,
            (cz as f32 + 0.5) * TREE_DENSITY_CELL_SIZE,
        )
}

/// Inputs to `place_patches`. Carries everything the algorithm
/// needs without forcing the caller to copy fields out of the
/// variant struct (which is only kept on the registry).
pub struct PlacementParams {
    /// World-space biome AABB. Patches stay strictly inside.
    pub biome_rect: Rect,
    pub variant_seed: u64,
    pub patches_per_m2: f32,
    pub patch_radius_range: (f32, f32),
    pub min_patch_spacing: f32,
    pub patch_edge_noise_amp: f32,
    /// Below this density a candidate is unconditionally rejected.
    pub canopy_threshold: f32,
    /// Between `canopy_soft_threshold..canopy_threshold` the patch's
    /// radius is linearly scaled down to the minimum.
    pub canopy_soft_threshold: f32,
    /// Splatmap weight floor for the variant's biome at the patch's
    /// centre + 8 perimeter samples. A candidate is rejected unless
    /// every sample exceeds this — keeps any patch from straddling a
    /// biome seam.
    pub biome_weight_threshold: f32,
    /// Density-at-rim taper width as a fraction of patch radius
    /// (typical 0.20–0.30).
    pub rim_falloff_fraction: f32,
    /// Per-m² blade target density at the patch centre. Total blade
    /// count is approximated by `density * π * radius² * (1 - rim/2)`.
    pub blades_per_m2: f32,
}

/// Run the placement algorithm:
/// 1. Insert `PatchOverride::Force` placements first — they claim
///    their footprint against subsequent procedural placement.
/// 2. Generate Poisson-disk-like candidates inside `biome_rect`,
///    seeded from `variant_seed`. Reject candidates that:
///    - are within `min_patch_spacing` of any existing placement
///    - fall inside any `PatchOverride::Suppress` rect
///    - have local tree-density ≥ `canopy_threshold`
///    - any of their 8 perimeter biome-weight samples falls below
///      `biome_weight_threshold`
/// 3. For accepted candidates whose density sits in
///    `canopy_soft_threshold..canopy_threshold`, scale the radius
///    down linearly — patches survive in mixed canopy but smaller.
/// 4. Each patch's `seed` is derived from `variant_seed + index` so
///    re-ordering doesn't change individual patches.
pub fn place_patches(
    params: &PlacementParams,
    overrides: &[PatchOverride],
    tree_density: &TreeDensityField,
    biome_weight_at: impl Fn(Vec2) -> f32,
) -> Vec<PatchPlacement> {
    let mut placements: Vec<PatchPlacement> = Vec::new();

    // 1) Force overrides first.
    for o in overrides
        .iter()
        .filter(|o| o.mode == PatchOverrideMode::Force)
    {
        if !params.biome_rect.contains(o.centre) {
            continue;
        }
        let canopy = tree_density.sample(o.centre);
        let blade_count = blade_count_for(o.radius, params);
        placements.push(PatchPlacement {
            centre: o.centre,
            radius: o.radius,
            blade_count,
            seed: derive_patch_seed(params.variant_seed, placements.len()),
            edge_noise_amp: params.patch_edge_noise_amp,
            canopy_density_at_centre: canopy,
        });
    }

    // 2) Procedural Poisson-like rejection sampling.
    let mut rng = WyRand::new(params.variant_seed);
    let attempts = poisson_attempts_for(params);
    for _ in 0..attempts {
        let radius_min = params.patch_radius_range.0;
        let radius_max = params.patch_radius_range.1;
        let raw_r = radius_min + rand01(&mut rng) * (radius_max - radius_min);
        let raw_centre = sample_inside(&params.biome_rect, raw_r, &mut rng);

        // Suppression overrides.
        if overrides.iter().any(|o| {
            o.mode == PatchOverrideMode::Suppress
                && raw_centre.distance(o.centre) < (raw_r + o.radius)
        }) {
            continue;
        }

        // Min-spacing rejection.
        if placements
            .iter()
            .any(|p| raw_centre.distance(p.centre) < (raw_r + p.radius + params.min_patch_spacing))
        {
            continue;
        }

        // Hard canopy rejection at centre.
        let canopy = tree_density.sample(raw_centre);
        if canopy >= params.canopy_threshold {
            continue;
        }

        // Soft canopy: scale radius down between soft and hard thresholds.
        let mut radius = raw_r;
        if canopy >= params.canopy_soft_threshold {
            let span = params.canopy_threshold - params.canopy_soft_threshold;
            let t = ((canopy - params.canopy_soft_threshold) / span.max(1e-3)).clamp(0.0, 1.0);
            radius = radius_min + (raw_r - radius_min) * (1.0 - t);
        }

        // Footprint-perimeter biome-weight check.
        if biome_weight_at(raw_centre) < params.biome_weight_threshold {
            continue;
        }
        let outer = radius * (1.0 + params.patch_edge_noise_amp);
        let mut all_in = true;
        for k in 0..8 {
            let theta = (k as f32) * std::f32::consts::TAU / 8.0;
            let p = raw_centre + Vec2::new(theta.cos(), theta.sin()) * outer;
            if biome_weight_at(p) < params.biome_weight_threshold {
                all_in = false;
                break;
            }
        }
        if !all_in {
            continue;
        }

        placements.push(PatchPlacement {
            centre: raw_centre,
            radius,
            blade_count: blade_count_for(radius, params),
            seed: derive_patch_seed(params.variant_seed, placements.len()),
            edge_noise_amp: params.patch_edge_noise_amp,
            canopy_density_at_centre: canopy,
        });
    }

    placements
}

fn poisson_attempts_for(params: &PlacementParams) -> u32 {
    // Aim for `patches_per_m² × biome_area`, with 4× the attempts to
    // absorb rejections. Capped at a reasonable upper bound so a
    // pathological biome rect doesn't blow out load times.
    let area = params.biome_rect.size().x * params.biome_rect.size().y;
    let target = (params.patches_per_m2 * area).round() as u32;
    (target.saturating_mul(4)).min(20_000)
}

fn blade_count_for(radius: f32, params: &PlacementParams) -> u32 {
    // Approximate effective area after rim falloff: outer 25% of the
    // disc averages 50% density, so effective area is
    // `π·r² × (1 - rim·0.25)`.
    let rim = params.rim_falloff_fraction.clamp(0.0, 0.5);
    let area = std::f32::consts::PI * radius * radius * (1.0 - rim * 0.25);
    let count = (params.blades_per_m2 * area).round() as u32;
    // Hard-cap at the shared mesh's blade slot count so the
    // template-mesh path never overflows.
    count.min(crate::mesh::MAX_BLADES_PER_PATCH)
}

fn derive_patch_seed(variant_seed: u64, index: usize) -> u32 {
    let mut h = variant_seed
        .wrapping_add(index as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 32;
    h as u32
}

fn sample_inside(rect: &Rect, radius: f32, rng: &mut WyRand) -> Vec2 {
    let inset = rect.inflate(-radius);
    let size = inset.size().max(Vec2::splat(0.0));
    Vec2::new(
        inset.min.x + rand01(rng) * size.x,
        inset.min.y + rand01(rng) * size.y,
    )
}

#[inline]
fn rand01(rng: &mut WyRand) -> f32 {
    (rng.rand() >> 32) as f32 / u32::MAX as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_density(rect: Rect) -> TreeDensityField {
        TreeDensityField::new(rect, std::iter::empty())
    }

    #[test]
    fn placement_is_deterministic() {
        let rect = Rect::new(-100.0, -50.0, 100.0, 50.0);
        let density = dummy_density(rect);
        let params = PlacementParams {
            biome_rect: rect,
            variant_seed: 0xCAFE_F00D,
            patches_per_m2: 0.0008,
            patch_radius_range: (8.0, 20.0),
            min_patch_spacing: 4.0,
            patch_edge_noise_amp: 0.18,
            canopy_threshold: 0.6,
            canopy_soft_threshold: 0.35,
            biome_weight_threshold: 0.6,
            rim_falloff_fraction: 0.25,
            blades_per_m2: 150.0,
        };
        let a = place_patches(&params, &[], &density, |_| 1.0);
        let b = place_patches(&params, &[], &density, |_| 1.0);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.seed, pb.seed);
            assert_eq!(pa.centre.x, pb.centre.x);
            assert_eq!(pa.centre.y, pb.centre.y);
            assert_eq!(pa.radius, pb.radius);
        }
    }

    #[test]
    fn force_override_lands() {
        let rect = Rect::new(-100.0, -50.0, 100.0, 50.0);
        let density = dummy_density(rect);
        let params = PlacementParams {
            biome_rect: rect,
            variant_seed: 1,
            patches_per_m2: 0.0,
            patch_radius_range: (8.0, 8.0),
            min_patch_spacing: 4.0,
            patch_edge_noise_amp: 0.0,
            canopy_threshold: 1.0,
            canopy_soft_threshold: 1.0,
            biome_weight_threshold: 0.5,
            rim_falloff_fraction: 0.0,
            blades_per_m2: 100.0,
        };
        let force = PatchOverride {
            centre: Vec2::new(0.0, 0.0),
            radius: 12.0,
            mode: PatchOverrideMode::Force,
        };
        let placements = place_patches(&params, &[force], &density, |_| 1.0);
        assert_eq!(placements.len(), 1);
        assert!((placements[0].centre - force.centre).length() < 1e-3);
        assert!((placements[0].radius - force.radius).abs() < 1e-3);
    }

    #[test]
    fn suppress_carves_hole() {
        let rect = Rect::new(-100.0, -50.0, 100.0, 50.0);
        let density = dummy_density(rect);
        let suppress = PatchOverride {
            centre: Vec2::new(0.0, 0.0),
            radius: 30.0,
            mode: PatchOverrideMode::Suppress,
        };
        let params = PlacementParams {
            biome_rect: rect,
            variant_seed: 0xDEAD_BEEF,
            patches_per_m2: 0.005,
            patch_radius_range: (5.0, 5.0),
            min_patch_spacing: 1.0,
            patch_edge_noise_amp: 0.0,
            canopy_threshold: 1.0,
            canopy_soft_threshold: 1.0,
            biome_weight_threshold: 0.5,
            rim_falloff_fraction: 0.0,
            blades_per_m2: 100.0,
        };
        let placements = place_patches(&params, &[suppress], &density, |_| 1.0);
        // No patch's footprint should overlap the suppressed disc.
        for p in &placements {
            assert!(p.centre.distance(suppress.centre) >= (p.radius + suppress.radius - 0.01));
        }
    }

    #[test]
    fn perimeter_check_keeps_patches_off_seam() {
        // Biome weight ramps from 0 to 1 across X. Patches placed at
        // X near 0 should be rejected because at least one perimeter
        // sample falls below the threshold.
        let rect = Rect::new(-50.0, -50.0, 50.0, 50.0);
        let density = dummy_density(rect);
        let params = PlacementParams {
            biome_rect: rect,
            variant_seed: 0xC0DE,
            patches_per_m2: 0.005,
            patch_radius_range: (8.0, 8.0),
            min_patch_spacing: 1.0,
            patch_edge_noise_amp: 0.0,
            canopy_threshold: 1.0,
            canopy_soft_threshold: 1.0,
            biome_weight_threshold: 0.6,
            rim_falloff_fraction: 0.0,
            blades_per_m2: 100.0,
        };
        let placements = place_patches(&params, &[], &density, |p| {
            ((p.x + 50.0) / 100.0).clamp(0.0, 1.0)
        });
        // No accepted patch should have its leftmost perimeter point
        // below the threshold.
        for p in &placements {
            let left = p.centre + Vec2::new(-p.radius, 0.0);
            let weight = ((left.x + 50.0) / 100.0).clamp(0.0, 1.0);
            assert!(
                weight >= params.biome_weight_threshold - 1e-3,
                "patch at {:?} has left perimeter weight {} below threshold",
                p.centre,
                weight,
            );
        }
    }
}
