//! Camera-local RT blade selection with persistent primitive ownership.
//!
//! The CPU mirrors the shaders' blade roots and survival (rim, clearing
//! discs, a tile's cover test) so dead blades take no slot. GPU arithmetic
//! may differ by a few ulps, so this is a budget/LOD decision, not geometric
//! clipping: the expansion re-derives every blade and collapses any it finds
//! dead.
//!
//! Work per update scales with the selected budget, not the blades in
//! range. The near band walks patches nearest first and stops once no
//! patch can hold a blade nearer than the current farthest pick. The far
//! band keeps each blade with a probability that falls with its distance;
//! blades are ranked by index, so each patch walks only the prefix of its
//! blades that its nearest point can keep, the way the raster LOD gate does.
use crate::field::{
    CoverSurvival, CoverTexel, MeadowGrid, cover_survives, hash_u32, hash01_u32, tile_root,
};
use crate::material::{ClearingProfiles, PatchTrunkSlot};
use crate::placement::{PatchPlacement, PatchShape};
use bevy::math::{Vec2, Vec3};
use bevy::platform::collections::{HashMap, HashSet};
use std::collections::BinaryHeap;
use std::sync::Arc;

static NEXT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Value of a vacant blade slot.
pub(crate) const INACTIVE: [u32; 2] = [u32::MAX; 2];
/// Radius of the exact near band around the selection centre, in metres. Also
/// the largest radius the exact-only reference mode accepts.
pub(crate) const RT_NEAR_DISTANCE: f32 = 18.0;
/// Casters end where raster grass stops casting shadows.
const FAR_DISTANCE: f32 = crate::mesh::SHADOW_MAX_DIST;
/// Eye travel, in metres, that triggers a reselection.
const RESELECT_DISTANCE: f32 = 0.5;
/// Share of the far capacity the budget aims for, leaving room for the
/// estimate's error before the hard cut.
const FAR_BUDGET_TARGET: f32 = 0.9;
/// Samples per side of the grid a patch's far-band blade count is
/// estimated over.
const ESTIMATE_SAMPLES: u32 = 4;

/// One live patch, by its stable index into the variant's patch buffers.
pub(crate) type LivePlacements = Arc<Vec<(u32, PatchPlacement)>>;

pub(crate) fn same_placements(a: &LivePlacements, b: &LivePlacements) -> bool {
    Arc::ptr_eq(a, b) || a == b
}

fn same_arc<T>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    a.as_ref().map(Arc::as_ptr) == b.as_ref().map(Arc::as_ptr)
}

/// Blade `blade`'s seed in a patch seeded `seed` (`derive_blade`).
fn blade_seed(seed: u32, blade: u32) -> u32 {
    hash_u32(seed ^ blade.wrapping_mul(0x27d4_eb2d))
}

/// The blade's root, `None` where the shader rejects both disc attempts.
fn blade_root(p: &PatchPlacement, blade: u32) -> Option<Vec2> {
    let seed = blade_seed(p.seed, blade);
    if p.is_tile() {
        return Some(p.centre + tile_root(p.radius, seed));
    }
    for attempt in 0u32..2 {
        let theta = hash01_u32(seed.wrapping_add(attempt * 17)) * std::f32::consts::TAU;
        let r = hash01_u32(seed.wrapping_add(attempt * 17 + 1)).sqrt();
        let candidate = Vec2::new(theta.cos(), theta.sin()) * r * p.radius;
        let harmonic = 1.0
            + p.edge_noise_amp
                * ((theta * 3.0 + (p.seed & 0xff) as f32 * 0.37).sin()
                    + 0.5 * (theta * 5.0 + (p.seed & 0x3f) as f32 * 1.7).sin());
        if candidate.length() <= p.radius * harmonic {
            return Some(p.centre + candidate);
        }
    }
    None
}

/// Height of a blade's root above the heightfield. MUST equal
/// `MEADOW_BLADE_GROUND_LIFT` in `meadow_shared.wesl`.
pub(crate) const BLADE_GROUND_LIFT: f32 = 0.02;

/// Everything a selection reads besides the eye.
#[derive(Clone, Default)]
pub(crate) struct RtInputs {
    pub(crate) placements: LivePlacements,
    /// The variant's clearing-disc rows, indexed like its patches, and the
    /// clearing profiles they reference.
    pub(crate) trunks: Arc<Vec<PatchTrunkSlot>>,
    pub(crate) profiles: Arc<ClearingProfiles>,
    pub(crate) cover: Option<Arc<MeadowGrid<CoverTexel>>>,
    pub(crate) survival: CoverSurvival,
    /// Ground heights, for 3D distances; without them distances are planar.
    pub(crate) heights: Option<Arc<MeadowGrid<f32>>>,
}

impl RtInputs {
    /// Whether `other` selects the same blades as `self` for any eye.
    pub(crate) fn same(&self, other: &Self) -> bool {
        same_placements(&self.placements, &other.placements)
            && Arc::ptr_eq(&self.trunks, &other.trunks)
            && Arc::ptr_eq(&self.profiles, &other.profiles)
            && same_arc(&self.cover, &other.cover)
            && same_arc(&self.heights, &other.heights)
            && self.survival == other.survival
    }

    /// Whether the blade rooted at `root` survives the patch's clearing
    /// discs and, on a tile, the cover field under them.
    pub(crate) fn live(&self, index: u32, p: &PatchPlacement, blade: u32, root: Vec2) -> bool {
        let mut stand = 1.0f32;
        let discs = self
            .trunks
            .get(index as usize)
            .map_or(&[][..], |slot| slot.discs());
        let mut ground_y = None;
        for disc in discs {
            if disc.centre().distance_squared(root) < disc.outer_radius().powi(2) {
                // The kernels' `ground_y`: a missing heightfield samples as 0.
                let ground_y = *ground_y.get_or_insert_with(|| {
                    self.heights.as_ref().map_or(0.0, |h| h.sample(root)) + BLADE_GROUND_LIFT
                });
                stand = stand.min(disc.effect(&self.profiles, root, ground_y, 0.0).0);
                // A cleared blade is dead on a tile and a disc patch alike.
                if stand <= 0.0 {
                    return false;
                }
            }
        }
        match (p.shape, &self.cover) {
            (PatchShape::Tile { cover_ceiling }, Some(cover)) => cover_survives(
                &self.survival,
                root,
                blade_seed(p.seed, blade),
                cover.sample(root).density * stand,
                cover_ceiling,
            ),
            _ => stand > 1e-4,
        }
    }

    /// A lower bound on the height between `eye_y` and patch `p`'s ground,
    /// from the heightfield at its centre and corners less a metre of
    /// slack for relief between them; 0 without a heightfield.
    fn height_gap(&self, p: &PatchPlacement, eye_y: f32) -> f32 {
        let Some(heights) = &self.heights else {
            return 0.0;
        };
        let r = p.outer_radius();
        let (lo, hi) = [
            Vec2::ZERO,
            Vec2::new(-r, -r),
            Vec2::new(r, -r),
            Vec2::new(-r, r),
            Vec2::splat(r),
        ]
        .iter()
        .map(|offset| heights.sample(p.centre + *offset))
        .fold((f32::MAX, f32::MIN), |(lo, hi), h| (lo.min(h), hi.max(h)));
        (eye_y - hi - 1.0).max(lo - 1.0 - eye_y).max(0.0)
    }

    /// Ground height at `root`, or `eye_y` without a heightfield.
    fn ground(&self, root: Vec2, eye_y: f32) -> f32 {
        self.heights
            .as_ref()
            .map_or(eye_y, |h| h.sample(root) + BLADE_GROUND_LIFT)
    }

    /// Squared eye distance of the blade at `root`.
    fn distance_squared(&self, root: Vec2, eye: Vec3) -> f32 {
        Vec3::new(root.x, self.ground(root, eye.y), root.y).distance_squared(eye)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Owner {
    patch: u32,
    blade: u32,
    incarnation: u64,
}

/// A near-band candidate, ordered nearest first.
#[derive(Clone, Copy)]
struct NearCandidate {
    owner: Owner,
    distance_squared: f32,
    rank: u32,
}
impl PartialEq for NearCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for NearCandidate {}
impl PartialOrd for NearCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NearCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance_squared
            .total_cmp(&other.distance_squared)
            .then(self.rank.cmp(&other.rank))
            .then(self.owner.cmp(&other.owner))
    }
}

/// A far-band candidate. Priority below zero marks a blade inside the near
/// band's reach, which the far band must hold for the complementary
/// transition; the rest rank by how deep they sit under their keep share.
#[derive(Clone, Copy)]
struct FarCandidate {
    owner: Owner,
    priority: f32,
}
fn far_order(a: &FarCandidate, b: &FarCandidate) -> std::cmp::Ordering {
    a.priority
        .total_cmp(&b.priority)
        .then(a.owner.cmp(&b.owner))
}

/// `meadow_blade_rank`: blade `blade`'s place in the LOD prefix order of
/// `count` blades, in (0, 1).
fn lod_rank(blade: u32, count: u32) -> f32 {
    (blade as f32 + 0.5) / count.max(1) as f32
}

/// Per-blade tie-break rank, decorrelated from the placement hash.
fn tie_break(p: &PatchPlacement, blade: u32) -> u32 {
    hash_u32(p.seed ^ blade.wrapping_mul(0x9e37_79b9))
}

/// The far band's share of live blades kept at eye distance `d`, past the
/// near band's reach `end`, under budget `budget` (`None`: unbounded). Full
/// up to `end`, thinning with distance; the budget term `1 - exp(-T /
/// beyond²)` approaches 1 continuously at `end`, so coverage has no cliff
/// at the boundary.
fn keep_share(d: f32, end: f32, budget: Option<f32>) -> f32 {
    let beyond = (d - end).max(0.0);
    if beyond <= 0.0 {
        return 1.0;
    }
    let ramp = (beyond / (FAR_DISTANCE - end).max(1e-3)).clamp(0.0, 1.0);
    let density = (1.0 - 0.75 * ramp) * (1.0 - 0.5 * ramp);
    density * budget.map_or(1.0, |t| 1.0 - (-t / (beyond * beyond)).exp())
}

/// A candidate patch: its live index, placement, incarnation and the
/// nearest planar distance any of its blades can lie from the eye.
struct Reach<'a> {
    index: u32,
    placement: &'a PatchPlacement,
    incarnation: u64,
    nearest: f32,
}

#[derive(Default)]
pub(crate) struct RtSelection {
    pub(crate) near_slots: Vec<[u32; 2]>,
    pub(crate) far_slots: Vec<[u32; 2]>,
    pub(crate) near_generation: u64,
    pub(crate) far_generation: u64,
    pub(crate) near_transition: [f32; 2],
    pub(crate) selection_center: Vec3,
    pub(crate) candidate_count: usize,
    pub(crate) coverage_radius: f32,
    exact_radius: Option<f32>,
    /// `(placement, incarnation)` per live patch index; a changed placement
    /// at an index is a new owner.
    incarnations: HashMap<u32, (PatchPlacement, u64)>,
    next_incarnation: u64,
    near_owners: Vec<Option<Owner>>,
    far_owners: Vec<Option<Owner>>,
    last_update: Option<(Vec3, usize, usize, RtInputs)>,
}

fn assign(
    selected: impl Iterator<Item = Owner>,
    capacity: usize,
    owners: &mut Vec<Option<Owner>>,
    slots: &mut Vec<[u32; 2]>,
    generation: &mut u64,
) {
    let mut newcomers: HashSet<_> = selected.collect();
    let mut changed = owners.len() != capacity;
    owners.resize(capacity, None);
    for owner in owners.iter_mut() {
        if let Some(id) = *owner
            && !newcomers.remove(&id)
        {
            *owner = None;
            changed = true;
        }
    }
    // Most owners survive camera motion. Only new owners need ordering;
    // retained slots never move, and unchanged sets do no sorting.
    if !newcomers.is_empty() {
        let mut new_owners: Vec<_> = newcomers.into_iter().collect();
        new_owners.sort_unstable();
        let mut free = owners.iter_mut().filter(|o| o.is_none());
        for owner in new_owners {
            *free.next().expect("selected blades fit capacity") = Some(owner);
        }
        changed = true;
    }
    if changed {
        slots.clear();
        slots.extend(
            owners
                .iter()
                .map(|o| o.map_or(INACTIVE, |id| [id.patch, id.blade])),
        );
        *generation = NEXT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl RtSelection {
    pub(crate) fn needs_update(
        &self,
        inputs: &RtInputs,
        eye: Vec3,
        near_capacity: usize,
        far_capacity: usize,
    ) -> bool {
        self.needs_viewer_update(eye, near_capacity, far_capacity)
            || self
                .last_update
                .as_ref()
                .is_some_and(|(_, _, _, last)| !last.same(inputs))
    }

    /// `needs_update` without the input comparison, for callers that know
    /// the inputs match the last update.
    pub(crate) fn needs_viewer_update(
        &self,
        eye: Vec3,
        near_capacity: usize,
        far_capacity: usize,
    ) -> bool {
        self.last_update
            .as_ref()
            .is_none_or(|(old_eye, old_near, old_far, _)| {
                old_eye.distance_squared(eye) > RESELECT_DISTANCE * RESELECT_DISTANCE
                    || *old_near != near_capacity
                    || *old_far != far_capacity
            })
    }

    #[cfg(test)]
    pub(crate) fn update(
        &mut self,
        inputs: &RtInputs,
        eye: Vec3,
        near_capacity: usize,
        far_capacity: usize,
    ) {
        self.update_mode(inputs, eye, near_capacity, far_capacity, None);
    }

    /// The live patches near enough to hold a caster, with their
    /// incarnations, nearest first.
    fn reach<'a>(&mut self, inputs: &'a RtInputs, eye: Vec3) -> Vec<Reach<'a>> {
        let same_patches = self
            .last_update
            .as_ref()
            .is_some_and(|(_, _, _, last)| Arc::ptr_eq(&last.placements, &inputs.placements));
        if !same_patches {
            let live: HashSet<_> = inputs.placements.iter().map(|(index, _)| *index).collect();
            self.incarnations.retain(|index, _| live.contains(index));
        }
        let mut reach = Vec::new();
        for (index, p) in inputs.placements.iter() {
            let planar = (p.centre.distance(Vec2::new(eye.x, eye.z)) - p.outer_radius()).max(0.0);
            if planar > FAR_DISTANCE || p.blade_count == 0 {
                continue;
            }
            let nearest = planar.hypot(inputs.height_gap(p, eye.y));
            if nearest > FAR_DISTANCE {
                continue;
            }
            let next = &mut self.next_incarnation;
            let entry = self.incarnations.entry(*index).or_insert_with(|| {
                *next += 1;
                (*p, *next)
            });
            if entry.0 != *p {
                *next += 1;
                *entry = (*p, *next);
            }
            reach.push(Reach {
                index: *index,
                placement: p,
                incarnation: entry.1,
                nearest,
            });
        }
        reach.sort_by(|a, b| a.nearest.total_cmp(&b.nearest).then(a.index.cmp(&b.index)));
        reach
    }

    /// The `capacity` live blades nearest `eye` within `radius`, and the
    /// number of live blades the walk met within it.
    fn select_near(
        inputs: &RtInputs,
        reach: &[Reach],
        eye: Vec3,
        radius: f32,
        capacity: usize,
    ) -> (Vec<NearCandidate>, usize) {
        if capacity == 0 {
            return (Vec::new(), 0);
        }
        let mut heap = BinaryHeap::with_capacity(capacity);
        let mut met = 0;
        let eye_xz = Vec2::new(eye.x, eye.z);
        for r in reach {
            // The heap's worst pick bounds what the rest can contribute.
            let limit_squared = match heap.peek() {
                Some(worst) if heap.len() == capacity => {
                    let NearCandidate {
                        distance_squared, ..
                    } = *worst;
                    distance_squared.min(radius * radius)
                }
                _ => radius * radius,
            };
            if r.nearest * r.nearest > limit_squared {
                break;
            }
            let p = r.placement;
            for blade in 0..p.blade_count {
                let Some(root) = blade_root(p, blade) else {
                    continue;
                };
                if root.distance_squared(eye_xz) > limit_squared {
                    continue;
                }
                let distance_squared = inputs.distance_squared(root, eye);
                if distance_squared > radius * radius || !inputs.live(r.index, p, blade, root) {
                    continue;
                }
                met += 1;
                let candidate = NearCandidate {
                    owner: Owner {
                        patch: r.index,
                        blade,
                        incarnation: r.incarnation,
                    },
                    distance_squared,
                    rank: tie_break(p, blade),
                };
                if heap.len() < capacity {
                    heap.push(candidate);
                } else if heap.peek().is_some_and(|worst| candidate < *worst) {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }
        (heap.into_vec(), met)
    }

    /// Far-band blade samples: `(eye distance, live blades it stands for)`
    /// over a grid of each patch's ground, within `start..=FAR_DISTANCE`.
    fn far_samples(inputs: &RtInputs, reach: &[Reach], eye: Vec3, start: f32) -> Vec<(f32, f32)> {
        let n = ESTIMATE_SAMPLES;
        let offsets = (0..n * n).map(|i| {
            (Vec2::new((i % n) as f32, (i / n) as f32) + 0.5) / n as f32 * 2.0 - Vec2::ONE
        });
        let tile_offsets: Vec<Vec2> = offsets.clone().collect();
        let disc_offsets: Vec<Vec2> = offsets.filter(|o| o.length_squared() <= 1.0).collect();
        let mut samples = Vec::new();
        for r in reach {
            let p = r.placement;
            let (offsets, ceiling) = match p.shape {
                PatchShape::Tile { cover_ceiling } => {
                    (&tile_offsets, Some(cover_ceiling.max(1e-4)))
                }
                PatchShape::Disc => (&disc_offsets, None),
            };
            let blades = p.blade_count as f32 / offsets.len() as f32;
            for offset in offsets {
                let point = p.centre + *offset * p.radius;
                let d = inputs.distance_squared(point, eye).sqrt();
                if d < start || d > FAR_DISTANCE {
                    continue;
                }
                let live = match (&inputs.cover, ceiling) {
                    (Some(cover), Some(ceiling)) => {
                        (cover.sample(point).density / ceiling).min(1.0)
                    }
                    _ => 1.0,
                };
                samples.push((d, blades * live));
            }
        }
        samples
    }

    /// The far band's budget term: unbounded when every eligible blade
    /// fits, else the largest that keeps the estimate from `samples` on
    /// target.
    fn far_budget(samples: &[(f32, f32)], end: f32, capacity: usize) -> Option<f32> {
        let estimate = |budget| -> f32 {
            samples
                .iter()
                .map(|&(d, blades)| blades * keep_share(d, end, budget))
                .sum()
        };
        let target = capacity as f32 * FAR_BUDGET_TARGET;
        if estimate(None) <= target {
            return None;
        }
        let (mut lo, mut hi) = (-12.0_f32, 12.0_f32);
        for _ in 0..16 {
            let mid = 0.5 * (lo + hi);
            if estimate(Some(mid.exp2())) > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(lo.exp2())
    }

    pub(crate) fn update_mode(
        &mut self,
        inputs: &RtInputs,
        eye: Vec3,
        near_capacity: usize,
        far_capacity: usize,
        exact_radius: Option<f32>,
    ) {
        if self.exact_radius != exact_radius {
            self.last_update = None;
            self.exact_radius = exact_radius;
        }
        let near_distance = exact_radius.unwrap_or(RT_NEAR_DISTANCE);
        // Hold the selection centre within the reselect distance; shader
        // transitions use the same centre so they never outrun the
        // currently selected slots.
        if !self.needs_update(inputs, eye, near_capacity, far_capacity) {
            return;
        }
        self.selection_center = eye;
        let reach = self.reach(inputs, eye);
        let (near, met) = Self::select_near(inputs, &reach, eye, near_distance, near_capacity);
        self.candidate_count = met;
        // When not saturated retain the full near band, avoiding a changing
        // transition whenever the outermost patch streams in or out.
        let end = if near_capacity == 0 {
            0.0
        } else if near.len() == near_capacity {
            near.iter()
                .map(|c| c.distance_squared)
                .fold(0.0, f32::max)
                .sqrt()
                .min(near_distance)
        } else {
            near_distance
        };
        self.coverage_radius = end;
        // Diagnostic casters are all exact, with no representation dither.
        // Both limits lie beyond the allowed exact diagnostic radius.
        self.near_transition = if exact_radius.is_some() {
            [51.0, 52.0]
        } else {
            [end * 0.75, end]
        };
        let mut far = Vec::new();
        if exact_radius.is_none() && far_capacity > 0 {
            let start = self.near_transition[0];
            let samples = Self::far_samples(inputs, &reach, eye, start);
            let budget = Self::far_budget(&samples, end, far_capacity);
            let eye_xz = Vec2::new(eye.x, eye.z);
            for r in &reach {
                let p = r.placement;
                // Keep shares fall with distance, so the patch's nearest
                // point bounds its blades' and its walk stops at that prefix.
                let bound = keep_share(r.nearest, end, budget);
                let prefix = ((bound * p.blade_count as f32).ceil() as u32).min(p.blade_count);
                for blade in 0..prefix {
                    let Some(root) = blade_root(p, blade) else {
                        continue;
                    };
                    if root.distance_squared(eye_xz) > FAR_DISTANCE * FAR_DISTANCE {
                        continue;
                    }
                    let d = inputs.distance_squared(root, eye).sqrt();
                    if d < start || d > FAR_DISTANCE {
                        continue;
                    }
                    let rank = lod_rank(blade, p.blade_count);
                    let share = keep_share(d, end, budget);
                    if rank >= share || !inputs.live(r.index, p, blade, root) {
                        continue;
                    }
                    far.push(FarCandidate {
                        owner: Owner {
                            patch: r.index,
                            blade,
                            incarnation: r.incarnation,
                        },
                        priority: if d <= end { -1.0 } else { rank / share },
                    });
                }
            }
            if far.len() > far_capacity {
                far.select_nth_unstable_by(far_capacity, far_order);
                far.truncate(far_capacity);
            }
        }
        assign(
            near.iter().map(|c| c.owner),
            near_capacity,
            &mut self.near_owners,
            &mut self.near_slots,
            &mut self.near_generation,
        );
        assign(
            far.iter().map(|c| c.owner),
            far_capacity,
            &mut self.far_owners,
            &mut self.far_slots,
            &mut self.far_generation,
        );
        self.last_update = Some((eye, near_capacity, far_capacity, inputs.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::TrunkDisc;
    use bevy::math::UVec2;

    fn patch(x: f32, seed: u32) -> PatchPlacement {
        PatchPlacement {
            centre: Vec2::new(x, 0.0),
            radius: 2.0,
            blade_count: 100,
            seed,
            edge_noise_amp: 0.2,
            canopy_density_at_centre: 0.0,
            palette: 0,
            shape: PatchShape::Disc,
        }
    }
    fn inputs(placements: &[(u32, PatchPlacement)]) -> RtInputs {
        RtInputs {
            placements: Arc::new(placements.to_vec()),
            ..Default::default()
        }
    }
    fn eye(x: f32) -> Vec3 {
        Vec3::new(x, 0.0, 0.0)
    }
    /// Every live blade of `inputs` within `radius` of `eye`, nearest first.
    fn live_within(inputs: &RtInputs, eye: Vec3, radius: f32) -> Vec<[u32; 2]> {
        let mut all = Vec::new();
        for (index, p) in inputs.placements.iter() {
            for blade in 0..p.blade_count {
                if let Some(root) = blade_root(p, blade)
                    && inputs.live(*index, p, blade, root)
                {
                    let d = inputs.distance_squared(root, eye);
                    if d <= radius * radius {
                        all.push((d, tie_break(p, blade), [*index, blade]));
                    }
                }
            }
        }
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        all.into_iter().map(|(_, _, id)| id).collect()
    }
    fn occupied(slots: &[[u32; 2]]) -> HashSet<[u32; 2]> {
        slots.iter().copied().filter(|s| *s != INACTIVE).collect()
    }

    #[test]
    fn nearest_coverage_capacity_and_retained_slots() {
        let patches = inputs(&[(0, patch(2.0, 12)), (1, patch(12.0, 13))]);
        let mut s = RtSelection::default();
        s.update(&patches, Vec3::ZERO, 50, 40);
        assert_eq!(s.near_slots.len(), 50);
        assert_eq!(s.far_slots.len(), 40);
        assert!(s.near_slots.iter().all(|id| id[0] == 0));
        let old: HashMap<_, _> = s
            .near_slots
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let selected = occupied(&s.near_slots);
        let furthest = s.near_transition[1];
        for b in 0..100 {
            if let Some(r) = blade_root(&patches.placements[0].1, b)
                && r.length() < furthest
            {
                assert!(selected.contains(&[0, b]));
            }
        }
        s.update(&patches, eye(0.6), 50, 40);
        for (i, id) in s.near_slots.iter().enumerate() {
            if let Some(old_i) = old.get(id) {
                assert_eq!(i, *old_i);
            }
        }
    }

    #[test]
    fn ownership_generation_and_determinism() {
        let mut list = vec![(0, patch(2.0, 12)), (1, patch(12.0, 13))];
        let mut s = RtSelection::default();
        s.update(&inputs(&list), Vec3::ZERO, 50, 40);
        let before = (
            s.near_slots.clone(),
            s.far_slots.clone(),
            s.near_generation,
            s.far_generation,
        );
        list.reverse();
        s.update(&inputs(&list), Vec3::ZERO, 50, 40);
        assert_eq!(
            before,
            (
                s.near_slots.clone(),
                s.far_slots.clone(),
                s.near_generation,
                s.far_generation
            )
        );
        let mut other = RtSelection::default();
        other.update(&inputs(&list), Vec3::ZERO, 50, 40);
        assert_eq!(other.near_slots, s.near_slots);
        assert_eq!(other.far_slots, s.far_slots);
        assert_ne!(other.near_generation, s.near_generation);
        list[1].1.seed += 1;
        s.update(&inputs(&list), Vec3::ZERO, 50, 40);
        assert!(s.near_generation > before.2);
        let generation = s.near_generation;
        s.update(&inputs(&[]), Vec3::ZERO, 50, 40);
        assert!(s.near_slots.iter().all(|s| *s == INACTIVE));
        assert!(s.near_generation > generation);
    }

    /// A patch leaving the live set frees only its own slots: the others'
    /// owners keep their slots, so their RT identity survives eviction.
    #[test]
    fn evicting_a_patch_keeps_other_owners_in_place() {
        let list = [
            (3, patch(-3.0, 21)),
            (7, patch(3.0, 22)),
            (9, patch(9.0, 23)),
        ];
        let mut s = RtSelection::default();
        s.update(&inputs(&list), Vec3::ZERO, 400, 400);
        let before_near = s.near_slots.clone();
        let before_far = s.far_slots.clone();
        s.update(&inputs(&[list[0], list[2]]), Vec3::ZERO, 400, 400);
        for (before, after) in [(before_near, &s.near_slots), (before_far, &s.far_slots)] {
            for (old, new) in before.iter().zip(after) {
                if old[0] == 7 {
                    assert_eq!(*new, INACTIVE);
                } else if *old != INACTIVE {
                    assert_eq!(old, new, "a surviving owner moved slot");
                }
            }
        }
    }

    /// A tile field with clearing discs: the pruned near selection is the
    /// brute-force nearest `capacity` live blades, and every selected blade
    /// in either band passes the survival test the shaders apply.
    #[test]
    fn selection_matches_brute_force_and_skips_dead_blades() {
        let dims = UVec2::new(17, 17);
        let texels = (0..dims.x * dims.y)
            .map(|i| CoverTexel::new(((i % dims.x) as f32 / 16.0).min(1.0), 1.0, 0.0))
            .collect();
        let cover = MeadowGrid::new(Vec2::splat(-16.0), Vec2::splat(16.0), dims, texels);
        let heights = MeadowGrid::new(
            Vec2::splat(-16.0),
            Vec2::splat(16.0),
            UVec2::new(2, 2),
            vec![0.0, 4.0, 0.0, 4.0],
        );
        let tiles: Vec<_> = (0..16u32)
            .map(|i| {
                let centre = Vec2::new((i % 4) as f32 * 8.0 - 12.0, (i / 4) as f32 * 8.0 - 12.0);
                (
                    i * 3 + 1,
                    PatchPlacement::tile(centre, 4.0, 600, 1000 + i, 1.0, 0),
                )
            })
            .collect();
        // Tile index 28 is centred at (-4, 4).
        let mut trunks = vec![PatchTrunkSlot::default(); 64];
        trunks[28] = PatchTrunkSlot::from_discs(&[TrunkDisc::new(Vec2::new(-4.0, 4.0), 2.0, 0.5)]);
        let field = RtInputs {
            placements: Arc::new(tiles),
            trunks: Arc::new(trunks),
            profiles: Arc::default(),
            cover: Some(Arc::new(cover)),
            survival: CoverSurvival {
                noise_scale_m: 3.0,
                clumpiness: 0.6,
            },
            heights: Some(Arc::new(heights)),
        };
        let eye = Vec3::new(1.5, 3.0, -2.0);
        let capacity = 300;
        let mut s = RtSelection::default();
        s.update(&field, eye, capacity, 2000);
        let reference: HashSet<[u32; 2]> = live_within(&field, eye, RT_NEAR_DISTANCE)
            .into_iter()
            .take(capacity)
            .collect();
        assert_eq!(occupied(&s.near_slots), reference);
        let by_index: HashMap<u32, PatchPlacement> = field.placements.iter().copied().collect();
        let selected = occupied(&s.near_slots);
        let far = occupied(&s.far_slots);
        assert!(far.len() > 500, "{}", far.len());
        for id in selected.union(&far) {
            let p = &by_index[&id[0]];
            let root = blade_root(p, id[1]).unwrap();
            assert!(
                field.live(id[0], p, id[1], root),
                "dead blade {id:?} selected"
            );
            if id[0] == 28 {
                assert!(root.distance(Vec2::new(-4.0, 4.0)) > 2.0);
            }
        }
    }

    /// Selection measures eye distance in 3D: an eye high above the grass
    /// selects nothing in the near band.
    #[test]
    fn high_eye_selects_by_3d_distance() {
        let heights = MeadowGrid::new(
            Vec2::splat(-50.0),
            Vec2::splat(50.0),
            UVec2::new(2, 2),
            vec![0.0; 4],
        );
        let mut field = inputs(&[(0, patch(0.0, 5))]);
        field.heights = Some(Arc::new(heights));
        let mut s = RtSelection::default();
        s.update(&field, Vec3::new(0.0, 30.0, 0.0), 50, 50);
        assert!(occupied(&s.near_slots).is_empty());
        assert!(!occupied(&s.far_slots).is_empty());
        s.update(&field, Vec3::new(0.0, 60.0, 0.0), 50, 50);
        assert!(occupied(&s.far_slots).is_empty());
    }

    #[test]
    fn far_budget_preserves_complementary_transition() {
        let list = inputs(&[
            (0, patch(2.0, 12)),
            (1, patch(22.0, 13)),
            (2, patch(30.0, 14)),
        ]);
        let mut s = RtSelection::default();
        s.update(&list, Vec3::ZERO, 50, 50);
        let far = occupied(&s.far_slots);
        for id in occupied(&s.near_slots) {
            let r = blade_root(&list.placements[0].1, id[1]).unwrap();
            if r.length() >= s.near_transition[0] {
                assert!(far.contains(&id));
            }
        }
    }

    #[test]
    fn saturated_far_budget_has_continuous_radial_coverage() {
        // Uniform disc with substantially more eligible blades than slots.
        // Retention thins gradually with distance; a global far cutoff
        // would drop from complete overlap to sparse coverage in the first
        // bin outside the near boundary.
        let p = PatchPlacement {
            centre: Vec2::ZERO,
            radius: 50.0,
            blade_count: 200_000,
            seed: 741,
            edge_noise_amp: 0.0,
            canopy_density_at_centre: 0.0,
            palette: 0,
            shape: PatchShape::Disc,
        };
        let list = inputs(&[(0, p)]);
        let mut s = RtSelection::default();
        s.update(&list, Vec3::ZERO, 8_000, 20_000);
        let far = occupied(&s.far_slots);
        assert!(far.len() > 17_000 && far.len() <= 20_000, "{}", far.len());
        let end = s.near_transition[1];
        let bounds = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0];
        let mut total = [0usize; 8];
        let mut selected = [0usize; 8];
        for blade in 0..p.blade_count {
            let Some(root) = blade_root(&p, blade) else {
                continue;
            };
            let delta = root.length() - end;
            if let Some(bin) = bounds
                .windows(2)
                .position(|w| delta >= w[0] && delta < w[1])
            {
                total[bin] += 1;
                selected[bin] += usize::from(far.contains(&[0, blade]));
            }
        }
        let rates: Vec<_> = selected
            .iter()
            .zip(total)
            .map(|(&n, d)| n as f32 / d as f32)
            .collect();
        assert!(
            rates[0] > 0.97,
            "boundary coverage must approach full: {rates:?}"
        );
        assert!(
            rates[1] > 0.94,
            "no cliff immediately outside first bin: {rates:?}"
        );
        for pair in rates.windows(2) {
            assert!(pair[1] <= pair[0] + 0.03, "{rates:?}");
        }
        assert!(
            rates[7] < 0.05,
            "must respect budget through gradual thinning: {rates:?}"
        );
        for id in occupied(&s.near_slots) {
            let r = blade_root(&p, id[1]).unwrap().length();
            if r >= s.near_transition[0] {
                assert!(far.contains(&id));
            }
        }
    }

    #[test]
    fn selection_center_hysteresis_and_input_changes() {
        let mut list = vec![(0, patch(2.0, 12))];
        let mut s = RtSelection::default();
        let same = inputs(&list);
        s.update(&same, Vec3::ZERO, 50, 50);
        let generation = s.near_generation;
        s.update(&same, eye(0.4), 50, 50);
        assert_eq!(s.selection_center, Vec3::ZERO);
        assert!(!s.needs_update(&same, eye(0.4), 50, 50));
        assert!(s.needs_update(&same, eye(0.6), 50, 50));
        assert!(s.needs_update(&same, Vec3::new(0.0, 0.6, 0.0), 50, 50));
        assert_eq!(s.near_generation, generation);
        s.update(&same, eye(0.6), 50, 50);
        assert_eq!(s.selection_center, eye(0.6));
        list[0].1.seed += 1;
        let changed = inputs(&list);
        assert!(s.needs_update(&changed, eye(0.6), 50, 50));
        assert!(!s.needs_viewer_update(eye(0.6), 50, 50));
        assert!(s.needs_viewer_update(eye(0.6), 51, 50));
        let mut retrunked = same.clone();
        retrunked.trunks = Arc::new(Vec::new());
        assert!(s.needs_update(&retrunked, eye(0.6), 50, 50));
        s.update(&changed, eye(0.7), 50, 50);
        assert_eq!(s.selection_center, eye(0.7));
    }

    #[test]
    fn exact_reference_has_complete_local_coverage_and_no_proxies() {
        let list = inputs(&[(0, patch(2.0, 12)), (1, patch(12.0, 13))]);
        let mut s = RtSelection::default();
        s.update_mode(&list, Vec3::ZERO, 200, 200, Some(4.0));
        let expected: HashSet<_> = live_within(&list, Vec3::ZERO, 4.0).into_iter().collect();
        let actual = occupied(&s.near_slots);
        assert_eq!(actual, expected);
        assert_eq!(s.candidate_count, expected.len());
        assert_eq!(s.coverage_radius, 4.0);
        assert!(s.near_transition[0] > 4.0);
        assert!(s.far_slots.iter().all(|s| *s == INACTIVE));
        // Mode and radius changes must bypass the stationary-camera cache.
        s.update_mode(&list, Vec3::ZERO, 200, 200, Some(1.0));
        assert_eq!(s.coverage_radius, 1.0);
        assert!(occupied(&s.near_slots).len() < actual.len());
        s.update(&list, Vec3::ZERO, 200, 200);
        assert_eq!(s.coverage_radius, 18.0);
        assert!(s.near_transition[0] < 18.0);
    }

    #[test]
    fn exact_reference_reports_capacity_limited_coverage() {
        let p = patch(2.0, 12);
        let list = inputs(&[(0, p)]);
        let mut s = RtSelection::default();
        s.update_mode(&list, Vec3::ZERO, 10, 20, Some(4.0));
        assert!(s.candidate_count > 10);
        assert!(s.coverage_radius < 4.0);
        assert_eq!(s.near_slots.len(), 10);
        let selected = occupied(&s.near_slots);
        for b in 0..p.blade_count {
            if blade_root(&p, b).is_some_and(|r| r.length() < s.coverage_radius) {
                assert!(selected.contains(&[0, b]));
            }
        }
        assert!(s.far_slots.iter().all(|s| *s == INACTIVE));
    }

    #[test]
    fn empty_and_zero_capacity() {
        let mut s = RtSelection::default();
        s.update(&inputs(&[]), Vec3::ZERO, 0, 0);
        assert!(s.near_slots.is_empty());
        assert_eq!(s.near_transition, [0.0, 0.0]);
        s.update(&inputs(&[(0, patch(2.0, 12))]), Vec3::ZERO, 0, 0);
        assert!(s.near_slots.is_empty());
        assert!(s.far_slots.is_empty());
    }

    /// The CPU roots carry the shader's derivation constants.
    #[test]
    fn roots_match_the_shader_constants() {
        let compute = include_str!("meadow_compute.wesl");
        let shared = include_str!("meadow_shared.wesl");
        for needle in [
            "hash_u32(p_seed ^ (blade_idx * 0x27D4EB2Du))",
            "hash_u32(blade_seed + attempt * 17u)",
            "sin(theta * 3.0 + f32(p_seed & 0xFFu) * 0.37)",
            "0.5 * sin(theta * 5.0 + f32(p_seed & 0x3Fu) * 1.7)",
        ] {
            assert!(
                compute.contains(needle),
                "`{needle}` missing from meadow_compute.wesl"
            );
        }
        assert!(shared.contains("return (2.0 * u - 1.0) * half;"));
    }
}
