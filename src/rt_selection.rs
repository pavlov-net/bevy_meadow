//! Camera-local RT blade selection with persistent primitive ownership.
//! CPU roots mirror the shader's two-attempt placement; GPU trig may differ
//! by a few ulps, so this is a budget/LOD decision, not geometric clipping.
use crate::placement::PatchPlacement;
use bevy::math::Vec2;
use bevy::platform::collections::{HashMap, HashSet};

static NEXT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Value of a vacant blade slot.
pub(crate) const INACTIVE: [u32; 2] = [u32::MAX; 2];
/// Radius of the exact near band around the selection centre, in metres. Also
/// the largest radius the exact-only reference mode accepts.
pub(crate) const RT_NEAR_DISTANCE: f32 = 18.0;
/// Casters end where raster grass stops casting shadows.
const FAR_DISTANCE: f32 = crate::mesh::SHADOW_MAX_DIST;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Owner {
    patch: u32,
    blade: u32,
    incarnation: u64,
}
struct CachedPatch {
    incarnation: u64,
    key: [u32; 9],
    roots: Vec<(u32, Vec2)>,
}
#[derive(Clone, Copy)]
struct Candidate {
    owner: Owner,
    distance_squared: f32,
    rank: u32,
    near_overlap: bool,
    far_priority: f32,
}

#[derive(Default)]
pub(crate) struct RtSelection {
    pub(crate) near_slots: Vec<[u32; 2]>,
    pub(crate) far_slots: Vec<[u32; 2]>,
    pub(crate) near_generation: u64,
    pub(crate) far_generation: u64,
    pub(crate) near_transition: [f32; 2],
    pub(crate) selection_center: Vec2,
    pub(crate) candidate_count: usize,
    pub(crate) coverage_radius: f32,
    exact_radius: Option<f32>,
    cache: HashMap<u32, CachedPatch>,
    next_incarnation: u64,
    near_owners: Vec<Option<Owner>>,
    far_owners: Vec<Option<Owner>>,
    last_update: Option<(Vec2, usize, usize, Vec<[u32; 9]>)>,
}

fn placement_key(index: u32, p: &PatchPlacement) -> [u32; 9] {
    [
        index,
        p.centre.x.to_bits(),
        p.centre.y.to_bits(),
        p.radius.to_bits(),
        p.blade_count,
        p.seed,
        p.edge_noise_amp.to_bits(),
        p.canopy_density_at_centre.to_bits(),
        0,
    ]
}
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn unit(x: u32) -> f32 {
    (x & 0x00ff_ffff) as f32 / 16_777_216.0
}
fn root(p: &PatchPlacement, blade: u32) -> Option<Vec2> {
    let seed = hash(p.seed ^ blade.wrapping_mul(0x27d4eb2d));
    for attempt in 0u32..2 {
        let h0 = hash(seed.wrapping_add(attempt * 17));
        let h1 = hash(seed.wrapping_add(attempt * 17 + 1));
        let theta = unit(h0) * std::f32::consts::TAU;
        let candidate = Vec2::new(theta.cos(), theta.sin()) * unit(h1).sqrt() * p.radius;
        let harmonic = 1.0
            + p.edge_noise_amp
                * ((theta * 3.0 + (p.seed & 0xff) as f32 * 0.37).sin()
                    + 0.5 * (theta * 5.0 + (p.seed & 0x3f) as f32 * 1.7).sin());
        if candidate.length() <= p.radius * harmonic {
            return Some(p.centre + candidate);
        }
    }
    None // Both rejected: the shader sets rim_factor to zero.
}
fn roots(p: &PatchPlacement) -> Vec<(u32, Vec2)> {
    (0..p.blade_count)
        .filter_map(|b| root(p, b).map(|r| (b, r)))
        .collect()
}
/// Per-blade selection rank, decorrelated from the placement hash.
fn blade_rank(p: &PatchPlacement, blade: u32) -> u32 {
    hash(p.seed ^ blade.wrapping_mul(0x9e3779b9))
}
fn near_order(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    a.distance_squared
        .total_cmp(&b.distance_squared)
        .then(a.rank.cmp(&b.rank))
        .then(a.owner.cmp(&b.owner))
}
fn far_order(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.near_overlap
        .cmp(&a.near_overlap)
        .then(a.far_priority.total_cmp(&b.far_priority))
        .then(a.rank.cmp(&b.rank))
        .then(a.owner.cmp(&b.owner))
}
fn bounded(
    candidates: &mut Vec<Candidate>,
    capacity: usize,
    order: fn(&Candidate, &Candidate) -> std::cmp::Ordering,
) {
    if candidates.len() > capacity {
        candidates.select_nth_unstable_by(capacity, order);
        candidates.truncate(capacity);
    }
}
fn assign(
    candidates: &mut [Candidate],
    capacity: usize,
    owners: &mut Vec<Option<Owner>>,
    slots: &mut Vec<[u32; 2]>,
    generation: &mut u64,
) {
    let mut newcomers: HashSet<_> = candidates.iter().map(|c| c.owner).collect();
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
        placements: &[(u32, PatchPlacement)],
        viewer: Vec2,
        near_capacity: usize,
        far_capacity: usize,
    ) -> bool {
        // Share this inexpensive gate with the async task scheduler.
        self.last_update
            .as_ref()
            .is_none_or(|(old_viewer, old_near, old_far, keys)| {
                old_viewer.distance_squared(viewer) > 0.25
                    || *old_near != near_capacity
                    || *old_far != far_capacity
                    || keys.len() != placements.len()
                    || !keys
                        .iter()
                        .zip(placements)
                        .all(|(key, (index, p))| *key == placement_key(*index, p))
            })
    }

    #[cfg(test)]
    pub(crate) fn update(
        &mut self,
        placements: &[(u32, PatchPlacement)],
        viewer: Vec2,
        near_capacity: usize,
        far_capacity: usize,
    ) {
        self.update_mode(placements, viewer, near_capacity, far_capacity, None);
    }

    pub(crate) fn update_mode(
        &mut self,
        placements: &[(u32, PatchPlacement)],
        viewer: Vec2,
        near_capacity: usize,
        far_capacity: usize,
        exact_radius: Option<f32>,
    ) {
        if self.exact_radius != exact_radius {
            self.last_update = None;
            self.exact_radius = exact_radius;
        }
        let near_distance = exact_radius.unwrap_or(RT_NEAR_DISTANCE);
        // Hold the selection center within 0.5m; shader transitions use the
        // same center so they never outrun the currently selected slots.
        if !self.needs_update(placements, viewer, near_capacity, far_capacity) {
            return;
        }
        self.selection_center = viewer;
        let live: HashSet<_> = placements.iter().map(|(index, _)| *index).collect();
        self.cache.retain(|index, _| live.contains(index));
        let mut near = Vec::new();
        for (index, p) in placements {
            // All generated roots lie within radius, even with a noisy edge.
            if p.centre.distance(viewer) > FAR_DISTANCE + p.radius {
                continue;
            }
            let key = placement_key(*index, p);
            let cached = self.cache.entry(*index).or_insert_with(|| CachedPatch {
                key,
                incarnation: {
                    self.next_incarnation += 1;
                    self.next_incarnation
                },
                roots: roots(p),
            });
            if cached.key != key {
                cached.key = key;
                self.next_incarnation += 1;
                cached.incarnation = self.next_incarnation;
                cached.roots = roots(p);
            }
            for &(blade, position) in &cached.roots {
                let distance_squared = position.distance_squared(viewer);
                if distance_squared <= near_distance * near_distance {
                    near.push(Candidate {
                        owner: Owner {
                            patch: *index,
                            blade,
                            incarnation: cached.incarnation,
                        },
                        distance_squared,
                        rank: blade_rank(p, blade),
                        near_overlap: false,
                        far_priority: 0.0,
                    });
                }
            }
        }
        self.candidate_count = near.len();
        bounded(&mut near, near_capacity, near_order);
        // When not saturated retain the full 18m band, avoiding a changing
        // transition whenever the outermost patch streams in or out.
        let end = if near.len() == near_capacity && near_capacity != 0 {
            near.iter()
                .map(|c| c.distance_squared)
                .fold(0.0, f32::max)
                .sqrt()
                .min(near_distance)
        } else if near_capacity == 0 {
            0.0
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
        let selected_near: HashSet<_> = near.iter().map(|c| c.owner).collect();
        let mut far = Vec::new();
        let start_squared = self.near_transition[0] * self.near_transition[0];
        let end_squared = end * end;
        for (index, p) in placements {
            if exact_radius.is_some() || p.centre.distance(viewer) > FAR_DISTANCE + p.radius {
                continue;
            }
            let Some(cached) = self.cache.get(index) else {
                continue;
            };
            for &(blade, position) in &cached.roots {
                let distance_squared = position.distance_squared(viewer);
                if distance_squared < start_squared
                    || distance_squared > FAR_DISTANCE * FAR_DISTANCE
                {
                    continue;
                }
                let rank = blade_rank(p, blade);
                let beyond_near = (distance_squared.sqrt() - end).max(0.0);
                let ramp = (beyond_near / (FAR_DISTANCE - end)).clamp(0.0, 1.0);
                // Full eligibility across the complementary transition;
                // thinning beyond it is never compensated by wider geometry.
                let density = (1.0 - 0.75 * ramp) * (1.0 - 0.5 * ramp);
                if unit(hash(rank ^ 0xa511e9b3)) >= density {
                    continue;
                }
                let owner = Owner {
                    patch: *index,
                    blade,
                    incarnation: cached.incarnation,
                };
                far.push(Candidate {
                    owner,
                    distance_squared,
                    rank,
                    near_overlap: distance_squared <= end_squared && selected_near.contains(&owner),
                    // Exponential weighted selection makes retention approach
                    // 1 continuously at the exact-near boundary. With budget
                    // cutoff T, retention is 1-exp(-T/distance²), rather than
                    // jumping directly to a global far-band keep fraction.
                    // Separate randomness from density and transition gates.
                    far_priority: {
                        // 23-bit bin midpoints are representable strictly
                        // inside (0,1), including both endpoint bins.
                        let u =
                            ((hash(rank ^ 0x68bc21eb) & 0x007f_ffff) as f32 + 0.5) / 8_388_608.0;
                        -(1.0 - u).ln() * beyond_near * beyond_near
                    },
                });
            }
        }
        bounded(&mut far, far_capacity, far_order);
        assign(
            &mut near,
            near_capacity,
            &mut self.near_owners,
            &mut self.near_slots,
            &mut self.near_generation,
        );
        assign(
            &mut far,
            far_capacity,
            &mut self.far_owners,
            &mut self.far_slots,
            &mut self.far_generation,
        );
        self.last_update = Some((
            viewer,
            near_capacity,
            far_capacity,
            placements
                .iter()
                .map(|(index, p)| placement_key(*index, p))
                .collect(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn patch(x: f32, seed: u32) -> PatchPlacement {
        PatchPlacement {
            centre: Vec2::new(x, 0.0),
            radius: 2.0,
            blade_count: 100,
            seed,
            edge_noise_amp: 0.2,
            canopy_density_at_centre: 0.0,
        }
    }
    #[test]
    fn nearest_coverage_capacity_and_retained_slots() {
        let patches = [(0, patch(2.0, 12)), (1, patch(12.0, 13))];
        let mut s = RtSelection::default();
        s.update(&patches, Vec2::ZERO, 50, 40);
        assert_eq!(s.near_slots.len(), 50);
        assert_eq!(s.far_slots.len(), 40);
        assert!(s.near_slots.iter().all(|id| id[0] == 0));
        let old: HashMap<_, _> = s
            .near_slots
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let selected: HashSet<_> = s.near_slots.iter().copied().collect();
        let furthest = s.near_transition[1];
        for b in 0..100 {
            if let Some(r) = root(&patches[0].1, b) {
                if r.length() < furthest {
                    assert!(selected.contains(&[0, b]));
                }
            }
        }
        s.update(&patches, Vec2::new(0.05, 0.0), 50, 40);
        for (i, id) in s.near_slots.iter().enumerate() {
            if let Some(old_i) = old.get(id) {
                assert_eq!(i, *old_i);
            }
        }
    }
    #[test]
    fn ownership_generation_and_determinism() {
        let mut patches = [(0, patch(2.0, 12)), (1, patch(12.0, 13))];
        let mut s = RtSelection::default();
        s.update(&patches, Vec2::ZERO, 50, 40);
        let before = (
            s.near_slots.clone(),
            s.far_slots.clone(),
            s.near_generation,
            s.far_generation,
        );
        patches.reverse();
        s.update(&patches, Vec2::ZERO, 50, 40);
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
        other.update(&patches, Vec2::ZERO, 50, 40);
        assert_eq!(other.near_slots, s.near_slots);
        assert_eq!(other.far_slots, s.far_slots);
        assert_ne!(other.near_generation, s.near_generation);
        assert_ne!(other.far_generation, s.far_generation);
        patches[1].1.seed += 1;
        s.update(&patches, Vec2::ZERO, 50, 40);
        assert!(s.near_generation > before.2);
        let generation = s.near_generation;
        s.update(&[], Vec2::ZERO, 50, 40);
        assert!(s.near_slots.iter().all(|s| *s == INACTIVE));
        assert!(s.near_generation > generation);
    }
    #[test]
    fn far_budget_preserves_complementary_transition() {
        let patches = [
            (0, patch(2.0, 12)),
            (1, patch(22.0, 13)),
            (2, patch(30.0, 14)),
        ];
        let mut s = RtSelection::default();
        s.update(&patches, Vec2::ZERO, 50, 50);
        let far: HashSet<_> = s.far_slots.iter().copied().collect();
        for id in &s.near_slots {
            let r = root(&patches[0].1, id[1]).unwrap();
            if r.length() >= s.near_transition[0] {
                assert!(far.contains(id));
            }
        }
    }
    #[test]
    fn saturated_far_budget_has_continuous_radial_coverage() {
        // Uniform disc with substantially more eligible blades than slots.
        // Retention thins gradually with distance; a global far hash cutoff
        // would drop from complete overlap to sparse coverage in the first
        // bin outside the near boundary.
        let p = PatchPlacement {
            centre: Vec2::ZERO,
            radius: 50.0,
            blade_count: 200_000,
            seed: 741,
            edge_noise_amp: 0.0,
            canopy_density_at_centre: 0.0,
        };
        let mut s = RtSelection::default();
        s.update(&[(0, p)], Vec2::ZERO, 8_000, 20_000);
        let far: HashSet<_> = s.far_slots.iter().copied().collect();
        assert_eq!(far.len(), 20_000);
        assert!(!far.contains(&INACTIVE));
        let end = s.near_transition[1];
        let bounds = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0];
        let mut total = [0usize; 8];
        let mut selected = [0usize; 8];
        for &(blade, position) in &s.cache[&0].roots {
            let delta = position.length() - end;
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
        eprintln!("near end {end:.2}m, far radial retention {rates:?}");
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
        for id in &s.near_slots {
            let r = root(&p, id[1]).unwrap().length();
            if r >= s.near_transition[0] {
                assert!(far.contains(id));
            }
        }
    }
    #[test]
    fn selection_center_hysteresis_and_placement_changes() {
        let mut patches = [(0, patch(2.0, 12))];
        let mut s = RtSelection::default();
        s.update(&patches, Vec2::ZERO, 50, 50);
        let generation = s.near_generation;
        s.update(&patches, Vec2::new(0.4, 0.0), 50, 50);
        assert_eq!(s.selection_center, Vec2::ZERO);
        assert!(!s.needs_update(&patches, Vec2::new(0.4, 0.0), 50, 50));
        assert!(s.needs_update(&patches, Vec2::new(0.6, 0.0), 50, 50));
        assert_eq!(s.near_generation, generation);
        s.update(&patches, Vec2::new(0.6, 0.0), 50, 50);
        assert_eq!(s.selection_center, Vec2::new(0.6, 0.0));
        patches[0].1.seed += 1;
        s.update(&patches, Vec2::new(0.7, 0.0), 50, 50);
        assert_eq!(s.selection_center, Vec2::new(0.7, 0.0));
    }
    #[test]
    fn exact_reference_has_complete_local_coverage_and_no_proxies() {
        let patches = [(0, patch(2.0, 12)), (1, patch(12.0, 13))];
        let mut s = RtSelection::default();
        s.update_mode(&patches, Vec2::ZERO, 200, 200, Some(4.0));
        let expected: HashSet<_> = patches
            .iter()
            .flat_map(|(p, placement)| {
                (0..placement.blade_count).filter_map(move |b| {
                    root(placement, b)
                        .filter(|r| r.length() <= 4.0)
                        .map(|_| [*p, b])
                })
            })
            .collect();
        let actual: HashSet<_> = s
            .near_slots
            .iter()
            .copied()
            .filter(|s| *s != INACTIVE)
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(s.candidate_count, expected.len());
        assert_eq!(s.coverage_radius, 4.0);
        assert!(s.near_transition[0] > 4.0);
        assert!(s.far_slots.iter().all(|s| *s == INACTIVE));
        // Mode and radius changes must bypass the stationary-camera cache.
        s.update_mode(&patches, Vec2::ZERO, 200, 200, Some(1.0));
        assert_eq!(s.coverage_radius, 1.0);
        assert!(s.near_slots.iter().filter(|s| **s != INACTIVE).count() < actual.len());
        s.update(&patches, Vec2::ZERO, 200, 200);
        assert_eq!(s.coverage_radius, 18.0);
        assert!(s.near_transition[0] < 18.0);
    }
    #[test]
    fn exact_reference_reports_capacity_limited_coverage() {
        let p = patch(2.0, 12);
        let mut s = RtSelection::default();
        s.update_mode(&[(0, p)], Vec2::ZERO, 10, 20, Some(4.0));
        assert!(s.candidate_count > 10);
        assert!(s.coverage_radius < 4.0);
        assert_eq!(s.near_slots.len(), 10);
        let selected: HashSet<_> = s.near_slots.iter().copied().collect();
        for b in 0..p.blade_count {
            if root(&p, b).is_some_and(|r| r.length() < s.coverage_radius) {
                assert!(selected.contains(&[0, b]));
            }
        }
        assert!(s.far_slots.iter().all(|s| *s == INACTIVE));
    }

    #[test]
    fn empty_and_zero_capacity() {
        let mut s = RtSelection::default();
        s.update(&[], Vec2::ZERO, 0, 0);
        assert!(s.near_slots.is_empty());
        assert_eq!(s.near_transition, [0.0, 0.0]);
        s.update(&[(0, patch(2.0, 12))], Vec2::ZERO, 0, 0);
        assert!(s.near_slots.is_empty());
        assert!(s.far_slots.is_empty());
    }
}
