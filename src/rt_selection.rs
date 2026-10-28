//! Camera-local raytracing caster selection: the decisions the CPU keeps.
//!
//! The selection itself runs on the GPU (`meadow_compute.wesl`, the
//! `rt_select_*` kernels). It ranks every live blade in reach by a lower
//! bound on its view-weighted distance from any pose within the margins of
//! the selection's ([`RtPose::within`]): the eye distance, stretched for
//! blades outside the view cone, so both bands spend their slots on what the
//! camera sees. It fits the near band's reach and the far band's budget term
//! to their capacities from histograms of those ranks. A selected blade
//! keeps its slot, and a slot whose blade the expansion drew last frame
//! keeps it until the blade stops drawing, so a slot only passes between
//! blades while neither is drawn and casters come and go one at a time.
//!
//! The CPU decides when to reselect: once the viewer has crossed
//! [`MeadowRtMargins::reselect_fraction`] of the margins, or the inputs
//! change. It widens the margins past [`MeadowViewer::caster_margins`] to
//! the viewer's own motion, lists the blades the GPU scans ([`RtScan`]) and
//! the patches whose blades must leave their slots ([`RtSelect::purge`]), and
//! keeps the slots' identity generation.
//!
//! [`MeadowViewer::caster_margins`]: crate::MeadowViewer::caster_margins
use crate::compute::gate_count;
use crate::field::{CoverSurvival, CoverTexel, MeadowGrid};
use crate::material::{ClearingProfiles, PatchTrunkSlot};
use crate::placement::PatchPlacement;
use bevy::math::{Vec2, Vec3};
use bevy::platform::collections::HashMap;
use std::f32::consts::PI;
use std::sync::Arc;

/// Largest view-weighted reach of the exact near band, in metres. Also the
/// largest radius the exact-only reference mode accepts.
pub(crate) const RT_NEAR_DISTANCE: f32 = 18.0;
/// Casters end where raster grass stops casting shadows.
pub(crate) const FAR_DISTANCE: f32 = crate::mesh::SHADOW_MAX_DIST;
/// Height of a blade's root above the heightfield. MUST equal
/// `MEADOW_BLADE_GROUND_LIFT` in `meadow_shared.wesl`.
pub(crate) const BLADE_GROUND_LIFT: f32 = 0.02;
/// MUST equal `RT_FAR_THINNING` in `meadow_compute.wesl`: the far band's
/// keep share before its budget term is `(1 - x * ramp) * (1 - y * ramp)`,
/// `ramp` rising from 0 at the near band's reach to 1 at [`FAR_DISTANCE`].
pub(crate) const FAR_THINNING: [f32; 2] = [0.75, 0.5];
/// Most blades a scan chunk (one workgroup's share of a scan) holds.
pub(crate) const SCAN_CHUNK: u32 = 2048;
/// MUST equal `RT_COVERAGE_NEAR` in `meadow_compute.wesl`: the bit of an
/// [`RtScan::coverage`] entry's end marking a patch the near band scans.
pub(crate) const COVERAGE_NEAR: u32 = 1 << 31;
/// How much of the viewer's largest recent steady motion per frame its
/// margins hold over a frame ([`Motion`]), and the share of it a frame
/// keeps.
const MOTION_SAFETY: f32 = 1.5;
const MOTION_DECAY: f32 = 0.98;
/// The widest margins motion widens the floor to, over the floor. Motion
/// past them is a cut, not motion.
const MOTION_CAP: f32 = 16.0;

static NEXT_IDENTITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_identity() -> u64 {
    NEXT_IDENTITY.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// How far the viewer may move from a raytracing caster selection's pose
/// while it still holds every caster the expansion can draw, and when the
/// next one runs. Wider margins select less exactly; narrower ones select
/// more often. The next selection lands the frame it is due, so a frame's
/// motion must stay within the share of the margins past
/// `reselect_fraction`: each selection widens these margins to the viewer's
/// own recent motion, up to sixteen times.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeadowRtMargins {
    /// Eye travel, in metres.
    pub distance: f32,
    /// View turn plus view-cone change, in radians, up to π/2.
    pub angle: f32,
    /// Share of the margins the viewer crosses before the next selection,
    /// in (0, 0.95].
    pub reselect_fraction: f32,
}

impl Default for MeadowRtMargins {
    fn default() -> Self {
        Self {
            distance: 0.5,
            angle: 8f32.to_radians(),
            reselect_fraction: 0.5,
        }
    }
}

impl MeadowRtMargins {
    /// The margins with their fields in range.
    fn clamped(&self) -> Self {
        Self {
            distance: self.distance.max(0.0),
            angle: self.angle.clamp(0.0, PI / 2.0),
            reselect_fraction: self.reselect_fraction.clamp(1e-3, 0.95),
        }
    }
}

/// One live patch, by its stable index into the variant's patch buffers.
pub(crate) type LivePlacements = Arc<Vec<(u32, PatchPlacement)>>;

fn same_placements(a: &LivePlacements, b: &LivePlacements) -> bool {
    Arc::ptr_eq(a, b) || a == b
}

fn same_arc<T>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    a.as_ref().map(Arc::as_ptr) == b.as_ref().map(Arc::as_ptr)
}

/// The viewer a selection ranks blades for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct RtPose {
    pub(crate) eye: Vec3,
    /// Unit view direction; zero ranks blades by eye distance alone.
    pub(crate) forward: Vec3,
    /// Half-angle, in radians, of the cone around `forward` that holds the
    /// view; 0 ranks blades by eye distance alone.
    pub(crate) cone: f32,
}

impl RtPose {
    fn weighted(&self) -> bool {
        self.cone > 0.0 && self.forward != Vec3::ZERO
    }

    /// The pose ranking blades by eye distance alone.
    pub(crate) fn by_distance(&self) -> Self {
        Self {
            eye: self.eye,
            ..Self::default()
        }
    }

    /// The view direction blades rank by, zero when they rank by eye
    /// distance alone.
    pub(crate) fn ranked_forward(&self) -> Vec3 {
        if self.weighted() {
            self.forward
        } else {
            Vec3::ZERO
        }
    }

    /// The cosine of the view cone's half-angle, kept off -1.
    pub(crate) fn cone_cos(&self) -> f32 {
        self.cone.cos().max(-1.0 + 1e-3)
    }

    /// The view's turn plus its cone's change to `other`.
    fn turn(&self, other: &Self) -> f32 {
        self.forward.dot(other.forward).clamp(-1.0, 1.0).acos() + (self.cone - other.cone).abs()
    }

    /// Whether `other` lies within `fraction` of `margins`: the eye within
    /// that share of their distance, and the view's turn and its cone's
    /// change together within that share of their angle.
    pub(crate) fn within(&self, other: &Self, margins: &MeadowRtMargins, fraction: f32) -> bool {
        if self.weighted() != other.weighted()
            || self.eye.distance_squared(other.eye) > (margins.distance * fraction).powi(2)
        {
            return false;
        }
        !self.weighted() || self.turn(other) <= margins.angle * fraction
    }

    /// A pose at `eye` looking along compass `heading` (degrees clockwise
    /// from -Z), pitched `pitch` degrees, holding a cone of `cone` degrees.
    #[cfg(test)]
    pub(crate) fn looking(eye: Vec3, heading: f32, pitch: f32, cone: f32) -> Self {
        let (h, p) = (heading.to_radians(), pitch.to_radians());
        Self {
            eye,
            forward: Vec3::new(h.sin() * p.cos(), p.sin(), -h.cos() * p.cos()),
            cone: cone.to_radians(),
        }
    }
}

/// MUST equal `RT_OUTSIDE_VIEW_WEIGHT` in `meadow_compute.wesl`: a blade
/// straight behind the view ranks like one this many times nearer inside
/// it.
#[cfg(test)]
pub(crate) const OUTSIDE_VIEW_WEIGHT: f32 = 3.0;
/// MUST equal `RT_UNBOUNDED_BUDGET` in `meadow_compute.wesl`: the far
/// band's budget term standing for "every eligible blade fits".
#[cfg(test)]
pub(crate) const UNBOUNDED_BUDGET: f32 = 1.0e6;

/// CPU reference of the kernels' ranking (`rt_rank_distance` and
/// `rt_least_rank_distance` in `meadow_compute.wesl`).
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct Ranker {
    pose: RtPose,
    margins: MeadowRtMargins,
}

#[cfg(test)]
impl Ranker {
    pub(crate) fn new(pose: &RtPose, margins: &MeadowRtMargins) -> Self {
        Self {
            pose: *pose,
            margins: *margins,
        }
    }

    /// The view weight of a direction at cosine `c` from the view: 1
    /// inside the cone, rising linearly with the cosine outside it to
    /// [`OUTSIDE_VIEW_WEIGHT`] straight behind.
    fn weight(&self, c: f32) -> f32 {
        let cone_cos = self.pose.cone_cos();
        let outside = ((cone_cos - c) / (cone_cos + 1.0)).clamp(0.0, 1.0);
        1.0 + (OUTSIDE_VIEW_WEIGHT - 1.0) * outside
    }

    /// `point`'s eye distance times its view weight.
    pub(crate) fn rank_distance(&self, point: Vec3) -> f32 {
        let dir = point - self.pose.eye;
        let r = dir.length();
        let forward = self.pose.ranked_forward();
        if forward == Vec3::ZERO || r <= 0.0 {
            return r;
        }
        r * self.weight(dir.dot(forward) / r)
    }

    /// A lower bound on [`Self::rank_distance`] of `point` from any pose
    /// within the margins: the eye up to their distance away, the view
    /// turned and its cone widened together by up to their angle. With
    /// `1 + cos x = 2 cos²(x / 2)`, widening the cone by an angle lowers a
    /// blade's weight no more than turning toward it by that angle does, so
    /// the bound turns the view by the whole angle.
    pub(crate) fn least_rank_distance(&self, point: Vec3) -> f32 {
        let dir = point - self.pose.eye;
        let r = dir.length();
        let margin = self.margins.distance;
        let nearest = (r - margin).max(0.0);
        let forward = self.pose.ranked_forward();
        if forward == Vec3::ZERO || nearest <= 0.0 {
            return nearest;
        }
        let c = (dir.dot(forward) / r).clamp(-1.0, 1.0);
        // The direction to `point` turns toward the view by at most the
        // margin angle plus what the eye's travel turns it.
        let x = margin / r;
        let (sin_m, cos_m) = self.margins.angle.sin_cos();
        let y = (1.0 - x * x).sqrt();
        let (sin_t, cos_t) = (sin_m * y + cos_m * x, cos_m * y - sin_m * x);
        let nearest_c = if c >= cos_t {
            1.0
        } else {
            c * cos_t + (1.0 - c * c).sqrt() * sin_t
        };
        nearest * self.weight(nearest_c)
    }
}

/// Everything a selection reads besides the pose. The selection reads the
/// GPU copies; these say when they change.
#[derive(Clone, Default)]
pub(crate) struct RtInputs {
    pub(crate) placements: LivePlacements,
    /// The variant's clearing-disc rows and the clearing profiles they
    /// reference.
    pub(crate) trunks: Arc<Vec<PatchTrunkSlot>>,
    pub(crate) profiles: Arc<ClearingProfiles>,
    pub(crate) cover: Option<Arc<MeadowGrid<CoverTexel>>>,
    pub(crate) survival: CoverSurvival,
    /// Ground heights, for the patches' 3D distances; without them the
    /// distances are planar.
    pub(crate) heights: Option<Arc<MeadowGrid<f32>>>,
}

impl RtInputs {
    /// Whether `other` selects the same blades as `self` for any pose.
    fn same(&self, other: &Self) -> bool {
        same_placements(&self.placements, &other.placements)
            && Arc::ptr_eq(&self.trunks, &other.trunks)
            && Arc::ptr_eq(&self.profiles, &other.profiles)
            && same_arc(&self.cover, &other.cover)
            && same_arc(&self.heights, &other.heights)
            && self.survival == other.survival
    }
}

/// The lowest and highest blade root of each live patch, kept while its
/// placement and the heights stay. A bilinear sample lies within the four
/// texels around it, so the texels under a patch's square, widened to
/// whole texels, bound its ground.
#[derive(Default)]
struct HeightBounds {
    heights: Option<Arc<MeadowGrid<f32>>>,
    by_patch: HashMap<u32, (PatchPlacement, f32, f32)>,
}

impl HeightBounds {
    fn update(&mut self, inputs: &RtInputs) {
        if !same_arc(&self.heights, &inputs.heights) {
            self.heights.clone_from(&inputs.heights);
            self.by_patch.clear();
        }
        let Some(heights) = &self.heights else {
            return;
        };
        for (index, p) in inputs.placements.iter() {
            if self
                .by_patch
                .get(index)
                .is_some_and(|(cached, ..)| cached == p)
            {
                continue;
            }
            let r = Vec2::splat(p.outer_radius());
            let lo = heights.pixel(p.centre - r).floor().as_uvec2();
            let hi = heights
                .pixel(p.centre + r)
                .ceil()
                .as_uvec2()
                .min(heights.dims - 1);
            let (mut low, mut high) = (f32::MAX, f32::MIN);
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let h = heights.texel(x, y);
                    low = low.min(h);
                    high = high.max(h);
                }
            }
            self.by_patch.insert(
                *index,
                (*p, low + BLADE_GROUND_LIFT, high + BLADE_GROUND_LIFT),
            );
        }
        // Patches that left are dropped once they outnumber the live ones.
        if self.by_patch.len() > 2 * inputs.placements.len() {
            let live: HashMap<u32, ()> = inputs.placements.iter().map(|(i, _)| (*i, ())).collect();
            self.by_patch.retain(|index, _| live.contains_key(index));
        }
    }

    /// The least height between `eye_y` and any blade root of patch
    /// `index`; 0 without a heightfield.
    fn gap(&self, index: u32, eye_y: f32) -> f32 {
        self.by_patch.get(&index).map_or(0.0, |&(_, low, high)| {
            (eye_y - high).max(low - eye_y).max(0.0)
        })
    }
}

/// The far band's share of live blades before its budget term at a
/// view-weighted distance `beyond` past the near band's reach `end`
/// (`rt_far_thinning` in `meadow_compute.wesl`).
fn thinning(beyond: f32, end: f32) -> f32 {
    let ramp = (beyond / (FAR_DISTANCE - end).max(1e-3)).clamp(0.0, 1.0);
    (1.0 - FAR_THINNING[0] * ramp) * (1.0 - FAR_THINNING[1] * ramp)
}

/// `rt_far_keep_share` in `meadow_compute.wesl`: the far band's share of
/// live blades kept at view-weighted distance `d`, past the near band's
/// reach `end`, under budget term `budget`. Full up to `end`, thinning with
/// distance; the budget term `1 - exp(-budget / beyond²)` approaches 1
/// continuously at `end`, so coverage has no cliff at the boundary. Rises
/// with `end` and `budget` and falls with `d`.
#[cfg(test)]
pub(crate) fn keep_share(d: f32, end: f32, budget: f32) -> f32 {
    let beyond = (d - end).max(0.0);
    if beyond <= 0.0 {
        return 1.0;
    }
    thinning(beyond, end) * (1.0 - (-budget / (beyond * beyond)).exp())
}

/// `rt_least_budget` in `meadow_compute.wesl`: the least budget term for
/// which [`keep_share`] at `d` exceeds `rank`, infinite where none does.
#[cfg(test)]
pub(crate) fn least_budget(d: f32, end: f32, rank: f32) -> f32 {
    let beyond = (d - end).max(0.0);
    if beyond <= 0.0 {
        return 0.0;
    }
    let share = rank / thinning(beyond, end);
    if share >= 1.0 {
        return f32::INFINITY;
    }
    -beyond * beyond * (-share).ln_1p()
}

/// The blades a selection scans: chunks of each patch's blade indices,
/// those of the patches that can hold a near-band blade first.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RtScan {
    /// `(patch index, first blade, end blade, f32 bits of the least 3D
    /// distance between the eye and the patch's blades)` per chunk.
    pub(crate) chunks: Vec<[u32; 4]>,
    /// How many of `chunks` the near band scans.
    pub(crate) near_chunks: u32,
    /// `(patch index, end blade)` per scanned patch, sorted by index: the
    /// patch's blades below the end are scanned, and [`COVERAGE_NEAR`]
    /// marks a patch the near band scans. Only a blade the coverage holds
    /// keeps a key in the selection, so a resident blade the scan missed
    /// (the GPU's ground lags the CPU's while a new atlas uploads) leaves
    /// its slot instead of escaping the fit.
    pub(crate) coverage: Vec<[u32; 2]>,
}

impl RtScan {
    /// Every patch of `inputs` near enough to `eye` to hold a caster, and
    /// in each the blades the bands can select: all of a patch's blades
    /// within `near_radius` of the margins' nearest eye, and, past it, the
    /// prefix of its blade indices the far band's thinning keeps at its
    /// nearest point for any near reach. `far` is off in the exact-only
    /// reference, which selects no far band. `margin` is the selection's
    /// margin distance; `bounds` holds the patches' ground.
    fn new(
        inputs: &RtInputs,
        bounds: &HeightBounds,
        eye: Vec3,
        near_radius: f32,
        far: bool,
        margin: f32,
    ) -> Self {
        let near_limit = near_radius + margin;
        let limit = if far {
            FAR_DISTANCE + margin
        } else {
            near_limit
        };
        let mut near = Vec::new();
        let mut rest = Vec::new();
        for (index, p) in inputs.placements.iter() {
            let planar = (p.centre.distance(Vec2::new(eye.x, eye.z)) - p.outer_radius()).max(0.0);
            if planar > limit || p.blade_count == 0 {
                continue;
            }
            let nearest = planar.hypot(bounds.gap(*index, eye.y));
            if nearest > limit {
                continue;
            }
            if nearest <= near_limit {
                near.push((*index, p.blade_count, nearest));
                continue;
            }
            // Blades are ranked by index, the far band keeps a blade only
            // while its rank is under the keep share, and keep shares fall
            // with distance and rise with the near reach.
            let beyond = (nearest - margin - RT_NEAR_DISTANCE).max(0.0);
            let end = gate_count(thinning(beyond, RT_NEAR_DISTANCE), p.blade_count);
            if end > 0 {
                rest.push((*index, end, nearest));
            }
        }
        near.sort_unstable_by_key(|(index, ..)| *index);
        rest.sort_unstable_by_key(|(index, ..)| *index);
        let split = |list: &[(u32, u32, f32)], chunks: &mut Vec<[u32; 4]>| {
            for &(index, end, nearest) in list {
                chunks.extend((0..end).step_by(SCAN_CHUNK as usize).map(|start| {
                    [
                        index,
                        start,
                        (start + SCAN_CHUNK).min(end),
                        nearest.to_bits(),
                    ]
                }));
            }
        };
        let mut chunks = Vec::new();
        split(&near, &mut chunks);
        let near_chunks = chunks.len() as u32;
        split(&rest, &mut chunks);
        let mut coverage: Vec<[u32; 2]> = near
            .iter()
            .map(|&(index, end, _)| [index, end | COVERAGE_NEAR])
            .chain(rest.iter().map(|&(index, end, _)| [index, end]))
            .collect();
        coverage.sort_unstable();
        Self {
            chunks,
            near_chunks,
            coverage,
        }
    }

    /// The scanned patch indices, sorted.
    fn patches(&self) -> Vec<u32> {
        self.coverage.iter().map(|c| c[0]).collect()
    }
}

/// One selection for the GPU to make.
#[derive(Clone, Debug)]
pub(crate) struct RtSelect {
    /// Distinguishes the selection in [`RtSelectState`]; never 0.
    pub(crate) serial: u32,
    pub(crate) pose: RtPose,
    /// The poses the selection holds its casters for.
    pub(crate) margins: MeadowRtMargins,
    /// The exact-only reference's radius; that reference selects no far
    /// band.
    pub(crate) exact_radius: Option<f32>,
    /// Vacate every slot before selecting.
    pub(crate) clear: bool,
    /// Patch indices whose blades leave their slots, sorted.
    pub(crate) purge: Vec<u32>,
    pub(crate) scan: RtScan,
}

impl RtSelect {
    /// The near band's largest reach.
    pub(crate) fn near_radius(&self) -> f32 {
        self.exact_radius.unwrap_or(RT_NEAR_DISTANCE)
    }
}

/// A selection's outcome, read back from the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct RtSelectState {
    pub(crate) serial: u32,
    /// The view-weighted distance the near band holds every live blade
    /// within, from any pose the selection holds.
    pub(crate) near_reach: f32,
    /// The largest far-band budget term the far slots hold every kept blade
    /// for, from any pose the selection holds.
    pub(crate) far_budget: f32,
    /// Live blades within the near band's largest reach, those its scans
    /// counted by the chunk included.
    pub(crate) near_candidates: u32,
    pub(crate) near_selected: u32,
    pub(crate) far_selected: u32,
    /// Slots held for blades drawn last frame that the selection left out.
    pub(crate) pinned: u32,
    /// Selected blades no slot was free for.
    pub(crate) dropped: u32,
    /// The reach and budget term the expansion draws with.
    pub(crate) drawn_near_reach: f32,
    pub(crate) drawn_far_budget: f32,
}

impl RtSelectState {
    /// Whether the drawn limits have reached the selection's.
    pub(crate) fn settled(&self) -> bool {
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * b.abs().max(1e-6);
        close(self.drawn_near_reach, self.near_reach)
            && close(self.drawn_far_budget, self.far_budget)
    }
}

/// The viewer's recent motion per frame, and the margins that hold it.
#[derive(Default)]
struct Motion {
    last: Option<RtPose>,
    /// The last frame's travel and turn.
    step: (f32, f32),
    travel: f32,
    turn: f32,
    margins: Option<MeadowRtMargins>,
}

impl Motion {
    /// The margins for a selection at `viewer`: `floor`, widened at once to
    /// hold [`MOTION_SAFETY`] times the viewer's largest recent motion per
    /// frame within the share past the reselect fraction, and narrowed once
    /// they are twice what that motion needs or it needs only the floor.
    /// Motion is steady over two frames: a one-frame jump, like a camera
    /// snapping to a new pose, is a cut.
    fn margins(&mut self, viewer: &RtPose, floor: &MeadowRtMargins) -> MeadowRtMargins {
        let floor = floor.clamped();
        let widest = (
            floor.distance * MOTION_CAP,
            (floor.angle * MOTION_CAP).min(PI / 2.0),
        );
        if let Some(last) = self.last {
            let step = (last.eye.distance(viewer.eye), last.turn(viewer));
            let (travel, turn) = (step.0.min(self.step.0), step.1.min(self.step.1));
            self.step = step;
            if travel <= widest.0 {
                self.travel = travel.max(self.travel * MOTION_DECAY);
            }
            if turn <= widest.1 {
                self.turn = turn.max(self.turn * MOTION_DECAY);
            }
        }
        self.last = Some(*viewer);
        let spare = MOTION_SAFETY / (1.0 - floor.reselect_fraction);
        let need = (
            (self.travel * spare).clamp(floor.distance, widest.0),
            (self.turn * spare).clamp(floor.angle, widest.1),
        );
        let margins = self.margins.get_or_insert(floor);
        let follow = |held: &mut f32, need: f32, floor: f32| {
            if need > *held || 2.0 * need < *held || need <= floor {
                *held = need;
            }
        };
        follow(&mut margins.distance, need.0, floor.distance);
        follow(&mut margins.angle, need.1, floor.angle);
        margins.reselect_fraction = floor.reselect_fraction;
        *margins
    }
}

/// The selection a variant's slots hold, as planned.
struct Planned {
    pose: RtPose,
    margins: MeadowRtMargins,
    inputs: RtInputs,
    /// The patches its scan visited, sorted.
    patches: Vec<u32>,
    /// Whether it was made only to clean up after the selection before it.
    cleanup: bool,
}

/// One variant's selection bookkeeping: when to reselect, which patches
/// leave, and when slots may pass from a drawn blade to another.
pub(crate) struct RtSelector {
    planned: Option<Planned>,
    /// The patches the selection before `planned` scanned, sorted: with
    /// `planned`'s, every patch that can own a slot.
    earlier_patches: Vec<u32>,
    serial: u32,
    exact_radius: Option<f32>,
    motion: Motion,
    bounds: HeightBounds,
    /// Changes when a slot may pass from a drawn blade to another one.
    pub(crate) identity: u64,
    /// How many times `identity` changed.
    pub(crate) identity_changes: u64,
    /// The selection state last read back.
    pub(crate) state: Option<RtSelectState>,
}

impl Default for RtSelector {
    fn default() -> Self {
        Self {
            planned: None,
            earlier_patches: Vec::new(),
            serial: 0,
            exact_radius: None,
            motion: Motion::default(),
            bounds: HeightBounds::default(),
            identity: next_identity(),
            identity_changes: 0,
            state: None,
        }
    }
}

impl RtSelector {
    /// The pose of the selection the slots hold.
    pub(crate) fn pose(&self) -> Option<RtPose> {
        self.planned.as_ref().map(|p| p.pose)
    }

    /// Whether the state last read back is the latest selection's.
    pub(crate) fn current(&self) -> bool {
        self.state.is_some_and(|s| s.serial == self.serial)
    }

    fn renew_identity(&mut self) {
        self.identity = next_identity();
        self.identity_changes += 1;
    }

    /// The selection due this frame for `inputs` seen from `viewer`, if
    /// any, for the caller to record this frame. `exact_radius` switches on
    /// the exact-only reference, which selects every live blade within that
    /// radius by eye distance alone. `floor` is the narrowest the margins
    /// get.
    pub(crate) fn update(
        &mut self,
        inputs: &RtInputs,
        viewer: RtPose,
        exact_radius: Option<f32>,
        floor: &MeadowRtMargins,
    ) -> Option<RtSelect> {
        let margins = self.motion.margins(&viewer, floor);
        let mut clear = self.serial == 0;
        if exact_radius != self.exact_radius {
            self.exact_radius = exact_radius;
            self.planned = None;
            clear = true;
        }
        if clear {
            self.renew_identity();
        }
        // A patch that left, or changed under its index, takes its blades
        // with it. Slots that held them pass to other blades, which may
        // happen the frame they were drawn.
        let mut purge = Vec::new();
        if let Some(planned) = &self.planned
            && !same_placements(&planned.inputs.placements, &inputs.placements)
        {
            let live: HashMap<u32, PatchPlacement> = inputs.placements.iter().copied().collect();
            purge.extend(
                planned
                    .inputs
                    .placements
                    .iter()
                    .filter(|(index, p)| live.get(index) != Some(p))
                    .map(|(index, _)| *index),
            );
            if purge.iter().any(|index| {
                planned.patches.binary_search(index).is_ok()
                    || self.earlier_patches.binary_search(index).is_ok()
            }) {
                self.renew_identity();
            }
        }
        let pose = match exact_radius {
            Some(_) => viewer.by_distance(),
            None => viewer,
        };
        let due = clear
            || !purge.is_empty()
            || self.planned.as_ref().is_none_or(|planned| {
                !planned
                    .pose
                    .within(&pose, &planned.margins, planned.margins.reselect_fraction)
                    || !planned.inputs.same(inputs)
            });
        // Once the drawn limits settle, a selection that had to keep slots
        // for blades drawn past its limits, or found no slot for some, is
        // made again to free them.
        let cleanup = self.planned.as_ref().is_some_and(|planned| {
            !planned.cleanup
                && self.current()
                && self
                    .state
                    .is_some_and(|s| (s.pinned > 0 || s.dropped > 0) && s.settled())
        });
        if !due && !cleanup {
            return None;
        }
        self.serial = self.serial.wrapping_add(1).max(1);
        self.bounds.update(inputs);
        let scan = RtScan::new(
            inputs,
            &self.bounds,
            pose.eye,
            exact_radius.unwrap_or(RT_NEAR_DISTANCE),
            exact_radius.is_none(),
            margins.distance,
        );
        purge.sort_unstable();
        purge.dedup();
        let earlier = self.planned.replace(Planned {
            pose,
            margins,
            inputs: inputs.clone(),
            patches: scan.patches(),
            cleanup: !due,
        });
        self.earlier_patches = earlier.map(|p| p.patches).unwrap_or_default();
        Some(RtSelect {
            serial: self.serial,
            pose,
            margins,
            exact_radius,
            clear,
            purge,
            scan,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::hash_u32;
    use crate::placement::PatchShape;
    use bevy::math::{Quat, UVec2};

    fn m() -> MeadowRtMargins {
        MeadowRtMargins::default()
    }
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
    fn scan(inputs: &RtInputs, eye: Vec3, near_radius: f32, far: bool) -> RtScan {
        let mut bounds = HeightBounds::default();
        bounds.update(inputs);
        RtScan::new(inputs, &bounds, eye, near_radius, far, m().distance)
    }

    /// The bound selections rank by is below the view-weighted distance
    /// from every pose within the margins, the cone's change included.
    #[test]
    fn least_rank_distance_bounds_every_held_pose() {
        let pose = RtPose::looking(Vec3::new(0.0, 1.7, 0.0), 0.0, -12.0, 40.0);
        let ranker = Ranker::new(&pose, &m());
        let mut seed = 1u32;
        let mut next = || {
            seed = hash_u32(seed);
            (seed & 0x00ff_ffff) as f32 / 16_777_216.0
        };
        let mut checked = 0;
        for _ in 0..40_000 {
            let point = Vec3::new(next() * 30.0 - 15.0, 0.0, next() * 30.0 - 15.0);
            let offset = Vec3::new(next() - 0.5, next() - 0.5, next() - 0.5).normalize_or_zero()
                * next()
                * m().distance;
            let axis = Vec3::new(next() - 0.5, next() - 0.5, next() - 0.5).normalize_or(Vec3::Y);
            let turn = next() * m().angle;
            let widen = (next() * 2.0 - 1.0) * (m().angle - turn);
            let other = RtPose {
                eye: pose.eye + offset,
                forward: Quat::from_axis_angle(axis, turn) * pose.forward,
                cone: pose.cone + widen,
            };
            if !pose.within(&other, &m(), 1.0) {
                continue;
            }
            checked += 1;
            assert!(
                ranker.least_rank_distance(point)
                    <= Ranker::new(&other, &m()).rank_distance(point) + 1e-4,
                "{point:?} from {other:?}"
            );
        }
        assert!(checked > 10_000, "{checked}");
    }

    #[test]
    fn least_budget_inverts_the_keep_share() {
        for (d, end, rank) in [(6.0, 3.0, 0.2), (20.0, 4.0, 0.01), (40.0, 2.0, 0.05)] {
            let budget = least_budget(d, end, rank);
            assert!((keep_share(d, end, budget) - rank).abs() < 1e-4);
        }
        assert_eq!(least_budget(2.0, 3.0, 0.9), 0.0);
        assert_eq!(least_budget(45.0, 3.0, 0.9), f32::INFINITY);
    }

    /// A cone change counts against the angular margin like a turn, so a
    /// field-of-view animation within it reselects only once it adds up.
    #[test]
    fn cone_changes_share_the_angular_margin() {
        let pose = RtPose::looking(Vec3::ZERO, 90.0, 0.0, 40.0);
        let zoomed = |cone: f32, turn: f32| RtPose {
            forward: Quat::from_rotation_y(turn.to_radians()) * pose.forward,
            cone: pose.cone + cone.to_radians(),
            ..pose
        };
        let fraction = m().reselect_fraction;
        let half = fraction * m().angle.to_degrees();
        for (cone, turn, held) in [
            (0.9, 0.0, true),
            (-0.9, 0.0, true),
            (0.6, 0.6, false),
            (1.1, 0.0, false),
        ] {
            assert_eq!(
                pose.within(&zoomed(cone * half, turn * half), &m(), fraction),
                held,
                "{cone} {turn}"
            );
        }
    }

    /// The scan holds every patch within reach, near-band patches first and
    /// whole, and past the near band only the prefix the thinning keeps.
    #[test]
    fn scan_lists_patches_in_reach() {
        let mut list: Vec<_> = (0..14u32)
            .map(|i| {
                (
                    i * 2 + 1,
                    PatchPlacement::tile(Vec2::new(i as f32 * 8.0, 0.0), 4.0, 5000, i, 1.0, 0),
                )
            })
            .collect();
        list.push((99, PatchPlacement::tile(Vec2::ZERO, 4.0, 0, 9, 1.0, 0)));
        let field = inputs(&list);
        let listed = scan(&field, Vec3::new(0.0, 1.5, 0.0), RT_NEAR_DISTANCE, true);
        let covered = |index: u32| -> u32 {
            listed
                .chunks
                .iter()
                .filter(|c| c[0] == index)
                .map(|c| c[2] - c[1])
                .sum()
        };
        let near: Vec<u32> = listed.chunks[..listed.near_chunks as usize]
            .iter()
            .map(|c| c[0])
            .collect();
        for (index, p) in &list {
            let gap = (p.centre.length() - p.outer_radius()).max(0.0);
            let blades = covered(*index);
            if p.blade_count == 0 || gap > FAR_DISTANCE + m().distance {
                assert_eq!(blades, 0, "{index}");
            } else if gap <= RT_NEAR_DISTANCE + m().distance {
                assert_eq!(blades, p.blade_count, "{index}");
                assert!(near.contains(index));
            } else {
                assert!(blades > 0 && blades < p.blade_count, "{index}: {blades}");
                assert!(!near.contains(index));
                // The prefix bounds every rank the far band can keep.
                let d = gap - m().distance;
                for end in [0.5, 6.0, RT_NEAR_DISTANCE] {
                    let keep = keep_share(d, end, UNBOUNDED_BUDGET);
                    assert!(keep * p.blade_count as f32 <= blades as f32, "{index}");
                }
            }
        }
        // Chunks tile each listed range without overlap.
        for c in &listed.chunks {
            assert!(c[1] < c[2] && c[2] - c[1] <= SCAN_CHUNK);
        }
        // The coverage lists every chunked range, near patches marked.
        for &[index, end] in &listed.coverage {
            assert_eq!(covered(index), end & !COVERAGE_NEAR, "{index}");
            assert_eq!(end & COVERAGE_NEAR != 0, near.contains(&index), "{index}");
        }
        assert_eq!(listed.patches(), {
            let mut p: Vec<u32> = listed.chunks.iter().map(|c| c[0]).collect();
            p.sort_unstable();
            p.dedup();
            p
        });
        // An exact-only scan lists only the near band.
        let exact = scan(&field, Vec3::new(0.0, 1.5, 0.0), 4.0, false);
        assert_eq!(exact.near_chunks as usize, exact.chunks.len());
        assert!(exact.coverage.len() < listed.coverage.len());
    }

    /// A high eye scans by 3D distance.
    #[test]
    fn scan_measures_height() {
        let heights = MeadowGrid::new(
            Vec2::splat(-50.0),
            Vec2::splat(50.0),
            UVec2::new(2, 2),
            vec![0.0; 4],
        );
        let mut field = inputs(&[(0, patch(0.0, 5))]);
        field.heights = Some(Arc::new(heights));
        let near = scan(&field, Vec3::new(0.0, 30.0, 0.0), RT_NEAR_DISTANCE, true);
        assert_eq!((near.near_chunks, near.chunks.len()), (0, 1));
        let gone = scan(&field, Vec3::new(0.0, 60.0, 0.0), RT_NEAR_DISTANCE, true);
        assert!(gone.chunks.is_empty());
    }

    /// The height gap bounds every blade root's height under the patch,
    /// relief between the samples included, and follows new heights.
    #[test]
    fn height_gap_bounds_relief_within_a_patch() {
        let dims = UVec2::splat(65);
        let grid = |up: f32| {
            let texels = (0..dims.x * dims.y)
                .map(
                    |i| match ((i % dims.x) as i32 - 32, (i / dims.x) as i32 - 32) {
                        (2, -1) => up,
                        (-1, 2) => -2.5,
                        _ => 0.0,
                    },
                )
                .collect();
            Arc::new(MeadowGrid::new(
                Vec2::splat(-32.0),
                Vec2::splat(32.0),
                dims,
                texels,
            ))
        };
        let mut field = inputs(&[(0, PatchPlacement::tile(Vec2::ZERO, 4.0, 1, 1, 1.0, 0))]);
        field.heights = Some(grid(3.0));
        let mut bounds = HeightBounds::default();
        bounds.update(&field);
        let p = field.placements[0].1;
        let heights = field.heights.clone().unwrap();
        for eye_y in [-6.0, -1.0, 0.5, 1.7, 4.0, 9.0] {
            let gap = bounds.gap(0, eye_y);
            for i in 0..=40 {
                for j in 0..=40 {
                    let root = p.centre + (Vec2::new(i as f32, j as f32) / 20.0 - 1.0) * p.radius;
                    let y = heights.sample(root) + BLADE_GROUND_LIFT;
                    assert!(
                        (eye_y - y).abs() >= gap - 1e-5,
                        "eye {eye_y}: {root:?} at {y}"
                    );
                }
            }
        }
        // The spike sits where corner samples miss it.
        let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(close(bounds.gap(0, 4.0), 4.0 - 3.0 - BLADE_GROUND_LIFT));
        field.heights = Some(grid(3.5));
        bounds.update(&field);
        assert!(close(bounds.gap(0, 4.0), 4.0 - 3.5 - BLADE_GROUND_LIFT));
    }

    /// Selections fall due on the reselect fraction of the margins and on
    /// input changes; the identity holds through viewer motion and patches
    /// streaming in, and changes when a patch in reach leaves or the mode
    /// changes.
    #[test]
    fn selections_fall_due_and_identity_holds() {
        let mut list = vec![(0, patch(2.0, 12)), (1, patch(6.0, 13))];
        let field = inputs(&list);
        let mut s = RtSelector::default();
        let pose = RtPose::looking(Vec3::ZERO, 90.0, 0.0, 40.0);
        let first = s.update(&field, pose, None, &m()).expect("first selection");
        assert!(first.clear && first.exact_radius.is_none() && first.purge.is_empty());
        let identity = s.identity;
        assert!(s.update(&field, pose, None, &m()).is_none());
        let moved = |x: f32, turn: f32| RtPose {
            eye: Vec3::new(x, 0.0, 0.0),
            forward: Quat::from_rotation_y(turn.to_radians()) * pose.forward,
            ..pose
        };
        let half = m().reselect_fraction * m().distance;
        assert!(
            s.update(&field, moved(0.8 * half, 3.0), None, &m())
                .is_none()
        );
        assert!(
            !s.update(&field, moved(1.2 * half, 0.0), None, &m())
                .unwrap()
                .clear
        );
        assert!(
            s.update(&field, moved(1.2 * half, 5.0), None, &m())
                .is_some()
        );
        assert_eq!(s.identity, identity);
        // A patch streaming in reselects without a purge.
        list.push((2, patch(30.0, 14)));
        let streamed = s
            .update(&inputs(&list), s.pose().unwrap(), None, &m())
            .unwrap();
        assert!(streamed.purge.is_empty());
        assert_eq!(s.identity, identity);
        // A far patch out of reach leaving purges without a new identity.
        list.push((3, patch(500.0, 15)));
        s.update(&inputs(&list), s.pose().unwrap(), None, &m())
            .unwrap();
        list.pop();
        let left = s
            .update(&inputs(&list), s.pose().unwrap(), None, &m())
            .unwrap();
        assert_eq!(left.purge, vec![3]);
        assert_eq!(s.identity, identity);
        // One in reach leaving, or changing in place, is a new identity.
        list[0].1.seed += 1;
        let replaced = s
            .update(&inputs(&list), s.pose().unwrap(), None, &m())
            .unwrap();
        assert_eq!(replaced.purge, vec![0]);
        assert_ne!(s.identity, identity);
        // The exact-only mode clears and selects only the near band.
        let identity = s.identity;
        let exact = s.update(&inputs(&list), pose, Some(4.0), &m()).unwrap();
        assert!(exact.clear && exact.pose.forward == Vec3::ZERO);
        assert_ne!(s.identity, identity);
        assert_eq!(exact.near_radius(), 4.0);
    }

    /// Once the drawn limits settle, a selection that kept slots for
    /// blades past its limits is made again, once.
    #[test]
    fn pinned_slots_clean_up_once_settled() {
        let field = inputs(&[(0, patch(2.0, 12))]);
        let mut s = RtSelector::default();
        let pose = RtPose::default();
        let first = s.update(&field, pose, None, &m()).unwrap();
        let state = RtSelectState {
            serial: first.serial,
            near_reach: 4.0,
            far_budget: 10.0,
            pinned: 12,
            drawn_near_reach: 4.5,
            drawn_far_budget: 10.0,
            ..Default::default()
        };
        s.state = Some(state);
        assert!(s.update(&field, pose, None, &m()).is_none());
        s.state = Some(RtSelectState {
            drawn_near_reach: 4.0,
            ..state
        });
        let cleanup = s.update(&field, pose, None, &m()).expect("cleanup");
        s.state = Some(RtSelectState {
            serial: cleanup.serial,
            drawn_near_reach: 4.0,
            ..state
        });
        assert!(s.update(&field, pose, None, &m()).is_none());
    }

    /// Fast steady motion widens the margins to hold a frame's travel and
    /// turn, so selections stay frames apart; a cut leaves them, and they
    /// narrow back once the motion stops.
    #[test]
    fn margins_follow_the_viewer() {
        let field = inputs(&[(0, patch(2.0, 12))]);
        let mut s = RtSelector::default();
        let mut selections = 0;
        // 30 m/s and 200°/s at 60 fps: 0.5 m and 3.3° a frame.
        let pose =
            |frame: f32| RtPose::looking(Vec3::new(0.5 * frame, 1.7, 0.0), 3.3 * frame, 0.0, 40.0);
        for frame in 0..120 {
            selections += usize::from(s.update(&field, pose(frame as f32), None, &m()).is_some());
        }
        let wide = s.motion.margins.unwrap();
        let spare = 1.0 - wide.reselect_fraction;
        assert!(wide.distance >= 0.5 / spare, "{wide:?}");
        assert!(wide.angle >= 3.3f32.to_radians() / spare, "{wide:?}");
        // At most every other frame, past the first frame's floor margins.
        assert!(selections <= 120 / 2 + 1, "{selections}");
        // A cut leaves them.
        let cut = RtPose {
            eye: Vec3::new(5000.0, 1.7, 0.0),
            ..pose(119.0)
        };
        s.update(&field, cut, None, &m());
        assert_eq!(s.motion.margins.unwrap(), wide);
        for _ in 0..400 {
            s.update(&field, cut, None, &m());
        }
        assert_eq!(s.motion.margins.unwrap(), m());
        // A snap within the widest margins, then standing still, leaves them
        // too.
        let snap = RtPose::looking(cut.eye + Vec3::new(3.0, 0.0, 0.0), 45.0, 0.0, 40.0);
        for _ in 0..3 {
            s.update(&field, snap, None, &m());
        }
        assert_eq!(s.motion.margins.unwrap(), m());
    }
}
