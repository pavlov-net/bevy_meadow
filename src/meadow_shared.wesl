// Meadow shared shader library, imported as `bevy_meadow::meadow_shared`
// by the compute kernel (`meadow_compute.wesl`), the raster passthrough
// shader (`meadow.wesl`), the mesh-shader path (`meadow_mesh*.wesl`) and the
// Solari receiver pass. It holds the buffer structs, the blade geometry and
// wind helpers, the hashes and per-blade palette variance both derivation
// paths share (so they derive bit-identical blades), the blade shading
// ramp, and the G-buffer helpers. Binding-free: each caller binds its
// buffers at its own slots and passes the values in.
//
// Struct layouts here are the contract with the Rust side
// (`material.rs::VariantParams`, `material.rs::PaletteParams`,
// `compute.rs`); std430 offsets are load-bearing.

// Capacity of a variant's palette table (`array<PaletteParams, 64>` in its
// own storage buffer). MUST equal `material.rs::MAX_MEADOW_PALETTES`.
const MEADOW_MAX_PALETTES: u32 = 64u;

// Mirror of `material.rs::PaletteParams` (3 × vec4 = 48 B): one entry of a
// variant's palette table, which each patch indexes by bits 8..15 of its
// flags (`meadow_patch_palette`). The CPU packs it with the amounts
// clamped and the dead-blade thresholds precomputed
// (`plugin.rs::MeadowPalette::params`).
struct PaletteParams {
    // rgb = live blade albedo (linear), w = per-blade colour jitter.
    live_jitter: vec4<f32>,
    // rgb = dry blade albedo (linear), w = single-dead-blade threshold: a
    // blade whose random falls under it is dead.
    dry_blade_share: vec4<f32>,
    // x = height variance, y = dead-patch noise frequency (1 / patch size in
    // metres, 0 = no patches), z = dead-patch share at neutral ground
    // dryness, w = widest soft patch edge, in the uniform patch noise.
    variance: vec4<f32>,
}

// Mirror of `material.rs::VariantParams` (13 × vec4 = 208 B). Bound at
// `@binding(100)` in raster (material group) and `@binding(0)` in
// compute.
struct VariantParams {
    height_range: vec4<f32>,
    width_range: vec4<f32>,
    wind: vec4<f32>,
    wind_direction: vec4<f32>,
    wind_state: vec4<f32>,
    lod: vec4<f32>,
    density: vec4<f32>,
    heightfield_extents: vec4<f32>,
    // xy = player eye XZ, z = player eye Y.
    viewer_world_xz: vec4<f32>,
    tuft: vec4<f32>,
    viewer_forward: vec4<f32>,
    // World-space extent of the cover field atlas: xy = min XZ, zw = max XZ.
    cover_extents: vec4<f32>,
    // x = 1 / survival noise scale (metres), y = clumpiness, z = largest
    // height scale in the field (at least 1), w = unused.
    cover: vec4<f32>,
}

// Transient (not a buffer layout) — function return/param only.
struct BladeAttrs {
    world_xz: vec2<f32>,
    yaw: f32,
    height: f32,
    width: f32,
    clump_seed: f32,
    rim_factor: f32,
    // Per-blade hash seed the palette variance (`meadow_blade_*`) draws from.
    seed: u32,
}
// Compute → raster, 48 B std430. The compute kernel derives + culls +
// compacts these once per frame; the passthrough VS reads one per
// blade instance and expands the 11 local verts. Only wind is
// recomputed in the VS (cheap; depends on time + per-vertex y_norm).
struct CompactedBladeRecord {
    world_root_xz: vec2<f32>,   // @0
    ground_y: f32,              // @8
    height: f32,                // @12
    width: f32,                 // @16
    yaw: f32,                   // @20
    clump_seed: f32,            // @24
    collapse_t: f32,            // @28  visibility*rim*alive, frozen for the frame
    prev_root_xz: vec2<f32>,    // @32  previous-frame root XZ (motion vector)
    prev_ground_y: f32,         // @40
    albedo: u32,                // @44 -> 48  `meadow_pack_albedo`; 0 in shadow views
}

// ---------- blade template ----------
//
// Vertex `vert_idx ∈ [0, 11)`: layer `i = 0..4` is the pair (2i, 2i+1) at
// length fraction `0.2 i`, left then right; idx 10 is the tip at length
// fraction 1. The local frame is x = across the blade, y = up, z = the
// direction the blade leans. Pure functions of the blade's height, width
// and yaw, so every path that expands a blade (raster, mesh shader,
// raytracing casters) builds the same surface.

// Arc length of a blade over its `height`: the bend brings a typical tip
// back to about that height. MUST equal `mesh.rs::BLADE_LENGTH_PER_HEIGHT`.
const MEADOW_BLADE_LENGTH: f32 = 1.1;

// Width scale of the base-to-tip chord triangle that conserves the
// template's silhouette area: the ribbon's area over the chord's
// (0.596 / 0.5) for `meadow_blade_taper` sampled at the five layers.
const MEADOW_CHORD_WIDTH_SCALE: f32 = 1.19;

// Tilt of the shading normal toward the blade edges, as the tangent of the
// tilt angle at the edge: the blade reads as slightly folded along its
// midrib, so its two halves catch the light differently.
const MEADOW_BLADE_ROUNDING: f32 = 0.6;

// Length fraction of a template vertex: 0 at the root, 1 at the tip.
fn blade_length_fraction(vert_idx: u32) -> f32 {
    if (vert_idx == 10u) {
        return 1.0;
    }
    return f32(vert_idx >> 1u) * 0.2;
}

// Template side of a vertex: -1 left edge, +1 right edge, 0 at the tip.
fn blade_side(vert_idx: u32) -> f32 {
    if (vert_idx == 10u) {
        return 0.0;
    }
    return f32(i32(vert_idx & 1u) * 2 - 1);
}

// Half-width factor along the blade: near-parallel sides low down, then a
// straight run to a sharp tip.
fn meadow_blade_taper(t: f32) -> f32 {
    return (1.0 - t) * (1.0 + 0.6 * t);
}

// The blade's lean as (angle from vertical at the root, further bend to
// the tip), in radians. Drawn from the fractional digits of the yaw, a
// hash every path carries. Most blades stand with a soft bend; about one
// in five lies well over.
fn meadow_blade_lean(blade: BladeAttrs) -> vec2<f32> {
    let u_bend = fract(blade.yaw * 1.4323945);
    let u_root = fract(blade.yaw * 4.7746483);
    let root = 0.08 + 0.32 * u_root;
    let u2 = u_bend * u_bend;
    let bend = 0.3 + 0.6 * u_bend + 0.9 * u2 * u2;
    return vec2<f32>(root, bend);
}

// Local-blade vertex position. The centreline is a constant-curvature arc
// of length `MEADOW_BLADE_LENGTH * height` that leaves the ground at the
// lean's root angle and turns by its bend toward the tip.
fn local_blade_position(vert_idx: u32, blade: BladeAttrs) -> vec3<f32> {
    let t = blade_length_fraction(vert_idx);
    let lean = meadow_blade_lean(blade);
    let length = MEADOW_BLADE_LENGTH * blade.height;
    // Integral of (sin a, cos a) along the arc; the bend is at least 0.3.
    let a = lean.x + lean.y * t;
    let up = length * (sin(a) - sin(lean.x)) / lean.y;
    let forward = length * (cos(lean.x) - cos(a)) / lean.y;
    let half_w = 0.5 * meadow_blade_taper(t) * blade.width;
    return vec3<f32>(blade_side(vert_idx) * half_w, up, forward);
}

// World-space normal of the blade's front face (the template triangles'
// counter-clockwise side) at a vertex, tilted toward its edge by
// `MEADOW_BLADE_ROUNDING`. `cy`/`sy` are the yaw's cosine and sine, as in
// the vertex expansion; `wind_full` is the tip's wind displacement, a
// shear along the length that leans the surface with it. Unnormalized, so
// interpolation rounds it between the edges.
fn meadow_blade_world_normal(
    vert_idx: u32,
    blade: BladeAttrs,
    cy: f32,
    sy: f32,
    wind_full: vec2<f32>,
) -> vec3<f32> {
    let lean = meadow_blade_lean(blade);
    let a = lean.x + lean.y * blade_length_fraction(vert_idx);
    let length = MEADOW_BLADE_LENGTH * blade.height;
    let across = vec3<f32>(cy, 0.0, -sy);
    let along = vec3<f32>(
        sy * length * sin(a) + wind_full.x,
        length * cos(a),
        cy * length * sin(a) + wind_full.y,
    );
    let face = normalize(cross(across, along));
    return face + across * (blade_side(vert_idx) * MEADOW_BLADE_ROUNDING);
}

// Weight of the blade's own normal in its shading normal at view depth
// `depth`: 1 near, falling to 0 (the upright normal) where blades shrink
// toward a pixel and per-blade normals would shimmer. It ends where the
// Solari casters start to fade (`RT_SHADOW_FADE_START`), before the near
// band thins, so the shading, shadow and density changes take separate
// bands.
const MEADOW_BLADE_NORMAL_FADE: vec2<f32> = vec2<f32>(15.0, 35.0);

fn meadow_blade_normal_weight(depth: f32) -> f32 {
    return 1.0 - smoothstep(MEADOW_BLADE_NORMAL_FADE.x, MEADOW_BLADE_NORMAL_FADE.y, depth);
}

// Template vertex `vert_idx` of a near-band blade, or of a far-band tuft
// (`tuft`), in the local frame; w is the shade/wind ramp `y_norm`: the
// length fraction up a blade, the height fraction up a tuft arm.
fn meadow_template_vertex(vert_idx: u32, tuft: bool, blade: BladeAttrs) -> vec4<f32> {
    if (tuft) {
        let local = tuft_local_position(vert_idx, blade);
        return vec4<f32>(local, local.y / max(blade.height, 1e-3));
    }
    return vec4<f32>(local_blade_position(vert_idx, blade), blade_length_fraction(vert_idx));
}

// World position of a template vertex (`meadow_template_vertex`): the local
// frame turned by the yaw's cosine/sine `cy`/`sy`, the tip's wind
// displacement `wind_full` scaled by `y_norm`, and the blade collapsed
// toward its root by `collapse_t` (0 puts every vertex at the root, so the
// blade's triangles are degenerate).
fn meadow_place_vertex(
    local_vertex: vec4<f32>,
    root_xz: vec2<f32>,
    ground_y: f32,
    cy: f32,
    sy: f32,
    wind_full: vec2<f32>,
    collapse_t: f32,
) -> vec3<f32> {
    let local = local_vertex.xyz;
    let rotated = vec2<f32>(cy * local.x + sy * local.z, -sy * local.x + cy * local.z);
    let xz = mix(root_xz, root_xz + rotated + wind_full * local_vertex.w, collapse_t);
    let y = mix(ground_y, ground_y + local.y, collapse_t);
    return vec3<f32>(xz.x, y, xz.y);
}

// Shading-normal varying of a template vertex at view depth `depth`: xyz
// the blade's own normal (`meadow_blade_world_normal`, with the tip wind
// `wind_full` it sways by), w its weight (`meadow_blade_normal_weight`).
// Tufts, and blades past the fade, carry weight 0 and skip the normal.
fn meadow_blade_normal_varying(
    vert_idx: u32,
    tuft: bool,
    blade: BladeAttrs,
    cy: f32,
    sy: f32,
    wind_full: vec2<f32>,
    depth: f32,
) -> vec4<f32> {
    var weight = 0.0;
    if (!tuft) {
        weight = meadow_blade_normal_weight(depth);
    }
    if (weight <= 0.0) {
        return vec4<f32>(0.0, 1.0, 0.0, 0.0);
    }
    return vec4<f32>(meadow_blade_world_normal(vert_idx, blade, cy, sy, wind_full), weight);
}

// Local-tuft vertex computation (far LOD band). `vert_idx ∈ [0, 21)`
// demuxes to `(arm = idx/3, corner = idx%3)`: 7 mini-blade arms fanned
// around one root, each a single triangle (corner 0/1 = base edge,
// 2 = tip). Each arm tilts outward (~25°) at its baked yaw, so the clump
// reads from any azimuth AND straight down — no camera-facing rotation,
// hence no billboard popping. The per-clump `blade.yaw` is applied on top
// by the caller (same as the blade). Pure — depends only on the record's
// height/width.
fn tuft_local_position(vert_idx: u32, blade: BladeAttrs) -> vec3<f32> {
    let arm = vert_idx / 3u;
    let corner = vert_idx % 3u;
    let arm_yaw = f32(arm) * (6.2831853 / 7.0);
    let ca = cos(arm_yaw);
    let sa = sin(arm_yaw);
    // Arm in its own frame: x = sideways, y = up, z = outward.
    let half_w = 0.6 * blade.width; // tufts a touch wider than a blade
    let h = blade.height;
    let tip_out = 0.45 * h; // ~25° outward tilt of the tip
    var local: vec3<f32>;
    if (corner == 2u) {
        local = vec3<f32>(0.0, h, tip_out); // tip: up + outward
    } else {
        let s = f32(i32(corner) * 2 - 1); // corner 0 → -1, 1 → +1
        local = vec3<f32>(s * half_w, 0.0, 0.0); // base edge at the root
    }
    // Rotate the arm by its baked yaw around +Y.
    let rx = ca * local.x + sa * local.z;
    let rz = -sa * local.x + ca * local.z;
    return vec3<f32>(rx, local.y, rz);
}

// Wind sway in world XZ at `y_norm` up the blade. Direction comes from
// `WindDirection`; amplitude/period are per-variant; speed/gustiness/crest
// spacing come from `MeadowWindState`. Gust crests travel along the wind
// direction so the meadow reads as one big wave passing through, not as
// every blade gusting in lockstep. Shared by the raster passthrough VS
// (`meadow.wesl`) and the raytracing expansion (`meadow_compute.wesl`) so
// RT shadows sway in lockstep with the visible grass; the parameters are
// the `VariantParams` fields of the same names (passed explicitly because
// each caller binds `variant_params` at a different group slot).
fn wind_sway(
    world_xz: vec2<f32>,
    blade_y_norm: f32,
    t: f32,
    clump: f32,
    wind: vec4<f32>,
    wind_direction: vec4<f32>,
    wind_state: vec4<f32>,
) -> vec2<f32> {
    let wind_time = select(t, wind_state.w - 1.0, wind_state.w > 0.0);
    let speed_mul = wind_state.x;
    let gustiness = wind_state.y;
    let crest_k = wind_state.z;
    let amp = wind.x * speed_mul;
    let period = max(wind.y, 1e-3);
    let dir = normalize(wind_direction.xy + vec2<f32>(0.0001, 0.0));
    // Subtracting `dot(world_xz, dir) * crest_k` makes the crest move
    // along +dir rather than against it. `clump * 1.57` (~π/2) gives
    // ~quarter-cycle per-blade phase scatter — enough to break perfect
    // lockstep without drowning the spatial gradient that produces the
    // traveling crest. The earlier `clump * 4π` swamped the spatial
    // signal so blades visibly sloshed independently.
    let phase = (wind_time / period) * 6.2831853 + clump * 1.57 - dot(world_xz, dir) * crest_k;
    let gust_osc = 0.6 + 0.4 * sin(wind_time * 0.25 + world_xz.x * 0.11 + world_xz.y * 0.09);
    let gust = mix(1.0, gust_osc, gustiness);
    let sway = sin(phase) * amp * gust * blade_y_norm;
    return dir * sway;
}

// ---------- hashes ----------
// The one hash family of both derivation paths and the variance below.

fn hash_u32(v: u32) -> u32 {
    var x = v;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash01_u32(v: u32) -> f32 {
    return f32(hash_u32(v) & 0x00FFFFFFu) / f32(0x01000000u);
}

fn hash01_pair(a: u32, b: u32) -> f32 {
    let h = hash_u32(a ^ (b * 0x9E3779B9u));
    return f32(h & 0x00FFFFFFu) / f32(0x01000000u);
}

// ---------- LOD gate ----------

// Largest blade count of a patch. MUST equal `mesh.rs::MAX_BLADES_PER_PATCH`.
const MEADOW_MAX_BLADES_PER_PATCH: u32 = 262144u;

// Blades of a patch that pass a survival `threshold` in [0, 1]: the gate
// keeps the first `ceil(threshold * blade_count)` blade indices. Every
// blade's root is drawn from a hash of its index, so a prefix is a uniform
// spatial subset, and work launched for the prefix alone covers every
// survivor (`compute.rs::task_slice_bound`).
fn meadow_gate_count(threshold: f32, blade_count: u32) -> u32 {
    return min(u32(ceil(clamp(threshold, 0.0, 1.0) * f32(blade_count))), blade_count);
}

// Grass farther than this from the player casts no shadow-map shadow, in
// metres. MUST equal `mesh.rs::SHADOW_MAX_DIST`.
const MEADOW_SHADOW_MAX_DIST: f32 = 50.0;

// Shadow-caster LOD ramp at planar player distance `dist`: 0 within
// `shadow_full_dist` (`VariantParams.density.z`), 1 at
// `MEADOW_SHADOW_MAX_DIST`. Keyed on player distance like the other bands,
// so a patch's caster count is the same in every cascade and view.
fn meadow_shadow_ramp(dist: f32, shadow_full_dist: f32) -> f32 {
    return clamp(
        (dist - shadow_full_dist) / max(MEADOW_SHADOW_MAX_DIST - shadow_full_dist, 1e-3),
        0.0,
        1.0,
    );
}

// Survival threshold (`meadow_gate_count`) of a patch's band in a view, at
// the planar player distance `dist` of its centre and the view's far LOD
// distance `lod_max`:
// - near blades (band 0) thin 1 -> 1 / max_widen (`lod.w`) across
//   [full_distance, tuft_start], the least share whose widening keeps
//   their area;
// - shadow casters (band 0, shadow views) thin from 1 to
//   `shadow_far_density` over `meadow_shadow_ramp`;
// - far tufts (band 1) ramp tuft.y -> tuft.z across [tuft_start, lod_max].
// The CPU twin is `compute.rs::task_slice_bound`.
fn meadow_band_threshold(
    params: VariantParams,
    band: u32,
    is_shadow: bool,
    dist: f32,
    lod_max: f32,
) -> f32 {
    let tuft_start = params.tuft.x;
    if (band == 1u) {
        let ramp_t = clamp((dist - tuft_start) / max(lod_max - tuft_start, 1e-3), 0.0, 1.0);
        return mix(params.tuft.y, params.tuft.z, ramp_t);
    }
    if (is_shadow) {
        return mix(1.0, params.density.w, meadow_shadow_ramp(dist, params.density.z));
    }
    let lod_full = params.lod.x;
    let ramp = clamp((dist - lod_full) / max(tuft_start - lod_full, 1e-3), 0.0, 1.0);
    return mix(1.0, 1.0 / params.lod.w, ramp);
}

// Area of a far-band tuft over that of a blade of the same width and
// height. MUST equal `plugin.rs::TUFT_BLADE_AREA_RATIO`.
const MEADOW_TUFT_BLADE_AREA: f32 = 7.03;

// Width scale of a survivor of band `band` (1 = tufts) that keeps share
// `keep` of its patch's blades: the inverse of the area share the
// survivors draw, up to `max_widen` (`VariantParams.lod.w`), so thinning
// keeps the band's blade area. CPU twin: `MeadowVariant::lod_widen`.
fn meadow_lod_widen(keep: f32, band: u32, max_widen: f32) -> f32 {
    let drawn = keep * select(1.0, MEADOW_TUFT_BLADE_AREA, band == 1u);
    return min(1.0 / max(drawn, 1e-4), max_widen);
}

// ---------- per-blade palette variance ----------
//
// Every blade takes its patch's palette entry and varies around it with
// zero mean, so the coverage-weighted mean colour of a patch is
// `mix(live, dry, dead_share)` and its mean height the unvaried one on
// ground of neutral dryness:
// - dead weight: a blade is dead where warped fractal world-space noise at
//   its root crosses the patch threshold (dead patches) or where its own
//   random falls under the single-blade threshold. The CPU splits the dead
//   share between the two so the expected weight equals it at neutral
//   ground dryness; the cover field's dryness scales the patch share by
//   `2 * dryness`, so dry ground browns in patches and moist ground stays
//   green.
// - height: `1 + variance * (blade + patch) / 2`, both terms in [-1, 1]
//   with zero mean and from the same two sources as the dead weight, so
//   dead blades and dead patches run short; times a ground-dryness factor
//   (`meadow_dryness_height`) that is 1 on neutral ground.
// - colour: a hue rotation about the grey axis and a value scale, each
//   `jitter * [-1, 1]`.
// Height applies to every surviving blade; the dead weight and colour only
// where a view shades the blade.

// Hue rotation in radians at a colour jitter of 1.
const MEADOW_HUE_JITTER_RAD: f32 = 1.0;

// Palette index packed into bits 8..15 of `PatchData`'s flags lane
// (`plugin.rs::upload_placements`, which clamps it into the padded table).
fn meadow_patch_palette(flags: f32) -> u32 {
    return (bitcast<u32>(flags) >> 8u) & 0xFFu;
}

// Factor bounding a blade's height above the variant's `height_range.y`.
fn meadow_max_height_scale(palette: PaletteParams) -> f32 {
    return 1.0 + palette.variance.x;
}

// Transient (not a buffer layout): the noise lattice cell about a point, the
// point's offset within it and its quintic interpolation weights.
struct MeadowNoiseLattice {
    cell: vec2<i32>,
    offset: vec2<f32>,
    weight: vec2<f32>,
}

// Lattice rotation (cos, sin) of the value noises: off the world axes so
// features don't line up with the world grid.
const MEADOW_NOISE_ROTATION: vec2<f32> = vec2<f32>(0.8, 0.6);

// The lattice about `p` turned by the rotation `rot` (cos, sin).
fn meadow_noise_lattice(p: vec2<f32>, rot: vec2<f32>) -> MeadowNoiseLattice {
    let q = vec2<f32>(rot.x * p.x - rot.y * p.y, rot.y * p.x + rot.x * p.y);
    let f = q - floor(q);
    return MeadowNoiseLattice(vec2<i32>(floor(q)), f, f * f * f * (f * (f * 6.0 - 15.0) + 10.0));
}

// Hashes of a lattice cell's corners (x0z0, x1z0, x0z1, x1z1), the noise
// `seed` offsetting x: `hash01_pair`'s hash.
fn meadow_corner_hashes(cell: vec2<i32>, seed: u32) -> vec4<u32> {
    let x0 = bitcast<u32>(cell.x) + seed;
    let k0 = bitcast<u32>(cell.y) * 0x9E3779B9u;
    let k1 = (bitcast<u32>(cell.y) + 1u) * 0x9E3779B9u;
    return vec4<u32>(
        hash_u32(x0 ^ k0),
        hash_u32((x0 + 1u) ^ k0),
        hash_u32(x0 ^ k1),
        hash_u32((x0 + 1u) ^ k1),
    );
}

// Bilinear blend of corner values (x0z0, x1z0, x0z1, x1z1) by `weight`.
fn meadow_lattice_blend(v: vec4<f32>, weight: vec2<f32>) -> f32 {
    let z = mix(v.xz, v.yw, weight.x);
    return mix(z.x, z.y, weight.y);
}

// Quintic value noise in [0, 1] with mean 0.5 and a bell-shaped spread.
fn meadow_value_noise(p: vec2<f32>, seed: u32) -> f32 {
    let lattice = meadow_noise_lattice(p, MEADOW_NOISE_ROTATION);
    let h = meadow_corner_hashes(lattice.cell, seed);
    return meadow_lattice_blend(vec4<f32>(h & vec4<u32>(0x00FFFFFFu)) / f32(0x01000000u), lattice.weight);
}

// Three independent value noises in [0, 1] off one lattice, from bits
// 0..10, 10..20 and 20..30 of each corner hash.
fn meadow_value_noise3(p: vec2<f32>, seed: u32) -> vec3<f32> {
    let lattice = meadow_noise_lattice(p, MEADOW_NOISE_ROTATION);
    let h = meadow_corner_hashes(lattice.cell, seed);
    let bits = vec4<u32>(1023u);
    return vec3<f32>(
        meadow_lattice_blend(vec4<f32>(h & bits) / 1024.0, lattice.weight),
        meadow_lattice_blend(vec4<f32>((h >> vec4<u32>(10u)) & bits) / 1024.0, lattice.weight),
        meadow_lattice_blend(vec4<f32>((h >> vec4<u32>(20u)) & bits) / 1024.0, lattice.weight),
    );
}

// Quintic gradient noise on the lattice turned by `rot`, zero mean and
// symmetric. Each corner's gradient is `±(dir + spread * r)`, `r` uniform in
// [-1, 1]² and the sign from bit 31 of the corner hash, with `dir` a
// world-space unit vector: features stretch across `dir`, the more the
// smaller `spread`.
fn meadow_oriented_noise(p: vec2<f32>, rot: vec2<f32>, seed: u32, dir: vec2<f32>, spread: f32) -> f32 {
    let lattice = meadow_noise_lattice(p, rot);
    let h = meadow_corner_hashes(lattice.cell, seed);
    let d = vec2<f32>(rot.x * dir.x - rot.y * dir.y, rot.y * dir.x + rot.x * dir.y);
    let sign = select(vec4<f32>(-1.0), vec4<f32>(1.0), (h >> vec4<u32>(31u)) != vec4<u32>(0u));
    let rx = vec4<f32>(h & vec4<u32>(0xFFFFu)) / 32768.0 - 1.0;
    let rz = vec4<f32>((h >> vec4<u32>(16u)) & vec4<u32>(0x7FFFu)) / 16384.0 - 1.0;
    let f = lattice.offset;
    let v = sign * (d.x + spread * rx) * (f.x - vec4<f32>(0.0, 1.0, 0.0, 1.0))
        + sign * (d.y + spread * rz) * (f.y - vec4<f32>(0.0, 0.0, 1.0, 1.0));
    return meadow_lattice_blend(v, lattice.weight);
}

// Quintic value noise remapped to a uniform [0, 1] distribution. The odd
// polynomial in `n - 0.5` is a least-squares fit of the raw noise's CDF
// (max error 0.0015), so a threshold at `1 - a` covers a share `a` of the
// ground.
fn meadow_uniform_noise(p: vec2<f32>) -> f32 {
    let n = meadow_value_noise(p, 0u) - 0.5;
    let n2 = n * n;
    return clamp(0.5 + n * (1.49767 + n2 * (-2.16135 + n2 * 0.73196)), 0.0, 1.0);
}

// Gradient spread of the dead-patch noise's detail octaves: features
// stretch mildly across the slowly turning direction.
const MEADOW_PATCH_SPREAD: f32 = 0.6;

// Domain-warped, anisotropic fractal noise for dead patches at `p` in units
// of the patch size, remapped to a uniform [0, 1] distribution. Its main
// lattice cell spans two patch sizes, so a feature spans about one. A
// low-frequency value-noise triple shifts the domain by up to 1.1 cells and
// sets a stretch direction that turns slowly across the world. A
// near-isotropic octave at 0.45 of the main frequency gathers patches into
// clusters that merge where the threshold drops, and three oriented detail
// octaves (weights 4, 2.6 and 1.7, lacunarity 2.03), each on its own lattice
// rotation (distinct modulo a quarter turn), stretch across the direction
// and fray the outlines, so thresholded patches come out ragged and
// elongated, without isolated round tops or lattice-aligned edges. The tanh
// of an odd polynomial fits the sum's CDF (max error 0.0003), so a threshold covers its share to within
// 0.1 % of it from a share of 0.003 up; CPU twin `field.rs::patch_noise`.
fn meadow_patch_fbm(p: vec2<f32>) -> f32 {
    let lp = 0.5 * p;
    let w = meadow_value_noise3(0.25 * lp, 0x51EDu);
    let q = lp + 2.2 * (w.xy - vec2<f32>(0.5));
    let angle = 6.28318531 * w.z;
    let dir = vec2<f32>(cos(angle), sin(angle));
    let sum = 3.0 * meadow_oriented_noise(0.45 * q, vec2<f32>(0.96, 0.28), 0x2B5Fu, dir, 1.2)
        + 4.0 * meadow_oriented_noise(q, vec2<f32>(0.8, 0.6), 0x9E37u, dir, MEADOW_PATCH_SPREAD)
        + 2.6 * meadow_oriented_noise(2.03 * q, vec2<f32>(0.28, 0.96), 0x7F4Au, dir, MEADOW_PATCH_SPREAD)
        + 1.7 * meadow_oriented_noise(4.1209 * q, vec2<f32>(0.6, 0.8), 0x3C6Fu, dir, MEADOW_PATCH_SPREAD);
    let n = sum / 11.3;
    let n2 = n * n;
    // The clamp keeps tanh finite where a driver builds it from exp; the
    // sum never reaches it.
    return 0.5 + 0.5 * tanh(clamp(n * (5.96009 + n2 * (14.33164 - n2 * 0.70197)), -9.0, 9.0));
}

// ---------- tile patches and the cover field ----------
//
// A tile patch (flags bit 1) is a world-aligned square of half-width
// `radius` whose blades fill it uniformly. Its blades read the cover field
// atlas (`field.rs::CoverTexel`): r = cover density in [0, 1], g = height
// scale, b = continuous palette index (< 0: the patch's own entry), a =
// ground dryness (0.5 neutral, `meadow_patch_dead`). A blade
// survives where the density, over the patch's cover ceiling (`PatchData`
// lane y), exceeds its threshold (`meadow_cover_threshold`), so the
// expected cover equals the density.

// Height of a blade's root above the heightfield. MUST equal
// `rt_selection.rs::BLADE_GROUND_LIFT`.
const MEADOW_BLADE_GROUND_LIFT: f32 = 0.02;

// Height at or below which a clearing profile holds grass flat; the disc
// clears it there. MUST equal `material.rs::FLAT_HEIGHT_CAP`.
const MEADOW_FLAT_HEIGHT_CAP: f32 = 0.001;

// Largest clearing-disc edge noise. MUST equal
// `material.rs::MAX_DISC_EDGE_NOISE`.
const MEADOW_MAX_DISC_EDGE_NOISE: f32 = 0.5;

// Texel `x` of row `row` of a clearing-profile table
// (`material.rs::ClearingProfiles`).
fn meadow_clearing_texel(profiles: texture_2d<f32>, row: u32, x: u32) -> f32 {
    return textureLoad(profiles, vec2<i32>(i32(x), i32(row)), 0).r;
}

// Sample `(ring, j)` of clearing-profile row `row` with `sectors` sectors,
// times `scale_lift.x` plus `scale_lift.y`.
fn meadow_clearing_sample(
    profiles: texture_2d<f32>,
    row: u32,
    sectors: u32,
    ring: u32,
    j: u32,
    scale_lift: vec2<f32>,
) -> f32 {
    return meadow_clearing_texel(profiles, row, 2u + ring * sectors + j) * scale_lift.x + scale_lift.y;
}

// A blade `height` tall under clearing-profile row `row`
// (`material.rs::MeadowClearingProfile`) at `u` of the distance where the
// profile has relaxed and `turns` of a turn from its local +x, the row's
// samples times `scale_lift.x` plus `scale_lift.y` (the profile origin above
// the blade's ground): x = its height under the cap, interpolated
// bilinearly between the four samples about it and rising to `height`
// across one ring step past the outer ring; y = 1 where those four samples
// hold grass flat, else 0. A row the table lacks or cannot hold caps
// nothing.
fn meadow_clearing_profile(
    profiles: texture_2d<f32>,
    row: u32,
    u: f32,
    turns: f32,
    scale_lift: vec2<f32>,
    height: f32,
) -> vec2<f32> {
    let dims = textureDimensions(profiles, 0);
    if (row >= dims.y || dims.x < 2u) {
        return vec2<f32>(height, 0.0);
    }
    let rings = u32(meadow_clearing_texel(profiles, row, 0u));
    let sectors = u32(meadow_clearing_texel(profiles, row, 1u));
    if (rings < 2u || sectors < 1u || 2u + rings * sectors > dims.x) {
        return vec2<f32>(height, 0.0);
    }
    let last = rings - 1u;
    let q = clamp(u * f32(rings), 0.0, f32(rings));
    let i = min(u32(q), last);
    let f = q - f32(i);
    let s = (turns - floor(turns)) * f32(sectors);
    let j0 = min(u32(s), sectors - 1u);
    let g = s - f32(j0);
    let j1 = (j0 + 1u) % sectors;
    let h00 = meadow_clearing_sample(profiles, row, sectors, i, j0, scale_lift);
    let h01 = meadow_clearing_sample(profiles, row, sectors, i, j1, scale_lift);
    let here = clamp(h00, 0.0, height) * (1.0 - g) + clamp(h01, 0.0, height) * g;
    var next = height;
    var flat = 0.0;
    if (i < last) {
        let h10 = meadow_clearing_sample(profiles, row, sectors, i + 1u, j0, scale_lift);
        let h11 = meadow_clearing_sample(profiles, row, sectors, i + 1u, j1, scale_lift);
        next = clamp(h10, 0.0, height) * (1.0 - g) + clamp(h11, 0.0, height) * g;
        if (max(max(h00, h01), max(h10, h11)) <= MEADOW_FLAT_HEIGHT_CAP) {
            flat = 1.0;
        }
    }
    return vec2<f32>(here * (1.0 - f) + next * f, flat);
}

// Clearing disc `(c, a, n)` (`material.rs::TrunkDisc`) at `root_xz` for a
// blade `height` tall growing from world height `ground_y`, its profile in
// table `profiles`: x = the share of grass it lets stand (0 in its core, 1
// past its fade, 0 where its profile holds grass flat), y = the blade's
// height under its profile. Edge noise scales the distance by a smooth
// noise sized to the disc, so both outlines wobble.
fn meadow_disc_effect(
    profiles: texture_2d<f32>,
    c: vec4<f32>,
    a: vec4<f32>,
    n: vec4<f32>,
    root_xz: vec2<f32>,
    ground_y: f32,
    height: f32,
) -> vec2<f32> {
    let centre = c.xy;
    let offset = root_xz - centre;
    let clear_outer = c.z + max(c.w, 1e-4);
    // Edge noise in [-1, 1], skipped where neither outline wobbles.
    var noise = 0.0;
    if (max(n.x, n.y) > 0.0) {
        let wavelength = max(0.5 * max(clear_outer, a.z), 0.25);
        noise = 2.0 * meadow_uniform_noise(offset / wavelength + centre * 0.37) - 1.0;
    }
    let dist = length(offset);
    var stand = smoothstep(
        c.z,
        clear_outer,
        dist * (1.0 + clamp(n.x, 0.0, MEADOW_MAX_DISC_EDGE_NOISE) * noise),
    );
    var capped = height;
    if (a.z > 0.0) {
        let d = dist * (1.0 + clamp(n.y, 0.0, MEADOW_MAX_DISC_EDGE_NOISE) * noise);
        var turns = -a.x;
        if (any(offset != vec2<f32>(0.0))) {
            turns = atan2(offset.y, offset.x) / 6.2831855 - a.x;
        }
        let scale_lift = vec2<f32>(a.w, a.y - ground_y);
        let profile = meadow_clearing_profile(profiles, u32(n.z), d / a.z, turns, scale_lift, height);
        capped = profile.x;
        if (profile.y > 0.0) {
            stand = 0.0;
        }
    }
    return vec2<f32>(stand, capped);
}

// Largest palette height variance. MUST equal
// `plugin.rs::MAX_HEIGHT_VARIANCE`.
const MEADOW_MAX_HEIGHT_VARIANCE: f32 = 0.9;

// Transient (not a buffer layout): a derived blade past its per-blade
// gates (`gate_blade` in the deriving kernels).
struct GatedBlade {
    ground_y: f32,
    // Height after variance, the field's scale, the tuft fade and the
    // clearing discs' cap.
    height: f32,
    // Clearing and rim collapse; 0 drops the blade.
    collapse_t: f32,
    // Palette-table entry the blade takes its colour from.
    palette: u32,
    // Dead-patch noise at the root (`meadow_patch_noise`).
    noise: f32,
    // Ground dryness at the root: the cover field's on a tile, else neutral.
    dryness: f32,
    // Width factor that keeps the thinned band's blade area
    // (`meadow_lod_widen`); 1 without the LOD gate.
    width_scale: f32,
}

// Ground dryness that keeps a palette entry's own dead-patch share. MUST
// equal `field.rs::CoverTexel::NEUTRAL_DRYNESS`.
const MEADOW_NEUTRAL_DRYNESS: f32 = 0.5;

// `PatchData` flag bit marking a tile patch (`plugin.rs::TILE_FLAG`).
const MEADOW_PATCH_FLAG_TILE: u32 = 2u;

fn meadow_patch_is_tile(flags: f32) -> bool {
    return (bitcast<u32>(flags) & MEADOW_PATCH_FLAG_TILE) != 0u;
}

// Planar radius that bounds a patch's blade roots.
fn meadow_patch_outer_radius(radius: f32, edge_noise: f32, tile: bool) -> f32 {
    if (tile) {
        return radius * 1.4142135;
    }
    return radius * (1.0 + edge_noise);
}

// Offset from its tile's centre of the tile blade with seed `blade_seed`,
// uniform over a square of half-width `half`.
fn meadow_tile_root(half: f32, blade_seed: u32) -> vec2<f32> {
    let u = vec2<f32>(hash01_u32(blade_seed), hash01_u32(blade_seed + 1u));
    return (2.0 * u - 1.0) * half;
}

// Survival threshold of a blade: its own uniform random `blade_u` mixed
// with the spatially correlated uniform noise `noise_u` by `clumpiness`,
// mapped through the CDF of the weighted sum so the threshold stays
// uniform in [0, 1]. Clumpiness 0 dithers blade by blade; 1 cuts whole
// noise-sized patches with a hard edge. CPU twin: `field.rs::cover_threshold`.
fn meadow_cover_threshold(blade_u: f32, noise_u: f32, clumpiness: f32) -> f32 {
    let b = clamp(clumpiness, 0.0, 1.0);
    let a = 1.0 - b;
    let lo = min(a, b);
    let hi = max(a, b);
    if (lo < 1e-4) {
        return select(noise_u, blade_u, a > b);
    }
    let z = a * blade_u + b * noise_u;
    if (z < lo) {
        return z * z / (2.0 * a * b);
    }
    if (z > hi) {
        let w = 1.0 - z;
        return 1.0 - w * w / (2.0 * a * b);
    }
    return (z - 0.5 * lo) / hi;
}

// Whether a tile blade at `root_xz` with seed `seed` survives the cover
// `density` under its patch's cover `ceiling`.
fn meadow_cover_survives(
    params: VariantParams,
    root_xz: vec2<f32>,
    seed: u32,
    density: f32,
    ceiling: f32,
) -> bool {
    let noise_u = meadow_uniform_noise(root_xz * params.cover.x);
    let threshold = meadow_cover_threshold(hash01_u32(seed + 37u), noise_u, params.cover.y);
    return density > threshold * max(ceiling, 1e-4);
}

// Transient (not a buffer layout): the four texels about a point of a
// world-aligned grid (`field.rs::MeadowGrid`) and the point's position
// between them.
struct MeadowGridFootprint {
    c00: vec4<f32>,
    c10: vec4<f32>,
    c01: vec4<f32>,
    c11: vec4<f32>,
    t: vec2<f32>,
}

// Whether `grid` holds a field: a bound grid is at least 2 texels a side.
fn meadow_grid_bound(grid: texture_2d<f32>) -> bool {
    let dims = textureDimensions(grid, 0);
    return dims.x >= 2u && dims.y >= 2u;
}

// The footprint of `world_xz` in the bound grid `grid` over `extents` (xy =
// min XZ, zw = max XZ), edge-clamped: texel `(i, j)` sits at
// `min + (max - min) * (i, j) / (dims - 1)`.
fn meadow_grid_footprint(grid: texture_2d<f32>, extents: vec4<f32>, world_xz: vec2<f32>) -> MeadowGridFootprint {
    let dims = vec2<f32>(textureDimensions(grid, 0));
    let size = max(extents.zw - extents.xy, vec2<f32>(1e-3));
    let uv = clamp((world_xz - extents.xy) / size, vec2<f32>(0.0), vec2<f32>(1.0));
    let pixel = uv * (dims - vec2<f32>(1.0));
    let p0 = vec2<i32>(floor(pixel));
    let p1 = min(p0 + vec2<i32>(1), vec2<i32>(dims) - vec2<i32>(1));
    return MeadowGridFootprint(
        textureLoad(grid, p0, 0),
        textureLoad(grid, vec2<i32>(p1.x, p0.y), 0),
        textureLoad(grid, vec2<i32>(p0.x, p1.y), 0),
        textureLoad(grid, p1, 0),
        pixel - vec2<f32>(p0),
    );
}

// Cover field sample at `world_xz`: bilinear density, height scale and
// ground dryness (x, y, w), the nearest texel's palette index (z). A field
// narrower than 2 texels (none bound) reads full cover at unit height with
// the patch palette on neutral ground.
fn meadow_sample_cover(
    cover_field: texture_2d<f32>,
    extents: vec4<f32>,
    world_xz: vec2<f32>,
) -> vec4<f32> {
    if (!meadow_grid_bound(cover_field)) {
        return vec4<f32>(1.0, 1.0, -1.0, MEADOW_NEUTRAL_DRYNESS);
    }
    let f = meadow_grid_footprint(cover_field, extents, world_xz);
    let blended = mix(mix(f.c00, f.c10, f.t.x), mix(f.c01, f.c11, f.t.x), f.t.y);
    let near_x = f.t.x >= 0.5;
    let palette = select(
        select(f.c00.b, f.c10.b, near_x),
        select(f.c01.b, f.c11.b, near_x),
        f.t.y >= 0.5,
    );
    return vec4<f32>(blended.rg, palette, blended.a);
}

// Palette entry of a blade from the field's continuous `index`: one of the
// two entries around it, the upper one with probability `fract(index)`. A
// negative index keeps `patch_palette`.
fn meadow_blade_palette(index: f32, seed: u32, patch_palette: u32) -> u32 {
    if (index < 0.0) {
        return patch_palette;
    }
    let picked = u32(floor(index + hash01_u32(seed + 41u)));
    return min(picked, MEADOW_MAX_PALETTES - 1u);
}

// Rank of blade `b` of `count` in the LOD prefix order, in (0, 1): the gate
// at threshold `t` keeps about the blades ranked below `t`.
fn meadow_blade_rank(b: u32, count: u32) -> f32 {
    return (f32(b) + 0.5) / f32(max(count, 1u));
}

// Dead-patch noise at a blade root on ground of `dryness`, shared by its
// height and colour: the uniform patch noise, or 0.5 where the entry has no
// patches or the ground none of its share, which both then ignore.
fn meadow_patch_noise(palette: PaletteParams, root_xz: vec2<f32>, dryness: f32) -> f32 {
    if (palette.variance.y <= 0.0 || meadow_patch_share(palette, dryness) <= 0.0) {
        return 0.5;
    }
    return meadow_patch_fbm(root_xz * palette.variance.y);
}

// Dead-patch share of a palette entry on ground of `dryness` (0.5 neutral):
// the entry's share scaled by `2 * dryness`.
fn meadow_patch_share(palette: PaletteParams, dryness: f32) -> f32 {
    return clamp(palette.variance.z * 2.0 * dryness, 0.0, 1.0);
}

// Chance a blade with patch noise `noise` is dead in a patch of `share`:
// past a soft edge no wider than twice the share or its complement, so the
// ramp stays inside [0, 1] and its expected value over the uniform noise is
// the share exactly.
fn meadow_patch_dead(palette: PaletteParams, noise: f32, share: f32) -> f32 {
    let edge = max(min(palette.variance.w, 2.0 * min(share, 1.0 - share)), 1e-4);
    let ramp = clamp((noise - (1.0 - share)) / edge + 0.5, 0.0, 1.0);
    return select(0.0, ramp, share > 0.0);
}

// Largest height factor of ground dryness (`meadow_dryness_height`).
const MEADOW_MAX_DRYNESS_HEIGHT: f32 = 1.1;

// Blade height factor of ground `dryness`: 1.1 on the wettest ground, 1 on
// neutral and 0.8 on the driest, linear between.
fn meadow_dryness_height(dryness: f32) -> f32 {
    let d = clamp(dryness, 0.0, 1.0) - MEADOW_NEUTRAL_DRYNESS;
    return 1.0 - d * select(0.2, 0.4, d > 0.0);
}

// A blade's height scale from its patch noise, ground dryness and seed: the
// dryness factor times `1 + variance * (blade + patch) / 2`. The blade term
// is its random in [-1, 1]; the patch term its dead-patch chance below the
// share, over the larger of the share and its complement, so in [-1, 1]
// with zero mean: patch blades run short, the rest a little tall, and
// ground without patches keeps its height.
fn meadow_blade_height_scale(palette: PaletteParams, noise: f32, dryness: f32, seed: u32) -> f32 {
    let blade_term = 2.0 * hash01_u32(seed + 19u) - 1.0;
    let share = meadow_patch_share(palette, dryness);
    let patch_term = (share - meadow_patch_dead(palette, noise, share)) / max(share, 1.0 - share);
    return meadow_dryness_height(dryness) * (1.0 + palette.variance.x * 0.5 * (blade_term + patch_term));
}

// Mean of the blade height shade (`meadow_blade_base_color`) over a blade.
// MUST equal `plugin.rs::MEAN_BLADE_SHADE`.
const MEADOW_MEAN_BLADE_SHADE: f32 = 0.85;

// Expected shaded albedo of the blades rooted about `root_xz` on ground of
// `dryness`, averaged over a footprint `footprint_m` metres across: their
// dead weight (the patch noise there, then single dead blades) between the
// live and dry colours, under the mean blade shade. Footprints past a
// quarter of a dead patch fade the patches toward their share, the
// noise's mean, so they do not alias; the per-blade colour jitter is
// zero-mean and left out. What a stand of blades too small to draw
// averages to, such as on far ground.
fn meadow_stand_albedo(
    palette: PaletteParams,
    root_xz: vec2<f32>,
    dryness: f32,
    footprint_m: f32,
) -> vec3<f32> {
    let share = meadow_patch_share(palette, dryness);
    let resolved = 1.0 - smoothstep(0.25, 1.0, footprint_m * palette.variance.y);
    var patch_dead = share;
    if (resolved > 0.0) {
        let noise = meadow_patch_noise(palette, root_xz, dryness);
        patch_dead = mix(share, meadow_patch_dead(palette, noise, share), resolved);
    }
    let dead = patch_dead + (1.0 - patch_dead) * palette.dry_blade_share.w;
    return mix(palette.live_jitter.rgb, palette.dry_blade_share.rgb, dead) * MEADOW_MEAN_BLADE_SHADE;
}

// A blade's linear albedo from its patch noise, ground dryness and seed.
fn meadow_blade_albedo(palette: PaletteParams, noise: f32, dryness: f32, seed: u32) -> vec3<f32> {
    // Each blade is live or dead: the soft patch edge thins the dead blades
    // out rather than tinting them.
    let patch_chance = meadow_patch_dead(palette, noise, meadow_patch_share(palette, dryness));
    let dead = hash01_u32(seed + 43u) < patch_chance || hash01_u32(seed + 19u) < palette.dry_blade_share.w;
    let base = select(palette.live_jitter.rgb, palette.dry_blade_share.rgb, dead);
    let jitter = palette.live_jitter.w;
    // Rodrigues rotation about the grey axis: keeps the channel mean and
    // turns the chroma by `angle`.
    let angle = jitter * (2.0 * hash01_u32(seed + 23u) - 1.0) * MEADOW_HUE_JITTER_RAD;
    let k = vec3<f32>(0.57735027);
    let cos_a = cos(angle);
    let rotated = base * cos_a + cross(k, base) * sin(angle) + k * dot(k, base) * (1.0 - cos_a);
    let value = 1.0 + jitter * (2.0 * hash01_u32(seed + 29u) - 1.0);
    return max(rotated * value, vec3<f32>(0.0));
}

// 8-bit gamma-2 (square-root) encoded albedo: close to sRGB's precision on
// dark albedos for a `sqrt` and a square.
fn meadow_pack_albedo(albedo: vec3<f32>) -> u32 {
    return pack4x8unorm(vec4<f32>(sqrt(clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0));
}

fn meadow_unpack_albedo(packed: u32) -> vec3<f32> {
    let s = unpack4x8unorm(packed).rgb;
    return s * s;
}

// ---------- blade shading ----------

// Blade albedo shaded up the blade, darker at the root:
// `mix(0.70, 1.00, y_norm)`, whose mean over the blade is
// `plugin.rs::MEAN_BLADE_SHADE`.
fn meadow_blade_base_color(albedo: vec3<f32>, y_norm: f32) -> vec3<f32> {
    return albedo * mix(0.70, 1.00, y_norm);
}

// Faint glow on the top of the blade.
fn meadow_blade_emissive(albedo: vec3<f32>, y_norm: f32) -> vec3<f32> {
    return albedo * smoothstep(0.7, 1.0, y_norm) * 0.06;
}
// Grass shading normal. Near blades shade with their own normal
// (`meadow_blade_world_normal`, interpolated), turned to the side facing
// the viewer, so the blades of a meadow catch the light differently. With
// `blade_weight` it blends to the upright normal, which keeps sub-pixel
// grass lighting coherent; weight 0 is the upright normal exactly.
//
// A hillside can put the camera below the normal's hemisphere, and
// Solari's BRDF rejects NdotV <= 0, making those blades black. The normal
// is kept whenever NdotV clears a margin and tilted just enough toward the
// viewer otherwise. Shared by both raster rendering paths.
fn meadow_shading_normal(
    blade_normal: vec3<f32>,
    blade_weight: f32,
    view_direction: vec3<f32>,
) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let length_squared = dot(view_direction, view_direction);
    if length_squared < 1e-12 { return up; }
    let v = view_direction * inverseSqrt(length_squared);
    var n = up;
    let blade_length_squared = dot(blade_normal, blade_normal);
    if blade_weight > 0.0 && blade_length_squared > 1e-12 {
        var facing = blade_normal * inverseSqrt(blade_length_squared);
        if dot(facing, v) < 0.0 { facing = -facing; }
        let blended = mix(up, facing, min(blade_weight, 1.0));
        let blended_length_squared = dot(blended, blended);
        if blended_length_squared > 1e-6 {
            n = blended * inverseSqrt(blended_length_squared);
        }
    }
    let n_dot_v = dot(n, v);
    let margin = 0.1;
    if n_dot_v >= margin { return n; }
    let tangent = n - v * n_dot_v;
    let tangent_length_squared = dot(tangent, tangent);
    // A normal facing straight away from the viewer has no unique tangent.
    if tangent_length_squared < 1e-12 { return v; }
    return tangent * inverseSqrt(tangent_length_squared) * sqrt(1.0 - margin * margin)
        + v * margin;
}

// G-buffer A bit 31 marks a meadow pixel as a thin two-sided receiver. The
// contract is private to meadow: both raster paths write it and meadow's
// Solari receiver pass (`meadow_receivers.wesl`) is its only reader. bevy_pbr's
// deferred flags grow up from bit 24 and the packed normal holds bits 0..23;
// meadow takes the flag byte's top two bits and alters neither.
const MEADOW_GBUFFER_THIN_RECEIVER_BIT: u32 = 1u << 31u;

// G-buffer A bit 30 marks a geometric-normal payload in B, read only by
// meadow's Solari receiver pass and meaningful only together with bit 31. B
// bits 0..15 keep IOR, specular and metallic; bits 16..23 and 24..31 carry the
// octahedral triangle normal as two unorm8 values. The payload overwrites the
// AO and clearcoat bytes that deferred lighting reads, so the caller gates it
// with `width_range.z`.
const MEADOW_GBUFFER_GEOMETRIC_NORMAL_BIT: u32 = 1u << 30u;

// Reconstruct each invocation's OWN rasterized position in view space.
// Differentiating absolute world-position varyings loses precision far from
// the origin. This keeps coordinates small without extra vertex varyings or
// sampling neighboring surfaces. Viewport must match the geometry pass,
// including resolution overrides; view_from_clip must include its jitter.
fn meadow_triangle_normal(
    frag_coord: vec4<f32>,
    viewport: vec4<f32>,
    view_from_clip: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
) -> vec3<f32> {
    // Same conversion as bevy_render::view::{frag_coord_to_ndc,
    // position_ndc_to_view}, kept binding-free for shared meadow modules.
    let uv = (frag_coord.xy - viewport.xy) / viewport.zw;
    let ndc = vec3<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), frag_coord.z);
    let homogeneous_view = view_from_clip * vec4<f32>(ndc, 1.0);
    let view_position = homogeneous_view.xyz / homogeneous_view.w;
    let view_normal = cross(dpdx(view_position), dpdy(view_position));
    // Inverse-transpose normal transform also handles a scaled camera.
    let normal_to_world = transpose(mat3x3<f32>(view_from_world[0].xyz,
        view_from_world[1].xyz, view_from_world[2].xyz));
    return normal_to_world * view_normal;
}

fn meadow_pack_geometric_normal(
    gbuffer: vec4<u32>,
    triangle_normal: vec3<f32>,
    enabled: bool,
) -> vec4<u32> {
    var result = gbuffer;
    let magnitude = abs(triangle_normal);
    let scale = max(magnitude.x, max(magnitude.y, magnitude.z));
    // Avoid normalizing a collapsed, non-finite, or numerically tiny
    // derivative cross product. Absence of bit 30 invokes consumer fallback.
    if (!enabled || !(scale > 1e-20) || !all(magnitude < vec3<f32>(1e30))) {
        return result;
    }
    let normal = triangle_normal / scale;
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let wrap = (1.0 - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    let oct = select(wrap, n.xy, n.z >= 0.0) * 0.5 + 0.5;
    let encoded = vec2<u32>(round(clamp(oct, vec2<f32>(0.0), vec2<f32>(1.0)) * 255.0));
    result.b = (result.b & 0xFFFFu) | (encoded.x << 16u) | (encoded.y << 24u);
    result.a |= MEADOW_GBUFFER_GEOMETRIC_NORMAL_BIT;
    return result;
}

// Octahedral coordinates in [0,1]² of the packed payload, in the convention of
// `bevy_render::utils::octahedral_decode`. Valid only when A bit 30 is set.
fn meadow_unpack_geometric_normal_oct(gbuffer_b: u32) -> vec2<f32> {
    return vec2<f32>(f32((gbuffer_b >> 16u) & 255u), f32(gbuffer_b >> 24u)) / 255.0;
}
