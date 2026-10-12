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
    // metres, 0 = no patches), z = patch-noise threshold, w = 1 / width of
    // the soft patch edge.
    variance: vec4<f32>,
}

// Mirror of `material.rs::VariantParams` (11 × vec4 = 176 B). Bound at
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
    viewer_world_xz: vec4<f32>,
    tuft: vec4<f32>,
    viewer_forward: vec4<f32>,
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
// toward a pixel and per-blade normals would shimmer.
const MEADOW_BLADE_NORMAL_FADE: vec2<f32> = vec2<f32>(15.0, 45.0);

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
// - near blades (band 0) taper 1 -> 0 across [full_distance, tuft_start];
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
    return 1.0 - clamp((dist - lod_full) / max(tuft_start - lod_full, 1e-3), 0.0, 1.0);
}

// ---------- per-blade palette variance ----------
//
// Every blade takes its patch's palette entry and varies around it with
// zero mean, so the coverage-weighted mean colour of a patch is
// `mix(live, dry, dead_share)` and its mean height the unvaried one:
// - dead weight: a blade is dead where low-frequency world-space noise at
//   its root crosses the patch threshold (small dead patches) or where its
//   own random falls under the single-blade threshold. The CPU splits the
//   dead share between the two so the expected weight equals it.
// - height: `1 + variance * (blade + patch) / 2`, both terms in [-1, 1]
//   from the same two sources, so dead blades and dead patches run short.
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

// Quintic value noise remapped to a uniform [0, 1] distribution. The raw
// noise has mean 0.5 and a bell-shaped spread; the odd polynomial in
// `n - 0.5` is a least-squares fit of its CDF (max error 0.0015), so a
// threshold at `1 - a` covers a share `a` of the ground.
fn meadow_uniform_noise(p: vec2<f32>) -> f32 {
    // Rotated off the lattice axes so patches don't line up with the world grid.
    let q = vec2<f32>(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y);
    let cell = vec2<i32>(floor(q));
    let f = q - floor(q);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let x0 = bitcast<u32>(cell.x);
    let z0 = bitcast<u32>(cell.y);
    let a = hash01_pair(x0, z0);
    let b = hash01_pair(x0 + 1u, z0);
    let c = hash01_pair(x0, z0 + 1u);
    let d = hash01_pair(x0 + 1u, z0 + 1u);
    let n = mix(mix(a, b, u.x), mix(c, d, u.x), u.y) - 0.5;
    let n2 = n * n;
    return clamp(0.5 + n * (1.49767 + n2 * (-2.16135 + n2 * 0.73196)), 0.0, 1.0);
}

// Dead-patch noise at a blade root, shared by its height and colour: the
// uniform noise, or 0.5 (no patch) when the entry has no patches.
fn meadow_patch_noise(palette: PaletteParams, root_xz: vec2<f32>) -> f32 {
    if (palette.variance.y <= 0.0) {
        return 0.5;
    }
    return meadow_uniform_noise(root_xz * palette.variance.y);
}

// A blade's height scale from its patch noise and seed.
fn meadow_blade_height_scale(palette: PaletteParams, noise: f32, seed: u32) -> f32 {
    let blade_random = hash01_u32(seed + 19u);
    return 1.0 + palette.variance.x * 0.5 * ((2.0 * blade_random - 1.0) + (1.0 - 2.0 * noise));
}

// A blade's linear albedo from its patch noise and seed.
fn meadow_blade_albedo(palette: PaletteParams, noise: f32, seed: u32) -> vec3<f32> {
    // The soft patch ramp's expected value over the uniform noise is the
    // patch share exactly (its edge stays inside [0, 1]).
    let patch_dead = clamp((noise - palette.variance.z) * palette.variance.w + 0.5, 0.0, 1.0);
    let blade_dead = select(0.0, 1.0, hash01_u32(seed + 19u) < palette.dry_blade_share.w);
    let base = mix(palette.live_jitter.rgb, palette.dry_blade_share.rgb, max(patch_dead, blade_dead));
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
