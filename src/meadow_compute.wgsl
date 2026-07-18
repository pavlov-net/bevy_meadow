// Meadow GPU-driven cull + compact compute kernel.
//
// Runs once per frame (render-graph node, before the camera driver).
// For every (active patch, view) pair it derives each blade, samples
// the heightfield once, applies the per-view LOD / density / frustum
// gates, and atomically appends survivors into that view's contiguous
// region of `out_blades`. The raster passes then issue one
// `draw_indexed_indirect` per view over its region (see `meadow.wgsl`).
//
// Dispatch: x = active-patch ordinal (indexes `active_patches`, which
// holds the rows of `patches` that are in currently-loaded chunks —
// streaming-correct, unlike iterating all placements), y = view slot.
// One workgroup per (patch, view); threads strip-mine the patch's
// blades in steps of 256.

#import bevy_meadow::shared::{
    VariantParams, CompactedBladeRecord, BladeAttrs, local_blade_position, wind_sway,
}

// ---------- compute-only structs + helpers ----------
// These live here, not in the composable `shared` module, because
// naga_oil forbids exported identifiers that naga's WGSL writer renames
// on round-trip (digit-ending names like `pad0` / `hash_u32`); an entry
// shader has no such restriction.

const MEADOW_VIEW_FLAG_SHADOW: u32 = 1u;

// LOD bands the kernel routes survivors into; MUST equal
// `mesh.rs::MEADOW_MAX_BANDS`. Band 0 = near blade, band 1 = far tuft.
const MEADOW_MAX_BANDS: u32 = 2u;

// Decorrelates a blade slot's tuft-band survival hash from its near-band
// survival hash, so the far band isn't a strict subset of the near band.
const TUFT_SALT: u32 = 0x5C3Bu;

struct PatchData {
    centre_xz_radius_seed: vec4<f32>,
    blade_count_noise_canopy_flags: vec4<f32>,
}

struct TrunkDisc {
    center_radius_fade: vec4<f32>,
}

struct PatchTrunkSlot {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    discs: array<TrunkDisc, 16>,
}

// Matches wgpu `DrawIndexedIndirect` (5 × u32 = 20 B). Static fields
// are CPU-written; `instance_count` is GPU-written by
// `write_instance_counts`.
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// Per-view culling input. `frustum` holds the 6 world-space half-space
// planes (normal.xyz, d).
// `params  = (flags, lod_max, base0, cap0)` — band 0 (near blade) region;
// flags bit 0 = shadow view.
// `params2 = (base1, cap1, shadow_near, _)` — band 1 (far tuft) region,
// plus the per-cascade near radial clip for shadow views.
struct MeadowViewCull {
    frustum: array<vec4<f32>, 6>,
    params: vec4<f32>,
    params2: vec4<f32>,
}

// `views` length MUST equal `mesh.rs::MEADOW_MAX_VIEWS`.
struct MeadowViewCullData {
    count: u32,
    views: array<MeadowViewCull, 6>,
}

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

// Distance-ramped shadow-caster density: full within
// `variant_params.density.z` (shadow_full_dist) of the player, ramping to
// `variant_params.density.w` (shadow_far_density) by the cascade's far
// radial clip. Each shadow cascade owns a distance slice `[shadow_near,
// lod_max]` (see `cull_and_compact`), so a blade casts into ~1 cascade
// rather than all 4. Only band 0 (blades) casts — tufts never do.

// ---------- bind group (compute group 0) ----------

@group(0) @binding(0) var<uniform> variant_params: VariantParams;
@group(0) @binding(1) var<storage, read> patches: array<PatchData>;
@group(0) @binding(2) var<storage, read> trunk_slots: array<PatchTrunkSlot>;
@group(0) @binding(3) var heightfield: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> view_cull: MeadowViewCullData;
@group(0) @binding(5) var<storage, read> active_patches: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_blades: array<CompactedBladeRecord>;
@group(0) @binding(7) var<storage, read_write> cursors: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> indirect: array<DrawIndexedIndirectArgs>;

// ---------- workgroup-aggregated compaction scratch ----------
// One global `atomicAdd` on `cursors[view]` per surviving blade was the
// kernel's single hottest instruction (Nsight: ~38%, contention on 6 global
// counters across thousands of threads). Instead, each 256-blade batch
// claims its survivors' slots with a fast workgroup-shared atomic, then a
// single thread reserves the batch's whole run in the view's region with
// ONE global atomic. Reduces global atomics from O(survivors) to
// O(batches) = O(blade_count / 256) per (patch, view).
var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_base: u32;

// ---------- per-blade derivation (binding-touching; compute-only) ----------

fn derive_blade(blade_idx: u32, p: PatchData) -> BladeAttrs {
    let centre = p.centre_xz_radius_seed.xy;
    let radius = p.centre_xz_radius_seed.z;
    let p_seed = bitcast<u32>(p.centre_xz_radius_seed.w);
    let blade_seed = hash_u32(p_seed ^ (blade_idx * 0x27D4EB2Du));

    var sample = vec2<f32>(0.0, 0.0);
    var dist_norm = 0.0;
    var hit = false;
    for (var attempt = 0u; attempt < 2u; attempt = attempt + 1u) {
        if (hit) {
            continue;
        }
        let h0 = hash_u32(blade_seed + attempt * 17u);
        let h1 = hash_u32(blade_seed + attempt * 17u + 1u);
        let theta = (f32(h0 & 0x00FFFFFFu) / f32(0x01000000u)) * 6.2831853;
        let r_norm = sqrt(f32(h1 & 0x00FFFFFFu) / f32(0x01000000u));
        let candidate = vec2<f32>(cos(theta), sin(theta)) * r_norm * radius;
        let amp = p.blade_count_noise_canopy_flags.y;
        let harmonic = 1.0 + amp *
            (sin(theta * 3.0 + f32(p_seed & 0xFFu) * 0.37) +
             0.5 * sin(theta * 5.0 + f32(p_seed & 0x3Fu) * 1.7));
        let allowed = radius * harmonic;
        if (length(candidate) <= allowed) {
            sample = candidate;
            dist_norm = length(candidate) / max(radius, 1e-3);
            hit = true;
        }
    }

    let yaw = hash01_u32(blade_seed + 7u) * 6.2831853;
    let h_lo = variant_params.height_range.x;
    let h_hi = variant_params.height_range.y;
    let height = h_lo + hash01_u32(blade_seed + 11u) * (h_hi - h_lo);
    let w_lo = variant_params.width_range.x;
    let w_hi = variant_params.width_range.y;
    let width = w_lo + hash01_u32(blade_seed + 13u) * (w_hi - w_lo);
    let clump_seed = hash01_u32(blade_seed + 17u);

    let rim_w = variant_params.lod.z;
    let rim_inner = 1.0 - rim_w;
    var rim_factor = 1.0 - smoothstep(rim_inner, 1.0, dist_norm);
    if (!hit) {
        rim_factor = 0.0;
    }

    return BladeAttrs(centre + sample, yaw, height, width, clump_seed, rim_factor);
}

fn blade_visibility(root_xz: vec2<f32>, patch_idx: u32) -> f32 {
    // Read `count` directly from the storage buffer and bail before
    // touching the discs. The previous `let slot = trunk_slots[patch_idx]`
    // copied the whole 272-byte struct — incl. the 16-disc array — into
    // function-local memory on every blade (the array's dynamic `[i]`
    // index forced a local spill), which Nsight flagged as ~1/3 of the
    // kernel. Indexing `trunk_slots[..].discs[i]` keeps the access in the
    // storage buffer (no copy), and most patches are treeless (count 0).
    let n = min(trunk_slots[patch_idx].count, 16u);
    if (n == 0u) {
        return 1.0;
    }
    var scale: f32 = 1.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let c = trunk_slots[patch_idx].discs[i].center_radius_fade;
        let centre = c.xy;
        let radius = c.z;
        let fade = max(c.w, 1e-4);
        let outer = radius + fade;
        let to_centre = root_xz - centre;
        let d_sq = dot(to_centre, to_centre);
        if (d_sq >= outer * outer) {
            continue;
        }
        let s = smoothstep(radius, outer, sqrt(d_sq));
        scale = min(scale, s);
        if (scale <= 0.0) {
            break;
        }
    }
    return scale;
}

fn sample_heightfield(world_xz: vec2<f32>) -> f32 {
    let extents_min = variant_params.heightfield_extents.xy;
    let extents_max = variant_params.heightfield_extents.zw;
    let extents_size = max(extents_max - extents_min, vec2<f32>(1e-3, 1e-3));
    let dims = vec2<f32>(textureDimensions(heightfield, 0));
    if (dims.x < 2.0 || dims.y < 2.0) {
        return 0.0;
    }
    let uv = clamp((world_xz - extents_min) / extents_size, vec2<f32>(0.0), vec2<f32>(1.0));
    let pixel = uv * (dims - vec2<f32>(1.0));
    let p0 = vec2<i32>(floor(pixel));
    let p1 = vec2<i32>(min(p0.x + 1, i32(dims.x) - 1), p0.y);
    let p2 = vec2<i32>(p0.x, min(p0.y + 1, i32(dims.y) - 1));
    let p3 = vec2<i32>(p1.x, p2.y);
    let t = pixel - vec2<f32>(p0);
    let h0 = textureLoad(heightfield, p0, 0).r;
    let h1 = textureLoad(heightfield, p1, 0).r;
    let h2 = textureLoad(heightfield, p2, 0).r;
    let h3 = textureLoad(heightfield, p3, 0).r;
    let top = mix(h0, h1, t.x);
    let bot = mix(h2, h3, t.x);
    return mix(top, bot, t.y);
}

// Conservative patch bounding-sphere vs. view frustum. Planes are
// normalized (Bevy `Frustum`), point outside a plane when
// `dot(n, c) + d <= -radius`. Order: sides 0-3, near 4, far 5.
// Shadow views skip the near plane (casters between the light and the
// cascade box are outside it but still cast in) and the far plane
// (`SHADOW_MAX_DIST` LOD already bounds shadow grass); main view also
// tests the near plane to drop behind-camera grass.
fn patch_sphere_culled(centre: vec3<f32>, radius: f32, vc: MeadowViewCull, is_shadow: bool) -> bool {
    var plane_count = 4u;
    if (!is_shadow) {
        plane_count = 5u; // include near
    }
    for (var i = 0u; i < plane_count; i = i + 1u) {
        let pl = vc.frustum[i];
        if (dot(pl.xyz, centre) + pl.w <= -radius) {
            return true;
        }
    }
    return false;
}

// ---------- kernels ----------

@compute @workgroup_size(256)
fn cull_and_compact(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let view_slot = wg.y;
    if (view_slot >= view_cull.count) {
        return;
    }
    let patch_idx = active_patches[wg.x];
    let p = patches[patch_idx];
    let blade_count = u32(p.blade_count_noise_canopy_flags.x);
    if (blade_count == 0u) {
        return;
    }

    let vc = view_cull.views[view_slot];
    let is_shadow = (u32(vc.params.x) & MEADOW_VIEW_FLAG_SHADOW) != 0u;

    // Patch-level frustum reject (uniform across the workgroup).
    let centre_xz = p.centre_xz_radius_seed.xy;
    let centre_y = sample_heightfield(centre_xz);
    let max_h = variant_params.height_range.y;
    let edge_noise = p.blade_count_noise_canopy_flags.y;
    let cull_radius = p.centre_xz_radius_seed.z * (1.0 + edge_noise) + max_h * 2.0;
    if (patch_sphere_culled(vec3<f32>(centre_xz.x, centre_y, centre_xz.y), cull_radius, vc, is_shadow)) {
        return;
    }

    let lod_full = variant_params.lod.x;
    let lod_max = vc.params.y;
    let tuft_start = variant_params.tuft.x;
    let dist = length(variant_params.viewer_world_xz.xy - centre_xz);
    let p_seed = bitcast<u32>(p.centre_xz_radius_seed.w);

    // Depth along the MAIN camera's forward axis, from the broadcast
    // view-depth plane `viewer_forward = (forward.xyz, -dot(forward, eye))`:
    // `dot(forward, P) + w` = distance from the camera. Shadow cascade
    // assignment uses THIS, not the planar viewer distance, because
    // receivers pick their cascade by view-space depth — clipping casters by
    // planar distance makes caster/receiver cascades drift apart as the
    // camera orbits, popping near shadows on and off. A uniform (rather than
    // reaching into another view's frustum) keeps this independent of which
    // view is rendering and of view-slot ordering.
    let vf = variant_params.viewer_forward;
    let view_depth =
        dot(vf.xyz, vec3<f32>(centre_xz.x, centre_y, centre_xz.y)) + vf.w;

    // Per-patch LOD band (uniform across the workgroup): near blade (0)
    // inside `tuft_start`, far tuft (1) beyond. A patch is one band at a
    // time — near blades taper to nothing by `tuft_start` while tufts ramp
    // in, so the handoff is sparse on both sides.
    var band = 0u;
    if (dist >= tuft_start) {
        band = 1u;
    }

    if (is_shadow) {
        // Tufts never cast.
        if (band == 1u) {
            return;
        }
        // Coarse-reject whole patches with no part in this cascade's depth
        // slice [shadow_near, lod_max] (camera view-depth). The EXACT clip is
        // per-blade below: a wide patch can span several cascades, and each
        // blade must land in the cascade the receiver under it samples — else
        // near shadows pop as the camera orbits. `cull_radius` bounds the
        // patch's extent so this only drops patches that are fully outside.
        let shadow_near = vc.params2.z;
        if (view_depth + cull_radius < shadow_near || view_depth - cull_radius > lod_max) {
            return;
        }
    }

    // Select the band's region (params.zw = near, params2.xy = far) and its
    // per-(view, band) append cursor.
    var base = u32(vc.params.z);
    var cap = u32(vc.params.w);
    if (band == 1u) {
        base = u32(vc.params2.x);
        cap = u32(vc.params2.y);
    }
    let cursor_idx = view_slot * MEADOW_MAX_BANDS + band;

    // Near-band blade survival tapers across [full_distance, tuft_start];
    // far-band tuft survival ramps tuft.y -> tuft.z across
    // [tuft_start, max_view], with a height fade over the last stretch so the
    // band dissolves into the ground.
    let near_t = clamp((dist - lod_full) / max(tuft_start - lod_full, 1e-3), 0.0, 1.0);
    let near_threshold = 1.0 - near_t;
    let tuft_ramp_t = clamp((dist - tuft_start) / max(lod_max - tuft_start, 1e-3), 0.0, 1.0);
    let tuft_density = mix(variant_params.tuft.y, variant_params.tuft.z, tuft_ramp_t);
    let height_fade_start = variant_params.tuft.w;
    let tuft_height_scale =
        clamp((lod_max - dist) / max(lod_max - height_fade_start, 1e-3), 0.0, 1.0);
    let shadow_full_dist = variant_params.density.z;
    let shadow_far_density = variant_params.density.w;

    // Strip-mine the patch's blades in lockstep 256-wide batches. Lockstep
    // (rather than `b = lid; b += 256`) keeps every thread on the same
    // iteration count — `num_batches` is uniform across the workgroup since
    // `blade_count` is — so the `workgroupBarrier()`s below sit in uniform
    // control flow. Each thread derives at most one survivor per batch and
    // holds it in `rec` across the aggregation barriers.
    let num_batches = (blade_count + 255u) / 256u;
    for (var batch = 0u; batch < num_batches; batch = batch + 1u) {
        let b = batch * 256u + lid;

        var survived = false;
        var rec: CompactedBladeRecord;
        // The final batch runs past `blade_count`; those lanes simply don't
        // survive but still reach the barriers below.
        if (b < blade_count) {
            var survived_gate = false;
            if (band == 0u) {
                if (is_shadow) {
                    // Distance-ramped shadow density: full within
                    // shadow_full_dist, thinning to shadow_far_density by the
                    // cascade's far clip. Uses the same camera view-depth as
                    // the cascade clip so the thinning is stable under camera
                    // orbit. Hash-distributed → a uniform spatial subset.
                    let ramp_t = clamp(
                        (view_depth - shadow_full_dist) / max(lod_max - shadow_full_dist, 1e-3),
                        0.0,
                        1.0,
                    );
                    let shadow_density = mix(1.0, shadow_far_density, ramp_t);
                    survived_gate = hash01_pair(p_seed, b) <= shadow_density;
                } else {
                    // Per-blade LOD hash gate (player-distance, view-independent).
                    survived_gate = hash01_u32(p_seed ^ b) <= near_threshold;
                }
            } else {
                // Far tuft band (main view only): sparse, decorrelated from
                // the near gate via TUFT_SALT.
                survived_gate = hash01_pair(p_seed, b ^ TUFT_SALT) <= tuft_density;
            }
            if (survived_gate) {
                let blade = derive_blade(b, p);
                // Per-blade shadow cascade clip (view-depth): a wide patch can
                // span several cascades, so assign each blade to the cascade
                // whose slice contains it — matching the receiver beneath it.
                var cast_ok = true;
                if (is_shadow) {
                    let bvd = dot(
                        vf.xyz,
                        vec3<f32>(blade.world_xz.x, centre_y, blade.world_xz.y),
                    ) + vf.w;
                    cast_ok = bvd >= vc.params2.z && bvd <= lod_max;
                }
                let collapse_t = blade_visibility(blade.world_xz, patch_idx) * blade.rim_factor;
                if (cast_ok && collapse_t > 1e-4) {
                    let ground_y = sample_heightfield(blade.world_xz) + 0.02;
                    // Far-band tufts fade height to zero near `max_view` so the
                    // band dissolves into the ground instead of ending hard.
                    var h = blade.height;
                    if (band == 1u) {
                        h = h * tuft_height_scale;
                    }
                    rec.world_root_xz = blade.world_xz;
                    rec.ground_y = ground_y;
                    rec.height = h;
                    rec.width = blade.width;
                    rec.yaw = blade.yaw;
                    rec.clump_seed = blade.clump_seed;
                    rec.collapse_t = collapse_t;
                    // Static placement: previous-frame root == current. Motion
                    // comes entirely from the wind delta, recomputed in the VS.
                    rec.prev_root_xz = blade.world_xz;
                    rec.prev_ground_y = ground_y;
                    rec.pad = 0.0;
                    survived = true;
                }
            }
        }

        // Workgroup-aggregated compaction. Claim a local slot in this batch's
        // run via the fast shared atomic, then thread 0 reserves the run in
        // the view's region with a SINGLE global atomic (vs. one per blade).
        workgroupBarrier();
        if (lid == 0u) {
            atomicStore(&wg_count, 0u);
        }
        workgroupBarrier();
        var local_idx = 0u;
        if (survived) {
            local_idx = atomicAdd(&wg_count, 1u);
        }
        workgroupBarrier();
        if (lid == 0u) {
            let m = atomicLoad(&wg_count);
            wg_base = atomicAdd(&cursors[cursor_idx], m);
        }
        workgroupBarrier();
        if (survived) {
            let slot = wg_base + local_idx;
            if (slot < cap) { // safety net; buffers are sized so this never trips
                out_blades[base + slot] = rec;
            }
        }
    }
}

// ---------- raytracing blade expansion ----------
//
// A standalone pass (NOT tied to the raster cull/compact above) that bakes
// the SHADOW-CASTER blade set into triangle vertex buffers for a raytracing
// BLAS, so grass casts shadows under a raytracer (solari) the same way it
// casts them into the shadow cascades in the non-RT path. Runs regardless of
// which raster path (compute or mesh-shader) draws the visible grass — the
// geometry it produces feeds the TLAS, not the raster.
//
// Selection mirrors the non-RT shadow gate (`cull_and_compact`, is_shadow):
// blades within `RT_SHADOW_DIST` of the viewer, thinned by the same
// `shadow_full_dist -> shadow_far_density` ramp, trunk-disc gated,
// `collapse_t`-faded — but as a single radial disc (RT shadow rays travel
// every direction, so no camera-frustum cull and no per-cascade slicing).
//
// Survivors split into two proxy bands so the triangle budget goes where
// shadow-ray precision needs it:
//
// * NEAR (inside `shadow_full_dist`): a 5-vert / 3-tri bent silhouette —
//   base edge, mid edge on the curl curve (lifted half the chord sag so the
//   proxy straddles the true ribbon surface within ±~1 cm), curled tip.
//   Shadow-ray origins sit ~1 cm off the G-buffer surface; a single
//   base-to-tip chord bows up to 0.105·height (several cm) in front of the
//   rendered ribbon and would self-shadow-speckle blades this close.
// * FAR (out to `RT_SHADOW_DIST`): a 1-tri base-to-tip chord, width × 4/3
//   (occlusion-area parity with the tapered ribbon — same trick as the
//   mesh-shader shadow proxy), further thinned toward `RT_FAR_THIN` with the
//   dropped area folded back into blade width. Chord-vs-curve deviation is
//   invisible at this range.
//
// Both gates are pure functions of the stable blade hash + distance, so the
// caster set is identical every frame (no flicker) and the CPU-side survivor
// estimate scales `rt_params` keeps so the counts fit the fixed capacities by
// construction; the cursors only guard the tail. Unused slots are padded
// with NaN positions each frame (inactive primitives — excluded from the
// BLAS entirely). The BLAS is rebuilt each frame over the full capacity.

// Radial shadow-caster extent for RT (metres). Matches `mesh.rs::SHADOW_MAX_DIST`.
const RT_SHADOW_DIST: f32 = 50.0;
// Fixed per-band blade capacities — MUST equal `compute.rs::RT_NEAR_MAX_BLADES`
// / `RT_FAR_MAX_BLADES`. `rt_params` keep-scales bound the expected survivor
// counts below these; the cursors drop the (rare) overflow tail.
const RT_NEAR_MAX_BLADES: u32 = 98304u;
const RT_FAR_MAX_BLADES: u32 = 131072u;
const RT_NEAR_VERTS: u32 = 5u;
const RT_FAR_VERTS: u32 = 3u;
// Extra RT-only thinning of the far band at `RT_SHADOW_DIST` (ramped in from
// 1.0 at the band split). MUST equal `compute.rs::RT_FAR_THIN`.
const RT_FAR_THIN: f32 = 0.5;
// Mid-edge pull-back along the curl axis: half of the per-segment chord sag
// `0.42·h·(1/2)²/4`. Chords of the convex curl curve bow in FRONT of it
// (+z), so pulling the shared mid verts back by half the sag centres each
// segment's deviation (≈ [-0.013·h, +0.020·h]) instead of the un-lifted
// one-sided +0.026·h.
const RT_NEAR_BEND_LIFT: f32 = 0.013125;

// solari packed vertex: a = (pos.xyz, normal.x), b = (normal.yz, uv.xy).
struct RtVertex {
    a: vec4<f32>,
    b: vec4<f32>,
    tangent: vec4<f32>,
}

// x = near keep scale (budget only; raster density is 1.0 in the near band),
// y = far keep scale (budget only; applied on top of the raster ramp and the
//     `RT_FAR_THIN` ramp), z/w unused. Written by the CPU each frame from the
// survivor estimates.
@group(0) @binding(9) var<uniform> rt_params: vec4<f32>;
@group(0) @binding(10) var<storage, read_write> rt_cursor_near: atomic<u32>;
@group(0) @binding(11) var<storage, read_write> rt_verts_near: array<RtVertex>;
@group(0) @binding(12) var<storage, read_write> rt_cursor_far: atomic<u32>;
@group(0) @binding(13) var<storage, read_write> rt_verts_far: array<RtVertex>;

// Two-band workgroup compaction accumulators (the raster kernel's
// `wg_count`/`wg_base` pair, one per band).
var<workgroup> rt_wg_count_near: atomic<u32>;
var<workgroup> rt_wg_base_near: u32;
var<workgroup> rt_wg_count_far: atomic<u32>;
var<workgroup> rt_wg_base_far: u32;

// Tip wind sway for one blade (`shared::wind_sway`, the same implementation
// the raster VS uses, so RT shadows sway in lockstep with the visible
// grass). Sway is linear in `y_norm`, so per-vertex sway is just this
// scaled — computed once per blade, not per vertex.
fn rt_wind_tip(blade: BladeAttrs) -> vec2<f32> {
    return wind_sway(
        blade.world_xz,
        1.0,
        variant_params.wind.z,
        blade.clump_seed,
        variant_params.wind,
        variant_params.wind_direction,
        variant_params.wind_state,
    );
}

// Near-band proxy template. `vi` 0/1 = base edge (ribbon template verts
// 0/1), 2/3 = mid edge (y = 0.5 — not a template vert; the ribbon's taper +
// curl formulas with the curl pulled back `RT_NEAR_BEND_LIFT`), 4 = the
// ribbon's curled tip (template vert 10). `width_boost` folds any near-band
// budget thinning back into aggregate occlusion.
fn rt_near_local(vi: u32, blade: BladeAttrs, width_boost: f32) -> vec3<f32> {
    var local: vec3<f32>;
    if (vi == 4u) {
        local = local_blade_position(10u, blade);
    } else if (vi < 2u) {
        local = local_blade_position(vi, blade);
    } else {
        let side = f32(i32(vi & 1u) * 2 - 1);
        let half_w = 0.5 * (1.0 - 0.25) * blade.width;
        let curl = (0.42 * 0.25 - RT_NEAR_BEND_LIFT) * blade.height;
        local = vec3<f32>(side * half_w, 0.5 * blade.height, curl);
    }
    local.x = local.x * width_boost;
    return local;
}

// Far-band proxy template: the ribbon's base edge + curled tip (template
// verts 0/1/10), one triangle. `width_boost` carries the 4/3 area-parity
// factor plus the reciprocal of the RT-only thinning so aggregate occlusion
// is conserved.
fn rt_far_local(vi: u32, blade: BladeAttrs, width_boost: f32) -> vec3<f32> {
    var local = local_blade_position(select(vi, 10u, vi == 2u), blade);
    local.x = local.x * width_boost;
    return local;
}

// Pack one proxy vertex, reproducing `meadow.wgsl`'s
// local->world+yaw+wind+collapse transform. `cy`/`sy`/`wind_tip` are
// per-blade values hoisted out of the per-vertex loop.
fn rt_pack_vertex(
    local: vec3<f32>,
    blade: BladeAttrs,
    ground_y: f32,
    collapse_t: f32,
    cy: f32,
    sy: f32,
    wind_tip: vec2<f32>,
) -> RtVertex {
    let rotated_x = cy * local.x + sy * local.z;
    let rotated_z = -sy * local.x + cy * local.z;
    let blade_world_xz = blade.world_xz + vec2<f32>(rotated_x, rotated_z);
    let blade_world_y = ground_y + local.y;
    let y_norm = local.y / max(blade.height, 1e-3);
    let target_world_xz = blade_world_xz + wind_tip * y_norm;
    let final_xz = mix(blade.world_xz, target_world_xz, collapse_t);
    let final_y = mix(ground_y, blade_world_y, collapse_t);
    var v: RtVertex;
    // normal = (0,1,0), uv = (0, y_norm), tangent = (1,0,0,1) — matches meadow.wgsl.
    v.a = vec4<f32>(final_xz.x, final_y, final_xz.y, 0.0);
    v.b = vec4<f32>(1.0, 0.0, 0.0, y_norm);
    v.tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return v;
}

@compute @workgroup_size(256)
fn expand_rt_blades(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let patch_idx = active_patches[wg.x];
    let p = patches[patch_idx];
    let blade_count = u32(p.blade_count_noise_canopy_flags.x);
    if (blade_count == 0u) {
        return;
    }

    let viewer = variant_params.viewer_world_xz.xy;
    let centre_xz = p.centre_xz_radius_seed.xy;
    let radius = p.centre_xz_radius_seed.z;
    let edge_noise = p.blade_count_noise_canopy_flags.y;
    // Coarse patch reject: whole patch outside the shadow disc.
    if (length(viewer - centre_xz) - radius * (1.0 + edge_noise) > RT_SHADOW_DIST) {
        return;
    }

    let p_seed = bitcast<u32>(p.centre_xz_radius_seed.w);
    let shadow_full_dist = variant_params.density.z;
    let shadow_far_density = variant_params.density.w;
    let keep_near = rt_params.x;
    let keep_far = rt_params.y;

    let num_batches = (blade_count + 255u) / 256u;
    for (var batch = 0u; batch < num_batches; batch = batch + 1u) {
        let b = batch * 256u + lid;

        var band_near = false;
        var band_far = false;
        var blade: BladeAttrs;
        var ground_y = 0.0;
        var collapse_t = 0.0;
        var near_width_boost = 1.0;
        var far_width_boost = 1.0;
        if (b < blade_count) {
            blade = derive_blade(b, p);
            let dist = length(viewer - blade.world_xz);
            if (dist <= RT_SHADOW_DIST) {
                let near = dist <= shadow_full_dist;
                // Same distance-ramped shadow density as the non-RT shadow
                // gate (planar viewer distance; no cascade view-depth here),
                // times the RT-only far thinning ramp and the CPU budget
                // scale. The budget scale blends near->far with the same
                // ramp and every thinning term is folded back into blade
                // width, so both the caster set and the aggregate occlusion
                // are continuous across the band split.
                var keep = keep_near;
                if (near) {
                    near_width_boost = 1.0 / clamp(keep_near, 1.0 / 3.0, 1.0);
                } else {
                    let ramp_t = clamp(
                        (dist - shadow_full_dist)
                            / max(RT_SHADOW_DIST - shadow_full_dist, 1e-3),
                        0.0,
                        1.0,
                    );
                    let raster_density = mix(1.0, shadow_far_density, ramp_t);
                    let extra_thin = mix(1.0, RT_FAR_THIN, ramp_t);
                    let keep_budget = mix(keep_near, keep_far, ramp_t);
                    keep = raster_density * extra_thin * keep_budget;
                    // Fold the RT-only thinning (not the raster ramp — that
                    // look is the parity target) back into blade width so
                    // aggregate occlusion is conserved. 4/3 = chord-vs-ribbon
                    // silhouette area parity.
                    far_width_boost =
                        (4.0 / 3.0) / clamp(extra_thin * keep_budget, 1.0 / 3.0, 1.0);
                }
                if (hash01_pair(p_seed, b) <= keep) {
                    collapse_t = blade_visibility(blade.world_xz, patch_idx) * blade.rim_factor;
                    if (collapse_t > 1e-4) {
                        ground_y = sample_heightfield(blade.world_xz) + 0.02;
                        band_near = near;
                        band_far = !near;
                    }
                }
            }
        }

        // Workgroup-aggregated compaction, one global atomic per band per
        // batch — mirrors `cull_and_compact`.
        workgroupBarrier();
        if (lid == 0u) {
            atomicStore(&rt_wg_count_near, 0u);
            atomicStore(&rt_wg_count_far, 0u);
        }
        workgroupBarrier();
        var local_idx = 0u;
        if (band_near) {
            local_idx = atomicAdd(&rt_wg_count_near, 1u);
        } else if (band_far) {
            local_idx = atomicAdd(&rt_wg_count_far, 1u);
        }
        workgroupBarrier();
        if (lid == 0u) {
            rt_wg_base_near = atomicAdd(&rt_cursor_near, atomicLoad(&rt_wg_count_near));
            rt_wg_base_far = atomicAdd(&rt_cursor_far, atomicLoad(&rt_wg_count_far));
        }
        workgroupBarrier();
        if (band_near) {
            let slot = rt_wg_base_near + local_idx;
            if (slot < RT_NEAR_MAX_BLADES) {
                let cy = cos(blade.yaw);
                let sy = sin(blade.yaw);
                let wind_tip = rt_wind_tip(blade);
                let base = slot * RT_NEAR_VERTS;
                for (var vi = 0u; vi < RT_NEAR_VERTS; vi = vi + 1u) {
                    rt_verts_near[base + vi] = rt_pack_vertex(
                        rt_near_local(vi, blade, near_width_boost),
                        blade,
                        ground_y,
                        collapse_t,
                        cy,
                        sy,
                        wind_tip,
                    );
                }
            }
        } else if (band_far) {
            let slot = rt_wg_base_far + local_idx;
            if (slot < RT_FAR_MAX_BLADES) {
                let cy = cos(blade.yaw);
                let sy = sin(blade.yaw);
                let wind_tip = rt_wind_tip(blade);
                let base = slot * RT_FAR_VERTS;
                for (var vi = 0u; vi < RT_FAR_VERTS; vi = vi + 1u) {
                    rt_verts_far[base + vi] = rt_pack_vertex(
                        rt_far_local(vi, blade, far_width_boost),
                        blade,
                        ground_y,
                        collapse_t,
                        cy,
                        sy,
                        wind_tip,
                    );
                }
            }
        }
    }
}

// Pad every unused slot in both bands with NaN positions: NaN triangles are
// "inactive" primitives to the BLAS builder — excluded from the tree entirely
// (unlike zeroed verts, which pile degenerate triangles into a BVH blob at
// the origin). Runs as a second pass in the same submission, after
// `expand_rt_blades` has settled the cursors. Also runs alone on frames when
// the expansion inputs are unavailable, so stale geometry vanishes instead of
// freezing at the last-written wind phase.
@compute @workgroup_size(256)
fn rt_pad_unused(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let nan = bitcast<f32>(0x7FC00000u);
    var v: RtVertex;
    v.a = vec4<f32>(nan, nan, nan, 0.0);
    v.b = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    v.tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);

    let used_near = min(atomicLoad(&rt_cursor_near), RT_NEAR_MAX_BLADES);
    if (i >= used_near && i < RT_NEAR_MAX_BLADES) {
        let base = i * RT_NEAR_VERTS;
        for (var k = 0u; k < RT_NEAR_VERTS; k = k + 1u) {
            rt_verts_near[base + k] = v;
        }
    }
    let used_far = min(atomicLoad(&rt_cursor_far), RT_FAR_MAX_BLADES);
    if (i >= used_far && i < RT_FAR_MAX_BLADES) {
        let base = i * RT_FAR_VERTS;
        for (var k = 0u; k < RT_FAR_VERTS; k = k + 1u) {
            rt_verts_far[base + k] = v;
        }
    }
}

// One thread per view; copy each band's compacted survivor count into its
// indirect draw args. Dispatched (1, 1, 1) after compaction. Workgroup
// size MUST equal `mesh.rs::MEADOW_MAX_VIEWS`.
@compute @workgroup_size(6)
fn write_instance_counts(@builtin(local_invocation_index) i: u32) {
    if (i >= view_cull.count) {
        return;
    }
    for (var band = 0u; band < MEADOW_MAX_BANDS; band = band + 1u) {
        let idx = i * MEADOW_MAX_BANDS + band;
        indirect[idx].instance_count = atomicLoad(&cursors[idx]);
    }
}
