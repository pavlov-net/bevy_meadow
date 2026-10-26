// Meadow GPU-driven cull + compact compute kernel.
//
// Runs once per frame (render-graph node, before the camera driver).
// For every (active patch, view) pair it derives each blade, samples
// the heightfield once, applies the per-view LOD / density / frustum
// gates, and atomically appends survivors into that view's contiguous
// region of `out_blades`. The raster passes then issue one
// `draw_indexed_indirect` per view over its region (see `meadow.wesl`).
//
// Dispatch: x = active-patch ordinal (indexes `active_patches`, which
// holds the rows of `patches` that are in currently-loaded chunks —
// streaming-correct, unlike iterating all placements), y = view slot.
// One workgroup per (patch, view); threads strip-mine the patch's
// blades in steps of 256.

import bevy_meadow::meadow_shared::{
    VariantParams, PaletteParams, MEADOW_MAX_PALETTES, CompactedBladeRecord, BladeAttrs,
    MEADOW_CHORD_WIDTH_SCALE, MEADOW_SHADOW_MAX_DIST, wind_sway, hash_u32, hash01_u32, hash01_pair,
    meadow_patch_palette, meadow_max_height_scale, meadow_patch_noise, meadow_blade_height_scale,
    meadow_blade_albedo, meadow_pack_albedo, meadow_gate_count, meadow_band_threshold,
    meadow_template_vertex, meadow_place_vertex, meadow_patch_is_tile, meadow_patch_outer_radius,
    meadow_tile_root, meadow_cover_survives, meadow_sample_cover, meadow_blade_palette,
    meadow_grid_bound, meadow_grid_footprint,
    meadow_blade_rank, MEADOW_MAX_HEIGHT_VARIANCE, MEADOW_MAX_DRYNESS_HEIGHT, MEADOW_NEUTRAL_DRYNESS,
    GatedBlade, meadow_disc_effect, MEADOW_BLADE_GROUND_LIFT,
};

// ---------- compute-only structs ----------
// These live here, not in `meadow_shared`, because only this kernel
// uses them.

const MEADOW_VIEW_FLAG_SHADOW: u32 = 1u;

// LOD bands the kernel routes survivors into; MUST equal
// `mesh.rs::MEADOW_MAX_BANDS`. Band 0 = near blade, band 1 = far tuft.
const MEADOW_MAX_BANDS: u32 = 2u;

struct PatchData {
    centre_xz_radius_seed: vec4<f32>,
    blade_count_noise_canopy_flags: vec4<f32>,
}

struct TrunkDisc {
    center_radius_fade: vec4<f32>,
    profile_frame: vec4<f32>,
    noise_profile: vec4<f32>,
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
// The variant's palette table: patches index it by their flags, tile
// blades by the cover field.
@group(0) @binding(14) var<storage, read> palettes: array<PaletteParams, 64>;
// The cover field atlas tile patches read (`meadow_shared::meadow_sample_cover`).
@group(0) @binding(15) var cover_field: texture_2d<f32>;
// The variant's clearing-profile table (`material.rs::ClearingProfiles`).
@group(0) @binding(16) var clearing_profiles: texture_2d<f32>;

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
    if (meadow_patch_is_tile(p.blade_count_noise_canopy_flags.w)) {
        sample = meadow_tile_root(radius, blade_seed);
        hit = true;
    }
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

    return BladeAttrs(centre + sample, yaw, height, width, clump_seed, rim_factor, blade_seed);
}

// Clearing discs at a blade root for a blade `height` tall growing from
// `ground_y` (`meadow_disc_effect`): x = the share of grass they let stand,
// y = the blade's height under their profiles.
fn blade_visibility(root_xz: vec2<f32>, patch_idx: u32, ground_y: f32, height: f32) -> vec2<f32> {
    // Read `count` directly from the storage buffer and bail before
    // touching the discs. Indexing `trunk_slots[..].discs[i]` keeps the
    // access in the storage buffer: copying the slot struct would spill
    // its disc array to function-local memory on every blade, and most
    // patches have no discs.
    let n = min(trunk_slots[patch_idx].count, 16u);
    var result = vec2<f32>(1.0, height);
    for (var i = 0u; i < n; i = i + 1u) {
        let c = trunk_slots[patch_idx].discs[i].center_radius_fade;
        let d = trunk_slots[patch_idx].discs[i].noise_profile;
        let reach = d.w;
        let to_centre = root_xz - c.xy;
        if (dot(to_centre, to_centre) >= reach * reach) {
            continue;
        }
        let a = trunk_slots[patch_idx].discs[i].profile_frame;
        result = min(
            result,
            meadow_disc_effect(clearing_profiles, c, a, d, root_xz, ground_y, height),
        );
        if (result.x <= 0.0) {
            break;
        }
    }
    return result;
}

// Planar radius bounding the patch's blade roots.
fn patch_outer_radius(p: PatchData) -> f32 {
    return meadow_patch_outer_radius(
        p.centre_xz_radius_seed.z,
        p.blade_count_noise_canopy_flags.y,
        meadow_patch_is_tile(p.blade_count_noise_canopy_flags.w),
    );
}

// Tallest blade the patch can hold: a tile's blades take any palette
// entry, the field's height scale and the wettest ground's gain, a disc's
// its own entry on neutral ground.
fn patch_max_height(p: PatchData) -> f32 {
    let flags = p.blade_count_noise_canopy_flags.w;
    if (meadow_patch_is_tile(flags)) {
        return variant_params.height_range.y * (1.0 + MEADOW_MAX_HEIGHT_VARIANCE)
            * MEADOW_MAX_DRYNESS_HEIGHT * variant_params.cover.z;
    }
    return variant_params.height_range.y
        * meadow_max_height_scale(palettes[meadow_patch_palette(flags)]);
}

// Blade `b` of patch `p` through its per-blade gates: on a tile, cover
// survival under the clearing discs and (with `lod`) the LOD rank at the
// blade's own eye distance against band `band`'s threshold; on a disc
// patch, the clearing discs' collapse; then the rim. A `collapse_t` of 0
// drops the blade.
fn gate_blade(
    blade: BladeAttrs,
    b: u32,
    p: PatchData,
    patch_idx: u32,
    band: u32,
    is_shadow: bool,
    lod_max: f32,
    lod: bool,
) -> GatedBlade {
    var out: GatedBlade;
    let flags = p.blade_count_noise_canopy_flags.w;
    out.palette = meadow_patch_palette(flags);
    out.noise = 0.5;
    out.dryness = MEADOW_NEUTRAL_DRYNESS;
    out.ground_y = sample_heightfield(blade.world_xz) + MEADOW_BLADE_GROUND_LIFT;
    var field_height = 1.0;
    var dist = length(variant_params.viewer_world_xz.xy - p.centre_xz_radius_seed.xy);
    // The discs' clearing gates the blade, their height caps once its height
    // is known.
    var collapse = 1.0;
    if (meadow_patch_is_tile(flags)) {
        let eye = variant_params.viewer_world_xz.xzy;
        dist = distance(eye, vec3<f32>(blade.world_xz.x, out.ground_y, blade.world_xz.y));
        // The rank needs no discs: reject on it first.
        if (lod) {
            let rank = meadow_blade_rank(b, u32(p.blade_count_noise_canopy_flags.x));
            if (rank >= meadow_band_threshold(variant_params, band, is_shadow, dist, lod_max)) {
                out.collapse_t = 0.0;
                return out;
            }
        }
        let cover = meadow_sample_cover(cover_field, variant_params.cover_extents, blade.world_xz);
        let ceiling = p.blade_count_noise_canopy_flags.y;
        let stand = blade_visibility(blade.world_xz, patch_idx, out.ground_y, 0.0).x;
        if (!meadow_cover_survives(variant_params, blade.world_xz, blade.seed, cover.x * stand, ceiling)) {
            out.collapse_t = 0.0;
            return out;
        }
        field_height = cover.y;
        out.dryness = cover.w;
        out.palette = meadow_blade_palette(cover.z, blade.seed, out.palette);
    } else {
        collapse = blade_visibility(blade.world_xz, patch_idx, out.ground_y, 0.0).x;
    }
    out.collapse_t = collapse * blade.rim_factor;
    if (out.collapse_t <= 1e-4) {
        return out;
    }
    let palette = palettes[out.palette];
    out.noise = meadow_patch_noise(palette, blade.world_xz, out.dryness);
    var height = blade.height * meadow_blade_height_scale(palette, out.noise, out.dryness, blade.seed) * field_height;
    if (band == 1u) {
        // Far-band tufts fade height to zero by `lod_max` so the band
        // dissolves into the ground instead of ending hard.
        let fade_start = variant_params.tuft.w;
        height = height * clamp((lod_max - dist) / max(lod_max - fade_start, 1e-3), 0.0, 1.0);
    }
    out.height = blade_visibility(blade.world_xz, patch_idx, out.ground_y, height).y;
    return out;
}

// Ground height at `world_xz`, bilinear; 0 without a heightfield.
fn sample_heightfield(world_xz: vec2<f32>) -> f32 {
    if (!meadow_grid_bound(heightfield)) {
        return 0.0;
    }
    let f = meadow_grid_footprint(heightfield, variant_params.heightfield_extents, world_xz);
    return mix(mix(f.c00.r, f.c10.r, f.t.x), mix(f.c01.r, f.c11.r, f.t.x), f.t.y);
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
    let flags = p.blade_count_noise_canopy_flags.w;
    let tile = meadow_patch_is_tile(flags);
    let outer = patch_outer_radius(p);
    let cull_radius = outer + patch_max_height(p) * 2.0;
    if (patch_sphere_culled(vec3<f32>(centre_xz.x, centre_y, centre_xz.y), cull_radius, vc, is_shadow)) {
        return;
    }

    let lod_max = vc.params.y;
    let tuft_start = variant_params.tuft.x;
    let dist = length(variant_params.viewer_world_xz.xy - centre_xz);
    // A tile gates its prefix at its nearest root and each blade again at
    // its own eye distance (`gate_blade`).
    var gate_dist = dist;
    if (tile) {
        gate_dist = max(dist - outer, 0.0);
    }

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

    let threshold = meadow_band_threshold(variant_params, band, is_shadow, gate_dist, lod_max);
    let survivors = meadow_gate_count(threshold, blade_count);

    // Strip-mine the surviving prefix in lockstep 256-wide batches.
    // Lockstep (rather than `b = lid; b += 256`) keeps every thread on the
    // same iteration count — `num_batches` is uniform across the workgroup
    // since `survivors` is — so the `workgroupBarrier()`s below sit in
    // uniform control flow. Each thread derives at most one survivor per
    // batch and holds it in `rec` across the aggregation barriers.
    let num_batches = (survivors + 255u) / 256u;
    for (var batch = 0u; batch < num_batches; batch = batch + 1u) {
        let b = batch * 256u + lid;

        var survived = false;
        var rec: CompactedBladeRecord;
        // The final batch runs past `survivors`; those lanes simply don't
        // survive but still reach the barriers below.
        if (b < survivors) {
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
            var gated: GatedBlade;
            if (cast_ok) {
                gated = gate_blade(blade, b, p, patch_idx, band, is_shadow, lod_max, true);
            }
            if (cast_ok && gated.collapse_t > 1e-4) {
                rec.world_root_xz = blade.world_xz;
                rec.ground_y = gated.ground_y;
                rec.height = gated.height;
                rec.width = blade.width;
                rec.yaw = blade.yaw;
                rec.clump_seed = blade.clump_seed;
                rec.collapse_t = gated.collapse_t;
                // Static placement: previous-frame root == current. Motion
                // comes entirely from the wind delta, recomputed in the VS.
                rec.prev_root_xz = blade.world_xz;
                rec.prev_ground_y = gated.ground_y;
                // Shadow views draw depth only; their records skip the colour.
                rec.albedo = 0u;
                if (!is_shadow) {
                    rec.albedo = meadow_pack_albedo(
                        meadow_blade_albedo(palettes[gated.palette], gated.noise, gated.dryness, blade.seed),
                    );
                }
                survived = true;
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
// Selection is CPU-side (`rt_selection.rs`) and independent of the raster
// shadow gate: persistent blade slots, nearest first. The near band uses
// exact visible ribbons; far slots use single-triangle chords. In the
// overlap a stable hash chooses one representation. No atomic compaction
// changes triangle ownership while wind moves the geometry.

// Fade caster density to zero before the hard range limit. This thinning
// is NOT compensated by width, which would cancel the intended fade.
const RT_SHADOW_FADE_START: f32 = 35.0;
// Fixed per-band blade capacities — MUST equal `compute.rs::RT_NEAR_MAX_BLADES`
// / `RT_FAR_MAX_BLADES`. CPU slot allocation enforces these exact limits.
const RT_NEAR_MAX_BLADES: u32 = 32768u;
const RT_FAR_MAX_BLADES: u32 = 131072u;
const RT_NEAR_VERTS: u32 = 11u;
const RT_FAR_VERTS: u32 = 3u;
// solari packed vertex: a = (pos.xyz, normal.x), b = (normal.yz, uv.xy).
struct RtVertex {
    a: vec4<f32>,
    b: vec4<f32>,
    tangent: vec4<f32>,
}

// CPU-selected persistent slots: (patch index, blade index), MAX = vacant.
// `rt_params[0].xy` delimit the near/far transition around the stable
// selection centre `rt_params[1].xyz`.
@group(0) @binding(9) var<uniform> rt_params: array<vec4<f32>, 2>;
@group(0) @binding(10) var<storage, read> rt_slots_near: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read_write> rt_verts_near: array<RtVertex>;
@group(0) @binding(12) var<storage, read> rt_slots_far: array<vec2<u32>>;
@group(0) @binding(13) var<storage, read_write> rt_verts_far: array<RtVertex>;

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

// Pack one caster vertex: a template vertex (`meadow_template_vertex`)
// placed like the raster paths place it. `cy`/`sy`/`wind_tip` are
// per-blade values hoisted out of the per-vertex loop.
fn rt_pack_vertex(
    local: vec4<f32>,
    blade: BladeAttrs,
    ground_y: f32,
    collapse_t: f32,
    cy: f32,
    sy: f32,
    wind_tip: vec2<f32>,
    palette_u: f32,
) -> RtVertex {
    let world = meadow_place_vertex(local, blade.world_xz, ground_y, cy, sy, wind_tip, collapse_t);
    var v: RtVertex;
    // normal = (0,1,0), uv = (palette_u, y_norm), tangent = (1,0,0,1). The
    // caster material's base colour texture holds one texel per palette
    // entry, so `palette_u` (the entry's texel centre) gives each blade its
    // patch's mean colour.
    v.a = vec4<f32>(world, 0.0);
    v.b = vec4<f32>(1.0, 0.0, palette_u, local.w);
    v.tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return v;
}

fn rt_empty_vertex() -> RtVertex {
    var v: RtVertex;
    v.a = vec4<f32>(vec3<f32>(bitcast<f32>(0x7FC00000u)), 0.0);
    v.b = vec4<f32>(0.0);
    v.tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return v;
}

// Every invocation owns a fixed slot. No atomic packing: wind cannot
// reorder triangle identities underneath Solari's temporal GI anchors.
fn rt_write_slot(slot: u32, key: vec2<u32>, near: bool) {
    let count = select(RT_FAR_VERTS, RT_NEAR_VERTS, near);
    var blade: BladeAttrs;
    var ground_y = 0.0;
    var collapse = 0.0;
    var palette_u = 0.0;
    let occupied = key.x != 0xFFFFFFFFu;
    if (occupied) {
        let p = patches[key.x];
        blade = derive_blade(key.y, p);
        let gated = gate_blade(blade, key.y, p, key.x, 0u, false, 0.0, false);
        blade.height = gated.height;
        ground_y = gated.ground_y;
        palette_u = (f32(gated.palette) + 0.5) / f32(MEADOW_MAX_PALETTES);
        let dist = distance(rt_params[1].xyz, vec3<f32>(blade.world_xz.x, ground_y, blade.world_xz.y));
        let transition =
            smoothstep(rt_params[0].x, max(rt_params[0].y, rt_params[0].x + 0.01), dist);
        let seed = bitcast<u32>(p.centre_xz_radius_seed.w);
        let h = hash01_pair(seed, key.y);
        // The same threshold selects exactly one representation in overlap.
        let chosen = select((h < transition), (h >= transition), near);
        let fade = 1.0 - smoothstep(RT_SHADOW_FADE_START, MEADOW_SHADOW_MAX_DIST, dist);
        let fade_hash = hash01_pair(seed ^ 0xA511E9B3u, key.y);
        if (chosen && fade_hash < fade) {
            collapse = gated.collapse_t;
        }
    }
    let cy = cos(blade.yaw);
    let sy = sin(blade.yaw);
    let wind_tip = rt_wind_tip(blade);
    for (var vi = 0u; vi < count; vi += 1u) {
        var vertex = rt_empty_vertex();
        if (occupied) {
            // Far slots are one proxy triangle: the ribbon's base edge and
            // tip (template verts 0/1/10), widened to the ribbon's
            // silhouette area (`MEADOW_CHORD_WIDTH_SCALE`). Area parity
            // only: the distance fade thins casters and no widening
            // replaces them.
            var template_vert = vi;
            if (!near) {
                template_vert = select(vi, 10u, vi == 2u);
            }
            var local = meadow_template_vertex(template_vert, false, blade);
            if (!near) {
                local.x = local.x * MEADOW_CHORD_WIDTH_SCALE;
            }
            // Suppressed owners retain finite, collapsed vertices. History
            // cannot accidentally sample NaN positions during a fade.
            vertex = rt_pack_vertex(local, blade, ground_y, collapse, cy, sy, wind_tip, palette_u);
        }
        if (near) { rt_verts_near[slot * count + vi] = vertex; }
        else { rt_verts_far[slot * count + vi] = vertex; }
    }
}

@compute @workgroup_size(256)
fn expand_rt_blades(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot < RT_NEAR_MAX_BLADES) { rt_write_slot(slot, rt_slots_near[slot], true); }
    if (slot < RT_FAR_MAX_BLADES) { rt_write_slot(slot, rt_slots_far[slot], false); }
}

// Clear on missing inputs so streaming cannot leave stale shadow geometry.
@compute @workgroup_size(256)
fn rt_pad_unused(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    let vertex = rt_empty_vertex();
    if (slot < RT_NEAR_MAX_BLADES) {
        for (var vi = 0u; vi < RT_NEAR_VERTS; vi += 1u) {
            rt_verts_near[slot * RT_NEAR_VERTS + vi] = vertex;
        }
    }
    if (slot < RT_FAR_MAX_BLADES) {
        for (var vi = 0u; vi < RT_FAR_VERTS; vi += 1u) {
            rt_verts_far[slot * RT_FAR_VERTS + vi] = vertex;
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
