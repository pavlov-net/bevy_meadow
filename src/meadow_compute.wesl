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
    MEADOW_CHORD_WIDTH_SCALE, MEADOW_SHADOW_MAX_DIST, MEADOW_RT_SHADOW_FADE_START, wind_sway,
    hash_u32, hash01_u32, hash01_pair,
    meadow_patch_palette, meadow_max_height_scale, meadow_patch_noise, meadow_blade_height_scale,
    meadow_blade_albedo, meadow_pack_albedo, meadow_gate_count, meadow_band_threshold,
    meadow_template_vertex, meadow_place_vertex, meadow_blade_face_normal, meadow_patch_is_tile,
    meadow_patch_outer_radius, meadow_tile_root, meadow_cover_survives, meadow_sample_cover,
    meadow_blade_palette,
    meadow_grid_bound, meadow_grid_footprint,
    meadow_blade_rank, MEADOW_MAX_HEIGHT_VARIANCE, MEADOW_MAX_DRYNESS_HEIGHT, MEADOW_NEUTRAL_DRYNESS,
    GatedBlade, meadow_disc_effect, meadow_lod_widen, MEADOW_BLADE_GROUND_LIFT,
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
    let tile = meadow_patch_is_tile(flags);
    if (tile) {
        let eye = variant_params.viewer_world_xz.xzy;
        dist = distance(eye, vec3<f32>(blade.world_xz.x, out.ground_y, blade.world_xz.y));
    }
    // A disc patch's prefix gate keeps its share; a tile ranks each blade at
    // its own distance, before the discs, which the rank does not need.
    out.width_scale = 1.0;
    if (lod) {
        let keep = meadow_band_threshold(variant_params, band, is_shadow, dist, lod_max);
        if (tile && meadow_blade_rank(b, u32(p.blade_count_noise_canopy_flags.x)) >= keep) {
            out.collapse_t = 0.0;
            return out;
        }
        out.width_scale = meadow_lod_widen(keep, band, variant_params.lod.w);
    }
    if (tile) {
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
    // time: near blades thin to `1 / max_widen` by `tuft_start`, where the
    // tufts take over.
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
                rec.width = blade.width * gated.width_scale;
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
// The selection below (`rt_select_*`) fills each band's persistent slots,
// least view-weighted distance first, independently of the raster shadow
// gate. The near band uses exact visible ribbons; far slots use
// single-triangle chords. Each frame this pass decides from the current
// pose which resident blades cast, by the rules the selection holds its
// slots for: a blade's view-weighted distance against the near band's
// reach, and its LOD rank against the far band's keep share. In the
// hand-over between the bands a stable hash chooses one representation. No
// atomic compaction changes triangle ownership while wind moves the geometry.

// Fixed per-band blade capacities. Slots run near band first.
const RT_NEAR_MAX_BLADES: u32 = 32768u;
const RT_FAR_MAX_BLADES: u32 = 262144u;
const RT_SLOTS: u32 = RT_NEAR_MAX_BLADES + RT_FAR_MAX_BLADES;
const RT_NEAR_VERTS: u32 = 11u;
const RT_FAR_VERTS: u32 = 3u;
// `RtSlot::flags`: the expansion drew the slot's blade last frame; the
// slot's owner changed since, so it has no motion history.
const RT_SLOT_DRAWN: u32 = 1u;
const RT_SLOT_FRESH: u32 = 2u;
// A blade straight behind the view ranks like one this many times nearer
// inside it.
const RT_OUTSIDE_VIEW_WEIGHT: f32 = 3.0;
// How the far band thins across its range: its keep share before the
// budget term is `(1 - x * ramp) * (1 - y * ramp)`.
const RT_FAR_THINNING: vec2<f32> = vec2<f32>(0.75, 0.5);
// The near band hands its blades over to the far band from this share of
// its reach on.
const RT_NEAR_HANDOVER: f32 = 0.75;

// solari packed vertex: a = (pos.xyz, normal.x), b = (normal.yz, uv.xy).
struct RtVertex {
    a: vec4<f32>,
    b: vec4<f32>,
    tangent: vec4<f32>,
}

// A persistent blade slot.
struct RtSlot {
    // Patch index + 1, 0 for a vacant slot.
    patch_id: u32,
    blade: u32,
    // `RT_SLOT_DRAWN` | `RT_SLOT_FRESH`.
    flags: u32,
    unused: u32,
}

// `compute.rs::RtParams`. The reaches are view-weighted distances
// (`rt_rank_distance`).
struct RtParams {
    // (hand-over start, near reach, far budget term, _), written by
    // `rt_ease_drawn`.
    drawn: vec4<f32>,
    // xyz = eye, w = cosine of the view cone's half-angle.
    eye: vec4<f32>,
    // xyz = unit view direction, zero to rank blades by eye distance alone;
    // w = sunward widening cap.
    forward: vec4<f32>,
    // x = share of the way the drawn limits ease toward the selection's
    // this frame.
    ease: vec4<f32>,
}

@group(0) @binding(9) var<uniform> rt_params: RtParams;
// The slots, near band first.
@group(0) @binding(10) var<storage, read_write> rt_slots: array<RtSlot>;
@group(0) @binding(11) var<storage, read_write> rt_verts_near: array<RtVertex>;
@group(0) @binding(12) var<storage, read_write> rt_previous_near: array<RtVertex>;
@group(0) @binding(13) var<storage, read_write> rt_verts_far: array<RtVertex>;
@group(0) @binding(23) var<storage, read_write> rt_previous_far: array<RtVertex>;

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

// Pack one caster vertex at `world` from template vertex `local`, with
// `normal` its front face's unit normal (the counter-clockwise side).
// Solari lights a hit with the interpolated vertex normal and offsets its
// rays along it, so it must be the blade's surface: the material is
// double-sided, and a hit on the back face flips it. uv = (palette_u,
// y_norm): the caster material's base colour texture holds one texel per
// palette entry, so `palette_u` (the entry's texel centre) gives each blade
// its patch's mean colour. The tangent is the blade's across axis.
fn rt_pack_vertex(world: vec3<f32>, local: vec4<f32>, normal: vec3<f32>, cy: f32, sy: f32, palette_u: f32) -> RtVertex {
    var v: RtVertex;
    v.a = vec4<f32>(world, normal.x);
    v.b = vec4<f32>(normal.yz, palette_u, local.w);
    v.tangent = vec4<f32>(cy, 0.0, -sy, 1.0);
    return v;
}

// The far band's share of live blades before its budget term at a
// view-weighted distance `beyond` past the near band's reach `end`, `ramp`
// rising to 1 at `MEADOW_SHADOW_MAX_DIST`.
fn rt_far_thinning(beyond: f32, end: f32) -> f32 {
    let ramp = clamp(beyond / max(MEADOW_SHADOW_MAX_DIST - end, 1e-3), 0.0, 1.0);
    return (1.0 - RT_FAR_THINNING.x * ramp) * (1.0 - RT_FAR_THINNING.y * ramp);
}

// Share of live blades the far band keeps at view-weighted distance `dist`,
// past the near band's reach `end`, under budget term `budget`: full up to
// `end`, thinning with distance, the budget term `1 - exp(-budget /
// beyond²)` approaching 1 continuously at `end`. Rises with `end` and
// `budget` and falls with `dist`.
fn rt_far_keep_share(dist: f32, end: f32, budget: f32) -> f32 {
    let beyond = max(dist - end, 0.0);
    if (beyond <= 0.0) {
        return 1.0;
    }
    return rt_far_thinning(beyond, end) * (1.0 - exp(-budget / (beyond * beyond)));
}

// The view weight of a direction at cosine `c` from the view, for a view
// cone of cosine `cone_cos`: 1 inside the cone, rising linearly with the
// cosine outside it to `RT_OUTSIDE_VIEW_WEIGHT` straight behind.
fn rt_view_weight(cone_cos: f32, c: f32) -> f32 {
    let outside = clamp((cone_cos - c) / (cone_cos + 1.0), 0.0, 1.0);
    return 1.0 + (RT_OUTSIDE_VIEW_WEIGHT - 1.0) * outside;
}

// `point`'s eye distance times its view weight from the current pose.
fn rt_rank_distance(point: vec3<f32>) -> f32 {
    let dir = point - rt_params.eye.xyz;
    let r = length(dir);
    let forward = rt_params.forward.xyz;
    if (r <= 0.0 || dot(forward, forward) == 0.0) {
        return r;
    }
    return r * rt_view_weight(rt_params.eye.w, dot(dir, forward) / r);
}

fn rt_empty_vertex() -> RtVertex {
    var v: RtVertex;
    v.a = vec4<f32>(vec3<f32>(bitcast<f32>(0x7FC00000u)), 0.0);
    v.b = vec4<f32>(0.0);
    v.tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return v;
}

// A slot's blade, derived and gated as the raster paths derive it.
struct RtBlade {
    blade: BladeAttrs,
    gated: GatedBlade,
    root: vec3<f32>,
    // Its place in the LOD prefix order of its patch's blades.
    rank: f32,
    patch_seed: u32,
}

fn rt_blade(patch_idx: u32, b: u32) -> RtBlade {
    let p = patches[patch_idx];
    var x: RtBlade;
    x.blade = derive_blade(b, p);
    x.gated = gate_blade(x.blade, b, p, patch_idx, 0u, false, 0.0, false);
    x.blade.height = x.gated.height;
    x.root = vec3<f32>(x.blade.world_xz.x, x.gated.ground_y, x.blade.world_xz.y);
    x.rank = meadow_blade_rank(b, u32(p.blade_count_noise_canopy_flags.x));
    x.patch_seed = bitcast<u32>(p.centre_xz_radius_seed.w);
    return x;
}

fn rt_live(x: RtBlade) -> bool {
    return x.gated.collapse_t > 1e-4;
}

// Every invocation owns a fixed slot. No atomic packing: wind cannot
// reorder triangle identities underneath Solari's temporal GI anchors.
// The slot's previous vertices take its current ones, or the new ones when
// its owner changed; a slot that stays collapsed keeps both.
fn rt_write_slot(slot: u32, near: bool) {
    let count = select(RT_FAR_VERTS, RT_NEAR_VERTS, near);
    let first = select(slot - RT_NEAR_MAX_BLADES, slot, near) * count;
    let s = rt_slots[slot];
    let occupied = s.patch_id != 0u;
    let fresh = (s.flags & RT_SLOT_FRESH) != 0u;
    var x: RtBlade;
    var collapse = 0.0;
    var widen = 1.0;
    if (occupied) {
        x = rt_blade(s.patch_id - 1u, s.blade);
        let dist = distance(rt_params.eye.xyz, x.root);
        let rank_distance = rt_rank_distance(x.root);
        // Past the near reach the transition reaches 1, which no hash
        // passes in the near band.
        let transition = smoothstep(
            rt_params.drawn.x,
            max(rt_params.drawn.y, rt_params.drawn.x + 0.01),
            rank_distance,
        );
        let h = hash01_pair(x.patch_seed, s.blade);
        // The same threshold selects exactly one representation in overlap.
        var chosen = select((h < transition), (h >= transition), near);
        if (!near) {
            let keep = rt_far_keep_share(rank_distance, rt_params.drawn.y, rt_params.drawn.z);
            chosen = chosen && x.rank < keep;
            // A chord wider than its blade stands in the gaps the drawn
            // blades are seen through. Chords of blades lying toward the
            // sun from the eye, whatever the view direction, widen up to
            // the sunward cap; the rest up to the LOD's.
            let sunward = saturate(dot(normalize(x.root - rt_params.eye.xyz), variant_params.sun.xyz));
            widen = mix(
                meadow_lod_widen(keep, 0u, variant_params.lod.w),
                meadow_lod_widen(keep, 0u, max(rt_params.forward.w, variant_params.lod.w)),
                sunward,
            );
        }
        let fade = 1.0 - smoothstep(MEADOW_RT_SHADOW_FADE_START, MEADOW_SHADOW_MAX_DIST, dist);
        let fade_hash = hash01_pair(x.patch_seed ^ 0xA511E9B3u, s.blade);
        if (chosen && fade_hash < fade) {
            collapse = x.gated.collapse_t;
        }
    }
    let drawn = collapse > 0.0;
    rt_slots[slot].flags = select(0u, RT_SLOT_DRAWN, drawn);
    if (!fresh && !drawn && (s.flags & RT_SLOT_DRAWN) == 0u) {
        return;
    }
    let blade = x.blade;
    let cy = cos(blade.yaw);
    let sy = sin(blade.yaw);
    let wind_tip = rt_wind_tip(blade);
    let palette_u = (f32(x.gated.palette) + 0.5) / f32(MEADOW_MAX_PALETTES);
    // Far slots are one proxy triangle: the ribbon's base edge and tip
    // (template verts 0/1/10), widened to the ribbon's silhouette area
    // (`MEADOW_CHORD_WIDTH_SCALE`) and by `widen` to keep the area of the
    // blades the far band's thinning drops. The distance fade thins casters
    // and no widening replaces them. Suppressed owners keep finite,
    // collapsed vertices, so history never samples NaN positions in a fade.
    var local: array<vec4<f32>, 11>;
    var world: array<vec3<f32>, 11>;
    for (var vi = 0u; vi < count; vi += 1u) {
        let template_vert = select(select(vi, 10u, vi == 2u), vi, near);
        local[vi] = meadow_template_vertex(template_vert, false, blade);
        if (!near) {
            local[vi].x *= MEADOW_CHORD_WIDTH_SCALE * widen;
        }
        world[vi] = meadow_place_vertex(local[vi], blade.world_xz, x.gated.ground_y, cy, sy, wind_tip, collapse);
    }
    // A collapsed blade faces down, away from every receiver above its
    // root. Solari moves a GI sample anchored on it to the collapsed
    // triangle, where its reconnection fails and the sample drops, so a slot
    // passes to another blade free of samples gathered on this one.
    var chord_normal = vec3<f32>(0.0, -1.0, 0.0);
    if (drawn && !near) {
        let e1 = world[1] - world[0];
        let e2 = world[2] - world[0];
        let n = cross(e1, e2);
        chord_normal = vec3<f32>(0.0, 1.0, 0.0);
        if (dot(n, n) > 1e-12 * dot(e1, e1) * dot(e2, e2)) {
            chord_normal = normalize(n);
        }
    }
    for (var vi = 0u; vi < count; vi += 1u) {
        var vertex = rt_empty_vertex();
        if (occupied) {
            var normal = chord_normal;
            if (drawn && near) {
                normal = meadow_blade_face_normal(vi, blade, cy, sy, wind_tip);
            }
            vertex = rt_pack_vertex(world[vi], local[vi], normal, cy, sy, palette_u);
        }
        // A fresh slot's blade has no history.
        let at = first + vi;
        if (near) {
            var previous = vertex;
            if (!fresh) {
                previous = rt_verts_near[at];
            }
            rt_previous_near[at] = previous;
            rt_verts_near[at] = vertex;
        } else {
            var previous = vertex;
            if (!fresh) {
                previous = rt_verts_far[at];
            }
            rt_previous_far[at] = previous;
            rt_verts_far[at] = vertex;
        }
    }
}

@compute @workgroup_size(256)
fn expand_rt_blades(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot < RT_NEAR_MAX_BLADES) { rt_write_slot(slot, true); }
    if (slot < RT_FAR_MAX_BLADES) { rt_write_slot(RT_NEAR_MAX_BLADES + slot, false); }
}

// Clear on missing inputs so streaming cannot leave stale shadow geometry.
@compute @workgroup_size(256)
fn rt_pad_unused(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    let vertex = rt_empty_vertex();
    if (slot < RT_NEAR_MAX_BLADES) {
        for (var vi = 0u; vi < RT_NEAR_VERTS; vi += 1u) {
            rt_verts_near[slot * RT_NEAR_VERTS + vi] = vertex;
            rt_previous_near[slot * RT_NEAR_VERTS + vi] = vertex;
        }
    }
    if (slot < RT_FAR_MAX_BLADES) {
        for (var vi = 0u; vi < RT_FAR_VERTS; vi += 1u) {
            rt_verts_far[slot * RT_FAR_VERTS + vi] = vertex;
            rt_previous_far[slot * RT_FAR_VERTS + vi] = vertex;
        }
    }
}

// ---------- raytracing caster selection ----------
//
// Fills each RT band's persistent slots the frame the CPU finds a selection
// due (`rt_selection.rs`). Every live blade the CPU's scan lists ranks by
// `rt_least_rank_distance`: a lower bound on its view-weighted distance from
// any pose within the selection's margins. The near band keeps the blades
// of least rank. The far band keeps a blade past the near band's reach
// while its LOD rank is under the keep share (`rt_far_keep_share`), and
// ranks blades by the least budget term that keeps them
// (`rt_least_budget`). Each band's limit is fitted to its capacity from a
// histogram of its keys, and blades are selected by histogram bin, so the
// counts and the choice agree.
//
// A selected blade keeps its slot (`rt_select_index` builds an owner table).
// A slot whose blade the expansion drew last frame keeps it while
// unselected, counted against the capacity by the fit. The rest are vacated
// (`rt_select_evict`) for the newly selected (`rt_select_place`), so a slot
// only passes between blades neither of which is drawn.
//
// Passes, in order: `rt_select_purge` (when patches left, or to clear),
// `rt_select_index`, `rt_select_count_near`, `rt_select_fit_near`,
// `rt_select_drawn_far`, `rt_select_count_far`, `rt_select_fit_far`,
// `rt_select_evict`, `rt_select_place`; then, every frame, `rt_ease_drawn`.

// Bins per key histogram, a multiple of 256.
const RT_SELECT_BINS: u32 = 16384u;
// Workgroups per dispatch row of a scan, the most a dimension holds.
const RT_SCAN_ROW: u32 = 65535u;
// The far band's budget term standing for "every eligible blade fits".
const RT_UNBOUNDED_BUDGET: f32 = 1.0e6;
// log2 of the far band's smallest nonzero bin edge.
const RT_LOG_BUDGET_MIN: f32 = -12.0;
// Bit of a coverage entry's end marking a patch the near band scans.
const RT_COVERAGE_NEAR: u32 = 0x80000000u;
// How far past the last selection's near reach and far budget term the
// next one's scans count blades one by one, by band.
const RT_BOUND_SLACK: vec2<f32> = vec2<f32>(1.5, 4.0);
// The key of a blade a band cannot hold.
const RT_KEY_NONE: f32 = -1.0;
const RT_TABLE_TOMBSTONE: u32 = 0xFFFFFFFFu;

// `compute.rs::RtSelectParams`.
struct RtSelectParams {
    // xyz = eye, w = the near band's largest reach.
    eye: vec4<f32>,
    // xyz = unit view direction, zero to rank blades by eye distance alone;
    // w = the cosine of the view cone's half-angle.
    forward: vec4<f32>,
    // x = margin distance, y/z = sine/cosine of the margin angle.
    margin: vec4<f32>,
    // x = near-band scan chunks, y = coverage entries, z = purged patches,
    // w = 1 when the far band selects, + 2 when every slot is vacated first.
    counts: vec4<u32>,
    // x = scan chunks, y = the selection's serial.
    run: vec4<u32>,
}

// `compute.rs::RtSelectStateGpu`. Per-band pairs run near band first. Each
// selection resets the words before `drawn`; the rest persist.
struct RtSelectState {
    serial: atomic<u32>,
    // 1 = far band cut inside the near reach, 2 = no far band,
    // 4 / 8 = the near / far fit stopped at its scans' bound.
    flags: atomic<u32>,
    // Near reach and far budget term (f32 bits).
    limit: array<atomic<u32>, 2>,
    boundary: array<atomic<u32>, 2>,
    selected: array<atomic<u32>, 2>,
    // Drawn slots whose blades hold no key.
    drawn_other: array<atomic<u32>, 2>,
    free: array<atomic<u32>, 2>,
    claimed: array<atomic<u32>, 2>,
    // Live blades within the near band's largest reach.
    near_candidates: atomic<u32>,
    // Drawn slots kept for blades the selection leaves out.
    pinned: atomic<u32>,
    // Selected blades no slot was free for.
    dropped: atomic<u32>,
    // The limits the expansion draws with (f32 bits).
    drawn: array<atomic<u32>, 2>,
    drawn_valid: atomic<u32>,
    // The limits past which the next selection's scans count blades by the
    // chunk (f32 bits), 0 to count them all.
    bound: array<atomic<u32>, 2>,
}

struct RtSelectScratch {
    state: RtSelectState,
    // `RT_SELECT_BINS` bins each of the near keys', the far keys', the drawn
    // near slots' and the drawn far slots' histograms; the free slots, near
    // band first; and the owner tables, two entries per slot, near band
    // first.
    words: array<atomic<u32>>,
}

@group(0) @binding(17) var<uniform> rt_select: RtSelectParams;
// The purged patch indices, sorted; then `(patch index, end blade)` per
// scanned patch, sorted (`rt_covered`); then `(patch index, first blade, end
// blade, f32 bits of the least 3D distance between the eye and the patch's
// blades)` per scan chunk, the near band's first.
@group(0) @binding(18) var<storage, read> rt_scan: array<u32>;
@group(0) @binding(19) var<storage, read_write> rt_scratch: RtSelectScratch;
// `rt_params`, which `rt_ease_drawn` writes the drawn limits into.
@group(0) @binding(22) var<storage, read_write> rt_drawn_params: RtParams;

var<workgroup> rt_fit_sums: array<vec2<u32>, 256>;

fn rt_band(near: bool) -> u32 {
    return select(1u, 0u, near);
}

// The start of a histogram in `rt_scratch.words`.
fn rt_hist(near: bool, drawn: bool) -> u32 {
    return RT_SELECT_BINS * (rt_band(near) + select(0u, 2u, drawn));
}

fn rt_free_base(near: bool) -> u32 {
    return 4u * RT_SELECT_BINS + select(RT_NEAR_MAX_BLADES, 0u, near);
}

// The first of `count` entries `stride` words apart from `base` in `rt_scan`
// that is at least `key`.
fn rt_lower_bound(base: u32, stride: u32, count: u32, key: u32) -> u32 {
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (rt_scan[base + stride * mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// A blade as the selection ranks it: its least view-weighted distance and
// its eye distance.
struct RtCandidate {
    blade: RtBlade,
    least: f32,
    dist: f32,
}

fn rt_candidate(patch_idx: u32, b: u32) -> RtCandidate {
    var c: RtCandidate;
    c.blade = rt_blade(patch_idx, b);
    c.least = rt_least_rank_distance(c.blade.root);
    c.dist = distance(c.blade.root, rt_select.eye.xyz);
    return c;
}

// A lower bound on `rt_rank_distance` of `point` from any pose within the
// selection's margins: the eye up to the margin distance away, the view
// turned and its cone widened together by up to the margin angle. Widening
// the cone lowers a blade's weight no more than turning toward it by the
// same angle, so the bound turns the view by the whole margin.
fn rt_least_rank_distance(point: vec3<f32>) -> f32 {
    let dir = point - rt_select.eye.xyz;
    let r = length(dir);
    let margin = rt_select.margin.x;
    let nearest = max(r - margin, 0.0);
    let forward = rt_select.forward.xyz;
    if (dot(forward, forward) == 0.0 || nearest <= 0.0) {
        return nearest;
    }
    let c = clamp(dot(dir, forward) / r, -1.0, 1.0);
    // The direction to `point` turns toward the view by at most the margin
    // angle plus what the eye's travel turns it.
    let x = margin / r;
    let y = sqrt(1.0 - x * x);
    let sin_t = rt_select.margin.y * y + rt_select.margin.z * x;
    let cos_t = rt_select.margin.z * y - rt_select.margin.y * x;
    var nearest_c = 1.0;
    if (c < cos_t) {
        nearest_c = c * cos_t + sqrt(1.0 - c * c) * sin_t;
    }
    return nearest * rt_view_weight(rt_select.forward.w, nearest_c);
}

// The least budget term for which the far band's keep share at view-
// weighted distance `d`, past the near reach `end`, exceeds LOD rank
// `rank`; the unbounded term where none does.
fn rt_least_budget(d: f32, end: f32, rank: f32) -> f32 {
    let beyond = max(d - end, 0.0);
    if (beyond <= 0.0) {
        return 0.0;
    }
    let share = rank / rt_far_thinning(beyond, end);
    if (share >= 1.0) {
        return RT_UNBOUNDED_BUDGET;
    }
    // -ln(1 - share), by its series where the subtraction loses digits.
    var neg_log = -log(1.0 - share);
    if (share < 1e-3) {
        neg_log = share * (1.0 + share * (0.5 + share / 3.0));
    }
    return beyond * beyond * neg_log;
}

// Band `near`'s key of `c` past the near reach `end`, or `RT_KEY_NONE`
// where the band cannot hold it: the near band's is the least view-weighted
// distance, the far band's the least budget term.
fn rt_key(c: RtCandidate, near: bool, end: f32) -> f32 {
    if (!rt_live(c.blade)) {
        return RT_KEY_NONE;
    }
    if (near) {
        return select(RT_KEY_NONE, c.least, c.least <= rt_select.eye.w);
    }
    if ((rt_select.counts.w & 1u) == 0u || c.dist > MEADOW_SHADOW_MAX_DIST + rt_select.margin.x) {
        return RT_KEY_NONE;
    }
    let key = rt_least_budget(c.least, end, c.blade.rank);
    return select(RT_KEY_NONE, key, key < RT_UNBOUNDED_BUDGET);
}

// The lower edge of bin `k` of band `near`'s histogram. The near band's
// bins split its largest reach evenly. The far band's bin 0 holds the keys
// of 0, bin 1 the rest below the edge of bin 2, and from there the edges
// rise geometrically to the unbounded budget term.
fn rt_edge(k: u32, near: bool) -> f32 {
    if (near) {
        return rt_select.eye.w * f32(k) / f32(RT_SELECT_BINS);
    }
    if (k < 2u) {
        return 0.0;
    }
    let span = log2(RT_UNBOUNDED_BUDGET) - RT_LOG_BUDGET_MIN;
    return exp2(RT_LOG_BUDGET_MIN + f32(k - 1u) * span / f32(RT_SELECT_BINS - 2u));
}

// The bin `k` of band `near` with `rt_edge(k) <= key < rt_edge(k + 1)`, by
// the same arithmetic as the edges.
fn rt_bin(key: f32, near: bool) -> u32 {
    var x: f32;
    var lowest = 0u;
    if (near) {
        x = key / rt_select.eye.w * f32(RT_SELECT_BINS);
    } else {
        if (key <= 0.0) {
            return 0u;
        }
        let span = log2(RT_UNBOUNDED_BUDGET) - RT_LOG_BUDGET_MIN;
        x = 1.0 + floor((log2(key) - RT_LOG_BUDGET_MIN) * f32(RT_SELECT_BINS - 2u) / span);
        lowest = 1u;
    }
    var k = u32(clamp(x, f32(lowest), f32(RT_SELECT_BINS - 1u)));
    if (k > lowest && key < rt_edge(k, near)) {
        k -= 1u;
    } else if (k + 1u < RT_SELECT_BINS && key >= rt_edge(k + 1u, near)) {
        k += 1u;
    }
    return k;
}

// The fitted near reach, where the far band's keys start.
fn rt_near_end() -> f32 {
    return bitcast<f32>(atomicLoad(&rt_scratch.state.limit[0]));
}

// Whether band `near` selects `c`, a blade the scan lists: its key lies
// below the fitted boundary bin.
fn rt_selected(c: RtCandidate, near: bool) -> bool {
    let key = rt_key(c, near, rt_near_end());
    return key >= 0.0 && rt_bin(key, near) < atomicLoad(&rt_scratch.state.boundary[rt_band(near)]);
}

// Whether the scan lists blade `b` of patch `patch_idx` for the near band
// (`near`) or the far band. Only a listed blade holds a key in the fit, so
// a slot holding any other blade counts as a drawn slot without a key, or
// leaves its slot.
fn rt_covered(patch_idx: u32, b: u32, near: bool) -> bool {
    let base = rt_select.counts.z;
    let i = rt_lower_bound(base, 2u, rt_select.counts.y, patch_idx);
    if (i >= rt_select.counts.y || rt_scan[base + 2u * i] != patch_idx) {
        return false;
    }
    let end = rt_scan[base + 2u * i + 1u];
    if (near) {
        return (end & RT_COVERAGE_NEAR) != 0u;
    }
    return b < (end & ~RT_COVERAGE_NEAR);
}

// Where `owner`'s probe sequence through its band's owner table starts.
struct RtProbe {
    base: u32,
    mask: u32,
    at: u32,
}

fn rt_probe(near: bool, owner: vec2<u32>) -> RtProbe {
    var probe: RtProbe;
    probe.base = 4u * RT_SELECT_BINS + RT_SLOTS + select(2u * RT_NEAR_MAX_BLADES, 0u, near);
    probe.mask = 2u * select(RT_FAR_MAX_BLADES, RT_NEAR_MAX_BLADES, near) - 1u;
    probe.at = hash_u32((owner.x * 0x9E3779B9u) ^ hash_u32(owner.y)) & probe.mask;
    return probe;
}

// Enter `slot`, which `owner` holds, in its band's owner table.
fn rt_table_insert(near: bool, owner: vec2<u32>, slot: u32) {
    var probe = rt_probe(near, owner);
    loop {
        let r = atomicCompareExchangeWeak(&rt_scratch.words[probe.base + probe.at], 0u, slot + 1u);
        if (r.exchanged) {
            return;
        }
        // A weak exchange may fail spuriously on an empty entry.
        if (r.old_value != 0u) {
            probe.at = (probe.at + 1u) & probe.mask;
        }
    }
}

// Remove `slot`, which `owner` holds, from its band's owner table. Entries
// are matched by slot, so the pass never reads an owner another invocation
// vacates.
fn rt_table_remove(near: bool, owner: vec2<u32>, slot: u32) {
    var probe = rt_probe(near, owner);
    var entry = atomicLoad(&rt_scratch.words[probe.base + probe.at]);
    while (entry != 0u && entry != slot + 1u) {
        probe.at = (probe.at + 1u) & probe.mask;
        entry = atomicLoad(&rt_scratch.words[probe.base + probe.at]);
    }
    if (entry != 0u) {
        atomicStore(&rt_scratch.words[probe.base + probe.at], RT_TABLE_TOMBSTONE);
    }
}

// Whether `owner` holds a slot of its band. Every entry left after
// `rt_select_evict` is a slot that keeps its owner.
fn rt_table_holds(near: bool, owner: vec2<u32>) -> bool {
    var probe = rt_probe(near, owner);
    var entry = atomicLoad(&rt_scratch.words[probe.base + probe.at]);
    while (entry != 0u) {
        let s = rt_slots[entry - 1u];
        if (entry != RT_TABLE_TOMBSTONE && s.patch_id == owner.x && s.blade == owner.y) {
            return true;
        }
        probe.at = (probe.at + 1u) & probe.mask;
        entry = atomicLoad(&rt_scratch.words[probe.base + probe.at]);
    }
    return false;
}

fn rt_free_slot(near: bool, slot: u32) {
    let i = atomicAdd(&rt_scratch.state.free[rt_band(near)], 1u);
    atomicStore(&rt_scratch.words[rt_free_base(near) + i], slot);
}

// A vacated slot, whose geometry the expansion clears.
fn rt_vacated() -> RtSlot {
    return RtSlot(0u, 0u, RT_SLOT_FRESH, 0u);
}

// Give `owner` a free slot of its band.
fn rt_place(near: bool, owner: vec2<u32>) {
    let band = rt_band(near);
    let i = atomicAdd(&rt_scratch.state.claimed[band], 1u);
    if (i >= atomicLoad(&rt_scratch.state.free[band])) {
        atomicAdd(&rt_scratch.state.dropped, 1u);
        return;
    }
    let slot = atomicLoad(&rt_scratch.words[rt_free_base(near) + i]);
    rt_slots[slot] = RtSlot(owner.x, owner.y, RT_SLOT_FRESH, 0u);
}

// The scan chunk of workgroup `wg`, dispatched in rows of `RT_SCAN_ROW`.
fn rt_chunk_index(wg: vec3<u32>) -> u32 {
    return wg.x + wg.y * RT_SCAN_ROW;
}

// Scan chunk `chunk` (`rt_scan`): (patch index, first blade, end blade,
// least eye distance less the margin).
struct RtChunk {
    patch_index: u32,
    first: u32,
    end: u32,
    nearest: f32,
}

fn rt_scan_chunk(chunk: u32) -> RtChunk {
    let at = rt_select.counts.z + 2u * rt_select.counts.y + 4u * chunk;
    let nearest = max(bitcast<f32>(rt_scan[at + 3u]) - rt_select.margin.x, 0.0);
    return RtChunk(rt_scan[at], rt_scan[at + 1u], rt_scan[at + 2u], nearest);
}

// The end of the prefix of `chunk`'s patch blades the far band can keep
// under budget term `budget` past the near reach `end`: keep shares fall
// with distance, view weights are at least 1, and blades rank by index, so
// the patch's nearest point bounds the share.
fn rt_far_prefix(chunk: RtChunk, end: f32, budget: f32) -> u32 {
    let keep = rt_far_keep_share(chunk.nearest, end, budget);
    let count = u32(patches[chunk.patch_index].blade_count_noise_canopy_flags.x);
    return clamp(meadow_gate_count(keep, count) + 1u, chunk.first, chunk.end);
}

// Vacate the slots of the purged patches, or every slot.
@compute @workgroup_size(256)
fn rt_select_purge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= RT_SLOTS || rt_slots[slot].patch_id == 0u) {
        return;
    }
    let patch_idx = rt_slots[slot].patch_id - 1u;
    let purged = rt_select.counts.z;
    let i = rt_lower_bound(0u, 1u, purged, patch_idx);
    if ((rt_select.counts.w & 2u) != 0u || (i < purged && rt_scan[i] == patch_idx)) {
        rt_slots[slot] = rt_vacated();
    }
}

// Histogram the key of drawn slot `slot` of band `near`.
fn rt_count_drawn(slot: u32, near: bool) {
    let s = rt_slots[slot];
    let key = rt_key(rt_candidate(s.patch_id - 1u, s.blade), near, rt_near_end());
    if (key >= 0.0 && rt_covered(s.patch_id - 1u, s.blade, near)) {
        atomicAdd(&rt_scratch.words[rt_hist(near, true) + rt_bin(key, near)], 1u);
    } else {
        atomicAdd(&rt_scratch.state.drawn_other[rt_band(near)], 1u);
    }
}

// Enter every owned slot in its band's table, and histogram the keys of
// the near band's drawn slots.
@compute @workgroup_size(256)
fn rt_select_index(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= RT_SLOTS) {
        return;
    }
    let s = rt_slots[slot];
    if (s.patch_id == 0u) {
        return;
    }
    let near = slot < RT_NEAR_MAX_BLADES;
    rt_table_insert(near, vec2<u32>(s.patch_id, s.blade), slot);
    if (near && (s.flags & RT_SLOT_DRAWN) != 0u) {
        rt_count_drawn(slot, true);
    }
}

// Histogram the far keys of the far band's drawn slots.
@compute @workgroup_size(256)
fn rt_select_drawn_far(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = RT_NEAR_MAX_BLADES + gid.x;
    if (gid.x < RT_FAR_MAX_BLADES && rt_slots[slot].patch_id != 0u && (rt_slots[slot].flags & RT_SLOT_DRAWN) != 0u) {
        rt_count_drawn(slot, false);
    }
}

// The bound band `near`'s scans count blades one by one up to, 0 for none.
fn rt_scan_bound(near: bool) -> f32 {
    return bitcast<f32>(atomicLoad(&rt_scratch.state.bound[rt_band(near)]));
}

// Histogram band `near`'s keys over a scan chunk. Past the band's bound the
// chunk's blades count, keyed or not, at the bound's bin, where the fit
// stops: the near band's in patches wholly past the bound, the far band's
// past the prefix the bound keeps.
fn rt_count(wg: vec3<u32>, lid: u32, near: bool) {
    let index = rt_chunk_index(wg);
    if (index >= select(rt_select.run.x, rt_select.counts.x, near)) {
        return;
    }
    let chunk = rt_scan_chunk(index);
    let end = rt_near_end();
    let bound = rt_scan_bound(near);
    var last = chunk.end;
    if (bound > 0.0) {
        if (!near) {
            last = rt_far_prefix(chunk, end, bound);
        } else if (chunk.nearest > bound) {
            last = chunk.first;
        }
        if (lid == 0u && last < chunk.end) {
            atomicAdd(&rt_scratch.words[rt_hist(near, false) + rt_bin(bound, near)], chunk.end - last);
        }
    }
    for (var b = chunk.first + lid; b < last; b += 256u) {
        let key = rt_key(rt_candidate(chunk.patch_index, b), near, end);
        if (key >= 0.0) {
            atomicAdd(&rt_scratch.words[rt_hist(near, false) + rt_bin(key, near)], 1u);
        }
    }
}

@compute @workgroup_size(256)
fn rt_select_count_near(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    rt_count(wg, lid, true);
}

@compute @workgroup_size(256)
fn rt_select_count_far(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    rt_count(wg, lid, false);
}

// A band's fit: its boundary bin, the blades below it, its candidates and
// whether it stopped at the scans' bound.
struct RtFit {
    boundary: u32,
    selected: u32,
    candidates: u32,
    bounded: bool,
}

// Fit band `near`'s boundary bin to `capacity` from its keys' histogram and
// its drawn slots' (more drawn slots hold blades without a key): the
// largest bin such that the blades below the next one and the drawn slots
// past it fit, or `RT_SELECT_BINS` when every candidate fits. The bin of
// slack holds blades whose key another pass rounds across the boundary's
// edge. A boundary past the bin of the scans' bound stops there, where the
// histogram stops counting exactly. Valid in the first invocation.
fn rt_fit(lid: u32, near: bool, capacity: u32) -> RtFit {
    let per = RT_SELECT_BINS / 256u;
    let hist = rt_hist(near, false);
    let drawn_hist = rt_hist(near, true);
    var sums = vec2<u32>(0u);
    for (var i = 0u; i < per; i += 1u) {
        let k = lid * per + i;
        sums += vec2<u32>(
            atomicLoad(&rt_scratch.words[hist + k]),
            atomicLoad(&rt_scratch.words[drawn_hist + k]),
        );
    }
    rt_fit_sums[lid] = sums;
    workgroupBarrier();
    var fit: RtFit;
    if (lid != 0u) {
        return fit;
    }
    var total = vec2<u32>(0u);
    for (var t = 0u; t < 256u; t += 1u) {
        total += rt_fit_sums[t];
    }
    let other = atomicLoad(&rt_scratch.state.drawn_other[rt_band(near)]);
    fit.candidates = total.x;
    fit.boundary = RT_SELECT_BINS;
    fit.selected = total.x;
    if (total.x + other > capacity) {
        let drawn = total.y + other;
        // The blades below a boundary plus the drawn slots past it never
        // fall as the boundary rises.
        var below = vec2<u32>(0u);
        var t = 0u;
        for (; t < 256u; t += 1u) {
            let next = below + rt_fit_sums[t];
            if (next.x + drawn - next.y > capacity) {
                break;
            }
            below = next;
        }
        var k = t * per;
        for (; k < min(t + 1u, 256u) * per; k += 1u) {
            let next = below + vec2<u32>(
                atomicLoad(&rt_scratch.words[hist + k]),
                atomicLoad(&rt_scratch.words[drawn_hist + k]),
            );
            if (next.x + drawn - next.y > capacity) {
                break;
            }
            below = next;
        }
        // `k` is the first bin that does not fit whole; the one before it is
        // the slack.
        fit.boundary = max(k, 1u) - 1u;
        fit.selected = select(0u, below.x - atomicLoad(&rt_scratch.words[hist + fit.boundary]), k > 0u);
    }
    let bound = rt_scan_bound(near);
    if (bound > 0.0 && fit.boundary < RT_SELECT_BINS) {
        let stop = rt_bin(bound, near);
        if (fit.boundary >= stop) {
            fit.bounded = true;
            fit.boundary = stop;
            fit.selected = 0u;
            for (var k = 0u; k < stop; k += 1u) {
                fit.selected += atomicLoad(&rt_scratch.words[hist + k]);
            }
        }
    }
    return fit;
}

// Store band `near`'s fit and the bound its next scans count blades one by
// one up to: `RT_BOUND_SLACK` past its limit, or none after a fit that
// fitted everything or stopped at its bound.
fn rt_store_fit(near: bool, fit: RtFit, limit: f32) {
    let band = rt_band(near);
    atomicStore(&rt_scratch.state.limit[band], bitcast<u32>(limit));
    atomicStore(&rt_scratch.state.boundary[band], fit.boundary);
    atomicStore(&rt_scratch.state.selected[band], fit.selected);
    var bound = 0.0;
    if (!fit.bounded && fit.boundary < RT_SELECT_BINS && limit > 0.0) {
        bound = min(limit * RT_BOUND_SLACK[band], select(RT_UNBOUNDED_BUDGET, rt_select.eye.w, near));
    }
    atomicStore(&rt_scratch.state.bound[band], bitcast<u32>(bound));
    if (fit.bounded) {
        atomicOr(&rt_scratch.state.flags, select(8u, 4u, near));
    }
}

@compute @workgroup_size(256)
fn rt_select_fit_near(@builtin(local_invocation_index) lid: u32) {
    let fit = rt_fit(lid, true, RT_NEAR_MAX_BLADES);
    if (lid != 0u) {
        return;
    }
    atomicStore(&rt_scratch.state.serial, rt_select.run.y);
    atomicStore(&rt_scratch.state.near_candidates, fit.candidates);
    rt_store_fit(true, fit, rt_edge(fit.boundary, true));
}

@compute @workgroup_size(256)
fn rt_select_fit_far(@builtin(local_invocation_index) lid: u32) {
    if ((rt_select.counts.w & 1u) == 0u) {
        if (lid == 0u) {
            rt_store_fit(false, RtFit(0u, 0u, 0u, false), 0.0);
            atomicOr(&rt_scratch.state.flags, 2u);
        }
        return;
    }
    let fit = rt_fit(lid, false, RT_FAR_MAX_BLADES);
    if (lid != 0u) {
        return;
    }
    var budget = RT_UNBOUNDED_BUDGET;
    if (fit.boundary < RT_SELECT_BINS) {
        budget = rt_edge(fit.boundary, false);
    }
    if (fit.boundary == 0u) {
        atomicOr(&rt_scratch.state.flags, 1u);
    }
    rt_store_fit(false, fit, budget);
}

// Keep each slot whose blade is selected or was drawn last frame; vacate
// and free the rest.
@compute @workgroup_size(256)
fn rt_select_evict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= RT_SLOTS) {
        return;
    }
    let near = slot < RT_NEAR_MAX_BLADES;
    let s = rt_slots[slot];
    if (s.patch_id != 0u) {
        if (rt_covered(s.patch_id - 1u, s.blade, near) && rt_selected(rt_candidate(s.patch_id - 1u, s.blade), near)) {
            return;
        }
        if ((s.flags & RT_SLOT_DRAWN) != 0u) {
            atomicAdd(&rt_scratch.state.pinned, 1u);
            return;
        }
        rt_table_remove(near, vec2<u32>(s.patch_id, s.blade), slot);
        rt_slots[slot] = rt_vacated();
    }
    rt_free_slot(near, slot);
}

// Give every selected blade without a slot a free one. A chunk past the
// near reach holds no near-band blade, and only the prefix the fitted far
// budget term keeps.
@compute @workgroup_size(256)
fn rt_select_place(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let index = rt_chunk_index(wg);
    if (index >= rt_select.run.x) {
        return;
    }
    let chunk = rt_scan_chunk(index);
    let end = rt_near_end();
    let near_band = index < rt_select.counts.x && chunk.nearest < end;
    var last = chunk.end;
    if (!near_band) {
        last = rt_far_prefix(chunk, end, bitcast<f32>(atomicLoad(&rt_scratch.state.limit[1])));
    }
    for (var b = chunk.first + lid; b < last; b += 256u) {
        let c = rt_candidate(chunk.patch_index, b);
        let owner = vec2<u32>(chunk.patch_index + 1u, b);
        if (near_band && rt_selected(c, true) && !rt_table_holds(true, owner)) {
            rt_place(true, owner);
        }
        if (rt_selected(c, false) && !rt_table_holds(false, owner)) {
            rt_place(false, owner);
        }
    }
}

// Ease the drawn limits toward the selection's, both ways, and write them
// into `rt_params`; they start at the first selection's. The expansion
// draws only resident blades, and a blade it drew keeps its slot, so a
// limit easing down past the selection's stays drawable.
@compute @workgroup_size(1)
fn rt_ease_drawn() {
    if (atomicLoad(&rt_scratch.state.serial) == 0u) {
        rt_drawn_params.drawn = vec4<f32>(0.0);
        return;
    }
    let reach = bitcast<f32>(atomicLoad(&rt_scratch.state.limit[0]));
    let budget = bitcast<f32>(atomicLoad(&rt_scratch.state.limit[1]));
    var near = reach;
    var far = 0.0;
    var start = reach;
    if ((atomicLoad(&rt_scratch.state.flags) & 2u) == 0u) {
        far = budget;
        if (atomicLoad(&rt_scratch.state.drawn_valid) != 0u) {
            let ease = rt_drawn_params.ease.x;
            let last_near = bitcast<f32>(atomicLoad(&rt_scratch.state.drawn[0]));
            let last_far = bitcast<f32>(atomicLoad(&rt_scratch.state.drawn[1]));
            near = mix(last_near, reach, ease);
            // Budget terms span decades: ease their logarithm.
            far = exp2(mix(log2(max(last_far, 1e-6)), log2(max(budget, 1e-6)), ease));
            if (abs(near - reach) <= 1e-3 * reach) {
                near = reach;
            }
            if (abs(far - budget) <= 1e-3 * max(budget, 1e-6)) {
                far = budget;
            }
        }
        start = near * RT_NEAR_HANDOVER;
    }
    atomicStore(&rt_scratch.state.drawn[0], bitcast<u32>(near));
    atomicStore(&rt_scratch.state.drawn[1], bitcast<u32>(far));
    atomicStore(&rt_scratch.state.drawn_valid, 1u);
    rt_drawn_params.drawn = vec4<f32>(start, near, far, 0.0);
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
