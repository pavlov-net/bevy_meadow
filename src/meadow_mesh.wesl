// Meadow mesh-shader render path — task + mesh entry points.
//
// Gated on the `mesh-shaders` cargo feature + runtime
// `EXPERIMENTAL_MESH_SHADER`. The task/mesh pipeline replaces the compute
// cull/compact pass + intermediate `out_blades` buffer + indirect draw
// for the views it serves, and is faster where supported. The compute
// path is the fallback on GPUs without mesh shaders (see `mesh_path.rs`).
//
// Stage split (shaped by Nsight: SM warp starvation, not ALU, dominates
// mesh pipelines with big per-workgroup outputs):
//
// - TASK stage, 128 threads per workgroup — one workgroup per (active
//   patch, 128-blade slice). Does everything the compute kernel's
//   `cull_and_compact` does at full warp width: patch frustum/LOD cull,
//   per-blade survival gates, `derive_blade`, trunk-disc collapse,
//   heightfield sample — then compacts survivors into the task PAYLOAD
//   and spawns just enough mesh workgroups to expand them.
// - MESH stage, 32 threads per workgroup — a pure expander. Each
//   workgroup owns ≤5 blades (55 verts / 45 tris) or ≤3 tufts (63 / 21),
//   keeping the per-workgroup output allocation small (~3 KB) so many
//   workgroups fit in flight — large outputs (the previous 253-vert
//   budget) throttle workgroup launch and starve the SMs. All 32 lanes
//   expand vertices cooperatively (lane ↔ vertex, not lane ↔ blade).
//
// Imports `BladeAttrs`, the blade geometry and the per-blade palette
// variance from the shared library, and the group-3 bindings, the mesh →
// fragment varyings, and the motion-vector helper shared with the PBR
// fragments from `meadow_mesh_bindings.wesl`, so none of them can drift
// between stages. The task stage resolves each blade's colour once and
// the mesh stage hands it to the fragment as a varying. Compiled by
// bevy's pipeline cache like every other WESL shader (`mesh_path.rs`
// queues the pipelines).
//
// Dispatch: one task workgroup per CPU-built `(patch_idx, slice_base)`
// pair covering exactly the active patches' real blade counts
// (`task_slices`; sizing a grid for the 65536-blade maximum instead
// launches ~mostly-empty task workgroups, millions per frame across the
// views). The flat work list can exceed wgpu's 65535 per-dimension
// dispatch cap, so it folds over a fixed-stride 2D grid — see
// `mesh.rs::MESH_TASK_DISPATCH_STRIDE`. The view is selected by a
// per-view-slot bind group (`mesh_view`), and the per-view cull data is
// the SAME `view_cull` storage buffer the compute path uploads (frusta
// + flags + lod_max + shadow_near; the mesh path ignores the base/cap
// region fields).
//
// Bindings live at @group(3) in EVERY pipeline built from this module:
// the main pipeline has bevy_pbr's view + binding-array groups at 0/1
// (the PBR fragment imports them at those slots) and an empty group 2; the
// depth-only shadow pipeline fills slots 0-2 with empty bind groups so
// the one module serves both layouts.

import bevy_meadow::meadow_shared::{
    BladeAttrs, MEADOW_CHORD_WIDTH_SCALE, MEADOW_MAX_BLADES_PER_PATCH, wind_sway, hash_u32,
    hash01_u32, meadow_template_vertex, meadow_place_vertex, meadow_blade_normal_varying,
    meadow_gate_count, meadow_band_threshold, meadow_shadow_ramp, meadow_patch_palette,
    meadow_max_height_scale, meadow_patch_noise, meadow_blade_height_scale, meadow_blade_albedo,
    meadow_pack_albedo, meadow_unpack_albedo, meadow_patch_is_tile, meadow_patch_outer_radius,
    meadow_tile_root, meadow_cover_survives, meadow_sample_cover, meadow_blade_palette,
    meadow_grid_bound, meadow_grid_footprint,
    meadow_blade_rank, MEADOW_MAX_HEIGHT_VARIANCE, MEADOW_MAX_DRYNESS_HEIGHT, MEADOW_NEUTRAL_DRYNESS,
    GatedBlade, PaletteParams, meadow_disc_effect, MEADOW_BLADE_GROUND_LIFT,
};
import bevy_meadow::meadow_mesh_bindings::{
    variant_params, patches, trunk_slots, heightfield, view_cull, task_slices, mesh_view, palettes,
    cover_field, clearing_profiles, PatchData, MeadowViewCull, MeadowVertexOut,
};

enable wgpu_mesh_shader;

// ---------- budget constants ----------
// MUST mirror `mesh.rs` (`MESH_TASK_BLADES` / `MESH_WG_BLADES` /
// `MESH_WG_TUFTS`).

// Blades scanned/derived per task workgroup (= its workgroup size, and
// the payload survivor capacity).
const MESH_TASK_BLADES: u32 = 128u;
// Blades expanded per mesh workgroup (band 0): 5 × 11 = 55 verts,
// 5 × 9 = 45 tris — inside the ≤64-vert small-output sweet spot.
const MESH_WG_BLADES: u32 = 5u;
// Tufts expanded per mesh workgroup (band 1): 3 × 21 = 63 verts, 21 tris.
const MESH_WG_TUFTS: u32 = 3u;
// Shadow-cascade blades per mesh workgroup: a shadow blade is a single
// proxy triangle (base edge + leaning, wind-swept tip = template verts
// 0/1/10), so 21 × 3 = 63 verts / 21 tris — 9× less rasterization than
// the full ribbon, with a silhouette that only loses the mid-blade
// bow (sub-texel at shadow-map density).
const MESH_WG_SHADOW_BLADES: u32 = 21u;
// Task-grid X stride (MUST mirror `mesh.rs::MESH_TASK_DISPATCH_STRIDE`):
// the flat work list folds over a 2D dispatch because it can exceed
// wgpu's 65535 per-dimension cap.
const MESH_TASK_DISPATCH_STRIDE: u32 = 32768u;
// Full mesh output budget: max(5 × 11, 3 × 21) verts, max(5 × 9, 3 × 7)
// tris across the two bands. The shadow output is sized separately —
// one proxy triangle per blade.
const MESH_OUT_VERTS: u32 = 63u;
const MESH_OUT_PRIMS: u32 = 45u;
const MESH_SHADOW_OUT_VERTS: u32 = 63u; // MESH_WG_SHADOW_BLADES × 3
const MESH_SHADOW_OUT_PRIMS: u32 = 21u; // MESH_WG_SHADOW_BLADES × 1

const MEADOW_VIEW_FLAG_SHADOW: u32 = 1u;

// ---------- derivation helpers (MIRROR: meadow_compute.wesl) ----------

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

// ---------- wind (MIRROR: meadow.wesl) ----------

fn wind_displacement(world_xz: vec2<f32>, blade_y_norm: f32, t: f32, clump: f32) -> vec2<f32> {
    return wind_sway(
        world_xz,
        blade_y_norm,
        t,
        clump,
        variant_params.wind,
        variant_params.wind_direction,
        variant_params.wind_state,
    );
}

// ---------- task stage ----------

// One fully-derived surviving blade, handed task → mesh in the payload.
// The mesh stage only expands geometry from these — no storage reads, no
// re-derivation. Tuft height fade is already applied to `height`.
struct SurvivorBlade {
    world_xz: vec2<f32>,
    yaw: f32,
    height: f32,
    width: f32,
    // `meadow_pack_albedo` of the blade colour; 0 for shadow views, which
    // draw depth only.
    albedo: u32,
    ground_y: f32,
    collapse_t: f32,
    // Full-amplitude (y_norm = 1) wind displacement at the current /
    // previous wind time. `wind_displacement`'s phase + gust depend only
    // on the blade ROOT, so sway is exactly linear in y_norm — hoisting
    // the sin/normalize-heavy evaluation here turns the per-vertex wind
    // into a single multiply-add (it used to run per vertex × 2 times).
    // For shadow casters this is pre-attenuated with camera distance
    // (sub-texel far blade shadows swaying read as temporal noise).
    wind_now: vec2<f32>,
    wind_prev: vec2<f32>,
}

struct MeadowTaskPayload {
    band: u32,
    count: u32,
    pad0: u32,
    pad1: u32,
    survivors: array<SurvivorBlade, MESH_TASK_BLADES>,
}

var<task_payload> payload: MeadowTaskPayload;

var<workgroup> task_emit: atomic<u32>;

// One task workgroup per CPU-built (patch, 128-blade slice) pair. The
// patch-uniform work (cull, band, thresholds) is computed redundantly
// per lane — same trade the compute kernel makes; it's a handful of ALU
// + 4 cached texture loads. Per-lane: gate + derive + compact into the
// payload at full 128-wide utilization.
@task
@payload(payload)
@workgroup_size(128)
fn meadow_task(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) -> @builtin(mesh_task_size) vec3<u32> {
    let view_slot = mesh_view.params.x;
    if (view_slot >= view_cull.count) {
        return vec3<u32>(0u, 0u, 0u);
    }
    let flat = wg.y * MESH_TASK_DISPATCH_STRIDE + wg.x;
    if (flat >= task_slices.count) {
        return vec3<u32>(0u, 0u, 0u);
    }
    let slice = task_slices.entries[flat];
    let patch_idx = slice.x;
    let slice_base = slice.y;
    let p = patches[patch_idx];
    let blade_count = min(u32(p.blade_count_noise_canopy_flags.x), MEADOW_MAX_BLADES_PER_PATCH);
    if (blade_count == 0u || slice_base >= blade_count) {
        return vec3<u32>(0u, 0u, 0u);
    }

    let vc = view_cull.views[view_slot];
    let is_shadow = (u32(vc.params.x) & MEADOW_VIEW_FLAG_SHADOW) != 0u;

    // Patch-level frustum reject — same math as the compute kernel.
    let centre_xz = p.centre_xz_radius_seed.xy;
    let centre_y = sample_heightfield(centre_xz);
    let flags = p.blade_count_noise_canopy_flags.w;
    let tile = meadow_patch_is_tile(flags);
    let outer = patch_outer_radius(p);
    let cull_radius = outer + patch_max_height(p) * 2.0;
    if (patch_sphere_culled(vec3<f32>(centre_xz.x, centre_y, centre_xz.y), cull_radius, vc, is_shadow)) {
        return vec3<u32>(0u, 0u, 0u);
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

    // Camera view-depth of the patch centre (see meadow_compute.wesl for
    // why cascade assignment uses this, not planar viewer distance).
    let vf = variant_params.viewer_forward;
    let view_depth =
        dot(vf.xyz, vec3<f32>(centre_xz.x, centre_y, centre_xz.y)) + vf.w;

    var band = 0u;
    if (dist >= tuft_start) {
        band = 1u;
    }

    if (is_shadow) {
        // Tufts never cast.
        if (band == 1u) {
            return vec3<u32>(0u, 0u, 0u);
        }
        let shadow_near = vc.params2.z;
        if (view_depth + cull_radius < shadow_near || view_depth - cull_radius > lod_max) {
            return vec3<u32>(0u, 0u, 0u);
        }
    }

    // The gate keeps a prefix of the blade indices; a slice past it has no
    // survivors. The CPU work list already stops near the prefix
    // (`compute.rs::task_slice_bound`).
    let threshold = meadow_band_threshold(variant_params, band, is_shadow, gate_dist, lod_max);
    let survivors = meadow_gate_count(threshold, blade_count);
    if (slice_base >= survivors) {
        return vec3<u32>(0u, 0u, 0u);
    }

    // Per-lane: derive lane `lid`'s blade slot. MIRROR of
    // `cull_and_compact`'s per-blade body (band routing, cascade clip,
    // per-blade gates).
    let b = slice_base + lid;
    var survived = false;
    var out: SurvivorBlade;
    if (b < survivors) {
        let blade = derive_blade(b, p);
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
            var wind_now = wind_displacement(
                blade.world_xz, 1.0, variant_params.wind.z, blade.clump_seed);
            var wind_prev = wind_displacement(
                blade.world_xz, 1.0, variant_params.wind.w, blade.clump_seed);
            if (is_shadow) {
                // Fade shadow-caster sway with player distance: a far
                // blade's shadow is sub-texel in the cascade, and
                // animating it reads as shimmering noise. Same ramp
                // the shadow density uses, so sway settles exactly
                // where the casters thin out.
                let sway_atten = 1.0 - meadow_shadow_ramp(dist, variant_params.density.z);
                wind_now = wind_now * sway_atten;
                wind_prev = wind_prev * sway_atten;
            }
            out.world_xz = blade.world_xz;
            out.yaw = blade.yaw;
            out.height = gated.height;
            out.width = blade.width;
            out.albedo = 0u;
            if (!is_shadow) {
                out.albedo = meadow_pack_albedo(
                    meadow_blade_albedo(palettes[gated.palette], gated.noise, gated.dryness, blade.seed),
                );
            }
            out.ground_y = gated.ground_y;
            out.collapse_t = gated.collapse_t;
            out.wind_now = wind_now;
            out.wind_prev = wind_prev;
            survived = true;
        }
    }

    // Compact survivors into the payload (workgroup-shared atomic, same
    // pattern as the compute kernel's aggregation).
    if (lid == 0u) {
        atomicStore(&task_emit, 0u);
    }
    workgroupBarrier();
    if (survived) {
        let slot = atomicAdd(&task_emit, 1u);
        payload.survivors[slot] = out;
    }
    workgroupBarrier();
    let n = atomicLoad(&task_emit);
    if (lid == 0u) {
        payload.band = band;
        payload.count = n;
    }

    var per_wg = MESH_WG_BLADES;
    if (is_shadow) {
        per_wg = MESH_WG_SHADOW_BLADES;
    } else if (band == 1u) {
        per_wg = MESH_WG_TUFTS;
    }
    let n_wgs = (n + per_wg - 1u) / per_wg;
    if (lid == 0u) {
        return vec3<u32>(n_wgs, 1u, 1u);
    }
    return vec3<u32>(0u, 0u, 0u);
}

// ---------- shared vertex expansion ----------

// Blade triangle table — MIRROR of `mesh.rs::BLADE_INDICES` (27 indices
// grouped as 9 tris).
const BLADE_TRIS = array<vec3<u32>, 9>(
    vec3<u32>(0u, 1u, 3u), vec3<u32>(0u, 3u, 2u),
    vec3<u32>(2u, 3u, 5u), vec3<u32>(2u, 5u, 4u),
    vec3<u32>(4u, 5u, 7u), vec3<u32>(4u, 7u, 6u),
    vec3<u32>(6u, 7u, 9u), vec3<u32>(6u, 9u, 8u),
    vec3<u32>(8u, 9u, 10u),
);

// ---------- mesh stage: full (main view) ----------

struct MeadowPrimOut {
    @builtin(triangle_indices) indices: vec3<u32>,
}

struct MeadowMeshOut {
    @builtin(vertices) vertices: array<MeadowVertexOut, MESH_OUT_VERTS>,
    @builtin(primitives) primitives: array<MeadowPrimOut, MESH_OUT_PRIMS>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> mesh_out: MeadowMeshOut;

// Pure expansion: workgroup `wg.x` owns survivors
// `[wg.x * per_wg, ...+count)` from the payload; the 32 lanes expand
// vertices and primitives cooperatively (lane ↔ vertex).
@mesh(mesh_out)
@payload(payload)
@workgroup_size(32)
fn meadow_mesh(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    var per_wg = MESH_WG_BLADES;
    var n_verts = 11u;
    var n_prims = 9u;
    if (payload.band == 1u) {
        per_wg = MESH_WG_TUFTS;
        n_verts = 21u;
        n_prims = 7u;
    }
    let base = wg.x * per_wg;
    let count = min(per_wg, payload.count - base);

    if (lid == 0u) {
        mesh_out.vertex_count = count * n_verts;
        mesh_out.primitive_count = count * n_prims;
    }

    let total_verts = count * n_verts;
    for (var t = lid; t < total_verts; t = t + 32u) {
        let s = payload.survivors[base + t / n_verts];
        let v = t % n_verts;
        let blade = BladeAttrs(s.world_xz, s.yaw, s.height, s.width, 0.0, 1.0, 0u);
        let tuft = payload.band == 1u;
        let cy = cos(s.yaw);
        let sy = sin(s.yaw);
        // The task stage computes the tip wind (see `SurvivorBlade`), so
        // the previous-frame position is the same template vertex placed
        // with the previous wind.
        let local = meadow_template_vertex(v, tuft, blade);
        let world = vec4<f32>(
            meadow_place_vertex(local, s.world_xz, s.ground_y, cy, sy, s.wind_now, s.collapse_t),
            1.0,
        );
        let prev =
            meadow_place_vertex(local, s.world_xz, s.ground_y, cy, sy, s.wind_prev, s.collapse_t);
        let clip = mesh_view.clip_from_world * world;
        mesh_out.vertices[t].position = clip;
        mesh_out.vertices[t].world_position = world;
        mesh_out.vertices[t].prev_world_position = vec4<f32>(prev, 1.0);
        mesh_out.vertices[t].y_norm = local.w;
        mesh_out.vertices[t].albedo = meadow_unpack_albedo(s.albedo);
        // Clip w is the view depth under a perspective projection.
        mesh_out.vertices[t].blade_normal = meadow_blade_normal_varying(
            v, tuft, blade, cy, sy, s.wind_now * s.collapse_t, clip.w);
    }

    let total_prims = count * n_prims;
    for (var t = lid; t < total_prims; t = t + 32u) {
        let s = t / n_prims;
        let pr = t % n_prims;
        var tri: vec3<u32>;
        if (payload.band == 1u) {
            // Tuft arms: sequential triangles (mesh.rs tuft indices).
            tri = vec3<u32>(pr * 3u, pr * 3u + 1u, pr * 3u + 2u);
        } else {
            tri = BLADE_TRIS[pr];
        }
        mesh_out.primitives[t].indices = tri + vec3<u32>(s * n_verts);
    }
}

// ---------- mesh stage: depth-only (shadow cascades) ----------

struct MeadowShadowVertexOut {
    @builtin(position) position: vec4<f32>,
}

struct MeadowShadowPrimOut {
    @builtin(triangle_indices) indices: vec3<u32>,
}

struct MeadowShadowMeshOut {
    @builtin(vertices) vertices: array<MeadowShadowVertexOut, MESH_SHADOW_OUT_VERTS>,
    @builtin(primitives) primitives: array<MeadowShadowPrimOut, MESH_SHADOW_OUT_PRIMS>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> shadow_mesh_out: MeadowShadowMeshOut;

// Proxy-silhouette template: base-left, base-right, tip. Verts 0/1 are
// the blade's base edge and vert 10 is the tip, so the triangle
// keeps the blade's height, width, yaw, lean, and tip wind sway —
// it only loses the mid-blade bow of the 5-layer ribbon.
const SHADOW_TEMPLATE_VERTS = array<u32, 3>(0u, 1u, 10u);

@mesh(shadow_mesh_out)
@payload(payload)
@workgroup_size(32)
fn meadow_mesh_shadow(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // Shadow views only ever spawn band 0 (the task stage rejects tufts),
    // and each blade is a single proxy triangle.
    let per_wg = MESH_WG_SHADOW_BLADES;
    let n_verts = 3u;
    let base = wg.x * per_wg;
    let count = min(per_wg, payload.count - base);

    if (lid == 0u) {
        shadow_mesh_out.vertex_count = count * n_verts;
        shadow_mesh_out.primitive_count = count;
    }

    let total_verts = count * n_verts;
    for (var t = lid; t < total_verts; t = t + 32u) {
        let s = payload.survivors[base + t / n_verts];
        let v = SHADOW_TEMPLATE_VERTS[t % n_verts];
        // `MEADOW_CHORD_WIDTH_SCALE` conserves occlusion area: the
        // base-to-tip triangle is narrower than the tapered ribbon's
        // silhouette, and PCF-filtered grass shadows read proportionally
        // fainter without it.
        let blade = BladeAttrs(
            s.world_xz, s.yaw, s.height, s.width * MEADOW_CHORD_WIDTH_SCALE, 0.0, 1.0, 0u);
        let world = meadow_place_vertex(
            meadow_template_vertex(v, false, blade),
            s.world_xz,
            s.ground_y,
            cos(s.yaw),
            sin(s.yaw),
            s.wind_now,
            s.collapse_t,
        );
        shadow_mesh_out.vertices[t].position = mesh_view.clip_from_world * vec4<f32>(world, 1.0);
    }

    for (var t = lid; t < count; t = t + 32u) {
        let v_base = t * n_verts;
        shadow_mesh_out.primitives[t].indices =
            vec3<u32>(v_base, v_base + 1u, v_base + 2u);
    }
}
