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
// This file is NOT a standalone WGSL module. `mesh_path.rs` assembles the
// final source as:
//
//     enable wgpu_mesh_shader;
//     <meadow_shared.wgsl minus its #define_import_path line>
//     <this file>
//
// so `VariantParams` / `BladeAttrs` / `local_blade_position` /
// `tuft_local_position` come from the shared library verbatim (no drift),
// and the `enable` directive precedes all declarations as WGSL requires.
//
// Dispatch: one task workgroup per CPU-built `(patch_idx, slice_base)`
// pair covering exactly the active patches' real blade counts
// (`task_slices`; sizing a grid for the 65536-blade maximum instead
// launches ~mostly-empty task workgroups, millions per frame across the
// views). The flat work list can exceed wgpu's 65535 per-dimension
// dispatch cap, so it folds over a fixed-stride 2D grid — see
// `mesh.rs::MESH_TASK_DISPATCH_STRIDE`. The view is selected by a
// dynamic-offset uniform (`mesh_view`), and the per-view cull data is
// the SAME `view_cull` storage buffer the compute path uploads (frusta
// + flags + lod_max + shadow_near; the mesh path ignores the base/cap
// region fields).
//
// Bindings live at @group(3) in EVERY pipeline built from this module:
// the main pipeline has bevy_pbr's view + binding-array groups at 0/1
// (the composed PBR fragment hardcodes them) and an empty group 2; the
// depth-only shadow pipeline fills slots 0-2 with empty bind groups so
// the one module serves both layouts.

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
// proxy triangle (base edge + curled/wind-swept tip = template verts
// 0/1/10), so 21 × 3 = 63 verts / 21 tris — 9× less rasterization than
// the full ribbon, with a silhouette that only loses the mid-blade curl
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
const TUFT_SALT: u32 = 0x5C3Bu;

// ---------- structs (MIRROR: meadow_compute.wgsl) ----------
// Any change to these structs or to the derivation helpers below MUST be
// copied to/from `meadow_compute.wgsl` — the two paths must derive
// bit-identical blades, or switching between them (the runtime toggle)
// shows different fields. `tests` in `mesh_path.rs` string-compares the
// mirrored helper bodies.

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

struct MeadowViewCull {
    frustum: array<vec4<f32>, 6>,
    params: vec4<f32>,
    params2: vec4<f32>,
}

struct MeadowViewCullData {
    count: u32,
    views: array<MeadowViewCull, 6>,
}

// ---------- mesh-path-only per-view uniform ----------
// Rust mirror: `mesh_path.rs::MeadowMeshViewUniform`. One entry per view
// slot in a dynamic-offset uniform buffer; the pass binds the right
// offset per view. Matrices are split jittered/unjittered so motion
// vectors exclude the TAA/DLSS jitter, mirroring bevy_pbr's prepass.
struct MeadowMeshView {
    // Rasterization matrix (includes temporal jitter when active).
    clip_from_world: mat4x4<f32>,
    // Jitter-free current matrix (motion-vector numerator).
    unjittered_clip_from_world: mat4x4<f32>,
    // Previous frame's jitter-free matrix (motion-vector denominator).
    prev_clip_from_world: mat4x4<f32>,
    // x = view slot (indexes `view_cull.views`); yzw reserved.
    params: vec4<u32>,
}

// ---------- bindings (@group(3); see header) ----------

@group(3) @binding(0) var<uniform> variant_params: VariantParams;
@group(3) @binding(1) var<storage, read> patches: array<PatchData>;
@group(3) @binding(2) var<storage, read> trunk_slots: array<PatchTrunkSlot>;
@group(3) @binding(3) var heightfield: texture_2d<f32>;
@group(3) @binding(4) var<storage, read> view_cull: MeadowViewCullData;
// CPU-built work list: one `(patch_idx, slice_base)` per task workgroup.
// `count` (not `arrayLength`) bounds the grid — the buffer is grown-only
// and the folded 2D dispatch's last row overshoots.
struct MeadowTaskSlices {
    count: u32,
    pad: u32,
    entries: array<vec2<u32>>,
}
@group(3) @binding(5) var<storage, read> task_slices: MeadowTaskSlices;
@group(3) @binding(6) var<uniform> mesh_view: MeadowMeshView;

// ---------- hash helpers (MIRROR: meadow_compute.wgsl) ----------

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

// ---------- derivation helpers (MIRROR: meadow_compute.wgsl) ----------

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

// ---------- wind + palette (MIRROR: meadow.wgsl) ----------

fn wind_displacement(world_xz: vec2<f32>, blade_y_norm: f32, t: f32, clump: f32) -> vec2<f32> {
    let speed_mul = variant_params.wind_state.x;
    let gustiness = variant_params.wind_state.y;
    let crest_k = variant_params.wind_state.z;
    let amp = variant_params.wind.x * speed_mul;
    let period = max(variant_params.wind.y, 1e-3);
    let dir = normalize(variant_params.wind_direction.xy + vec2<f32>(0.0001, 0.0));
    let phase = (t / period) * 6.2831853 + clump * 1.57 - dot(world_xz, dir) * crest_k;
    let gust_osc = 0.6 + 0.4 * sin(t * 0.25 + world_xz.x * 0.11 + world_xz.y * 0.09);
    let gust = mix(1.0, gust_osc, gustiness);
    let sway = sin(phase) * amp * gust * blade_y_norm;
    return dir * sway;
}

fn season_palette() -> vec3<f32> {
    let a_idx = u32(variant_params.season_blend.x);
    let b_idx = u32(variant_params.season_blend.y);
    let t = variant_params.season_blend.z;
    var a = variant_params.palette_summer.rgb;
    var b = variant_params.palette_summer.rgb;
    if (a_idx == 0u) { a = variant_params.palette_spring.rgb; }
    else if (a_idx == 1u) { a = variant_params.palette_summer.rgb; }
    else if (a_idx == 2u) { a = variant_params.palette_autumn.rgb; }
    else { a = variant_params.palette_winter.rgb; }
    if (b_idx == 0u) { b = variant_params.palette_spring.rgb; }
    else if (b_idx == 1u) { b = variant_params.palette_summer.rgb; }
    else if (b_idx == 2u) { b = variant_params.palette_autumn.rgb; }
    else { b = variant_params.palette_winter.rgb; }
    return mix(a, b, clamp(t, 0.0, 1.0));
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
    clump_seed: f32,
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
    let blade_count = min(u32(p.blade_count_noise_canopy_flags.x), 65536u);
    if (blade_count == 0u || slice_base >= blade_count) {
        return vec3<u32>(0u, 0u, 0u);
    }

    let vc = view_cull.views[view_slot];
    let is_shadow = (u32(vc.params.x) & MEADOW_VIEW_FLAG_SHADOW) != 0u;

    // Patch-level frustum reject — same math as the compute kernel.
    let centre_xz = p.centre_xz_radius_seed.xy;
    let centre_y = sample_heightfield(centre_xz);
    let max_h = variant_params.height_range.y;
    let edge_noise = p.blade_count_noise_canopy_flags.y;
    let cull_radius = p.centre_xz_radius_seed.z * (1.0 + edge_noise) + max_h * 2.0;
    if (patch_sphere_culled(vec3<f32>(centre_xz.x, centre_y, centre_xz.y), cull_radius, vc, is_shadow)) {
        return vec3<u32>(0u, 0u, 0u);
    }

    let lod_full = variant_params.lod.x;
    let lod_max = vc.params.y;
    let tuft_start = variant_params.tuft.x;
    let dist = length(variant_params.viewer_world_xz.xy - centre_xz);

    // Camera view-depth of the patch centre (see meadow_compute.wgsl for
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

    // Same survival-threshold formulas as `cull_and_compact`.
    let near_t = clamp((dist - lod_full) / max(tuft_start - lod_full, 1e-3), 0.0, 1.0);
    let near_threshold = 1.0 - near_t;
    let tuft_ramp_t = clamp((dist - tuft_start) / max(lod_max - tuft_start, 1e-3), 0.0, 1.0);
    let tuft_density = mix(variant_params.tuft.y, variant_params.tuft.z, tuft_ramp_t);
    let height_fade_start = variant_params.tuft.w;
    var tuft_height_scale = 1.0;
    if (band == 1u) {
        tuft_height_scale =
            clamp((lod_max - dist) / max(lod_max - height_fade_start, 1e-3), 0.0, 1.0);
    }

    var threshold = near_threshold;
    if (band == 1u) {
        threshold = tuft_density;
    } else if (is_shadow) {
        let shadow_full_dist = variant_params.density.z;
        let shadow_far_density = variant_params.density.w;
        let ramp_t = clamp(
            (view_depth - shadow_full_dist) / max(lod_max - shadow_full_dist, 1e-3),
            0.0,
            1.0,
        );
        threshold = mix(1.0, shadow_far_density, ramp_t);
    }
    if (threshold <= 0.0) {
        return vec3<u32>(0u, 0u, 0u);
    }

    // Per-lane: gate + derive lane `lid`'s blade slot. MIRROR of
    // `cull_and_compact`'s per-blade body (band routing, cascade clip,
    // trunk-disc collapse, heightfield sample).
    let p_seed = bitcast<u32>(p.centre_xz_radius_seed.w);
    let b = slice_base + lid;
    var survived = false;
    var out: SurvivorBlade;
    if (b < blade_count) {
        var survived_gate = false;
        if (band == 0u) {
            if (is_shadow) {
                survived_gate = hash01_pair(p_seed, b) <= threshold;
            } else {
                survived_gate = hash01_u32(p_seed ^ b) <= threshold;
            }
        } else {
            survived_gate = hash01_pair(p_seed, b ^ TUFT_SALT) <= threshold;
        }
        if (survived_gate) {
            let blade = derive_blade(b, p);
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
                var wind_now = wind_displacement(
                    blade.world_xz, 1.0, variant_params.wind.z, blade.clump_seed);
                var wind_prev = wind_displacement(
                    blade.world_xz, 1.0, variant_params.wind.w, blade.clump_seed);
                if (is_shadow) {
                    // Fade shadow-caster sway with camera distance: a far
                    // blade's shadow is sub-texel in the cascade, and
                    // animating it reads as shimmering noise. Same ramp
                    // the shadow density uses, so sway settles exactly
                    // where the casters thin out.
                    let sway_atten = 1.0 - clamp(
                        (view_depth - variant_params.density.z)
                            / max(lod_max - variant_params.density.z, 1e-3),
                        0.0,
                        1.0,
                    );
                    wind_now = wind_now * sway_atten;
                    wind_prev = wind_prev * sway_atten;
                }
                out.world_xz = blade.world_xz;
                out.yaw = blade.yaw;
                out.height = blade.height * tuft_height_scale;
                out.width = blade.width;
                out.clump_seed = blade.clump_seed;
                out.ground_y = ground_y;
                out.collapse_t = collapse_t;
                out.wind_now = wind_now;
                out.wind_prev = wind_prev;
                survived = true;
            }
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

// Expand one template vertex of a blade/tuft to world space.
// `wind_full` is the blade's full-amplitude wind displacement (task-
// computed; sway is linear in y_norm — see `SurvivorBlade`), so the
// previous-frame position is this same function with the previous wind.
// Returns (world.xyz, y_norm).
fn blade_vertex(
    vert_idx: u32,
    band: u32,
    blade: BladeAttrs,
    ground_y: f32,
    collapse_t: f32,
    wind_full: vec2<f32>,
) -> vec4<f32> {
    var local: vec3<f32>;
    if (band == 1u) {
        local = tuft_local_position(vert_idx, blade);
    } else {
        local = local_blade_position(vert_idx, blade);
    }
    let cy = cos(blade.yaw);
    let sy = sin(blade.yaw);
    let rotated_x = cy * local.x + sy * local.z;
    let rotated_z = -sy * local.x + cy * local.z;
    let blade_world_xz = blade.world_xz + vec2<f32>(rotated_x, rotated_z);
    let blade_world_y = ground_y + local.y;
    let y_norm = local.y / max(blade.height, 1e-3);

    let target_world_xz = blade_world_xz + wind_full * y_norm;

    let final_xz = mix(blade.world_xz, target_world_xz, collapse_t);
    let final_y = mix(ground_y, blade_world_y, collapse_t);
    return vec4<f32>(final_xz.x, final_y, final_xz.y, y_norm);
}

// ---------- mesh stage: full (main view) ----------

struct MeadowVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    // Previous-frame world position (prev wind time; static root), for
    // per-fragment motion vectors written to the meadow-owned MV target.
    @location(1) prev_world_position: vec4<f32>,
    // x = y_norm (shade/tip ramp), y = clump luminance.
    @location(2) misc: vec2<f32>,
}

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
        let blade = BladeAttrs(s.world_xz, s.yaw, s.height, s.width, s.clump_seed, 1.0);
        let cur = blade_vertex(v, payload.band, blade, s.ground_y, s.collapse_t, s.wind_now);
        let prev = blade_vertex(v, payload.band, blade, s.ground_y, s.collapse_t, s.wind_prev);
        let world = vec4<f32>(cur.xyz, 1.0);
        mesh_out.vertices[t].position = mesh_view.clip_from_world * world;
        mesh_out.vertices[t].world_position = world;
        mesh_out.vertices[t].prev_world_position = vec4<f32>(prev.xyz, 1.0);
        mesh_out.vertices[t].misc = vec2<f32>(cur.w, 0.85 + 0.30 * s.clump_seed);
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
// the blade's base edge and vert 10 is the curled tip, so the triangle
// keeps the blade's height, width, yaw, forward curl, and tip wind sway —
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
        // Width × 4/3 conserves occlusion area: the base-to-tip triangle
        // covers 3/4 of the tapered ribbon's silhouette, and PCF-filtered
        // grass shadows read proportionally fainter without it.
        let blade =
            BladeAttrs(s.world_xz, s.yaw, s.height, s.width * (4.0 / 3.0), s.clump_seed, 1.0);
        let cur = blade_vertex(v, 0u, blade, s.ground_y, s.collapse_t, s.wind_now);
        shadow_mesh_out.vertices[t].position =
            mesh_view.clip_from_world * vec4<f32>(cur.xyz, 1.0);
    }

    for (var t = lid; t < count; t = t + 32u) {
        let v_base = t * n_verts;
        shadow_mesh_out.primitives[t].indices =
            vec3<u32>(v_base, v_base + 1u, v_base + 2u);
    }
}

// ---------- fragments: flat-lit fallback ----------
//
// Used when PBR fragment composition is unavailable (debug / composition
// failure). Season palette × height shade × fixed lambert — deliberately
// simple; the shipping fragment is the naga_oil-composed PBR one
// (`mesh_path.rs::PBR_FRAGMENT_SOURCE / compose_pbr_fragment`).

fn flat_lit_color(misc: vec2<f32>) -> vec4<f32> {
    let palette = season_palette();
    let shade = mix(0.70, 1.00, misc.x);
    let lum = misc.y;
    let ndotl = 0.85;
    return vec4<f32>(palette * shade * lum * ndotl, 1.0);
}

@fragment
fn meadow_frag_flat(in: MeadowVertexOut) -> @location(0) vec4<f32> {
    return flat_lit_color(in.misc);
}

// Jitter-free current/previous NDC delta — same UV-delta convention as
// bevy_pbr's motion-vector prepass. Written to the MEADOW-OWNED motion
// target (color target 1 of the main pass; z = valid flag) — bevy's real
// MV prepass texture rides inside the mesh-view bind group as a sampled
// resource, so it can't be an attachment of the same pass; a fullscreen
// composite copies valid texels across afterwards (`meadow_mv_composite`).
fn meadow_motion_vector(world: vec4<f32>, prev_world: vec4<f32>) -> vec2<f32> {
    let clip = mesh_view.unjittered_clip_from_world * vec4<f32>(world.xyz, 1.0);
    let prev_clip = mesh_view.prev_clip_from_world * vec4<f32>(prev_world.xyz, 1.0);
    let ndc = clip.xy / clip.w;
    let prev_ndc = prev_clip.xy / prev_clip.w;
    return (ndc - prev_ndc) * vec2<f32>(0.5, -0.5);
}

struct MeadowFlatMvOut {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec4<f32>,
}

@fragment
fn meadow_frag_flat_mv(in: MeadowVertexOut) -> MeadowFlatMvOut {
    var out: MeadowFlatMvOut;
    out.color = flat_lit_color(in.misc);
    out.motion = vec4<f32>(
        meadow_motion_vector(in.world_position, in.prev_world_position),
        1.0,
        0.0,
    );
    return out;
}
