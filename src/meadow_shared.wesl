// Meadow shared shader library (importable as `bevy_meadow::meadow_shared`).
//
// Only the structs/helpers used by BOTH the compute kernel
// (`meadow_compute.wesl`) and the raster passthrough shader
// (`meadow.wesl`) live here. Compute-only types (PatchData,
// PatchTrunkSlot, the view-cull + indirect structs, the hash helpers)
// live in `meadow_compute.wesl` directly. This file is also spliced
// verbatim into the mesh-shader module source
// (`mesh_path.rs::assemble_geom_source`), so it must stay plain WGSL —
// no imports or `@if` directives.
//
// Struct layouts here are the contract with the Rust side
// (`material.rs::VariantParams`, `compute.rs`); std430 offsets are
// load-bearing.

// Mirror of `material.rs::VariantParams` (16 × vec4 = 256 B). Bound at
// `@binding(100)` in raster (material group) and `@binding(0)` in
// compute.
struct VariantParams {
    height_range: vec4<f32>,
    width_range: vec4<f32>,
    wind: vec4<f32>,
    wind_direction: vec4<f32>,
    wind_state: vec4<f32>,
    palette_spring: vec4<f32>,
    palette_summer: vec4<f32>,
    palette_autumn: vec4<f32>,
    palette_winter: vec4<f32>,
    season_blend: vec4<f32>,
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
    pad: f32,                   // @44 -> 48
}

// Local-blade vertex computation. `vert_idx ∈ [0, 11)`; layer = idx
// pair (i*2, i*2+1) for i = 0..4, with idx 10 the tip. Half-width
// tapers as `1 - y²`; forward curl quadratic in y. Pure — depends only
// on the blade's height/width.
fn local_blade_position(vert_idx: u32, blade: BladeAttrs) -> vec3<f32> {
    if (vert_idx == 10u) {
        let y = 1.0;
        let curl = 0.42 * y * y * blade.height;
        return vec3<f32>(0.0, y * blade.height, curl);
    }
    let layer = vert_idx >> 1u;
    let side = i32(vert_idx & 1u) * 2 - 1; // -1 or +1
    let y = f32(layer) * 0.2;
    let half_w = 0.5 * (1.0 - y * y) * blade.width;
    let curl = 0.42 * y * y * blade.height;
    return vec3<f32>(f32(side) * half_w, y * blade.height, curl);
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

// Seasonal palette blend: `season_blend.xy` index the from/to palettes
// (0 spring, 1 summer, 2 autumn, 3 winter), `.z` is the blend factor.
// Takes the `VariantParams` fields explicitly (like `wind_sway`) so the
// raster material and the mesh path each bind `variant_params` at their
// own group slot and share this one body.
fn season_palette_of(
    season_blend: vec4<f32>,
    palette_spring: vec4<f32>,
    palette_summer: vec4<f32>,
    palette_autumn: vec4<f32>,
    palette_winter: vec4<f32>,
) -> vec3<f32> {
    let a_idx = u32(season_blend.x);
    let b_idx = u32(season_blend.y);
    let t = season_blend.z;
    var a = palette_summer.rgb;
    var b = palette_summer.rgb;
    if (a_idx == 0u) { a = palette_spring.rgb; }
    else if (a_idx == 1u) { a = palette_summer.rgb; }
    else if (a_idx == 2u) { a = palette_autumn.rgb; }
    else { a = palette_winter.rgb; }
    if (b_idx == 0u) { b = palette_spring.rgb; }
    else if (b_idx == 1u) { b = palette_summer.rgb; }
    else if (b_idx == 2u) { b = palette_autumn.rgb; }
    else { b = palette_winter.rgb; }
    return mix(a, b, clamp(t, 0.0, 1.0));
}

// Meadow's smoothed upright normal keeps dense grass lighting coherent, but
// a hillside can put the camera below that normal's hemisphere. Solari's
// BRDF rejects NdotV <= 0, making those blades black at camera elevation.
// Preserve the upright normal whenever possible and tilt it just enough
// toward the viewer otherwise. Shared by both raster rendering paths.
fn meadow_shading_normal(view_direction: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let length_squared = dot(view_direction, view_direction);
    if length_squared < 1e-12 { return up; }
    let v = view_direction * inverseSqrt(length_squared);
    let n_dot_v = v.y;
    let margin = 0.1;
    if n_dot_v >= margin { return up; }
    let tangent = up - v * n_dot_v;
    let tangent_length_squared = dot(tangent, tangent);
    // Directly beneath a blade there is no unique upward tangent.
    if tangent_length_squared < 1e-12 { return v; }
    return tangent * inverseSqrt(tangent_length_squared) * sqrt(1.0 - margin * margin)
        + v * margin;
}

// G-buffer A bit 27 marks a meadow pixel as a thin two-sided receiver. The
// contract is private to meadow: both raster paths write it and meadow's
// Solari receiver pass (`meadow_receivers.wesl`) is its only reader. bevy_pbr's
// deferred flags occupy bits 24..26 and the packed normal bits 0..23; neither
// is altered.
const MEADOW_GBUFFER_THIN_RECEIVER_BIT: u32 = 1u << 27u;

// G-buffer A bit 28 marks a geometric-normal payload in B, read only by
// meadow's Solari receiver pass and meaningful only together with bit 27. B
// bits 0..15 keep reflectance/metallic; bits 16..23 and 24..31 carry the
// octahedral triangle normal as two unorm8 values. The payload overwrites the
// AO and clearcoat bytes that deferred lighting reads, so the caller gates it
// with `width_range.z`.
const MEADOW_GBUFFER_GEOMETRIC_NORMAL_BIT: u32 = 1u << 28u;

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
    // derivative cross product. Absence of bit 28 invokes consumer fallback.
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
// `bevy_render::utils::octahedral_decode`. Valid only when A bit 28 is set.
fn meadow_unpack_geometric_normal_oct(gbuffer_b: u32) -> vec2<f32> {
    return vec2<f32>(f32((gbuffer_b >> 16u) & 255u), f32(gbuffer_b >> 24u)) / 255.0;
}
