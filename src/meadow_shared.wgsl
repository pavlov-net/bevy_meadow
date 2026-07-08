// Meadow shared shader library.
//
// Only the structs/helpers used by BOTH the compute kernel
// (`meadow_compute.wgsl`) and the raster passthrough shader
// (`meadow.wgsl`) live here. Compute-only types (PatchData,
// PatchTrunkSlot, the view-cull + indirect structs, the hash helpers)
// live in `meadow_compute.wgsl` directly — this is a *composable*
// module (`#define_import_path`), and naga_oil forbids exported
// identifiers that naga's WGSL writer would rename on round-trip
// (notably names ending in a digit, e.g. `pad0` / `hash_u32`; see
// bevy_pbr `utils.wgsl:86`). Keeping those out of the export surface
// sidesteps the rule; the one shared pad field is named `pad`.
//
// Struct layouts here are the contract with the Rust side
// (`material.rs::VariantParams`, `compute.rs`); std430 offsets are
// load-bearing.

#define_import_path bevy_meadow::shared

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
