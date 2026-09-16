// The meadow mesh path's bind group (@group(3), `mesh_path.rs::
// MEADOW_GROUP`; layout in `meadow_mesh_bgl_desc`) and the declarations
// shared across its stages. The task/mesh module (`meadow_mesh.wesl`),
// the PBR fragments (`meadow_mesh_pbr_fragment.wesl`,
// `meadow_mesh_deferred_fragment.wesl`), and their common PbrInput
// assembly (`meadow_mesh_pbr_common.wesl`) import from here, so every
// binding, the mesh → fragment varyings, the per-view matrices, and the
// palette have exactly one definition. WESL imports are item-level, so a
// fragment that only reads the palette pulls in nothing else.

import bevy_meadow::meadow_shared::{VariantParams, season_palette_of};

// ---------- structs (MIRROR: meadow_compute.wesl) ----------
// Any change to these structs or to the derivation helpers in
// `meadow_mesh.wesl` MUST be copied to/from `meadow_compute.wesl` — the
// two paths must derive bit-identical blades, or switching between them
// (the runtime toggle) shows different fields. `tests` in `mesh_path.rs`
// string-compares the mirrored helper bodies.

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

// CPU-built work list: one `(patch_idx, slice_base)` per task workgroup.
// `count` (not `arrayLength`) bounds the grid — the buffer is grown-only
// and the folded 2D dispatch's last row overshoots.
struct MeadowTaskSlices {
    count: u32,
    pad: u32,
    entries: array<vec2<u32>>,
}

// Rust mirror: `mesh_path.rs::MeadowMeshViewUniform`. One entry per view
// slot, each bound by its own bind group; the pass binds the right
// group per view. Matrices are split jittered/unjittered so motion
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

// ---------- bindings ----------

@group(3) @binding(0) var<uniform> variant_params: VariantParams;
@group(3) @binding(1) var<storage, read> patches: array<PatchData>;
@group(3) @binding(2) var<storage, read> trunk_slots: array<PatchTrunkSlot>;
@group(3) @binding(3) var heightfield: texture_2d<f32>;
@group(3) @binding(4) var<storage, read> view_cull: MeadowViewCullData;
@group(3) @binding(5) var<storage, read> task_slices: MeadowTaskSlices;
@group(3) @binding(6) var<uniform> mesh_view: MeadowMeshView;

// ---------- shared stage interface ----------

// Mesh-stage output / fragment input.
struct MeadowVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    // Previous-frame world position (prev wind time; static root), for
    // per-fragment motion vectors written to the meadow-owned MV target.
    @location(1) prev_world_position: vec4<f32>,
    // x = y_norm (shade/tip ramp), y = clump luminance.
    @location(2) misc: vec2<f32>,
}

// Same blend as `meadow.wesl::season_palette`, over this group's
// `variant_params`.
fn season_palette() -> vec3<f32> {
    return season_palette_of(
        variant_params.season_blend,
        variant_params.palette_spring,
        variant_params.palette_summer,
        variant_params.palette_autumn,
        variant_params.palette_winter,
    );
}

// Jitter-free current/previous NDC delta — same UV-delta convention as
// bevy_pbr's motion-vector prepass (`calculate_motion_vector`), computed
// from the mesh-view matrices instead of the previous-view uniform so no
// extra view binding is needed. Forward views write it to the
// MEADOW-OWNED motion target (color target 1 of the main pass; z = valid
// flag) — bevy's real MV prepass texture rides inside the mesh-view bind
// group as a sampled resource, so it can't be an attachment of the same
// pass; a fullscreen composite copies valid texels across afterwards
// (`meadow_mv_composite.wgsl`). Deferred views write it straight into
// bevy's MV prepass attachment.
fn meadow_motion_vector(world: vec4<f32>, prev_world: vec4<f32>) -> vec2<f32> {
    let clip = mesh_view.unjittered_clip_from_world * vec4<f32>(world.xyz, 1.0);
    let prev_clip = mesh_view.prev_clip_from_world * vec4<f32>(prev_world.xyz, 1.0);
    let ndc = clip.xy / clip.w;
    let prev_ndc = prev_clip.xy / prev_clip.w;
    return (ndc - prev_ndc) * vec2<f32>(0.5, -0.5);
}
