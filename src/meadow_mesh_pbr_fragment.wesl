// Forward PBR fragment for the mesh path's main-view pipeline: the
// shared `PbrInput` (`meadow_mesh_pbr_common.wesl`) through
// `apply_pbr_lighting`, like the compute path's `meadow.wesl` forward
// fragment.
//
// Compiled by bevy's pipeline cache with the shader defs of the meadow
// material's own specialized forward pipeline, so the view-binding
// declarations pulled in through `mesh_view_bindings` match the real
// view bind group layout by construction.

import bevy_pbr::render::{
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
};
import bevy_meadow::meadow_mesh_bindings::{MeadowVertexOut, meadow_motion_vector};
import bevy_meadow::meadow_mesh_pbr_common::meadow_blade_pbr_input;

fn shade_meadow_fragment(in: MeadowVertexOut, is_front: bool) -> vec4<f32> {
    let pbr_input = meadow_blade_pbr_input(in, is_front);

    var color = apply_pbr_lighting(pbr_input);
    color = main_pass_post_lighting_processing(pbr_input, color);
    return color;
}

@fragment
fn fragment(in: MeadowVertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    return shade_meadow_fragment(in, is_front);
}

// MV variant: also writes the motion vector + valid flag to the
// meadow-owned motion target (color target 1) — composited into bevy's
// MV prepass texture by `meadow_mesh_mv_composite_pass`.
struct MeadowPbrMvOut {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec4<f32>,
}

@fragment
fn fragment_mv(in: MeadowVertexOut, @builtin(front_facing) is_front: bool) -> MeadowPbrMvOut {
    var out: MeadowPbrMvOut;
    out.color = shade_meadow_fragment(in, is_front);
    out.motion = vec4<f32>(
        meadow_motion_vector(in.world_position, in.prev_world_position),
        1.0,
        0.0,
    );
    return out;
}
