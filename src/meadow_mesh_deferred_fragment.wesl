// Deferred G-buffer fragment for the mesh path on deferred main views:
// the shared `PbrInput` (`meadow_mesh_pbr_common.wesl`) packed through
// `deferred_gbuffer_from_pbr_input` + the lighting-pass id — bevy's
// deferred lighting pass (or solari) shades it from the G-buffer,
// exactly like the compute path's `meadow.wesl` deferred arm via
// `deferred_output`. The output struct is bevy's own
// `prepass::io::FragmentOutput`, so the def set (the meadow material's
// specialized deferred prepass pipeline's) makes the output locations
// match bevy's prepass attachment slots by construction.
//
// Motion vectors are written directly to bevy's MV prepass attachment
// (location 1): the only view binding this fragment statically uses is
// the `View` uniform (through `deferred_gbuffer_from_pbr_input`'s
// exposure term), served by the prepass view layout — nothing in the
// pipeline's bind groups samples the deferred/MV textures, so they can
// be attachments of this very pass. Bevy's own `calculate_motion_vector`
// would drag in the previous-view uniform; the mesh-view matrices at
// group 3 carry the same data.

import bevy_pbr::deferred::functions::deferred_gbuffer_from_pbr_input;
import bevy_pbr::prepass::io::FragmentOutput;
import bevy_meadow::meadow_mesh_bindings::{MeadowVertexOut, meadow_motion_vector};
import bevy_meadow::meadow_mesh_pbr_common::meadow_blade_pbr_input;

@fragment
fn fragment_deferred(
    in: MeadowVertexOut,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    let pbr_input = meadow_blade_pbr_input(in, is_front);

    var out: FragmentOutput;
    out.deferred = deferred_gbuffer_from_pbr_input(pbr_input);
    out.deferred_lighting_pass_id = pbr_input.material.deferred_lighting_pass_id;
@if(NORMAL_PREPASS)
    out.normal = vec4<f32>(pbr_input.N * 0.5 + vec3<f32>(0.5), 1.0);
@if(MOTION_VECTOR_PREPASS)
    out.motion_vector = meadow_motion_vector(in.world_position, in.prev_world_position);
    return out;
}
