// PbrInput assembly shared by the mesh path's forward and deferred PBR
// fragments. Mirrors what the compute path's `meadow.wesl` fragments
// produce (season palette × height shade × clump luminance,
// shadow-receiver bit forced, tip emissive), but builds the `PbrInput`
// by hand — there is no material or mesh bind group on these pipelines,
// so `pbr_input_from_standard_material` is not usable. The hardcoded
// material values mirror the meadow `StandardMaterial`
// (`plugin.rs::register_variant`: white base, perceptual_roughness 0.85,
// double-sided).
//
// Kept apart from `meadow_mesh_bindings.wesl` so the def-free task/mesh
// module never pulls bevy_pbr into its import closure.

import bevy_pbr::render::{
    pbr_types::{PbrInput, pbr_input_new},
    pbr_functions::calculate_view,
    mesh_view_bindings::view,
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
};
import bevy_meadow::meadow_shared::meadow_shading_normal;
import bevy_meadow::meadow_mesh_bindings::{MeadowVertexOut, season_palette};

// Both paths need the view direction to keep the grass shading normal in
// the visible hemisphere. Deferred lighting reconstructs that direction
// from depth when it consumes the resulting G-buffer.
fn meadow_blade_pbr_input(in: MeadowVertexOut, is_front: bool) -> PbrInput {
    var pbr_input = pbr_input_new();
    let palette = season_palette();
    let shade = mix(0.70, 1.00, in.misc.x);
    let lum = in.misc.y;
    // Mirror of meadow.wesl::apply_blade_palette with the meadow
    // StandardMaterial's values baked in (white base → blade_lum = lum,
    // roughness 0.85).
    pbr_input.material.base_color = vec4<f32>(palette * shade * lum, 1.0);
    pbr_input.material.perceptual_roughness = 0.85;
    let tip_glow = smoothstep(0.7, 1.0, in.misc.x) * 0.06;
    pbr_input.material.emissive = vec4<f32>(palette * tip_glow, 1.0);
    pbr_input.frag_coord = in.position;
    pbr_input.world_position = in.world_position;
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr_input.V = calculate_view(in.world_position, pbr_input.is_orthographic);
    pbr_input.world_normal = meadow_shading_normal(pbr_input.V);
    pbr_input.N = pbr_input.world_normal;
    // No real MeshUniform row exists for mesh-emitted geometry; force the
    // shadow-receiver bit the same way meadow.wesl does.
    pbr_input.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;
    return pbr_input;
}
