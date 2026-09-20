//! Fills Solari's per-view receiver-override texture for meadow pixels, so rays
//! leaving a blade start on the blade instead of being pushed along the upright
//! shading normal. Solari clears the texture before this pass; pixels left
//! untouched keep its ordinary ray-origin policy.
//!
//! Three tiers, per pixel flagged in the G-buffer by the meadow raster paths:
//! - thin receiver only: the depth-reconstructed position, no offset;
//! - packed geometric normal: that position plus the blade's face normal;
//! - exact anchor: the raytracing triangle under the pixel, when a camera ray
//!   finds the near-band ribbon that coincides with the rasterised blade.
//! An unmatched pixel keeps the packed-normal tier, never a different triangle.
import bevy_render::utils::octahedral_decode;
import bevy_render::view::View;
import bevy_solari::realtime::gbuffer_utils::reconstruct_world_position;
import bevy_solari::realtime::receiver_override::{
    encode_receiver_override_surface,
    encode_receiver_override_surface_normal,
    encode_receiver_override_triangle,
};
import bevy_solari::scene::bindings::{geometry_ids, trace_ray, RAY_T_MIN, RAY_T_MAX};
import bevy_solari::scene::triangle_anchor::resolve_triangle_anchor;
import bevy_meadow::meadow_shared::{
    MEADOW_GBUFFER_THIN_RECEIVER_BIT,
    MEADOW_GBUFFER_GEOMETRIC_NORMAL_BIT,
    meadow_unpack_geometric_normal_oct,
};

enable wgpu_ray_query;

// Mirror of `solari.rs::MeadowReceiverParams`.
struct MeadowReceiverParams {
    // `MeadowReceiverOrigins` toggles, 0 or 1.
    unshifted: u32,
    geometric_normals: u32,
    exact_anchors: u32,
    // `RaytracingInstanceTag` value carried by near-band (exact ribbon) casters.
    near_tag: u32,
}

// Group 0 is Solari's raytracing scene bind group.
@group(1) @binding(0) var<uniform> view: View;
@group(1) @binding(1) var gbuffer: texture_2d<u32>;
@group(1) @binding(2) var depth_buffer: texture_depth_2d;
@group(1) @binding(3) var receiver_overrides: texture_storage_2d<rgba32uint, write>;
@group(1) @binding(4) var<uniform> params: MeadowReceiverParams;

// Sine of the angular uncertainty of a packed geometric normal. The oct8x2
// payload (two unorm8 octahedral coordinates) can land on the other side of a
// ray that grazes the blade plane; inside this band Solari applies no origin
// offset. Pinned by the `oct8x2_*` tests in `mesh_path.rs`.
const PACKED_NORMAL_TANGENT_GUARD = 0.02;

// Position window for accepting a camera-ray hit as the rasterised blade, in
// metres. Scaled with coordinate magnitude, and capped so large world
// coordinates cannot match a neighbouring blade centimetres away.
const MATCH_TOLERANCE_MIN = 0.00025;
const MATCH_TOLERANCE_MAX = 0.002;
// Minimum |cos| between the raytracing triangle normal and the packed normal.
const MATCH_NORMAL_COSINE = 0.995;

// Camera-ray match of a meadow pixel against the raytracing scene. Returns the
// TRIANGLE texel on success and `fallback` otherwise. Conservative, not a
// reconstruction-error bound: coincident blades stay ambiguous without raster
// primitive ids, and an unmatched pixel is preferred over a wrong anchor.
fn match_receiver(
    pixel: vec2<u32>,
    world_position: vec3<f32>,
    packed_normal: vec3<f32>,
    fallback: vec4<u32>,
) -> vec4<u32> {
    if !(dot(packed_normal, packed_normal) > 0.5) { return fallback; }
    // The ray starts at this pixel's near plane, which serves perspective and
    // orthographic cameras and excludes geometry clipped from the raster view.
    // Like Solari's G-buffer resolve, this assumes a viewport origin of zero.
    if any(view.main_pass_viewport.xy != vec2(0.0)) { return fallback; }
    let camera_origin = reconstruct_world_position(pixel, 1.0, view.main_pass_viewport.zw, view.world_from_clip);
    let delta = world_position - camera_origin;
    let distance = length(delta);
    if !(distance > RAY_T_MIN) || !(distance < RAY_T_MAX) { return fallback; }
    let magnitude = max(max(abs(world_position.x), abs(world_position.y)), abs(world_position.z));
    let tolerance = clamp(8.0 * 1.1920928955078125e-7 * magnitude, MATCH_TOLERANCE_MIN, MATCH_TOLERANCE_MAX);
    // Closest hit: the visible blade is the first surface along the camera ray.
    let hit = trace_ray(camera_origin, delta / distance, RAY_T_MIN, distance + tolerance, RAY_FLAG_NONE);
    if hit.kind == RAY_QUERY_INTERSECTION_NONE { return fallback; }
    // Only near-band ribbons reproduce the rasterised blade; far chords, terrain
    // and meshes carry other tags and never anchor a pixel.
    let slot = hit.instance_custom_data;
    if slot >= arrayLength(&geometry_ids) { return fallback; }
    if geometry_ids[slot].user_tag != params.near_tag { return fallback; }
    let anchor = resolve_triangle_anchor(slot, hit.primitive_index, hit.barycentrics, false);
    if !anchor.valid { return fallback; }
    if !(length(anchor.position - world_position) <= tolerance)
        || !(abs(dot(anchor.normal, packed_normal)) >= MATCH_NORMAL_COSINE) {
        return fallback;
    }
    return encode_receiver_override_triangle(slot, hit.primitive_index, hit.barycentrics);
}

@compute @workgroup_size(8, 8, 1)
fn resolve_meadow_receivers(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if any(pixel >= vec2<u32>(view.main_pass_viewport.zw)) || any(pixel >= textureDimensions(receiver_overrides)) { return; }
    let depth = textureLoad(depth_buffer, pixel, 0);
    if !(depth > 0.0) { return; }
    let gpixel = textureLoad(gbuffer, pixel, 0);
    if (gpixel.a & MEADOW_GBUFFER_THIN_RECEIVER_BIT) == 0u { return; }

    let has_normal = (gpixel.a & MEADOW_GBUFFER_GEOMETRIC_NORMAL_BIT) != 0u && params.geometric_normals != 0u;
    if !has_normal {
        if params.unshifted != 0u {
            textureStore(receiver_overrides, pixel, encode_receiver_override_surface());
        }
        return;
    }

    // The octahedral floats pass through unchanged, so Solari decodes the same
    // normal this pass matches against.
    let oct = meadow_unpack_geometric_normal_oct(gpixel.b);
    var texel = encode_receiver_override_surface_normal(oct, PACKED_NORMAL_TANGENT_GUARD);
    if params.exact_anchors != 0u {
        let world_position = reconstruct_world_position(pixel, depth, view.main_pass_viewport.zw, view.world_from_clip);
        texel = match_receiver(pixel, world_position, octahedral_decode(oct), texel);
    }
    textureStore(receiver_overrides, pixel, texel);
}
