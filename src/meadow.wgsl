// Meadow blade raster shader — passthrough vertex + PBR fragment.
//
// Blade geometry is derived, culled, and compacted GPU-side once per
// frame by `meadow_compute.wgsl` into a per-view `CompactedBladeRecord`
// buffer. This shader is a thin passthrough: the custom
// `DrawMeadowPatch` render command issues one `draw_indexed_indirect`
// per view whose `first_instance` is that view's base offset, so
// `@builtin(instance_index)` indexes straight into `blades[]`. The VS
// reads one record per blade instance, expands the 11 local verts (via
// `local_blade_position`), and recomputes only the wind sway (cheap,
// time-varying). `vert_idx` = `position.y` (0..10), encoded by `mesh.rs`.
//
// Structs + binding-free helpers live in `meadow_shared.wgsl`; the
// per-blade derivation (hash placement, heightfield sample, LOD gate)
// lives in `meadow_compute.wgsl`.

#import bevy_pbr::{
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
}

// Time fields are stored in `variant_params.wind.zw` (current,
// previous) rather than read from `bevy_render::globals::Globals`
// because the prepass pipeline layout doesn't include the globals
// bind group at binding 11 — referencing `globals` from this
// shader hits a runtime "binding missing from pipeline layout"
// validation error. Bevy's `tick_meadow_time` system writes
// `wind.z = time.elapsed_secs()` each frame.

// `pbr_fragment` accesses `in.world_normal` unconditionally
// (`bevy_pbr/render/pbr_fragment.wgsl:59`); the field only exists on
// `VertexOutput` under `NORMAL_PREPASS_OR_DEFERRED_PREPASS`.
// naga_oil validates the whole import tree at composition time —
// even uncalled functions — so importing `pbr_fragment` in a
// permutation without `world_normal` fails compilation. We gate the
// imports directly on Bevy's own shader defs rather than via an
// intermediate `#define MEADOW_HAS_PBR_INPUT`, because naga_oil only
// honours `#define` directives at the top of the shader, not inside
// `#ifdef` blocks.
#ifdef PREPASS_PIPELINE
    #import bevy_pbr::prepass_io::{Vertex, VertexOutput, FragmentOutput}
    #ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
        #import bevy_pbr::{
            pbr_fragment::pbr_input_from_standard_material,
            pbr_functions::alpha_discard,
            pbr_types::PbrInput,
            pbr_deferred_functions::deferred_output,
            mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
        }
    #endif
#else
    #import bevy_pbr::{
        forward_io::{Vertex, VertexOutput, FragmentOutput},
        pbr_fragment::pbr_input_from_standard_material,
        pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
        pbr_types::{PbrInput, STANDARD_MATERIAL_FLAGS_UNLIT_BIT},
        mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    }
#endif

#import bevy_meadow::shared::{
    VariantParams, CompactedBladeRecord, BladeAttrs, local_blade_position, tuft_local_position,
}

// ---------- Bindings ----------

// Per-variant params (wind + season). `patches`/`trunk_slots`/
// `heightfield` (material bindings 101/102/105) are still declared by
// the `MeadowExt` `AsBindGroup` so the GPU buffers exist for the
// compute pass to read — but this raster shader no longer references
// them (derivation moved to `meadow_compute.wgsl`), so they're omitted
// here. A bind group may carry bindings the shader doesn't use.
@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> variant_params: VariantParams;

// Per-view compacted blade records, written by `meadow_compute.wgsl`.
// Group 4 is injected into the pipeline layout by `MeadowExt::specialize`
// (groups 0-3 are view / view-array / mesh / material). The draw binds
// the variant's whole blade buffer; the view's base offset rides in via
// the indirect draw's `first_instance`, so `in.instance_index` indexes
// `blades[]` directly.
@group(4) @binding(0)
var<storage, read> blades: array<CompactedBladeRecord>;

// Wind sway in world XZ. Direction comes from `WindDirection`;
// amplitude/period are per-variant; speed/gustiness/crest spacing
// come from `MeadowWindState` via `variant_params.wind_state`. Gust
// crests travel along the wind direction so the meadow reads as one
// big wave passing through, not as every blade gusting in lockstep.
fn wind_displacement(world_xz: vec2<f32>, blade_y_norm: f32, t: f32, clump: f32) -> vec2<f32> {
    let speed_mul = variant_params.wind_state.x;
    let gustiness = variant_params.wind_state.y;
    let crest_k = variant_params.wind_state.z;
    let amp = variant_params.wind.x * speed_mul;
    let period = max(variant_params.wind.y, 1e-3);
    let dir = normalize(variant_params.wind_direction.xy + vec2<f32>(0.0001, 0.0));
    // Subtracting `dot(world_xz, dir) * crest_k` makes the crest move
    // along +dir rather than against it. `clump * 1.57` (~π/2) gives
    // ~quarter-cycle per-blade phase scatter — enough to break perfect
    // lockstep without drowning the spatial gradient that produces the
    // traveling crest. The earlier `clump * 4π` swamped the spatial
    // signal so blades visibly sloshed independently.
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

// ---------- Vertex shader ----------

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;

    // Passthrough: read the compacted record for this blade. The
    // indirect draw's `first_instance` is the view's base offset, so
    // `in.instance_index` indexes `blades[]` directly. All the heavy
    // per-blade work (placement, heightfield sample, LOD, trunk-disc
    // collapse) was done once in `meadow_compute.wgsl`; `collapse_t`
    // is baked, so this VS only expands the local geometry and
    // recomputes wind.
    let rec = blades[in.instance_index];

    // `vert_idx` (0..10) comes from POSITION.y — encoded by
    // `mesh::build_blade_template_mesh`.
    let vert_idx = u32(in.position.y);

    // Reconstruct a `BladeAttrs` for the local geometry + the
    // downstream `#ifdef` blocks (`rim_factor` is irrelevant here —
    // it's already folded into `rec.collapse_t`).
    let blade = BladeAttrs(rec.world_root_xz, rec.yaw, rec.height, rec.width, rec.clump_seed, 1.0);
    let ground_y = rec.ground_y;
    let collapse_t = rec.collapse_t;

    // Local blade space → world XZ. yaw rotates the (x, z) of the local
    // frame; height-curl moves the tip forward in z. POSITION.x marks the
    // LOD band (0 = blade, 1 = tuft) so the one shared VS expands the right
    // geometry. The band is uniform across an indirect draw (blades and
    // tufts are separate draws), so this branch is dynamically uniform —
    // `if` evaluates only the taken arm, vs `select` paying the tuft
    // transcendentals on every blade vertex (and vice-versa).
    var local: vec3<f32>;
    if (in.position.x > 0.5) {
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

    // Wind sway in world XZ on the upper portion of the blade —
    // recomputed each frame (time-varying); cheap relative to the
    // full derivation.
    let wind = wind_displacement(blade.world_xz, y_norm, variant_params.wind.z, blade.clump_seed);
    let target_world_xz = blade_world_xz + wind;
    let root_world_xz = blade.world_xz;

    // Mix root → upright-blade-with-wind by collapse_t. At
    // collapse_t = 0 the vertex sits at (root_xz, ground_y), so
    // all 11 verts of this blade converge and triangles are
    // degenerate.
    let final_xz = mix(root_world_xz, target_world_xz, collapse_t);
    let final_y = mix(ground_y, blade_world_y, collapse_t);
    let final_pos = vec3<f32>(final_xz.x, final_y, final_xz.y);

    let world_position4 = vec4<f32>(final_pos, 1.0);
    out.position = position_world_to_clip(world_position4.xyz);
    out.world_position = world_position4;

#ifdef VERTEX_NORMALS
#ifndef PREPASS_PIPELINE
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
#else
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
#endif
#endif
#endif

#ifdef VERTEX_TANGENTS
#ifndef PREPASS_PIPELINE
    out.world_tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);
#else
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    out.world_tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);
#endif
#endif
#endif

#ifdef PREPASS_PIPELINE
#ifdef MOTION_VECTOR_PREPASS
    // Motion vectors: re-expand the blade at the previous-frame root
    // (`rec.prev_*`, == current root for static placement) and wind
    // (`prev_time`) so TAA can cancel the per-vertex sway delta.
    let prev_t = variant_params.wind.w;
    let prev_wind = wind_displacement(rec.prev_root_xz, y_norm, prev_t, blade.clump_seed);
    let prev_blade_world_xz = rec.prev_root_xz + vec2<f32>(rotated_x, rotated_z);
    let prev_target_xz = prev_blade_world_xz + prev_wind;
    let prev_final_xz = mix(rec.prev_root_xz, prev_target_xz, collapse_t);
    let prev_final_y = mix(rec.prev_ground_y, rec.prev_ground_y + local.y, collapse_t);
    out.previous_world_position = vec4<f32>(prev_final_xz.x, prev_final_y, prev_final_xz.y, 1.0);
#endif
#endif

#ifdef VERTEX_UVS_A
    out.uv = vec2<f32>(0.0, y_norm);
#endif

#ifdef VERTEX_UVS_B
    out.uv_b = blade.world_xz;
#endif

#ifdef VERTEX_COLORS
    let lum = 0.85 + 0.30 * blade.clump_seed;
    out.color = vec4<f32>(lum, lum, lum, 1.0);
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = in.instance_index;
#endif

    return out;
}

// ---------- Fragment shader ----------
//
// The fragment function and its helpers are gated on the same
// `(PREPASS_PIPELINE, PREPASS_FRAGMENT)` shape Bevy's own
// `pbr_prepass.wgsl:23` uses. The shadow pass compiles this
// shader with `PREPASS_PIPELINE + DEPTH_PREPASS` and *no*
// `PREPASS_FRAGMENT` — `prepass_io::FragmentOutput` itself doesn't
// exist in that permutation (the struct is gated on
// `PREPASS_FRAGMENT` in `bevy_pbr/src/prepass/prepass_io.wgsl`),
// so emitting an `@fragment fn fragment(...) -> FragmentOutput`
// in source fails composition. Skip the fragment function
// entirely when `PREPASS_FRAGMENT` is unset.

#ifndef PREPASS_PIPELINE
@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    apply_blade_palette(in, &pbr_input);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color,
    );

    var out: FragmentOutput;
    if ((pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u) {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}

fn apply_blade_palette(in: VertexOutput, pbr_input: ptr<function, PbrInput>) {
    let blade_lum = (*pbr_input).material.base_color.rgb;
    let palette = season_palette();
    let shade = mix(0.70, 1.00, in.uv.y);
    (*pbr_input).material.base_color = vec4<f32>(palette * shade * blade_lum, 1.0);
    let tip_glow = smoothstep(0.7, 1.0, in.uv.y) * 0.06;
    (*pbr_input).material.emissive = vec4<f32>(palette * tip_glow, 1.0);
    // Patch over the garbage `pbr_input.flags` that
    // `pbr_input_from_vertex_output` reads from `mesh[in.instance_index]`.
    // `instance_index` is the per-(view, band) base + survivor ordinal,
    // which indexes `blades[]` directly (see `crate::render`) but bears no
    // relation to this driver entity's single `MeshUniform` row — so the
    // `mesh[...]` lookup returns a wrong / out-of-range row with no
    // `MESH_FLAGS_SHADOW_RECEIVER_BIT` set, causing `apply_pbr_lighting`
    // to skip `fetch_directional_shadow` (gated on that bit at
    // `pbr_functions.wgsl:615`), leaving blades fully lit. Force the bit
    // on so grass receives tree / terrain shadows.
    (*pbr_input).flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;
}
#else
#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    apply_blade_palette(in, &pbr_input);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color,
    );
    return deferred_output(in, pbr_input);
#else
    // Motion-vector-only prepass — VertexOutput has no
    // `world_normal`, so we can't build a `PbrInput`. The
    // rasterizer wrote depth + motion-vector from `out.position`
    // and `out.previous_world_position` already; defaulting the
    // FragmentOutput is fine.
    var stub: FragmentOutput;
    return stub;
#endif
}

#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
fn apply_blade_palette(in: VertexOutput, pbr_input: ptr<function, PbrInput>) {
    let blade_lum = (*pbr_input).material.base_color.rgb;
    let palette = season_palette();
    let shade = mix(0.70, 1.00, in.uv.y);
    (*pbr_input).material.base_color = vec4<f32>(palette * shade * blade_lum, 1.0);
    let tip_glow = smoothstep(0.7, 1.0, in.uv.y) * 0.06;
    (*pbr_input).material.emissive = vec4<f32>(palette * tip_glow, 1.0);
    // Same shadow-receiver patch as the forward path — `deferred_output`
    // packs the bit into the GBuffer (`pbr_deferred_types.wgsl:15`), and
    // the deferred lighting pass unpacks it before sampling shadows.
    (*pbr_input).flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;
}
#endif
#endif // PREPASS_FRAGMENT
#endif // PREPASS_PIPELINE
