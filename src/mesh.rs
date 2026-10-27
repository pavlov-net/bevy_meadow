//! Shared LOD meshes the meadow renderer uses. Both built once at
//! plugin startup and shared by every patch in every variant; indexed
//! by LOD band (`MeadowMeshes::lod_meshes`):
//!
//! - **Band 0 — blade template** — exactly **one blade**: 11 vertices,
//!   27 indices. The near band the player resolves as individual blades.
//! - **Band 1 — tuft template** — a fanned clump of 7 mini-blades
//!   (21 vertices, 21 indices). One tuft instance stands in for many
//!   blades' silhouette in the far band; it reads as grass from any
//!   camera angle (no rotation, so no billboard popping) and casts no
//!   shadows.
//!
//! The compute kernel (`meadow_compute.wesl`) culls + compacts blades
//! per (view, band) into `CompactedBladeRecord`s; `DrawMeadowPatch`
//! (see `crate::render`) issues one `draw_indexed_indirect` per
//! (view, band) and `@builtin(instance_index)` indexes the band's
//! region directly. The vertex shader recovers `vert_idx` from
//! `POSITION.y` and the band from `POSITION.x` (0 = blade, 1 = tuft),
//! then expands the local geometry. Both meshes carry an identical
//! vertex-buffer layout (POSITION/NORMAL/UV/TANGENT/COLOR) so the single
//! specialized pipeline draws either without respecialization.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::resource::Resource;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

/// Per-patch blade budget, clamped at placement time. A placement /
/// performance cap, not a correctness invariant: the GPU path indexes
/// `blades[]` by per-view-band base offset + survivor ordinal, so blade
/// count is bounded by buffer capacity and the kernel's `slot < cap`
/// guard — not by any bit-packed index. Retunable together with
/// `MEADOW_MAX_BLADES_PER_PATCH` in `meadow_shared.wesl`, which MUST equal it.
pub const MAX_BLADES_PER_PATCH: u32 = 262_144;

/// Vertices per blade: 5 layers × 2 sides + 1 tip = 11.
pub const BLADE_VERTS_PER_BLADE: u32 = 11;

/// Indices per blade: 4 quad layers (12 indices) + 1 tip triangle (3) = 27.
pub const BLADE_INDICES_PER_BLADE: u32 = 27;

/// Arc length of a blade over its height. A blade bends along this length,
/// so no part of it lies farther than this multiple of its height from its
/// root (before wind sway). MUST equal `MEADOW_BLADE_LENGTH` in
/// `meadow_shared.wesl`.
pub const BLADE_LENGTH_PER_HEIGHT: f32 = 1.1;

/// Fanned arms per far-LOD tuft.
pub const TUFT_ARMS: u32 = 7;

/// Vertices per tuft: 7 arms × 3 (base-left, base-right, tip) = 21.
pub const TUFT_VERTS_PER_BLADE: u32 = TUFT_ARMS * 3;

/// Indices per tuft: 7 arms × one triangle = 21.
pub const TUFT_INDICES_PER_BLADE: u32 = TUFT_ARMS * 3;

/// Number of LOD bands the compute pass routes survivors into and the
/// raster side draws per view: band 0 = near blade, band 1 = far tuft.
/// Compile-time const — sizes the per-band `cursors` / `indirect`
/// arrays and the per-(view, band) base/cap bookkeeping. MUST equal
/// `MEADOW_MAX_BANDS` in `meadow_compute.wesl`.
pub const MEADOW_MAX_BANDS: usize = 2;

/// Bytes per `CompactedBladeRecord` (compute → raster). Must match the
/// `CompactedBladeRecord` std430 layout in `meadow_shared.wesl`.
pub const BLADE_RECORD_SIZE: u64 = 48;

/// Bytes per `DrawIndexedIndirectArgs` record (wgpu `DrawIndexedIndirect`:
/// 5 × u32). One record per (view, band) in the per-variant indirect buffer.
pub const INDIRECT_ARGS_SIZE: u64 = 20;

/// Grass farther than this casts no shadow-map shadow. Cascades reach
/// 140 m but grass shadows beyond ~50 m read as nothing. Caps each shadow
/// view's far radial clip (camera view depth,
/// `min(cascade_far, SHADOW_MAX_DIST)`) and ends the shadow density ramp
/// (planar player distance, `meadow_shared.wesl::meadow_shadow_ramp`).
/// MUST equal `MEADOW_SHADOW_MAX_DIST` in `meadow_shared.wesl`.
///
/// The ramp's start and far density live in `VariantParams.density.z/.w`
/// (`shadow_full_dist` / `shadow_far_density`) so they hot-reload without a
/// Rust rebuild; this const is also a Rust-side buffer-sizing input below.
pub const SHADOW_MAX_DIST: f32 = 50.0;

/// Max simultaneous cull views the compute pass writes: 1 main camera +
/// the directional cascades (currently 4) + 1 headroom slot. Sizes the
/// per-view regions of the blade + indirect + cursor buffers, so keep it
/// as tight as the cascade count allows (each extra slot costs a
/// `cap_shadow` region of VRAM). Excess views beyond this are dropped by
/// `build_meadow_view_slots`'s `slot >= MEADOW_MAX_VIEWS` guard.
/// MUST equal the array sizes hardcoded in `meadow_compute.wesl`
/// (`array<MeadowViewCull, N>` + the `write_instance_counts` workgroup size).
pub const MEADOW_MAX_VIEWS: usize = 6;

/// Mesh-shader path: blades scanned + derived per TASK workgroup (also
/// its workgroup size and the payload survivor capacity), and the
/// granularity of the CPU-built `(patch, slice)` task work list
/// (`compute.rs` builds it next to the active-patch list, feature-gated
/// there). The task stage owns the whole cull/derive/compact step at
/// full warp utilization; the mesh stage is a pure expander. MUST equal
/// `MESH_TASK_BLADES` in `meadow_mesh.wesl`. Ungated: `compute.rs`
/// (always compiled) references it.
pub const MESH_TASK_BLADES: u32 = 128;

/// Mesh-shader path: task-grid X stride. The flat task work list can
/// exceed wgpu's 65535 per-dimension dispatch cap (a large active set is
/// ~100k slices), so `draw_mesh_tasks` folds it over a fixed-stride 2D
/// grid: `x = min(n, STRIDE)`, `y = ceil(n / STRIDE)`, and the task
/// shader reconstructs `flat = wg.y * STRIDE + wg.x`, bounds-checked
/// against the count in the work-list header. MUST equal
/// `MESH_TASK_DISPATCH_STRIDE` in `meadow_mesh.wesl`.
#[cfg(feature = "mesh-shaders")]
pub const MESH_TASK_DISPATCH_STRIDE: u32 = 32_768;

/// Mesh-shader path: bytes per `SurvivorBlade` in the task payload
/// (12 × 4 B). Must match the struct in `meadow_mesh.wesl`; sizes the
/// runtime `max_task_payload_size` support check.
#[cfg(feature = "mesh-shaders")]
pub const MESH_SURVIVOR_BLADE_BYTES: u32 = 48;

/// Mesh-shader path: blades expanded per MESH workgroup (band 0).
/// 5 × 11 = 55 verts / 5 × 9 = 45 tris — small per-workgroup outputs are
/// load-bearing: large mesh outputs throttle workgroup launch and starve
/// the SMs (profiling showed 1.5/48 warp slots occupied at the previous
/// 253-vert budget). MUST equal `MESH_WG_BLADES` in `meadow_mesh.wesl`.
#[cfg(feature = "mesh-shaders")]
pub const MESH_WG_BLADES: u32 = 5;

/// Mesh-shader path: tufts expanded per MESH workgroup (band 1).
/// 3 × 21 = 63 verts / 21 tris. MUST equal `MESH_WG_TUFTS` in
/// `meadow_mesh.wesl`.
#[cfg(feature = "mesh-shaders")]
pub const MESH_WG_TUFTS: u32 = 3;

/// Mesh-shader path: blades expanded per MESH workgroup in SHADOW
/// cascades, where a blade is a single proxy triangle (base edge + tip —
/// template verts 0/1/10; drops the mid-blade bend, sub-texel at
/// shadow-map density): 21 × 3 = 63 verts / 21 tris, 9× less shadow
/// rasterization than the full ribbon. MUST equal
/// `MESH_WG_SHADOW_BLADES` in `meadow_mesh.wesl`.
#[cfg(feature = "mesh-shaders")]
pub const MESH_WG_SHADOW_BLADES: u32 = 21;

/// Mesh-shader path: largest per-workgroup mesh output across the bands
/// (band 1 tufts: 3 × 21 verts; band 0 blades: 5 × 9 tris). MUST equal
/// `MESH_OUT_VERTS` / `MESH_OUT_PRIMS` in `meadow_mesh.wesl`; sizes the
/// runtime output-limit support checks.
#[cfg(feature = "mesh-shaders")]
pub const MESH_OUT_VERTS: u32 = MESH_WG_TUFTS * TUFT_VERTS_PER_BLADE;
#[cfg(feature = "mesh-shaders")]
pub const MESH_OUT_PRIMS: u32 = MESH_WG_BLADES * 9;

pub(crate) const BLADE_INDICES: [u32; 27] = [
    0, 1, 3, 0, 3, 2, // layer 0 → 1
    2, 3, 5, 2, 5, 4, // layer 1 → 2
    4, 5, 7, 4, 7, 6, // layer 2 → 3
    6, 7, 9, 6, 9, 8, // layer 3 → 4
    8, 9, 10, // tip
];

/// Process-wide handles for the shared LOD meshes, indexed by band.
#[derive(Resource, Clone)]
pub struct MeadowMeshes {
    /// LOD templates: `[0]` = blade (11v/27i), `[1]` = tuft (21v/21i).
    /// Both share a byte-identical vertex-buffer layout; the band is
    /// carried in `POSITION.x` (0 = blade, 1 = tuft). The render driver
    /// entity carries `lod_meshes[0]` as its `Mesh3d` (for pipeline
    /// specialization / visibility); `DrawMeadowPatch` rebinds the
    /// per-band mesh slices when it issues each band's indirect draw.
    pub lod_meshes: [Handle<Mesh>; 2],
}

/// Build the shared single-blade template mesh.
///
/// Layout: 11 vertices, 27 indices forming a 5-layer ribbon with a
/// pointed tip. `POSITION = (0, vert_idx as f32, 0)` so the vertex
/// shader recovers `vert_idx` from `position.y`. NORMAL is upright
/// and UV.y carries the layer's height fraction (0..1) for the
/// `pbr_input` plumbing the deferred path expects; the shader
/// overwrites the world normal explicitly anyway.
///
/// Cost: 11 × 12 B POSITION + 11 × 12 B NORMAL + 11 × 8 B UV +
/// 11 × 16 B TANGENT + 11 × 16 B COLOR + 27 × 4 B indices ≈ 810 B per
/// process. Same mesh handle is reused
/// by every variant.
pub fn build_blade_template_mesh() -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(BLADE_VERTS_PER_BLADE as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(BLADE_VERTS_PER_BLADE as usize);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(BLADE_VERTS_PER_BLADE as usize);

    for vert_idx in 0..BLADE_VERTS_PER_BLADE {
        // POSITION.y carries `vert_idx` so the shader can recover it
        // without a separate `@builtin(vertex_index)` handshake; the
        // existing import paths assume POSITION is present.
        positions.push([0.0, vert_idx as f32, 0.0]);
        normals.push([0.0, 1.0, 0.0]);
        let layer = vert_idx.min(BLADE_VERTS_PER_BLADE - 1) as f32 / 10.0;
        uvs.push([0.0, layer]);
    }

    build_template_mesh(positions, normals, uvs, BLADE_INDICES.to_vec())
}

/// Assemble a meadow template mesh from per-vertex attribute arrays + an
/// index list, with a CONSTANT flat tangent (rather than
/// `generate_tangents()`): the shader hardcodes the world normal/tangent
/// and never reads mikktspace output, so a flat tangent is
/// rendering-invisible — and it keeps every LOD mesh's vertex buffer
/// byte-identical in layout (POSITION/NORMAL/UV/TANGENT/COLOR) so the
/// single specialized pipeline draws every band. Centralizes that
/// invariant. The white COLOR is never read: it turns on `VERTEX_COLORS`,
/// whose varying carries the per-blade albedo the vertex shader writes.
fn build_template_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Mesh {
    let tangents = vec![[1.0, 0.0, 0.0, 1.0]; positions.len()];
    let colors = vec![[1.0_f32, 1.0, 1.0, 1.0]; positions.len()];
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_indices(Indices::U32(indices))
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
}

/// Build the shared far-LOD tuft template mesh: 7 mini-blade arms
/// fanned around one root (21 verts / 21 indices, one triangle per arm).
///
/// Like the blade, `POSITION.y` carries `vert_idx` (0..20) and
/// `POSITION.x = 1.0` marks the tuft band so the shared vertex shader
/// expands it via `tuft_local_position` (in `meadow_shared.wesl`) instead
/// of `local_blade_position`. `vert_idx` demuxes to `(arm = idx/3,
/// corner = idx%3)`; corner 2 is the tip (UV.y = 1, for the shade ramp),
/// corners 0/1 the base. The fan's baked per-arm yaw means it reads as a
/// clump from any azimuth and straight down, with no camera-facing
/// rotation (hence no billboard popping). Layout matches the blade mesh
/// exactly (POSITION/NORMAL/UV/TANGENT/COLOR) so the same pipeline draws
/// it.
pub fn build_tuft_template_mesh() -> Mesh {
    let n = TUFT_VERTS_PER_BLADE as usize;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n);

    for vert_idx in 0..TUFT_VERTS_PER_BLADE {
        let corner = vert_idx % 3;
        positions.push([1.0, vert_idx as f32, 0.0]);
        normals.push([0.0, 1.0, 0.0]);
        // Tip (corner 2) → UV.y = 1 (lit/bright + wind-swept); base → 0.
        uvs.push([0.0, if corner == 2 { 1.0 } else { 0.0 }]);
    }

    let indices: Vec<u32> = (0..TUFT_INDICES_PER_BLADE).collect();
    build_template_mesh(positions, normals, uvs, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;
    use bevy::mesh::VertexAttributeValues;

    /// Both LOD meshes must carry the same vertex attributes so the single
    /// specialized pipeline draws either band without respecialization.
    #[test]
    fn lod_meshes_share_attribute_set() {
        for mesh in [build_blade_template_mesh(), build_tuft_template_mesh()] {
            assert!(mesh.attribute(Mesh::ATTRIBUTE_POSITION).is_some());
            assert!(mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
            assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
            assert!(mesh.attribute(Mesh::ATTRIBUTE_TANGENT).is_some());
            assert!(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
        }
    }

    #[test]
    fn lod_mesh_vertex_and_index_counts() {
        let blade = build_blade_template_mesh();
        assert_eq!(blade.count_vertices(), BLADE_VERTS_PER_BLADE as usize);
        assert_eq!(
            blade.indices().map(Indices::len),
            Some(BLADE_INDICES_PER_BLADE as usize)
        );
        let tuft = build_tuft_template_mesh();
        assert_eq!(tuft.count_vertices(), TUFT_VERTS_PER_BLADE as usize);
        assert_eq!(
            tuft.indices().map(Indices::len),
            Some(TUFT_INDICES_PER_BLADE as usize)
        );
    }

    /// POSITION.x carries the LOD band the VS branches on (0 blade, 1 tuft).
    #[test]
    fn position_x_marks_the_band() {
        let band_marker = |mesh: &Mesh| match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(v)) => v[0][0],
            _ => panic!("POSITION must be Float32x3"),
        };
        assert_eq!(band_marker(&build_blade_template_mesh()), 0.0);
        assert_eq!(band_marker(&build_tuft_template_mesh()), 1.0);
    }

    /// The shaders carry their own copies of the blade cap, the blade's arc
    /// length, the shadow distance, the tuft's area ratio and the mean blade
    /// shade.
    #[test]
    fn constants_match_the_shaders() {
        let shared = include_str!("meadow_shared.wesl");
        for line in [
            format!("const MEADOW_MAX_BLADES_PER_PATCH: u32 = {MAX_BLADES_PER_PATCH}u;"),
            format!("const MEADOW_BLADE_LENGTH: f32 = {BLADE_LENGTH_PER_HEIGHT:?};"),
            format!("const MEADOW_SHADOW_MAX_DIST: f32 = {SHADOW_MAX_DIST:?};"),
            format!(
                "const MEADOW_TUFT_BLADE_AREA: f32 = {:?};",
                crate::plugin::TUFT_BLADE_AREA_RATIO
            ),
            format!(
                "const MEADOW_MEAN_BLADE_SHADE: f32 = {:?};",
                crate::plugin::MEAN_BLADE_SHADE
            ),
        ] {
            assert!(
                shared.contains(&line),
                "`{line}` missing from meadow_shared.wesl"
            );
        }
    }

    /// CPU twin of `meadow_shared.wesl::meadow_leaf_phases`: the leaf phase
    /// along `v` and along `-v`.
    fn leaf_phases(v: Vec3, l: Vec3) -> (f32, f32) {
        let c = v.dot(l).clamp(-1.0, 1.0);
        let (b, s) = (c.acos(), (1.0 - c * c).sqrt());
        let scale = 1.5 * std::f32::consts::PI;
        (
            ((std::f32::consts::PI - b) * c + s) / scale,
            (s - b * c) / scale,
        )
    }

    /// The leaf phase along `v` ([`leaf_phases`]).
    fn leaf_phase(v: Vec3, l: Vec3) -> f32 {
        leaf_phases(v, l).0
    }

    /// CPU twin of `meadow_shared.wesl::meadow_canopy_normal`.
    fn canopy_normal(up: Vec3, v: Vec3, l: Vec3, light: f32) -> Vec3 {
        let target = light.clamp(0.0, 1.0);
        let target_sin = (1.0 - target * target).sqrt();
        let l_dot_v = l.dot(v);
        let v_across = v - l * l_dot_v;
        let v_across_length = v_across.length();
        let mut toward = up - l * up.dot(l);
        let mut toward_length = toward.length();
        if toward_length < 1e-4 {
            (toward, toward_length) = (v_across, v_across_length);
        }
        if toward_length < 1e-4 {
            return up;
        }
        let side = toward / toward_length;
        let nearest = l * target + side * target_sin;
        if nearest.dot(v) >= VIEW_MARGIN || v_across_length < 1e-4 {
            return nearest;
        }
        let e_v = v_across / v_across_length;
        let e_w = l.cross(e_v);
        let reach = (VIEW_MARGIN - target * l_dot_v) / (target_sin * v_across_length).max(1e-6);
        let across = if reach <= 1.0 {
            let turn = if side.dot(e_w) < 0.0 { -1.0 } else { 1.0 };
            e_v * reach + e_w * (turn * (1.0 - reach * reach).max(0.0).sqrt())
        } else {
            e_v
        };
        l * target + across * target_sin
    }

    /// `meadow_shared.wesl::MEADOW_VIEW_MARGIN`.
    const VIEW_MARGIN: f32 = 0.1;

    /// `meadow_shared.wesl::MEADOW_CANOPY_MIN_COS`.
    const CANOPY_MIN_COS: f32 = 0.02;

    /// `meadow_shared.wesl::MEADOW_LUMA`.
    const LUMA: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

    /// CPU twin of `meadow_shared.wesl::meadow_canopy_lit`.
    fn canopy_lit(up: Vec3, v: Vec3, l: Vec3) -> f32 {
        let (mu_v, mu_l) = (up.dot(v).max(CANOPY_MIN_COS), up.dot(l).max(CANOPY_MIN_COS));
        mu_l / (mu_l + mu_v)
    }

    /// CPU twin of `meadow_shared.wesl::meadow_canopy_sky`.
    fn canopy_sky(up: Vec3, v: Vec3) -> f32 {
        let mu = up.dot(v).max(CANOPY_MIN_COS);
        0.5 + 0.5754 * mu - (0.5889 + 0.5436 * mu) * mu * (1.0 + 1.0 / mu).ln()
    }

    /// CPU twin of `meadow_shared.wesl::meadow_canopy_sky_open`.
    fn canopy_sky_open(up: Vec3, v: Vec3) -> f32 {
        0.5 + up.dot(v).max(0.0) / 3.0
    }

    /// CPU twin of `meadow_shared.wesl::meadow_canopy_bounce`.
    fn canopy_bounce(mu_l: f32, deep_lit: f32, back: f32) -> f32 {
        deep_lit * (0.033 + 0.031 * back) + mu_l * (0.042 + 0.213 * back)
    }

    /// The deep canopy's bounce for the sun `l` seen along `v`
    /// ([`canopy_bounce`]).
    fn deep_bounce(v: Vec3, l: Vec3) -> f32 {
        canopy_bounce(l.y, canopy_lit(Vec3::Y, v, l), leaf_phases(v, l).1)
    }

    /// The canopy normal's Lambert factor in
    /// `meadow_shared.wesl::meadow_canopy_shade`: the root of
    /// `t = light / 2 (1 + cos_l t + sin_l sqrt(1 - t^2))`, at most 1.
    fn canopy_split(light: f32, cos_l: f32) -> f32 {
        let sin_l = (1.0 - cos_l * cos_l).max(0.0).sqrt();
        let h = 0.5 * light;
        if h * (1.0 + cos_l) >= 1.0 {
            return 1.0;
        }
        let p = 1.0 - h * cos_l;
        let hs = h * sin_l;
        ((h * p + hs * (1.0 - light * cos_l).max(0.0).sqrt()) / (p * p + hs * hs)).min(1.0)
    }

    /// CPU twin of `meadow_shared.wesl::meadow_canopy_shade`: normal, albedo.
    fn canopy_shade(
        up: Vec3,
        v: Vec3,
        l: Vec3,
        lit: f32,
        deep_lit: f32,
        deep: f32,
        sun_over_sky: f32,
        albedo: Vec3,
    ) -> (Vec3, Vec3) {
        let mut sky = canopy_sky_open(up, v);
        if deep > 0.0 {
            sky += (canopy_sky(up, v) - sky) * deep;
        }
        let cos_l = up.dot(l);
        if cos_l <= 0.0 {
            return (up, albedo * sky);
        }
        let (phase, back) = leaf_phases(v, l);
        let bounce = if deep > 0.0 {
            canopy_bounce(cos_l, deep_lit, back) * deep
        } else {
            0.0
        };
        let sun = Vec3::splat(phase * lit) + albedo * (1.0 + albedo) * bounce;
        let sun_luma = (albedo * sun).dot(LUMA) / albedo.dot(LUMA).max(1e-4);
        let light = sun_luma / sky.max(1e-4);
        let normal = canopy_normal(up, v, l, canopy_split(light, cos_l));
        let lights = sun_over_sky * normal.dot(l).max(0.0) + sky_view(normal, up);
        (
            normal,
            albedo * (sun * sun_over_sky + Vec3::splat(sky)) / lights.max(1e-4),
        )
    }

    /// CPU twin of `meadow_shared.wesl::meadow_sky_view`.
    fn sky_view(n: Vec3, up: Vec3) -> f32 {
        0.5 + 0.5 * n.dot(up)
    }

    /// A direction `elevation` degrees above the horizon at `azimuth`.
    fn direction(elevation: f32, azimuth: f32) -> Vec3 {
        let (e, a) = (elevation.to_radians(), azimuth.to_radians());
        Vec3::new(e.cos() * a.sin(), e.sin(), e.cos() * a.cos())
    }

    /// The canopy shading lights each channel, under a sun and a uniform
    /// sky, by the leaf phase times the lit share (2/3 with the light behind
    /// the viewer, none looking into it) plus the neighbours' bounce in that
    /// channel's albedo, and the sky's share, splitting them exactly as the
    /// leaves do wherever its normal can; it is the upright normal with the
    /// light below the horizon. The shader keeps the twin's lines. The
    /// canopy normal faces the viewer by the view margin wherever its cone
    /// allows.
    #[test]
    fn canopy_normal_lights_like_the_leaves() {
        let shared = include_str!("meadow_shared.wesl");
        for needle in [
            "return vec2<f32>((PI - b) * c + s, s - b * c) / (1.5 * PI);",
            "let target_cos = clamp(light, 0.0, 1.0);",
            "sky = mix(sky, meadow_canopy_sky(up, v), deep);",
            "bounce = meadow_canopy_bounce(cos_l, deep_lit, phases.y) * deep;",
            "let sun = vec3<f32>(phases.x * lit) + albedo * (1.0 + albedo) * bounce;",
            "let sun_luma = dot(albedo * sun, MEADOW_LUMA) / max(dot(albedo, MEADOW_LUMA), 1e-4);",
            "if (h * (1.0 + cos_l) < 1.0) {",
            "t = min((h * p + hs * sqrt(max(1.0 - light * cos_l, 0.0))) / (p * p + hs * hs), 1.0);",
            "let normal = meadow_canopy_normal(up, v, l, t);",
            "let lights = sun_over_sky * max(dot(normal, l), 0.0) + meadow_sky_view(normal, up);",
            "return MeadowCanopyShade(normal, albedo * (sun_over_sky * sun + sky) / max(lights, 1e-4));",
            "return 0.5 + 0.5 * dot(n, up);",
            "let nearest = l * target_cos + side * target_sin;",
            "let turn = select(1.0, -1.0, dot(side, e_w) < 0.0) * sqrt(max(1.0 - reach * reach, 0.0));",
            "across = e_v * reach + e_w * turn;",
            "return l * target_cos + across * target_sin;",
            "const MEADOW_VIEW_MARGIN: f32 = 0.1;",
            "const MEADOW_CANOPY_MIN_COS: f32 = 0.02;",
            "return mu_l / (mu_l + mu_v);",
            "leaves_sun = sun * lit * (1.0 + dot(v, l)) * (0.045 - 0.0248 * h_up2 * h_up2);",
            "surface_sun = sun * n_dot_l * d * vis;",
            "return saturate((leaves_sun + 0.014) / max(surface_sun + surface_sky, 1e-6));",
        ] {
            assert!(shared.contains(needle), "`{needle}` missing");
        }
        let l = direction(20.0, 0.0);
        let (toward, back) = leaf_phases(l, l);
        assert!((toward - 2.0 / 3.0).abs() < 1e-5 && back.abs() < 1e-5);
        let (toward, back) = leaf_phases(-l, l);
        assert!(toward.abs() < 1e-5 && (back - 2.0 / 3.0).abs() < 1e-5);
        for sun in [1.0, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0] {
            for view in [1.0, 3.0, 5.0, 10.0, 30.0, 60.0, 80.0] {
                for azimuth in (0..12).map(|i| i as f32 * 30.0) {
                    let (l, v) = (direction(sun, 0.0), direction(view, azimuth));
                    assert!((leaf_phases(v, l).1 - leaf_phase(-v, l)).abs() < 1e-5);
                    for (lit, albedo, k, deep) in [
                        (0.3, Vec3::new(0.12, 0.10, 0.04), 0.5, 1.0),
                        (0.3, Vec3::new(0.12, 0.10, 0.04), 4.0, 0.5),
                        (0.7, Vec3::new(0.08, 0.11, 0.03), 20.0, 0.0),
                        (1.0, Vec3::new(0.05, 0.09, 0.02), 4.0, 1.0),
                        (1.0, Vec3::new(0.05, 0.09, 0.02), 20.0, 0.2),
                    ] {
                        let deep_lit = canopy_lit(Vec3::Y, v, l);
                        let (n, shaded) =
                            canopy_shade(Vec3::Y, v, l, lit, deep_lit, deep, k, albedo);
                        assert!((n.length() - 1.0).abs() < 1e-4, "{n}");
                        let target = n.dot(l).min(1.0);
                        let sky = canopy_sky_open(Vec3::Y, v)
                            + (canopy_sky(Vec3::Y, v) - canopy_sky_open(Vec3::Y, v)) * deep;
                        let bounce = deep_bounce(v, l) * deep;
                        let sun_light = leaf_phase(v, l) * lit + albedo * (1.0 + albedo) * bounce;
                        // Sun and sky together, in every channel.
                        let rendered = shaded * (k * n.dot(l).max(0.0) + sky_view(n, Vec3::Y));
                        let want = albedo * (sun_light * k + Vec3::splat(sky));
                        assert!(
                            (rendered - want).abs().max_element() < 1e-4 * want.max_element(),
                            "{sun} {view} {azimuth}: {n} {shaded}"
                        );
                        // Split between them as the leaves split them, wherever
                        // a normal of the cone does and faces the viewer.
                        let light = (albedo * sun_light).dot(LUMA) / albedo.dot(LUMA) / sky;
                        if light * (1.0 + l.y) < 2.0 && n.dot(v) > VIEW_MARGIN + 0.01 {
                            let split = n.dot(l) / sky_view(n, Vec3::Y);
                            assert!(
                                (split - light).abs() < 1e-3 * light + 1e-5,
                                "{sun} {view} {azimuth}: {split} vs {light}"
                            );
                        }
                        // It faces the viewer by the margin wherever any normal
                        // of its cone can.
                        let best = target * l.dot(v)
                            + (1.0 - target * target).sqrt() * (v - l * l.dot(v)).length();
                        if best >= VIEW_MARGIN + 1e-3 {
                            assert!(
                                n.dot(v) >= VIEW_MARGIN - 1e-3,
                                "{sun} {view} {azimuth}: {n}"
                            );
                        }
                    }
                }
            }
        }
        let albedo = Vec3::new(0.1, 0.12, 0.05);
        assert_eq!(
            canopy_shade(
                Vec3::Y,
                Vec3::X,
                direction(-5.0, 0.0),
                1.0,
                1.0,
                0.0,
                4.0,
                albedo
            ),
            (Vec3::Y, albedo * 0.5)
        );
    }

    /// The canopy normal's Lambert factor is the root of its sky-view
    /// balance, found by bisection, or 1 where the balance has none below 1.
    #[test]
    fn canopy_split_is_the_balance_root() {
        for e in 1..=360 {
            let cos_l = (f64::from(e) * 0.25).to_radians().sin();
            let sin_l = (1.0 - cos_l * cos_l).sqrt();
            for i in 0..=800 {
                let light = f64::from(i) * 0.005;
                let balance = |t: f64| {
                    0.5 * light * (1.0 + cos_l * t + sin_l * (1.0 - t * t).max(0.0).sqrt()) - t
                };
                let want = if balance(1.0) >= 0.0 {
                    1.0
                } else {
                    let (mut lo, mut hi) = (0.0, 1.0);
                    for _ in 0..60 {
                        let mid = 0.5 * (lo + hi);
                        if balance(mid) >= 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    0.5 * (lo + hi)
                };
                let t = canopy_split(light as f32, cos_l as f32);
                assert!(
                    (f64::from(t) - want).abs() < 1e-5,
                    "{light} {cos_l}: {t} vs {want}"
                );
            }
        }
    }

    /// The canopy's sky shares are the leaf phase of every sky direction
    /// over a uniform sky, times its lit share deep in the canopy or alone
    /// for open leaves, by quadrature. Its bounce is within 0.03 of a Monte
    /// Carlo of the deep canopy (spherical, opaque, two-sided Lambertian
    /// leaves of albedo 0.15, `G` 0.5, orders past the second divided out
    /// as `1 + albedo`), listed as sun elevation, view elevation, view
    /// azimuth from the sun and bounce.
    #[test]
    fn canopy_sky_and_bounce_match_the_leaves() {
        let shared = include_str!("meadow_shared.wesl");
        for needle in [
            "return 0.5 + 0.5754 * mu - (0.5889 + 0.5436 * mu) * mu * log_term;",
            "return 0.5 + max(dot(up, v), 0.0) / 3.0;",
            "return deep_lit * (0.033 + 0.031 * back) + mu_l * (0.042 + 0.213 * back);",
        ] {
            assert!(shared.contains(needle), "`{needle}` missing");
        }
        let n = 160;
        let d_theta = std::f32::consts::FRAC_PI_2 / n as f32;
        let d_phi = std::f32::consts::TAU / (2 * n) as f32;
        for view in [1.5_f32, 3.0, 6.0, 12.0, 20.0, 35.0, 50.0, 70.0, 90.0] {
            let v = direction(view, 0.0);
            let (mut deep, mut open) = (0.0, 0.0);
            for i in 0..n {
                let theta = (i as f32 + 0.5) * d_theta;
                for j in 0..2 * n {
                    let phi = (j as f32 + 0.5) * d_phi;
                    let s = Vec3::new(
                        theta.sin() * phi.cos(),
                        theta.cos(),
                        theta.sin() * phi.sin(),
                    );
                    let phase =
                        leaf_phase(v, s) * theta.sin() * d_theta * d_phi / std::f32::consts::PI;
                    open += phase;
                    deep += phase * s.y / (s.y + v.y);
                }
            }
            let open_twin = canopy_sky_open(Vec3::Y, v);
            assert!(
                (open - open_twin).abs() < 2e-3,
                "{view}: {open} vs {open_twin}"
            );
            let deep_twin = canopy_sky(Vec3::Y, v);
            assert!(
                (deep - deep_twin).abs() < 2e-3,
                "{view}: {deep} vs {deep_twin}"
            );
        }
        for (sun, view, azimuth, monte_carlo) in [
            (8.0, 2.0, 180.0, 0.069),
            (20.0, 2.0, 0.0, 0.034),
            (20.0, 2.0, 180.0, 0.108),
            (20.0, 15.0, 120.0, 0.075),
            (35.0, 6.0, 60.0, 0.072),
            (35.0, 65.0, 180.0, 0.056),
            (50.0, 2.0, 180.0, 0.146),
            (66.0, 6.0, 180.0, 0.142),
            (66.0, 35.0, 0.0, 0.056),
            (85.0, 65.0, 120.0, 0.051),
        ] {
            let b = deep_bounce(direction(view, azimuth), direction(sun, 0.0));
            assert!(
                (b - monte_carlo).abs() < 0.03,
                "{sun} {view} {azimuth}: {b} vs {monte_carlo}"
            );
        }
    }

    /// The blade template's normals (lean, bend, rounding, random yaw) with
    /// their blade width's weight.
    fn template_normals() -> Vec<(Vec3, f32)> {
        let taper = |t: f32| (1.0 - t) * (1.0 + 0.6 * t);
        let mut normals = Vec::new();
        for i in 0..40_000u32 {
            let h = |k: u32| crate::field::hash01_u32(i.wrapping_mul(7).wrapping_add(k));
            let (u_bend, u_root, t, s, yaw) = (
                h(0),
                h(1),
                h(2),
                2.0 * h(3) - 1.0,
                h(4) * std::f32::consts::TAU,
            );
            let a = 0.08 + 0.32 * u_root + (0.3 + 0.6 * u_bend + 0.9 * u_bend.powi(4)) * t;
            let n = Vec3::new(0.6 * s, -a.sin(), a.cos()).normalize();
            let (c, si) = (yaw.cos(), yaw.sin());
            normals.push((
                Vec3::new(c * n.x + si * n.z, n.y, -si * n.x + c * n.z),
                taper(t),
            ));
        }
        normals
    }

    /// bevy_pbr's `D_GGX(roughness, n.h) * V_SmithGGXCorrelated(...) * n.l`
    /// at the blades' perceptual roughness 0.85.
    fn ggx_lobe(n_dot_v: f32, n_dot_l: f32, n_dot_h: f32) -> f32 {
        let alpha = 0.85 * 0.85;
        let a2 = alpha * alpha;
        let k = alpha / (1.0 - n_dot_h * n_dot_h + n_dot_h * n_dot_h * a2);
        let d = k * k / std::f32::consts::PI;
        let lambda_v = n_dot_l * ((n_dot_v - a2 * n_dot_v) * n_dot_v + a2).sqrt();
        let lambda_l = n_dot_v * ((n_dot_l - a2 * n_dot_l) * n_dot_l + a2).sqrt();
        d * 0.5 / (lambda_v + lambda_l) * n_dot_l
    }

    /// The blade template's own GGX lobes (perceptual roughness 0.85), over
    /// the faces a viewer sees, reflect the sun within 0.006 of
    /// `(1 + v.l) (0.045 - 0.0248 (h.up)^4)` at views up to 12 degrees off
    /// the ground: the canopy sheen's leaves.
    #[test]
    fn blade_sheen_is_the_backscatter() {
        let normals = template_normals();
        for sun in [5.0, 20.0, 35.0, 50.0, 70.0, 85.0] {
            for view in [1.0, 3.0, 6.0, 12.0] {
                for azimuth in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0] {
                    let (l, v) = (direction(sun, 0.0), direction(view, azimuth));
                    let h = (v + l).normalize();
                    let (reflected, seen) =
                        normals
                            .iter()
                            .fold((0.0, 0.0), |(reflected, seen), &(n, w)| {
                                let face = if n.dot(v) > 0.0 { n } else { -n };
                                let (n_dot_v, n_dot_l) = (face.dot(v), face.dot(l));
                                let lobe = if n_dot_l > 0.0 {
                                    ggx_lobe(n_dot_v, n_dot_l, face.dot(h).max(0.0))
                                } else {
                                    0.0
                                };
                                (reflected + w * n_dot_v * lobe, seen + w * n_dot_v)
                            });
                    let fit = (1.0 + v.dot(l)) * (0.045 - 0.0248 * h.y.powi(4));
                    assert!(
                        (reflected / seen - fit).abs() < 0.006,
                        "{sun} {view} {azimuth}: {} vs {fit}",
                        reflected / seen
                    );
                }
            }
        }
    }

    /// The blade template's normals (lean, bend, rounding, each blade facing
    /// its viewer) light within 0.04 of the spherical leaf phase at the
    /// views far grass is seen at.
    #[test]
    fn blade_normals_follow_the_spherical_phase() {
        let normals = template_normals();
        for sun in [5.0, 20.0, 40.0] {
            for view in [1.0, 4.0, 10.0] {
                for azimuth in [0.0, 90.0, 180.0] {
                    let (l, v) = (direction(sun, 0.0), direction(view, azimuth));
                    let (lit, seen) = normals.iter().fold((0.0, 0.0), |(lit, seen), &(n, w)| {
                        (
                            lit + w * (n.dot(v) * n.dot(l)).max(0.0),
                            seen + w * n.dot(v).abs(),
                        )
                    });
                    let phase = leaf_phase(v, l);
                    assert!(
                        (lit / seen - phase).abs() < 0.04,
                        "{sun} {view} {azimuth}: {} vs {phase}",
                        lit / seen
                    );
                }
            }
        }
    }

    /// A blade holds `BLADE_AREA_PER_WIDTH_HEIGHT` and a tuft
    /// `TUFT_BLADE_AREA_RATIO` blades' area: the ribbon of
    /// `local_blade_position` (five layers, each segment a chord of the
    /// bent centreline) over the leans of `meadow_blade_lean`, against the
    /// arm triangles of `tuft_local_position`.
    #[test]
    fn tuft_area_ratio_matches_the_templates() {
        let taper = |t: f32| (1.0 - t) * (1.0 + 0.6 * t);
        let layers = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let leans = 1000;
        let mut blade = 0.0;
        for i in 0..leans {
            let u = (i as f32 + 0.5) / leans as f32;
            let bend = 0.3 + 0.6 * u + 0.9 * u.powi(4);
            for k in 0..5 {
                let dt = layers[k + 1] - layers[k];
                let chord = BLADE_LENGTH_PER_HEIGHT * 2.0 * (0.5 * bend * dt).sin() / bend;
                // Half widths at the segment's ends; the tip has none.
                let top = if k < 4 { taper(layers[k + 1]) } else { 0.0 };
                blade += 0.5 * (taper(layers[k]) + top) * chord / leans as f32;
            }
        }
        assert!(
            (blade - crate::plugin::BLADE_AREA_PER_WIDTH_HEIGHT).abs() < 1e-3,
            "{blade}"
        );
        let arm = 0.5 * 1.2 * (1.0_f32 + 0.45 * 0.45).sqrt();
        let ratio = TUFT_ARMS as f32 * arm / blade;
        assert!(
            (ratio - crate::plugin::TUFT_BLADE_AREA_RATIO).abs() < 0.01,
            "{ratio}"
        );
    }
}
