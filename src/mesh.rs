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
//! The compute kernel (`meadow_compute.wgsl`) culls + compacts blades
//! per (view, band) into `CompactedBladeRecord`s; `DrawMeadowPatch`
//! (see `crate::render`) issues one `draw_indexed_indirect` per
//! (view, band) and `@builtin(instance_index)` indexes the band's
//! region directly. The vertex shader recovers `vert_idx` from
//! `POSITION.y` and the band from `POSITION.x` (0 = blade, 1 = tuft),
//! then expands the local geometry. Both meshes carry an identical
//! vertex-buffer layout (POSITION/NORMAL/UV/TANGENT) so the single
//! specialized pipeline draws either without respecialization.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::resource::Resource;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

/// Per-patch blade budget, clamped at placement time. A placement /
/// performance cap, not a correctness invariant: the GPU path indexes
/// `blades[]` by per-view-band base offset + survivor ordinal, so blade
/// count is bounded by buffer capacity and the kernel's `slot < cap`
/// guard — not by any bit-packed index. Freely retunable.
pub const MAX_BLADES_PER_PATCH: u32 = 65_536;

/// Vertices per blade: 5 layers × 2 sides + 1 tip = 11.
pub const BLADE_VERTS_PER_BLADE: u32 = 11;

/// Indices per blade: 4 quad layers (12 indices) + 1 tip triangle (3) = 27.
pub const BLADE_INDICES_PER_BLADE: u32 = 27;

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
/// `MEADOW_MAX_BANDS` in `meadow_compute.wgsl`.
pub const MEADOW_MAX_BANDS: usize = 2;

/// Bytes per `CompactedBladeRecord` (compute → raster). Must match the
/// `CompactedBladeRecord` std430 layout in `meadow_shared.wgsl`.
pub const BLADE_RECORD_SIZE: u64 = 48;

/// Bytes per `DrawIndexedIndirectArgs` record (wgpu `DrawIndexedIndirect`:
/// 5 × u32). One record per (view, band) in the per-variant indirect buffer.
pub const INDIRECT_ARGS_SIZE: u64 = 20;

/// Grass farther than this (camera view-depth) casts no shadow. Cascades
/// reach 140 m but grass shadows beyond ~50 m read as nothing. Caps each
/// shadow view's far radial clip (`min(cascade_far, SHADOW_MAX_DIST)`).
///
/// The shadow *density ramp* (full near the camera → sparse far) lives in
/// `VariantParams.density.z/.w` (`shadow_full_dist` / `shadow_far_density`)
/// so it hot-reloads without a Rust rebuild; this const is the matching hard
/// cut, plus the Rust-side buffer-sizing input below.
pub const SHADOW_MAX_DIST: f32 = 50.0;

/// Max simultaneous cull views the compute pass writes: 1 main camera +
/// the directional cascades (currently 4) + 1 headroom slot. Sizes the
/// per-view regions of the blade + indirect + cursor buffers, so keep it
/// as tight as the cascade count allows (each extra slot costs a
/// `cap_shadow` region of VRAM). Excess views beyond this are dropped by
/// `build_meadow_view_slots`'s `slot >= MEADOW_MAX_VIEWS` guard.
/// MUST equal the array sizes hardcoded in `meadow_compute.wgsl`
/// (`array<MeadowViewCull, N>` + the `write_instance_counts` workgroup size).
pub const MEADOW_MAX_VIEWS: usize = 6;

/// Mesh-shader path: blades scanned + derived per TASK workgroup (also
/// its workgroup size and the payload survivor capacity), and the
/// granularity of the CPU-built `(patch, slice)` task work list
/// (`compute.rs` builds it next to the active-patch list, feature-gated
/// there). The task stage owns the whole cull/derive/compact step at
/// full warp utilization; the mesh stage is a pure expander. MUST equal
/// `MESH_TASK_BLADES` in `meadow_mesh.wgsl`. Ungated: `compute.rs`
/// (always compiled) references it.
pub const MESH_TASK_BLADES: u32 = 128;

/// Mesh-shader path: task-grid X stride. The flat task work list can
/// exceed wgpu's 65535 per-dimension dispatch cap (a large active set is
/// ~100k slices), so `draw_mesh_tasks` folds it over a fixed-stride 2D
/// grid: `x = min(n, STRIDE)`, `y = ceil(n / STRIDE)`, and the task
/// shader reconstructs `flat = wg.y * STRIDE + wg.x`, bounds-checked
/// against the count in the work-list header. MUST equal
/// `MESH_TASK_DISPATCH_STRIDE` in `meadow_mesh.wgsl`.
#[cfg(feature = "mesh-shaders")]
pub const MESH_TASK_DISPATCH_STRIDE: u32 = 32_768;

/// Mesh-shader path: bytes per `SurvivorBlade` in the task payload
/// (12 × f32). Must match the struct in `meadow_mesh.wgsl`; sizes the
/// runtime `max_task_payload_size` support check.
#[cfg(feature = "mesh-shaders")]
pub const MESH_SURVIVOR_BLADE_BYTES: u32 = 48;

/// Mesh-shader path: blades expanded per MESH workgroup (band 0).
/// 5 × 11 = 55 verts / 5 × 9 = 45 tris — small per-workgroup outputs are
/// load-bearing: large mesh outputs throttle workgroup launch and starve
/// the SMs (profiling showed 1.5/48 warp slots occupied at the previous
/// 253-vert budget). MUST equal `MESH_WG_BLADES` in `meadow_mesh.wgsl`.
#[cfg(feature = "mesh-shaders")]
pub const MESH_WG_BLADES: u32 = 5;

/// Mesh-shader path: tufts expanded per MESH workgroup (band 1).
/// 3 × 21 = 63 verts / 21 tris. MUST equal `MESH_WG_TUFTS` in
/// `meadow_mesh.wgsl`.
#[cfg(feature = "mesh-shaders")]
pub const MESH_WG_TUFTS: u32 = 3;

/// Mesh-shader path: blades expanded per MESH workgroup in SHADOW
/// cascades, where a blade is a single proxy triangle (base edge + tip —
/// template verts 0/1/10; drops the mid-blade curl bow, sub-texel at
/// shadow-map density): 21 × 3 = 63 verts / 21 tris, 9× less shadow
/// rasterization than the full ribbon. MUST equal
/// `MESH_WG_SHADOW_BLADES` in `meadow_mesh.wgsl`.
#[cfg(feature = "mesh-shaders")]
pub const MESH_WG_SHADOW_BLADES: u32 = 21;

/// Mesh-shader path: largest per-workgroup mesh output across the bands
/// (band 1 tufts: 3 × 21 verts; band 0 blades: 5 × 9 tris). MUST equal
/// `MESH_OUT_VERTS` / `MESH_OUT_PRIMS` in `meadow_mesh.wgsl`; sizes the
/// runtime output-limit support checks.
#[cfg(feature = "mesh-shaders")]
pub const MESH_OUT_VERTS: u32 = MESH_WG_TUFTS * TUFT_VERTS_PER_BLADE;
#[cfg(feature = "mesh-shaders")]
pub const MESH_OUT_PRIMS: u32 = MESH_WG_BLADES * 9;

const BLADE_INDICES: [u32; 27] = [
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
/// 27 × 4 B indices ≈ 460 B per process. Same mesh handle is reused
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
/// byte-identical in layout (POSITION/NORMAL/UV/TANGENT) so the single
/// specialized pipeline draws every band. Centralizes that invariant.
fn build_template_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Mesh {
    let tangents = vec![[1.0, 0.0, 0.0, 1.0]; positions.len()];
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_indices(Indices::U32(indices))
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
}

/// Build the shared far-LOD tuft template mesh: 7 mini-blade arms
/// fanned around one root (21 verts / 21 indices, one triangle per arm).
///
/// Like the blade, `POSITION.y` carries `vert_idx` (0..20) and
/// `POSITION.x = 1.0` marks the tuft band so the shared vertex shader
/// expands it via `tuft_local_position` (in `meadow_shared.wgsl`) instead
/// of `local_blade_position`. `vert_idx` demuxes to `(arm = idx/3,
/// corner = idx%3)`; corner 2 is the tip (UV.y = 1, for the shade ramp),
/// corners 0/1 the base. The fan's baked per-arm yaw means it reads as a
/// clump from any azimuth and straight down, with no camera-facing
/// rotation (hence no billboard popping). Layout matches the blade mesh
/// exactly (POSITION/NORMAL/UV/TANGENT) so the same pipeline draws it.
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
}
