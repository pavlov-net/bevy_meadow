//! `MeadowMaterial` — `ExtendedMaterial<StandardMaterial, MeadowExt>`,
//! one asset per registered variant. Carries the per-variant scalar
//! params (wind, palette, LOD curve), the dynamic-sized `PatchData`
//! and `PatchTrunkSlot` storage buffers, and the world heightfield
//! atlas the vertex shader samples per blade.
//!
//! Going through `MaterialExtension` instead of a pure custom
//! pipeline lets the fragment write via
//! `pbr_input_from_standard_material` + `deferred_output`, picking up
//! PBR, shadow receiving, and the engine's GI integrations for free.

use bevy::asset::Asset;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use crate::compute::meadow_draw_bind_group_layout;

pub(crate) const MEADOW_SHADER: &str = "embedded://bevy_meadow/meadow.wgsl";

/// Maximum trunk discs collapsed per patch. The plan estimates ~10
/// per patch given typical tree spacing; 16 leaves headroom. This is
/// per-slot, not a global cap — each patch's slot is fixed-size, but
/// the variant's `trunk_slots` storage buffer holds an arbitrary
/// number of slots (one per patch).
pub const MAX_TRUNK_DISCS_PER_PATCH: usize = 16;

pub type MeadowMaterial = ExtendedMaterial<StandardMaterial, MeadowExt>;

/// Per-variant scalar parameters. Bound at uniform slot 100. All the
/// per-frame churn (wind direction, season palette lerp) lands here
/// and only here, so per-frame uploads of a variant material are a
/// single small uniform write rather than a full bind-group rebuild.
#[derive(ShaderType, Debug, Clone, Copy)]
pub struct VariantParams {
    /// Min/max blade height in metres, packed `(min, max, _, _)`.
    pub height_range: Vec4,
    /// Min/max blade width in metres, packed `(min, max, _, _)`.
    pub width_range: Vec4,
    /// `x = amplitude` (peak XZ displacement of a blade tip in
    /// metres), `y = period` (seconds per gust cycle),
    /// `z = current_time`, `w = previous_time` (for motion-vector
    /// TAA; both in `globals`' wrapped seconds). The time fields are
    /// stamped render-side by `prepare_meadow_variant_params` for the
    /// compute / mesh / RT kernels; the raster VS reads the same clock
    /// straight from the `globals` binding. They stay zero on the
    /// asset — a per-frame asset write would fire
    /// `AssetEvent::Modified` and re-extract every variant material.
    pub wind: Vec4,
    /// Globally-shared wind direction, normalised. The
    /// `WindDirection` resource broadcasts to every variant on
    /// resource change.
    pub wind_direction: Vec4,
    /// Wind dynamics packed `(speed_mul, gustiness, crest_wavenumber, _)`.
    /// Broadcast from the `MeadowWindState` resource; see that type's
    /// doc for per-field semantics.
    pub wind_state: Vec4,
    /// Spring / summer / autumn / winter base albedos (linear),
    /// each packed as `(r, g, b, _)`. The shader lerps between
    /// adjacent corners via `season_blend`.
    pub palette_spring: Vec4,
    pub palette_summer: Vec4,
    pub palette_autumn: Vec4,
    pub palette_winter: Vec4,
    /// `xy = (current_index_a, current_index_b)`, `z =
    /// blend_t in [0, 1]` between them. Written by the season
    /// system on `Changed<WorldClock>`.
    pub season_blend: Vec4,
    /// `x = full_distance` (every blade survives at this range or
    /// closer), `y = max_view_distance` (no blade survives at this
    /// range or beyond — `VisibilityRange` then culls the patch
    /// entity itself in a small soft band), `z = rim_falloff_fraction`
    /// (smoothstep on the outer edge of the noisy patch radius),
    /// `w = unused`. The shader's per-blade hash gate fades survival
    /// probability linearly across `full..max_view`.
    pub lod: Vec4,
    /// `x = blades_per_m²`, `y = patch_edge_noise_amp`,
    /// `z = shadow_full_dist` (shadow-caster density is full within this
    /// player distance), `w = shadow_far_density` (caster density at the
    /// shadow cull, ramped down from 1.0). The shadow density ramp lives
    /// here so it hot-reloads without a Rust rebuild.
    pub density: Vec4,
    /// World-space extent of the bound `heightfield` texture: `xy =
    /// world_min_xz`, `zw = world_max_xz`. The shader maps a blade's
    /// world XZ into texel coords via `(world_xz - xy) / (zw - xy) *
    /// dims`. Set by the consumer's atlas builder via
    /// `MeadowHeightfield::set` on world enter.
    pub heightfield_extents: Vec4,
    /// Player camera world position, packed `(x, z, _, _)`. Broadcast
    /// from `MeadowViewer` every frame. The shader's per-blade
    /// distance LOD gate reads this rather than `view.world_position`,
    /// because `view.world_position` is the projection origin of the
    /// current view — for shadow cascade passes that's the
    /// directional light, not the player, so reading it there causes
    /// every blade to survive the hash gate out to the cascade limit
    /// (~12× the visible blade count, dominating cascade depth time).
    /// A uniform decouples the LOD math from which view is rendering.
    pub viewer_world_xz: Vec4,
    /// Far-LOD tuft band, packed `(tuft_start, density_near, density_far,
    /// height_fade_start)`. Patches whose player distance exceeds
    /// `tuft_start` render as sparse fanned tufts (band 1) instead of
    /// blades; per-tuft survival ramps `density_near → density_far` across
    /// `tuft_start..max_view_distance` (`lod.y`), and tuft height fades to
    /// zero over `height_fade_start..max_view_distance` so the band
    /// dissolves into the ground rather than ending at a hard edge.
    pub tuft: Vec4,
    /// Main camera view-depth plane, packed `(forward.xyz, -dot(forward,
    /// eye))`. A world point `P`'s signed distance from the camera along its
    /// forward axis is `dot(forward, P) + w`. Shadow cascade assignment uses
    /// this (broadcast from `MeadowViewer`, like `viewer_world_xz`) so a
    /// caster lands in the same cascade the receiver under it samples — Bevy
    /// picks a receiver's cascade by view-space depth, so clipping casters by
    /// planar distance instead makes the two drift apart as the camera
    /// orbits and pops near shadows on and off.
    pub viewer_forward: Vec4,
}

impl Default for VariantParams {
    fn default() -> Self {
        Self {
            height_range: Vec4::new(0.18, 0.38, 0.0, 0.0),
            width_range: Vec4::new(0.04, 0.07, 0.0, 0.0),
            wind: Vec4::new(0.18, 4.5, 0.0, 0.0),
            wind_direction: Vec4::new(1.0, 0.0, 0.0, 0.0),
            wind_state: Vec4::new(1.0, 0.6, 0.6, 0.0),
            palette_spring: Vec4::new(0.16, 0.30, 0.09, 0.0),
            palette_summer: Vec4::new(0.18, 0.32, 0.11, 0.0),
            palette_autumn: Vec4::new(0.36, 0.28, 0.10, 0.0),
            palette_winter: Vec4::new(0.55, 0.55, 0.55, 0.0),
            season_blend: Vec4::new(0.0, 1.0, 0.0, 0.0),
            lod: Vec4::new(40.0, 110.0, 0.25, 0.0),
            density: Vec4::new(150.0, 0.18, 18.0, 0.25),
            // Default to a 1×1 metre extent at the origin — degenerate
            // but well-formed; the shader's bilerp short-circuits when
            // the bound heightfield is the 1×1 fallback so blades
            // ride at Y=0 until the consumer sets a real atlas.
            heightfield_extents: Vec4::new(0.0, 0.0, 1.0, 1.0),
            // Origin — overwritten every frame by `broadcast_viewer`
            // from the consumer-written `MeadowViewer` resource.
            viewer_world_xz: Vec4::ZERO,
            // (tuft_start, density_near, density_far, height_fade_start).
            tuft: Vec4::new(55.0, 0.12, 0.02, 98.0),
            // Overwritten every frame by `broadcast_viewer`; ZERO until then
            // means view_depth = 0, so no grass shadows render pre-camera.
            viewer_forward: Vec4::ZERO,
        }
    }
}

/// Per-patch row in the variant's `patches` uniform array, indexed
/// by the patch entity's `MeshTag`. Packed for cheap shader fetch:
/// the WGSL side reads one `vec4` for centre/radius/seed and a
/// second `vec4` for blade_count/edge_noise_amp/canopy_density/flags.
#[derive(ShaderType, Debug, Clone, Copy, Default)]
pub struct PatchData {
    /// `xy = world centre`, `z = radius (metres)`, `w = seed (f32 from u32 bits)`.
    pub centre_xz_radius_seed: Vec4,
    /// `x = blade_count (f32)`, `y = edge_noise_amp`,
    /// `z = canopy_density_at_centre`, `w = flags (bit 0 = audio_enabled)`.
    pub blade_count_noise_canopy_flags: Vec4,
}

/// One trunk disc collapsed in this patch's local trunk list.
/// `xy = world centre`, `z = solid radius`, `w = fade band`.
#[derive(ShaderType, Debug, Clone, Copy, Default)]
pub struct TrunkDisc {
    pub center_radius_fade: Vec4,
}

/// Per-patch slot of trunk discs. Padded so the slot is
/// `std140`-aligned to 16 B; the explicit padding keeps the layout
/// stable across any future field reordering.
#[derive(ShaderType, Debug, Clone, Copy)]
pub struct PatchTrunkSlot {
    pub count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub discs: [TrunkDisc; MAX_TRUNK_DISCS_PER_PATCH],
}

impl Default for PatchTrunkSlot {
    fn default() -> Self {
        Self {
            count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            discs: [TrunkDisc::default(); MAX_TRUNK_DISCS_PER_PATCH],
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Default, Debug, Clone)]
pub struct MeadowExt {
    #[uniform(100)]
    pub variant_params: VariantParams,
    /// Per-patch rows. Updated via `upload_placements` after
    /// `place_patches` returns.
    #[storage(101, read_only)]
    pub patches: Handle<ShaderBuffer>,
    /// Per-patch trunk-disc slot rows. Same dynamic-sized story as
    /// `patches`. The shader indexes `trunk_slots[patch_idx]` to
    /// fade blades inside any tree's trunk disc.
    #[storage(102, read_only)]
    pub trunk_slots: Handle<ShaderBuffer>,
    /// World-XZ → ground Y heightfield atlas, R32Float. Sampled
    /// per-blade in the vertex shader to plant blades at the actual
    /// terrain height; replaces the per-patch tilted-plane
    /// approximation that floated grass on slopes. The texture is
    /// declared `unfilterable_float` (R32Float can't be linearly
    /// sampled without the `float32-filterable` wgpu feature) and
    /// the shader does manual bilinear via `textureLoad`. Set by
    /// the consumer's atlas builder via `MeadowHeightfield`; until
    /// then, Bevy's `FallbackImage` provides a 1×1 zeros texture
    /// so the bind group is well-formed and blades ride at Y=0.
    #[texture(105, sample_type = "float", filterable = false)]
    pub heightfield: Handle<Image>,
}

impl MaterialExtension for MeadowExt {
    fn vertex_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }
    fn deferred_vertex_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }
    fn fragment_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        MEADOW_SHADER.into()
    }

    /// Inject the group-4 bind-group layout (the per-variant compacted
    /// blade buffer the passthrough VS reads) into every pipeline
    /// permutation (main / prepass / deferred / shadow). The draw bind
    /// group is built from the *same* descriptor in
    /// `compute::prepare_meadow_gpu_buffers` (the realloc branch), so
    /// layout and bind group stay compatible.
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.layout.push(meadow_draw_bind_group_layout());
        Ok(())
    }
}
