//! GPU-driven patch-based grass rendering for Bevy. A compute pass
//! (`compute.rs` + `meadow_compute.wgsl`) derives, culls, and compacts
//! every blade once per frame into per-view buffers from a hash of
//! `(blade_idx, patch.seed)`; the raster passes then issue one indirect
//! draw per view (`render.rs::DrawMeadowPatch`) with a passthrough
//! vertex shader. Draw count is `O(views)`, independent of patch count.
//! One material per registered variant; `MeadowPatch` entities exist
//! for gameplay + the streaming active-patch set but do not render —
//! a single per-variant render-driver entity owns the draws.
//!
//! See `README.md` for the architecture + runtime assumptions.
//!
//! Public surface:
//! - `MeadowPlugin` — register at app startup.
//! - `MeadowVariant` + `MeadowVariantRegistry` — declare one variant
//!   per biome from `enter_world`.
//! - `place_patches` + `TreeDensityField` + `PatchOverride` — pure
//!   CPU placement, deterministic from `(variant.seed, biome AABB,
//!   tree positions)`.
//! - `MeadowPatch` — required-components struct; one per placed
//!   patch.
//! - `WindDirection` — world-level resource broadcast to every
//!   variant's material on change.
//! - `MeadowWindState` — world-level resource carrying wind dynamics
//!   (speed multiplier, gustiness, crest wavenumber). Broadcast to
//!   every variant on change.

pub mod compute;
pub mod material;
pub mod mesh;
#[cfg(feature = "mesh-shaders")]
pub mod mesh_path;
pub mod placement;
pub mod plugin;
pub mod render;

#[cfg(feature = "mesh-shaders")]
pub use mesh_path::MeadowForceComputePath;

pub use compute::{MeadowRaytracingConfig, MeadowRtBuffers, RtBandBuffers, RtVariantBuffers};
pub use material::{
    MAX_TRUNK_DISCS_PER_PATCH, MeadowExt, MeadowMaterial, PatchData, PatchTrunkSlot, TrunkDisc,
    VariantParams,
};
pub use mesh::{
    BLADE_INDICES_PER_BLADE, BLADE_VERTS_PER_BLADE, MAX_BLADES_PER_PATCH, MEADOW_MAX_BANDS,
    MeadowMeshes, TUFT_INDICES_PER_BLADE, TUFT_VERTS_PER_BLADE, build_blade_template_mesh,
    build_tuft_template_mesh,
};
pub use placement::{
    PatchOverride, PatchOverrideMode, PatchPlacement, PlacementParams, TreeDensityField,
    place_patches,
};
pub use plugin::{
    MeadowHeightfield, MeadowLodCurve, MeadowPatch, MeadowPlugin, MeadowSeasonState, MeadowVariant,
    MeadowVariantId, MeadowVariantRegistry, MeadowViewer, MeadowWindState, PatchAudioTag,
    SeasonalPalette, WindDirection, WindParams, upload_placements, upload_trunk_slots,
};
