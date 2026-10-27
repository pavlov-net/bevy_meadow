//! GPU-driven patch-based grass rendering for Bevy. A compute pass
//! (`compute.rs` + `meadow_compute.wesl`) derives, culls, and compacts
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
//! - `PatchShape` — a patch is a noisy disc, or a world-aligned square
//!   tile whose blades grow from the cover field.
//! - `MeadowCoverField` + `CoverSurvival` + `MeadowGrid` — the world-level
//!   cover density / height / palette grid tile blades sample, and how its
//!   density turns into surviving blades.
//! - `MeadowPatch` — required-components struct; one per placed
//!   patch.
//! - `MeadowPalette` — one entry of a variant's palette table; each patch
//!   selects an entry, and its blades vary around it. `MeadowPalettes`
//!   holds the consumer-written current tables.
//! - `WindDirection` — world-level resource broadcast to every
//!   variant's material on change.
//! - `MeadowWindState` — world-level resource carrying wind dynamics
//!   (speed multiplier, gustiness, crest wavenumber). Broadcast to
//!   every variant on change.
//! - `solari::MeadowSolariPlugin` (`solari` feature) — blades as
//!   `bevy_solari` shadow casters, and ray origins that start on the blade
//!   for meadow pixels Solari lights.

pub mod compute;
pub mod field;
pub mod material;
pub mod mesh;
#[cfg(feature = "mesh-shaders")]
pub mod mesh_path;
pub mod placement;
pub mod plugin;
pub mod render;
mod rt_selection;
#[cfg(all(feature = "solari", not(target_family = "wasm")))]
pub mod solari;

#[cfg(feature = "mesh-shaders")]
pub use mesh_path::MeadowForceComputePath;

pub use compute::{
    MeadowRaytracingConfig, MeadowRtBuffers, MeadowRtDiagnostics, MeadowRtTelemetry,
    MeadowRtVariantStats, RtBandBuffers, RtVariantBuffers,
};
pub use field::{CoverSurvival, CoverTexel, MeadowCoverField, MeadowGrid};
pub use material::{
    CLEARING_PROFILE_TABLE_SIZE, ClearingProfileId, ClearingProfiles, FLAT_HEIGHT_CAP,
    MAX_DISC_EDGE_NOISE, MAX_MEADOW_PALETTES, MAX_TRUNK_DISCS_PER_PATCH, MeadowClearingProfile,
    MeadowExt, MeadowMaterial, NO_HEIGHT_CAP, PaletteParams, PatchData, PatchTrunkSlot, TrunkDisc,
    VariantParams,
};
pub use mesh::{
    BLADE_INDICES_PER_BLADE, BLADE_LENGTH_PER_HEIGHT, BLADE_VERTS_PER_BLADE, MAX_BLADES_PER_PATCH,
    MEADOW_MAX_BANDS, MeadowMeshes, TUFT_INDICES_PER_BLADE, TUFT_VERTS_PER_BLADE,
    build_blade_template_mesh, build_tuft_template_mesh,
};
pub use placement::{
    PatchOverride, PatchOverrideMode, PatchPlacement, PatchShape, PlacementParams,
    TreeDensityField, place_patches,
};
pub use plugin::{
    BLADE_AREA_PER_WIDTH_HEIGHT, MEAN_BLADE_SHADE, MeadowHeightfield, MeadowLodCurve,
    MeadowPalette, MeadowPalettes, MeadowPatch, MeadowPlugin, MeadowVariant, MeadowVariantId,
    MeadowVariantRegistry, MeadowViewer, MeadowWindState, PatchAudioTag, TUFT_BLADE_AREA_RATIO,
    TUFT_REFERENCE_BLADES_PER_M2, WindDirection, WindParams, palette_table,
    upload_clearing_profiles, upload_placements, upload_trunk_rows, upload_trunk_slots,
};

pub use plugin::{MeadowSolariGeometryNormals, MeadowWindFreeze};
