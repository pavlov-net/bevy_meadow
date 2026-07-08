# bevy_meadow

GPU-driven, patch-based grass for Bevy. A compute pass derives, culls, and
LODs every blade on the GPU each frame, then compacts the survivors into
per-view buffers. Rendering is a handful of indirect draws per frame,
independent of how many patches or blades exist.

Patches are placed on the CPU and spawned as ECS entities so gameplay can see
them (audio, spatial queries, streaming), but they do not render individually
-- one per-variant render driver owns every draw. You author one material per
biome variant and drive appearance at runtime by writing a few world
resources.

## Features

- **GPU-driven blades.** Per-frame cull, LOD, and compaction on the GPU; draw
  count scales with views, not patch or blade count.
- **Two-band LOD.** Individual blades up close; sparse fanned tufts in the
  distance that fade out with range. No billboard popping (tuft geometry is
  baked, never camera-facing).
- **Seasons.** A four-corner palette (spring/summer/autumn/winter) blended live
  from a world season state.
- **Wind.** A shared gust direction (agrees with other foliage) plus dynamics
  -- speed, gustiness, and traveling gust crests -- scaled per variant.
- **Terrain-following.** Blades sample a world heightfield atlas so grass sits
  on the ground, not a flat plane.
- **Tree-aware placement.** Patches bias toward clearings and away from canopy,
  with authored force/suppress overrides. Grass parts around tree trunks.
- **Streaming-correct.** Grass is generated only over loaded chunks; buffers
  don't thrash as chunks cross in and out.
- **PBR for free.** Built on `ExtendedMaterial<StandardMaterial, _>`, so blades
  get PBR shading, shadow receiving, deferred, and GI from the engine.
- **Optional mesh-shader path.** A task/mesh-pipeline renderer that runs about
  2x faster where the GPU supports it, behind the `mesh-shaders` feature. The
  compute path is the automatic fallback.

## Installation

`bevy_meadow` tracks Bevy's `main`, so it depends on Bevy by git and is not
published to crates.io. Add both as git dependencies:

```toml
[dependencies]
bevy = { git = "https://github.com/bevyengine/bevy" }
bevy_meadow = { git = "https://github.com/pavlov-net/bevy_meadow" }
```

Edition 2024, MSRV 1.89, MIT. The `mesh-shaders` feature is off by default; see
[Mesh-shader path](#mesh-shader-path).

## Quick start

Add the plugin, then, once per biome: author and register a variant, place
patches, upload them to the GPU, and spawn one `MeadowPatch` per placement.

```rust
use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;
use bevy_meadow::{
    MeadowPlugin, MeadowVariant, MeadowMaterial, MeadowVariantRegistry, MeadowPatch,
    PlacementParams, TreeDensityField, place_patches, upload_placements,
};

app.add_plugins(MeadowPlugin);

fn setup_biome(
    mut registry: ResMut<MeadowVariantRegistry>,
    mut materials: ResMut<Assets<MeadowMaterial>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut commands: Commands,
) {
    // 1. Author a variant (one per biome) and register it. Defaults target a
    //    dense temperate meadow; override what you care about.
    let variant = MeadowVariant {
        blades_per_m2: 200.0,
        height_range: (0.2, 0.45),
        seed: 0xC0FFEE,
        ..default()
    };
    let id = registry.register(variant, &mut materials, &mut buffers);

    // 2. Place patches on the CPU. Deterministic from (seed, rect, trees).
    let biome_rect = Rect::from_center_size(Vec2::ZERO, Vec2::splat(500.0));
    let tree_density = TreeDensityField::new(biome_rect, tree_positions());
    let params = PlacementParams {
        biome_rect,
        variant_seed: 0xC0FFEE,
        patches_per_m2: 0.0008,
        patch_radius_range: (8.0, 20.0),
        min_patch_spacing: 4.0,
        patch_edge_noise_amp: 0.18,
        canopy_threshold: 0.6,
        canopy_soft_threshold: 0.35,
        biome_weight_threshold: 0.6,
        rim_falloff_fraction: 0.25,
        blades_per_m2: 200.0,
    };
    // The last argument answers "how strongly does this biome own point p?"
    // (splatmap weight) so patches don't straddle a biome seam.
    let placements = place_patches(&params, &[], &tree_density, |_p| 1.0);

    // 3. Upload placements into the variant's GPU buffers. Nothing renders
    //    until this runs.
    upload_placements(id, &placements, &mut registry, &materials, &mut buffers);

    // 4. Spawn a MeadowPatch per placement that lands in a loaded chunk.
    //    Required components (Transform, Aabb, VisibilityRange, ...) auto-add.
    for (i, p) in placements.iter().enumerate() {
        commands.spawn(MeadowPatch {
            variant: id,
            patch_index: i as u32,
            blade_count: p.blade_count,
            centre: p.centre,
            radius: p.radius,
        });
    }
}
```

`MeadowPatch` carries no `Mesh3d`: the patch entity exists for gameplay,
audio, and the streaming active set, while the per-variant render driver issues
the actual draws. Size the entity's `Aabb` / `VisibilityRange` from
`p.outer_radius()` (the noisy-blob cull radius).

## Driving it at runtime

The plugin follows one rule: **write a world resource, and the plugin fans the
change out to every registered variant.** Most resources broadcast only on
change; `MeadowViewer` is read every frame.

```rust
use bevy_meadow::{MeadowViewer, WindDirection, MeadowWindState, MeadowSeasonState};

// Required every frame: the LOD/density pivot and shadow-cascade reference.
// Without it, LOD has no camera to measure distance from.
fn drive_meadow_viewer(
    mut viewer: ResMut<MeadowViewer>,
    camera: Single<&GlobalTransform, With<Camera3d>>,
) {
    let eye = camera.translation();
    let fwd = camera.forward().as_vec3();
    viewer.eye_xz = eye.xz();
    // (forward.xyz, -dot(forward, eye)) -- the camera view-depth plane.
    viewer.depth_plane = fwd.extend(-fwd.dot(eye));
}
```

`MeadowViewer` is deliberately decoupled from whichever view is rendering:
shadow-cascade passes render from the light, so LOD reads the player position
from here instead of the render view.

The other driver resources:

- **`WindDirection(Vec2)`** and **`MeadowWindState`** (speed, gustiness, crest
  wavenumber) -- world wind, shared so meadow and tree foliage gust together.
- **`MeadowSeasonState`** (`from_idx`, `to_idx`, `blend_t`) -- blends between
  two of the variant's four seasonal palette corners. Drive it from your own
  clock.
- **`MeadowHeightfield`** (`atlas`, `world_min`, `world_max`) -- an `R32Float`
  world-XZ -> ground-Y atlas, set on world enter. Until you set one, blades sit
  at Y=0.

## Authoring a variant

A `MeadowVariant` is the per-biome content knobset. It has a `Default` tuned for
a dense temperate meadow; override fields as needed. The groups:

- **Density and size** -- `patches_per_m2`, `blades_per_m2`, `patch_radius_range`,
  `min_patch_spacing`, `height_range`, `width_range`, `patch_edge_noise_amp`.
- **Canopy avoidance** -- `canopy_threshold` (hard reject) and
  `canopy_soft_threshold` (shrink), plus `biome_weight_threshold` for seams.
- **Wind** (`WindParams`) -- per-variant `amplitude` and `period`; direction and
  dynamics come from the shared resources.
- **Palette** (`SeasonalPalette`) -- the four seasonal albedo corners.
- **LOD** (`MeadowLodCurve`) -- see below.
- `seed`, `audio_tag`, and authored `overrides`.

### Levels of detail

`MeadowLodCurve` controls how grass thins with distance. Every blade survives
within `full_distance`; a per-blade hash gate fades survival to zero by
`max_view_distance`, after which `VisibilityRange` culls the patch entity.
Past `tuft_start`, patches switch from individual blades to sparse fanned
**tufts** whose density ramps `tuft_density_near -> tuft_density_far` and whose
height fades out by `height_fade_start`, so the far edge dissolves into the
ground instead of ending at a hard line. Shadow-caster density has its own ramp
(`shadow_full_dist`, `shadow_far_density`), full near the camera and sparse
beyond -- cheap far cascades, grounded contact shadows up close.

### Placement

`place_patches` is a pure, deterministic CPU pass: same `(variant_seed,
biome_rect, tree positions)` in, same `Vec<PatchPlacement>` out. It scatters
Poisson-disk-like candidates and rejects any that crowd a neighbor, sit under
canopy (via `TreeDensityField`, a coarse tree-density grid), or straddle a
biome seam (via your `biome_weight_at` closure sampled at the centre and eight
perimeter points).

`PatchOverride`s run first: `Force` guarantees a patch (a story clearing,
a set piece) and claims its footprint against procedural placement; `Suppress`
carves a hole (a cave mouth, a plaza).

### Trees and audio

Where trees stand in grass, upload per-patch **trunk discs** with
`upload_trunk_slots`; the shader fades blades inside each disc so grass doesn't
poke through a trunk. Each patch holds up to `MAX_TRUNK_DISCS_PER_PATCH` discs.

A variant can set an `audio_tag`; it rides onto each patch as `PatchAudioTag`,
so your audio system can swap footstep and ambience when the player enters the
grass.

## How it works

Optional reading -- skip it to use the crate.

### The frame

1. **Compute** (`meadow_compute.wgsl`). A render-graph node dispatches
   `cull_and_compact` over `(active patch x view)`. Per pair it frustum-rejects
   the patch, then for each blade derives placement from a hash, samples the
   heightfield once, applies the per-view LOD / density / shadow gates, and
   atomically appends survivors into that view's region of the `blades` buffer.
   Survivors route into two LOD **bands** (near blade, far tuft).
   `write_instance_counts` copies each region's survivor count into its indirect
   draw args.
2. **Raster** (`meadow.wgsl`). The render driver yields one phase item per view.
   `DrawMeadowPatch` picks the view's slot, binds the variant's blade buffer,
   and issues one `draw_indexed_indirect` **per (view, band)** -- so the main
   view draws two (blades + tufts) and each shadow cascade draws one (tufts
   don't cast). A passthrough vertex shader reads one compacted record per
   instance, expands the template (11 verts for a blade, 21 for a tuft, selected
   by a band marker), and recomputes only the wind sway. The fragment is the
   engine's PBR path, so shadows, deferred, and GI come for free.

Draw count is therefore a small constant per view, independent of patch and
blade count.

### Streaming and buffer sizing

The `patches` storage buffer holds *every* placement for a region, but the
compute pass iterates only the **active set** -- the `patch_index`es of live
`MeadowPatch` entities -- so grass is never generated over unloaded terrain.

Buffers are sized off an LOD-weighted estimate of what the *active* footprint
actually renders, not the full placement list (which runs to tens of millions
of blades and would blow wgpu's ~2 GB storage-binding limit). Three caps drive
the layout: near blades and far tufts for the main view, and a tighter cap for
shadow casters. Each is rounded up to a granularity and only ever grown, so
chunks crossing in and out don't trigger reallocation.

### Shadows

Grass shadows come from a single dominant directional light (the brightest
shadow-casting one). Each of its cascades clips grass to its own radial depth
slice, so a blade casts into roughly one cascade rather than all of them, and
`SHADOW_MAX_DIST` bounds the union. A second shadow-casting directional light
gets no grass shadows.

### Runtime assumptions

These hold for a typical single-player 3D camera but aren't enforced by types;
revisit them if the camera or view setup changes.

- **The main camera's render-world view carries a `Frustum`**, and it is the
  *first* non-light view with one -- that view becomes LOD slot 0. A second
  frustum'd non-light view (e.g. a 2D/UI camera) could be mis-assigned slot 0.
- **`@builtin(instance_index)` includes `first_instance`** under indirect draws.
  Each (view, band) region's base offset rides in via `first_instance`, and the
  vertex shader indexes `blades[]` directly. A backend that zero-bases
  `instance_index` would need a bound buffer slice instead.
- **The render driver reaches every view.** It carries a world-spanning `Aabb`
  and `NoFrustumCulling` so it lands in every cascade's visible set. Expect one
  meadow phase item per view; zero shadow items means the driver isn't reaching
  the cascades.
- **`MEADOW_MAX_VIEWS`** caps simultaneous cull views (main + cascades); raise
  it if cascade count grows.

### VRAM

Dominated by the per-variant `blades` buffer:
`(cap_main_near + cap_main_far + (MEADOW_MAX_VIEWS - 1) * cap_shadow) * 48 B`.
For a dense biome-scale variant that lands around 100-200 MB while active;
typically one variant is active at a time. Placement is static, so there's no
ping-pong buffer -- motion vectors come purely from the wind delta.

## Mesh-shader path

An alternative renderer on GPU task/mesh pipelines
(`mesh_path.rs` + `meadow_mesh.wgsl`), behind the `mesh-shaders` feature. Where
the GPU supports it, it runs about 2x faster than the compute path.

A 128-wide task stage does the whole cull / gate / derive / compact step over a
CPU-built work list and hands finished survivors straight to the mesh stage
through the task payload -- no `blades` buffer, no indirect plumbing. The mesh
stage is a pure expander with small outputs (large mesh outputs starve the SMs).
It serves the main camera and every directional shadow cascade, and drives the
engine's real per-view bind group, so PBR lighting and shadow receiving match
the compute path. Placement is hash-identical between the two, and `cargo test`
string-compares the shared derivation helpers and compiles the assembled mesh
module, so the two paths can't silently drift.

The path is opt-in and self-selecting:

- **Enabled** by the `mesh-shaders` feature (which pulls in `wgpu`, `naga_oil`,
  and `naga`) *and* a GPU that reports the mesh/task-shader capability with
  enough per-workgroup budget. Where either is missing, the compute path runs
  automatically with byte-identical results.
- **`MeadowForceComputePath`** (a main-world resource) forces the compute path
  for A/B comparison; wire it into a debug UI.
- **Fallbacks.** Deferred configs use the compute path. Motion vectors are
  written only in single-sample (DLSS/TAA) configs; under MSAA the main pass
  writes color and depth only.

## License

Licensed under the [MIT License](LICENSE). Copyright (c) 2026 Stuart Parmenter.
