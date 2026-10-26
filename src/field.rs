//! World-aligned grids the blade kernels sample, and the cover field tile
//! patches grow from.
//!
//! A [`MeadowGrid`] spans `world_min..world_max` with texel `(i, j)` at
//! `world_min + (world_max - world_min) * (i, j) / (dims - 1)`, so its edge
//! texels sit on the extent's edges and a bilinear sample between texels
//! matches the shaders' `pixel = uv * (dims - 1)` lookup exactly. The
//! resources that carry one ([`MeadowCoverField`],
//! [`MeadowHeightfield`](crate::MeadowHeightfield)) keep the grid on the CPU
//! beside its GPU atlas, so raytracing caster selection tests blades against
//! the data the kernels read.
//!
//! The cover field drives tile patches ([`PatchShape::Tile`](crate::PatchShape)):
//! every blade samples its density, height scale and palette index, and
//! survives where the density beats a threshold that mixes the blade's own
//! random with spatially correlated noise ([`CoverSurvival`]). The expected
//! share of surviving blades equals the density, at any clumpiness.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

/// A world-aligned grid of texels; see the module docs for its layout.
#[derive(Debug, Clone, PartialEq)]
pub struct MeadowGrid<T> {
    pub world_min: Vec2,
    pub world_max: Vec2,
    pub dims: UVec2,
    /// Row-major, `dims.x` texels per row, rows along +Z.
    pub texels: Vec<T>,
}

impl<T: Copy> MeadowGrid<T> {
    /// A grid of `texels` over `world_min..world_max`.
    ///
    /// # Panics
    ///
    /// If either dimension is below 2 or `texels` doesn't hold
    /// `dims.x * dims.y` texels.
    pub fn new(world_min: Vec2, world_max: Vec2, dims: UVec2, texels: Vec<T>) -> Self {
        assert!(dims.x >= 2 && dims.y >= 2, "a meadow grid is at least 2×2");
        assert_eq!(texels.len(), (dims.x * dims.y) as usize);
        Self {
            world_min,
            world_max,
            dims,
            texels,
        }
    }

    /// Texel `(x, y)`.
    pub fn texel(&self, x: u32, y: u32) -> T {
        self.texels[(y * self.dims.x + x) as usize]
    }

    /// The four texels around `world_xz` (`[x0y0, x1y0, x0y1, x1y1]`) and
    /// the position between them, edge-clamped like the shaders.
    fn footprint(&self, world_xz: Vec2) -> ([T; 4], Vec2) {
        let size = (self.world_max - self.world_min).max(Vec2::splat(1e-3));
        let uv = ((world_xz - self.world_min) / size).clamp(Vec2::ZERO, Vec2::ONE);
        let pixel = uv * (self.dims - UVec2::ONE).as_vec2();
        let p0 = pixel.floor().as_uvec2();
        let p1 = (p0 + UVec2::ONE).min(self.dims - UVec2::ONE);
        (
            [
                self.texel(p0.x, p0.y),
                self.texel(p1.x, p0.y),
                self.texel(p0.x, p1.y),
                self.texel(p1.x, p1.y),
            ],
            pixel - p0.as_vec2(),
        )
    }
}

impl MeadowGrid<f32> {
    /// Bilinear sample at `world_xz`.
    pub fn sample(&self, world_xz: Vec2) -> f32 {
        let (texels, t) = self.footprint(world_xz);
        lattice_blend(texels, t)
    }

    /// The grid as an `R32Float` atlas for the GPU.
    pub fn to_image(&self) -> Image {
        grid_image(
            self.dims,
            TextureFormat::R32Float,
            bytemuck::cast_slice(&self.texels),
        )
    }
}

/// One cover field texel, as the `Rgba32Float` atlas holds it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoverTexel {
    /// Share of the ground grass covers, in `[0, 1]`.
    pub density: f32,
    /// Factor on blade height.
    pub height_scale: f32,
    /// Continuous index into the variant's palette table: a blade takes one
    /// of the two entries around it, the upper with probability
    /// `fract(palette)`. Negative keeps the patch's own entry.
    pub palette: f32,
    /// How readily the ground browns off: the palette entry's dead-patch
    /// share scales by `2 * dryness`, clamped to `[0, 1]`, so 0 grows no
    /// dead patches, 0.5 the entry's own share and 1 twice it. Single dead
    /// blades keep the entry's share.
    pub dryness: f32,
}

impl CoverTexel {
    /// Dryness that keeps a palette entry's own dead-patch share. MUST equal
    /// `MEADOW_NEUTRAL_DRYNESS` in `meadow_shared.wesl`.
    pub const NEUTRAL_DRYNESS: f32 = 0.5;

    /// A texel on ground of neutral dryness.
    pub fn new(density: f32, height_scale: f32, palette: f32) -> Self {
        Self {
            density,
            height_scale,
            palette,
            dryness: Self::NEUTRAL_DRYNESS,
        }
    }

    pub fn with_dryness(self, dryness: f32) -> Self {
        Self { dryness, ..self }
    }
}

impl MeadowGrid<CoverTexel> {
    /// The sample tile blades take at `world_xz` (`meadow_sample_cover`):
    /// bilinear density, height scale and dryness, the nearest texel's
    /// palette index.
    pub fn sample(&self, world_xz: Vec2) -> CoverTexel {
        let (texels, t) = self.footprint(world_xz);
        let lerp2 = |f: fn(&CoverTexel) -> f32| lattice_blend(texels.map(|texel| f(&texel)), t);
        let [a, b, c, d] = texels;
        let nearest = match (t.x >= 0.5, t.y >= 0.5) {
            (false, false) => a,
            (true, false) => b,
            (false, true) => c,
            (true, true) => d,
        };
        CoverTexel {
            density: lerp2(|texel| texel.density),
            height_scale: lerp2(|texel| texel.height_scale),
            palette: nearest.palette,
            dryness: lerp2(|texel| texel.dryness),
        }
    }

    /// The grid as an `Rgba32Float` atlas for the GPU.
    pub fn to_image(&self) -> Image {
        grid_image(
            self.dims,
            TextureFormat::Rgba32Float,
            bytemuck::cast_slice(&self.texels),
        )
    }

    /// The largest height scale in the grid (at least 1): bounds blade
    /// height for culling.
    pub fn max_height_scale(&self) -> f32 {
        self.texels
            .iter()
            .fold(1.0_f32, |max, texel| max.max(texel.height_scale))
    }
}

/// A GPU-only atlas of `bytes`; the caller keeps the CPU copy.
pub(crate) fn grid_image(dims: UVec2, format: TextureFormat, bytes: &[u8]) -> Image {
    Image::new(
        Extent3d {
            width: dims.x,
            height: dims.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes.to_vec(),
        format,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// How tile blades turn cover density into blades.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverSurvival {
    /// Feature size of the survival noise, in metres: the size of the clumps
    /// and gaps at high clumpiness.
    pub noise_scale_m: f32,
    /// Mix of the noise into each blade's survival threshold, in `[0, 1]`:
    /// 0 thins blade by blade (an even dither), 1 cuts whole noise-sized
    /// clumps with hard edges, between gives soft-edged clumps.
    pub clumpiness: f32,
}

impl Default for CoverSurvival {
    fn default() -> Self {
        Self {
            noise_scale_m: 6.0,
            clumpiness: 0.5,
        }
    }
}

/// World-level cover field for tile patches: a `Rgba32Float` atlas of
/// [`CoverTexel`]s and its CPU grid. Set it with [`Self::set_grid`];
/// every variant's kernels read it. Without a grid, tile blades grow at
/// full cover and unit height with their patch's palette entry.
#[derive(Resource, Debug, Clone)]
pub struct MeadowCoverField {
    /// The GPU atlas of `grid`.
    pub atlas: Handle<Image>,
    pub grid: Option<Arc<MeadowGrid<CoverTexel>>>,
    pub survival: CoverSurvival,
    /// `grid`'s largest height scale, at least 1.
    max_height_scale: f32,
}

impl Default for MeadowCoverField {
    fn default() -> Self {
        Self {
            atlas: Handle::default(),
            grid: None,
            survival: CoverSurvival::default(),
            max_height_scale: 1.0,
        }
    }
}

impl MeadowCoverField {
    /// Replace the field with `grid`, uploading it as a new atlas.
    pub fn set_grid(&mut self, grid: MeadowGrid<CoverTexel>, images: &mut Assets<Image>) {
        self.atlas = images.add(grid.to_image());
        self.max_height_scale = grid.max_height_scale();
        self.grid = Some(Arc::new(grid));
    }

    /// Remove the field.
    pub fn clear(&mut self) {
        *self = Self {
            survival: self.survival,
            ..default()
        };
    }

    /// The field's `(extents, params)` lanes of
    /// [`VariantParams`](crate::VariantParams) (`cover_extents`, `cover`).
    pub(crate) fn params(&self) -> (Vec4, Vec4) {
        let extents = self
            .grid
            .as_ref()
            .map_or(Vec4::new(0.0, 0.0, 1.0, 1.0), |grid| {
                Vec4::new(
                    grid.world_min.x,
                    grid.world_min.y,
                    grid.world_max.x,
                    grid.world_max.y,
                )
            });
        let survival = Vec4::new(
            self.survival.noise_scale_m.max(1e-3).recip(),
            self.survival.clumpiness.clamp(0.0, 1.0),
            self.max_height_scale,
            0.0,
        );
        (extents, survival)
    }
}

// ---------- CPU twins of the shader derivation (`meadow_shared.wesl`) ----------

pub(crate) fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

pub(crate) fn hash01_u32(v: u32) -> f32 {
    (hash_u32(v) & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// `meadow_tile_root`.
pub(crate) fn tile_root(half: f32, blade_seed: u32) -> Vec2 {
    let u = Vec2::new(
        hash01_u32(blade_seed),
        hash01_u32(blade_seed.wrapping_add(1)),
    );
    (2.0 * u - Vec2::ONE) * half
}

/// `MEADOW_NOISE_ROTATION`.
const NOISE_ROTATION: Vec2 = Vec2::new(0.8, 0.6);

/// `meadow_noise_lattice` and `meadow_corner_hashes`: the corner hashes
/// (x0z0, x1z0, x0z1, x1z1) of the lattice cell about `p` turned by `rot`
/// (cos, sin), the offset within it and the quintic weights.
fn noise_lattice(p: Vec2, rot: Vec2, seed: u32) -> ([u32; 4], Vec2, Vec2) {
    let q = Vec2::new(rot.x * p.x - rot.y * p.y, rot.y * p.x + rot.x * p.y);
    let floor = q.floor();
    let cell = floor.as_ivec2();
    let f = q - floor;
    let x0 = (cell.x as u32).wrapping_add(seed);
    let x1 = x0.wrapping_add(1);
    let k0 = (cell.y as u32).wrapping_mul(0x9e37_79b9);
    let k1 = (cell.y as u32).wrapping_add(1).wrapping_mul(0x9e37_79b9);
    (
        [
            hash_u32(x0 ^ k0),
            hash_u32(x1 ^ k0),
            hash_u32(x0 ^ k1),
            hash_u32(x1 ^ k1),
        ],
        f,
        f * f * f * (f * (f * 6.0 - 15.0) + 10.0),
    )
}

/// `meadow_lattice_blend`: bilinear blend of corner values (x0z0, x1z0,
/// x0z1, x1z1) by `u`.
fn lattice_blend([a, b, c, d]: [f32; 4], u: Vec2) -> f32 {
    let top = a + (b - a) * u.x;
    let bottom = c + (d - c) * u.x;
    top + (bottom - top) * u.y
}

/// `meadow_value_noise`.
fn value_noise(p: Vec2, seed: u32) -> f32 {
    let (h, _, u) = noise_lattice(p, NOISE_ROTATION, seed);
    lattice_blend(h.map(|h| (h & 0x00ff_ffff) as f32 / 16_777_216.0), u)
}

/// `meadow_value_noise3`.
#[cfg(test)]
fn value_noise3(p: Vec2, seed: u32) -> Vec3 {
    let (h, _, u) = noise_lattice(p, NOISE_ROTATION, seed);
    let channel = |shift: u32| lattice_blend(h.map(|h| ((h >> shift) & 1023) as f32 / 1024.0), u);
    Vec3::new(channel(0), channel(10), channel(20))
}

/// `meadow_oriented_noise`.
#[cfg(test)]
fn oriented_noise(p: Vec2, rot: Vec2, seed: u32, dir: Vec2, spread: f32) -> f32 {
    let (h, f, u) = noise_lattice(p, rot, seed);
    let d = Vec2::new(rot.x * dir.x - rot.y * dir.y, rot.y * dir.x + rot.x * dir.y);
    let corners = [Vec2::ZERO, Vec2::X, Vec2::Y, Vec2::ONE];
    let v = std::array::from_fn(|i| {
        let sign = if h[i] >> 31 != 0 { 1.0 } else { -1.0 };
        let r = Vec2::new(
            (h[i] & 0xffff) as f32 / 32_768.0 - 1.0,
            ((h[i] >> 16) & 0x7fff) as f32 / 16_384.0 - 1.0,
        );
        (sign * (d + spread * r)).dot(f - corners[i])
    });
    lattice_blend(v, u)
}

/// `meadow_uniform_noise`: rotated quintic value noise remapped to a
/// uniform distribution.
pub(crate) fn uniform_noise(p: Vec2) -> f32 {
    let n = value_noise(p, 0) - 0.5;
    let n2 = n * n;
    (0.5 + n * (1.49767 + n2 * (-2.16135 + n2 * 0.73196))).clamp(0.0, 1.0)
}

/// `meadow_patch_fbm`: domain-warped, anisotropic four-octave gradient
/// noise, in units of the patch size, remapped to a uniform distribution.
#[cfg(test)]
pub(crate) fn patch_noise(p: Vec2) -> f32 {
    const SPREAD: f32 = 0.6;
    let lp = 0.5 * p;
    let w = value_noise3(0.25 * lp, 0x51ed);
    let q = lp + 2.2 * (w.truncate() - Vec2::splat(0.5));
    let angle = std::f32::consts::TAU * w.z;
    let dir = Vec2::new(angle.cos(), angle.sin());
    let sum = 3.0 * oriented_noise(0.45 * q, Vec2::new(0.96, 0.28), 0x2b5f, dir, 1.2)
        + 4.0 * oriented_noise(q, Vec2::new(0.8, 0.6), 0x9e37, dir, SPREAD)
        + 2.6 * oriented_noise(2.03 * q, Vec2::new(0.28, 0.96), 0x7f4a, dir, SPREAD)
        + 1.7 * oriented_noise(4.1209 * q, Vec2::new(0.6, 0.8), 0x3c6f, dir, SPREAD);
    let n = sum / 11.3;
    let n2 = n * n;
    0.5 + 0.5
        * (n * (5.96009 + n2 * (14.33164 - n2 * 0.70197)))
            .clamp(-9.0, 9.0)
            .tanh()
}

/// `meadow_cover_threshold`: the blade's uniform `blade_u` mixed with the
/// uniform `noise_u` by `clumpiness`, mapped back to a uniform threshold
/// through the CDF of the weighted sum.
pub(crate) fn cover_threshold(blade_u: f32, noise_u: f32, clumpiness: f32) -> f32 {
    let b = clumpiness.clamp(0.0, 1.0);
    let a = 1.0 - b;
    let lo = a.min(b);
    let hi = a.max(b);
    if lo < 1e-4 {
        return if a > b { blade_u } else { noise_u };
    }
    let z = a * blade_u + b * noise_u;
    if z < lo {
        z * z / (2.0 * a * b)
    } else if z > hi {
        let w = 1.0 - z;
        1.0 - w * w / (2.0 * a * b)
    } else {
        (z - 0.5 * lo) / hi
    }
}

/// `meadow_cover_survives`.
pub(crate) fn cover_survives(
    survival: &CoverSurvival,
    root: Vec2,
    seed: u32,
    density: f32,
    ceiling: f32,
) -> bool {
    let noise_u = uniform_noise(root / survival.noise_scale_m.max(1e-3));
    let threshold = cover_threshold(
        hash01_u32(seed.wrapping_add(37)),
        noise_u,
        survival.clumpiness,
    );
    density > threshold * ceiling.max(1e-4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The threshold maps two uniforms to a uniform at every clumpiness, so
    /// the expected cover is the density.
    #[test]
    fn threshold_stays_uniform_at_any_clumpiness() {
        for clumpiness in [0.0, 0.1, 0.3, 0.5, 0.7, 0.95, 1.0] {
            let mut bins = [0u32; 10];
            let n = 200_000u32;
            for i in 0..n {
                let t = cover_threshold(hash01_u32(i), hash01_u32(i ^ 0x5bd1_e995), clumpiness);
                bins[((t * 10.0) as usize).min(9)] += 1;
            }
            for (bin, &count) in bins.iter().enumerate() {
                let share = count as f32 / n as f32;
                assert!(
                    (share - 0.1).abs() < 0.006,
                    "clumpiness {clumpiness}: bin {bin} holds {share}"
                );
            }
        }
    }

    /// Survivors over a uniform field match its density at every
    /// clumpiness, with or without a cover ceiling.
    #[test]
    fn expected_cover_equals_density() {
        for clumpiness in [0.0, 0.5, 1.0] {
            let survival = CoverSurvival {
                noise_scale_m: 3.0,
                clumpiness,
            };
            for (density, ceiling) in [(0.2, 1.0), (0.5, 1.0), (0.85, 1.0), (0.3, 0.4)] {
                let n = 400_000u32;
                let alive = (0..n)
                    .filter(|&i| {
                        let seed = hash_u32(i);
                        let root =
                            Vec2::new(hash01_u32(seed), hash01_u32(seed.wrapping_add(1))) * 400.0;
                        cover_survives(&survival, root, seed, density, ceiling)
                    })
                    .count();
                let cover = alive as f32 / n as f32 * ceiling;
                assert!(
                    (cover - density).abs() < 0.015,
                    "clumpiness {clumpiness}, density {density}: cover {cover}"
                );
            }
        }
    }

    /// Clumpiness grows the cover's spatial variance: per 1 m cell, survival
    /// spreads from binomial noise at 0 toward all-or-nothing at 1.
    #[test]
    fn clumpiness_grows_patch_contrast() {
        let variance = |clumpiness: f32| {
            let survival = CoverSurvival {
                noise_scale_m: 6.0,
                clumpiness,
            };
            let cells = 64u32;
            let per_cell = 200u32;
            let shares: Vec<f32> = (0..cells * cells)
                .map(|cell| {
                    let origin = Vec2::new((cell % cells) as f32, (cell / cells) as f32);
                    let alive = (0..per_cell)
                        .filter(|&i| {
                            let seed = hash_u32(cell * per_cell + i);
                            let root = origin
                                + Vec2::new(hash01_u32(seed), hash01_u32(seed.wrapping_add(1)));
                            cover_survives(&survival, root, seed, 0.5, 1.0)
                        })
                        .count();
                    alive as f32 / per_cell as f32
                })
                .collect();
            let mean = shares.iter().sum::<f32>() / shares.len() as f32;
            shares.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / shares.len() as f32
        };
        let (dither, soft, hard) = (variance(0.0), variance(0.5), variance(1.0));
        // Binomial at p = 0.5 over 200 blades.
        assert!(dither < 0.002, "{dither}");
        assert!(soft > 4.0 * dither, "{soft} vs {dither}");
        assert!(hard > 2.0 * soft, "{hard} vs {soft}");
        assert!(hard > 0.15, "{hard}");
    }

    #[test]
    fn grid_samples_on_texel_fences() {
        let grid = MeadowGrid::new(
            Vec2::new(-2.0, 10.0),
            Vec2::new(2.0, 14.0),
            UVec2::new(3, 3),
            (0..9).map(|i| i as f32).collect(),
        );
        assert_eq!(grid.sample(Vec2::new(-2.0, 10.0)), 0.0);
        assert_eq!(grid.sample(Vec2::new(2.0, 14.0)), 8.0);
        assert_eq!(grid.sample(Vec2::new(0.0, 12.0)), 4.0);
        assert_eq!(grid.sample(Vec2::new(-1.0, 10.0)), 0.5);
        assert_eq!(grid.sample(Vec2::new(-50.0, 50.0)), 6.0);
        let cover = MeadowGrid::new(
            Vec2::ZERO,
            Vec2::ONE,
            UVec2::new(2, 2),
            vec![
                CoverTexel::new(0.0, 1.0, 1.0).with_dryness(0.0),
                CoverTexel::new(1.0, 1.0, 2.0).with_dryness(1.0),
                CoverTexel::new(0.0, 3.0, 3.0),
                CoverTexel::new(1.0, 3.0, 4.0),
            ],
        );
        let s = cover.sample(Vec2::new(0.75, 0.25));
        assert_eq!(
            (s.density, s.height_scale, s.palette, s.dryness),
            (0.75, 1.5, 2.0, 0.6875)
        );
        assert_eq!(CoverTexel::new(1.0, 1.0, 0.0).dryness, 0.5);
        assert_eq!(cover.max_height_scale(), 3.0);
    }

    /// The warped patch noise is uniform in `[0, 1]`, so a threshold at
    /// `1 - a` covers a share `a` of the ground, small shares included.
    #[test]
    fn patch_noise_is_uniform() {
        let n = 1_000_000u32;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let seed = hash_u32(i ^ 0x68e3_1da4);
                let p = Vec2::new(hash01_u32(seed), hash01_u32(seed.wrapping_add(1)));
                patch_noise((p - 0.5) * 4000.0)
            })
            .collect();
        let mut bins = [0u32; 20];
        for &u in &samples {
            bins[((u * 20.0) as usize).min(19)] += 1;
        }
        for (bin, &count) in bins.iter().enumerate() {
            let share = count as f32 / n as f32;
            assert!((share - 0.05).abs() < 0.0015, "bin {bin} holds {share}");
        }
        for share in [0.005, 0.01, 0.03, 0.1, 0.35, 0.65, 0.9] {
            let covered = samples.iter().filter(|&&u| u > 1.0 - share).count() as f32 / n as f32;
            assert!(
                (covered - share).abs() < 0.00025 + 0.005 * share,
                "share {share}: covered {covered}"
            );
        }
    }

    /// The patch noise is continuous at the blade scale but varies across
    /// a patch: roots a hundredth of a patch apart agree, roots a patch
    /// apart don't.
    #[test]
    fn patch_noise_is_smooth_at_the_blade_scale() {
        let (mut near, mut far) = (0.0f32, 0.0f32);
        for i in 0..10_000u32 {
            let p = Vec2::new(hash01_u32(i), hash01_u32(i ^ 0x9e37_79b9)) * 1000.0;
            near = near.max((patch_noise(p) - patch_noise(p + Vec2::new(0.01, 0.0))).abs());
            far += (patch_noise(p) - patch_noise(p + Vec2::new(1.0, 0.0))).abs();
        }
        assert!(near < 0.1, "{near}");
        assert!(far / 10_000.0 > 0.2, "{}", far / 10_000.0);
    }

    /// The CPU twins carry the shader's constants.
    #[test]
    fn twins_match_the_shader_constants() {
        let shared = include_str!("meadow_shared.wesl");
        for needle in [
            "x = x * 0x7feb352du;",
            "x = x * 0x846ca68bu;",
            "hash_u32(a ^ (b * 0x9E3779B9u))",
            "let x0 = bitcast<u32>(cell.x) + seed;",
            "let u = vec2<f32>(hash01_u32(blade_seed), hash01_u32(blade_seed + 1u));",
            "return (2.0 * u - 1.0) * half;",
            "let k0 = bitcast<u32>(cell.y) * 0x9E3779B9u;",
            "let k1 = (bitcast<u32>(cell.y) + 1u) * 0x9E3779B9u;",
            "hash_u32((x0 + 1u) ^ k0),",
            "hash_u32((x0 + 1u) ^ k1),",
            "let z = mix(v.xz, v.yw, weight.x);",
            "return mix(z.x, z.y, weight.y);",
            "f * f * f * (f * (f * 6.0 - 15.0) + 10.0)",
            "vec4<f32>(h & vec4<u32>(0x00FFFFFFu)) / f32(0x01000000u)",
            "return vec4<f32>(blended.rg, palette, blended.a);",
            "0.5 + n * (1.49767 + n2 * (-2.16135 + n2 * 0.73196))",
            "const MEADOW_NOISE_ROTATION: vec2<f32> = vec2<f32>(0.8, 0.6);",
            "vec2<f32>(rot.x * p.x - rot.y * p.y, rot.y * p.x + rot.x * p.y)",
            "vec2<f32>(rot.x * dir.x - rot.y * dir.y, rot.y * dir.x + rot.x * dir.y)",
            "meadow_lattice_blend(vec4<f32>((h >> vec4<u32>(10u)) & bits) / 1024.0, lattice.weight)",
            "meadow_lattice_blend(vec4<f32>((h >> vec4<u32>(20u)) & bits) / 1024.0, lattice.weight)",
            "(h >> vec4<u32>(31u)) != vec4<u32>(0u)",
            "let rx = vec4<f32>(h & vec4<u32>(0xFFFFu)) / 32768.0 - 1.0;",
            "let rz = vec4<f32>((h >> vec4<u32>(16u)) & vec4<u32>(0x7FFFu)) / 16384.0 - 1.0;",
            "let v = sign * (d.x + spread * rx) * (f.x - vec4<f32>(0.0, 1.0, 0.0, 1.0))",
            "+ sign * (d.y + spread * rz) * (f.y - vec4<f32>(0.0, 0.0, 1.0, 1.0));",
            "const MEADOW_PATCH_SPREAD: f32 = 0.6;",
            "let lp = 0.5 * p;",
            "let w = meadow_value_noise3(0.25 * lp, 0x51EDu);",
            "let q = lp + 2.2 * (w.xy - vec2<f32>(0.5));",
            "let angle = 6.28318531 * w.z;",
            "let dir = vec2<f32>(cos(angle), sin(angle));",
            "let bits = vec4<u32>(1023u);",
            "meadow_lattice_blend(vec4<f32>(h & bits) / 1024.0, lattice.weight)",
            "hash_u32(x0 ^ k0),",
            "hash_u32(x0 ^ k1),",
            "select(vec4<f32>(-1.0), vec4<f32>(1.0), (h >> vec4<u32>(31u)) != vec4<u32>(0u))",
            "let sum = 3.0 * meadow_oriented_noise(0.45 * q, vec2<f32>(0.96, 0.28), 0x2B5Fu, dir, 1.2)",
            "+ 4.0 * meadow_oriented_noise(q, vec2<f32>(0.8, 0.6), 0x9E37u, dir, MEADOW_PATCH_SPREAD)",
            "+ 2.6 * meadow_oriented_noise(2.03 * q, vec2<f32>(0.28, 0.96), 0x7F4Au, dir, MEADOW_PATCH_SPREAD)",
            "+ 1.7 * meadow_oriented_noise(4.1209 * q, vec2<f32>(0.6, 0.8), 0x3C6Fu, dir, MEADOW_PATCH_SPREAD);",
            "let n = sum / 11.3;",
            "return 0.5 + 0.5 * tanh(clamp(n * (5.96009 + n2 * (14.33164 - n2 * 0.70197)), -9.0, 9.0));",
            "return vec4<f32>(1.0, 1.0, -1.0, MEADOW_NEUTRAL_DRYNESS);",
            "meadow_cover_threshold(hash01_u32(seed + 37u), noise_u, params.cover.y)",
            "return z * z / (2.0 * a * b);",
            "return (z - 0.5 * lo) / hi;",
        ] {
            assert!(
                shared.contains(needle),
                "`{needle}` missing from meadow_shared.wesl"
            );
        }
        let neutral = format!(
            "const MEADOW_NEUTRAL_DRYNESS: f32 = {:?};",
            CoverTexel::NEUTRAL_DRYNESS
        );
        assert!(shared.contains(&neutral), "`{neutral}` missing");
    }
}
