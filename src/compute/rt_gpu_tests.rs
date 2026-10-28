//! The raytracing caster selection and expansion, run on the GPU through
//! the passes the renderer records, against the CPU reference ranking
//! (`rt_selection::Ranker`). Skipped without a GPU adapter.

use std::sync::OnceLock;

use bevy::platform::collections::HashSet;
use bevy::render::render_resource::{
    Extent3d, PipelineLayoutDescriptor, RawComputePipelineDescriptor, ShaderModuleDescriptor,
    ShaderSource, TextureDataOrder, TextureDimension, TextureFormat, TextureViewDescriptor,
};
use bevy::render::renderer::initialize_renderer;
use bevy::render::settings::{Backends, RenderResources, WgpuSettings};
use bevy::shader::Shader;

use super::*;
use crate::field::tests::{blade_seed, tile_root};
use crate::field::{CoverTexel, MeadowCoverField, MeadowGrid, hash_u32};
use crate::material::{PatchData, PatchTrunkSlot, TrunkDisc};
use crate::placement::PatchPlacement;
use crate::plugin::{MeadowPalette, palette_table, patch_row, variant_params_from};
use crate::rt_selection::{
    BLADE_GROUND_LIFT, FAR_DISTANCE, Ranker, RtInputs, UNBOUNDED_BUDGET, keep_share, least_budget,
};
use crate::shader_tests::Library;

/// The test device, `None` without an adapter.
fn device() -> Option<&'static (RenderDevice, RenderQueue)> {
    static DEVICE: OnceLock<Option<(RenderDevice, RenderQueue)>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            std::panic::catch_unwind(|| {
                let RenderResources(device, queue, ..) =
                    bevy::tasks::block_on(initialize_renderer(
                        Backends::from_env().unwrap_or(Backends::PRIMARY),
                        None,
                        &WgpuSettings::default(),
                    ));
                (device, queue)
            })
            .ok()
        })
        .as_ref()
}

/// `meadow_compute.wesl`, composed and validated by bevy's shader cache.
fn compute_wgsl() -> String {
    let mut lib = Library::meadow();
    let shader = lib.add(Shader::from_wesl(
        include_str!("../meadow_compute.wesl"),
        "embedded://bevy_meadow/meadow_compute.wesl",
    ));
    lib.compile(shader, &[]).to_string()
}

/// The kernels compose and validate, and every entry point the pipelines
/// name survives.
#[test]
fn rt_kernels_compile_and_validate() {
    let wgsl = compute_wgsl();
    for entry in RT_ENTRIES {
        assert!(
            wgsl.contains(&format!("fn {entry}(")),
            "entry `{entry}` lost"
        );
    }
    assert_eq!(
        RT_NEAR_MAX_BLADES * crate::mesh::BLADE_INDICES_PER_BLADE / 3,
        294_912,
        "near fidelity must not increase the triangle budget",
    );
    assert_eq!(
        RT_FAR_MAX_BLADES * RT_FAR_INDICES_PER_BLADE / 3,
        262_144,
        "the far band's triangle budget",
    );
}

/// A field of 8 m tiles over a rolling heightfield, one tile with a
/// clearing disc.
struct Scene {
    inputs: RtInputs,
    /// `inputs` with more tiles streamed in, doubling the grass by the
    /// field's centre.
    grown: RtInputs,
    disc: TrunkDisc,
}

const TILES_PER_SIDE: u32 = 16;
const BLADES_PER_TILE: u32 = 9600;

fn scene() -> Scene {
    let half = TILES_PER_SIDE as f32 * 4.0;
    let dims = UVec2::splat(129);
    let heights = (0..dims.x * dims.y)
        .map(|i| {
            let p = Vec2::new((i % dims.x) as f32, (i / dims.x) as f32) * (2.0 * half / 128.0)
                - Vec2::splat(half);
            // Rolling ground with relief of a metre and more inside a tile.
            1.5 * (p.x / 9.0).sin()
                + (p.y / 7.0).cos()
                + 0.03 * p.x
                + 1.2 * (p.x * 1.1).sin() * (p.y * 0.9).sin()
        })
        .collect();
    let heights = MeadowGrid::new(Vec2::splat(-half), Vec2::splat(half), dims, heights);
    let tiles = TILES_PER_SIDE * TILES_PER_SIDE;
    let centre = |cell: u32| {
        Vec2::new(
            (cell % TILES_PER_SIDE) as f32 * 8.0 + 4.0 - half,
            (cell / TILES_PER_SIDE) as f32 * 8.0 + 4.0 - half,
        )
    };
    // A tile on every cell, then a second on each of the four round the
    // centre.
    let c = TILES_PER_SIDE / 2;
    let cells = (0..tiles).chain([
        (c - 1) * TILES_PER_SIDE + c - 1,
        (c - 1) * TILES_PER_SIDE + c,
        c * TILES_PER_SIDE + c - 1,
        c * TILES_PER_SIDE + c,
    ]);
    let grown: Vec<(u32, PatchPlacement)> = cells
        .enumerate()
        .map(|(i, cell)| {
            let i = i as u32;
            let tile =
                PatchPlacement::tile(centre(cell), 4.0, BLADES_PER_TILE, hash_u32(i + 1), 1.0, 0);
            (i, tile)
        })
        .collect();
    let disc = TrunkDisc::new(Vec2::new(6.0, 2.0), 1.5, 0.5);
    let mut trunks = vec![PatchTrunkSlot::default(); grown.len()];
    let tile = grown
        .iter()
        .find(|(_, p)| (disc.centre() - p.centre).abs().max_element() < 4.0)
        .unwrap()
        .0;
    trunks[tile as usize] = PatchTrunkSlot::from_discs(&[disc]);
    let inputs = RtInputs {
        placements: Arc::new(grown[..tiles as usize].to_vec()),
        trunks: Arc::new(trunks),
        heights: Some(Arc::new(heights)),
        ..Default::default()
    };
    Scene {
        grown: RtInputs {
            placements: Arc::new(grown),
            ..inputs.clone()
        },
        inputs,
        disc,
    }
}

/// One blade as the CPU reference sees it.
struct Blade {
    id: [u32; 2],
    point: Vec3,
    rank: f32,
    /// Whether the clearing disc may cut it, which the reference leaves to
    /// the GPU.
    cleared: bool,
}

impl Scene {
    /// The blades of `inputs`, one of the scene's.
    fn blades(&self, inputs: &RtInputs) -> Vec<Blade> {
        let heights = inputs.heights.as_ref().unwrap();
        let mut blades = Vec::new();
        for (index, p) in inputs.placements.iter() {
            for b in 0..p.blade_count {
                let root = p.centre + tile_root(p.radius, blade_seed(p.seed, b));
                blades.push(Blade {
                    id: [*index, b],
                    point: Vec3::new(root.x, heights.sample(root) + BLADE_GROUND_LIFT, root.y),
                    rank: (b as f32 + 0.5) / p.blade_count as f32,
                    cleared: root.distance(self.disc.centre()) < self.disc.outer_radius() + 0.05,
                });
            }
        }
        blades
    }

    fn eye(&self, xz: Vec2) -> Vec3 {
        Vec3::new(
            xz.x,
            self.inputs.heights.as_ref().unwrap().sample(xz) + 1.7,
            xz.y,
        )
    }
}

/// `inputs` without patch `gone`.
fn without(inputs: &RtInputs, gone: u32) -> RtInputs {
    RtInputs {
        placements: Arc::new(
            inputs
                .placements
                .iter()
                .copied()
                .filter(|(index, _)| *index != gone)
                .collect(),
        ),
        ..inputs.clone()
    }
}

/// The casters' frame drawing `inputs` from `viewer`, easing the drawn
/// limits by `ease`.
fn extract(inputs: &RtInputs, viewer: RtPose, exact_radius: Option<f32>, ease: f32) -> RtExtract {
    RtExtract {
        inputs: inputs.clone(),
        viewer,
        margins: MeadowRtMargins::default(),
        exact_radius,
        sunward_widen: 1.0,
        ease,
    }
}

/// The renderer's selection and expansion over a [`Scene`].
struct Harness {
    device: RenderDevice,
    queue: RenderQueue,
    /// In [`RT_ENTRIES`] order.
    pipelines: Vec<ComputePipeline>,
    layouts: RtLayouts,
    variant: UniformBuffer<VariantParams>,
    patches: Buffer,
    trunks: Buffer,
    palettes: Buffer,
    heightfield: TextureView,
    cover: TextureView,
    blank: TextureView,
    rtv: RtVariantBuffers,
}

impl Harness {
    fn new(scene: &Scene) -> Option<Self> {
        let (device, queue) = device()?.clone();
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("meadow_compute"),
            source: ShaderSource::Wgsl(compute_wgsl().into()),
        });
        let layout = |desc: BindGroupLayoutDescriptor| {
            device.create_bind_group_layout(desc.label.as_ref(), &desc.entries)
        };
        let layouts = RtLayouts {
            select: layout(rt_select_layout()),
            expand: layout(rt_expand_layout()),
            pad: layout(rt_pad_layout()),
        };
        let pipelines = RT_ENTRIES
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(layouts.of(i))],
                    immediate_size: 0,
                });
                device.create_compute_pipeline(&RawComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();

        let heights = scene.inputs.heights.as_ref().unwrap();
        let mut params = variant_params_from(&MeadowVariant::default());
        params.heightfield_extents = Vec4::new(
            heights.world_min.x,
            heights.world_min.y,
            heights.world_max.x,
            heights.world_max.y,
        );
        (params.cover_extents, params.cover) = MeadowCoverField::default().params();
        params.sun = Vec3::new(0.3, 0.6, -0.7).normalize().extend(0.0);
        let mut variant = UniformBuffer::from(params);
        variant.write_buffer(&device, &queue);
        let storage = |label: &str, contents: &[u8]| {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        };
        let rows: Vec<PatchData> = scene
            .grown
            .placements
            .iter()
            .map(|(_, p)| patch_row(p))
            .collect();
        let blank = Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![0; 16],
            TextureFormat::Rgba32Float,
            Default::default(),
        );
        let blank = texture_view(&device, &queue, &blank);
        Some(Self {
            pipelines,
            layouts,
            patches: storage("patches", bytemuck::cast_slice(&rows)),
            trunks: storage("trunks", bytemuck::cast_slice(&scene.grown.trunks)),
            palettes: storage(
                "palettes",
                bytemuck::cast_slice(&palette_table(&[MeadowPalette::default()])),
            ),
            heightfield: texture_view(&device, &queue, &heights.to_image()),
            cover: blank.clone(),
            blank,
            rtv: RtVariantBuffers::new(&device, &queue, make_rt_band_indices(&device)),
            variant,
            device,
            queue,
        })
    }

    /// Rewrite the clearing-disc rows.
    fn set_trunks(&mut self, slots: &[PatchTrunkSlot]) {
        self.queue
            .write_buffer(&self.trunks, 0, bytemuck::cast_slice(slots));
    }

    /// Grow the tiles from `grid`.
    fn set_cover(&mut self, grid: &MeadowGrid<CoverTexel>) {
        self.cover = texture_view(&self.device, &self.queue, &grid.to_image());
        let mut field = MeadowCoverField::default();
        field.set_grid(grid.clone(), &mut Assets::<Image>::default());
        let params = self.variant.get_mut();
        (params.cover_extents, params.cover) = field.params();
        self.variant.write_buffer(&self.device, &self.queue);
    }

    /// Record with `record`, run it, and take the selection state back
    /// through the renderer's readback.
    fn run(
        &mut self,
        record: impl FnOnce(
            &mut RtVariantBuffers,
            &mut CommandEncoder,
            &RtPasses,
            &RtBlades,
            &RtLayouts,
        ),
    ) {
        let blades = RtBlades {
            params: self.variant.buffer().unwrap(),
            patches: &self.patches,
            trunks: &self.trunks,
            palettes: &self.palettes,
            textures: [&self.heightfield, &self.cover, &self.blank],
        };
        let passes = RtPasses(std::array::from_fn(|i| &self.pipelines[i]));
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        record(&mut self.rtv, &mut encoder, &passes, &blades, &self.layouts);
        self.queue.submit([encoder.finish()]);
        self.rtv.poll_readbacks();
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("the frame completes");
        self.rtv.poll_readbacks();
    }

    /// Run one frame drawing `inputs` from `viewer`, easing the drawn limits
    /// by `ease`, with the selection due there; returns that selection.
    fn frame(
        &mut self,
        inputs: &RtInputs,
        viewer: RtPose,
        exact_radius: Option<f32>,
        ease: f32,
    ) -> Option<RtSelect> {
        let rt = extract(inputs, viewer, exact_radius, ease);
        let (device, queue) = (self.device.clone(), self.queue.clone());
        let mut select = None;
        self.run(|rtv, encoder, passes, blades, layouts| {
            select = rtv.select(
                &device,
                &queue,
                encoder,
                passes,
                &layouts.select,
                blades,
                &rt,
                None,
            );
            rtv.expand(&device, encoder, passes, &layouts.expand, blades);
        });
        select
    }

    /// Run one frame drawing from `viewer` without selecting there.
    fn draw(&mut self, viewer: RtPose) {
        let params = extract(&RtInputs::default(), viewer, None, 1.0).params();
        let (device, queue) = (self.device.clone(), self.queue.clone());
        self.run(|rtv, encoder, passes, blades, layouts| {
            queue.write_buffer(
                &rtv.params,
                RT_PARAMS_CPU as u64,
                &bytemuck::bytes_of(&params)[RT_PARAMS_CPU..],
            );
            let group = &rtv.select_group.as_ref().unwrap().1;
            passes.record(encoder, "ease", group, &[(RtPass::Ease, 1)]);
            rtv.expand(&device, encoder, passes, &layouts.expand, blades);
        });
    }

    /// Clear the bands, as for a frame whose inputs are missing.
    fn pad(&mut self) {
        let device = self.device.clone();
        self.run(|rtv, encoder, passes, _, layouts| {
            rtv.pad(&device, encoder, passes, &layouts.pad);
        });
    }

    fn read(&self, buffer: &Buffer, size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(MapMode::Read, |_| {});
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("the copy completes");
        let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
        staging.unmap();
        bytes
    }

    fn raw_state(&self) -> RtSelectStateGpu {
        bytemuck::pod_read_unaligned(&self.read(&self.rtv.select.scratch, RT_STATE_SIZE))
    }

    /// The selection state, as the readback reports it.
    fn state(&self) -> RtSelectState {
        let state = self.raw_state().decode();
        assert_eq!(self.rtv.selector.state, Some(state), "a stale readback");
        state
    }

    /// Each slot's `(patch, blade)` and whether it was drawn, near band
    /// first.
    fn slots(&self) -> Vec<(Option<[u32; 2]>, bool)> {
        let bytes = self.read(&self.rtv.select.slots, RT_SLOT_SIZE * u64::from(RT_SLOTS));
        bytemuck::cast_slice::<u8, [u32; 4]>(&bytes)
            .iter()
            .map(|s| ((s[0] != 0).then(|| [s[0] - 1, s[1]]), s[2] & 1 != 0))
            .collect()
    }

    /// Each slot's `(patch, blade)`, near band first.
    fn owners(&self) -> Vec<Option<[u32; 2]>> {
        self.slots().into_iter().map(|(owner, _)| owner).collect()
    }

    /// How many of `range`'s slots were drawn.
    fn drawn(&self, range: std::ops::Range<usize>) -> usize {
        self.slots()[range]
            .iter()
            .filter(|(_, drawn)| *drawn)
            .count()
    }

    /// Each slot's vertices as `(position, normal)`, near band first.
    fn slot_vertices(&self) -> Vec<Vec<(Vec3, Vec3)>> {
        let band = |buffers: &RtBandBuffers, per_slot: u32| {
            let bytes = self.read(
                &buffers.vertices,
                u64::from(buffers.vertex_count) * RT_VERTEX_SIZE,
            );
            bytemuck::cast_slice::<u8, [f32; (RT_VERTEX_SIZE / 4) as usize]>(&bytes)
                .chunks(per_slot as usize)
                .map(|slot| {
                    slot.iter()
                        .map(|v| (Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5])))
                        .collect()
                })
                .collect::<Vec<_>>()
        };
        let mut slots = band(&self.rtv.near, RT_NEAR_VERTS_PER_BLADE);
        slots.extend(band(&self.rtv.far, RT_FAR_VERTS_PER_BLADE));
        slots
    }
}

impl RtLayouts {
    /// The layout of [`RT_ENTRIES`]`[entry]`.
    fn of(&self, entry: usize) -> &BindGroupLayout {
        match entry {
            e if e == RtPass::Expand as usize => &self.expand,
            e if e == RtPass::Pad as usize => &self.pad,
            _ => &self.select,
        }
    }
}

fn texture_view(device: &RenderDevice, queue: &RenderQueue, image: &Image) -> TextureView {
    device
        .create_texture_with_data(
            queue,
            &image.texture_descriptor,
            TextureDataOrder::default(),
            image.data.as_ref().unwrap(),
        )
        .create_view(&TextureViewDescriptor::default())
}

const NEAR_SLOTS: std::ops::Range<usize> = 0..RT_NEAR_MAX_BLADES as usize;
const ALL_SLOTS: std::ops::Range<usize> = 0..RT_SLOTS as usize;

/// The occupied ids of one band's slots, failing on a blade held twice.
fn band(owners: &[Option<[u32; 2]>], near: bool) -> HashSet<[u32; 2]> {
    let slots = if near {
        &owners[..RT_NEAR_MAX_BLADES as usize]
    } else {
        &owners[RT_NEAR_MAX_BLADES as usize..]
    };
    let mut held = HashSet::default();
    for id in slots.iter().flatten() {
        assert!(held.insert(*id), "{id:?} holds two slots");
    }
    held
}

/// Relative slack between the GPU's and the CPU's rounding.
const SLACK: f32 = 1e-4;

/// The selection the GPU made at `pose` holds every blade the reference
/// keys within its limits, and only blades near them, within capacity.
fn assert_selection(
    blades: &[Blade],
    pose: &RtPose,
    state: &RtSelectState,
    owners: &[Option<[u32; 2]>],
) {
    let ranker = Ranker::new(pose, &MeadowRtMargins::default());
    let (near, far) = (band(owners, true), band(owners, false));
    let keys: Vec<f32> = blades
        .iter()
        .map(|b| ranker.least_rank_distance(b.point))
        .collect();
    let reach = state.near_reach;
    assert!(near.len() <= RT_NEAR_MAX_BLADES as usize);
    assert!(far.len() <= RT_FAR_MAX_BLADES as usize);
    assert_eq!(state.dropped, 0);
    // The reach is the capacity's: within a bin of the exact cut.
    let mut sorted: Vec<f32> = blades
        .iter()
        .zip(&keys)
        .filter(|(b, d)| !b.cleared && **d <= RT_NEAR_DISTANCE)
        .map(|(_, d)| *d)
        .collect();
    sorted.sort_by(f32::total_cmp);
    if sorted.len() > RT_NEAR_MAX_BLADES as usize {
        let exact = sorted[RT_NEAR_MAX_BLADES as usize];
        assert!(
            reach <= exact + 1e-3,
            "reach {reach} past the exact cut {exact}"
        );
        assert!(
            near.len() as f32 >= 0.98 * RT_NEAR_MAX_BLADES as f32,
            "{} near blades",
            near.len()
        );
    }
    let budget = state.far_budget;
    let mut far_needed = 0;
    for (b, &d) in blades.iter().zip(&keys) {
        if b.cleared {
            continue;
        }
        if d < reach * (1.0 - SLACK) {
            assert!(
                near.contains(&b.id),
                "{:?} at {d} within reach {reach}",
                b.id
            );
        }
        if near.contains(&b.id) {
            assert!(
                d <= reach * (1.0 + SLACK) + 1e-5,
                "{:?} at {d} past reach {reach}",
                b.id
            );
        }
        let key = least_budget(d, reach, b.rank);
        let within = b.point.distance(pose.eye) <= FAR_DISTANCE;
        if within && key < budget * (1.0 - SLACK) {
            far_needed += 1;
            assert!(
                far.contains(&b.id),
                "{:?} needs {key} under budget {budget}",
                b.id
            );
        }
        if far.contains(&b.id) {
            assert!(
                key <= budget * (1.0 + SLACK),
                "{:?} needs {key} over budget {budget}",
                b.id
            );
        }
    }
    assert!(far_needed > RT_FAR_MAX_BLADES as usize / 2, "{far_needed}");
    // The disc clears its core.
    for b in blades.iter().filter(|b| b.cleared) {
        if near.contains(&b.id) || far.contains(&b.id) {
            assert!(
                b.point.xz().distance(Vec2::new(6.0, 2.0)) > 1.5 - 0.01,
                "{:?} in the clearing",
                b.id
            );
        }
    }
}

/// Every blade the expansion's rules draw from `pose` under the drawn
/// limits, capped at the selection's, is resident.
fn assert_drawable_resident(
    blades: &[Blade],
    pose: &RtPose,
    state: &RtSelectState,
    owners: &[Option<[u32; 2]>],
) {
    let ranker = Ranker::new(pose, &MeadowRtMargins::default());
    let (near, far) = (band(owners, true), band(owners, false));
    let reach = state.drawn_near_reach.min(state.near_reach);
    let budget = state.drawn_far_budget.min(state.far_budget);
    for b in blades.iter().filter(|b| !b.cleared) {
        let d = ranker.rank_distance(b.point);
        if d < reach * (1.0 - SLACK) {
            assert!(
                near.contains(&b.id),
                "{:?} drawable at {d} under {reach}",
                b.id
            );
        }
        if b.point.distance(pose.eye) <= FAR_DISTANCE
            && b.rank < keep_share(d, reach, budget) * (1.0 - SLACK)
        {
            assert!(far.contains(&b.id), "far {:?} drawable at {d}", b.id);
        }
    }
}

// The device prints nothing; a skip says why.
#[allow(clippy::print_stderr)]
fn harness(scene: &Scene) -> Option<Harness> {
    let harness = Harness::new(scene);
    if harness.is_none() {
        eprintln!("skipping: no GPU adapter");
    }
    harness
}

/// A selection holds what the CPU reference ranks within its fitted
/// limits, at capacity, and the expansion's drawable blades are resident
/// from any pose within the margins.
#[test]
fn gpu_selection_matches_the_reference() {
    let scene = scene();
    let Some(mut gpu) = harness(&scene) else {
        return;
    };
    let blades = scene.blades(&scene.inputs);
    let pose = RtPose::looking(scene.eye(Vec2::new(1.3, -0.7)), 30.0, -12.0, 40.0);
    assert!(gpu.frame(&scene.inputs, pose, None, 1.0).is_some());
    let state = gpu.state();
    assert!(state.near_reach < RT_NEAR_DISTANCE && state.far_budget < UNBOUNDED_BUDGET);
    let owners = gpu.owners();
    assert_eq!(band(&owners, true).len(), state.near_selected as usize);
    assert_eq!(band(&owners, false).len(), state.far_selected as usize);
    assert_selection(&blades, &pose, &state, &owners);
    // Poses at the corners of the margins' reselect share, then of the
    // margins, the cone's change shared with the turn: `(step m, turn °,
    // widen °, reselects)`.
    for (step, turn, widen, past_share) in [
        (0.24, 3.0, 0.9, false),
        (0.24, -3.9, 0.0, false),
        (0.2, 0.0, -3.9, false),
        (0.49, 5.0, 2.9, true),
        (0.49, -7.9, 0.0, true),
        (0.3, 0.0, -7.9, true),
    ] {
        let edge = RtPose {
            eye: pose.eye + Vec3::new(1.0, 0.0, 0.2).normalize() * step,
            forward: Quat::from_rotation_y(f32::to_radians(turn)) * pose.forward,
            cone: pose.cone + f32::to_radians(widen),
        };
        if past_share {
            gpu.draw(edge);
        } else {
            assert!(gpu.frame(&scene.inputs, edge, None, 1.0).is_none());
        }
        assert_drawable_resident(&blades, &edge, &gpu.state(), &owners);
    }
}

/// Walking and turning fast, every blade the expansion drew keeps its
/// slot, every selected blade already resident keeps its slot, the
/// drawable blades are resident each frame, and no blade waits for a slot.
/// Tiles streaming in, or out of reach, keep the slots' identity; a tile in
/// reach streaming out gives its slots up under a new one.
#[test]
fn drawn_blades_keep_their_slots() {
    let scene = scene();
    let Some(mut gpu) = harness(&scene) else {
        return;
    };
    let blades = scene.blades(&scene.inputs);
    let start = Vec2::new(-3.0, 1.0);
    let mut reselections = 0;
    let mut pinned = 0;
    let grown = scene.blades(&scene.grown);
    // A corner tile, out of reach, then the tile under the walk.
    let (corner, under) = (0, 8 * TILES_PER_SIDE + 7);
    let past_corner = without(&scene.grown, corner);
    let past_under = without(&past_corner, under);
    let (corner_blades, under_blades) = (scene.blades(&past_corner), scene.blades(&past_under));
    let mut identity = None;
    for step in 0..34 {
        let t = step as f32;
        // Tiles stream in by the walk, denser than the field: the selection
        // tightens past the blades drawn last frame. Then two stream out.
        let (inputs, blades) = match step {
            0..12 => (&scene.inputs, &blades),
            12..20 => (&scene.grown, &grown),
            20..26 => (&past_corner, &corner_blades),
            _ => (&past_under, &under_blades),
        };
        // 1.5 m/s at 60 fps, turning 3.5° a frame (210°/s), then back.
        let heading = 20.0 + 3.5 * t.min(40.0 - t);
        let eye = scene.eye(start + Vec2::new(0.025, 0.01) * t);
        let pose = RtPose::looking(eye, heading, -10.0, 40.0);
        let before = gpu.slots();
        let next = gpu.frame(inputs, pose, None, 0.25);
        let now = gpu.rtv.selector.identity;
        if step == 26 {
            assert_eq!(next.as_ref().unwrap().purge, vec![under]);
            assert_ne!(Some(now), identity, "a tile in reach left");
            identity = Some(now);
        }
        assert_eq!(now, *identity.get_or_insert(now), "step {step}");
        reselections += usize::from(next.is_some());
        let state = gpu.state();
        let owners = gpu.owners();
        assert_eq!(state.dropped, 0, "step {step}");
        if step >= 26 {
            assert!(
                owners.iter().flatten().all(|id| id[0] != under),
                "step {step}"
            );
        }
        pinned += state.pinned;
        for (slot, (was, drawn)) in before.iter().enumerate() {
            if *drawn && was.is_some_and(|id| step != 26 || id[0] != under) {
                assert_eq!(
                    owners[slot], *was,
                    "step {step}: drawn slot {slot} lost its blade"
                );
            }
        }
        for range in [NEAR_SLOTS, NEAR_SLOTS.end..ALL_SLOTS.end] {
            let now: HashMap<[u32; 2], usize> = range
                .clone()
                .filter_map(|slot| owners[slot].map(|id| (id, slot)))
                .collect();
            for slot in range {
                if let Some(id) = before[slot].0
                    && let Some(&moved) = now.get(&id)
                {
                    assert_eq!(moved, slot, "step {step}: {id:?} moved slot");
                }
            }
        }
        assert_drawable_resident(blades, &pose, &state, &owners);
    }
    assert!(reselections > 10, "{reselections}");
    assert!(
        pinned > 0,
        "the selection never tightened past a drawn blade"
    );
}

/// A purged patch leaves its slots; the exact-only reference clears the
/// slots and holds every live blade within its radius by eye distance
/// alone, with no far band.
#[test]
fn purges_and_the_exact_reference() {
    let scene = scene();
    let Some(mut gpu) = harness(&scene) else {
        return;
    };
    let blades = scene.blades(&scene.inputs);
    let pose = RtPose::looking(scene.eye(Vec2::new(0.5, 0.5)), 200.0, -10.0, 40.0);
    gpu.frame(&scene.inputs, pose, None, 1.0);
    let owned = |gpu: &Harness, patch: u32| gpu.owners().iter().flatten().any(|id| id[0] == patch);
    let gone = scene
        .inputs
        .placements
        .iter()
        .find(|(_, p)| p.centre.distance(Vec2::new(4.0, 4.0)) < 1.0)
        .unwrap()
        .0;
    assert!(owned(&gpu, gone));
    let identity = gpu.rtv.selector.identity;
    let purge = gpu.frame(&without(&scene.inputs, gone), pose, None, 1.0);
    assert_eq!(purge.unwrap().purge, vec![gone]);
    assert_ne!(gpu.rtv.selector.identity, identity);
    assert!(!owned(&gpu, gone));

    let radius = 3.0;
    let exact = gpu.frame(&scene.inputs, pose, Some(radius), 1.0).unwrap();
    assert!(exact.clear);
    let state = gpu.state();
    let owners = gpu.owners();
    let (near, far) = (band(&owners, true), band(&owners, false));
    assert!(far.is_empty());
    assert_eq!(state.near_reach, radius);
    assert_eq!(state.drawn_near_reach, radius);
    let ranker = Ranker::new(&exact.pose, &MeadowRtMargins::default());
    let mut held = 0;
    for b in blades.iter().filter(|b| !b.cleared) {
        let d = ranker.least_rank_distance(b.point);
        if d < radius * (1.0 - SLACK) {
            assert!(near.contains(&b.id), "{:?} at {d}", b.id);
            held += 1;
        } else if d > radius * (1.0 + SLACK) {
            assert!(!near.contains(&b.id), "{:?} at {d}", b.id);
        }
    }
    assert!(held > 1000, "{held}");
    assert!(gpu.drawn(NEAR_SLOTS) > 1000);
}

impl Harness {
    /// Fit both bands to histograms `hists` (near keys, far keys, drawn near
    /// slots' keys, drawn far slots' keys, in `(bin, count)` runs) and
    /// `others` drawn slots without a key per band, alone.
    fn fit(&mut self, hists: [&[(u32, u32)]; 4], others: [u32; 2]) -> RtSelectStateGpu {
        if self.rtv.select_group.is_none() {
            // An empty selection binds the selection's group.
            self.frame(&RtInputs::default(), RtPose::default(), None, 1.0);
        }
        let hists_start = (RT_STATE_SIZE / 4) as usize;
        let mut words = vec![0u32; hists_start + 4 * RT_SELECT_BINS as usize];
        let drawn_other = std::mem::offset_of!(RtSelectStateGpu, drawn_other) / 4;
        words[drawn_other..drawn_other + 2].copy_from_slice(&others);
        for (h, runs) in hists.iter().enumerate() {
            for &(bin, count) in runs.iter() {
                words[hists_start + h * RT_SELECT_BINS as usize + bin as usize] = count;
            }
        }
        self.queue
            .write_buffer(&self.rtv.select.scratch, 0, bytemuck::cast_slice(&words));
        self.run(|rtv, encoder, passes, _, _| {
            let group = &rtv.select_group.as_ref().unwrap().1;
            passes.record(
                encoder,
                "fit",
                group,
                &[(RtPass::FitNear, 1), (RtPass::FitFar, 1)],
            );
        });
        self.raw_state()
    }
}

/// The fit keeps a bin of slack below the first bin that overflows, counts
/// drawn slots past the boundary against the capacity, and reports a far
/// band cut inside the near reach.
#[test]
fn fits_leave_slack_and_count_drawn_slots() {
    let scene = scene();
    let Some(mut gpu) = harness(&scene) else {
        return;
    };
    let near = RT_NEAR_MAX_BLADES;
    let far = RT_FAR_MAX_BLADES;
    // Everything fits.
    let roomy = gpu.fit([&[(10, 100)], &[(0, 50), (900, 100)], &[], &[]], [0, 0]);
    assert_eq!(roomy.boundary, [RT_SELECT_BINS; 2]);
    assert_eq!(f32::from_bits(roomy.limit[0]), RT_NEAR_DISTANCE);
    assert_eq!(f32::from_bits(roomy.limit[1]), UNBOUNDED_BUDGET);
    assert_eq!(roomy.selected, [100, 150]);
    // Bin 200 overflows: the boundary is the bin before it.
    let tight = gpu.fit(
        [
            &[(100, near - 1000), (199, 400), (200, 2000)],
            &[(0, 1000), (5000, far - 2000), (5001, 5000)],
            &[],
            &[],
        ],
        [0, 0],
    );
    assert_eq!(tight.boundary, [199, 5000]);
    assert_eq!(tight.selected, [near - 1000, 1000]);
    assert_eq!(tight.flags & 1, 0);
    // Drawn slots past a boundary hold their slots: with 3000 drawn blades
    // keyed in bin 300 and 500 without keys, bin 100 no longer fits.
    let pinned = gpu.fit(
        [
            &[(50, near - 4000), (100, 1000), (300, 3000)],
            &[],
            &[(300, 3000)],
            &[],
        ],
        [500, 0],
    );
    assert_eq!(pinned.boundary[0], 99);
    assert_eq!(pinned.selected[0], near - 4000);
    // More far blades inside the near reach than slots: the cut.
    let cut = gpu.fit([&[], &[(0, far + 10)], &[], &[]], [0, 0]);
    assert_eq!(cut.boundary[1], 0);
    assert_eq!(f32::from_bits(cut.limit[1]), 0.0);
    assert_ne!(cut.flags & 1, 0, "{cut:?}");
}

/// Drawn blades that die (a clearing disc grows over them, or the cover
/// thins under them) keep their slots for the frame and count against the
/// capacity without a key; the next frame they leave, with no blade waiting
/// for a slot and the slots' identity kept. While they hold their slots
/// undrawn, their vertices all sit at one finite point facing up, so
/// Solari drops history anchored on them; a vacant slot's are NaN, or
/// collapsed from its last owner.
#[test]
fn dying_drawn_blades_hold_their_slots_without_a_key() {
    let scene = scene();
    let Some(mut gpu) = harness(&scene) else {
        return;
    };
    let pose = RtPose::looking(scene.eye(Vec2::new(-2.0, -1.0)), 120.0, -10.0, 40.0);
    gpu.frame(&scene.inputs, pose, None, 1.0);
    // A second frame draws the blades the first placed.
    gpu.frame(&scene.inputs, pose, None, 1.0);
    let identity = gpu.rtv.selector.identity;
    assert!(gpu.drawn(NEAR_SLOTS) > 1000);
    // A disc clears the ground under the eye.
    let disc = TrunkDisc::new(Vec2::new(-2.0, -1.0), 3.0, 0.5);
    let mut trunks = (*scene.inputs.trunks).clone();
    let under = scene
        .inputs
        .placements
        .iter()
        .find(|(_, p)| (Vec2::new(-2.0, -1.0) - p.centre).abs().max_element() < 4.0)
        .unwrap()
        .0;
    trunks[under as usize] = PatchTrunkSlot::from_discs(&[disc]);
    gpu.set_trunks(&trunks);
    let cleared = RtInputs {
        trunks: Arc::new(trunks),
        ..scene.inputs.clone()
    };
    let next = gpu.frame(&cleared, pose, None, 1.0);
    assert!(next.is_some(), "a changed clearing reselects");
    let state = gpu.state();
    let raw = gpu.raw_state();
    assert!(raw.drawn_other[0] > 100, "{raw:?}");
    assert!(state.pinned >= raw.drawn_other[0], "{state:?}");
    assert_eq!(state.dropped, 0);
    assert_eq!(gpu.rtv.selector.identity, identity);
    let collapsed = |vertices: &[(Vec3, Vec3)]| {
        vertices[0].0.is_finite() && vertices.iter().all(|(p, _)| *p == vertices[0].0)
    };
    let (mut undrawn, mut vacant) = (0, 0);
    for (slot, ((owner, drawn), vertices)) in
        gpu.slots().into_iter().zip(gpu.slot_vertices()).enumerate()
    {
        match (owner, drawn) {
            (Some(_), true) => {
                assert!(vertices.iter().all(|(p, _)| p.is_finite()), "slot {slot}");
                assert!(!collapsed(&vertices), "slot {slot}");
            }
            (Some(_), false) => {
                undrawn += 1;
                assert!(
                    collapsed(&vertices) && vertices.iter().all(|(_, n)| *n == Vec3::Y),
                    "slot {slot}: {vertices:?}"
                );
            }
            (None, _) => {
                vacant += 1;
                assert!(
                    vertices.iter().all(|(p, _)| p.is_nan()) || collapsed(&vertices),
                    "slot {slot}: {vertices:?}"
                );
            }
        }
    }
    assert!(undrawn > 100, "{undrawn}");
    assert!(vacant > 50, "{vacant}");
    // The cover thins to nothing over half the field.
    let dims = UVec2::splat(17);
    let texels = (0..dims.x * dims.y)
        .map(|i| CoverTexel::new(if i % dims.x < 8 { 0.0 } else { 1.0 }, 1.0, 0.0))
        .collect();
    let half = TILES_PER_SIDE as f32 * 4.0;
    let cover = MeadowGrid::new(Vec2::splat(-half), Vec2::splat(half), dims, texels);
    gpu.set_cover(&cover);
    let thinned = RtInputs {
        cover: Some(Arc::new(cover)),
        ..cleared
    };
    for _ in 0..3 {
        gpu.frame(&thinned, pose, None, 1.0);
        assert_eq!(gpu.state().dropped, 0);
    }
    assert_eq!(
        gpu.rtv.selector.identity, identity,
        "cover changes keep the identity"
    );
    // Every resident blade now stands on cover.
    let points: HashMap<[u32; 2], Vec3> = scene
        .blades(&scene.inputs)
        .into_iter()
        .map(|b| (b.id, b.point))
        .collect();
    for id in gpu.owners().iter().flatten() {
        assert!(
            points[id].x >= -half + 2.0 * half * 7.0 / 16.0 - 1e-3,
            "{id:?}"
        );
    }
}

/// After a frame whose inputs were missing, the selection starts over: it
/// clears the slots, and its casters draw the frame it lands, as do the
/// exact-only reference's when it switches on.
#[test]
fn cleared_slots_draw_at_once() {
    let scene = scene();
    let Some(mut gpu) = harness(&scene) else {
        return;
    };
    let pose = RtPose::looking(scene.eye(Vec2::new(3.0, 2.0)), 300.0, -10.0, 40.0);
    gpu.frame(&scene.inputs, pose, None, 1.0);
    assert!(gpu.drawn(ALL_SLOTS) > 10_000, "{}", gpu.drawn(ALL_SLOTS));
    let before = gpu.owners();
    let identity = gpu.rtv.selector.identity;
    gpu.pad();
    let moved = RtPose::looking(scene.eye(Vec2::new(-20.0, 10.0)), 30.0, -10.0, 40.0);
    let reset = gpu.frame(&scene.inputs, moved, None, 1.0);
    assert!(reset.unwrap().clear);
    assert_ne!(gpu.rtv.selector.identity, identity);
    assert!(gpu.drawn(ALL_SLOTS) > 10_000);
    let state = gpu.state();
    let owners = gpu.owners();
    assert_eq!(
        owners.iter().flatten().count(),
        (state.near_selected + state.far_selected) as usize
    );
    assert_ne!(owners, before);
    let exact = gpu.frame(&scene.inputs, moved, Some(3.0), 1.0);
    assert!(exact.unwrap().clear);
    assert!(gpu.drawn(ALL_SLOTS) > 100, "{}", gpu.drawn(ALL_SLOTS));
}

/// A scan of more chunks than a dispatch row holds selects from all of
/// them.
#[test]
fn scans_past_a_dispatch_row() {
    let half = 40.0;
    let step = 0.3;
    let side = (2.0 * half / step) as u32;
    let placements: Vec<(u32, PatchPlacement)> = (0..side * side)
        .map(|i| {
            let centre = Vec2::new((i % side) as f32, (i / side) as f32) * step - half;
            (
                i,
                PatchPlacement::tile(centre, 0.1, 4, hash_u32(i + 7), 1.0, 0),
            )
        })
        .collect();
    let heights = MeadowGrid::new(
        Vec2::splat(-half - 1.0),
        Vec2::splat(half + 1.0),
        UVec2::splat(2),
        vec![0.0; 4],
    );
    let inputs = RtInputs {
        placements: Arc::new(placements),
        trunks: Arc::new(vec![PatchTrunkSlot::default(); (side * side) as usize]),
        heights: Some(Arc::new(heights)),
        ..Default::default()
    };
    let scene = Scene {
        grown: inputs.clone(),
        inputs,
        disc: TrunkDisc::new(Vec2::splat(1e4), 0.1, 0.0),
    };
    let Some(mut gpu) = harness(&scene) else {
        return;
    };
    // Looking along +Z, toward the patches the last chunks hold.
    let pose = RtPose::looking(scene.eye(Vec2::ZERO), 180.0, -10.0, 40.0);
    let select = gpu.frame(&scene.inputs, pose, None, 1.0).unwrap();
    assert!(
        select.scan.chunks.len() > RT_SCAN_ROW as usize,
        "{}",
        select.scan.chunks.len()
    );
    let state = gpu.state();
    let far = band(&gpu.owners(), false);
    let ranker = Ranker::new(&pose, &MeadowRtMargins::default());
    let mut past_row = 0;
    for (k, chunk) in select.scan.chunks.iter().enumerate() {
        let b = &scene.blades(&RtInputs {
            placements: Arc::new(vec![scene.inputs.placements[chunk[0] as usize]]),
            ..scene.inputs.clone()
        })[0];
        let key = least_budget(
            ranker.least_rank_distance(b.point),
            state.near_reach,
            b.rank,
        );
        if b.point.distance(pose.eye) <= FAR_DISTANCE && key < state.far_budget * (1.0 - SLACK) {
            assert!(far.contains(&b.id), "chunk {k}: {:?}", b.id);
            past_row += usize::from(k >= RT_SCAN_ROW as usize);
        }
    }
    assert!(past_row > 100, "{past_row}");
}

/// A variant's selection lives with its buffers through frames it is
/// missing from the extract, as without a registry, and both are freed
/// once it has been inactive for [`RT_INACTIVE_FREE_FRAMES`].
#[test]
fn selections_live_as_long_as_their_buffers() {
    let Some((device, queue)) = device() else {
        return;
    };
    let id = MeadowVariantId(7);
    let mut rt = MeadowRtBuffers::default();
    rt.by_variant.insert(
        id,
        RtVariantBuffers::new(device, queue, make_rt_band_indices(device)),
    );
    let identity = rt.by_variant[&id].selector.identity;
    let absent = MeadowExtractedVariants::default();
    for _ in 1..RT_INACTIVE_FREE_FRAMES {
        rt.free_inactive(&absent);
    }
    assert_eq!(rt.by_variant[&id].selector.identity, identity);
    rt.free_inactive(&absent);
    assert!(rt.by_variant.is_empty());
}
