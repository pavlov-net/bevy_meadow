//! Cover-field demo. Tile patches fill a square of ground; a procedural
//! cover field thins them from a dense lawn on one side to bare ground on
//! the other, two palette entries blend across it, and a clearing profile
//! caps the grass below a shrub's drooping skirt and branch. Press Space to
//! cycle clumpiness.

use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;
use bevy_meadow::{
    CoverTexel, MeadowClearingProfile, MeadowCoverField, MeadowGrid, MeadowLodCurve,
    MeadowMaterial, MeadowPalette, MeadowPatch, MeadowPlugin, MeadowVariant, MeadowVariantRegistry,
    MeadowViewer, PatchPlacement, TrunkDisc, upload_clearing_profiles, upload_placements,
    upload_trunk_slots,
};

/// Tiles per side and their edge length, metres.
const TILES: i32 = 16;
const TILE_SIZE: f32 = 8.0;
const BLADES_PER_M2: f32 = 200.0;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, MeadowPlugin, FreeCameraPlugin))
        .insert_resource(ClearColor(Color::srgb(0.55, 0.72, 0.95)))
        .add_systems(Startup, (setup_scene, setup_meadow))
        .add_systems(Update, (update_meadow_viewer, cycle_clumpiness))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 25.0, 70.0).looking_at(Vec3::ZERO, Vec3::Y),
        FreeCamera::default(),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.35, 0.0)),
    ));
    let half = TILES as f32 * TILE_SIZE * 0.5;
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(half * 1.1)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.24, 0.2, 0.13),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.02, 0.0),
    ));
    // The shrub the clearing sits under: a high crown over a ring of
    // drooping skirt, and a branch lying on the ground toward +X.
    let foliage = materials.add(Color::srgb(0.08, 0.16, 0.05));
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(1.2))),
        MeshMaterial3d(foliage.clone()),
        Transform::from_xyz(0.0, 1.9, 0.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Torus::new(0.8, 1.6))),
        MeshMaterial3d(foliage.clone()),
        Transform::from_xyz(0.0, 0.45, 0.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.3))),
        MeshMaterial3d(foliage),
        Transform::from_xyz(1.2, 0.15, 0.0),
    ));
}

fn setup_meadow(
    mut commands: Commands,
    mut registry: ResMut<MeadowVariantRegistry>,
    mut meadow_materials: ResMut<Assets<MeadowMaterial>>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    mut images: ResMut<Assets<Image>>,
    mut cover: ResMut<MeadowCoverField>,
) {
    let lush = MeadowPalette::default();
    let dry = MeadowPalette {
        live: Vec3::new(0.2, 0.17, 0.09),
        dry: Vec3::new(0.2, 0.17, 0.09),
        ..lush
    };
    let variant = MeadowVariant {
        blades_per_m2: BLADES_PER_M2,
        palettes: vec![lush, dry],
        lod: MeadowLodCurve {
            full_distance: 50.0,
            max_view_distance: 120.0,
            ..default()
        },
        ..default()
    };
    let id = registry.register(variant, &mut meadow_materials, &mut shader_buffers);

    // A 2 m cover grid over the tiled square: density falls from 1 at -X to
    // 0 at +X, the palette index runs from lush (0) to dry (1).
    let half = TILES as f32 * TILE_SIZE * 0.5;
    let dims = UVec2::splat(half as u32 + 1);
    let texels = (0..dims.x * dims.y)
        .map(|i| {
            let u = (i % dims.x) as f32 / (dims.x - 1) as f32;
            CoverTexel::new(1.0 - u, 1.0, u)
        })
        .collect();
    cover.set_grid(
        MeadowGrid::new(Vec2::splat(-half), Vec2::splat(half), dims, texels),
        &mut images,
    );

    let mut placements = Vec::new();
    for z in 0..TILES {
        for x in 0..TILES {
            let centre = Vec2::new(x as f32 + 0.5, z as f32 + 0.5) * TILE_SIZE - half;
            // The tile's densest texel: the cover falls with +X.
            let ceiling =
                (1.0 - (centre.x - TILE_SIZE * 0.5 + half) / (2.0 * half)).clamp(0.0, 1.0);
            let blades = (TILE_SIZE * TILE_SIZE * BLADES_PER_M2 * ceiling).ceil() as u32;
            let seed = (z * TILES + x) as u32;
            placements.push(PatchPlacement::tile(
                centre,
                TILE_SIZE * 0.5,
                blades,
                seed.wrapping_mul(0x9E37_79B9),
                ceiling,
                0,
            ));
        }
    }
    upload_placements(
        id,
        &placements,
        &mut registry,
        &meadow_materials,
        &mut shader_buffers,
    );

    // A shrub at the origin, bare at its stem. Its clearing profile, in the
    // world's frame on the ground at y = 0, is the underside of its foliage
    // less 5 cm in 8 directions, out to the skirt's rim at 1.6 m: grass under
    // the high crown keeps its height and stays short under the skirt and the
    // branch. The stem's edge wobbles more than the skirt's. The disc reaches
    // the four tiles that meet there.
    let underside = [0.65, 0.67, 0.74, 0.87, 0.12, 0.02, 0.05, 0.4];
    let sectors = 8;
    let heights = underside
        .iter()
        .enumerate()
        .flat_map(|(ring, h)| {
            (0..sectors).map(move |sector| {
                let branch = sector == 0 && (4..7).contains(&ring);
                if branch { 0.0 } else { *h }
            })
        })
        .collect();
    let profile = MeadowClearingProfile {
        radius: 1.6,
        rings: underside.len() as u32,
        sectors,
        heights,
    };
    let ids = upload_clearing_profiles(id, &[profile], &mut registry, &mut images);
    let shrub = TrunkDisc::new(Vec2::ZERO, 0.15, 0.2)
        .with_profile(ids[0], 0.0, Vec2::X, 1.0, 1.0)
        .with_edge_noise(0.35)
        .with_profile_noise(0.1);
    let slots: Vec<_> = placements
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            p.centre.distance(shrub.centre()) < shrub.outer_radius() + p.outer_radius()
        })
        .map(|(index, _)| (index as u32, vec![shrub]))
        .collect();
    upload_trunk_slots(
        id,
        &slots,
        &mut registry,
        &meadow_materials,
        &mut shader_buffers,
    );

    for (index, p) in placements.iter().enumerate() {
        if p.blade_count == 0 {
            continue;
        }
        commands.spawn((
            MeadowPatch {
                variant: id,
                patch_index: index as u32,
                blade_count: p.blade_count,
                centre: p.centre,
                radius: p.radius,
            },
            Transform::from_xyz(p.centre.x, 0.0, p.centre.y),
        ));
    }
}

fn cycle_clumpiness(keys: Res<ButtonInput<KeyCode>>, mut cover: ResMut<MeadowCoverField>) {
    if keys.just_pressed(KeyCode::Space) {
        cover.survival.clumpiness = (cover.survival.clumpiness + 0.25) % 1.25;
        info!("clumpiness {}", cover.survival.clumpiness);
    }
}

fn update_meadow_viewer(
    camera: Query<&GlobalTransform, With<Camera3d>>,
    mut viewer: ResMut<MeadowViewer>,
) {
    let Ok(transform) = camera.single() else {
        return;
    };
    let eye = transform.translation();
    let forward = transform.forward();
    viewer.eye = eye;
    viewer.depth_plane = Vec4::new(forward.x, forward.y, forward.z, -forward.dot(eye));
}
