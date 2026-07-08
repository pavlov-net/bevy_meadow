//! Minimal meadow demo. Register one variant, scatter patches across a flat
//! biome, and fly a camera over the grass.

use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::VisibilityRange;
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::math::Rect;
use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;
use bevy_meadow::{
    MeadowLodCurve, MeadowMaterial, MeadowPatch, MeadowPlugin, MeadowVariant,
    MeadowVariantRegistry, MeadowViewer, PlacementParams, TreeDensityField, place_patches,
    upload_placements,
};

const BIOME_HALF_EXTENT: f32 = 60.0;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, MeadowPlugin, FreeCameraPlugin))
        .insert_resource(ClearColor(Color::srgb(0.55, 0.72, 0.95)))
        .add_systems(Startup, (setup_scene, setup_meadow))
        .add_systems(Update, update_meadow_viewer)
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 18.0, 42.0).looking_at(Vec3::ZERO, Vec3::Y),
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

    // Flat reference ground
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(BIOME_HALF_EXTENT * 2.2)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.22, 0.28, 0.12),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.02, 0.0),
    ));
}

fn setup_meadow(
    mut commands: Commands,
    mut registry: ResMut<MeadowVariantRegistry>,
    mut meadow_materials: ResMut<Assets<MeadowMaterial>>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let variant = MeadowVariant {
        patches_per_m2: 0.0012,
        blades_per_m2: 180.0,
        lod: MeadowLodCurve {
            full_distance: 50.0,
            max_view_distance: 120.0,
            ..default()
        },
        ..default()
    };

    let variant_id = registry.register(variant.clone(), &mut meadow_materials, &mut shader_buffers);

    let biome_rect = Rect::from_corners(
        Vec2::splat(-BIOME_HALF_EXTENT),
        Vec2::splat(BIOME_HALF_EXTENT),
    );
    let tree_density = TreeDensityField::new(biome_rect, []);
    let params = PlacementParams {
        biome_rect,
        variant_seed: variant.seed,
        patches_per_m2: variant.patches_per_m2,
        patch_radius_range: variant.patch_radius_range,
        min_patch_spacing: variant.min_patch_spacing,
        patch_edge_noise_amp: variant.patch_edge_noise_amp,
        canopy_threshold: variant.canopy_threshold,
        canopy_soft_threshold: variant.canopy_soft_threshold,
        biome_weight_threshold: variant.biome_weight_threshold,
        rim_falloff_fraction: variant.rim_falloff_fraction,
        blades_per_m2: variant.blades_per_m2,
    };

    let placements = place_patches(&params, &variant.overrides, &tree_density, |_| 1.0);
    upload_placements(
        variant_id,
        &placements,
        &mut registry,
        &meadow_materials,
        &mut shader_buffers,
    );

    let max_blade_height = variant.height_range.1;
    let visibility = VisibilityRange::abrupt(0.0, variant.lod.max_view_distance);

    for (patch_index, placement) in placements.into_iter().enumerate() {
        if placement.blade_count == 0 {
            continue;
        }
        let outer = placement.outer_radius();
        commands.spawn((
            MeadowPatch {
                variant: variant_id,
                patch_index: patch_index as u32,
                blade_count: placement.blade_count,
                centre: placement.centre,
                radius: placement.radius,
            },
            Transform::from_xyz(placement.centre.x, 0.0, placement.centre.y),
            visibility.clone(),
            Aabb {
                center: Vec3A::new(
                    placement.centre.x,
                    max_blade_height * 0.5,
                    placement.centre.y,
                ),
                half_extents: Vec3A::new(outer, max_blade_height * 0.5, outer),
            },
        ));
    }

    info!(
        "spawned meadow patches across a {} m biome",
        BIOME_HALF_EXTENT * 2.0
    );
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
    viewer.eye_xz = eye.xz();
    viewer.depth_plane = Vec4::new(forward.x, forward.y, forward.z, -forward.dot(eye));
}
