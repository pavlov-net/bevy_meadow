// Fullscreen composite: copy valid texels from the meadow-owned motion
// target into bevy's Rg16Float MV prepass texture. Fragment only — the
// vertex stage is bevy's `FullscreenShader`. Plain WGSL: nothing
// mesh-shader about it.

@group(0) @binding(0) var meadow_motion: texture_2d<f32>;

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let v = textureLoad(meadow_motion, vec2<i32>(pos.xy), 0);
    if (v.z == 0.0) {
        discard;
    }
    return v.xy;
}
