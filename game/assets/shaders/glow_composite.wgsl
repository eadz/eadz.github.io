// Adds the two glow octaves back over the HDR frame, before tonemapping.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct Glow {
    threshold: f32,
    knee: f32,
    intensity: f32,
    stretch: f32,
}

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var tight: texture_2d<f32>;
@group(0) @binding(3) var wide: texture_2d<f32>;
@group(0) @binding(4) var<uniform> glow: Glow;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSampleLevel(frame, linear_sampler, in.uv, 0.0);
    let halo = textureSampleLevel(tight, linear_sampler, in.uv, 0.0).rgb * 0.55
        + textureSampleLevel(wide, linear_sampler, in.uv, 0.0).rgb;
    return vec4(scene.rgb + halo * glow.intensity, scene.a);
}
