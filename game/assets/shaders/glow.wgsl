// A small, conservative bloom for browsers whose WebGPU trips over Bevy's (which renders into
// the mip chain of one packed-float texture using blend constants). This one uses separate
// half-float textures, fixed passes and no blend state:
//
//   prefilter (1/4 res) -> blur -> downsample (1/16 res) -> wide, horizontally stretched blur
//   -> composite both octaves over the frame.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct Glow {
    threshold: f32,
    knee: f32,
    intensity: f32,
    // Horizontal tap spacing of the wide octave: the anamorphic streak.
    stretch: f32,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> glow: Glow;

fn texel() -> vec2<f32> {
    return 1.0 / vec2<f32>(textureDimensions(source));
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(source, source_sampler, uv, 0.0).rgb;
}

// Box downsample, weighted against very bright texels so a lamp a few pixels wide glows
// steadily instead of sparkling as it crosses texel boundaries.
fn downsample_at(uv: vec2<f32>, stable: bool) -> vec3<f32> {
    let t = texel();
    var sum = vec3(0.0);
    var weights = 0.0;
    for (var i = 0; i < 4; i++) {
        let offset = vec2(f32(i & 1) * 2.0 - 1.0, f32(i >> 1u) * 2.0 - 1.0);
        let c = min(tap(uv + offset * t), vec3(48.0));
        let w = select(1.0, 1.0 / (1.0 + max(c.r, max(c.g, c.b))), stable);
        sum += c * w;
        weights += w;
    }
    return sum / weights;
}

@fragment
fn prefilter(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = downsample_at(in.uv, true);
    // Soft-knee threshold.
    let brightness = max(color.r, max(color.g, color.b));
    let soft = clamp(brightness - glow.threshold + glow.knee, 0.0, 2.0 * glow.knee);
    let contribution = max(soft * soft / (4.0 * glow.knee + 1e-4), brightness - glow.threshold);
    return vec4(color * max(contribution, 0.0) / max(brightness, 1e-4), 1.0);
}

@fragment
fn downsample(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return vec4(downsample_at(in.uv, false), 1.0);
}

fn blur(uv: vec2<f32>, step: vec2<f32>) -> vec4<f32> {
    // 9-tap Gaussian, sigma = 2 taps.
    var weights = array<f32, 5>(0.2042, 0.1802, 0.1238, 0.0663, 0.0276);
    var sum = tap(uv) * weights[0];
    for (var i = 1; i < 5; i++) {
        sum += (tap(uv + step * f32(i)) + tap(uv - step * f32(i))) * weights[i];
    }
    return vec4(sum, 1.0);
}

@fragment
fn blur_h(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2(texel().x, 0.0));
}

@fragment
fn blur_h_wide(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2(texel().x * glow.stretch, 0.0));
}

@fragment
fn blur_v(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2(0.0, texel().y));
}
