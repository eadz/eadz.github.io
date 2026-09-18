// 35mm film emulation, applied after tonemapping: chromatic fringing, grade, vignette, grain.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct FilmGrade {
    time: f32,
    grain: f32,
    vignette: f32,
    aberration: f32,
    fade: f32,
    _pad: vec3<f32>,
}

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var screen_sampler: sampler;
@group(0) @binding(2) var<uniform> film: FilmGrade;

fn hash12(p: vec2<f32>) -> f32 {
    var q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

// Two uniform samples summed give a triangular distribution, close enough to grain density.
fn grain_at(cell: vec2<f32>, frame: f32) -> f32 {
    let a = hash12(cell + vec2(frame * 17.13, frame * 3.71));
    let b = hash12(cell * 1.37 + vec2(frame * 5.19, frame * 23.47));
    return a + b - 1.0;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(screen_texture));
    let aspect = dims.x / dims.y;
    let centered = in.uv - 0.5;
    // Normalise so the vignette is round on the short axis of a portrait frame.
    let lens = centered * vec2(1.0, 1.0 / max(aspect, 0.001)) * min(aspect, 1.0) * 2.0;
    let r2 = dot(centered, centered);

    // Lateral chromatic aberration grows towards the edge of the lens.
    let fringe = centered * r2 * film.aberration * 8.0;
    var color = vec3(
        textureSample(screen_texture, screen_sampler, in.uv + fringe).r,
        textureSample(screen_texture, screen_sampler, in.uv).g,
        textureSample(screen_texture, screen_sampler, in.uv - fringe).b,
    );

    // Print grade: crush the toe, cool the shadows, keep highlights warm.
    let luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = pow(max(color, vec3(0.0)), vec3(1.12));
    color += vec3(-0.006, 0.0, 0.012) * (1.0 - smoothstep(0.0, 0.35, luma));
    color *= mix(vec3(1.0), vec3(1.04, 1.0, 0.94), smoothstep(0.4, 1.0, luma));

    let falloff = smoothstep(0.45, 1.75, length(lens));
    color *= 1.0 - film.vignette * falloff;

    // Grain clumps across ~1.5 px and is strongest in the mid-to-dark tones.
    let frame = floor(film.time * 24.0);
    let fine = grain_at(floor(in.position.xy / 1.5), frame);
    let coarse = grain_at(floor(in.position.xy / 3.0) + 91.7, frame);
    let response = mix(1.0, 0.4, smoothstep(0.3, 0.9, luma));
    color += (fine * 0.8 + coarse * 0.2) * film.grain * response * (0.35 + luma * 1.6);

    color *= 1.0 - film.fade;
    return vec4(max(color, vec3(0.0)), 1.0);
}
