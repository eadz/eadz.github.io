// Volumetric atmosphere: height fog, raymarched lamp in-scatter with analytic shadow volumes,
// and screen-space crepuscular rays. Runs in HDR, before bloom and tonemapping.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View

const MAX_LAMPS: u32 = 4u;
const MAX_OCCLUDERS: u32 = 24u;
const KEY_RAY_SAMPLES: i32 = 28;

struct Lamp {
    // xyz = world position, w = in-scatter intensity
    position: vec4<f32>,
    // rgb = colour, w = falloff radius
    color: vec4<f32>,
}

struct Occluder {
    // xy = centre, zw = half extents
    rect: vec4<f32>,
    // x = depth, y = lamp index
    info: vec4<f32>,
}

struct Atmosphere {
    // rgb = haze colour, w = density at the reference height
    fog_color: vec4<f32>,
    // x = reference height, y = height falloff, z = fog front depth, w = raymarch back depth
    fog_shape: vec4<f32>,
    // xy = key light screen position, z = ray strength, w = sky glow strength
    key_light: vec4<f32>,
    key_color: vec4<f32>,
    // x = backdrop (0 none, 1 night city, 2 daylight ice shelf), y = height of the ground
    // plane, z = ground brightness,
    // w = how far the haze reaches across open sky
    sky: vec4<f32>,
    // xy = screen position of the planet, z = radius in screen heights, w = brightness
    planet: vec4<f32>,
    // x = seconds, y = raymarch steps, z = lamp count, w = occluder count
    counts: vec4<f32>,
    lamps: array<Lamp, MAX_LAMPS>,
    occluders: array<Occluder, MAX_OCCLUDERS>,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var<uniform> atm: Atmosphere;
@group(0) @binding(4) var<uniform> view: View;

fn hash13(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i), hash13(i + vec3(1.0, 0.0, 0.0)), s.x),
            mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), s.x), s.y),
        mix(mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), s.x),
            mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), s.x), s.y),
        s.z);
}

fn interleaved_gradient_noise(pixel: vec2<f32>, frame: u32) -> f32 {
    let p = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Exponential height fog. Denser towards the abyss, thinning out with altitude.
fn height_density(y: f32) -> f32 {
    let e = clamp(-(y - atm.fog_shape.x) * atm.fog_shape.y, -8.0, 2.2);
    return atm.fog_color.w * exp(e);
}

// Closed-form optical depth of the height fog along a segment.
fn optical_depth(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let len = distance(a, b);
    let k = atm.fog_shape.y;
    let dy = (b.y - a.y) * k;
    var integral = 1.0;
    if abs(dy) > 1e-3 {
        integral = (1.0 - exp(-clamp(dy, -8.0, 8.0))) / dy;
    }
    return height_density(a.y) * len * min(integral, 12.0);
}

struct Shadowing {
    // How much of each lamp reaches the sample, given the slabs standing in the way.
    lamps: array<f32, MAX_LAMPS>,
    // How far the shadow here has condensed towards something solid: zero at the slab,
    // one where the shadow volume meets the gameplay plane and becomes walkable.
    gloom: f32,
}

fn lamp_shadowing(p: vec3<f32>) -> Shadowing {
    var vis = array<f32, MAX_LAMPS>(1.0, 1.0, 1.0, 1.0);
    var gloom = 0.0;
    let count = u32(atm.counts.w);
    for (var i = 0u; i < count; i++) {
        let occ = atm.occluders[i];
        let depth = occ.info.x;
        if p.z <= depth + 0.25 {
            continue;
        }
        let li = u32(occ.info.y);
        let lamp = atm.lamps[li].position.xyz;
        // Project the sample back through the lamp onto the occluder's plane.
        let s = (depth - lamp.z) / (p.z - lamp.z);
        let q = lamp.xy + (p.xy - lamp.xy) * s;
        let inside = occ.rect.zw - abs(q - occ.rect.xy);
        let penumbra = (0.10 + 0.012 * (p.z - depth)) * s;
        let shade = smoothstep(-penumbra, penumbra, min(inside.x, inside.y));
        vis[li] *= 1.0 - shade;
        let condensed = smoothstep(depth - 4.0, -2.0, p.z) * (1.0 - smoothstep(1.0, 6.0, p.z));
        gloom = max(gloom, shade * condensed);
    }
    return Shadowing(vis, gloom);
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
}

fn lamp_inscatter(p: vec3<f32>, ray: vec3<f32>, vis: array<f32, MAX_LAMPS>) -> vec3<f32> {
    var light = vec3(0.0);
    let count = u32(atm.counts.z);
    for (var i = 0u; i < count; i++) {
        let lamp = atm.lamps[i];
        let v = p - lamp.position.xyz;
        let d2 = dot(v, v);
        let dir = v * inverseSqrt(d2);
        let radius = lamp.color.w;
        let falloff = lamp.position.w / (1.0 + d2 / (radius * radius));
        // The lamps are spots aimed at the gameplay plane (+z).
        let cone = smoothstep(0.15, 0.6, dir.z);
        let phase = henyey_greenstein(dot(dir, -ray), 0.62);
        light += lamp.color.rgb * (falloff * cone * phase * vis[i]);
    }
    return light;
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.xx + q.yz) * q.zy);
}

// One octave of city: a lattice of blocks, each with at most one light. `spot` is the
// light's radius in cell units along x and z, already widened to at least a pixel.
fn city_lights(p: vec2<f32>, cell_size: f32, spot: vec2<f32>) -> vec3<f32> {
    let g = p / cell_size;
    let cell = floor(g);
    let rnd = hash22(cell);
    let rnd2 = hash22(cell + 71.3);
    // Districts: broad patches of dense and sparse city, cut by lit avenues.
    let district = value_noise(vec3(cell * cell_size * 0.0016, 3.0));
    let avenue = f32(i32(cell.x) % 9 == 0 || i32(cell.y) % 7 == 0);
    let chance = smoothstep(0.38, 0.7, district) * 0.5 + avenue * 0.3 * smoothstep(0.25, 0.5, district);
    if rnd.x > chance {
        return vec3(0.0);
    }
    let at = 0.25 + 0.5 * rnd2;
    let d = (fract(g) - at) / spot;
    // Only half-conserve energy as lights widen with distance: strictly correct would melt
    // the far city into an even glow, and the sparkle is the point.
    let glow = exp(-dot(d, d)) * sqrt((0.16 * 0.16) / (spot.x * spot.y));
    let warm = vec3(1.0, 0.56, 0.2);
    let cold = vec3(0.62, 0.8, 1.0);
    let tint = mix(warm, cold, step(0.72, rnd.y));
    let twinkle = 0.8 + 0.2 * sin(atm.counts.x * (1.0 + rnd2.x * 3.0) + rnd.y * 40.0);
    let hot = 1.0 + 9.0 * step(0.94, rnd2.x);
    return tint * glow * (0.3 + 1.7 * rnd2.y * rnd2.y) * hot * twinkle;
}

// What lies beyond the architecture on open-air levels: stars, a planet rising behind the
// horizon, and the city far below. Stars and planet are effectively at infinity and the
// camera never rotates, so they are fixed in screen space; the city is a real ground plane
// and slides with true parallax.
fn exterior_sky(ray: vec3<f32>, uv: vec2<f32>, pixel: vec2<f32>, dims: vec2<f32>, cam: vec3<f32>) -> vec3<f32> {
    let aspect = dims.x / dims.y;
    let up = ray.y;
    var color = mix(vec3(0.020, 0.034, 0.060), vec3(0.002, 0.004, 0.010), smoothstep(0.0, 0.16, up));

    // Stars, a pixel or so wide, thinning out into the horizon haze.
    let star_cell = floor(pixel / 9.0);
    let star = hash22(star_cell);
    let star_at = (star_cell + 0.2 + 0.6 * hash22(star_cell + 17.0)) * 9.0;
    let star_d = (pixel - star_at) / (0.6 + 0.9 * star.y);
    let star_glow = exp(-dot(star_d, star_d)) * pow(star.x, 9.0) * 2.5;
    var heavens = vec3(0.85, 0.92, 1.0) * star_glow;

    // The planet: a lit limb, a dark body that blots out the stars, and a thin atmosphere.
    let rel = (uv - atm.planet.xy) * vec2(aspect, 1.0) / atm.planet.z;
    let r2 = dot(rel, rel);
    if r2 < 1.0 {
        let normal = vec3(rel.x, -rel.y, sqrt(1.0 - r2));
        let sun = normalize(vec3(-0.78, 0.5, -0.12));
        let lit = smoothstep(-0.05, 0.35, dot(normal, sun));
        let bands = value_noise(vec3(rel.y * 9.0 + value_noise(vec3(rel * 3.0, 1.0)) * 1.5, rel.x * 0.7, 5.0));
        let surface = mix(vec3(0.20, 0.30, 0.44), vec3(0.62, 0.74, 0.86), bands);
        let limb = pow(1.0 - normal.z, 3.0);
        heavens = surface * lit * (0.55 + 0.45 * normal.z) + vec3(0.35, 0.55, 0.9) * limb * (0.15 + lit);
        heavens += vec3(0.010, 0.016, 0.028);
    } else {
        let halo = exp(-(sqrt(r2) - 1.0) * 14.0);
        heavens += vec3(0.25, 0.42, 0.75) * halo * 0.35;
    }
    color += heavens * atm.planet.w * smoothstep(-0.004, 0.035, up);

    // Light pollution banked up along the horizon.
    color += vec3(0.10, 0.075, 0.05) * exp(-abs(up) * 38.0) * atm.sky.z;

    if up < -0.002 {
        let t = (atm.sky.y - cam.y) / ray.y;
        let p = (cam + ray * t).xz;
        // World-space size of a pixel on the plane: t * pixel angle across the view,
        // stretched by 1 / sin(elevation) along it.
        let pixel_angle = 2.0 / (view.clip_from_view[1][1] * dims.y);
        let across = t * pixel_angle;
        let along = across / max(-ray.y, 0.01);
        let footprint = sqrt(across * along);
        let lod = max(log2(footprint * 3.0 / 12.0), 0.0);
        let cell = 12.0 * exp2(floor(lod));
        // Keep every light at least ~0.8 px in both screen axes so none of them shimmer.
        let spot_near = max(vec2(0.16), 0.8 * vec2(across, along) / cell);
        let spot_far = max(vec2(0.16), 0.4 * vec2(across, along) / cell);
        let lights = mix(city_lights(p, cell, spot_near), city_lights(p, cell * 2.0, spot_far), fract(lod));
        let reach = exp(-t / 5200.0) * smoothstep(0.002, 0.02, -up);
        // A faint sodium haze hangs over the streets, which also keeps a black figure on the
        // ledge readable against the city.
        color = vec3(0.016, 0.017, 0.022) * atm.sky.z + lights * atm.sky.z * 9.0 * reach
            + vec3(0.06, 0.045, 0.03) * atm.sky.z * exp(-t / 2600.0) * 0.4;
    }
    return color;
}

// Worley noise: distance to the nearest and second-nearest feature point, plus a cell id.
fn voronoi(g: vec2<f32>) -> vec3<f32> {
    let cell = floor(g);
    var f1 = 8.0;
    var f2 = 8.0;
    var id = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = cell + vec2(f32(x), f32(y));
            let rnd = hash22(c);
            let d = distance(g, c + rnd);
            if d < f1 {
                f2 = f1;
                f1 = d;
                id = rnd.x;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    return vec3(f1, f2, id);
}

// Daylight backdrop: a hazy polar sky, the ghost of the planet, and a frozen sea of floes
// split by dark leads of open water, glittering where the sun catches it.
fn ice_shelf_sky(ray: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>, cam: vec3<f32>) -> vec3<f32> {
    let aspect = dims.x / dims.y;
    let up = ray.y;
    let horizon = vec3(0.62, 0.74, 0.88);
    let to_sun = (atm.key_light.xy - uv) * vec2(aspect, 1.0);
    let sun_glow = vec3(1.0, 0.94, 0.84) * 0.3 / (1.0 + dot(to_sun, to_sun) * 9.0);
    var color = mix(horizon, vec3(0.05, 0.13, 0.32), smoothstep(-0.01, 0.10, up)) + sun_glow;

    // The planet is still up there, washed out to a pale limb by the daylight.
    let rel = (uv - atm.planet.xy) * vec2(aspect, 1.0) / atm.planet.z;
    let r2 = dot(rel, rel);
    if r2 < 1.0 {
        let normal = vec3(rel.x, -rel.y, sqrt(1.0 - r2));
        let lit = smoothstep(-0.05, 0.4, dot(normal, normalize(vec3(-0.78, 0.5, -0.12))));
        color += vec3(0.75, 0.85, 1.0) * lit * (0.4 + 0.6 * normal.z) * atm.planet.w
            * smoothstep(0.0, 0.05, up);
    }

    if up < -0.002 {
        let t = (atm.sky.y - cam.y) / ray.y;
        let p = (cam + ray * t).xz;
        let pixel_angle = 2.0 / (view.clip_from_view[1][1] * dims.y);
        let across = t * pixel_angle;
        let footprint = across / sqrt(max(-ray.y, 0.01));

        // Floes. Leads between them are widened to at least a pixel, then faded out where
        // the floes themselves go sub-pixel.
        let scale = 1.0 / 34.0;
        // Warp the lattice so the floes are ragged plates, not a honeycomb.
        let warp = vec2(value_noise(vec3(p * 0.013, 11.0)), value_noise(vec3(p * 0.013, 23.0))) - 0.5;
        let v = voronoi(p * scale + warp * 1.6);
        let soften = footprint * scale;
        let lead = (1.0 - smoothstep(0.06, 0.24 + soften * 1.5, v.y - v.x))
            * (1.0 - smoothstep(0.25, 0.7, soften));
        // A few wide channels of open water wander through the pack.
        let wander = value_noise(vec3(p * 0.0021, 7.0));
        let channel = 1.0 - smoothstep(0.040, 0.075, abs(wander - 0.5));
        let water = max(channel, lead * 0.9);

        let drift = value_noise(vec3(p * 0.045, 2.0));
        let ridges = value_noise(vec3(p * 0.19, 4.0));
        var ice = vec3(0.80, 0.88, 0.97) * (0.70 + 0.30 * v.z) * (0.8 + 0.4 * drift);
        ice += vec3(0.16) * smoothstep(0.6, 0.85, ridges) * (1.0 - smoothstep(0.5, 2.0, footprint));
        // Sun glitter: needle-sharp up close, melting into a sheen with distance.
        let spark = hash22(floor(p * 1.3));
        let twinkle = 0.5 + 0.5 * sin(atm.counts.x * (2.0 + spark.y * 5.0) + spark.y * 60.0);
        let near_glitter = step(0.965, spark.x) * twinkle * 9.0 * (1.0 - smoothstep(0.4, 1.4, footprint * 1.3));
        let far_sheen = 0.22 * smoothstep(0.4, 1.4, footprint * 1.3) * (0.6 + 0.4 * drift);
        ice += vec3(1.0, 0.97, 0.9) * (near_glitter + far_sheen);

        let fresnel = pow(1.0 + ray.y, 7.0);
        let sea = mix(vec3(0.030, 0.10, 0.20), vec3(0.34, 0.50, 0.70), fresnel);
        let ground = mix(ice, sea, water) * atm.sky.z;
        color = mix(horizon + sun_glow, ground, exp(-t / 5200.0));
    }
    return color;
}

// Snow on the wind, in two screen-space layers that slide with the camera for parallax.
fn snowfall(pixel: vec2<f32>, cam: vec3<f32>) -> f32 {
    var snow = 0.0;
    for (var layer = 0; layer < 2; layer++) {
        let l = f32(layer);
        let cell_size = 46.0 + 38.0 * l;
        let speed = 1.0 + 0.7 * l;
        let q = pixel + vec2(cam.x * (5.0 + 4.0 * l), -cam.y * (5.0 + 4.0 * l))
            - atm.counts.x * vec2(26.0, 48.0) * speed;
        let cell = floor(q / cell_size);
        let rnd = hash22(cell + 19.0 * l);
        let sway = sin(atm.counts.x * (0.8 + rnd.y) + rnd.x * 30.0) * 5.0;
        let at = (cell + 0.2 + 0.6 * rnd) * cell_size + vec2(sway, 0.0);
        let d = (q - at) / (1.1 + 0.9 * l);
        snow += exp(-dot(d, d)) * step(0.45, rnd.y) * (0.35 + 0.3 * l);
    }
    return snow;
}

fn view_distance(depth: f32) -> f32 {
    // Infinite reverse-z perspective: depth = near / view_z.
    return view.clip_from_view[3][2] / max(depth, 1e-6);
}

// Cheap estimate of how hazy (and therefore how luminous) the scene is at a given depth.
fn haze_amount(depth: f32) -> f32 {
    let reach = max(view_distance(depth) - (view.world_position.z - atm.fog_shape.z), 0.0);
    return 1.0 - exp(-reach * atm.fog_color.w * 1.6);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let dims = vec2<f32>(textureDimensions(depth_texture));
    var scene = textureSampleLevel(scene_texture, scene_sampler, in.uv, 0.0);
    let depth = textureLoad(depth_texture, pixel, 0);
    let jitter = interleaved_gradient_noise(in.position.xy, view.frame_count);

    // Reconstruct the view ray.
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let near_h = view.world_from_clip * vec4(ndc, 1.0, 1.0);
    let cam = view.world_position;
    let ray = normalize(near_h.xyz / near_h.w - cam);
    let cos_forward = dot(ray, -normalize(view.world_from_view[2].xyz));
    var t_surface = 6000.0;
    var far_reach = 2500.0;
    if depth > 1e-6 {
        t_surface = view_distance(depth) / cos_forward;
    } else if atm.sky.x > 1.5 {
        scene = vec4(ice_shelf_sky(ray, in.uv, dims, cam), scene.a);
        far_reach = atm.sky.w;
    } else if atm.sky.x > 0.5 {
        scene = vec4(exterior_sky(ray, in.uv, in.position.xy, dims, cam), scene.a);
        far_reach = atm.sky.w;
    }

    // Screen-space vector to the cold key light, aspect corrected.
    let aspect = dims.x / dims.y;
    let to_key = (atm.key_light.xy - in.uv) * vec2(aspect, 1.0);
    let key_glow = atm.key_light.w / (1.0 + dot(to_key, to_key) * 5.0);
    let ambient = atm.fog_color.rgb + atm.key_color.rgb * key_glow * 0.55;

    var transmittance = 1.0;
    var inscatter = vec3(0.0);

    // Raymarch the playable slab of fog, where the lamps and their shadow volumes live.
    let t_front = (atm.fog_shape.z - cam.z) / ray.z;
    let t_back = min((atm.fog_shape.w - cam.z) / ray.z, t_surface);
    if t_back > t_front {
        let steps = i32(atm.counts.y);
        let dt = (t_back - t_front) / f32(steps);
        let drift = vec3(atm.counts.x * 0.6, atm.counts.x * -0.15, atm.counts.x * 0.2);
        for (var i = 0; i < steps; i++) {
            let p = cam + ray * (t_front + (f32(i) + jitter) * dt);
            let wisps = 0.55 + 0.9 * value_noise(p * vec3(0.06, 0.09, 0.06) + drift * 0.1);
            let shadowing = lamp_shadowing(p);
            // Condensing shadow reads as dark smoke: it soaks up light instead of scattering it.
            let gloom = shadowing.gloom;
            let density = height_density(p.y) * wisps + gloom * 0.22;
            let step_transmittance = exp(-density * dt);
            let light = (ambient + lamp_inscatter(p, ray, shadowing.lamps)) * (1.0 - gloom);
            inscatter += transmittance * (1.0 - step_transmittance) * light;
            transmittance *= step_transmittance;
        }
    }

    // Everything behind the slab gets closed-form height fog.
    if t_surface > t_back && t_back > 0.0 {
        let start = max(t_back, t_front);
        let far_t = exp(-optical_depth(cam + ray * start, cam + ray * min(t_surface, far_reach)));
        let far_ambient = atm.fog_color.rgb * 1.15 + atm.key_color.rgb * key_glow;
        inscatter += transmittance * (1.0 - far_t) * far_ambient;
        transmittance *= far_t;
    }

    // Crepuscular rays: walk towards the key light and count how much luminous haze is
    // visible along the way. Silhouettes between here and the light carve out the shafts.
    var shafts = 0.0;
    var weight_sum = 0.0;
    var weight = 1.0;
    for (var i = 0; i < KEY_RAY_SAMPLES; i++) {
        let f = (f32(i) + jitter) / f32(KEY_RAY_SAMPLES);
        let sample_uv = clamp(in.uv + (atm.key_light.xy - in.uv) * f * 0.92, vec2(0.0), vec2(1.0));
        let sample_depth = textureLoad(depth_texture, vec2<i32>(sample_uv * (dims - 1.0)), 0);
        shafts += haze_amount(sample_depth) * weight;
        weight_sum += weight;
        weight *= 0.965;
    }
    shafts = pow(shafts / weight_sum, 2.2);
    let key_falloff = 1.0 / (1.0 + dot(to_key, to_key) * 2.4);
    let rays = atm.key_color.rgb * (shafts * key_falloff * atm.key_light.z) * (1.0 - transmittance);

    var color = scene.rgb * transmittance + inscatter + rays;
    if atm.sky.x > 1.5 {
        color += vec3(0.9, 0.95, 1.0) * snowfall(in.position.xy, cam);
    }
    return vec4(color, scene.a);
}
