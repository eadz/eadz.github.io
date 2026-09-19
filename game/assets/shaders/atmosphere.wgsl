// Volumetric atmosphere: height fog, raymarched lamp in-scatter with analytic shadow volumes,
// and screen-space crepuscular rays. Runs in HDR, before bloom and tonemapping.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View

const MAX_LAMPS: u32 = 4u;
const MAX_OCCLUDERS: u32 = 24u;
const MAX_WARDENS: u32 = 2u;
const MAX_BLINDS: u32 = 4u;
const MAX_BANDS: u32 = 4u;
const KEY_RAY_SAMPLES: i32 = 28;
// Light scattered per metre of penumbra mist.
const VEIL_COLOR: vec3<f32> = vec3<f32>(0.012, 0.018, 0.028);

struct Lamp {
    // xyz = world position, w = in-scatter intensity. For the sun, xy = its slope instead:
    // how far its light travels across the gameplay plane per metre towards the lens.
    position: vec4<f32>,
    // rgb = colour, w = falloff radius. A radius of zero marks the sun: parallel rays.
    color: vec4<f32>,
}

struct Occluder {
    // xy = centre, zw = half extents
    rect: vec4<f32>,
    // x = depth, y = lamp index, z = solidity of its shadow on the gameplay plane:
    // 1 a platform, 0.5 penumbra (solid only where two overlap), 0 erased by a Warden
    info: vec4<f32>,
}

// A searchlight on the lens side. Its beam is the pyramid from the emitter through the
// footprint rectangle on the gameplay plane: the same rectangle the collision rules use.
struct Warden {
    // xyz = emitter, w = beam strength
    emitter: vec4<f32>,
    // xy = centre of the footprint, zw = half extents
    footprint: vec4<f32>,
    // rgb = colour, w = pool strength
    color: vec4<f32>,
}

// A slab on the lens side that shades the plane from the Wardens.
struct Blind {
    // xy = centre, zw = half extents
    rect: vec4<f32>,
    // x = depth (positive)
    info: vec4<f32>,
}

// A band of weather: an upright rectangle of air, the same one the collision rules cut the
// shadow platforms with.
struct Band {
    // xy = centre, zw = half extents
    rect: vec4<f32>,
    // x = 0 for a hole in the haze, 1 for a squall
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
    // x = Warden count, y = blind count, z = samples along each beam
    warden_counts: vec4<f32>,
    wardens: array<Warden, MAX_WARDENS>,
    blinds: array<Blind, MAX_BLINDS>,
    // x = band count, y = 1 when the air away from the bands is clear, z = how much of the
    // haze is left in clear air, w = density a squall adds
    weather: vec4<f32>,
    bands: array<Band, MAX_BANDS>,
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
    // Penumbra: a single shadow of the kind that is only solid where two overlap. Drawn as
    // pale drifting mist, so it reads as a shadow's ghost and never as something to stand on.
    veil: f32,
}

fn lamp_shadowing(p: vec3<f32>) -> Shadowing {
    var vis = array<f32, MAX_LAMPS>(1.0, 1.0, 1.0, 1.0);
    var gloom = 0.0;
    // Penumbrae count half each: one alone is a faint veil, two overlapping are an umbra.
    var half_gloom = 0.0;
    let count = u32(atm.counts.w);
    for (var i = 0u; i < count; i++) {
        let occ = atm.occluders[i];
        let depth = occ.info.x;
        if p.z <= depth + 0.25 {
            continue;
        }
        let li = u32(occ.info.y);
        let lamp = atm.lamps[li].position.xyz;
        // Project the sample back onto the occluder's plane: along the sun's parallel rays,
        // which neither spread nor magnify...
        var s = 1.0;
        var q = p.xy - lamp.xy * (p.z - depth);
        if atm.lamps[li].color.w > 0.0 {
            // ...or through the lamp.
            s = (depth - lamp.z) / (p.z - lamp.z);
            q = lamp.xy + (p.xy - lamp.xy) * s;
        }
        let inside = occ.rect.zw - abs(q - occ.rect.xy);
        // Capped, so that a slab a long way off (the vessel) still throws a crisp edge.
        let penumbra = min(0.10 + 0.012 * (p.z - depth), 0.75) * s;
        let shade = smoothstep(-penumbra, penumbra, min(inside.x, inside.y));
        vis[li] *= 1.0 - shade;
        let condensed = smoothstep(max(depth - 4.0, -60.0), -2.0, p.z) * (1.0 - smoothstep(1.0, 6.0, p.z));
        let solidity = occ.info.z;
        let whole = step(0.75, solidity);
        gloom = max(gloom, shade * condensed * whole);
        half_gloom += shade * condensed * solidity * (1.0 - whole);
    }
    let umbra = smoothstep(0.62, 0.95, half_gloom);
    gloom = max(gloom, umbra);
    return Shadowing(vis, gloom, min(half_gloom, 0.5) * 2.0 * (1.0 - umbra));
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
        if lamp.color.w <= 0.0 {
            // The sun: the same light everywhere, brightest looking up its rays.
            let rays = normalize(vec3(lamp.position.xy, 1.0));
            light += lamp.color.rgb * (lamp.position.w * henyey_greenstein(dot(rays, -ray), 0.5) * vis[i]);
            continue;
        }
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

struct Air {
    // How much of the ordinary haze is here: one in haze, zero in clear air.
    haze: f32,
    // How far inside a squall this is: zero outside, one well inside.
    squall: f32,
    // One on the edge of a squall or a hole, falling off either side: the wall of spindrift
    // that marks exactly where shadows start and stop existing.
    edge: f32,
}

// The weather at a point, from the same rectangles the collision rules use. The edges are
// kept tight (a third of a metre) so that what is drawn is what holds.
fn air_at(p: vec2<f32>) -> Air {
    var clear = atm.weather.y;
    var squall = 0.0;
    var edge = 0.0;
    let count = u32(atm.weather.x);
    for (var i = 0u; i < count; i++) {
        let band = atm.bands[i];
        let inside = band.rect.zw - abs(p - band.rect.xy);
        let d = min(inside.x, inside.y);
        let within = smoothstep(-0.3, 0.3, d);
        let is_squall = band.info.x;
        squall = max(squall, within * is_squall);
        clear = max(clear, within * (1.0 - is_squall));
        edge = max(edge, exp(-abs(d) * 0.9));
    }
    return Air(max(1.0 - clear, squall), squall, edge);
}

// Narrows the range of t to where a * t + b <= 0.
fn clip_range(range: vec2<f32>, a: f32, b: f32) -> vec2<f32> {
    var r = range;
    if a > 1e-6 {
        r.y = min(r.y, -b / a);
    } else if a < -1e-6 {
        r.x = max(r.x, -b / a);
    } else if b > 0.0 {
        r.y = r.x - 1.0;
    }
    return r;
}

// How far the beams reach behind the gameplay plane before the haze has swallowed them.
const BEAM_END: f32 = -8.0;

// Is `p` hidden from a light at `emitter` by one of the blinds?
fn blinded(p: vec3<f32>, emitter: vec3<f32>) -> f32 {
    var open = 1.0;
    let count = u32(atm.warden_counts.y);
    for (var i = 0u; i < count; i++) {
        let blind = atm.blinds[i];
        let depth = blind.info.x;
        if p.z >= depth - 0.25 || depth >= emitter.z {
            continue;
        }
        let s = (emitter.z - depth) / (emitter.z - p.z);
        let q = emitter.xy + (p.xy - emitter.xy) * s;
        let inside = blind.rect.zw - abs(q - blind.rect.xy);
        open *= 1.0 - smoothstep(-0.06, 0.06, min(inside.x, inside.y) );
    }
    return open;
}

// Cold light scattered towards the lens by the Wardens' beams. Each beam is a pyramid, so
// the stretch of the view ray inside it is found in closed form (six half-spaces), and only
// that stretch is sampled: crisp edges, a handful of samples, nothing for pixels it misses.
fn warden_beams(cam: vec3<f32>, ray: vec3<f32>, t_surface: f32, jitter: f32) -> vec3<f32> {
    var light = vec3(0.0);
    let count = u32(atm.warden_counts.x);
    let samples = i32(atm.warden_counts.z);
    for (var w = 0u; w < count; w++) {
        let warden = atm.wardens[w];
        let e = warden.emitter.xyz;
        let c = warden.footprint.xy;
        let h = warden.footprint.zw;
        let reach = e.z - cam.z;
        var range = vec2(0.0, t_surface);
        // Between the emitter and where the beam dies away behind the plane.
        range = clip_range(range, ray.z, cam.z - e.z + 1.5);
        range = clip_range(range, -ray.z, BEAM_END - cam.z);
        // Inside the four faces of the pyramid.
        for (var axis = 0; axis < 2; axis++) {
            let lean = (c[axis] - e[axis]);
            let slope = ray[axis] * e.z + lean * ray.z;
            let offset = (cam[axis] - e[axis]) * e.z - lean * reach;
            range = clip_range(range, slope + h[axis] * ray.z, offset - h[axis] * reach);
            range = clip_range(range, -slope + h[axis] * ray.z, -offset - h[axis] * reach);
        }
        let span = range.y - range.x;
        if span <= 0.0 {
            continue;
        }
        let throw_length = distance(e, vec3(c, 0.0));
        var sum = 0.0;
        for (var i = 0; i < samples; i++) {
            let p = cam + ray * (range.x + (f32(i) + jitter) / f32(samples) * span);
            let d = distance(p, e) / throw_length;
            let falloff = min(1.0 / (d * d), 5.0);
            let fade = smoothstep(BEAM_END, 0.0, p.z);
            let dust = 0.55 + 0.9 * value_noise(p * vec3(0.16, 0.16, 0.10) + vec3(atm.counts.x * 0.25, atm.counts.x * -0.12, 0.0));
            sum += falloff * fade * dust * blinded(p, e);
        }
        light += warden.color.rgb * (warden.emitter.w * 0.03 * sum * span / f32(samples));
    }
    return light;
}

// The pool: every surface near the gameplay plane whose xy lies inside a footprint (and
// outside every blind's shade) is washed with the beam's light, whatever it is made of. A
// black shadow lit from the front is not black any more, and neither is the Shade.
fn warden_pool(surface: vec3<f32>) -> vec3<f32> {
    var light = vec3(0.0);
    if abs(surface.z) > 9.0 {
        return light;
    }
    let count = u32(atm.warden_counts.x);
    let blinds = u32(atm.warden_counts.y);
    for (var w = 0u; w < count; w++) {
        let warden = atm.wardens[w];
        let inside = warden.footprint.zw - abs(surface.xy - warden.footprint.xy);
        var wash = smoothstep(0.0, 0.12, min(inside.x, inside.y));
        if wash <= 0.0 {
            continue;
        }
        let e = warden.emitter.xyz;
        for (var i = 0u; i < blinds; i++) {
            let blind = atm.blinds[i];
            let k = e.z / (e.z - blind.info.x);
            let centre = e.xy + (blind.rect.xy - e.xy) * k;
            let shaded = blind.rect.zw * k - abs(surface.xy - centre);
            wash *= 1.0 - smoothstep(0.0, 0.12, min(shaded.x, shaded.y));
        }
        let grain = 0.8 + 0.2 * value_noise(surface * 1.7);
        light += warden.color.rgb * (warden.color.w * wash * grain);
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
        if atm.warden_counts.x > 0.5 {
            scene = vec4(scene.rgb + warden_pool(cam + ray * t_surface), scene.a);
        }
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

    let weathered = atm.weather.x + atm.weather.y > 0.5;

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
            var gloom = shadowing.gloom;
            var veil = shadowing.veil;
            var density = height_density(p.y) * wisps;
            // Snow scatters far more of the light that falls on it than haze does.
            var snow_lit = 1.0;
            if weathered {
                // In clear air there is next to nothing for light or shadow to fall on. Where
                // there is weather the haze is never so thin that the difference cannot be
                // seen; a squall is thicker still, and every edge is a wall of spindrift.
                var air = air_at(p.xy);
                // The wall is drawn near the gameplay plane only. The lens has a little
                // perspective, and a wall as deep as the fog would smear sideways across it.
                air.edge *= 1.0 - smoothstep(6.0, 28.0, -p.z);
                let body = smoothstep(atm.fog_shape.w, atm.fog_shape.w + 30.0, p.z);
                density = density * mix(atm.weather.z, 1.0, air.haze)
                    + atm.weather.w * body * wisps * (max(air.squall, air.haze * 0.3) + air.edge * 1.5);
                gloom *= air.haze;
                veil *= air.haze;
                snow_lit += 1.6 * air.squall + 2.5 * air.edge;
            }
            density += gloom * 0.22;
            let step_transmittance = exp(-density * dt);
            let light = (ambient + lamp_inscatter(p, ray, shadowing.lamps)) * ((1.0 - gloom) * snow_lit);
            inscatter += transmittance * (1.0 - step_transmittance) * light;
            if veil > 0.0 {
                let curl = value_noise(p * vec3(0.30, 0.42, 0.30) + drift * 0.45);
                inscatter += transmittance * VEIL_COLOR * (veil * (0.35 + curl * curl * 1.3) * dt);
            }
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
    if atm.warden_counts.x > 0.5 {
        color += warden_beams(cam, ray, t_surface, jitter);
    }
    // Backdrop 3 is the same daylight as 2 in still, dry air: the summit, above the weather.
    var snow = select(0.0, 1.0, atm.sky.x > 1.5 && atm.sky.x < 2.5);
    if weathered {
        // Judged where the view ray crosses the gameplay plane: the snow comes down hard
        // inside a squall, and hardly at all in clear air.
        let air = air_at((cam + ray * (-cam.z / ray.z)).xy);
        snow = snow * mix(0.15, 1.0, air.haze) + 3.0 * air.squall;
    }
    if snow > 0.0 {
        color += vec3(0.9, 0.95, 1.0) * (snowfall(in.position.xy, cam) * snow);
    }
    return vec4(color, scene.a);
}
