#include <metal_stdlib>
using namespace metal;

// Bounds are relative to the observation; angles are radians and radii m.
struct Cell { float w, e, s, n, b, t; };

inline float one_minus_cos_psi(float lon, float lat, float obs_lat) {
    float a = sin(0.5f * lat);
    float b = sin(0.5f * lon);
    return 2.0f * (a * a + cos(obs_lat) * cos(obs_lat + lat) * b * b);
}

kernel void tesseroid_gravity(device const float *pairs [[buffer(0)]],
                             device float *result [[buffer(1)]],
                             constant uint &count [[buffer(2)]],
                             constant uint &sources [[buffer(3)]],
                             constant uint &gz [[buffer(4)]],
                             constant uint &radial [[buffer(5)]],
                             device uint *status [[buffer(6)]],
                             uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    float sum = 0.0f, correction = 0.0f;
    for (uint source = 0; source < sources; ++source) {
        device const float *p = pairs + (index * sources + source) * 9;
        float rho = p[8];
        if (rho == 0.0f) continue;
        float obs_lat = p[6], radius = p[7];
        Cell stack[256];
        uint length = 1;
        stack[0] = {p[0], p[1], p[2], p[3], p[4], p[5]};
        while (length != 0) {
            Cell c = stack[--length];
            float lon = (c.w + c.e) * 0.5f;
            float lat = (c.s + c.n) * 0.5f;
            float dr = (c.b + c.t) * 0.5f;
            float source_radius = radius + dr;
            float omc = one_minus_cos_psi(lon, lat, obs_lat);
            float distance = sqrt(dr * dr + 2.0f * radius * source_radius * omc);
            float lon_size = (radius + c.t) * 2.0f * asin(clamp(cos(obs_lat + lat) * sin((c.e - c.w) * 0.5f), -1.0f, 1.0f));
            float lat_size = (radius + c.t) * (c.n - c.s);
            float q = gz ? 2.5f : 1.0f;
            bool sx = distance < q * lon_size;
            bool sy = distance < q * lat_size;
            bool sz = radial && distance < q * (c.t - c.b);
            if (sx || sy || sz) {
                uint nx = sx ? 2 : 1, ny = sy ? 2 : 1, nz = sz ? 2 : 1;
                if (length + nx * ny * nz > 256) { status[index] = 1; return; }
                float xm = (c.w + c.e) * 0.5f, ym = (c.s + c.n) * 0.5f, zm = (c.b + c.t) * 0.5f;
                if ((sx && (xm == c.w || xm == c.e)) ||
                    (sy && (ym == c.s || ym == c.n)) ||
                    (sz && (zm == c.b || zm == c.t))) {
                    status[index] = 2;
                    return;
                }
                for (uint ix = 0; ix < nx; ++ix)
                    for (uint iy = 0; iy < ny; ++iy)
                        for (uint iz = 0; iz < nz; ++iz) {
                            Cell child = {sx && ix ? xm : c.w, sx && !ix ? xm : c.e,
                                          sy && iy ? ym : c.s, sy && !iy ? ym : c.n,
                                          sz && iz ? zm : c.b, sz && !iz ? zm : c.t};
                            stack[length++] = child;
                        }
                continue;
            }
            float hx = (c.e - c.w) * 0.5f, hy = (c.n - c.s) * 0.5f, hz = (c.t - c.b) * 0.5f;
            float factor = 6.6743e-11f * rho * hx * hy * hz * (gz ? 1.0e5f : 1.0f);
            constexpr float node = 0.5773502691896258f;
            for (uint ix = 0; ix < 2; ++ix)
                for (uint iy = 0; iy < 2; ++iy)
                    for (uint iz = 0; iz < 2; ++iz) {
                        float x = lon + (ix ? node : -node) * hx;
                        float y = lat + (iy ? node : -node) * hy;
                        float z = dr + (iz ? node : -node) * hz;
                        float r = radius + z;
                        float h = one_minus_cos_psi(x, y, obs_lat);
                        float l2 = z * z + 2.0f * radius * r * h;
                        float inv = rsqrt(l2);
                        float value = factor * r * r * cos(obs_lat + y) * inv;
                        if (gz) value *= (-z + r * h) / l2;
                        float adjusted = value - correction;
                        float next = sum + adjusted;
                        correction = (next - sum) - adjusted;
                        sum = next;
                    }
        }
    }
    result[index] = sum;
}
