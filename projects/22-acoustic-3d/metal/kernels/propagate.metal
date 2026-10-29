#include <metal_stdlib>
using namespace metal;

// Flat indexing is x-fastest, followed by y and z. Each axis wraps independently.
// For extreme scales the host uploads Courant values in velocity and sets
// dt_over_spacing to 1; the update itself remains float32 in both cases.
kernel void step(device const float *velocity [[buffer(0)]],
                 device const float *previous [[buffer(1)]],
                 device const float *current [[buffer(2)]],
                 device float *next [[buffer(3)]],
                 constant uint &nx [[buffer(4)]],
                 constant uint &ny [[buffer(5)]],
                 constant uint &nz [[buffer(6)]],
                 constant float &dt_over_spacing [[buffer(7)]],
                 uint i [[thread_position_in_grid]]) {
    uint plane = nx * ny;
    uint count = plane * nz;
    if (i >= count) return;
    uint x = i % nx;
    uint y = (i / nx) % ny;
    uint z = i / plane;
    uint xm = x == 0 ? i + nx - 1 : i - 1;
    uint xp = x + 1 == nx ? i - (nx - 1) : i + 1;
    uint ym = y == 0 ? i + (ny - 1) * nx : i - nx;
    uint yp = y + 1 == ny ? i - (ny - 1) * nx : i + nx;
    uint zm = z == 0 ? i + (nz - 1) * plane : i - plane;
    uint zp = z + 1 == nz ? i - (nz - 1) * plane : i + plane;
    float c = current[i];
    float lap = ((current[xm] - c) + (current[xp] - c))
              + ((current[ym] - c) + (current[yp] - c))
              + ((current[zm] - c) + (current[zp] - c));
    float courant = velocity[i] * dt_over_spacing;
    next[i] = (2.0f * c - previous[i]) + (courant * courant) * lap;
}
