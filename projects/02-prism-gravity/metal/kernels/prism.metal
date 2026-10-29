#include <metal_stdlib>
using namespace metal;

// Fukushima's continuous extensions at the axes and the origin.
inline float safe_log(float x, float y, float z, float r) {
    if (r == 0.0f) return 0.0f;
    if (x < 0.0f) {
        if (y == 0.0f && z == 0.0f) return -log(-2.0f * x);
        return log((y * y + z * z) / (r - x));
    }
    return log(x + r);
}
inline float safe_atan(float y, float x) {
    if (x == 0.0f) {
        if (y > 0.0f) return 1.5707963267948966f;
        if (y < 0.0f) return -1.5707963267948966f;
        return 0.0f;
    }
    return atan(y / x);
}
inline float corner(float x, float y, float z, uint field) {
    float r = sqrt(x*x + y*y + z*z);
    if (field == 0) {
        return x*y*safe_log(z,x,y,r) + y*z*safe_log(x,y,z,r)
             + x*z*safe_log(y,z,x,r)
             - 0.5f*x*x*safe_atan(z*y,x*r)
             - 0.5f*y*y*safe_atan(z*x,y*r)
             - 0.5f*z*z*safe_atan(x*y,z*r);
    }
    if (field == 1) return -(y*safe_log(z,x,y,r) + z*safe_log(y,z,x,r) - x*safe_atan(y*z,x*r));
    if (field == 2) return -(z*safe_log(x,y,z,r) + x*safe_log(z,x,y,r) - y*safe_atan(x*z,y*r));
    // g_z is downward, the negative of the upward component.
    return x*safe_log(y,z,x,r) + y*safe_log(x,y,z,r) - z*safe_atan(x*y,z*r);
}

kernel void prism_gravity(device const float *coordinates [[buffer(0)]],
                          device const float *prisms [[buffer(1)]],
                          device const float *density [[buffer(2)]],
                          device float *output [[buffer(3)]],
                          constant uint &n [[buffer(4)]],
                          constant uint &p [[buffer(5)]],
                          constant uint &field [[buffer(6)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    // Widen before multiplying: N and P can each span the uint range,
    // while their flattened coordinate/bound offsets can exceed that range.
    const ulong coordinate_offset = ulong(tid) * 3;
    float ox = coordinates[coordinate_offset], oy = coordinates[coordinate_offset+1], oz = coordinates[coordinate_offset+2];
    float total = 0.0f;
    for (uint q = 0; q < p; ++q) {
        if (density[q] == 0.0f) continue;
        const ulong prism_offset = ulong(q) * 6;
        float sum = 0.0f;
        for (uint i = 0; i < 2; ++i) {
            float x = prisms[prism_offset + (1-i)] - ox;
            for (uint j = 0; j < 2; ++j) {
                float y = prisms[prism_offset + 2 + (1-j)] - oy;
                for (uint k = 0; k < 2; ++k) {
                    float z = prisms[prism_offset + 4 + (1-k)] - oz;
                    float value = corner(x,y,z,field);
                    sum += ((i+j+k) & 1) ? -value : value;
                }
            }
        }
        total += density[q] * sum;
    }
    output[tid] = total * (field == 0 ? 6.67430e-11f : 6.67430e-6f);
}
