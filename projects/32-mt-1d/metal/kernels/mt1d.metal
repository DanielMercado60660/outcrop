#include <metal_stdlib>
using namespace metal;

// Complex numbers are stored as (real, imaginary).
inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cdiv(float2 a, float2 b) {
    float scale = max(abs(b.x), abs(b.y));
    float2 bs = b / scale;
    float2 as = a / scale;
    float d = dot(bs, bs);
    return float2(as.x * bs.x + as.y * bs.y, as.y * bs.x - as.x * bs.y) / d;
}

kernel void mt1d(device const float *rho [[buffer(0)]],
                 device const float *thickness [[buffer(1)]],
                 device const float *frequency [[buffer(2)]],
                 device float *apparent [[buffer(3)]],
                 device float *phase [[buffer(4)]],
                 constant uint &layers [[buffer(5)]],
                 constant uint &count [[buffer(6)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float mu0 = 1.25663706212e-6f;
    const float pi = 3.14159265358979323846f;
    // z is physical impedance divided by sqrt(mu0 * omega). Its squared
    // magnitude is apparent resistivity, and its phase is unchanged.
    // Handle the half-space directly, including resistivities near FLT_MAX.
    if (layers == 1) {
        apparent[i] = rho[0];
        phase[i] = 45.0f;
        return;
    }
    float bottom = sqrt(rho[layers - 1]) * 0.7071067811865475f;
    float2 z = float2(bottom, bottom);

    for (uint j = layers - 1; j > 0; --j) {
        uint k = j - 1;
        float scale = sqrt(rho[k]) * 0.7071067811865475f;
        float2 intrinsic = float2(scale, scale);
        float depth = thickness[k] * ((sqrt(frequency[i]) * sqrt(pi * mu0)) / sqrt(rho[k]));
        // Beyond this depth the reflection is negligible; avoid sin/cos of
        // an infinite or very large argument when the product overflows.
        if (depth > 20.0f) {
            z = intrinsic;
            continue;
        }
        // exp(-2 * (1+i) * depth) is bounded for thick layers.
        float decay = exp(-2.0f * depth);
        float2 propagation = decay * float2(cos(2.0f * depth), -sin(2.0f * depth));
        float2 reflection = cmul(cdiv(intrinsic - z, intrinsic + z), propagation);
        z = cmul(intrinsic, cdiv(float2(1.0f, 0.0f) - reflection,
                                  float2(1.0f, 0.0f) + reflection));
    }
    apparent[i] = dot(z, z);
    phase[i] = atan2(z.y, z.x) * (180.0f / pi);
}
