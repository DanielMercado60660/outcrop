#include <metal_stdlib>
using namespace metal;

kernel void plate_flexure(device const float *coordinate [[buffer(0)]],
                          device float *deflection [[buffer(1)]],
                          constant float &amplitude [[buffer(2)]],
                          constant uint &broken [[buffer(3)]],
                          constant uint &n [[buffer(4)]],
                          constant float &alpha_mantissa [[buffer(5)]],
                          constant int &alpha_exponent [[buffer(6)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    if (amplitude == 0.0f) {
        deflection[i] = 0.0f;
        return;
    }
    // All per-coordinate arithmetic is float32 on the GPU. The mantissa is
    // in [1, 2], so this division cannot overflow a finite coordinate.
    // ldexp handles lengths outside float32 range without 0 * infinity at x=0.
    float distance = broken != 0 ? coordinate[i] : abs(coordinate[i]);
    float u = ldexp(distance / alpha_mantissa, -alpha_exponent);
    // Avoid evaluating trigonometric functions at infinity in the far field.
    if (u > 104.0f) {
        deflection[i] = 0.0f;
        return;
    }
    float profile = cos(u);
    if (broken == 0) profile += sin(u);
    deflection[i] = amplitude * exp(-u) * profile;
}
