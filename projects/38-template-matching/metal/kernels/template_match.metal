#include <metal_stdlib>
using namespace metal;

// Apply one positive power-of-two scale to the whole window before centering.
// Reconstruct subnormal inputs from their integer significands so a device's
// flush-to-zero arithmetic cannot turn a nonconstant tiny trace into a constant.
// Ordinary values take a bit-exact exponent adjustment, preserving small
// representable variations even on top of a large DC offset.
inline float scaled_sample(uint bits, uint scale_exponent) {
    uint sign = bits & 0x80000000u;
    uint exponent = (bits >> 23) & 0xffu;
    uint fraction = bits & 0x7fffffu;
    int adjusted = int(exponent) - int(scale_exponent) + 127;
    if (exponent != 0u && adjusted > 0) {
        return as_type<float>(sign | (uint(adjusted) << 23) | fraction);
    }
    uint significand = fraction | (exponent == 0u ? 0u : 0x800000u);
    float value = ldexp(float(significand),
                       int(max(exponent, 1u)) - int(scale_exponent) - 23);
    return sign == 0u ? value : -value;
}

kernel void template_match(device const uint *data [[buffer(0)]],
                           device const uint *templates [[buffer(1)]],
                           device float *correlation [[buffer(2)]],
                           constant uint &stations [[buffer(3)]],
                           constant uint &samples [[buffer(4)]],
                           constant uint &length [[buffer(5)]],
                           constant uint &windows [[buffer(6)]],
                           constant uint &count [[buffer(7)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    // C-order output is (template, station, valid starting sample).
    uint k = i % windows;
    uint pair = i / windows;
    uint station = pair % stations;
    uint xbase = station * samples + k;
    uint ybase = pair * length;
    uint xmax = 0u, ymax = 0u;
    for (uint j = 0; j < length; ++j) {
        xmax = max(xmax, data[xbase + j] & 0x7fffffffu);
        ymax = max(ymax, templates[ybase + j] & 0x7fffffffu);
    }
    uint xe = max(xmax >> 23, 1u), ye = max(ymax >> 23, 1u);
    // Scaled magnitudes are below two, so subtraction, sums and squared norms
    // cannot overflow. An entirely subnormal trace is lifted into normal range.
    // Translation by the first scaled sample makes constant traces exactly zero.
    float x0 = scaled_sample(data[xbase], xe);
    float y0 = scaled_sample(templates[ybase], ye);
    float sx = 0.0f, sy = 0.0f;
    for (uint j = 0; j < length; ++j) {
        sx += scaled_sample(data[xbase + j], xe) - x0;
        sy += scaled_sample(templates[ybase + j], ye) - y0;
    }
    float mx = sx / float(length), my = sy / float(length);
    float xx = 0.0f, yy = 0.0f, xy = 0.0f;
    for (uint j = 0; j < length; ++j) {
        float x = (scaled_sample(data[xbase + j], xe) - x0) - mx;
        float y = (scaled_sample(templates[ybase + j], ye) - y0) - my;
        xx += x * x;
        yy += y * y;
        xy += x * y;
    }
    // Divide by the norms separately to avoid overflow in the product xx*yy.
    // No epsilon: any strictly positive variance contributes its coefficient.
    correlation[i] = (xx > 0.0f && yy > 0.0f)
        ? clamp((xy / sqrt(xx)) / sqrt(yy), -1.0f, 1.0f) : 0.0f;
}
