#include <metal_stdlib>
using namespace metal;

inline float2 multiply(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 conjugate(float2 a) { return float2(a.x, -a.y); }

// Each butterfly owns both outputs. Dispatch boundaries synchronize stages.
kernel void fft_reorder(device const float2 *input [[buffer(0)]],
                        device float2 *output [[buffer(1)]],
                        constant uint &n [[buffer(2)]],
                        constant uint &bits [[buffer(3)]],
                        uint2 id [[thread_position_in_grid]]) {
    uint reversed = 0, j = id.x;
    for (uint b = 0; b < bits; ++b) {
        reversed = (reversed << 1) | (j & 1);
        j >>= 1;
    }
    output[id.y * n + reversed] = input[id.y * n + id.x];
}

kernel void fft_stage(device float2 *data [[buffer(0)]],
                      device const float2 *twiddle [[buffer(1)]],
                      constant uint &n [[buffer(2)]],
                      constant uint &half_width [[buffer(3)]],
                      constant uint &inverse [[buffer(4)]],
                      uint2 id [[thread_position_in_grid]]) {
    uint j = id.x % half_width;
    uint i = id.y * n + (id.x / half_width) * (2 * half_width) + j;
    float2 w = twiddle[j * (n / (2 * half_width))];
    if (inverse) w = conjugate(w);
    float2 a = data[i], b = multiply(data[i + half_width], w);
    data[i] = a + b;
    data[i + half_width] = a - b;
}

kernel void fft_scale(device float2 *data [[buffer(0)]],
                      constant float &scale [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    data[i] *= scale;
}

// Bluestein gives an exact n-point DFT through a padded convolution.
// The gather's frequency grid is never padded.
kernel void chirp_prepare(device const float2 *input [[buffer(0)]],
                          device const float2 *chirp [[buffer(1)]],
                          device float2 *work [[buffer(2)]],
                          constant uint &n [[buffer(3)]],
                          constant uint &m [[buffer(4)]],
                          constant uint &inverse [[buffer(5)]],
                          uint2 id [[thread_position_in_grid]]) {
    float2 v = float2(0.0f);
    if (id.x < n) {
        v = input[id.y * n + id.x];
        if (inverse) v = conjugate(v);
        v = multiply(v, chirp[id.x]);
    }
    work[id.y * m + id.x] = v;
}

kernel void convolution_product(device float2 *data [[buffer(0)]],
                                device const float2 *spectrum [[buffer(1)]],
                                constant uint &m [[buffer(2)]],
                                uint2 id [[thread_position_in_grid]]) {
    uint i = id.y * m + id.x;
    data[i] = multiply(data[i], spectrum[id.x]);
}

kernel void chirp_finish(device const float2 *work [[buffer(0)]],
                         device const float2 *chirp [[buffer(1)]],
                         device float2 *output [[buffer(2)]],
                         constant uint &n [[buffer(3)]],
                         constant uint &m [[buffer(4)]],
                         constant uint &inverse [[buffer(5)]],
                         uint2 id [[thread_position_in_grid]]) {
    float2 v = multiply(work[id.y * m + id.x], chirp[id.x]);
    if (inverse) v = conjugate(v) / float(n);
    output[id.y * n + id.x] = v;
}

kernel void real_to_complex(device const float *input [[buffer(0)]],
                            device float2 *output [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    output[i] = float2(input[i], 0.0f);
}

kernel void complex_to_real(device const float2 *input [[buffer(0)]],
                            device float *output [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    output[i] = input[i].x;
}

kernel void transpose(device const float2 *input [[buffer(0)]],
                      device float2 *output [[buffer(1)]],
                      constant uint &width [[buffer(2)]],
                      constant uint &height [[buffer(3)]],
                      uint2 id [[thread_position_in_grid]]) {
    output[id.x * height + id.y] = input[id.y * width + id.x];
}

kernel void apply_mask(device float2 *data [[buffer(0)]],
                       device const float *mask [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    data[i] *= mask[i];
}

// One dot product per frequency, shared by all output traces.
kernel void radon_coefficients(device const float2 *input [[buffer(0)]],
                               device const float2 *phase [[buffer(1)]],
                               device float2 *coefficients [[buffer(2)]],
                               constant uint &nx [[buffer(3)]],
                               constant uint &nt [[buffer(4)]],
                               uint f [[thread_position_in_grid]]) {
    float2 sum = float2(0.0f);
    for (uint x = 0; x < nx; ++x) {
        uint i = x * nt + f;
        sum += multiply(conjugate(phase[i]), input[i]);
    }
    coefficients[f] = sum / float(nx);
}

kernel void project_radon(device float2 *data [[buffer(0)]],
                          device const float2 *phase [[buffer(1)]],
                          device const float2 *coefficients [[buffer(2)]],
                          constant uint &nt [[buffer(3)]],
                          uint2 id [[thread_position_in_grid]]) {
    uint i = id.y * nt + id.x;
    data[i] -= multiply(phase[i], coefficients[id.x]);
}
