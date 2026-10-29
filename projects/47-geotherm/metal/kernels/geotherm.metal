#include <metal_stdlib>
using namespace metal;

// Metal has no erf intrinsic. For 0 <= x <= 1 integrate the Taylor series
// exp(-x*x) term by term. Horner evaluation preserves relative accuracy at
// small x (and gives exactly zero at x = 0), unlike subtracting an erfc
// approximation from one. The first omitted term at x = 1 is < 1.3e-9.
// Above one, A&S 7.1.26 has absolute approximation error <= 1.5e-7;
// erf(x) >= erf(1) there, so this also provides ample relative accuracy.
inline float erf_nonnegative(float x) {
    if (x <= 1.0f) {
        float s = x * x;
        float p = 1.0f / 76204800.0f;
        p = fma(p, s, -1.0f / 6894720.0f);
        p = fma(p, s, 1.0f / 685440.0f);
        p = fma(p, s, -1.0f / 75600.0f);
        p = fma(p, s, 1.0f / 9360.0f);
        p = fma(p, s, -1.0f / 1320.0f);
        p = fma(p, s, 1.0f / 216.0f);
        p = fma(p, s, -1.0f / 42.0f);
        p = fma(p, s, 1.0f / 10.0f);
        p = fma(p, s, -1.0f / 3.0f);
        p = fma(p, s, 1.0f);
        return 1.1283791670955126f * x * p;
    }
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float p = 1.061405429f;
    p = fma(p, t, -1.453152027f);
    p = fma(p, t, 1.421413741f);
    p = fma(p, t, -0.284496736f);
    p = fma(p, t, 0.254829592f);
    return 1.0f - (p * t) * exp(-x * x);
}

kernel void geotherm(device const float *depth [[buffer(0)]],
                     device const float *age [[buffer(1)]],
                     device float *temperature [[buffer(2)]],
                     device float *flux [[buffer(3)]],
                     constant uint &n [[buffer(4)]], constant uint &d [[buffer(5)]],
                     constant float &surface [[buffer(6)]], constant float &delta [[buffer(7)]],
                     constant float &length_scale [[buffer(8)]], constant float &flux_scale [[buffer(9)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n * d) return;
    uint a = i / d;
    float root_age = sqrt(age[a]);
    float x = depth[i % d] / (2.0f * length_scale * root_age);
    // z is positive downward, hence upward flux is +k*dT/dz.
    temperature[i] = surface + delta * erf_nonnegative(x);
    flux[i] = (flux_scale / root_age) * exp(-x * x);
}

kernel void invert_age(device const float *flux [[buffer(0)]], device float *age [[buffer(1)]],
                       constant uint &n [[buffer(2)]], constant float &coefficient [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    // q(0, a) = coefficient / sqrt(a), with a already in Myr.
    float ratio = coefficient / flux[i];
    age[i] = ratio * ratio;
}

kernel void misfit(device const float *age [[buffer(0)]], device const float *observed [[buffer(1)]],
                   device float *gradient [[buffer(2)]], device float *squared_residual [[buffer(3)]],
                   constant uint &n [[buffer(4)]], constant float &coefficient [[buffer(5)]],
                   constant float &scale [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float predicted = coefficient / sqrt(age[i]);
    float residual = (predicted - observed[i]) / scale;
    squared_residual[i] = residual * residual;
    // dJ/da_i = (q_i-observed_i)/(n*scale^2) * (-q_i/(2*a_i)).
    gradient[i] = residual * (-0.5f * predicted / age[i]) / (scale * float(n));
}

// Each thread combines at most four entries in a balanced tree. Repeated
// dispatches give O(log(n)) summation depth instead of a serial O(n) sum,
// so similarly sized residuals do not disappear into a growing accumulator.
// Separate source/destination buffers avoid races between threadgroups;
// bounds checks cover singleton inputs and incomplete groups of four.
kernel void reduce_misfit(device const float *input [[buffer(0)]],
                         device float *output [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         constant uint &n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    uint base = 4u * i;
    if (base >= count) return;
    float a = input[base];
    float b = base + 1u < count ? input[base + 1u] : 0.0f;
    float c = base + 2u < count ? input[base + 2u] : 0.0f;
    float d = base + 3u < count ? input[base + 3u] : 0.0f;
    float sum = (a + b) + (c + d);
    // Normalize exactly once, on the final pass, including n == 1.
    output[i] = count <= 4u ? 0.5f * sum / float(n) : sum;
}
