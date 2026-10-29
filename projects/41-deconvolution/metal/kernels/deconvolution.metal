#include <metal_stdlib>
using namespace metal;

kernel void autocorrelation(device const float *x [[buffer(0)]], device float *r [[buffer(1)]],
                            constant uint &n [[buffer(2)]], constant uint &count [[buffer(3)]],
                            uint k [[thread_position_in_grid]]) {
    if (k >= count) return;
    float sum = 0.0f;
    if (k < n) for (uint t = 0; t < n - k; ++t) sum += x[t] * x[t + k];
    r[k] = sum;
}

kernel void system(device const float *r [[buffer(0)]], device const float *x [[buffer(1)]],
                   device float *a [[buffer(2)]], device float *b [[buffer(3)]],
                   constant uint &p [[buffer(4)]], constant uint &lag [[buffer(5)]],
                   constant float &loading [[buffer(6)]], constant uint &pred [[buffer(7)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= p) return;
    b[i] = pred != 0 ? r[lag + i] : (i == 0 ? x[0] : 0.0f);
    float diagonal = pred != 0 ? loading * r[0] : loading;
    for (uint j = 0; j < p; ++j) {
        uint d = i > j ? i - j : j - i;
        a[i * p + j] = r[d] + (i == j ? diagonal : 0.0f);
    }
}

// Partial pivoting and back substitution for the small filter systems.
kernel void solve(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                  constant uint &n [[buffer(2)]], device uint *status [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    for (uint k = 0; k < n; ++k) {
        uint pivot = k;
        float biggest = fabs(a[k * n + k]);
        for (uint i = k + 1; i < n; ++i) {
            float v = fabs(a[i * n + k]);
            if (v > biggest) { biggest = v; pivot = i; }
        }
        if (biggest == 0.0f) {
            status[0] = 1;
            return;
        }
        if (pivot != k) {
            for (uint j = k; j < n; ++j) {
                float tmp = a[k * n + j]; a[k * n + j] = a[pivot * n + j]; a[pivot * n + j] = tmp;
            }
            float tmp = b[k]; b[k] = b[pivot]; b[pivot] = tmp;
        }
        for (uint i = k + 1; i < n; ++i) {
            float factor = a[i * n + k] / a[k * n + k];
            for (uint j = k + 1; j < n; ++j) a[i * n + j] -= factor * a[k * n + j];
            b[i] -= factor * b[k];
        }
    }
    for (uint ii = n; ii > 0; --ii) {
        uint i = ii - 1;
        float value = b[i];
        for (uint j = i + 1; j < n; ++j) value -= a[i * n + j] * b[j];
        b[i] = value / a[i * n + i];
    }
}

kernel void apply_filter(device const float *x [[buffer(0)]], device const float *c [[buffer(1)]],
                         device float *y [[buffer(2)]], constant uint &n [[buffer(3)]],
                         constant uint &p [[buffer(4)]], constant uint &lag [[buffer(5)]],
                         constant uint &pred [[buffer(6)]], uint t [[thread_position_in_grid]]) {
    if (t >= n) return;
    float sum = 0.0f;
    for (uint j = 0; j < p; ++j) {
        uint delay = lag + j;
        if (delay > t) break;
        sum += c[j] * x[t - delay];
    }
    y[t] = pred != 0 ? x[t] - sum : sum;
}
