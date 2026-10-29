#include <metal_stdlib>
using namespace metal;

inline float interior(device const float *old, device const float *prev,
                      uint i, float r2) {
    // Displacement leapfrog: r2 = (c * dt / dx)^2.
    float lap = old[i - 1] - 2.0f * old[i] + old[i + 1];
    return 2.0f * old[i] - prev[i] + r2 * lap;
}

kernel void advance(device float *history [[buffer(0)]],
                    device const float *initial_previous [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    constant uint &step [[buffer(3)]],
                    constant float &r2 [[buffer(4)]],
                    constant float &q [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    device const float *old = history + step * n;
    device const float *prev = step == 0 ? initial_previous : history + (step - 1) * n;
    device float *next = history + (step + 1) * n;
    // Recompute the adjacent interior update from the two old time levels.
    // Reading next[1] or next[n-2] here would race another GPU thread.
    // This also works for n=3, where both walls share the same interior point.
    if (i == 0) {
        float adjacent = interior(old, prev, 1, r2);
        next[i] = old[1] + q * (adjacent - old[0]);
    } else if (i == n - 1) {
        float adjacent = interior(old, prev, n - 2, r2);
        next[i] = old[n - 2] + q * (adjacent - old[n - 1]);
    } else {
        next[i] = interior(old, prev, i, r2);
    }
}
