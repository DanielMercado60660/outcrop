#include <metal_stdlib>
using namespace metal;
inline float sample_nmo(device const float *g, device const float *offsets, uint trace, uint t, uint nt, float dt, float velocity) {
    float moveout = offsets[trace] / velocity / dt;
    float x = sqrt(float(t) * float(t) + moveout * moveout);
    if (x > float(nt - 1)) return 0.0f;
    uint lo = uint(x);
    if (lo == nt - 1) return g[trace * nt + lo];
    float a = g[trace * nt + lo], b = g[trace * nt + lo + 1];
    return a + (x - float(lo)) * (b - a);
}
kernel void nmo_kernel(device const float *g [[buffer(0)]], device const float *offsets [[buffer(1)]], device float *out [[buffer(2)]], constant uint &nt [[buffer(3)]], constant uint &no [[buffer(4)]], constant float &dt [[buffer(5)]], constant float &velocity [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    if (id >= nt * no) return;
    out[id] = sample_nmo(g, offsets, id / nt, id % nt, nt, dt, velocity);
}
kernel void stack_kernel(device const float *g [[buffer(0)]], device const float *offsets [[buffer(1)]], device float *out [[buffer(2)]], constant uint &nt [[buffer(3)]], constant uint &no [[buffer(4)]], constant float &dt [[buffer(5)]], constant float &velocity [[buffer(6)]], uint t [[thread_position_in_grid]]) {
    if (t >= nt) return;
    float sum = 0.0f;
    for (uint i = 0; i < no; ++i) sum += sample_nmo(g, offsets, i, t, nt, dt, velocity);
    out[t] = no ? sum / float(no) : 0.0f;
}
kernel void semblance_terms(device const float *g [[buffer(0)]], device const float *offsets [[buffer(1)]], device const float *velocities [[buffer(2)]], device float *numer [[buffer(3)]], device float *denom [[buffer(4)]], constant uint &nt [[buffer(5)]], constant uint &no [[buffer(6)]], constant uint &nv [[buffer(7)]], constant float &dt [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    if (id >= nt * nv) return;
    uint t = id % nt;
    float velocity = velocities[id / nt];
    float sum = 0.0f, squares = 0.0f;
    for (uint i = 0; i < no; ++i) {
        float a = sample_nmo(g, offsets, i, t, nt, dt, velocity);
        sum += a;
        squares += a * a;
    }
    numer[id] = sum * sum;
    denom[id] = squares;
}
// Keep the low-order bits when terms enter and leave a sliding window.
inline void add_compensated(thread float &hi, thread float &lo, float x) {
    float s = hi + x;
    float v = s - hi;
    float e = (hi - (s - v)) + (x - v);
    float low = lo + e;
    float total = s + low;
    lo = low - (total - s);
    hi = total;
}
kernel void semblance_window(device const float *numer [[buffer(0)]], device const float *denom [[buffer(1)]], device float *out [[buffer(2)]], constant uint &nt [[buffer(3)]], constant uint &no [[buffer(4)]], constant uint &nv [[buffer(5)]], constant uint &half_window [[buffer(6)]], uint velocity [[thread_position_in_grid]]) {
    if (velocity >= nv || nt == 0) return;
    uint base = velocity * nt;
    uint end = min(half_window, nt - 1);
    float nh = 0.0f, nl = 0.0f, dh = 0.0f, dl = 0.0f;
    uint nonzero = 0;
    for (uint k = 0; k <= end; ++k) {
        add_compensated(nh, nl, numer[base + k]);
        add_compensated(dh, dl, denom[base + k]);
        nonzero += denom[base + k] > 0.0f ? 1u : 0u;
    }
    for (uint t = 0; t < nt; ++t) {
        float n = nh + nl, d = dh + dl;
        out[base + t] = nonzero > 0 && d > 0.0f && no > 0 ? n / (float(no) * d) : 0.0f;
        if (t >= half_window) {
            uint k = t - half_window;
            add_compensated(nh, nl, -numer[base + k]);
            add_compensated(dh, dl, -denom[base + k]);
            nonzero -= denom[base + k] > 0.0f ? 1u : 0u;
        }
        if (end < nt - 1) {
            ++end;
            add_compensated(nh, nl, numer[base + end]);
            add_compensated(dh, dl, denom[base + end]);
            nonzero += denom[base + end] > 0.0f ? 1u : 0u;
        }
        if (nonzero == 0) { nh = 0.0f; nl = 0.0f; dh = 0.0f; dl = 0.0f; }
    }
}
