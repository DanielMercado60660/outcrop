#include <metal_stdlib>
using namespace metal;

inline uint bit_reverse_index(uint k, uint n) {
    uint r = 0;
    for (uint m = n; m > 1; m >>= 1) { r = (r << 1) | (k & 1); k >>= 1; }
    return r;
}

kernel void fft_init(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                     device float2 *out [[buffer(2)]], constant uint &n [[buffer(3)]],
                     constant uint &windows [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
    uint t = id.x, w = id.y; if (t >= n || w >= windows) return;
    uint j = bit_reverse_index(t, n);
    out[w*n+j] = float2(a[w*n+t], 0);
    out[(windows+w)*n+j] = float2(b[w*n+t], 0);
}

kernel void fft_stage(device const float2 *src [[buffer(0)]], device float2 *dst [[buffer(1)]],
                      constant uint &n [[buffer(2)]], constant uint &windows [[buffer(3)]],
                      constant uint &stage [[buffer(4)]], constant uint &inverse [[buffer(5)]],
                      uint2 id [[thread_position_in_grid]]) {
    uint t = id.x, seq = id.y; if (t >= n || seq >= windows * (inverse != 0 ? 3u : 2u)) return;
    // `half` is a reserved scalar type in MSL; use a descriptive index name.
    uint half_width = 1u << stage;
    uint width = half_width * 2, j = t % half_width, base = (t / width) * width;
    float c; float s = sincos((inverse != 0 ? 1.0f : -1.0f) * 6.2831853071795864769f * float(j) / float(width), c);
    float2 a = src[seq*n+base+j], b = src[seq*n+base+j+half_width];
    float2 z = float2(b.x*c-b.y*s, b.x*s+b.y*c);
    dst[seq*n+t] = (t % width < half_width) ? a+z : a-z;
}

kernel void fft_whiten(device const float2 *src [[buffer(0)]], device float2 *dst [[buffer(1)]],
                       device const uint *mask [[buffer(2)]], constant uint &n [[buffer(3)]],
                       constant uint &windows [[buffer(4)]], constant float &eps [[buffer(5)]],
                       uint2 id [[thread_position_in_grid]]) {
    uint k = id.x, w = id.y; if (k >= n || w >= windows) return;
    // Whiten the real FFT's nonnegative bins and reconstruct their conjugates.
    uint p = k <= n/2 ? k : n-k;
    float2 a = src[w*n+p], b = src[(windows+w)*n+p];
    if (p == 0 || (n % 2 == 0 && p == n/2)) { a.y = 0; b.y = 0; }
    float ma = length(a), mb = length(b);
    a = mask[p] && ma > eps ? a/ma : float2(0);
    b = mask[p] && mb > eps ? b/mb : float2(0);
    if (k > n/2) { a.y = -a.y; b.y = -b.y; }
    // conj(A) * B: delaying b gives a positive correlation lag.
    float2 c = float2(a.x*b.x+a.y*b.y, a.x*b.y-a.y*b.x);
    uint j = bit_reverse_index(k,n);
    dst[w*n+j] = a; dst[(windows+w)*n+j] = b; dst[(2*windows+w)*n+j] = c;
}

kernel void fft_finish(device const float2 *src [[buffer(0)]], device float *a [[buffer(1)]],
                       device float *b [[buffer(2)]], device float *corr [[buffer(3)]],
                       constant uint &n [[buffer(4)]], constant uint &windows [[buffer(5)]],
                       uint2 id [[thread_position_in_grid]]) {
    uint t = id.x, w = id.y; if (t >= n || w >= windows) return;
    a[w*n+t] = src[w*n+t].x / float(n); b[w*n+t] = src[(windows+w)*n+t].x / float(n);
    // fftshift reads unshifted[t + ceil(n/2)], putting zero lag at floor(n/2).
    uint lag = (t + (n+1)/2) % n;
    corr[w*n+t] = src[(2*windows+w)*n+lag].x / float(n);
}

inline void add_compensated(thread float &sum, thread float &err, float value) {
    float y = value - err;
    float t = sum + y;
    err = (t - sum) - y;
    sum = t;
}

kernel void whiten_spectrum(device const float *a [[buffer(0)]],
                            device const float *b [[buffer(1)]],
                            device const uint *mask [[buffer(2)]],
                            device float2 *sa [[buffer(3)]],
                            device float2 *sb [[buffer(4)]],
                            constant uint &n [[buffer(5)]],
                            constant uint &windows [[buffer(6)]],
                            constant float &eps [[buffer(7)]],
                            uint2 id [[thread_position_in_grid]]) {
    uint k = id.x, w = id.y, bins = n / 2 + 1;
    if (k >= bins || w >= windows) return;
    uint idx = w * bins + k;
    if (mask[k] == 0) { sa[idx] = float2(0); sb[idx] = float2(0); return; }
    float ar = 0, ai = 0, br = 0, bi = 0;
    float ear = 0, eai = 0, ebr = 0, ebi = 0;
    for (uint t = 0; t < n; ++t) {
        uint phase = (k * t) % n;
        float c, s;
        if (k == 0) { c = 1; s = 0; }
        else if (n % 2 == 0 && k == n / 2) { c = (t % 2 == 0) ? 1.0f : -1.0f; s = 0; }
        else { float angle = 6.2831853071795864769f * (float(phase) / float(n)); s = sincos(angle, c); }
        float av = a[w * n + t], bv = b[w * n + t];
        add_compensated(ar, ear, av * c); add_compensated(ai, eai, -av * s);
        add_compensated(br, ebr, bv * c); add_compensated(bi, ebi, -bv * s);
    }
    float am = length(float2(ar, ai)), bm = length(float2(br, bi));
    sa[idx] = am > eps ? float2(ar, ai) / am : float2(0);
    sb[idx] = bm > eps ? float2(br, bi) / bm : float2(0);
}

kernel void inverse_and_correlate(device const float2 *sa [[buffer(0)]],
                                  device const float2 *sb [[buffer(1)]],
                                  device float *wa [[buffer(2)]],
                                  device float *wb [[buffer(3)]],
                                  device float *corr [[buffer(4)]],
                                  constant uint &n [[buffer(5)]],
                                  constant uint &windows [[buffer(6)]],
                                  uint2 id [[thread_position_in_grid]]) {
    uint t = id.x, w = id.y, bins = n / 2 + 1;
    if (t >= n || w >= windows) return;
    uint lag = (t + (n + 1) / 2) % n;
    float va = 0, vb = 0, vc = 0, ea = 0, eb = 0, ec = 0;
    for (uint k = 0; k < bins; ++k) {
        float2 x = sa[w * bins + k], y = sb[w * bins + k];
        float2 cross = float2(x.x * y.x + x.y * y.y, x.x * y.y - x.y * y.x);
        // Only interior real-FFT bins have a distinct negative-frequency partner.
        float weight = (k == 0 || (n % 2 == 0 && k == n / 2)) ? 1.0f : 2.0f;
        uint p = (k * t) % n;
        uint q = (k * lag) % n;
        float ct, cl;
        float st = sincos(6.2831853071795864769f * (float(p) / float(n)), ct);
        float sl = sincos(6.2831853071795864769f * (float(q) / float(n)), cl);
        add_compensated(va, ea, weight * (x.x * ct - x.y * st));
        add_compensated(vb, eb, weight * (y.x * ct - y.y * st));
        add_compensated(vc, ec, weight * (cross.x * cl - cross.y * sl));
    }
    wa[w * n + t] = va / float(n);
    wb[w * n + t] = vb / float(n);
    corr[w * n + t] = vc / float(n);
}

kernel void stack_windows(device const float *corr [[buffer(0)]],
                          device float *stack [[buffer(1)]],
                          constant uint &n [[buffer(2)]],
                          constant uint &windows [[buffer(3)]],
                          uint t [[thread_position_in_grid]]) {
    if (t >= n) return;
    float sum = 0, err = 0;
    for (uint w = 0; w < windows; ++w) add_compensated(sum, err, corr[w * n + t]);
    stack[t] = sum / float(windows);
}
