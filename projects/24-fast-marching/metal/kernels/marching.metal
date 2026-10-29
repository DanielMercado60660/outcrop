#include <metal_stdlib>
using namespace metal;

inline int neighbor(uint i, uint axis, int dir, constant uint *p) {
    uint stride = p[6 + axis];
    uint coord = (i / stride) % p[3 + axis];
    if ((dir < 0 && coord == 0) || (dir > 0 && coord + 1 >= p[3 + axis])) return -1;
    return int(i) + dir * int(stride);
}

kernel void initialize(constant uint *p [[buffer(0)]], constant float *h [[buffer(1)]],
                       device const float *phi [[buffer(2)]], device const float *speed [[buffer(3)]],
                       device float *t [[buffer(4)]], device uint *frozen [[buffer(5)]],
                       device int *pos [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    pos[i] = -1;
    float v = phi[i];
    if (v == 0.0f) { t[i] = 0.0f; frozen[i] = 1; return; }
    float sum = 0.0f;
    for (uint a = 0; a < p[1]; ++a) {
        float best = INFINITY;
        for (int dir = -1; dir <= 1; dir += 2) {
            int j = neighbor(i, a, dir, p);
            if (j < 0) continue;
            float q = phi[j];
            bool crossing = (v < 0.0f && q > 0.0f) || (v > 0.0f && q < 0.0f) || (p[2] == 2 && q == 0.0f);
            if (crossing) {
                float denominator = v - q;
                float fraction;
                if (isfinite(denominator)) {
                    fraction = abs(v / denominator);
                } else {
                    float scale = max(abs(v), abs(q));
                    float av = abs(v) / scale, aq = abs(q) / scale;
                    fraction = av / (av + aq);
                }
                best = min(best, h[a] * fraction);
            }
        }
        if (isfinite(best)) { float inv = 1.0f / best; sum += inv * inv; }
    }
    if (sum > 0.0f) { t[i] = (1.0f / sqrt(sum)) / speed[i]; frozen[i] = 1; }
    else { t[i] = INFINITY; frozen[i] = 0; }
}

inline bool less_node(uint x, uint y, device float *t) {
    return t[x] < t[y] || (t[x] == t[y] && x < y);
}
inline void swap_nodes(uint a, uint b, device uint *heap, device int *pos) {
    uint x = heap[a], y = heap[b]; heap[a] = y; heap[b] = x; pos[x] = int(b); pos[y] = int(a);
}
inline void up(uint at, device uint *heap, device int *pos, device float *t) {
    while (at > 0) { uint parent = (at - 1) / 2; if (!less_node(heap[at], heap[parent], t)) break; swap_nodes(at, parent, heap, pos); at = parent; }
}
inline void enqueue(uint i, device uint *heap, device int *pos, device float *t, thread uint &size) {
    if (pos[i] < 0) { pos[i] = int(size); heap[size++] = i; }
    up(uint(pos[i]), heap, pos, t);
}
inline uint pop(device uint *heap, device int *pos, device float *t, thread uint &size) {
    uint result = heap[0]; pos[result] = -1; --size;
    if (size != 0) {
        heap[0] = heap[size]; pos[heap[0]] = 0; uint at = 0;
        while (true) { uint l = 2 * at + 1; if (l >= size) break; uint r = l + 1; uint best = l;
            if (r < size && less_node(heap[r], heap[l], t)) best = r;
            if (!less_node(heap[best], heap[at], t)) break;
            swap_nodes(at, best, heap, pos); at = best;
        }
    }
    return result;
}

inline float root(thread float *a, thread float *w, uint count, float invspeed, bool dropping) {
    for (uint i = 1; i < count; ++i) { float av = a[i], wv = w[i]; uint j = i;
        while (j > 0 && a[j-1] > av) { a[j] = a[j-1]; w[j] = w[j-1]; --j; } a[j] = av; w[j] = wv;
    }
    // Scale time offsets by slowness when its square would overflow.
    bool scaled = invspeed > 1.0e18f;
    float base = a[0], sw = 0.0f, sb = 0.0f;
    float sc = scaled ? -1.0f : -(invspeed * invspeed), value = base;
    for (uint i = 0; i < count; ++i) {
        if (dropping && i > 0 && value < a[i]) break;
        float d = a[i] - base; if (scaled) d /= invspeed;
        sw += w[i]; sb += w[i] * d; sc += w[i] * d * d;
        float disc = sb * sb - sw * sc;
        if (disc < 0.0f) return NAN;
        float offset = (sb + sqrt(disc)) / sw;
        value = base + (scaled ? invspeed * offset : offset);
    }
    return value;
}
inline void update(uint i, constant uint *p, constant float *h, device const float *phi,
                   device const float *speed, device float *t, device uint *frozen,
                   device uint *heap, device int *pos, thread uint &size) {
    if (frozen[i]) return;
    float a[3], w[3], a1[3], w1[3]; uint count = 0;
    for (uint axis = 0; axis < p[1]; ++axis) {
        int neg = neighbor(i, axis, -1, p), positive = neighbor(i, axis, 1, p);
        int best = -1, dir = -1;
        if (neg >= 0 && frozen[neg]) best = neg;
        if (positive >= 0 && frozen[positive] && (best < 0 || t[positive] < t[best])) { best = positive; dir = 1; }
        if (best < 0) continue;
        float weight = 1.0f / (h[axis] * h[axis]);
        a1[count] = t[best]; w1[count] = weight;
        a[count] = a1[count]; w[count] = weight;
        if (p[2] == 2) {
            int second = neighbor(uint(best), axis, dir, p);
            if (second >= 0 && frozen[second] && t[second] <= t[best] &&
                ((phi[i] >= 0.0f && phi[second] >= 0.0f) || (phi[i] <= 0.0f && phi[second] <= 0.0f))) {
                a[count] = t[best] > 8.0e37f
                    ? t[best] + (t[best] - t[second]) / 3.0f
                    : (4.0f * t[best] - t[second]) / 3.0f;
                w[count] = 2.25f * weight;
            }
        }
        ++count;
    }
    if (count == 0) return;
    float inv = 1.0f / speed[i];
    float value = root(a, w, count, inv, p[2] == 2);
    if (isnan(value)) value = root(a1, w1, count, inv, true);
    if (value < t[i]) { t[i] = value; enqueue(i, heap, pos, t, size); }
}

kernel void march(constant uint *p [[buffer(0)]], constant float *h [[buffer(1)]],
                  device const float *phi [[buffer(2)]], device const float *speed [[buffer(3)]],
                  device float *t [[buffer(4)]], device uint *frozen [[buffer(5)]],
                  device uint *heap [[buffer(6)]], device int *pos [[buffer(7)]],
                  device uint *state [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint size = state[0];
    if (state[1] == 0) {
        uint end = min(state[2] + 4096u, p[0]);
        for (uint i = state[2]; i < end; ++i) if (!frozen[i]) update(i, p, h, phi, speed, t, frozen, heap, pos, size);
        state[2] = end;
        state[0] = size;
        if (end < p[0]) return;
        state[1] = 1;
    }
    for (uint step = 0; step < 4096 && size > 0; ++step) {
        uint i = pop(heap, pos, t, size); frozen[i] = 1;
        for (uint a = 0; a < p[1]; ++a) for (int dir = -1; dir <= 1; dir += 2) {
            int j = neighbor(i, a, dir, p);
            if (j >= 0 && !frozen[j]) update(uint(j), p, h, phi, speed, t, frozen, heap, pos, size);
        }
    }
    state[0] = size;
}
