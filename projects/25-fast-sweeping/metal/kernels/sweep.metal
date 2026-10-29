#include <metal_stdlib>
using namespace metal;

kernel void initialize(device const float *phi [[buffer(0)]],
                       device const float *speed [[buffer(1)]],
                       device float *time [[buffer(2)]], device uint *fixed [[buffer(3)]],
                       constant uint &n0 [[buffer(4)]], constant uint &n1 [[buffer(5)]],
                       constant uint &n2 [[buffer(6)]], constant float &h [[buffer(7)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= n0*n1*n2) return;
    float p = phi[id];
    if (p == 0.0f) { time[id] = 0.0f; fixed[id] = 1; return; }
    uint coords[3] = { id / (n1*n2), (id / n2) % n1, id % n2 };
    uint sizes[3] = { n0, n1, n2 };
    uint strides[3] = { n1*n2, n2, 1 };
    float distances[3] = { INFINITY, INFINITY, INFINITY };
    float smallest = INFINITY;
    for (uint axis = 0; axis < 3; ++axis) {
        for (int dir = -1; dir <= 1; dir += 2) {
            if ((dir < 0 && coords[axis] == 0) ||
                (dir > 0 && coords[axis] + 1 >= sizes[axis])) continue;
            uint neighbor = dir < 0 ? id - strides[axis] : id + strides[axis];
            float q = phi[neighbor];
            if (q == 0.0f || (p < 0.0f && q > 0.0f) || (p > 0.0f && q < 0.0f)) {
                float ap = abs(p), aq = abs(q);
                float ratio = ap >= aq ? aq / ap : ap / aq;
                float fraction = ap >= aq ? 1.0f / (1.0f + ratio) : ratio / (1.0f + ratio);
                distances[axis] = min(distances[axis], h * fraction);
            }
        }
        smallest = min(smallest, distances[axis]);
    }
    if (isfinite(smallest)) {
        if (smallest == 0.0f) { time[id] = NAN; fixed[id] = 1; return; }
        float sum = 0.0f;
        for (uint axis = 0; axis < 3; ++axis) {
            float ratio = smallest / distances[axis];
            sum += ratio * ratio;
        }
        time[id] = (smallest / sqrt(sum)) / speed[id];
        fixed[id] = 1;
    } else {
        time[id] = INFINITY;
        fixed[id] = 0;
    }
}

kernel void sweep(device const float *speed [[buffer(0)]],
                  device float *time [[buffer(1)]], device const uint *fixed [[buffer(2)]],
                  constant uint &n0 [[buffer(3)]], constant uint &n1 [[buffer(4)]],
                  constant uint &n2 [[buffer(5)]], constant float &h [[buffer(6)]],
                  constant uint &parity [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= n0*n1*n2 || fixed[id]) return;
    uint x = id / (n1*n2), y = (id / n2) % n1, z = id % n2;
    if (((x + y + z) & 1u) != parity) return;
    float a = INFINITY, b = INFINITY, c = INFINITY;
    uint s = n1*n2;
    if (x > 0) a = time[id-s];
    if (x+1 < n0) a = min(a, time[id+s]);
    if (y > 0) b = time[id-n2];
    if (y+1 < n1) b = min(b, time[id+n2]);
    if (z > 0) c = time[id-1];
    if (z+1 < n2) c = min(c, time[id+1]);
    if (a > b) { float t = a; a = b; b = t; }
    if (b > c) { float t = b; b = c; c = t; }
    if (a > b) { float t = a; a = b; b = t; }
    if (!isfinite(a)) return;
    float hs = h / speed[id];
    float update = a + hs;
    if (update > b) {
        float d = (b - a) / hs;
        update = a + hs * ((d + sqrt(max(0.0f, 2.0f - d*d))) * 0.5f);
        if (update > c) {
            float e = (c - a) / hs;
            float disc = 3.0f - 2.0f*(d*d + e*e - d*e);
            update = a + hs * ((d + e + sqrt(max(0.0f, disc))) / 3.0f);
        }
    }
    time[id] = min(time[id], update);
}

