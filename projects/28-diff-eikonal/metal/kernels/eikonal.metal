#include <metal_stdlib>
using namespace metal;

// Preserve the float32 stable-increment arithmetic, including in the loss
// reduction. Runtime compilation also disables fast math via Gpu::compile.
#pragma clang fp contract(off)

struct Minima {
    float z;
    float x;
    int iz;
    int ix;
};

// Exterior neighbors are absent. Visit the negative direction first and
// replace only on strict improvement: equal neighbors choose lower C index.
inline Minima axis_minima(device const float *t, uint i, uint nz, uint nx) {
    uint z = i / nx;
    uint x = i % nx;
    Minima m = {INFINITY, INFINITY, -1, -1};
    if (z > 0) {
        m.iz = int(i - nx);
        m.z = t[i - nx];
    }
    if (z + 1 < nz && t[i + nx] < m.z) {
        m.iz = int(i + nx);
        m.z = t[i + nx];
    }
    if (x > 0) {
        m.ix = int(i - 1);
        m.x = t[i - 1];
    }
    if (x + 1 < nx && t[i + 1] < m.x) {
        m.ix = int(i + 1);
        m.x = t[i + 1];
    }
    return m;
}

inline float godunov_root(Minima m, float s, float dz, float dx) {
    float a = m.z;
    float b = m.x;
    float ha = dz;
    float hb = dx;
    if (b < a) {
        float tmp = a; a = b; b = tmp;
        tmp = ha; ha = hb; hb = tmp;
    }
    if (isinf(a)) return INFINITY;
    float one_axis = a + s * ha;
    if (one_axis <= b) return one_axis;

    // Solve for u = T-a, never a quadratic expanded in absolute time.
    // Keep each intermediate in float32 and retain the API's operation order.
    float delta = b - a;
    float ha2 = ha * ha;
    float hb2 = hb * hb;
    float A = 1.0f / ha2 + 1.0f / hb2;
    float B = (-2.0f * delta) / hb2;
    float C = (delta * delta) / hb2 - s * s;
    float discriminant = B * B - (4.0f * A) * C;
    float u = (-B + sqrt(max(discriminant, 0.0f))) / (2.0f * A);
    return a + u;
}

kernel void initialize(constant uint *p [[buffer(0)]],
                       device const uint *sources [[buffer(1)]],
                       device float *traveltime [[buffer(2)]],
                       device uint2 *rounded_steps [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    uint n = p[4];
    if (tid >= p[2] * n) return;
    traveltime[tid] = tid % n == sources[tid / n] ? 0.0f : INFINITY;
    rounded_steps[tid] = uint2(n, 0u);
}

kernel void sweep_cycle(constant uint *p [[buffer(0)]],
                        constant float *h [[buffer(1)]],
                        device const float *slowness [[buffer(2)]],
                        device const uint *sources [[buffer(3)]],
                        device float *traveltime [[buffer(4)]],
                        device uint *changed [[buffer(5)]],
                        device uint2 *rounded_steps [[buffer(6)]],
                        uint shot [[thread_position_in_grid]]) {
    if (shot >= p[2]) return;
    uint nz = p[0], nx = p[1], n = p[4];
    device float *t = traveltime + shot * n;
    device uint2 *rounded = rounded_steps + shot * n;
    uint source = sources[shot];
    bool any = false;
    for (uint direction = 0; direction < 4; ++direction) {
        for (uint rz = 0; rz < nz; ++rz) {
            uint z = (direction & 2u) ? nz - 1 - rz : rz;
            for (uint rx = 0; rx < nx; ++rx) {
                uint x = (direction & 1u) ? nx - 1 - rx : rx;
                uint i = z * nx + x;
                if (i == source) continue;
                Minima m = axis_minima(t, i, nz, nx);
                float candidate = godunov_root(m, slowness[i], h[0], h[1]);
                if (candidate < t[i]) {
                    t[i] = candidate;
                    rounded[i] = uint2(n, 0u);
                    // When a+s*ha rounds to a, the root selects its one-axis
                    // branch even if both axis minima equal a. Save the
                    // parent selected NOW: final equal minima can otherwise
                    // point back to a child and create a spurious cycle.
                    // A parent that stays equal cannot decrease again, so
                    // its depth remains valid for this equal-time edge.
                    if (candidate == min(m.z, m.x)) {
                        uint parent = uint(m.x < m.z ? m.ix : m.iz);
                        rounded[i] = uint2(parent, rounded[parent].y + 1u);
                    }
                    any = true;
                }
            }
        }
    }
    changed[shot] = any ? 1u : 0u;
}

// Sort nodes by (time, rounded-step depth, C index) on the GPU, then traverse
// backwards for triangular adjoint back substitution. Depth makes children
// precede equal-time parents regardless of source position. Heapsort needs
// only one node-index array per shot and O(n log n) comparisons.
inline bool later(uint a, uint b, device const float *t,
                  device const uint2 *rounded) {
    return t[a] > t[b] || (t[a] == t[b] &&
        (rounded[a].y > rounded[b].y ||
         (rounded[a].y == rounded[b].y && a > b)));
}

inline void sift_down(device uint *order, uint root, uint count,
                      device const float *t, device const uint2 *rounded) {
    while (2u * root + 1u < count) {
        uint child = 2u * root + 1u;
        if (child + 1u < count && later(order[child + 1u], order[child], t, rounded)) ++child;
        if (!later(order[child], order[root], t, rounded)) break;
        uint tmp = order[root];
        order[root] = order[child];
        order[child] = tmp;
        root = child;
    }
}

kernel void discrete_adjoint(constant uint *p [[buffer(0)]],
                             constant float *h [[buffer(1)]],
                             device const float *slowness [[buffer(2)]],
                             device const uint *sources [[buffer(3)]],
                             device const uint *receivers [[buffer(4)]],
                             device const float *observed [[buffer(5)]],
                             device const float *weights [[buffer(6)]],
                             device const float *traveltime [[buffer(7)]],
                             device float *predicted [[buffer(8)]],
                             device float *loss_terms [[buffer(9)]],
                             device float *adjoint_rhs [[buffer(10)]],
                             device float *shot_gradients [[buffer(11)]],
                             device uint *node_order [[buffer(12)]],
                             device const uint2 *rounded_steps [[buffer(13)]],
                             uint shot [[thread_position_in_grid]]) {
    if (shot >= p[2]) return;
    uint nz = p[0], nx = p[1], nr = p[3], n = p[4];
    device const float *t = traveltime + shot * n;
    device float *rhs = adjoint_rhs + shot * n;
    device float *g = shot_gradients + shot * n;
    device uint *order = node_order + shot * n;
    device const uint2 *rounded = rounded_steps + shot * n;
    for (uint i = 0; i < n; ++i) {
        rhs[i] = 0.0f;
        g[i] = 0.0f;
        order[i] = i;
    }

    // Each duplicate receiver is an independent weighted observation.
    for (uint r = 0; r < nr; ++r) {
        uint k = shot * nr + r;
        uint i = receivers[r];
        float prediction = t[i];
        predicted[k] = prediction;
        float residual = prediction - observed[k];
        float square = residual * residual;
        loss_terms[k] = weights[k] * square;
        rhs[i] += weights[k] * residual;
    }

    for (uint start = n / 2; start > 0; --start) sift_down(order, start - 1, n, t, rounded);
    for (uint end = n; end > 1; --end) {
        uint tmp = order[0];
        order[0] = order[end - 1];
        order[end - 1] = tmp;
        sift_down(order, 0, end - 1, t, rounded);
    }

    float hz2 = h[0] * h[0];
    float hx2 = h[1] * h[1];
    for (uint rank = n; rank > 0; --rank) {
        uint i = order[rank - 1];
        // Dirichlet T_source is independent of ALL slownesses. A receiver
        // residual at the source still contributes to the objective above.
        if (i == sources[shot]) continue;
        Minima m = axis_minima(t, i, nz, nx);
        float qz = m.iz >= 0 && t[i] > m.z ? (t[i] - m.z) / hz2 : 0.0f;
        float qx = m.ix >= 0 && t[i] > m.x ? (t[i] - m.x) / hx2 : 0.0f;

        // F_i = 1/2 * [sum_d ((T_i-a_d)_+/h_d)^2 - s_i^2].
        // J_ii = qz+qx, J_i,parent = -q_d, dF_i/ds_i = -s_i.
        // Solve J^T lambda = P^T W(PT-d), then g_i = s_i*lambda_i.
        float diagonal = qz + qx;
        if (diagonal == 0.0f && rounded[i].x < n) {
            // The selected update was T_i = T_parent + s_i*ha, whose
            // derivatives remain dT_i/dT_parent=1 and dT_i/ds_i=ha even
            // when the addition rounded to T_parent. Do not divide 0/0.
            uint parent = rounded[i].x;
            float ha = parent / nx == i / nx ? h[1] : h[0];
            g[i] = rhs[i] * ha;
            rhs[parent] += rhs[i];
            continue;
        }
        float lambda = rhs[i] / diagonal;
        g[i] = slowness[i] * lambda;
        if (qz > 0.0f) rhs[m.iz] += qz * lambda;
        if (qx > 0.0f) rhs[m.ix] += qx * lambda;
    }
}

kernel void reduce_gradient(constant uint *p [[buffer(0)]],
                            device const float *shot_gradients [[buffer(1)]],
                            device float *gradient [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p[4]) return;
    float sum = 0.0f;
    for (uint shot = 0; shot < p[2]; ++shot) sum += shot_gradients[shot * p[4] + i];
    gradient[i] = sum;
}

kernel void reduce_loss(constant uint *p [[buffer(0)]],
                        device const float *terms [[buffer(1)]],
                        device float *objective [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    // Deterministic compensated float32 summation reduces loss-rounding
    // noise in centered finite differences without changing the SUM loss.
    float sum = 0.0f;
    float correction = 0.0f;
    for (uint k = 0; k < p[2] * p[3]; ++k) {
        float y = terms[k] - correction;
        float next = sum + y;
        correction = (next - sum) - y;
        sum = next;
    }
    objective[0] = 0.5f * sum;
}
