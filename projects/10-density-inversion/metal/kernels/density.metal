#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Normalized high/low pairs. Every operation, including Cholesky, uses float32.
// Error-free sums and fma product residuals retain low bits through signed dot
// products and the moderately conditioned normal equations.
typedef float2 Pair;
inline Pair fp(float a) { return Pair(a, 0.0f); }
inline Pair renorm(float a, float b) {
    float s = a + b;
    return Pair(s, b - (s - a));
}
inline Pair addp(Pair a, Pair b) {
    float s = a.x + b.x;
    float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v);
    e += a.y + b.y;
    return renorm(s, e);
}
inline Pair subp(Pair a, Pair b) { return addp(a, -b); }
inline Pair mulp(Pair a, Pair b) {
    float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e += a.x * b.y + a.y * b.x;
    return renorm(p, e);
}
inline Pair divp(Pair a, Pair b) {
    float q = a.x / b.x;
    Pair r = subp(a, mulp(b, fp(q)));
    return renorm(q, (r.x + r.y) / b.x);
}
inline Pair sqrtp(Pair a) {
    float q = sqrt(a.x);
    Pair r = subp(a, mulp(fp(q), fp(q)));
    return renorm(q, (r.x + r.y) / (2.0f * q));
}
inline float value(Pair a) { return a.x + a.y; }

kernel void apply_matrix(device const float* g [[buffer(0)]],
                         device const float* v [[buffer(1)]],
                         device float* result [[buffer(2)]],
                         constant uint& nd [[buffer(3)]],
                         constant uint& nc [[buffer(4)]],
                         constant uint& transpose [[buffer(5)]],
                         uint i [[thread_position_in_grid]]) {
    uint no = transpose ? nc : nd;
    uint ni = transpose ? nd : nc;
    if (i >= no) return;
    Pair s = fp(0.0f);
    for (uint j = 0; j < ni; ++j) {
        float gij = transpose ? g[j * nc + i] : g[i * nc + j];
        s = addp(s, mulp(fp(gij), fp(v[j])));
    }
    result[i] = value(s);
}

// dims = [n_data,n_cells,nx,ny,nz]; scalar pairs = [h/scale,offset/scale,beta],
// scale = max(h,offset). The common length scale cancels from the ratio.
kernel void depth_weights(device const uint* dims [[buffer(0)]],
                          device const Pair* scalars [[buffer(1)]],
                          device Pair* weights [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= dims[1]) return;
    uint k = i / (dims[2] * dims[3]);
    Pair numerator = addp(mulp(fp(0.5f), scalars[0]), scalars[1]);
    Pair depth = mulp(fp(float(dims[4] - k) - 0.5f), scalars[0]);
    weights[i] = divp(numerator, addp(depth, scalars[1]));
}

inline Pair edge(uint i, uint j, uint axis,
                 device const Pair* w, device const float* alpha) {
    return mulp(fp(0.5f * alpha[axis]),
                addp(mulp(w[i], w[i]), mulp(w[j], w[j])));
}

kernel void model_delta(device const float* model [[buffer(0)]],
                        device const float* reference [[buffer(1)]],
                        device Pair* delta [[buffer(2)]],
                        constant uint& nc [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < nc) delta[i] = subp(fp(model[i]), fp(reference[i]));
}

kernel void data_objective(device const float* g [[buffer(0)]],
                           device const float* model [[buffer(1)]],
                           device const float* data [[buffer(2)]],
                           device const float* sigma [[buffer(3)]],
                           device const uint* dims [[buffer(4)]],
                           device Pair* dual [[buffer(5)]],
                           device Pair* terms [[buffer(6)]],
                           uint r [[thread_position_in_grid]]) {
    if (r >= dims[0]) return;
    Pair s = fp(0.0f);
    for (uint j = 0; j < dims[1]; ++j)
        s = addp(s, mulp(fp(g[r * dims[1] + j]), fp(model[j])));
    Pair residual = divp(subp(s, fp(data[r])), fp(sigma[r]));
    dual[r] = divp(residual, fp(sigma[r]));
    terms[r] = mulp(residual, residual);
}

kernel void regularization(device const Pair* delta [[buffer(0)]],
                           device const Pair* w [[buffer(1)]],
                           device const uint* dims [[buffer(2)]],
                           device const float* alpha [[buffer(3)]],
                           device Pair* action [[buffer(4)]],
                           device Pair* terms [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= dims[1]) return;
    uint nx = dims[2], ny = dims[3];
    uint coord[3] = {i % nx, (i / nx) % ny, i / (nx * ny)};
    uint stride[3] = {1, nx, nx * ny};
    Pair q = mulp(mulp(w[i], w[i]), delta[i]);
    Pair phi = mulp(q, delta[i]);
    for (uint a = 0; a < 3; ++a) {
        if (coord[a] > 0) {
            uint j = i - stride[a];
            Pair diff = subp(delta[i], delta[j]);
            q = addp(q, mulp(edge(i, j, a, w, alpha), diff));
        }
        if (coord[a] + 1 < dims[a + 2]) {
            uint j = i + stride[a];
            Pair diff = subp(delta[i], delta[j]);
            Pair ed = edge(i, j, a, w, alpha);
            q = addp(q, mulp(ed, diff));
            // Positive-axis edges only in phi: each unordered pair once.
            phi = addp(phi, mulp(ed, mulp(diff, diff)));
        }
    }
    action[i] = q;
    terms[i] = phi;
}

kernel void hand_gradient(device const float* g [[buffer(0)]],
                          device const Pair* dual [[buffer(1)]],
                          device const Pair* reg [[buffer(2)]],
                          device const Pair* w [[buffer(3)]],
                          device const uint* dims [[buffer(4)]],
                          device const Pair* scalars [[buffer(5)]],
                          device float* gradient [[buffer(6)]],
                          device float* weights [[buffer(7)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= dims[1]) return;
    Pair s = fp(0.0f);
    for (uint r = 0; r < dims[0]; ++r)
        s = addp(s, mulp(fp(g[r * dims[1] + i]), dual[r]));
    gradient[i] = value(mulp(fp(2.0f), addp(s, mulp(scalars[2], reg[i]))));
    weights[i] = value(w[i]);
}

kernel void sum_objective(device const Pair* dterms [[buffer(0)]],
                          device const Pair* mterms [[buffer(1)]],
                          device const uint* dims [[buffer(2)]],
                          device const Pair* scalars [[buffer(3)]],
                          device float* result [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    Pair pd = fp(0.0f), pm = fp(0.0f);
    for (uint r = 0; r < dims[0]; ++r) pd = addp(pd, dterms[r]);
    for (uint j = 0; j < dims[1]; ++j) pm = addp(pm, mterms[j]);
    result[0] = value(addp(pd, mulp(scalars[2], pm)));
    result[1] = value(pd);
    result[2] = value(pm);
}

// Entry of the symmetric smallness + face-difference precision matrix R.
inline Pair reg_entry(uint i, uint j, device const Pair* w,
                      device const uint* dims, device const float* alpha) {
    uint nx = dims[2], ny = dims[3];
    uint coord[3] = {i % nx, (i / nx) % ny, i / (nx * ny)};
    uint stride[3] = {1, nx, nx * ny};
    Pair q = i == j ? mulp(w[i], w[i]) : fp(0.0f);
    for (uint a = 0; a < 3; ++a) {
        if (coord[a] > 0) {
            uint neighbor = i - stride[a];
            Pair ed = edge(i, neighbor, a, w, alpha);
            if (i == j) q = addp(q, ed);
            else if (j == neighbor) q = subp(q, ed);
        }
        if (coord[a] + 1 < dims[a + 2]) {
            uint neighbor = i + stride[a];
            Pair ed = edge(i, neighbor, a, w, alpha);
            if (i == j) q = addp(q, ed);
            else if (j == neighbor) q = subp(q, ed);
        }
    }
    return q;
}

kernel void shifted_data(device const float* g [[buffer(0)]],
                         device const float* data [[buffer(1)]],
                         device const float* sigma [[buffer(2)]],
                         device const float* reference [[buffer(3)]],
                         device const uint* dims [[buffer(4)]],
                         device Pair* dual [[buffer(5)]],
                         uint r [[thread_position_in_grid]]) {
    if (r >= dims[0]) return;
    Pair s = fp(0.0f);
    for (uint j = 0; j < dims[1]; ++j)
        s = addp(s, mulp(fp(g[r * dims[1] + j]), fp(reference[j])));
    Pair residual = subp(fp(data[r]), s);
    dual[r] = divp(divp(residual, fp(sigma[r])), fp(sigma[r]));
}

kernel void normal_system(device const float* g [[buffer(0)]],
                          device const float* sigma [[buffer(1)]],
                          device const Pair* w [[buffer(2)]],
                          device const uint* dims [[buffer(3)]],
                          device const Pair* scalars [[buffer(4)]],
                          device const float* alpha [[buffer(5)]],
                          device const Pair* dual [[buffer(6)]],
                          device Pair* h [[buffer(7)]],
                          device Pair* rhs [[buffer(8)]],
                          uint2 ij [[thread_position_in_grid]]) {
    uint i = ij.x, j = ij.y, n = dims[1];
    if (i >= n || j > i) return;
    Pair s = fp(0.0f), b = fp(0.0f);
    for (uint r = 0; r < dims[0]; ++r) {
        Pair gi = fp(g[r * n + i]), gj = fp(g[r * n + j]);
        s = addp(s, mulp(divp(gi, fp(sigma[r])), divp(gj, fp(sigma[r]))));
        if (j == 0) b = addp(b, mulp(gi, dual[r]));
    }
    h[i * n + j] = addp(s, mulp(scalars[2], reg_entry(i, j, w, dims, alpha)));
    if (j == 0) rhs[i] = b;
}

// Left-looking Cholesky overwrites the lower triangle in place. Dispatch
// boundaries order the pivot and column; each column element has one writer.
kernel void cholesky_pivot(device Pair* l [[buffer(0)]],
                           constant uint& n [[buffer(1)]],
                           constant uint& k [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    Pair s = l[k * n + k];
    for (uint t = 0; t < k; ++t) s = subp(s, mulp(l[k * n + t], l[k * n + t]));
    l[k * n + k] = sqrtp(s);
}

kernel void cholesky_column(device Pair* l [[buffer(0)]],
                            constant uint& n [[buffer(1)]],
                            constant uint& k [[buffer(2)]],
                            uint offset [[thread_position_in_grid]]) {
    uint i = k + 1 + offset;
    if (i >= n) return;
    Pair s = l[i * n + k];
    for (uint t = 0; t < k; ++t) s = subp(s, mulp(l[i * n + t], l[k * n + t]));
    l[i * n + k] = divp(s, l[k * n + k]);
}

kernel void triangular_solve(device const Pair* l [[buffer(0)]],
                             device const Pair* rhs [[buffer(1)]],
                             device Pair* work [[buffer(2)]],
                             device Pair* delta [[buffer(3)]],
                             constant uint& n [[buffer(4)]],
                             uint id [[thread_position_in_grid]]) {
    if (id != 0) return;
    for (uint i = 0; i < n; ++i) {
        Pair s = rhs[i];
        for (uint j = 0; j < i; ++j) s = subp(s, mulp(l[i * n + j], work[j]));
        work[i] = divp(s, l[i * n + i]);
    }
    for (uint t = 0; t < n; ++t) {
        uint i = n - 1 - t;
        Pair s = work[i];
        for (uint j = i + 1; j < n; ++j) s = subp(s, mulp(l[j * n + i], delta[j]));
        delta[i] = divp(s, l[i * n + i]);
    }
}

kernel void finish_model(device const Pair* delta [[buffer(0)]],
                         device const float* reference [[buffer(1)]],
                         device float* model [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) model[i] = value(addp(delta[i], fp(reference[i])));
}
