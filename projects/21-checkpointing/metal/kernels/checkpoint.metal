#include <metal_stdlib>
using namespace metal;

// The runtime also disables fast math. Keep accumulation order deterministic.
#pragma clang fp contract(off)

struct Params {
    uint nz, nx, nzp, nxp, shots, nt, receivers, radius, zoff, xoff;
    float iz, ix, iz2, ix2, dt2;
};

inline uint plane(constant Params& p) { return p.nzp * p.nxp; }
inline uint cells(constant Params& p) { return p.shots * plane(p); }
inline bool active(uint z, uint x, constant Params& p) {
    return z >= p.radius && z < p.nzp - p.radius
        && x >= p.radius && x < p.nxp - p.radius;
}
inline float c1(uint j, uint r) {
    return r == 1 ? 0.5f : (j == 1 ? 2.0f / 3.0f : -1.0f / 12.0f);
}
inline float c2(uint j, uint r) {
    return r == 1 ? 1.0f : (j == 1 ? 4.0f / 3.0f : -1.0f / 12.0f);
}

inline float d1(device const float* u, uint i, uint stride, uint r, float ih) {
    float v = 0.0f;
    for (uint j = 1; j <= r; ++j)
        v += c1(j, r) * (u[i + j * stride] - u[i - j * stride]);
    return v * ih;
}
inline float d2(device const float* u, uint i, uint stride, uint r, float ih2) {
    float v = 0.0f;
    for (uint j = 1; j <= r; ++j)
        v += c2(j, r) * ((u[i + j * stride] - u[i]) + (u[i - j * stride] - u[i]));
    return v * ih2;
}

// D(a*psi), with the fixed profile multiplying BEFORE differentiation.
inline float d_memory(device const float4* m, device const float4* profile,
                      uint i, uint q, uint stride, uint component, uint r, float ih) {
    float v = 0.0f;
    for (uint j = 1; j <= r; ++j)
        v += c1(j, r) * (profile[q + j].x * m[i + j * stride][component]
                        - profile[q - j].x * m[i - j * stride][component]);
    return v * ih;
}

// Return (D_z u, D_x u, T_z, T_x). This exact helper is used by the
// original descent, every replay, and the velocity part of the one-step VJP.
inline float4 spatial(constant Params& p, device const float* u,
                      device const float4* m, device const float4* pz,
                      device const float4* px, uint i, uint z, uint x) {
    float4 az = pz[z], ax = px[x];
    float dz = d1(u, i, p.nxp, p.radius, p.iz);
    float dx = d1(u, i, 1, p.radius, p.ix);
    float tz = (1.0f + az.y) * d2(u, i, p.nxp, p.radius, p.iz2) + az.z * dz
             + d_memory(m, pz, i, z, p.nxp, 0, p.radius, p.iz);
    float tx = (1.0f + ax.y) * d2(u, i, 1, p.radius, p.ix2) + ax.z * dx
             + d_memory(m, px, i, x, 1, 1, p.radius, p.ix);
    return float4(dz, dx, tz, tx);
}

kernel void extend_velocity(constant Params& p [[buffer(0)]],
                            device const float* velocity [[buffer(1)]],
                            device float2* coefficients [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= plane(p)) return;
    int z = clamp(int(i / p.nxp) - int(p.zoff), 0, int(p.nz) - 1);
    int x = clamp(int(i % p.nxp) - int(p.xoff), 0, int(p.nx) - 1);
    float v = velocity[uint(z) * p.nx + uint(x)];
    coefficients[i] = float2((v * v) * p.dt2, (2.0f * v) * p.dt2);
}

kernel void record_receivers(constant Params& p [[buffer(0)]],
                             device const float* u [[buffer(1)]],
                             device const uint* indices [[buffer(2)]],
                             device float* predicted [[buffer(3)]],
                             constant uint& k [[buffer(4)]],
                             uint trace [[thread_position_in_grid]]) {
    if (trace >= p.shots * p.receivers) return;
    predicted[trace * p.nt + k] = u[indices[trace]];
}

kernel void copy_state(constant Params& p [[buffer(0)]],
                       device const float* u [[buffer(1)]],
                       device const float* prev [[buffer(2)]],
                       device const float4* memory [[buffer(3)]],
                       device float* out_u [[buffer(4)]],
                       device float* out_prev [[buffer(5)]],
                       device float4* out_memory [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= cells(p)) return;
    out_u[i] = u[i];
    out_prev[i] = prev[i];
    out_memory[i] = memory[i];
}

kernel void clear_state(constant Params& p [[buffer(0)]],
                        device float* u [[buffer(1)]],
                        device float* prev [[buffer(2)]],
                        device float4* memory [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= cells(p)) return;
    u[i] = 0.0f;
    prev[i] = 0.0f;
    memory[i] = float4(0.0f);
}

kernel void forward_step(constant Params& p [[buffer(0)]],
                         device const float2* coefficients [[buffer(1)]],
                         device const float4* pz [[buffer(2)]],
                         device const float4* px [[buffer(3)]],
                         device const float* source [[buffer(4)]],
                         device const uint* source_offsets [[buffer(5)]],
                         device const uint* source_ids [[buffer(6)]],
                         device const float* u [[buffer(7)]],
                         device const float* prev [[buffer(8)]],
                         device const float4* memory [[buffer(9)]],
                         device float* next_u [[buffer(10)]],
                         device float* next_prev [[buffer(11)]],
                         device float4* next_memory [[buffer(12)]],
                         constant uint& k [[buffer(13)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= cells(p)) return;
    uint local = i % plane(p), z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) {
        next_u[i] = 0.0f;
        next_prev[i] = 0.0f;
        next_memory[i] = float4(0.0f);
        return;
    }
    float4 az = pz[z], ax = px[x], m = memory[i];
    float4 s = spatial(p, u, memory, pz, px, i, z, x);
    float sz = (1.0f + az.y) * s.z + az.x * m.z;
    float sx = (1.0f + ax.y) * s.w + ax.x * m.w;
    float value = (2.0f * u[i] - prev[i]) + coefficients[local].x * (sz + sx);
    for (uint j = source_offsets[i]; j < source_offsets[i + 1]; ++j)
        value -= coefficients[local].x * source[source_ids[j] * p.nt + k];
    next_u[i] = value;
    next_prev[i] = u[i];
    next_memory[i] = float4(az.y * s.x + az.x * m.x,
                            ax.y * s.y + ax.x * m.y,
                            az.y * s.z + az.x * m.z,
                            ax.y * s.w + ax.x * m.w);
}

inline void accumulate(float value, thread float& sum, thread float& correction) {
    float y = value - correction;
    float t = sum + y;
    correction = (t - sum) - y;
    sum = t;
}

kernel void loss_blocks(constant Params& p [[buffer(0)]],
                        device const float* predicted [[buffer(1)]],
                        device const float* observed [[buffer(2)]],
                        device float* partial [[buffer(3)]],
                        uint block [[thread_position_in_grid]]) {
    uint n = p.shots * p.receivers * p.nt, begin = block * 128;
    if (begin >= n) return;
    float sum = 0.0f, correction = 0.0f;
    for (uint j = begin; j < min(begin + 128, n); ++j) {
        float r = predicted[j] - observed[j];
        accumulate(r * r, sum, correction);
    }
    partial[block] = sum;
}

kernel void loss_finish(device const float* partial [[buffer(0)]],
                        device float* objective [[buffer(1)]],
                        constant uint& blocks [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    float sum = 0.0f, correction = 0.0f;
    for (uint j = 0; j < blocks; ++j) accumulate(partial[j], sum, correction);
    objective[0] = 0.5f * sum;
}

// Reverse the same one-step map for ALL budgets, using only its input state.
// Let l=bar(u_next), ls=C*l. T-bar=(1+b)*ls+b*bar(zeta_next),
// D1u-bar=D(b)*T-bar+b*bar(psi_next). Guards are projected out before
// transpose stencils. No forward state is advanced or saved during this VJP.
kernel void adjoint_local(constant Params& p [[buffer(0)]],
                          device const float2* coefficients [[buffer(1)]],
                          device const float4* pz [[buffer(2)]],
                          device const float4* px [[buffer(3)]],
                          device const float* lambda_u [[buffer(4)]],
                          device const float4* lambda_memory [[buffer(5)]],
                          device const float* u [[buffer(6)]],
                          device const float4* memory [[buffer(7)]],
                          device const float* source [[buffer(8)]],
                          device const uint* source_offsets [[buffer(9)]],
                          device const uint* source_ids [[buffer(10)]],
                          device float4* temporary [[buffer(11)]],
                          device float* extended_gradient [[buffer(12)]],
                          constant uint& k [[buffer(13)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= cells(p)) return;
    uint local = i % plane(p), z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) { temporary[i] = float4(0.0f); return; }
    float4 az = pz[z], ax = px[x], lm = lambda_memory[i];
    float l = lambda_u[i], ls = coefficients[local].x * l;
    float tz = (1.0f + az.y) * ls + az.y * lm.z;
    float tx = (1.0f + ax.y) * ls + ax.y * lm.w;
    temporary[i] = float4(tz, tx, az.z * tz + az.y * lm.x, ax.z * tx + ax.y * lm.y);
    float4 s = spatial(p, u, memory, pz, px, i, z, x), m = memory[i];
    float sz = (1.0f + az.y) * s.z + az.x * m.z;
    float sx = (1.0f + ax.y) * s.w + ax.x * m.w;
    float rhs = sz + sx;
    // Includes the -v(source)^2 dt^2 f velocity derivative.
    for (uint j = source_offsets[i]; j < source_offsets[i + 1]; ++j)
        rhs -= source[source_ids[j] * p.nt + k];
    extended_gradient[i] += (coefficients[local].y * rhs) * l;
}

inline float d_component(device const float4* t, uint i, uint stride,
                          uint component, uint r, float ih) {
    float v = 0.0f;
    for (uint j = 1; j <= r; ++j)
        v += c1(j, r) * (t[i + j * stride][component] - t[i - j * stride][component]);
    return v * ih;
}

// D2^T((1+b)*T-bar): D2 is symmetric, and b multiplies the cotangent
// at its originating output cell. This matters at asymmetric PML faces.
inline float d2_weighted(device const float4* t, device const float4* profile,
                         uint i, uint q, uint stride, uint component, uint r, float ih2) {
    float v = 0.0f, center = (1.0f + profile[q].y) * t[i][component];
    for (uint j = 1; j <= r; ++j) {
        float hi = (1.0f + profile[q + j].y) * t[i + j * stride][component];
        float lo = (1.0f + profile[q - j].y) * t[i - j * stride][component];
        v += c2(j, r) * ((hi - center) + (lo - center));
    }
    return v * ih2;
}

kernel void adjoint_gather(constant Params& p [[buffer(0)]],
                           device const float2* coefficients [[buffer(1)]],
                           device const float4* pz [[buffer(2)]],
                           device const float4* px [[buffer(3)]],
                           device const float4* temporary [[buffer(4)]],
                           device const float4* old_memory [[buffer(5)]],
                           device float4* new_memory [[buffer(6)]],
                           device float* lambda_u [[buffer(7)]],
                           device float* lambda_prev [[buffer(8)]],
                           device const uint* receiver_offsets [[buffer(9)]],
                           device const uint* receiver_ids [[buffer(10)]],
                           device const float* predicted [[buffer(11)]],
                           device const float* observed [[buffer(12)]],
                           constant uint& k [[buffer(13)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= cells(p)) return;
    uint local = i % plane(p), z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) {
        lambda_u[i] = 0.0f;
        lambda_prev[i] = 0.0f;
        new_memory[i] = float4(0.0f);
        return;
    }
    float4 az = pz[z], ax = px[x], lm = old_memory[i];
    float l = lambda_u[i], ls = coefficients[local].x * l;
    float gz = d2_weighted(temporary, pz, i, z, p.nxp, 0, p.radius, p.iz2)
              - d_component(temporary, i, p.nxp, 2, p.radius, p.iz);
    float gx = d2_weighted(temporary, px, i, x, 1, 1, p.radius, p.ix2)
              - d_component(temporary, i, 1, 3, p.radius, p.ix);
    // D1^T=-D1 and (D1*diag(a))^T=-diag(a)*D1.
    new_memory[i] = float4(
        az.x * (lm.x - d_component(temporary, i, p.nxp, 0, p.radius, p.iz)),
        ax.x * (lm.y - d_component(temporary, i, 1, 1, p.radius, p.ix)),
        az.x * (lm.z + ls), ax.x * (lm.w + ls));
    float residual = 0.0f;
    for (uint j = receiver_offsets[i]; j < receiver_offsets[i + 1]; ++j) {
        uint index = receiver_ids[j] * p.nt + k;
        residual += predicted[index] - observed[index];
    }
    // Gather only reads neighboring values from completed temporary, never
    // neighboring lambda buffers: these in-place scalar writes are race free.
    lambda_u[i] = (((2.0f * l + lambda_prev[i]) + gz) + gx) + residual;
    lambda_prev[i] = -l;
}

kernel void fold_gradient(constant Params& p [[buffer(0)]],
                          device const float* extended_gradient [[buffer(1)]],
                          device float* gradient [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.nz * p.nx) return;
    uint z = i / p.nx, x = i % p.nx;
    // Each original cell owns its full replication preimage, corners included.
    // These bounds also work for a physical axis containing only one cell.
    uint z0 = z == 0 ? p.radius : p.zoff + z;
    uint z1 = z + 1 == p.nz ? p.nzp - p.radius : p.zoff + z + 1;
    uint x0 = x == 0 ? p.radius : p.xoff + x;
    uint x1 = x + 1 == p.nx ? p.nxp - p.radius : p.xoff + x + 1;
    float sum = 0.0f, correction = 0.0f;
    for (uint shot = 0; shot < p.shots; ++shot)
        for (uint iz = z0; iz < z1; ++iz)
            for (uint ix = x0; ix < x1; ++ix)
                accumulate(extended_gradient[shot * plane(p) + iz * p.nxp + ix], sum, correction);
    gradient[i] = sum;
}
