#include <metal_stdlib>
using namespace metal;

// All arithmetic, histories, cotangents and reductions are float32.
#pragma clang fp contract(off)

struct Params {
    uint nz, nx, nzp, nxp, shots, nt, receivers, pad;
    float ih, ih2, dt2, inv_m;
};

inline uint plane(constant Params& p) { return p.nzp * p.nxp; }
inline uint cells(constant Params& p) { return p.shots * plane(p); }
inline bool active(uint z, uint x, constant Params& p) {
    return z >= 2 && z < p.nzp - 2 && x >= 2 && x < p.nxp - 2;
}

inline float d1(device const float* u, uint i, uint stride, float ih) {
    return ((2.0f / 3.0f) * (u[i + stride] - u[i - stride])
           - (1.0f / 12.0f) * (u[i + 2 * stride] - u[i - 2 * stride])) * ih;
}

inline float d2(device const float* u, uint i, uint stride, float ih2) {
    return ((4.0f / 3.0f) * (u[i + stride] + u[i - stride] - 2.0f * u[i])
           - (1.0f / 12.0f) * (u[i + 2 * stride] + u[i - 2 * stride] - 2.0f * u[i])) * ih2;
}

// D(a psi): the profile multiplies at the sampled neighbor, before D.
inline float d_memory(device const float4* m, device const float4* profile,
                      uint i, uint q, uint stride, uint component, float ih) {
    float near = profile[q + 1].x * m[i + stride][component]
               - profile[q - 1].x * m[i - stride][component];
    float far = profile[q + 2].x * m[i + 2 * stride][component]
              - profile[q - 2].x * m[i - 2 * stride][component];
    return ((2.0f / 3.0f) * near - (1.0f / 12.0f) * far) * ih;
}

kernel void extend_velocity(constant Params& p [[buffer(0)]],
                            device const float* velocity [[buffer(1)]],
                            device float2* coefficients [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= plane(p)) return;
    int z = clamp(int(i / p.nxp) - int(p.pad), 0, int(p.nz) - 1);
    int x = clamp(int(i % p.nxp) - int(p.pad), 0, int(p.nx) - 1);
    float v = velocity[uint(z) * p.nx + uint(x)];
    coefficients[i] = float2(v * v * p.dt2, 2.0f * v * p.dt2);
}

kernel void record_receivers(constant Params& p [[buffer(0)]],
                             device const float* u [[buffer(1)]],
                             device const uint* receiver_indices [[buffer(2)]],
                             device float* predicted [[buffer(3)]],
                             constant uint& k [[buffer(4)]],
                             uint trace [[thread_position_in_grid]]) {
    if (trace >= p.shots * p.receivers) return;
    predicted[trace * p.nt + k] = u[receiver_indices[trace]];
}

kernel void forward_step(constant Params& p [[buffer(0)]],
                         device const float2* coefficients [[buffer(1)]],
                         device const float4* pz [[buffer(2)]],
                         device const float4* px [[buffer(3)]],
                         device const float* source [[buffer(4)]],
                         device const uint* source_offsets [[buffer(5)]],
                         device const uint* source_ids [[buffer(6)]],
                         device const float* u [[buffer(7)]],
                         device float* delta [[buffer(8)]],
                         device const float4* memory [[buffer(9)]],
                         device float* next_u [[buffer(10)]],
                         device float4* next_memory [[buffer(11)]],
                         device float* history [[buffer(12)]],
                         constant uint& k [[buffer(13)]],
                         uint i [[thread_position_in_grid]]) {
    uint n = cells(p);
    if (i >= n) return;
    uint local = i % plane(p);
    uint z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) {
        next_u[i] = 0.0f;
        delta[i] = 0.0f;
        next_memory[i] = float4(0.0f);
        history[k * n + i] = 0.0f;
        return;
    }
    float4 az = pz[z], ax = px[x]; // (a,b,c,unused)
    float4 m = memory[i];         // (psi_z,psi_x,zeta_z,zeta_x)
    float dz = d1(u, i, p.nxp, p.ih), dx = d1(u, i, 1, p.ih);
    float tz = (1.0f + az.y) * d2(u, i, p.nxp, p.ih2) + az.z * dz
             + d_memory(memory, pz, i, z, p.nxp, 0, p.ih);
    float tx = (1.0f + ax.y) * d2(u, i, 1, p.ih2) + ax.z * dx
             + d_memory(memory, px, i, x, 1, 1, p.ih);
    float sz = (1.0f + az.y) * tz + az.x * m.z;
    float sx = (1.0f + ax.y) * tx + ax.x * m.w;
    float forcing = 0.0f;
    for (uint j = source_offsets[i]; j < source_offsets[i + 1]; ++j)
        forcing += source[source_ids[j] * p.nt + k];
    float rhs = (sz + sx) - forcing;
    history[k * n + i] = rhs;
    next_memory[i] = float4(az.y * dz + az.x * m.x,
                            ax.y * dx + ax.x * m.y,
                            az.y * tz + az.x * m.z,
                            ax.y * tx + ax.x * m.w);

    // delta^k = u^k - u^(k-1). This is the specified centered leapfrog
    // recurrence, algebraically factored to avoid subtracting large u's.
    float next_delta = fma(coefficients[local].x, rhs, delta[i]);
    delta[i] = next_delta;
    next_u[i] = u[i] + next_delta;
}

// Kahan reductions are deterministic and retain the small FD loss differences.
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
    uint n = p.shots * p.receivers * p.nt;
    uint begin = block * 128;
    if (begin >= n) return;
    float sum = 0.0f, correction = 0.0f;
    for (uint j = begin; j < min(begin + 128, n); ++j) {
        float r = predicted[j] - observed[j];
        accumulate(r * r, sum, correction);
    }
    partial[block] = sum;
}

kernel void loss_finish(constant Params& p [[buffer(0)]],
                        device const float* partial [[buffer(1)]],
                        device float* objective [[buffer(2)]],
                        constant uint& blocks [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    float sum = 0.0f, correction = 0.0f;
    for (uint j = 0; j < blocks; ++j) accumulate(partial[j], sum, correction);
    objective[0] = (0.5f * p.inv_m) * sum;
}

// Reverse delta'=delta+C*rhs; u'=u+delta'. Let l = bar(u')+bar(delta').
// T-bar = (1+b)*C*l + b*bar(zeta'); D1u-bar = c*T-bar+b*bar(psi').
// The temporary packs (Tz-bar,Tx-bar,Dzu-bar,Dxu-bar). It is projected to
// zero on guards BEFORE the transpose stencils gather neighboring values.
kernel void adjoint_local(constant Params& p [[buffer(0)]],
                          device const float2* coefficients [[buffer(1)]],
                          device const float4* pz [[buffer(2)]],
                          device const float4* px [[buffer(3)]],
                          device const float* lambda_u [[buffer(4)]],
                          device const float* lambda_delta [[buffer(5)]],
                          device const float4* lambda_memory [[buffer(6)]],
                          device const float* history [[buffer(7)]],
                          device float4* temporary [[buffer(8)]],
                          device float* extended_gradient [[buffer(9)]],
                          constant uint& k [[buffer(10)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= cells(p)) return;
    uint local = i % plane(p);
    uint z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) {
        temporary[i] = float4(0.0f);
        return;
    }
    float4 az = pz[z], ax = px[x], m = lambda_memory[i];
    float l = lambda_u[i] + lambda_delta[i];
    float ls = coefficients[local].x * l;
    float tz = (1.0f + az.y) * ls + az.y * m.z;
    float tx = (1.0f + ax.y) * ls + ax.y * m.w;
    temporary[i] = float4(tz, tx, az.z * tz + az.y * m.x, ax.z * tx + ax.y * m.y);
    extended_gradient[i] = fma(coefficients[local].y * history[k * cells(p) + i],
                                l, extended_gradient[i]);
}

inline float d_component(device const float4* t, uint i, uint stride, uint component, float ih) {
    return ((2.0f / 3.0f) * (t[i + stride][component] - t[i - stride][component])
           - (1.0f / 12.0f) * (t[i + 2 * stride][component] - t[i - 2 * stride][component])) * ih;
}

// D2^T((1+b)*T-bar). D2 is symmetric; profile factors belong inside D2.
inline float d2_weighted(device const float4* t, device const float4* profile,
                         uint i, uint q, uint stride, uint component, float ih2) {
    float center = (1.0f + profile[q].y) * t[i][component];
    float near = (1.0f + profile[q + 1].y) * t[i + stride][component]
               + (1.0f + profile[q - 1].y) * t[i - stride][component];
    float far = (1.0f + profile[q + 2].y) * t[i + 2 * stride][component]
              + (1.0f + profile[q - 2].y) * t[i - 2 * stride][component];
    return ((4.0f / 3.0f) * (near - 2.0f * center)
           - (1.0f / 12.0f) * (far - 2.0f * center)) * ih2;
}

kernel void adjoint_gather(constant Params& p [[buffer(0)]],
                           device const float2* coefficients [[buffer(1)]],
                           device const float4* pz [[buffer(2)]],
                           device const float4* px [[buffer(3)]],
                           device const float4* temporary [[buffer(4)]],
                           device const float4* old_memory [[buffer(5)]],
                           device float4* new_memory [[buffer(6)]],
                           device float* lambda_u [[buffer(7)]],
                           device float* lambda_delta [[buffer(8)]],
                           device const uint* receiver_offsets [[buffer(9)]],
                           device const uint* receiver_ids [[buffer(10)]],
                           device const float* predicted [[buffer(11)]],
                           device const float* observed [[buffer(12)]],
                           constant uint& k [[buffer(13)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= cells(p)) return;
    uint local = i % plane(p);
    uint z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) {
        lambda_u[i] = 0.0f;
        lambda_delta[i] = 0.0f;
        new_memory[i] = float4(0.0f);
        return;
    }
    float4 az = pz[z], ax = px[x], m = old_memory[i];
    float l = lambda_u[i] + lambda_delta[i];
    float ls = coefficients[local].x * l;
    // D1^T = -D1 on the full grid, followed by the guard projection.
    float gz = d2_weighted(temporary, pz, i, z, p.nxp, 0, p.ih2)
              - d_component(temporary, i, p.nxp, 2, p.ih);
    float gx = d2_weighted(temporary, px, i, x, 1, 1, p.ih2)
              - d_component(temporary, i, 1, 3, p.ih);
    new_memory[i] = float4(
        az.x * (m.x - d_component(temporary, i, p.nxp, 0, p.ih)),
        ax.x * (m.y - d_component(temporary, i, 1, 1, p.ih)),
        az.x * (m.z + ls), ax.x * (m.w + ls));
    float residual = 0.0f;
    for (uint j = receiver_offsets[i]; j < receiver_offsets[i + 1]; ++j) {
        uint index = receiver_ids[j] * p.nt + k;
        residual += (predicted[index] - observed[index]) * p.inv_m;
    }
    // Only per-cell reads of the lambda buffers occur in this dispatch;
    // stencil gathers use the completed temporary buffer, so this is race-free.
    lambda_u[i] = ((lambda_u[i] + gz) + gx) + residual;
    lambda_delta[i] = l;
}

kernel void fold_gradient(constant Params& p [[buffer(0)]],
                          device const float* extended_gradient [[buffer(1)]],
                          device float* gradient [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.nz * p.nx) return;
    uint z = i / p.nx, x = i % p.nx;
    // Preimage of clip under edge replication, excluding guards. These
    // bounds also handle a one-cell model axis and all replicated corners.
    uint z0 = z == 0 ? 2 : p.pad + z;
    uint z1 = z + 1 == p.nz ? p.nzp - 2 : p.pad + z + 1;
    uint x0 = x == 0 ? 2 : p.pad + x;
    uint x1 = x + 1 == p.nx ? p.nxp - 2 : p.pad + x + 1;
    float sum = 0.0f, correction = 0.0f;
    for (uint shot = 0; shot < p.shots; ++shot)
        for (uint iz = z0; iz < z1; ++iz)
            for (uint ix = x0; ix < x1; ++ix)
                accumulate(extended_gradient[shot * plane(p) + iz * p.nxp + ix], sum, correction);
    gradient[i] = sum;
}
