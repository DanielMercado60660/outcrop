#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Params {
    uint nz, nx, nzp, nxp, shots, nt, ns, nr, pad, misfit;
    float ihz, ihx, ihz2, ihx2, dt2;
};
inline uint plane(constant Params& p) { return p.nzp * p.nxp; }
inline uint cells(constant Params& p) { return plane(p) * p.shots; }
inline bool active(uint z, uint x, constant Params& p) {
    return z >= 2 && x >= 2 && z < p.nzp - 2 && x < p.nxp - 2;
}

kernel void extend_model(constant Params& p [[buffer(0)]],
                         device const float* velocity [[buffer(1)]],
                         device const float* scatter [[buffer(2)]],
                         device float2* coefficients [[buffer(3)]],
                         device float* padded_scatter [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= plane(p)) return;
    uint z = uint(clamp(int(i / p.nxp) - int(p.pad), 0, int(p.nz) - 1));
    uint x = uint(clamp(int(i % p.nxp) - int(p.pad), 0, int(p.nx) - 1));
    float v = velocity[z * p.nx + x];
    coefficients[i] = float2(v * v * p.dt2, 2.0f * v * p.dt2);
    padded_scatter[i] = scatter[z * p.nx + x];
}

kernel void record_receivers(constant Params& p [[buffer(0)]],
                             device const float* s [[buffer(1)]],
                             device const uint* indices [[buffer(2)]],
                             device float* predicted [[buffer(3)]],
                             constant uint& k [[buffer(4)]],
                             uint trace [[thread_position_in_grid]]) {
    if (trace < p.shots * p.nr) predicted[trace * p.nt + k] = s[indices[trace]];
}

// The same CFS-PML algebra as projects 13/14, for one axis:
// T=(1+b)D2 q + (Db)D1 q + D1(a psi), H=(1+b)T+a zeta;
// psi'=b D1 q+a psi, zeta'=b T+a zeta.
// Memory planes are (psi_z,zeta_z,psi_x,zeta_x), per wavefield.
inline float axis(device const float* q,
                  device const float* psi, device const float* zeta,
                  device float* next_psi, device float* next_zeta,
                  device const float4* profile,
                  uint i, uint pos, uint stride, float ih, float ih2) {
    float d = ((2.0f / 3.0f) * (q[i + stride] - q[i - stride])
              - (1.0f / 12.0f) * (q[i + 2 * stride] - q[i - 2 * stride])) * ih;
    float lap = ((4.0f / 3.0f) * ((q[i + stride] - q[i]) + (q[i - stride] - q[i]))
                - (1.0f / 12.0f) * ((q[i + 2 * stride] - q[i]) + (q[i - 2 * stride] - q[i]))) * ih2;
    float ap = ((2.0f / 3.0f) * (profile[pos + 1].x * psi[i + stride]
                                                - profile[pos - 1].x * psi[i - stride])
                - (1.0f / 12.0f) * (profile[pos + 2].x * psi[i + 2 * stride]
                                                  - profile[pos - 2].x * psi[i - 2 * stride])) * ih;
    float4 pr = profile[pos]; // a,b,Db,unused
    float t = (1.0f + pr.y) * lap + pr.z * d + ap;
    next_psi[i] = pr.y * d + pr.x * psi[i];
    next_zeta[i] = pr.y * t + pr.x * zeta[i];
    return (1.0f + pr.y) * t + pr.x * zeta[i];
}

kernel void forward_step(constant Params& p [[buffer(0)]],
                          device const float2* coefficients [[buffer(1)]],
                          device const float* scatter [[buffer(2)]],
                          device const float4* pz [[buffer(3)]],
                          device const float4* px [[buffer(4)]],
                          device const float* source [[buffer(5)]],
                          device const uint* offsets [[buffer(6)]],
                          device const uint* ids [[buffer(7)]],
                          device const float* u [[buffer(8)]],
                          device const float* previous_u [[buffer(9)]],
                          device float* next_u [[buffer(10)]],
                          device const float* s [[buffer(11)]],
                          device const float* previous_s [[buffer(12)]],
                          device float* next_s [[buffer(13)]],
                          device const float* memory [[buffer(14)]],
                          device float* next_memory [[buffer(15)]],
                          device float* history [[buffer(16)]],
                          constant uint& k [[buffer(17)]],
                          uint i [[thread_position_in_grid]]) {
    uint n = cells(p);
    if (i >= n) return;
    uint local = i % plane(p), z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) return; // Every guard buffer was initialized to zero.
    float hu = axis(u, memory, memory + n, next_memory, next_memory + n,
                    pz, i, z, p.nxp, p.ihz, p.ihz2)
             + axis(u, memory + 2 * n, memory + 3 * n, next_memory + 2 * n, next_memory + 3 * n,
                    px, i, x, 1, p.ihx, p.ihx2);
    float forcing = 0.0f;
    for (uint j = offsets[i]; j < offsets[i + 1]; ++j) forcing += source[ids[j] * p.nt + k];
    float rhs = hu - forcing;
    history[k * n + i] = rhs;
    float2 c = coefficients[local];
    next_u[i] = 2.0f * u[i] - previous_u[i] + c.x * rhs;
    if (p.misfit) {
        float hs = axis(s, memory + 4 * n, memory + 5 * n, next_memory + 4 * n, next_memory + 5 * n,
                        pz, i, z, p.nxp, p.ihz, p.ihz2)
                 + axis(s, memory + 6 * n, memory + 7 * n, next_memory + 6 * n, next_memory + 7 * n,
                        px, i, x, 1, p.ihx, p.ihx2);
        // Tangent w.r.t. velocity: the source injection also contributes
        // -2 v dt^2 dv f. Scatter is velocity (m/s), not reflectivity.
        next_s[i] = 2.0f * s[i] - previous_s[i] + c.x * hs + c.y * scatter[local] * rhs;
    }
}

// Let l=bar(s^{k+1}). Reverse each local PML expression first, then
// gather the transposed stencils in a separate dispatch. The cotangent
// memory planes use the same order as the forward scattered memory.
// temporary holds (bar(T),bar(D2 s),bar(D1 s)) for each axis.
kernel void adjoint_local(constant Params& p [[buffer(0)]],
                          device const float2* coefficients [[buffer(1)]],
                          device const float4* pz [[buffer(2)]],
                          device const float4* px [[buffer(3)]],
                          device const float* lambda [[buffer(4)]],
                          device const float* memory [[buffer(5)]],
                          device float* temporary [[buffer(6)]],
                          uint i [[thread_position_in_grid]]) {
    uint n = cells(p);
    if (i >= n) return;
    uint local = i % plane(p), z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) {
        for (uint j = 0; j < 6; ++j) temporary[j * n + i] = 0.0f;
        return;
    }
    float w = coefficients[local].x * lambda[i];
    for (uint ax = 0; ax < 2; ++ax) {
        float4 pr = ax == 0 ? pz[z] : px[x];
        float t = (1.0f + pr.y) * w + pr.y * memory[(2 * ax + 1) * n + i];
        temporary[(3 * ax) * n + i] = t;
        temporary[(3 * ax + 1) * n + i] = (1.0f + pr.y) * t;
        temporary[(3 * ax + 2) * n + i] = pr.z * t + pr.y * memory[(2 * ax) * n + i];
    }
}

inline float d1_transpose(device const float* q, uint i, uint stride, float ih) {
    return ((2.0f / 3.0f) * (q[i - stride] - q[i + stride])
            - (1.0f / 12.0f) * (q[i - 2 * stride] - q[i + 2 * stride])) * ih;
}
inline float d2_transpose(device const float* q, uint i, uint stride, float ih2) {
    return ((4.0f / 3.0f) * ((q[i + stride] - q[i]) + (q[i - stride] - q[i]))
            - (1.0f / 12.0f) * ((q[i + 2 * stride] - q[i]) + (q[i - 2 * stride] - q[i]))) * ih2;
}

kernel void adjoint_gather(constant Params& p [[buffer(0)]],
                           device const float2* coefficients [[buffer(1)]],
                           device const float4* pz [[buffer(2)]],
                           device const float4* px [[buffer(3)]],
                           device const float* lambda [[buffer(4)]],
                           device const float* later_lambda [[buffer(5)]],
                           device float* previous_lambda [[buffer(6)]],
                           device const float* memory [[buffer(7)]],
                           device float* previous_memory [[buffer(8)]],
                           device const float* temporary [[buffer(9)]],
                           device const float* history [[buffer(10)]],
                           device float* gradient [[buffer(11)]],
                           device const float* predicted [[buffer(12)]],
                           device const float* data [[buffer(13)]],
                           device const uint* offsets [[buffer(14)]],
                           device const uint* ids [[buffer(15)]],
                           constant uint& k [[buffer(16)]],
                           uint i [[thread_position_in_grid]]) {
    uint n = cells(p);
    if (i >= n) return;
    uint local = i % plane(p), z = local / p.nxp, x = local % p.nxp;
    if (!active(z, x, p)) return;
    float value = 2.0f * lambda[i] - later_lambda[i];
    float w = coefficients[local].x * lambda[i];
    for (uint ax = 0; ax < 2; ++ax) {
        uint stride = ax == 0 ? p.nxp : 1;
        float ih = ax == 0 ? p.ihz : p.ihx, ih2 = ax == 0 ? p.ihz2 : p.ihx2;
        float a = ax == 0 ? pz[z].x : px[x].x;
        value += d2_transpose(temporary + (3 * ax + 1) * n, i, stride, ih2)
               + d1_transpose(temporary + (3 * ax + 2) * n, i, stride, ih);
        previous_memory[(2 * ax) * n + i] = a * (memory[(2 * ax) * n + i]
            + d1_transpose(temporary + (3 * ax) * n, i, stride, ih));
        previous_memory[(2 * ax + 1) * n + i] = a * (w + memory[(2 * ax + 1) * n + i]);
    }
    gradient[i] += coefficients[local].y * history[k * n + i] * lambda[i];
    // This injection builds bar(s^k), for the next reverse iteration.
    // Thus data[0] contributes no scatter gradient, and source[nt-1]
    // has no recorded effect, matching the initial-sample convention.
    for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
        uint sample = ids[j] * p.nt + k;
        value += p.misfit ? predicted[sample] - data[sample] : data[sample];
    }
    previous_lambda[i] = value;
}

inline void accumulate(float value, thread float& sum, thread float& correction) {
    float y = value - correction;
    float t = sum + y;
    correction = (t - sum) - y;
    sum = t;
}

kernel void fold_gradient(constant Params& p [[buffer(0)]],
                          device const float* padded [[buffer(1)]],
                          device float* gradient [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.nz * p.nx) return;
    uint z = i / p.nx, x = i % p.nx;
    // Preimage of edge replication, including all PML copies and corners,
    // excluding the fixed FD guards. Also valid for a one-node axis.
    uint z0 = z == 0 ? 2 : p.pad + z;
    uint z1 = z + 1 == p.nz ? p.nzp - 2 : p.pad + z + 1;
    uint x0 = x == 0 ? 2 : p.pad + x;
    uint x1 = x + 1 == p.nx ? p.nxp - 2 : p.pad + x + 1;
    float sum = 0.0f, correction = 0.0f;
    for (uint shot = 0; shot < p.shots; ++shot)
        for (uint iz = z0; iz < z1; ++iz)
            for (uint ix = x0; ix < x1; ++ix)
                accumulate(padded[shot * plane(p) + iz * p.nxp + ix], sum, correction);
    gradient[i] = sum;
}

kernel void loss_blocks(constant Params& p [[buffer(0)]],
                        device const float* predicted [[buffer(1)]],
                        device const float* data [[buffer(2)]],
                        device float* partial [[buffer(3)]],
                        uint block [[thread_position_in_grid]]) {
    uint n = p.shots * p.nr * p.nt, start = block * 128;
    if (start >= n) return;
    float sum = 0.0f, correction = 0.0f;
    for (uint j = start; j < min(start + 128, n); ++j) {
        float r = predicted[j] - data[j];
        accumulate(r * r, sum, correction);
    }
    partial[block] = sum;
}
kernel void loss_finish(device const float* partial [[buffer(0)]],
                        device float* loss [[buffer(1)]],
                        constant uint& blocks [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    float sum = 0.0f, correction = 0.0f;
    for (uint j = 0; j < blocks; ++j) accumulate(partial[j], sum, correction);
    loss[0] = 0.5f * sum;
}
