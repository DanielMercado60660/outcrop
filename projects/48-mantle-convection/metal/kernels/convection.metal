#include <metal_stdlib>
using namespace metal;

// Preserve cancellation-resistant stencil order; explicit fma remains fused.
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)

// Each FFT is an unnormalized forward DFT of a real odd extension:
// S(a)[k] = sum_i a[i] sin(pi*i*k/N) = -imag(FFT(odd(a)))[k]/2.
// S*S = (N/2) I; the two inverse factors 2/N occur in the z and x solves.
inline uint bit_reversed(uint i, uint length) {
    uint r = 0;
    for (uint p = length; p > 1; p >>= 1) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

inline void fft_line(threadgroup float2* a, device const float2* twiddle,
                     uint length, uint lane, uint width) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = 2; span <= length; span <<= 1) {
        const uint halfspan = span >> 1;
        for (uint b = lane; b < length / 2; b += width) {
            uint j = b % halfspan;
            uint lo = (b / halfspan) * span + j;
            float2 omega = twiddle[j * (length / span)];
            float2 x = a[lo];
            float2 y = a[lo + halfspan];
            float2 v = float2(fma(omega.x, y.x, -omega.y * y.y),
                              fma(omega.x, y.y,  omega.y * y.x));
            a[lo] = x + v;
            a[lo + halfspan] = x - v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void fft_rhs_x(device const float* t [[buffer(0)]],
                      device float* a [[buffer(1)]],
                      device const float2* twiddle [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      constant float& ra [[buffer(4)]],
                      uint lane [[thread_index_in_threadgroup]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float2 line[256];
    uint z = group.x + 1, side = n + 1, length = 2 * n;
    if (z >= n) return;
    for (uint p = lane; p < length; p += threads.x) {
        uint x = p <= n ? p : length - p;
        float value = 0.0f;
        if (x > 0 && x < n)
            value = (ra * (0.5f * float(n))) * (t[z * side + x + 1] - t[z * side + x - 1]);
        if (p > n) value = -value;
        line[bit_reversed(p, length)] = float2(value, 0.0f);
    }
    fft_line(line, twiddle, length, lane, threads.x);
    // Transpose so the vertical transform reads a contiguous line.
    for (uint k = lane + 1; k < n; k += threads.x)
        a[k * side + z] = -0.5f * line[k].y;
}

kernel void fft_solve_z(device const float* a [[buffer(0)]],
                        device float* b [[buffer(1)]],
                        device const float2* twiddle [[buffer(2)]],
                        device const float* inverse [[buffer(3)]],
                        constant uint& n [[buffer(4)]],
                        uint lane [[thread_index_in_threadgroup]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float2 line[256];
    threadgroup float modes[129];
    uint kx = group.x + 1, side = n + 1, length = 2 * n;
    if (kx >= n) return;
    for (uint p = lane; p < length; p += threads.x) {
        uint z = p <= n ? p : length - p;
        float value = (z > 0 && z < n) ? a[kx * side + z] : 0.0f;
        if (p > n) value = -value;
        line[bit_reversed(p, length)] = float2(value, 0.0f);
    }
    fft_line(line, twiddle, length, lane, threads.x);
    for (uint k = lane; k <= n; k += threads.x)
        modes[k] = (k > 0 && k < n) ? -0.5f * line[k].y * inverse[kx * side + k] : 0.0f;
    // Consume all FFT outputs before overwriting the scratch.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = lane; p < length; p += threads.x) {
        float value = modes[p <= n ? p : length - p];
        if (p > n) value = -value;
        line[bit_reversed(p, length)] = float2(value, 0.0f);
    }
    fft_line(line, twiddle, length, lane, threads.x);
    for (uint z = lane + 1; z < n; z += threads.x)
        b[z * side + kx] = -0.5f * line[z].y;
}

kernel void fft_inverse_x(device const float* b [[buffer(0)]],
                          device float* psi [[buffer(1)]],
                          device const float2* twiddle [[buffer(2)]],
                          constant uint& n [[buffer(3)]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float2 line[256];
    uint z = group.x + 1, side = n + 1, length = 2 * n;
    if (z >= n) return;
    for (uint p = lane; p < length; p += threads.x) {
        uint k = p <= n ? p : length - p;
        float value = (k > 0 && k < n) ? b[z * side + k] : 0.0f;
        if (p > n) value = -value;
        line[bit_reversed(p, length)] = float2(value, 0.0f);
    }
    fft_line(line, twiddle, length, lane, threads.x);
    for (uint x = lane + 1; x < n; x += threads.x)
        psi[z * side + x] = -line[x].y / float(n);
    // Wall entries were zero-initialized and are never overwritten.
}

// Direct DST-I transforms for arbitrary N, including the locked odd N=17.
kernel void dense_rhs_x(device const float* t [[buffer(0)]],
                        device float* a [[buffer(1)]],
                        device const float* sine [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        constant float& ra [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= (n - 1) * (n - 1)) return;
    uint z = id / (n - 1) + 1, kx = id % (n - 1) + 1, side = n + 1;
    float sum = 0.0f;
    for (uint x = 1; x < n; ++x) {
        float rhs = (ra * (0.5f * float(n))) * (t[z * side + x + 1] - t[z * side + x - 1]);
        sum = fma(sine[kx * side + x], rhs, sum);
    }
    a[kx * side + z] = sum;
}

kernel void dense_forward_z(device const float* a [[buffer(0)]],
                            device float* b [[buffer(1)]],
                            device const float* sine [[buffer(2)]],
                            device const float* inverse [[buffer(3)]],
                            constant uint& n [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= (n - 1) * (n - 1)) return;
    uint kx = id / (n - 1) + 1, kz = id % (n - 1) + 1, side = n + 1;
    float sum = 0.0f;
    for (uint z = 1; z < n; ++z)
        sum = fma(sine[kz * side + z], a[kx * side + z], sum);
    b[kx * side + kz] = sum * inverse[kx * side + kz];
}

kernel void dense_inverse_z(device const float* b [[buffer(0)]],
                            device float* a [[buffer(1)]],
                            device const float* sine [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= (n - 1) * (n - 1)) return;
    uint z = id / (n - 1) + 1, kx = id % (n - 1) + 1, side = n + 1;
    float sum = 0.0f;
    for (uint kz = 1; kz < n; ++kz)
        sum = fma(sine[kz * side + z], b[kx * side + kz], sum);
    a[z * side + kx] = sum;
}

kernel void dense_inverse_x(device const float* a [[buffer(0)]],
                            device float* psi [[buffer(1)]],
                            device const float* sine [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= (n - 1) * (n - 1)) return;
    uint z = id / (n - 1) + 1, x = id % (n - 1) + 1, side = n + 1;
    float sum = 0.0f;
    for (uint kx = 1; kx < n; ++kx)
        sum = fma(sine[kx * side + x], a[z * side + kx], sum);
    psi[z * side + x] = sum * (2.0f / float(n));
}

inline float2 velocity_at(device const float* psi, uint z, uint x, uint n) {
    uint side = n + 1, id = z * side + x;
    float u = 0.0f, w = 0.0f;
    if (x > 0 && x < n) {
        if (z == 0) u = float(n) * psi[id + side];
        else if (z == n) u = -float(n) * psi[id - side];
        else u = (0.5f * float(n)) * (psi[id + side] - psi[id - side]);
    }
    if (z > 0 && z < n) {
        if (x == 0) w = -float(n) * psi[id + 1];
        else if (x == n) w = float(n) * psi[id - 1];
        else w = -(0.5f * float(n)) * (psi[id + 1] - psi[id - 1]);
    }
    return float2(u, w);
}

kernel void velocity_field(device const float* psi [[buffer(0)]],
                           device float* u [[buffer(1)]],
                           device float* w [[buffer(2)]],
                           constant uint& n [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    uint side = n + 1;
    if (id >= side * side) return;
    float2 v = velocity_at(psi, id / side, id % side, n);
    u[id] = v.x;
    w[id] = v.y;
}

kernel void midpoint_stage(device const float* eval [[buffer(0)]],
                           device const float* base [[buffer(1)]],
                           device const float* psi [[buffer(2)]],
                           device float* result [[buffer(3)]],
                           constant uint& n [[buffer(4)]],
                           constant float& dt [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
    uint side = n + 1;
    if (id >= side * side) return;
    uint z = id / side, x = id % side;
    if (z == 0 || z == n) {
        result[id] = base[id];
        return;
    }
    // Reflected ghosts at x walls. The wall remains an evolved node.
    float left = eval[z * side + (x == 0 ? 1 : x - 1)];
    float right = eval[z * side + (x == n ? n - 1 : x + 1)];
    float down = eval[id - side], up = eval[id + side], center = eval[id];
    float lap = ((left - center) + (right - center) + (down - center) + (up - center)) * float(n * n);
    float tx = (0.5f * float(n)) * (right - left);
    float tz = (0.5f * float(n)) * (up - down);
    float2 v = velocity_at(psi, z, x, n);
    float rate = fma(-v.y, tz, fma(-v.x, tx, lap));
    // Stage 2 aliases base/result but only reads base[id]; eval is the
    // separate midpoint buffer, so neighboring read/write races cannot occur.
    result[id] = fma(dt, rate, base[id]);
}

kernel void observables(device const float* t [[buffer(0)]],
                        device const float* u [[buffer(1)]],
                        device const float* w [[buffer(2)]],
                        device float* result [[buffer(3)]],
                        constant uint& n [[buffer(4)]],
                        uint lane [[thread_index_in_threadgroup]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float3 partial[256];
    if (group.x != 0) return;
    uint side = n + 1;
    float3 sum = float3(0.0f);
    for (uint id = lane; id < side * side; id += threads.x) {
        uint z = id / side, x = id % side;
        float wx = (x == 0 || x == n) ? 0.5f : 1.0f;
        float wz = (z == 0 || z == n) ? 0.5f : 1.0f;
        sum.z += wx * wz * fma(u[id], u[id], w[id] * w[id]);
        if (z == 0) {
            sum.x += wx * (0.5f * float(n)) * (4.0f * t[(n - 1) * side + x] - t[(n - 2) * side + x] - 3.0f * t[n * side + x]);
            sum.y += wx * (0.5f * float(n)) * (3.0f * t[x] - 4.0f * t[side + x] + t[2 * side + x]);
        }
    }
    partial[lane] = sum;
    for (uint i = lane + threads.x; i < 256; i += threads.x) partial[i] = float3(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        for (uint i = lane; i < stride; i += threads.x) partial[i] += partial[i + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        result[0] = partial[0].x / float(n);
        result[1] = partial[0].y / float(n);
        result[2] = sqrt(partial[0].z / float(n * n));
    }
}
