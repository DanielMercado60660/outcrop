#include <metal_stdlib>
using namespace metal;

kernel void update_magnetic(device const float *e [[buffer(0)]],
                            device float *hx [[buffer(1)]],
                            device float *hz [[buffer(2)]],
                            constant uint &nx [[buffer(3)]],
                            constant uint &nz [[buffer(4)]],
                            constant float &cx [[buffer(5)]],
                            constant float &cz [[buffer(6)]],
                            uint k [[thread_position_in_grid]]) {
    if (k >= nx * nz) return;
    uint i = k / nz, j = k % nz;
    uint next_z = i * nz + (j + 1 == nz ? 0 : j + 1);
    uint next_x = (i + 1 == nx ? 0 : i + 1) * nz + j;
    hx[k] += cz * (e[next_z] - e[k]);
    hz[k] -= cx * (e[next_x] - e[k]);
}

kernel void update_electric(device float *e [[buffer(0)]],
                            device const float *hx [[buffer(1)]],
                            device const float *hz [[buffer(2)]],
                            device const float *eps [[buffer(3)]],
                            constant uint &nx [[buffer(4)]],
                            constant uint &nz [[buffer(5)]],
                            constant float &cx [[buffer(6)]],
                            constant float &cz [[buffer(7)]],
                            uint k [[thread_position_in_grid]]) {
    if (k >= nx * nz) return;
    uint i = k / nz, j = k % nz;
    uint prev_z = i * nz + (j == 0 ? nz - 1 : j - 1);
    uint prev_x = (i == 0 ? nx - 1 : i - 1) * nz + j;
    e[k] += (cz * (hx[k] - hx[prev_z]) - cx * (hz[k] - hz[prev_x])) / eps[k];
}
