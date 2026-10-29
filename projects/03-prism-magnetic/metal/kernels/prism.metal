#include <metal_stdlib>
using namespace metal;

// Outside a uniformly magnetized prism,
// B_i = mu_0/(4*pi) sum_j M_j d_i d_j integral_prism 1/r dV.
// The six independent tensor entries are evaluated at the eight corners:
// xx = -atan(y*z/(x*r)), xy = log(r+z), and cyclic permutations.
// The arctangent uses its principal branch, not atan2's full angular range.
inline float prism_atan(float numerator, float denominator) {
    if (denominator == 0.0f) {
        if (numerator > 0.0f) return 1.5707963267948966f;
        if (numerator < 0.0f) return -1.5707963267948966f;
        return 0.0f;
    }
    return atan(numerator / denominator);
}

inline float prism_log(float a, float b, float c, float r) {
    // Rewrite log(r+c) for negative c to avoid subtractive cancellation.
    if (c < 0.0f) {
        float transverse = a * a + b * b;
        if (transverse == 0.0f) return -log(-2.0f * c);
        return log(transverse / (r - c));
    }
    if (r + c == 0.0f) return 0.0f;
    return log(r + c);
}

kernel void prism_magnetic(
    device const float *coordinates [[buffer(0)]],
    device const float *prisms [[buffer(1)]],
    device const float *magnetization [[buffer(2)]],
    device const float *field [[buffer(3)]],
    device float *be [[buffer(4)]], device float *bn [[buffer(5)]],
    device float *bu [[buffer(6)]], device float *total [[buffer(7)]],
    constant uint &n [[buffer(8)]], constant uint &count [[buffer(9)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    // Widen before multiplying: N and P fit uint, but flattened offsets may
    // exceed it. Metal ulong keeps all address arithmetic in 64 bits.
    const ulong coordinate_offset = ulong(i) * 3;
    float ox = coordinates[coordinate_offset], oy = coordinates[coordinate_offset+1], oz = coordinates[coordinate_offset+2];
    float bx = 0.0f, by = 0.0f, bz = 0.0f;
    for (uint p = 0; p < count; ++p) {
        const ulong magnetization_offset = ulong(p) * 3;
        const ulong prism_offset = ulong(p) * 6;
        float mx = magnetization[magnetization_offset], my = magnetization[magnetization_offset+1], mz = magnetization[magnetization_offset+2];
        if (mx == 0.0f && my == 0.0f && mz == 0.0f) continue;
        float values[2][2][2][6];
        for (uint a = 0; a < 2; ++a) {
            // Subtract first, including for translated geometries: every
            // distance and transcendental argument uses source-local offsets.
            float x = prisms[prism_offset+a] - ox;
            for (uint b = 0; b < 2; ++b) {
                float y = prisms[prism_offset+2+b] - oy;
                for (uint c = 0; c < 2; ++c) {
                    float z = prisms[prism_offset+4+c] - oz;
                    float r = sqrt(x*x + y*y + z*z);
                    values[a][b][c][0] = -prism_atan(y*z, x*r);
                    values[a][b][c][1] = -prism_atan(x*z, y*r);
                    values[a][b][c][2] = -prism_atan(x*y, z*r);
                    values[a][b][c][3] = prism_log(x, y, z, r);
                    values[a][b][c][4] = prism_log(x, z, y, r);
                    values[a][b][c][5] = prism_log(y, z, x, r);
                }
            }
        }
        float t[6];
        for (uint j = 0; j < 6; ++j) {
            // Delta_x Delta_y Delta_z: upper-minus-lower in each axis.
            // Pair nearby terms before the remaining corner cancellation.
            float d00 = values[0][0][1][j] - values[0][0][0][j];
            float d01 = values[0][1][1][j] - values[0][1][0][j];
            float d10 = values[1][0][1][j] - values[1][0][0][j];
            float d11 = values[1][1][1][j] - values[1][1][0][j];
            t[j] = (d11 - d10) - (d01 - d00);
        }
        float xx = t[0], yy = t[1], zz = t[2], xy = t[3], xz = t[4], yz = t[5];
        // The prism potential is even about each of its three midplanes.
        // A mixed derivative is odd in both differentiated coordinates, so
        // it vanishes on either corresponding midplane. Enforce this exact
        // identity before applying magnetization: alternating float32 logs
        // can otherwise leave a nonzero residual in an exactly zero field.
        // Compare opposite source-local offsets, without a rounded global
        // midpoint or a tolerance that could erase a small off-axis signal.
        bool center_x = (prisms[prism_offset] - ox) == -(prisms[prism_offset+1] - ox);
        bool center_y = (prisms[prism_offset+2] - oy) == -(prisms[prism_offset+3] - oy);
        bool center_z = (prisms[prism_offset+4] - oz) == -(prisms[prism_offset+5] - oz);
        if (center_x || center_y) xy = 0.0f;
        if (center_x || center_z) xz = 0.0f;
        if (center_y || center_z) yz = 0.0f;
        // mu_0/(4*pi) * 1e9 converts A/m to nT: 100.
        bx += 100.0f * (xx*mx + xy*my + xz*mz);
        by += 100.0f * (xy*mx + yy*my + yz*mz);
        bz += 100.0f * (xz*mx + yz*my + zz*mz);
    }
    be[i] = bx; bn[i] = by; bu[i] = bz;
    // Project the same float32 components returned above onto the supplied
    // ENU direction. Negative upward means a downward inducing inclination.
    total[i] = bx*field[0] + by*field[1] + bz*field[2];
}
