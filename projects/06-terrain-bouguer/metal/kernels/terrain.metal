#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Continuous extension of log(x+r), including the negative axis.
inline float safe_log(float x, float y, float z, float r) {
    if (r == 0.0f) return 0.0f;
    if (x < 0.0f) {
        if (y == 0.0f && z == 0.0f) return -log(-2.0f*x);
        return log((y*y + z*z)/(r-x));
    }
    return log(x+r);
}
inline float safe_atan(float y, float x) {
    if (x == 0.0f) {
        if (y > 0.0f) return 1.5707963267948966f;
        if (y < 0.0f) return -1.5707963267948966f;
        return 0.0f;
    }
    return atan(y/x);
}
// Downward component of the rectangular-prism corner primitive.
inline float gz_corner(float x, float y, float z) {
    float r = sqrt(x*x + y*y + z*z);
    return x*safe_log(y,z,x,r) + y*safe_log(x,y,z,r) - z*safe_atan(x*y,z*r);
}
// For positive distances a and b=a+h below the observer, integrate vertically:
//   integral_a^b t/(x*x+y*y+t*t)^(3/2) dt = 1/r_a - 1/r_b
//     = h*(a+b)/(r_a*r_b*(r_a+r_b)).
// The rationalized expression retains thin and far contributions in float32.
// Only the smooth horizontal integral needs quadrature (6-point GL per axis).
inline float integrated_prism(float xc, float yc, float top, float dx, float dy, float thickness, uint tiles) {
    const float nodes[6] = {
        -0.9324695142031521f, -0.6612093864662645f, -0.2386191860831969f,
         0.2386191860831969f,  0.6612093864662645f,  0.9324695142031521f
    };
    const float weights[6] = {
        0.1713244923791704f, 0.3607615730481386f, 0.4679139345726910f,
        0.4679139345726910f, 0.3607615730481386f, 0.1713244923791704f
    };
    float sx = dx / float(tiles), sy = dy / float(tiles);
    float bottom = top + thickness;
    float total = 0.0f;
    for (uint j = 0; j < tiles; ++j) {
        float cy = yc + ((float(j)+0.5f)*sy - 0.5f*dy);
        for (uint i = 0; i < tiles; ++i) {
            float cx = xc + ((float(i)+0.5f)*sx - 0.5f*dx);
            float part = 0.0f;
            for (uint a = 0; a < 6; ++a) {
                float x = cx + 0.5f*sx*nodes[a];
                for (uint b = 0; b < 6; ++b) {
                    float y = cy + 0.5f*sy*nodes[b];
                    float horizontal2 = x*x + y*y;
                    float rt = sqrt(horizontal2 + top*top);
                    float rb = sqrt(horizontal2 + bottom*bottom);
                    // Factor the rationalized expression to avoid cubic
                    // distance products and retain thickness independently.
                    float vertical = (thickness/rt)*((top+bottom)/rb)/(rt+rb);
                    part += weights[a]*weights[b]*vertical;
                }
            }
            total += part;
        }
    }
    return total*(0.25f*sx*sy);
}
kernel void terrain_bouguer(
    device const float *east [[buffer(0)]], device const float *north [[buffer(1)]],
    device const float *surface [[buffer(2)]], device const float *reference [[buffer(3)]],
    device const float *density [[buffer(4)]], device const float *coordinates [[buffer(5)]],
    device float *layer [[buffer(6)]], device float *slab [[buffer(7)]],
    device float *terrain [[buffer(8)]], device float *correction [[buffer(9)]],
    constant uint &nx [[buffer(10)]], constant uint &ny [[buffer(11)]],
    constant uint &n [[buffer(12)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    float ox = coordinates[3*tid], oy = coordinates[3*tid+1], oz = coordinates[3*tid+2];
    float dx = east[1]-east[0], dy = north[1]-north[0];
    float sum = 0.0f, compensation = 0.0f;
    for (uint j = 0; j < ny; ++j) {
        for (uint i = 0; i < nx; ++i) {
            uint q = j*nx+i;
            float h = surface[q]-reference[q];
            if (h == 0.0f || density[q] == 0.0f) continue;
            float xc = east[i]-ox, yc = north[j]-oy;
            float lo = min(surface[q],reference[q])-oz;
            float hi = max(surface[q],reference[q])-oz;
            float prism = 0.0f;
            float width = max(dx, dy);
            float gx = max(abs(xc)-0.5f*dx, 0.0f);
            float gy = max(abs(yc)-0.5f*dy, 0.0f);
            float gap = length(float3(gx, gy, hi));
            // At gap >= 2 widths, each horizontal half-width/gap <= 1/4.
            // Six-point GL resolves the smooth integrand without subtracting
            // corner terms. This also handles lateral distance, independently
            // of prism thickness; all 700 m benchmark prisms satisfy it.
            if (gap >= 2.0f*width) {
                // No subdivision is needed in this regime. Keep the one-tile
                // path explicit for the dominant distant-prism workload.
                prism = integrated_prism(xc, yc, -hi, dx, dy, abs(h), 1u);
            } else if (abs(h) < 0.001f*width) {
                uint tiles = uint(clamp(ceil(2.0f*width / max(gap, width/16.0f)), 1.0f, 32.0f));
                prism = integrated_prism(xc, yc, -hi, dx, dy, abs(h), tiles);
            } else {
                for (uint a = 0; a < 2; ++a) {
                    float x = xc + (a == 0 ? 0.5f*dx : -0.5f*dx);
                    for (uint b = 0; b < 2; ++b) {
                        float y = yc + (b == 0 ? 0.5f*dy : -0.5f*dy);
                        for (uint c = 0; c < 2; ++c) {
                            float z = c == 0 ? hi : lo;
                            float term = gz_corner(x,y,z);
                            prism += ((a+b+c)&1) ? -term : term;
                        }
                    }
                }
            }
            float value = (h > 0.0f ? density[q] : -density[q])*prism;
            float adjusted = value-compensation;
            float next = sum+adjusted;
            compensation = (next-sum)-adjusted;
            sum = next;
        }
    }
    int ix = clamp(int(floor((ox-east[0])/dx + 0.5f)), 0, int(nx)-1);
    int iy = clamp(int(floor((oy-north[0])/dy + 0.5f)), 0, int(ny)-1);
    uint cell = uint(iy)*nx + uint(ix);
    float l = sum*6.67430e-6f;
    float s = (2.0f*3.14159265358979323846f*6.67430e-6f)*density[cell]*(surface[cell]-reference[cell]);
    layer[tid] = l;
    slab[tid] = s;
    terrain[tid] = l-s;
    correction[tid] = -l;
}
