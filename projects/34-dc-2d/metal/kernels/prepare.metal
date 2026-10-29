#include <metal_stdlib>
using namespace metal;

// Ratio K1(x)/K0(x). For large x the shared exponential/sqrt(x) factor
// cancels, so evaluating the scaled polynomials also avoids underflow.
inline float bessel_ratio(float x) {
    if (x <= 2.0f) {
        float y = x*x/4.0f;
        float t = x*x/(3.75f*3.75f);
        float i0 = 1.0f+t*(3.5156229f+t*(3.0899424f+t*(1.2067492f+t*(0.2659732f+t*(0.0360768f+t*0.0045813f)))));
        float i1 = x*(0.5f+t*(0.87890594f+t*(0.51498869f+t*(0.15084934f+t*(0.02658733f+t*(0.00301532f+t*0.00032411f))))));
        float k0 = -log(x/2.0f)*i0+(-0.57721566f+y*(0.42278420f+y*(0.23069756f+y*(0.03488590f+y*(0.00262698f+y*(0.00010750f+y*0.00000740f))))));
        float k1 = log(x/2.0f)*i1+(1.0f+y*(0.15443144f+y*(-0.67278579f+y*(-0.18156897f+y*(-0.01919402f+y*(-0.00110404f+y*(-0.00004686f)))))))/x;
        return k1/k0;
    }
    float y = 2.0f/x;
    float k0 = 1.25331414f+y*(-0.07832358f+y*(0.02189568f+y*(-0.01062446f+y*(0.00587872f+y*(-0.00251540f+y*0.00053208f)))));
    float k1 = 1.25331414f+y*(0.23498619f+y*(-0.03655620f+y*(0.01504268f+y*(-0.00780353f+y*(0.00325614f+y*(-0.00068245f))))));
    return k1/k0;
}

// Face conductances: face area / sum of the two half-cell resistances.
kernel void faces(device const float* hx [[buffer(0)]],
                  device const float* hz [[buffer(1)]],
                  device const float* sig [[buffer(2)]],
                  device float* east [[buffer(3)]],
                  device float* up [[buffer(4)]],
                  constant uint& nx [[buffer(5)]],
                  constant uint& nz [[buffer(6)]],
                  uint c [[thread_position_in_grid]]) {
    uint x = c%nx, z = c/nx;
    float s = sig[x*nz+z];
    east[c] = x+1 < nx ? hz[z]/(hx[x]/(2.0f*s)+hx[x+1]/(2.0f*sig[(x+1)*nz+z])) : 0.0f;
    up[c] = z+1 < nz ? hx[x]/(hz[z]/(2.0f*s)+hz[z+1]/(2.0f*sig[x*nz+z+1])) : 0.0f;
}

kernel void diagonal(device const float* hx [[buffer(0)]],
                     device const float* hz [[buffer(1)]],
                     device const float* sig [[buffer(2)]],
                     device const float* xn [[buffer(3)]],
                     device const float* zn [[buffer(4)]],
                     device const float* xc [[buffer(5)]],
                     device const float* zc [[buffer(6)]],
                     device const float* ky [[buffer(7)]],
                     device const float* east [[buffer(8)]],
                     device const float* up [[buffer(9)]],
                     device float* diag [[buffer(10)]],
                     constant uint& nx [[buffer(11)]],
                     constant uint& nz [[buffer(12)]],
                     uint i [[thread_position_in_grid]]) {
    uint n = nx*nz, c = i%n, x = c%nx, z = c/nx;
    float s = sig[x*nz+z], k = ky[i/n];
    float d = 0.0f;
    if (z > 0) d += up[c-nx];
    if (x > 0) d += east[c-1];
    d += east[c];
    d += up[c];
    d += k*k*s*hx[x]*hz[z];
    float top = zn[nz];
    float mid = 0.5f*(xn[0]+xn[nx]);
    // Robin elimination adds sigma*area*alpha/(1+halfwidth*alpha).
    // Both sides must be included even for a single column.
    for (uint side = 0; side < 2; ++side) {
        if ((side == 0 && x == 0) || (side == 1 && x+1 == nx)) {
            float dx = abs((side == 0 ? xn[0] : xn[nx])-mid);
            float r = length(float2(dx, zc[z]-top));
            float alpha = k*bessel_ratio(k*r)*dx/r;
            d += s*hz[z]*alpha/(1.0f+alpha*hx[x]*0.5f);
        }
    }
    if (z == 0) {
        float dz = top-zn[0];
        float r = length(float2(xc[x]-mid, dz));
        float alpha = k*bessel_ratio(k*r)*dz/r;
        d += s*hx[x]*alpha/(1.0f+alpha*hz[z]*0.5f);
    }
    // No boundary term on top: zero normal flux.
    diag[i] = d;
}
