#include <metal_stdlib>
using namespace metal;
// Solve each nonzero source at unit current; integrate rescales on the GPU.
kernel void init(device float*x [[buffer(0)]],device float*r [[buffer(1)]],device float*p [[buffer(2)]],device const float*d [[buffer(3)]],device const uint*src [[buffer(4)]],device const float*cur [[buffer(5)]],constant uint&n [[buffer(6)]],constant uint&ns [[buffer(7)]],uint i [[thread_position_in_grid]]) {uint s=i/n,c=i%n,row=s%ns;float v=(cur[row]!=0.0f?1.0f:0.0f)*((c==src[2*row]?1.0f:0.0f)-(c==src[2*row+1]?1.0f:0.0f));x[i]=0;r[i]=v;p[i]=v/d[(s/ns)*n+c];}
// State: preconditioned residual, step, direction coefficient, status,
// initial squared residual. Status is 0 active, 1 converged, -1 failed.
kernel void initdot(device const float*r [[buffer(0)]],device const float*p [[buffer(1)]],device float*st [[buffer(2)]],constant uint&n [[buffer(3)]],uint s [[thread_position_in_grid]]) {float v=0,norm=0;for(uint j=0;j<n;j++){float a=r[s*n+j];v+=a*p[s*n+j];norm+=a*a;}st[5*s]=v;st[5*s+1]=0;st[5*s+2]=0;st[5*s+3]=!isfinite(v)||!isfinite(norm)?-1.0f:(norm==0.0f?1.0f:(v>0.0f?0.0f:-1.0f));st[5*s+4]=norm;}
kernel void mat(device const float*p [[buffer(0)]],device float*q [[buffer(1)]],device const float*d [[buffer(2)]],device const float*e [[buffer(3)]],device const float*u [[buffer(4)]],constant uint&nx [[buffer(5)]],constant uint&nz [[buffer(6)]],constant uint&ns [[buffer(7)]],uint i [[thread_position_in_grid]]) {uint n=nx*nz,s=i/n,c=i%n,x=c%nx,z=c/nx;float v=d[(s/ns)*n+c]*p[i];if(x+1<nx)v-=e[c]*p[i+1];if(x>0)v-=e[c-1]*p[i-1];if(z+1<nz)v-=u[c]*p[i+nx];if(z>0)v-=u[c-nx]*p[i-nx];q[i]=v;}
kernel void step(device const float*p [[buffer(0)]],device const float*q [[buffer(1)]],device float*st [[buffer(2)]],constant uint&n [[buffer(3)]],uint s [[thread_position_in_grid]]) {if(st[5*s+3]!=0.0f)return;float v=0;for(uint j=0;j<n;j++)v+=p[s*n+j]*q[s*n+j];float a=st[5*s]/v;if(!(v>0.0f)||!isfinite(v)||!isfinite(a)){st[5*s+3]=-1.0f;st[5*s+1]=0.0f;return;}st[5*s+1]=a;}
kernel void update(device float*x [[buffer(0)]],device float*r [[buffer(1)]],device const float*p [[buffer(2)]],device const float*q [[buffer(3)]],device const float*st [[buffer(4)]],constant uint&n [[buffer(5)]],uint i [[thread_position_in_grid]]) {uint s=i/n;if(st[5*s+3]!=0.0f)return;float a=st[5*s+1];x[i]+=a*p[i];r[i]-=a*q[i];}
kernel void dot(device const float*r [[buffer(0)]],device const float*d [[buffer(1)]],device float*st [[buffer(2)]],constant uint&n [[buffer(3)]],constant uint&ns [[buffer(4)]],uint s [[thread_position_in_grid]]) {if(st[5*s+3]!=0.0f)return;float v=0,norm=0;for(uint j=0;j<n;j++){float a=r[s*n+j];v+=a*a/d[(s/ns)*n+j];norm+=a*a;}float old=st[5*s];if(!isfinite(v)||!isfinite(norm)){st[5*s+3]=-1.0f;return;}if(norm<=1e-16f*st[5*s+4]){st[5*s+3]=1.0f;return;}float beta=v/old;if(!(v>0.0f)||!(old>0.0f)||!isfinite(beta)){st[5*s+3]=-1.0f;return;}st[5*s+2]=beta;st[5*s]=v;}
kernel void direction(device float*p [[buffer(0)]],device const float*r [[buffer(1)]],device const float*d [[buffer(2)]],device const float*st [[buffer(3)]],constant uint&n [[buffer(4)]],constant uint&ns [[buffer(5)]],uint i [[thread_position_in_grid]]) {uint s=i/n;if(st[5*s+3]!=0.0f)return;p[i]=r[i]/d[(s/ns)*n+i%n]+st[5*s+2]*p[i];}

// Clamped linear interpolation of cell centers, including a one-cell axis.
inline uint bracket(device const float* centers, uint count, float position,
                    thread uint& upper, thread float& fraction) {
    if (position <= centers[0]) {
        upper = 0; fraction = 0.0f; return 0;
    }
    if (position >= centers[count-1]) {
        upper = count-1; fraction = 0.0f; return count-1;
    }
    uint low = 0, high = count-1;
    while (high-low > 1) {
        uint mid = low+(high-low)/2;
        if (centers[mid] < position) low = mid;
        else high = mid;
    }
    upper = high;
    fraction = (position-centers[low])/(centers[high]-centers[low]);
    return low;
}

inline float sample_potential(device const float* phi, device const float* xc,
                    device const float* zc, uint nx, uint nz, float x) {
    uint xb, zb;
    float tx, tz;
    uint xa = bracket(xc, nx, x, xb, tx);
    uint za = bracket(zc, nz, 0.0f, zb, tz);
    // Use convex weights so an electrode on a center picks that cell exactly.
    float lower = (1.0f-tx)*phi[za*nx+xa]+tx*phi[za*nx+xb];
    float upper = (1.0f-tx)*phi[zb*nx+xa]+tx*phi[zb*nx+xb];
    return (1.0f-tz)*lower+tz*upper;
}

// One thread per (Fourier mode, source), preserving the solver's layout.
kernel void receivers(device const float* phi [[buffer(0)]],
                      device const float* xc [[buffer(1)]],
                      device const float* zc [[buffer(2)]],
                      device const float* mn [[buffer(3)]],
                      device float* delta [[buffer(4)]],
                      constant uint& nx [[buffer(5)]],
                      constant uint& nz [[buffer(6)]],
                      constant uint& ns [[buffer(7)]],
                      uint system [[thread_position_in_grid]]) {
    uint row = system%ns;
    device const float* field = phi+system*nx*nz;
    delta[system] = sample_potential(field, xc, zc, nx, nz, mn[2*row])
                  -sample_potential(field, xc, zc, nx, nz, mn[2*row+1]);
}

kernel void integrate(device const float* delta [[buffer(0)]],
                      device const float* weights [[buffer(1)]],
                      device const float* currents [[buffer(2)]],
                      device const float* ab [[buffer(3)]],
                      device const float* mn [[buffer(4)]],
                      device float* voltage [[buffer(5)]],
                      device float* rho [[buffer(6)]],
                      constant uint& ns [[buffer(7)]],
                      constant uint& nk [[buffer(8)]],
                      uint row [[thread_position_in_grid]]) {
    // Float32 compensated Fourier sum avoids losing small weighted modes.
    // weights already include 1/pi; apply no further transform normalization.
    float sum = 0.0f, correction = 0.0f;
    for (uint mode = 0; mode < nk; ++mode) {
        float term = weights[mode]*delta[mode*ns+row]-correction;
        float next = sum+term;
        correction = (next-sum)-term;
        sum = next;
    }
    float v = currents[row]*sum;
    float a = ab[2*row], b = ab[2*row+1];
    float m = mn[2*row], n = mn[2*row+1];
    float g = (1.0f/abs(a-m)-1.0f/abs(b-m)-1.0f/abs(a-n)+1.0f/abs(b-n))
              /(2.0f*M_PI_F);
    voltage[row] = v;
    rho[row] = v/(currents[row]*g);
}
