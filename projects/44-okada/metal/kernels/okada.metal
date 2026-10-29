#include <metal_stdlib>
using namespace metal;

// At the free surface the real and image UA terms cancel for displacement
// and horizontal derivatives; the remaining field is the image UB0 term.
kernel void point_surface(device const float *xyz [[buffer(0)]],
                          device const float *slip [[buffer(1)]],
                          device float *disp [[buffer(2)]],
                          device float *grad [[buffer(3)]],
                          constant float &depth [[buffer(4)]],
                          constant float &dip [[buffer(5)]],
                          constant float &alpha [[buffer(6)]],
                          constant uint &n [[buffer(7)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float x = xyz[3*i], y = xyz[3*i+1], d = depth;
    // A surface point source has r >= depth > 0.  Do not snap physical
    // lengths: a fixed epsilon would turn a shallow source into r == 0.
    // Evaluate in power-of-two length units so the inverse powers through
    // r^-7 stay finite even when the caller uses very small or large units.
    // The largest normalized coordinate is in [1, 2); binary rescaling
    // preserves the significands used by the original equations.
    int length_exp = ilogb(max(max(abs(x), abs(y)), d));
    x = ldexp(x, -length_exp);
    y = ldexp(y, -length_exp);
    d = ldexp(d, -length_exp);
    // Potency also changes with length units.  Normalize it before forming
    // the field, then combine both exponents only at the final writeback.
    float potency_max = max(max(abs(slip[0]), abs(slip[1])), abs(slip[2]));
    int potency_exp = potency_max == 0.0f ? 0 : ilogb(potency_max);
    float sd=sin(dip*(M_PI_F/180.0f)), cd=cos(dip*(M_PI_F/180.0f));
    if (abs(cd)<1e-6f) { cd=0; sd=sd>0 ? 1.0f : -1.0f; }
    float a=(1-alpha)/alpha, p=y*cd+d*sd, q=y*sd-d*cd;
    float x2=x*x,y2=y*y,d2=d*d,xy=x*y,r2=x2+y2+d2,r=sqrt(r2),r3=r*r2,r5=r3*r2;
    float a3=1-3*x2/r2,a5=1-5*x2/r2,qr=3*q/r5,qrx=5*qr*x/r2;
    float uy=sd-5*y*q/r2,vy=(p*sd+q*cd)-5*y*p*q/r2,wy=uy+sd;
    float rd=r+d,d12=1/(r*rd*rd),d32=d12*(2*r+d)/r2,d33=d12*(3*r+d)/(r2*rd);
    float d53=d12*(8*r2+9*r*d+3*d2)/(r2*r2*rd);
    float d54=d12*(5*r2+4*r*d+d2)/r3*d12;
    float fi1=y*(d12-x2*d33),fi2=x*(d12-y2*d33),fi3=x/r3-fi2,fi4=-xy*d32,fi5=1/(r*rd)-x2*d32;
    float fj1=-3*xy*(d33-x2*d54),fj2=1/r3-3*d12+3*x2*y2*d54,fj3=a3/r3-fj2,fj4=-3*xy/r5-fj1;
    float fk1=-y*(d32-x2*d53),fk2=-x*(d32-y2*d53),fk3=-3*x*d/r5-fk2;
    float u[9]={0};
    float s=ldexp(slip[0], -potency_exp)/(2*M_PI_F);
    u[0]+=s*(-x2*qr-a*fi1*sd); u[1]+=s*(-xy*qr-a*fi2*sd); u[2]+=s*(-d*x*qr-a*fi4*sd);
    u[3]+=s*(-x*qr*(1+a5)-a*fj1*sd); u[4]+=s*(-y*qr*a5-a*fj2*sd); u[5]+=s*(-d*qr*a5-a*fk1*sd);
    u[6]+=s*(-3*x2/r5*uy-a*fj2*sd); u[7]+=s*(-3*xy/r5*uy-x*qr-a*fj4*sd); u[8]+=s*(-3*d*x/r5*uy-a*fk2*sd);
    s=ldexp(slip[1], -potency_exp)/(2*M_PI_F); float sc=sd*cd;
    u[0]+=s*(-x*p*qr+a*fi3*sc); u[1]+=s*(-y*p*qr+a*fi1*sc); u[2]+=s*(-d*p*qr+a*fi5*sc);
    u[3]+=s*(-p*qr*a5+a*fj3*sc); u[4]+=s*(y*p*qrx+a*fj1*sc); u[5]+=s*(d*p*qrx+a*fk3*sc);
    u[6]+=s*(-3*x/r5*vy+a*fj1*sc); u[7]+=s*(-3*y/r5*vy-p*qr+a*fj2*sc); u[8]+=s*(-3*d/r5*vy+a*fk1*sc);
    s=ldexp(slip[2], -potency_exp)/(2*M_PI_F); float ss=sd*sd;
    u[0]+=s*(x*q*qr-a*fi3*ss); u[1]+=s*(y*q*qr-a*fi1*ss); u[2]+=s*(d*q*qr-a*fi5*ss);
    u[3]+=s*(q*qr*a5-a*fj3*ss); u[4]+=s*(-y*q*qrx-a*fj1*ss); u[5]+=s*(-d*q*qrx-a*fk3*ss);
    u[6]+=s*(x*qr*wy-a*fj1*ss); u[7]+=s*(qr*(y*wy+q)-a*fj2*ss); u[8]+=s*(d*qr*wy-a*fk1*ss);
    // Point displacement is potency / length^2; each horizontal partial
    // adds one inverse length.  ldexp avoids an overflowing scale factor
    // (and keeps exact symmetry zeros zero).
    for(uint k=0;k<3;k++) {
        disp[3*i+k]=ldexp(u[k], potency_exp-2*length_exp);
        grad[6*i+2*k]=ldexp(u[3+k], potency_exp-3*length_exp);
        grad[6*i+2*k+1]=ldexp(u[6+k], potency_exp-3*length_exp);
    }
}
