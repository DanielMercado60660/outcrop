#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Params {
    uint ny, nx, cells, grid, shots, sources, receivers, nt;
    uint source_groups, receiver_groups, model_ny, model_nx, top, left, history;
    float dy, dx, dt;
};

// State planes: vy,vx,syy,sxx,sxy, then the eight derivative memories
// in SCHEME.md order. Coefficients: lambda,mu,mu_yx,by,bx.
// Forward state/coefficient planes have matching residual planes after the
// first 13/5 planes. Every arithmetic operation is float32. Keeping the
// roundoff residual prevents cancellation in centered objective differences
// from amplifying per-step and material-conversion rounding. The output and
// adjoint history are rounded float32; no temporal compression is used.
// History planes per step: div_y,div_x,ey,ex,g (corrected derivatives).

inline float2 split_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    return float2(s, (a - (s - v)) + (b - v));
}

inline float2 pair_add(float2 a, float2 b) {
    float2 s = split_sum(a.x, b.x);
    float e = (s.y + a.y) + b.y;
    return split_sum(s.x, e);
}

inline float2 pair_sub(float2 a, float2 b) {
    return pair_add(a, -b);
}

inline float2 pair_mul(float2 a, float2 b) {
    float hi = a.x * b.x;
    float lo = fma(a.x, b.x, -hi);
    lo = lo + a.x * b.y;
    lo = lo + a.y * b.x;
    return split_sum(hi, lo);
}

inline float2 pair_scale(float2 a, float b) {
    return pair_mul(a, float2(b, 0.0f));
}

// Quotient plus one residual correction; denominators are masked before
// calling this function. Explicit fma in pair_mul recovers product roundoff.
inline float2 pair_div(float2 a, float2 b) {
    float q = a.x / b.x;
    float2 r = pair_sub(a, pair_scale(b, q));
    return split_sum(q, (r.x + r.y) / b.x);
}

inline float2 state_load(device const float* s, uint k, constant Params& p) {
    return float2(s[k], s[13*p.grid+k]);
}

inline void state_store(device float* s, uint k, float2 v, constant Params& p) {
    s[k] = v.x;
    s[13*p.grid+k] = v.y;
}

inline float2 coeff_load(device const float* c, uint k, constant Params& p) {
    return float2(c[k], c[5*p.cells+k]);
}

inline void coeff_store(device float* c, uint k, float2 v, constant Params& p) {
    c[k] = v.x;
    c[5*p.cells+k] = v.y;
}

inline float rounded(float2 a) { return a.x + a.y; }

inline uint model_index(uint i, uint j, constant Params& p) {
    uint y = uint(clamp(int(i) - int(p.top), 0, int(p.model_ny) - 1));
    uint x = uint(clamp(int(j) - int(p.left), 0, int(p.model_nx) - 1));
    return y * p.model_nx + x;
}

inline float2 nodal_mu(device const float* model, uint i, uint j, constant Params& p) {
    uint m = p.model_ny * p.model_nx;
    uint k = model_index(i,j,p);
    float vs = model[m+k];
    return pair_scale(pair_mul(float2(vs,0.0f),float2(vs,0.0f)),model[2*m+k]);
}

kernel void material(device const float* model [[buffer(0)]],
                     device float* c [[buffer(1)]],
                     constant Params& p [[buffer(2)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.cells) return;
    uint i = k / p.nx, j = k % p.nx, m = p.model_ny * p.model_nx;
    uint q = model_index(i,j,p);
    float vp = model[q], vs = model[m+q], rho = model[2*m+q];
    float2 vs2 = pair_mul(float2(vs,0.0f),float2(vs,0.0f));
    float2 vp2 = pair_mul(float2(vp,0.0f),float2(vp,0.0f));
    float2 mu = pair_scale(vs2,rho);
    coeff_store(c,k,pair_scale(pair_sub(vp2,pair_scale(vs2,2.0f)),rho),p);
    coeff_store(c,p.cells+k,mu,p);
    float2 by = float2(0.0f), bx = float2(0.0f), h = float2(0.0f);
    if (i+1 < p.ny) {
        float2 sum = split_sum(rho,model[2*m+model_index(i+1,j,p)]);
        if (sum.x > 0.0f) by = pair_div(float2(2.0f,0.0f),sum);
    }
    if (j+1 < p.nx) {
        float2 sum = split_sum(rho,model[2*m+model_index(i,j+1,p)]);
        if (sum.x > 0.0f) bx = pair_div(float2(2.0f,0.0f),sum);
    }
    if (i+1 < p.ny && j+1 < p.nx && mu.x > 0.0f) {
        float2 a = nodal_mu(model,i+1,j,p), b = nodal_mu(model,i,j+1,p);
        float2 d = nodal_mu(model,i+1,j+1,p);
        if (a.x > 0.0f && b.x > 0.0f && d.x > 0.0f) {
            float2 one = float2(1.0f,0.0f);
            float2 sum = pair_add(pair_div(one,mu),pair_div(one,a));
            sum = pair_add(sum,pair_div(one,b));
            sum = pair_add(sum,pair_div(one,d));
            h = pair_div(float2(4.0f,0.0f),sum);
        }
    }
    coeff_store(c,2*p.cells+k,h,p);
    coeff_store(c,3*p.cells+k,by,p);
    coeff_store(c,4*p.cells+k,bx,p);
}

inline float2 corrected(float2 d, device float* s, uint k, float2 ab,
                        constant Params& p) {
    float2 next = pair_add(pair_scale(state_load(s,k,p),ab.x),pair_scale(d,ab.y));
    state_store(s,k,next,p);
    return pair_add(d,next);
}

kernel void record(device const float* s [[buffer(0)]],
                   device const uint* ids [[buffer(1)]],
                   device float* traces [[buffer(2)]],
                   constant Params& p [[buffer(3)]],
                   constant uint& t [[buffer(4)]], uint r [[thread_position_in_grid]]) {
    if (r >= p.shots*p.receivers) return;
    uint k = ids[r], n = p.shots*p.receivers*p.nt, q = r*p.nt+t;
    traces[q] = rounded(state_load(s,k,p));
    traces[n+q] = rounded(state_load(s,p.grid+k,p));
    traces[2*n+q] = rounded(pair_scale(pair_add(state_load(s,2*p.grid+k,p),
                                               state_load(s,3*p.grid+k,p)),-0.5f));
}

kernel void velocity(device float* s [[buffer(0)]],
                     device const float* c [[buffer(1)]],
                     device const float2* ab [[buffer(2)]],
                     device float* hist [[buffer(3)]],
                     constant Params& p [[buffer(4)]],
                     constant uint& t [[buffer(5)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.grid) return;
    uint q = k % p.cells, i = q/p.nx, j = q%p.nx, g = p.grid;
    float2 syy = state_load(s,2*g+k,p), sxx = state_load(s,3*g+k,p);
    float2 sxy = state_load(s,4*g+k,p), zero = float2(0.0f);
    float2 dy = float2(p.dy,0.0f), dx = float2(p.dx,0.0f);
    float2 d0 = pair_div(pair_sub(i+1<p.ny ? state_load(s,2*g+k+p.nx,p) : zero,syy),dy);
    float2 d1 = pair_div(pair_sub(sxy,j>0 ? state_load(s,4*g+k-1,p) : zero),dx);
    float2 d2 = pair_div(pair_sub(sxy,i>0 ? state_load(s,4*g+k-p.nx,p) : zero),dy);
    float2 d3 = pair_div(pair_sub(j+1<p.nx ? state_load(s,3*g+k+1,p) : zero,sxx),dx);
    float2 cy = pair_add(corrected(d0,s,5*g+k,ab[p.ny+i],p),
                         corrected(d1,s,6*g+k,ab[2*p.ny+j],p));
    float2 cx = pair_add(corrected(d2,s,7*g+k,ab[i],p),
                         corrected(d3,s,8*g+k,ab[2*p.ny+p.nx+j],p));
    float2 bydt = pair_scale(coeff_load(c,3*p.cells+q,p),p.dt);
    float2 bxdt = pair_scale(coeff_load(c,4*p.cells+q,p),p.dt);
    state_store(s,k,pair_add(state_load(s,k,p),pair_mul(bydt,cy)),p);
    state_store(s,g+k,pair_add(state_load(s,g+k,p),pair_mul(bxdt,cx)),p);
    if (p.history) {
        ulong h = ulong(t)*5*ulong(g)+k;
        hist[h] = rounded(cy);
        hist[h+g] = rounded(cx);
    }
}

// One thread per unique source cell, with sources in original index order.
// Coincident sources never use atomics and are reproducible across replays.
kernel void inject(device float* s [[buffer(0)]],
                   device const float* c [[buffer(1)]],
                   device const float* sy [[buffer(2)]],
                   device const float* sx [[buffer(3)]],
                   device const uint* groups [[buffer(4)]],
                   device const uint* entries [[buffer(5)]],
                   constant Params& p [[buffer(6)]],
                   constant uint& t [[buffer(7)]], uint group [[thread_position_in_grid]]) {
    if (group >= p.source_groups) return;
    uint k = groups[3*group], q = k%p.cells;
    float2 vy = state_load(s,k,p), vx = state_load(s,p.grid+k,p);
    float2 bydt = pair_scale(coeff_load(c,3*p.cells+q,p),p.dt);
    float2 bxdt = pair_scale(coeff_load(c,4*p.cells+q,p),p.dt);
    for (uint a=groups[3*group+1]; a<groups[3*group+2]; ++a) {
        uint f = entries[a]*p.nt+t;
        vy = pair_add(vy,pair_scale(bydt,sy[f]));
        vx = pair_add(vx,pair_scale(bxdt,sx[f]));
    }
    state_store(s,k,vy,p);
    state_store(s,p.grid+k,vx,p);
}

kernel void stress(device float* s [[buffer(0)]],
                   device const float* c [[buffer(1)]],
                   device const float2* ab [[buffer(2)]],
                   device float* hist [[buffer(3)]],
                   constant Params& p [[buffer(4)]],
                   constant uint& t [[buffer(5)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.grid) return;
    uint q = k%p.cells, i = q/p.nx, j = q%p.nx, g = p.grid;
    float2 vy = state_load(s,k,p), vx = state_load(s,g+k,p), zero = float2(0.0f);
    float2 dy = float2(p.dy,0.0f), dx = float2(p.dx,0.0f);
    float2 d0 = pair_div(pair_sub(vy,i>0 ? state_load(s,k-p.nx,p) : zero),dy);
    float2 d1 = pair_div(pair_sub(vx,j>0 ? state_load(s,g+k-1,p) : zero),dx);
    float2 d2 = pair_div(pair_sub(i+1<p.ny ? state_load(s,g+k+p.nx,p) : zero,vx),dy);
    float2 d3 = pair_div(pair_sub(j+1<p.nx ? state_load(s,k+1,p) : zero,vy),dx);
    float2 ey = corrected(d0,s,9*g+k,ab[i],p);
    float2 ex = corrected(d1,s,10*g+k,ab[2*p.ny+j],p);
    float2 shear = pair_add(corrected(d2,s,11*g+k,ab[p.ny+i],p),
                            corrected(d3,s,12*g+k,ab[2*p.ny+p.nx+j],p));
    float2 w = pair_mul(coeff_load(c,q,p),pair_add(ey,ex));
    float2 mu2 = pair_scale(coeff_load(c,p.cells+q,p),2.0f);
    float2 yy = pair_scale(pair_add(w,pair_mul(mu2,ey)),p.dt);
    float2 xx = pair_scale(pair_add(w,pair_mul(mu2,ex)),p.dt);
    float2 xy = pair_mul(pair_scale(coeff_load(c,2*p.cells+q,p),p.dt),shear);
    state_store(s,2*g+k,pair_add(state_load(s,2*g+k,p),yy),p);
    state_store(s,3*g+k,pair_add(state_load(s,3*g+k,p),xx),p);
    state_store(s,4*g+k,pair_add(state_load(s,4*g+k,p),xy),p);
    if (p.history) {
        ulong h = ulong(t)*5*ulong(g)+k;
        hist[h+2*g] = rounded(ey);
        hist[h+3*g] = rounded(ex);
        hist[h+4*g] = rounded(shear);
    }
}

// Reverse psi_new=a*psi_old+b*D, C=D+psi_new. Incoming memory is
// the adjoint from future steps; overwrite with the adjoint of psi_old.
inline float reverse_corrected(float bar_c, device float* bar_memory, float2 ab) {
    float bar_new = *bar_memory + bar_c;
    *bar_memory = ab.x*bar_new;
    return bar_c + ab.y*bar_new;
}

kernel void reverse_stress(device float* a [[buffer(0)]],
                           device const float* c [[buffer(1)]],
                           device const float2* ab [[buffer(2)]],
                           device const float* hist [[buffer(3)]],
                           device float* cg [[buffer(4)]],
                           device float* d [[buffer(5)]],
                           constant Params& p [[buffer(6)]],
                           constant uint& t [[buffer(7)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.grid) return;
    uint q=k%p.cells, i=q/p.nx, j=q%p.nx, g=p.grid;
    ulong h=ulong(t)*5*ulong(g)+k;
    float ey=hist[h+2*g], ex=hist[h+3*g], shear=hist[h+4*g];
    float yy=a[2*g+k], xx=a[3*g+k], xy=a[4*g+k];
    float bar_w=p.dt*(yy+xx);
    cg[k] += bar_w*(ey+ex);
    cg[g+k] += (2.0f*p.dt)*(yy*ey+xx*ex);
    cg[2*g+k] += (p.dt*xy)*shear;
    float bar_ey=c[q]*bar_w+(2.0f*c[p.cells+q])*(p.dt*yy);
    float bar_ex=c[q]*bar_w+(2.0f*c[p.cells+q])*(p.dt*xx);
    float bar_shear=(p.dt*c[2*p.cells+q])*xy;
    d[k] = reverse_corrected(bar_ey,a+9*g+k,ab[i]);
    d[g+k] = reverse_corrected(bar_ex,a+10*g+k,ab[2*p.ny+j]);
    d[2*g+k] = reverse_corrected(bar_shear,a+11*g+k,ab[p.ny+i]);
    d[3*g+k] = reverse_corrected(bar_shear,a+12*g+k,ab[2*p.ny+p.nx+j]);
}

// Explicit transpose stencils, with no scatter races and zero outside grid.
// (D-)^T q=(q[k]-q[k+1])/h; (D+)^T q=(q[k-1]-q[k])/h.
kernel void gather_velocity(device float* a [[buffer(0)]],
                            device const float* d [[buffer(1)]],
                            constant Params& p [[buffer(2)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.grid) return;
    uint q=k%p.cells, i=q/p.nx, j=q%p.nx, g=p.grid;
    float yy=(d[k]-(i+1<p.ny ? d[k+p.nx] : 0.0f))/p.dy;
    float yx=((j>0 ? d[3*g+k-1] : 0.0f)-d[3*g+k])/p.dx;
    float xx=(d[g+k]-(j+1<p.nx ? d[g+k+1] : 0.0f))/p.dx;
    float xy=((i>0 ? d[2*g+k-p.nx] : 0.0f)-d[2*g+k])/p.dy;
    a[k] += yy+yx;
    a[g+k] += xx+xy;
}

kernel void reverse_source(device const float* a [[buffer(0)]],
                           device const float* c [[buffer(1)]],
                           device const float* sy [[buffer(2)]],
                           device const float* sx [[buffer(3)]],
                           device const uint* groups [[buffer(4)]],
                           device const uint* entries [[buffer(5)]],
                           device float* cg [[buffer(6)]],
                           device float* sg [[buffer(7)]],
                           constant Params& p [[buffer(8)]],
                           constant uint& t [[buffer(9)]], uint group [[thread_position_in_grid]]) {
    if (group >= p.source_groups) return;
    uint k=groups[3*group], q=k%p.cells, g=p.grid;
    float ay=a[k], ax=a[g+k], gy=0.0f, gx=0.0f;
    uint n=p.shots*p.sources*p.nt;
    for (uint e=groups[3*group+1]; e<groups[3*group+2]; ++e) {
        uint f=entries[e]*p.nt+t;
        sg[f]=(p.dt*c[3*p.cells+q])*ay;
        sg[n+f]=(p.dt*c[4*p.cells+q])*ax;
        gy += (p.dt*sy[f])*ay;
        gx += (p.dt*sx[f])*ax;
    }
    cg[3*g+k] += gy;
    cg[4*g+k] += gx;
}

kernel void reverse_velocity(device float* a [[buffer(0)]],
                             device const float* c [[buffer(1)]],
                             device const float2* ab [[buffer(2)]],
                             device const float* hist [[buffer(3)]],
                             device float* cg [[buffer(4)]],
                             device float* d [[buffer(5)]],
                             constant Params& p [[buffer(6)]],
                             constant uint& t [[buffer(7)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.grid) return;
    uint q=k%p.cells, i=q/p.nx, j=q%p.nx, g=p.grid;
    ulong h=ulong(t)*5*ulong(g)+k;
    float ay=a[k], ax=a[g+k];
    cg[3*g+k] += (p.dt*hist[h])*ay;
    cg[4*g+k] += (p.dt*hist[h+g])*ax;
    float cy=(p.dt*c[3*p.cells+q])*ay, cx=(p.dt*c[4*p.cells+q])*ax;
    d[k]=reverse_corrected(cy,a+5*g+k,ab[p.ny+i]);
    d[g+k]=reverse_corrected(cy,a+6*g+k,ab[2*p.ny+j]);
    d[2*g+k]=reverse_corrected(cx,a+7*g+k,ab[i]);
    d[3*g+k]=reverse_corrected(cx,a+8*g+k,ab[2*p.ny+p.nx+j]);
}

kernel void gather_stress(device float* a [[buffer(0)]],
                          device const float* d [[buffer(1)]],
                          constant Params& p [[buffer(2)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.grid) return;
    uint q=k%p.cells, i=q/p.nx, j=q%p.nx, g=p.grid;
    a[2*g+k] += ((i>0 ? d[k-p.nx] : 0.0f)-d[k])/p.dy;
    a[3*g+k] += ((j>0 ? d[3*g+k-1] : 0.0f)-d[3*g+k])/p.dx;
    float x=(d[g+k]-(j+1<p.nx ? d[g+k+1] : 0.0f))/p.dx;
    float y=(d[2*g+k]-(i+1<p.ny ? d[2*g+k+p.nx] : 0.0f))/p.dy;
    a[4*g+k] += x+y;
}

kernel void receiver_adjoint(device float* a [[buffer(0)]],
                             device const float* residual [[buffer(1)]],
                             device const uint* groups [[buffer(2)]],
                             device const uint* entries [[buffer(3)]],
                             constant Params& p [[buffer(4)]],
                             constant uint& t [[buffer(5)]], uint group [[thread_position_in_grid]]) {
    if (group >= p.receiver_groups) return;
    uint k=groups[3*group], n=p.shots*p.receivers*p.nt;
    float ay=a[k], ax=a[p.grid+k];
    for (uint e=groups[3*group+1]; e<groups[3*group+2]; ++e) {
        uint f=entries[e]*p.nt+t;
        ay += residual[f];
        ax += residual[n+f];
    }
    a[k]=ay;
    a[p.grid+k]=ax;
}

inline float harmonic_pullback(float mu, float h, float bar_h) {
    // h==0 includes every harmonic stencil touching vacuum or a high edge.
    float r=h/mu;
    return (0.25f*r*r)*bar_h;
}

kernel void material_adjoint(device const float* model [[buffer(0)]],
                             device const float* c [[buffer(1)]],
                             device const float* cg [[buffer(2)]],
                             device float* pg [[buffer(3)]],
                             constant Params& p [[buffer(4)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.cells) return;
    uint i=k/p.nx, j=k%p.nx, m=p.model_ny*p.model_nx, q=model_index(i,j,p);
    float vp=model[q], vs=model[m+q], rho=model[2*m+q];
    if (rho == 0.0f) {
        pg[k]=0.0f; pg[p.cells+k]=0.0f; pg[2*p.cells+k]=0.0f;
        return;
    }
    float mu=c[p.cells+k], gl=0.0f, gm=0.0f, gr=0.0f;
    for (uint shot=0; shot<p.shots; ++shot) {
        uint u=shot*p.cells+k, g=p.grid;
        gl += cg[u];
        gm += cg[g+u];
        // Four harmonic means depend on this normal-node shear modulus.
        gm += harmonic_pullback(mu,c[2*p.cells+k],cg[2*g+u]);
        if (i>0) gm += harmonic_pullback(mu,c[2*p.cells+k-p.nx],cg[2*g+u-p.nx]);
        if (j>0) gm += harmonic_pullback(mu,c[2*p.cells+k-1],cg[2*g+u-1]);
        if (i>0 && j>0)
            gm += harmonic_pullback(mu,c[2*p.cells+k-p.nx-1],cg[2*g+u-p.nx-1]);
        // d[2/(rho0+rho1)]/d rho = -by^2/2, including vacuum/solid pairs.
        float by=c[3*p.cells+k], bx=c[4*p.cells+k];
        gr -= (0.5f*by*by)*cg[3*g+u];
        gr -= (0.5f*bx*bx)*cg[4*g+u];
        if (i>0) {
            by=c[3*p.cells+k-p.nx];
            gr -= (0.5f*by*by)*cg[3*g+u-p.nx];
        }
        if (j>0) {
            bx=c[4*p.cells+k-1];
            gr -= (0.5f*bx*bx)*cg[4*g+u-1];
        }
    }
    pg[k]=(2.0f*rho*vp)*gl;
    pg[p.cells+k]=(2.0f*rho*vs)*(gm-2.0f*gl);
    pg[2*p.cells+k]=(vp*vp-2.0f*(vs*vs))*gl+(vs*vs)*gm+gr;
}

// Transpose of edge replication: every padded copy sums into its input node.
kernel void unpad(device const float* pg [[buffer(0)]],
                   device float* mg [[buffer(1)]],
                   constant Params& p [[buffer(2)]], uint k [[thread_position_in_grid]]) {
    uint m=p.model_ny*p.model_nx;
    if (k >= m) return;
    uint i=k/p.model_nx, j=k%p.model_nx;
    uint lo_y=i==0 ? 0 : p.top+i, hi_y=i+1==p.model_ny ? p.ny : p.top+i+1;
    uint lo_x=j==0 ? 0 : p.left+j, hi_x=j+1==p.model_nx ? p.nx : p.left+j+1;
    float vp=0.0f, vs=0.0f, rho=0.0f;
    for (uint y=lo_y; y<hi_y; ++y) {
        for (uint x=lo_x; x<hi_x; ++x) {
            uint q=y*p.nx+x;
            vp += pg[q]; vs += pg[p.cells+q]; rho += pg[2*p.cells+q];
        }
    }
    mg[k]=vp; mg[m+k]=vs; mg[2*m+k]=rho;
}
