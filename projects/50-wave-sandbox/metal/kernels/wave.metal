#include <metal_stdlib>
using namespace metal;
#ifndef RADIUS
#define RADIUS 4
#endif
#pragma clang fp contract(off)

struct Params {
    uint w, h, ny, nx, p, nt, nr, periodic, save_history;
    float dt2, iy, ix, scale;
};
struct Profile { float a, b, c; };
#if RADIUS == 1
constant float C[2] = {0, 0.5f};
constant float D[2] = {-2, 1};
#elif RADIUS == 2
constant float C[3] = {0, 2.0f/3, -1.0f/12};
constant float D[3] = {-2.5f, 4.0f/3, -1.0f/12};
#elif RADIUS == 3
constant float C[4] = {0, 0.75f, -3.0f/20, 1.0f/60};
constant float D[4] = {-49.0f/18, 1.5f, -3.0f/20, 1.0f/90};
#else
constant float C[5] = {0, 0.8f, -0.2f, 4.0f/105, -1.0f/280};
constant float D[5] = {-205.0f/72, 1.6f, -0.2f, 8.0f/315, -1.0f/560};
#endif

// Buffers contain evolving nodes. These reads implement hidden zero guards
// or periodic closure without allocating/storing redundant guard cells.
inline int index_at(int y, int x, constant Params& p) {
    if (p.periodic) {
        y = (y % int(p.h) + int(p.h)) % int(p.h);
        x = (x % int(p.w) + int(p.w)) % int(p.w);
    } else if (y < 0 || x < 0 || y >= int(p.h) || x >= int(p.w)) return -1;
    return y*int(p.w)+x;
}
inline float at(device const float* f, int y, int x, constant Params& p) {
    int k = index_at(y,x,p);
    return k < 0 ? 0.0f : f[k];
}
inline float d1(device const float* f, int y, int x, bool axis_y, constant Params& p) {
    float s = 0;
    for (int j=1; j<=RADIUS; ++j) {
        int dy = axis_y ? j : 0, dx = axis_y ? 0 : j;
        s += C[j]*(at(f,y+dy,x+dx,p)-at(f,y-dy,x-dx,p));
    }
    return s*(axis_y ? p.iy : p.ix);
}
inline float d2(device const float* f, int y, int x, bool axis_y, constant Params& p) {
    float s = D[0]*f[y*p.w+x];
    for (int j=1; j<=RADIUS; ++j) {
        int dy = axis_y ? j : 0, dx = axis_y ? 0 : j;
        s += D[j]*(at(f,y+dy,x+dx,p)+at(f,y-dy,x-dx,p));
    }
    float ih = axis_y ? p.iy : p.ix;
    return s*(ih*ih);
}
inline float product_at(device const float* psi, device const Profile* profile,
                        int y, int x, bool axis_y, constant Params& p) {
    int k = index_at(y,x,p);
    if (k < 0) return 0;
    return profile[axis_y ? y : x].a*psi[k];
}
inline float product_d1(device const float* psi, device const Profile* profile,
                        int y, int x, bool axis_y, constant Params& p) {
    float s = 0;
    for (int j=1; j<=RADIUS; ++j) {
        int dy = axis_y ? j : 0, dx = axis_y ? 0 : j;
        s += C[j]*(product_at(psi,profile,y+dy,x+dx,axis_y,p)
                  -product_at(psi,profile,y-dy,x-dx,axis_y,p));
    }
    return s*(axis_y ? p.iy : p.ix);
}

kernel void project(device float* s [[buffer(0)]], constant Params& p [[buffer(1)]],
                    uint k [[thread_position_in_grid]]) {
    uint n=p.w*p.h;
    if(k>=n) return;
    uint y=k/p.w, x=k%p.w;
    if(y>=p.p && y<p.p+p.ny) { s[2*n+k]=0; s[4*n+k]=0; }
    if(x>=p.p && x<p.p+p.nx) { s[3*n+k]=0; s[5*n+k]=0; }
}
kernel void gather(device const float* s [[buffer(0)]], device const uint* loc [[buffer(1)]],
                   device float* rec [[buffer(2)]], constant Params& p [[buffer(3)]],
                   constant uint& t [[buffer(4)]], uint r [[thread_position_in_grid]]) {
    if(r<p.nr) rec[r*p.nt+t]=s[loc[r]];
}
kernel void advance(device const float* old [[buffer(0)]], device float* next [[buffer(1)]],
                    device const float* velocity [[buffer(2)]],
                    device const Profile* py [[buffer(3)]], device const Profile* px [[buffer(4)]],
                    device const int* source_map [[buffer(5)]], device const float* source [[buffer(6)]],
                    device float* history [[buffer(7)]], constant Params& p [[buffer(8)]],
                    constant uint& t [[buffer(9)]], uint k [[thread_position_in_grid]]) {
    uint n=p.w*p.h;
    if(k>=n) return;
    int y=k/p.w, x=k%p.w;
    float sy, sx;
    // Here the product derivative and coefficient derivative are identically
    // zero. Avoid auxiliary reads in the bulk of large models.
    if(p.periodic || (y>=int(p.p)+RADIUS && y<int(p.p+p.ny)-RADIUS &&
                      x>=int(p.p)+RADIUS && x<int(p.p+p.nx)-RADIUS)) {
        sy=d2(old,y,x,true,p); sx=d2(old,y,x,false,p);
        next[2*n+k]=0; next[3*n+k]=0; next[4*n+k]=0; next[5*n+k]=0;
    } else {
        Profile ay=py[y], ax=px[x];
        float gy=d1(old,y,x,true,p), gx=d1(old,y,x,false,p);
        float ty=(1+ay.b)*d2(old,y,x,true,p)+ay.c*gy
                 +product_d1(old+2*n,py,y,x,true,p);
        float tx=(1+ax.b)*d2(old,y,x,false,p)+ax.c*gx
                 +product_d1(old+3*n,px,y,x,false,p);
        sy=(1+ay.b)*ty+ay.a*old[4*n+k];
        sx=(1+ax.b)*tx+ax.a*old[5*n+k];
        bool outside_y=y<int(p.p)||y>=int(p.p+p.ny);
        bool outside_x=x<int(p.p)||x>=int(p.p+p.nx);
        next[2*n+k]=outside_y ? ay.b*gy+ay.a*old[2*n+k] : 0;
        next[3*n+k]=outside_x ? ax.b*gx+ax.a*old[3*n+k] : 0;
        next[4*n+k]=outside_y ? ay.b*ty+ay.a*old[4*n+k] : 0;
        next[5*n+k]=outside_x ? ax.b*tx+ax.a*old[5*n+k] : 0;
    }
    float rhs=sy+sx;
    int s=source_map[k];
    if(s>=0) rhs-=source[s*p.nt+t];
    float v=velocity[k];
    next[k]=2*old[k]-old[n+k]+(v*v)*p.dt2*rhs;
    next[n+k]=old[k];
    if(p.save_history) history[t*n+k]=rhs;
}

// Transpose of the old-memory recurrence, with lambda on the updated state:
//   Q = V^2 dt^2 lambda_u
//   Tbar = (1+b) Q + b lambda_zeta
//   A = (1+b) Tbar; B = c Tbar + b lambda_psi; C = Tbar.
// Then ubar receives D2(A)-D1(B), psi_bar receives -a D1(C),
// and the local a-memory branches are added by adjoint_step. D2 is symmetric
// and D1 is skew-symmetric on both the zero-guard and periodic domains.
// Three planes per axis: cotangents of D2(u), D1(u), D1(a*psi).
kernel void adjoint_terms(device const float* l [[buffer(0)]], device const float* v [[buffer(1)]],
                          device const Profile* py [[buffer(2)]], device const Profile* px [[buffer(3)]],
                          device float* terms [[buffer(4)]], constant Params& p [[buffer(5)]],
                          uint k [[thread_position_in_grid]]) {
    uint n=p.w*p.h;
    if(k>=n) return;
    Profile ay=py[k/p.w], ax=px[k%p.w];
    float q=(v[k]*v[k])*p.dt2*l[k];
    float ty=(1+ay.b)*q+ay.b*l[4*n+k];
    float tx=(1+ax.b)*q+ax.b*l[5*n+k];
    terms[k]=(1+ay.b)*ty;
    terms[n+k]=ay.c*ty+ay.b*l[2*n+k];
    terms[2*n+k]=ty;
    terms[3*n+k]=(1+ax.b)*tx;
    terms[4*n+k]=ax.c*tx+ax.b*l[3*n+k];
    terms[5*n+k]=tx;
}
kernel void adjoint_step(device const float* l [[buffer(0)]], device float* prev [[buffer(1)]],
                         device const float* v [[buffer(2)]],
                         device const Profile* py [[buffer(3)]], device const Profile* px [[buffer(4)]],
                         device const float* terms [[buffer(5)]], device const float* history [[buffer(6)]],
                         device float* grad [[buffer(7)]], device const float* residual [[buffer(8)]],
                         device const int* head [[buffer(9)]], device const int* links [[buffer(10)]],
                         constant Params& p [[buffer(11)]], constant uint& t [[buffer(12)]],
                         uint k [[thread_position_in_grid]]) {
    uint n=p.w*p.h;
    if(k>=n) return;
    int y=k/p.w,x=k%p.w;
    Profile ay=py[y],ax=px[x];
    float u=2*l[k]+l[n+k];
    u+=d2(terms,y,x,true,p)-d1(terms+n,y,x,true,p);
    u+=d2(terms+3*n,y,x,false,p)-d1(terms+4*n,y,x,false,p);
    for(int r=head[k];r>=0;r=links[r]) u+=residual[r*p.nt+t];
    prev[k]=u;
    prev[n+k]=-l[k];
    float q=(v[k]*v[k])*p.dt2*l[k];
    bool oy=y<int(p.p)||y>=int(p.p+p.ny);
    bool ox=x<int(p.p)||x>=int(p.p+p.nx);
    prev[2*n+k]=oy ? ay.a*(l[2*n+k]-d1(terms+2*n,y,x,true,p)) : 0;
    prev[3*n+k]=ox ? ax.a*(l[3*n+k]-d1(terms+5*n,y,x,false,p)) : 0;
    prev[4*n+k]=oy ? ay.a*(q+l[4*n+k]) : 0;
    prev[5*n+k]=ox ? ax.a*(q+l[5*n+k]) : 0;
    grad[k]+=(2*v[k])*p.dt2*history[t*n+k]*l[k];
}
kernel void objective(device const float* predicted [[buffer(0)]], device const float* observed [[buffer(1)]],
                      device float* residual [[buffer(2)]], device float* value [[buffer(3)]],
                      constant Params& p [[buffer(4)]], uint k [[thread_position_in_grid]]) {
    if(k!=0) return;
    // Compensated float32 reduction of the unnormalized sum.
    float sum=0, correction=0;
    for(uint j=0;j<p.nr*p.nt;++j) {
        float r=predicted[j]-observed[j]; residual[j]=r;
        float term=0.5f*r*r-correction;
        float next=sum+term; correction=(next-sum)-term; sum=next;
    }
    value[0]=sum;
}
kernel void extension_transpose(device const float* extended [[buffer(0)]], device float* grad [[buffer(1)]],
                                constant Params& p [[buffer(2)]], uint k [[thread_position_in_grid]]) {
    if(k>=p.ny*p.nx) return;
    uint y=k/p.nx,x=k%p.nx;
    uint y0=y==0 ? 0 : y+p.p, y1=y==p.ny-1 ? p.h : y+p.p+1;
    uint x0=x==0 ? 0 : x+p.p, x1=x==p.nx-1 ? p.w : x+p.p+1;
    float sum=0;
    for(uint j=y0;j<y1;++j) for(uint i=x0;i<x1;++i) sum+=extended[j*p.w+i];
    grad[k]=sum;
}
struct Raster { float4 position [[position]]; };
vertex Raster fullscreen(uint id [[vertex_id]]) {
    Raster out;
    out.position=float4(id==1 ? 3.0f : -1.0f, id==2 ? 3.0f : -1.0f, 0, 1);
    return out;
}
fragment float4 color(float4 position [[position]], device const float* u [[buffer(0)]],
                       constant Params& p [[buffer(1)]]) {
    uint x=uint(position.x),y=uint(position.y);
    float q=clamp(u[(y+p.p)*p.w+x+p.p]/p.scale,-1.0f,1.0f);
    return float4(max(q,0.0f),0,max(-q,0.0f),1);
}
kernel void probe(texture2d<float,access::read> image [[texture(0)]],
                  device const uint* loc [[buffer(0)]], device float4* samples [[buffer(1)]],
                  constant Params& p [[buffer(2)]], constant uint& t [[buffer(3)]],
                  uint r [[thread_position_in_grid]]) {
    if(r<p.nr) {
        uint k=loc[r];
        samples[t*p.nr+r]=image.read(uint2(k%p.w-p.p,k/p.w-p.p));
    }
}
kernel void read_image(texture2d<float,access::read> image [[texture(0)]],
                       device float4* rgba [[buffer(0)]], constant Params& p [[buffer(1)]],
                       uint k [[thread_position_in_grid]]) {
    if(k<p.ny*p.nx) rgba[k]=image.read(uint2(k%p.nx,k/p.nx));
}
