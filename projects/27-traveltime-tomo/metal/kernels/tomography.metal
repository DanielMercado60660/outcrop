#include <metal_stdlib>
using namespace metal;

kernel void ray_matrix(device const float *xe [[buffer(0)]], device const float *ze [[buffer(1)]], device const float *src [[buffer(2)]], device const float *rec [[buffer(3)]], device float *out [[buffer(4)]], constant uint &nx [[buffer(5)]], constant uint &nz [[buffer(6)]], constant uint &nr [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    uint nc=nx*nz; if(id>=nr*nc) return;
    uint r=id/nc, c=id%nc, ix=c%nx, iz=c/nx;
    float x0=src[2*r], z0=src[2*r+1], x1=rec[2*r], z1=rec[2*r+1];
    if(x0>x1 || (x0==x1 && z0>z1)) { float t=x0; x0=x1; x1=t; t=z0; z0=z1; z1=t; }
    float dx=x1-x0, dz=z1-z0, lo=0.0f, hi=1.0f;
    if(dx==0.0f) {
        if(x0<xe[ix] || x0>xe[ix+1] || (x0==xe[ix+1] && ix+1<nx)) { out[id]=0.0f; return; }
    } else { float a=(xe[ix]-x0)/dx, b=(xe[ix+1]-x0)/dx; lo=max(lo,min(a,b)); hi=min(hi,max(a,b)); }
    if(dz==0.0f) {
        if(z0<ze[iz] || z0>ze[iz+1] || (z0==ze[iz+1] && iz+1<nz)) { out[id]=0.0f; return; }
    } else { float a=(ze[iz]-z0)/dz, b=(ze[iz+1]-z0)/dz; lo=max(lo,min(a,b)); hi=min(hi,max(a,b)); }
    float scale=max(abs(dx),abs(dz));
    float length=scale==0.0f ? 0.0f : scale*sqrt((dx/scale)*(dx/scale)+(dz/scale)*(dz/scale));
    out[id]=max(hi-lo,0.0f)*length;
}

inline void add_comp(float v, thread float &sum, thread float &err) { float y=v-err; float t=sum+y; err=(t-sum)-y; sum=t; }
kernel void normal(device const float *a [[buffer(0)]], device float *h [[buffer(1)]], constant uint &m [[buffer(2)]], constant uint &n [[buffer(3)]], constant float &lambda2 [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if(id>=n*n) return; uint i=id/n, j=id%n; float s=0.0f,e=0.0f;
    for(uint r=0;r<m;r++) add_comp(a[r*n+i]*a[r*n+j],s,e);
    h[id]=s+(i==j ? lambda2 : 0.0f);
}
kernel void factor(device float *h [[buffer(0)]], constant uint &n [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    if(id) return;
    for(uint i=0;i<n;i++) for(uint j=0;j<=i;j++) {
        float s=0.0f,e=0.0f; for(uint k=0;k<j;k++) add_comp(h[i*n+k]*h[j*n+k],s,e);
        float v=h[i*n+j]-s; h[i*n+j]=(i==j) ? sqrt(v) : v/h[j*n+j];
    }
}
kernel void residual(device const float *a [[buffer(0)]], device const float *t [[buffer(1)]], device const float *s [[buffer(2)]], device float *r [[buffer(3)]], constant uint &m [[buffer(4)]], constant uint &n [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if(i>=m) return; float sum=0.0f,e=0.0f; for(uint j=0;j<n;j++) add_comp(a[i*n+j]*s[j],sum,e); r[i]=t[i]-sum;
}
kernel void rhs(device const float *a [[buffer(0)]], device const float *r [[buffer(1)]], device const float *s [[buffer(2)]], device const float *prior [[buffer(3)]], device float *b [[buffer(4)]], constant uint &m [[buffer(5)]], constant uint &n [[buffer(6)]], constant float &lambda2 [[buffer(7)]], uint j [[thread_position_in_grid]]) {
    if(j>=n) return; float sum=0.0f,e=0.0f; for(uint i=0;i<m;i++) add_comp(a[i*n+j]*r[i],sum,e); b[j]=sum-lambda2*(s[j]-prior[j]);
}
kernel void correction(device const float *h [[buffer(0)]], device float *b [[buffer(1)]], device float *s [[buffer(2)]], constant uint &n [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if(id) return;
    for(uint i=0;i<n;i++) { float sum=0.0f,e=0.0f; for(uint j=0;j<i;j++) add_comp(h[i*n+j]*b[j],sum,e); b[i]=(b[i]-sum)/h[i*n+i]; }
    for(uint ii=n;ii>0;ii--) { uint i=ii-1; float sum=0.0f,e=0.0f; for(uint j=i+1;j<n;j++) add_comp(h[j*n+i]*b[j],sum,e); b[i]=(b[i]-sum)/h[i*n+i]; }
    for(uint i=0;i<n;i++) s[i]+=b[i];
}
kernel void predict(device const float *a [[buffer(0)]], device const float *s [[buffer(1)]], device float *p [[buffer(2)]], constant uint &m [[buffer(3)]], constant uint &n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if(i>=m) return; float sum=0.0f,e=0.0f; for(uint j=0;j<n;j++) add_comp(a[i*n+j]*s[j],sum,e); p[i]=sum;
}
kernel void objective(device const float *r [[buffer(0)]], device const float *s [[buffer(1)]], device const float *prior [[buffer(2)]], device float *out [[buffer(3)]], constant uint &m [[buffer(4)]], constant uint &n [[buffer(5)]], constant float &lambda2 [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    if(id) return; float sum=0.0f,e=0.0f; for(uint i=0;i<m;i++) add_comp(r[i]*r[i],sum,e);
    for(uint j=0;j<n;j++) { float d=s[j]-prior[j]; add_comp(lambda2*d*d,sum,e); } out[0]=0.5f*sum;
}
kernel void gradient(device const float *a [[buffer(0)]], device const float *r [[buffer(1)]], device const float *s [[buffer(2)]], device const float *prior [[buffer(3)]], device float *g [[buffer(4)]], constant uint &m [[buffer(5)]], constant uint &n [[buffer(6)]], constant float &lambda2 [[buffer(7)]], uint j [[thread_position_in_grid]]) {
    if(j>=n) return; float sum=0.0f,e=0.0f; for(uint i=0;i<m;i++) add_comp(-a[i*n+j]*r[i],sum,e); g[j]=sum+lambda2*(s[j]-prior[j]);
}
