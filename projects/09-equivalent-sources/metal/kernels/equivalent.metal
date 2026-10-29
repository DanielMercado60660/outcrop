#include <metal_stdlib>
using namespace metal;
inline float green(device const float *x, uint i, device const float *p, uint j) {
    float dx=x[3*i]-p[3*j], dy=x[3*i+1]-p[3*j+1], dz=x[3*i+2]-p[3*j+2];
    return 1.0f/sqrt(dx*dx+dy*dy+dz*dz);
}
kernel void make_matrix(device const float *x [[buffer(0)]], device const float *p [[buffer(1)]], device const float *w [[buffer(2)]], device const uint *di [[buffer(3)]], device const uint *si [[buffer(4)]], device float *a [[buffer(5)]], device float *s [[buffer(6)]], constant uint &n [[buffer(7)]], constant uint &m [[buffer(8)]], constant float &damping [[buffer(9)]], uint j [[thread_position_in_grid]]) {
    if (j>=m) return;
    float mean=0.0f;
    for (uint i=0;i<n;i++) mean+=green(x,di[i],p,si[j]);
    mean/=float(n);
    float variance=0.0f;
    for (uint i=0;i<n;i++) { float d=green(x,di[i],p,si[j])-mean; variance+=d*d; }
    float scale=sqrt(variance/float(n));
    if (scale==0.0f) scale=1.0f;
    s[j]=scale;
    for (uint i=0;i<n;i++) a[i*m+j]=sqrt(w[di[i]])*green(x,di[i],p,si[j])/scale;
    for (uint i=0;i<m;i++) a[(n+i)*m+j]=i==j?sqrt(damping):0.0f;
}
kernel void make_rhs(device const float *r [[buffer(0)]], device const float *w [[buffer(1)]], device const uint *di [[buffer(2)]], device float *b [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &m [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i<n) b[i]=sqrt(w[di[i]])*r[di[i]]; else if(i<n+m) b[i]=0.0f;
}
// One-sided Jacobi SVD of the weighted, scaled Jacobian. Rotating the
// columns and the right singular vectors together avoids squaring its
// condition number and yields the minimum-norm solution in beta space.
kernel void solve_svd(device float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *s [[buffer(2)]], device float *c [[buffer(3)]], device float *v [[buffer(4)]], device float *beta [[buffer(5)]], constant uint &n [[buffer(6)]], constant uint &m [[buffer(7)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    for (uint i=0; i<m; ++i)
        for (uint j=0; j<m; ++j) v[i*m+j] = i==j ? 1.0f : 0.0f;
    for (uint sweep=0; sweep<100; ++sweep) {
        bool changed=false;
        for (uint p=0; p<m; ++p) for (uint q=p+1; q<m; ++q) {
            float app=0.0f, aqq=0.0f, apq=0.0f;
            for (uint i=0; i<n; ++i) {
                float x=a[i*m+p], y=a[i*m+q];
                app+=x*x; aqq+=y*y; apq+=x*y;
            }
            if (app==0.0f || aqq==0.0f || abs(apq)<=1.0e-6f*sqrt(app)*sqrt(aqq)) continue;
            float tau=(aqq-app)/(2.0f*apq);
            float t=copysign(1.0f, tau)/(abs(tau)+sqrt(1.0f+tau*tau));
            float cs=rsqrt(1.0f+t*t), sn=cs*t;
            for (uint i=0; i<n; ++i) {
                float x=a[i*m+p], y=a[i*m+q];
                a[i*m+p]=cs*x-sn*y; a[i*m+q]=sn*x+cs*y;
            }
            for (uint i=0; i<m; ++i) {
                float x=v[i*m+p], y=v[i*m+q];
                v[i*m+p]=cs*x-sn*y; v[i*m+q]=sn*x+cs*y;
            }
            changed=true;
        }
        if (!changed) break;
    }
    float max_norm=0.0f;
    for (uint j=0; j<m; ++j) {
        float norm2=0.0f;
        for (uint i=0; i<n; ++i) norm2+=a[i*m+j]*a[i*m+j];
        beta[j]=norm2;
        max_norm=max(max_norm,sqrt(norm2));
    }
    float cutoff=1.1920929e-7f*float(max(n,m))*max_norm;
    for (uint j=0; j<m; ++j) {
        float norm2=beta[j], dot=0.0f;
        if (sqrt(norm2)>cutoff) {
            for (uint i=0; i<n; ++i) dot+=a[i*m+j]*b[i];
            beta[j]=dot/norm2;
        } else beta[j]=0.0f;
    }
    for (uint i=0; i<m; ++i) {
        float value=0.0f;
        for (uint j=0; j<m; ++j) value+=v[i*m+j]*beta[j];
        c[i]=value/s[i];
    }
}
kernel void solve(device float *a [[buffer(0)]], device float *b [[buffer(1)]], device const float *s [[buffer(2)]], device float *c [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &m [[buffer(5)]], uint tid [[thread_position_in_grid]]) {
    if(tid!=0) return;
    uint rows=n+m;
    for(uint j=0;j<m;j++) {
        float norm2=0.0f;
        for(uint i=j;i<rows;i++) norm2+=a[i*m+j]*a[i*m+j];
        float norm=sqrt(norm2);
        if(norm==0.0f) continue;
        float alpha=a[j*m+j]>=0.0f?-norm:norm;
        a[j*m+j]-=alpha;
        float vv=0.0f;
        for(uint i=j;i<rows;i++) vv+=a[i*m+j]*a[i*m+j];
        float factor=2.0f/vv;
        for(uint k=j+1;k<m;k++) {
            float dot=0.0f;
            for(uint i=j;i<rows;i++) dot+=a[i*m+j]*a[i*m+k];
            dot*=factor;
            for(uint i=j;i<rows;i++) a[i*m+k]-=dot*a[i*m+j];
        }
        float dot=0.0f;
        for(uint i=j;i<rows;i++) dot+=a[i*m+j]*b[i];
        dot*=factor;
        for(uint i=j;i<rows;i++) b[i]-=dot*a[i*m+j];
        a[j*m+j]=alpha;
    }
    for(uint jj=m;jj>0;jj--) {
        uint j=jj-1; float v=b[j];
        for(uint k=j+1;k<m;k++) v-=a[j*m+k]*b[k];
        b[j]=a[j*m+j]==0.0f?0.0f:v/a[j*m+j];
    }
    for(uint j=0;j<m;j++) c[j]=b[j]/s[j];
}
kernel void update_residual(device const float *x [[buffer(0)]], device const float *p [[buffer(1)]], device const uint *si [[buffer(2)]], device const float *c [[buffer(3)]], device float *r [[buffer(4)]], constant uint &n [[buffer(5)]], constant uint &m [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if(i>=n) return; float field=0.0f;
    for(uint j=0;j<m;j++) field+=green(x,i,p,si[j])*c[j];
    r[i]-=field;
}
kernel void accumulate(device const uint *si [[buffer(0)]], device const float *c [[buffer(1)]], device float *total [[buffer(2)]], constant uint &m [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if(tid!=0) return; for(uint j=0;j<m;j++) total[si[j]]+=c[j];
}
kernel void predict(device const float *x [[buffer(0)]], device const float *p [[buffer(1)]], device const float *c [[buffer(2)]], device float *out [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &m [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if(i>=n) return; float value=0.0f;
    for(uint j=0;j<m;j++) value+=green(x,i,p,j)*c[j];
    out[i]=value;
}
