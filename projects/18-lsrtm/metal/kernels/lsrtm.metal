#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct Params { uint nz,nx,mz,mx,shots,nsrc,nrec,nt,pad,n; float hz,hx,dt2; };
inline bool active(uint i, constant Params& c) {
    uint z=(i%c.n)/c.mx, x=i%c.mx;
    return z>=2 && z<c.mz-2 && x>=2 && x<c.mx-2;
}
inline float d1(device const float* q,uint i,uint s,float h) {
    return ((2.0f/3.0f)*(q[i+s]-q[i-s])-(1.0f/12.0f)*(q[i+2*s]-q[i-2*s]))/h;
}
inline float d2(device const float* q,uint i,uint s,float h) {
    return ((4.0f/3.0f)*((q[i+s]-q[i])+(q[i-s]-q[i]))
        -(1.0f/12.0f)*((q[i+2*s]-q[i])+(q[i-2*s]-q[i])))/(h*h);
}
// Profiles are six padded planes: az,bz,dbz,ax,bx,dbx.
inline float dap(device const float* q,device const float* a,uint i,uint j,uint s,float h) {
    return ((2.0f/3.0f)*(a[j+s]*q[i+s]-a[j-s]*q[i-s])
        -(1.0f/12.0f)*(a[j+2*s]*q[i+2*s]-a[j-2*s]*q[i-2*s]))/h;
}
kernel void extend_model(device const float* m [[buffer(0)]],device const float* mask [[buffer(1)]],device float* dv [[buffer(2)]],
    constant Params& c [[buffer(3)]],uint j [[thread_position_in_grid]]) {
    if(j>=c.n)return;
    uint z=clamp(int(j/c.mx)-int(c.pad),0,int(c.nz)-1);
    uint x=clamp(int(j%c.mx)-int(c.pad),0,int(c.nx)-1);
    dv[j]=m[z*c.nx+x]*mask[z*c.nx+x];
}
// State planes: current, previous, psi_z, psi_x, zeta_z, zeta_x.
kernel void forward(device const float* in [[buffer(0)]],device float* out [[buffer(1)]],device const float* prof [[buffer(2)]],
    device const float* v [[buffer(3)]],device const float* dv [[buffer(4)]],device const float* src [[buffer(5)]],
    device const uint* sloc [[buffer(6)]],device float* drive [[buffer(7)]],constant Params& c [[buffer(8)]],
    constant uint& t [[buffer(9)]],constant uint& background [[buffer(10)]],uint i [[thread_position_in_grid]]) {
    uint total=c.n*c.shots;
    if(i>=total || !active(i,c))return;
    uint j=i%c.n, shot=i/c.n;
    float H=0;
    for(uint axis=0;axis<2;axis++) {
        uint stride=axis==0?c.mx:1;
        float h=axis==0?c.hz:c.hx;
        device const float* a=prof+(3*axis)*c.n;
        device const float* b=a+c.n;
        device const float* db=b+c.n;
        device const float* psi=in+(2+axis)*total;
        float g=d1(in,i,stride,h);
        float T=(1+b[j])*d2(in,i,stride,h)+db[j]*g+dap(psi,a,i,j,stride,h);
        H+=(1+b[j])*T+a[j]*in[(4+axis)*total+i];
        out[(2+axis)*total+i]=b[j]*g+a[j]*psi[i];
        out[(4+axis)*total+i]=b[j]*T+a[j]*in[(4+axis)*total+i];
    }
    float next=2*in[i]-in[total+i]+c.dt2*v[j]*v[j]*H;
    if(background) {
        float f=0;
        for(uint k=0;k<c.nsrc;k++) if(sloc[shot*c.nsrc+k]==j) f+=src[(shot*c.nsrc+k)*c.nt+t];
        next-=c.dt2*v[j]*v[j]*f;
        drive[t*total+i]=2*c.dt2*v[j]*(H-f);
    } else next+=dv[j]*drive[t*total+i];
    out[i]=next;
    out[total+i]=in[i];
}
kernel void sample(device const float* state [[buffer(0)]],device const uint* loc [[buffer(1)]],device float* data [[buffer(2)]],
    constant Params& c [[buffer(3)]],constant uint& t [[buffer(4)]],uint k [[thread_position_in_grid]]) {
    if(k<c.shots*c.nrec) data[k*c.nt+t]=state[(k/c.nrec)*c.n+loc[k]];
}
// Gather form of the exact transpose avoids floating atomic additions.
kernel void reverse_prepare(device const float* in [[buffer(0)]],device const float* prof [[buffer(1)]],device const float* v [[buffer(2)]],
    device float* tmp [[buffer(3)]],constant Params& c [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    uint total=c.n*c.shots;
    if(i>=total || !active(i,c))return;
    uint j=i%c.n;
    float V=c.dt2*v[j]*v[j]*in[i];
    for(uint axis=0;axis<2;axis++) {
        float b=prof[(3*axis+1)*c.n+j], db=prof[(3*axis+2)*c.n+j];
        float U=(1+b)*V+b*in[(4+axis)*total+i];
        tmp[axis*total+i]=(1+b)*U;
        tmp[(2+axis)*total+i]=db*U+b*in[(2+axis)*total+i];
        tmp[(4+axis)*total+i]=U;
    }
}
kernel void reverse_step(device const float* in [[buffer(0)]],device float* out [[buffer(1)]],device const float* tmp [[buffer(2)]],
    device const float* prof [[buffer(3)]],device const float* v [[buffer(4)]],device const float* drive [[buffer(5)]],
    device const float* data [[buffer(6)]],device const uint* loc [[buffer(7)]],device float* grad [[buffer(8)]],
    constant Params& c [[buffer(9)]],constant uint& t [[buffer(10)]],uint i [[thread_position_in_grid]]) {
    uint total=c.n*c.shots;
    if(i>=total || !active(i,c))return;
    uint j=i%c.n,shot=i/c.n;
    float L=in[i], V=c.dt2*v[j]*v[j]*L;
    grad[i]+=drive[t*total+i]*L;
    float q=2*L+in[total+i];
    for(uint axis=0;axis<2;axis++) {
        uint stride=axis==0?c.mx:1;
        float h=axis==0?c.hz:c.hx, a=prof[3*axis*c.n+j];
        q+=d2(tmp+axis*total,i,stride,h)-d1(tmp+(2+axis)*total,i,stride,h);
        out[(2+axis)*total+i]=a*(in[(2+axis)*total+i]-d1(tmp+(4+axis)*total,i,stride,h));
        out[(4+axis)*total+i]=a*(V+in[(4+axis)*total+i]);
    }
    for(uint k=0;k<c.nrec;k++) if(loc[shot*c.nrec+k]==j)q+=data[(shot*c.nrec+k)*c.nt+t];
    out[i]=q;
    out[total+i]=-L;
}
kernel void fold(device const float* padded [[buffer(0)]],device const float* mask [[buffer(1)]],device float* grad [[buffer(2)]],
    constant Params& c [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=c.nz*c.nx)return;
    uint z=i/c.nx,x=i%c.nx;
    uint zlo=z==0?2:z+c.pad, zhi=z==c.nz-1?c.mz-2:z+c.pad+1;
    uint xlo=x==0?2:x+c.pad, xhi=x==c.nx-1?c.mx-2:x+c.pad+1;
    float sum=0,cor=0;
    for(uint s=0;s<c.shots;s++)for(uint zz=zlo;zz<zhi;zz++)for(uint xx=xlo;xx<xhi;xx++) {
        float y=padded[s*c.n+zz*c.mx+xx]-cor, t=sum+y; cor=(t-sum)-y;sum=t;
    }
    grad[i]=mask[i]*sum;
}
kernel void axpy(device float* y [[buffer(0)]],device const float* x [[buffer(1)]],constant float& a [[buffer(2)]],constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) { if(i<n)y[i]+=a*x[i]; }
kernel void scale(device float* y [[buffer(0)]],constant float& a [[buffer(1)]],constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) { if(i<n)y[i]*=a; }
kernel void subtract(device const float* d [[buffer(0)]],device const float* p [[buffer(1)]],device float* r [[buffer(2)]],constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {if(i<n)r[i]=d[i]-p[i];}
kernel void dot(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device float* sums [[buffer(2)]],constant uint& n [[buffer(3)]],
    uint block [[thread_position_in_grid]]) {
    uint end=min(n,(block+1)*256);float sum=0,cor=0;
    for(uint i=block*256;i<end;i++){float y=a[i]*b[i]-cor,t=sum+y;cor=(t-sum)-y;sum=t;}
    sums[block]=sum;
}
