#include <metal_stdlib>
using namespace metal;
struct Params { uint nz,nx,shots,nt,ns,nr,radius,time; float inv_h,inv_h2,dt2; };
kernel void record(device const float* u [[buffer(0)]],device const uint* loc [[buffer(1)]],device float* out [[buffer(2)]],constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.shots*p.nr)return; uint shot=i/p.nr; out[i*p.nt+p.time]=u[shot*p.nz*p.nx+loc[i]];
}
kernel void step(device const float* u [[buffer(0)]],device const float* prev [[buffer(1)]],device float* next [[buffer(2)]],device const float* pz [[buffer(3)]],device float* npz [[buffer(4)]],device const float* px [[buffer(5)]],device float* npx [[buffer(6)]],device float* zz [[buffer(7)]],device float* zx [[buffer(8)]],device const float* v2 [[buffer(9)]],device const float* az [[buffer(10)]],device const float* ax [[buffer(11)]],device const float* coeff [[buffer(12)]],constant Params& p [[buffer(13)]],uint i [[thread_position_in_grid]]) {
 uint cells=p.nz*p.nx; if(i>=cells*p.shots)return; uint cell=i%cells,z=cell/p.nx,x=cell%p.nx;
 if(z<p.radius || z>=p.nz-p.radius || x<p.radius || x>=p.nx-p.radius)return;
 float center=u[i],gz=0,gx=0,lz=0,lx=0,apz=0,apx=0;
 for(uint j=1;j<=p.radius;++j){uint dz=j*p.nx; float c=coeff[j-1],d=coeff[4+j-1]; gz+=c*(u[i+dz]-u[i-dz]); gx+=c*(u[i+j]-u[i-j]); lz+=d*((u[i+dz]-center)+(u[i-dz]-center)); lx+=d*((u[i+j]-center)+(u[i-j]-center)); apz+=c*(az[z+j]*pz[i+dz]-az[z-j]*pz[i-dz]); apx+=c*(ax[x+j]*px[i+j]-ax[x-j]*px[i-j]);}
 gz*=p.inv_h; gx*=p.inv_h; lz*=p.inv_h2; lx*=p.inv_h2; apz*=p.inv_h; apx*=p.inv_h;
 float bz=az[p.nz+z],bx=ax[p.nx+x]; float tz=(1.0f+bz)*lz+az[2*p.nz+z]*gz+apz; float tx=(1.0f+bx)*lx+ax[2*p.nx+x]*gx+apx;
 float value=2.0f*center-prev[i]+v2[cell]*p.dt2*((1.0f+bz)*tz+az[z]*zz[i]+(1.0f+bx)*tx+ax[x]*zx[i]);
 npz[i]=bz*gz+az[z]*pz[i]; npx[i]=bx*gx+ax[x]*px[i]; zz[i]=bz*tz+az[z]*zz[i]; zx[i]=bx*tx+ax[x]*zx[i];
 next[i]=value;
}

// One thread owns each distinct (shot, cell). Stable host grouping preserves
// sequential float32 subtraction for coincident sources without atomic races.
kernel void inject(device float* next [[buffer(0)]],device const float* v2 [[buffer(1)]],device const float* source [[buffer(2)]],device const uint* cells [[buffer(3)]],device const uint* offsets [[buffer(4)]],device const uint* indices [[buffer(5)]],constant Params& p [[buffer(6)]],constant uint& count [[buffer(7)]],uint group [[thread_position_in_grid]]) {
 if(group>=count)return;
 uint i=cells[group],cell=i%(p.nz*p.nx);
 float value=next[i];
 for(uint s=offsets[group];s<offsets[group+1];++s)
  value-=v2[cell]*p.dt2*source[indices[s]*p.nt+p.time];
 next[i]=value;
}
