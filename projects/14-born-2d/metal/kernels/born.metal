#include <metal_stdlib>
using namespace metal;
struct P { uint ny,nx,shots,nt,ns,nr,r,t,periodic,save_history; float ih,ih2,dt2; };
inline uint at(int y,int x,uint s,constant P& p) { if(p.periodic){y=((y%int(p.ny))+int(p.ny))%int(p.ny);x=((x%int(p.nx))+int(p.nx))%int(p.nx);} return s*p.ny*p.nx+uint(y)*p.nx+uint(x); }
inline bool active(int y,int x,constant P& p){return p.periodic || (y>=int(p.r)&&x>=int(p.r)&&y<int(p.ny-p.r)&&x<int(p.nx-p.r));}
inline float c1(uint j,constant P& p){return p.r==1?0.5f:(j==1?2.0f/3.0f:-1.0f/12.0f);}
inline float c2(uint j,constant P& p){return p.r==1?1.0f:(j==1?4.0f/3.0f:-1.0f/12.0f);}
inline float axis(device const float* q,device const float* psi,device const float* zeta,device float* pn,device float* zn,device const float* prof,uint y,uint x,uint shot,uint ax,constant P& p){
 uint i=at(y,x,shot,p), pos=ax==0?y:x, len=ax==0?p.ny:p.nx; float d=0,l=0,ap=0;
 for(uint j=1;j<=p.r;j++){int dy=ax==0?int(j):0,dx=ax==1?int(j):0; uint hi=at(int(y)+dy,int(x)+dx,shot,p),lo=at(int(y)-dy,int(x)-dx,shot,p); d+=c1(j,p)*(q[hi]-q[lo]); l+=c2(j,p)*((q[hi]-q[i])+(q[lo]-q[i])); if(!p.periodic) ap+=c1(j,p)*(prof[pos+j]*psi[hi]-prof[pos-j]*psi[lo]); }
 d*=p.ih;l*=p.ih2;ap*=p.ih; float a=prof[pos],b=prof[len+pos],db=prof[2*len+pos]; float T=(1+b)*l+db*d+ap; pn[i]=b*d+a*psi[i]; zn[i]=b*T+a*zeta[i]; return (1+b)*T+a*zeta[i];
}
kernel void record(device const float* u [[buffer(0)]],device const float* s [[buffer(1)]],device const uint* loc [[buffer(2)]],device float* bu [[buffer(3)]],device float* bs [[buffer(4)]],constant P& p [[buffer(5)]],uint i [[thread_position_in_grid]]) {if(i>=p.shots*p.nr)return; uint k=i*p.nt+p.t, idx=(i/p.nr)*p.ny*p.nx+loc[i];bu[k]=u[idx];bs[k]=s[idx];}
kernel void step(device const float* u [[buffer(0)]],device const float* um [[buffer(1)]],device float* un [[buffer(2)]],device const float* s [[buffer(3)]],device const float* sm [[buffer(4)]],device float* sn [[buffer(5)]],device const float* mem [[buffer(6)]],device float* memn [[buffer(7)]],device const float* v [[buffer(8)]],device const float* dv [[buffer(9)]],device const float* py [[buffer(10)]],device const float* px [[buffer(11)]],device const float* src [[buffer(12)]],device const uint* loc [[buffer(13)]],device float* hist [[buffer(14)]],constant P& p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
 uint cells=p.ny*p.nx,total=cells*p.shots;if(i>=total)return;uint cell=i%cells,shot=i/cells,y=cell/p.nx,x=cell%p.nx;if(!active(y,x,p))return;
 float hu=axis(u,mem,mem+total,memn,memn+total,py,y,x,shot,0,p)+axis(u,mem+2*total,mem+3*total,memn+2*total,memn+3*total,px,y,x,shot,1,p);
 float hs=axis(s,mem+4*total,mem+5*total,memn+4*total,memn+5*total,py,y,x,shot,0,p)+axis(s,mem+6*total,mem+7*total,memn+6*total,memn+7*total,px,y,x,shot,1,p);
 float f=0;for(uint j=0;j<p.ns;j++)if(loc[shot*p.ns+j]==cell)f+=src[(shot*p.ns+j)*p.nt+p.t]; float force=hu-f;if(p.save_history)hist[p.t*total+i]=force;float vel=v[cell];un[i]=2*u[i]-um[i]+p.dt2*vel*vel*force;sn[i]=2*s[i]-sm[i]+p.dt2*(vel*vel*hs+2*vel*dv[cell]*force);
}
// Reverse the local PML algebra first, then apply the transposed stencils
// in adjstep. Guard entries of tmp are zero, matching the forward boundary.
kernel void adjprep(device const float* l [[buffer(0)]],device const float* mem [[buffer(1)]],device float* tmp [[buffer(2)]],device const float* v [[buffer(3)]],device const float* py [[buffer(4)]],device const float* px [[buffer(5)]],constant P& p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
 uint cells=p.ny*p.nx,total=cells*p.shots;if(i>=total)return;uint cell=i%cells,y=cell/p.nx,x=cell%p.nx;if(!active(y,x,p)){for(uint k=0;k<6;k++)tmp[k*total+i]=0;return;}
 float w=p.dt2*v[cell]*v[cell]*l[i]; for(uint ax=0;ax<2;ax++){device const float* pr=ax==0?py:px;uint pos=ax==0?y:x,len=ax==0?p.ny:p.nx;float b=pr[len+pos],db=pr[2*len+pos];float t=(1+b)*w+b*mem[(2*ax+1)*total+i];tmp[(3*ax)*total+i]=t;tmp[(3*ax+1)*total+i]=(1+b)*t;tmp[(3*ax+2)*total+i]=db*t+b*mem[(2*ax)*total+i];}
}
kernel void adjstep(device const float* l [[buffer(0)]],device const float* lm [[buffer(1)]],device float* ln [[buffer(2)]],device const float* mem [[buffer(3)]],device float* memn [[buffer(4)]],device const float* tmp [[buffer(5)]],device const float* v [[buffer(6)]],device const float* py [[buffer(7)]],device const float* px [[buffer(8)]],device const float* hist [[buffer(9)]],device float* grad [[buffer(10)]],device const float* data [[buffer(11)]],device const float* obs [[buffer(12)]],device const uint* rloc [[buffer(13)]],constant P& p [[buffer(14)]],uint i [[thread_position_in_grid]]) {
 uint cells=p.ny*p.nx,total=cells*p.shots;if(i>=total)return;uint cell=i%cells,shot=i/cells,y=cell/p.nx,x=cell%p.nx;if(!active(y,x,p))return; float value=2*l[i]-lm[i];
 for(uint ax=0;ax<2;ax++){device const float* pr=ax==0?py:px;uint pos=ax==0?y:x,len=ax==0?p.ny:p.nx; float dd=0,dt=0,dp=0;for(uint j=1;j<=p.r;j++){int dy=ax==0?int(j):0,dx=ax==1?int(j):0;uint hi=at(int(y)+dy,int(x)+dx,shot,p),lo=at(int(y)-dy,int(x)-dx,shot,p);dd+=c2(j,p)*((tmp[(3*ax+1)*total+hi]-tmp[(3*ax+1)*total+i])+(tmp[(3*ax+1)*total+lo]-tmp[(3*ax+1)*total+i]));dt+=c1(j,p)*(tmp[(3*ax+2)*total+lo]-tmp[(3*ax+2)*total+hi]);dp+=c1(j,p)*(tmp[(3*ax)*total+lo]-tmp[(3*ax)*total+hi]);} value+=dd*p.ih2+dt*p.ih;float a=pr[pos];memn[(2*ax)*total+i]=a*(mem[(2*ax)*total+i]+dp*p.ih);memn[(2*ax+1)*total+i]=a*(p.dt2*v[cell]*v[cell]*l[i]+mem[(2*ax+1)*total+i]);}
 grad[i]+=p.dt2*2*v[cell]*hist[p.t*total+i]*l[i];for(uint r=0;r<p.nr;r++)if(rloc[shot*p.nr+r]==cell){uint k=(shot*p.nr+r)*p.nt+p.t;value+=data[k]-obs[k];}ln[i]=value;
}
