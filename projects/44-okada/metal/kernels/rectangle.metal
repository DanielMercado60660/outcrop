#include <metal_stdlib>
using namespace metal;
struct D { float v,x,y; D(float a=0):v(a),x(0),y(0){} D(float a,float b,float c):v(a),x(b),y(c){} };
D operator+(D a,D b){return D(a.v+b.v,a.x+b.x,a.y+b.y);}
D operator-(D a,D b){return D(a.v-b.v,a.x-b.x,a.y-b.y);}
D operator-(D a){return D(-a.v,-a.x,-a.y);}
D operator*(D a,D b){return D(a.v*b.v,a.x*b.v+a.v*b.x,a.y*b.v+a.v*b.y);}
D operator/(D a,D b){float t=1.0f/b.v;return D(a.v*t,(a.x-a.v*t*b.x)*t,(a.y-a.v*t*b.y)*t);}
D dsqrt(D a){float r=sqrt(a.v);return D(r,a.x/(2*r),a.y/(2*r));}
D dlog(D a){return D(log(a.v),a.x/a.v,a.y/a.v);}
D datan(D a){float t=1/(1+a.v*a.v);return D(atan(a.v),a.x*t,a.y*t);}
D snap(D a){if(abs(a.v)<1e-6f)a.v=0;return a;}
struct V{D a,b,c;};
V vadd(V a,V b){return {a.a+b.a,a.b+b.b,a.c+b.c};}
V vmul(V a,D s){return {a.a*s,a.b*s,a.c*s};}
struct Blocks{V a,b,c;};
Blocks corner(D xi,D eta,D q,bool kxi,bool ket,float z,float sd,float cd,float alpha,device const float *slip){
 xi=snap(xi);eta=snap(eta);q=snap(q);
 float a1=(1-alpha)/2,a2=alpha/2,a3=(1-alpha)/alpha,a4=1-alpha,a5=alpha;
 D x2=xi*xi,e2=eta*eta,q2=q*q,r2=x2+e2+q2,r=dsqrt(r2),r3=r*r2;
 D y=eta*cd+q*sd,d=eta*sd-q*cd,rx=r+xi,re=r+eta;
 D alx=kxi ? -dlog(r-xi):dlog(rx),ale=ket ? -dlog(r-eta):dlog(re);
 D x11=kxi ? D(0):D(1)/(r*rx),y11=ket ? D(0):D(1)/(r*re);
 D x32=kxi ? D(0):(r+rx)*x11*x11/r,y32=ket ? D(0):(r+re)*y11*y11/r;
 D xe=xi*eta,qr=q*r,tt;
 // Okada's zero-q angular convention needs a separate corner-edge rule.
 // When xi*eta also vanishes, both the angular value and its derivatives
 // are zero.  Retaining eta*x11*dq here introduces a spurious gradient
 // at vertical faults (including the negative-dip KET logarithm branch).
 // Away from that edge, -atan(q*r/(xi*eta)) retains the regular limiting
 // derivative while giving the prescribed zero angular value at q=0.
 if(q.v==0){if(xe.v==0)tt=D(0);else tt=-datan(qr/xe);}else tt=datan(xe/qr);
 D xy=xi*y11,qx=q*x11,qy=q*y11;
 V ua1={tt/2+a2*xi*qy,a2*q/r,a1*ale-a2*q*qy};
 V ua2={a2*q/r,tt/2+a2*eta*qx,a1*alx-a2*q*qx};
 V ua3={-a1*ale-a2*q*qy,-a1*alx-a2*q*qx,tt/2-a2*(eta*qx+xi*qy)};
 D rd=r+d,ai3,ai4;
 if(cd!=0){D xx=dsqrt(x2+q2),angle;if(xi.v==0)angle=D(0);else angle=datan((eta*(xx+q*cd)+xx*(r+xx)*sd)/(xi*(r+xx)*cd));ai4=(xi/rd*sd*cd+2*angle)/(cd*cd);ai3=(y*cd/rd-ale+sd*dlog(rd))/(cd*cd);}
 else{ai3=(eta/rd+y*q/(rd*rd)-ale)/2;ai4=xi*y/(2*rd*rd);}
 D ai1=-xi/rd*cd-ai4*sd,ai2=dlog(rd)+ai3*sd;
 V ub1={-xi*qy-tt-a3*ai1*sd,-q/r+a3*y/rd*sd,q*qy-a3*ai2*sd};
 V ub2={-q/r+a3*ai3*sd*cd,-eta*qx-tt-a3*xi/rd*sd*cd,q*qx+a3*ai4*sd*cd};
 V ub3={q*qy-a3*ai3*sd*sd,q*qx+a3*xi/rd*sd*sd,eta*qx+xi*qy-tt-a3*ai4*sd*sd};
 D c=d+z,z32=sd/r3-(q*cd-z)*y32;
 V uc1={a4*xy*cd-a5*xi*q*z32,a4*(cd/r+2*qy*sd)-a5*c*q/r3,a4*qy*cd-a5*(c*eta/r3-z*y11+x2*z32)};
 V uc2={a4*cd/r-qy*sd-a5*c*q/r3,a4*y*x11-a5*c*eta*q*x32,-d*x11-xy*sd-a5*c*(x11-q2*x32)};
 V uc3={-a4*(sd/r+qy*cd)-a5*(z*y11-q2*z32),2*a4*xy*sd+d*x11-a5*c*(x11-q2*x32),a4*(y*x11+xy*cd)+a5*q*(c*eta*x32+xi*z32)};
 float s0=slip[0],s1=slip[1],s2=slip[2];
 return {vmul(vadd(vadd(vmul(ua1,s0),vmul(ua2,s1)),vmul(ua3,s2)),1/(2*M_PI_F)),vmul(vadd(vadd(vmul(ub1,s0),vmul(ub2,s1)),vmul(ub3,s2)),1/(2*M_PI_F)),vmul(vadd(vadd(vmul(uc1,s0),vmul(uc2,s1)),vmul(uc3,s2)),1/(2*M_PI_F))};
}
kernel void rectangle(device const float *xyz [[buffer(0)]],device const float *slip [[buffer(1)]],device float *disp [[buffer(2)]],device float *grad [[buffer(3)]],constant float &depth [[buffer(4)]],constant float &sd [[buffer(5)]],constant float &cd [[buffer(6)]],constant float &alpha [[buffer(7)]],device const float *bounds [[buffer(8)]],constant uint &n [[buffer(9)]],uint idx [[thread_position_in_grid]]){
 if(idx>=n)return;
 D x=D(xyz[3*idx],1,0),y=D(xyz[3*idx+1],0,1);float z=xyz[3*idx+2];
 D xi[2]={snap(x-bounds[0]),snap(x-bounds[1])};V total={D(0),D(0),D(0)};
 for(int image=0;image<2;image++){
  float dep=depth+(image ? -z:z);D p=y*cd+dep*sd,q=snap(y*sd-dep*cd);
  D e[2]={snap(p-bounds[2]),snap(p-bounds[3])};
  D r12=dsqrt(xi[0]*xi[0]+e[1]*e[1]+q*q),r21=dsqrt(xi[1]*xi[1]+e[0]*e[0]+q*q),r22=dsqrt(xi[1]*xi[1]+e[1]*e[1]+q*q);
  bool kx[2]={xi[0].v<0&&r21.v+xi[1].v<1e-6f,xi[0].v<0&&r22.v+xi[1].v<1e-6f};
  bool ke[2]={e[0].v<0&&r12.v+e[1].v<1e-6f,e[0].v<0&&r22.v+e[1].v<1e-6f};
  for(int i=0;i<2;i++)for(int j=0;j<2;j++){
   Blocks t=corner(xi[i],e[j],q,kx[j],ke[i],z,sd,cd,alpha,slip);V u;
   if(image){V v=vadd(vadd(t.a,t.b),vmul(t.c,z)),w=vadd(vadd(t.a,t.b),vmul(t.c,-z));u={v.a,v.b*cd-v.c*sd,w.b*sd+w.c*cd};}
   else u={-t.a.a,-t.a.b*cd+t.a.c*sd,-t.a.b*sd-t.a.c*cd};
   total=vadd(total,vmul(u,i==j ? 1:-1));
  }
 }
 D o[3]={total.a,total.b,total.c};for(uint k=0;k<3;k++){disp[3*idx+k]=o[k].v;grad[6*idx+2*k]=o[k].x;grad[6*idx+2*k+1]=o[k].y;}
}
