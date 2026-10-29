#include <metal_stdlib>
using namespace metal;
// q is horizontal slowness times surface velocity.
float2 layer_integral(device const float *interfaces, device const float *v, uint layers, float depth, float q) {
    float x=0.0f, t=0.0f, top=0.0f;
    for (uint j=0; j<layers && top<depth; ++j) {
        float bottom = j+1<layers ? min(depth,interfaces[j]) : depth;
        float h=bottom-top, s=q*(v[j]/v[0]);
        float c=sqrt(max(1.0f-s*s,1.0e-14f));
        x+=h*s/c; t+=h/(v[j]*c); top=bottom;
    }
    return float2(x,t);
}

// sqrt(1+x*x), scaled to avoid overflow for large positive tangents.
float unit_hypot(float x) {
    if (x>1.0f) {
        float inverse=1.0f/x;
        return x*sqrt(1.0f+inverse*inverse);
    }
    return sqrt(1.0f+x*x);
}

// u = tan(theta_max), where theta_max is the angle in a fastest traversed
// layer. Unlike p*vmax, u retains the distance from critical incidence when
// the offset/depth ratio is large. For r = v/vmax and a^2 = 1-r^2,
// tan(theta) = r*u/hypot(1,a*u), sec(theta) = hypot(1,u)/hypot(1,a*u).
float2 bent_layer_integral(device const float *interfaces, device const float *v, uint layers, float depth, float vmax, float u) {
    float x=0.0f, t=0.0f, top=0.0f;
    float norm=unit_hypot(u);
    for (uint j=0; j<layers && top<depth; ++j) {
        float bottom=j+1<layers ? min(depth,interfaces[j]) : depth;
        float h=bottom-top, r=v[j]/vmax;
        // Subtract velocities before dividing so close velocities retain
        // their difference; the fastest layers have a=0 exactly.
        float a=sqrt(((vmax-v[j])/vmax)*(1.0f+r));
        float denominator=unit_hypot(a*u);
        x+=h*(u/denominator)*r;
        t+=(h/v[j])*(norm/denominator);
        top=bottom;
    }
    return float2(x,t);
}
kernel void layered(device const float *interfaces [[buffer(0)]], device const float *v [[buffer(1)]], device const float *rays [[buffer(2)]], device float *endpoints [[buffer(3)]], device float *times [[buffer(4)]], device float *angles [[buffer(5)]], constant uint &layers [[buffer(6)]], constant float &depth [[buffer(7)]], constant uint &n [[buffer(8)]], constant uint &bend [[buffer(9)]], uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    float2 result;
    if (bend) {
        float vmax=v[0];
        for (uint j=1;j<layers;++j) { if (interfaces[j-1]>=depth) break; vmax=max(vmax,v[j]); }
        float fastest_height=0.0f, top=0.0f;
        for (uint j=0;j<layers && top<depth;++j) {
            float bottom=j+1<layers ? min(depth,interfaces[j]) : depth;
            if (v[j]==vmax) fastest_height+=bottom-top;
            top=bottom;
        }
        float target=abs(rays[i]);
        // x(u) >= fastest_height*u, so this is a guaranteed upper bound.
        // Expand from the homogeneous slope to keep the bracket narrow even
        // when the fastest layer is very thin, then bisect in tangent space.
        float upper=target/fastest_height;
        float lo=0.0f, hi=min(max(1.0f,target/depth),upper);
        for (uint k=0;k<128 && hi<upper;++k) {
            if (bent_layer_integral(interfaces,v,layers,depth,vmax,hi).x>=target) break;
            lo=hi;
            hi=min(2.0f*hi,upper);
        }
        for (uint k=0;k<32;++k) {
            float mid=lo+(hi-lo)*0.5f;
            if (bent_layer_integral(interfaces,v,layers,depth,vmax,mid).x<target) lo=mid;
            else hi=mid;
        }
        float u=lo+(hi-lo)*0.5f;
        float r=v[0]/vmax, a=sqrt(((vmax-v[0])/vmax)*(1.0f+r));
        angles[i]=copysign(atan2(r*u,unit_hypot(a*u)),rays[i]);
        result=bent_layer_integral(interfaces,v,layers,depth,vmax,u);
    } else { result=layer_integral(interfaces,v,layers,depth,sin(rays[i])); }
    endpoints[2*i]=bend ? rays[i] : result.x;
    endpoints[2*i+1]=depth; times[i]=result.y;
}
kernel void gradient(device const float *rays [[buffer(0)]], device float *endpoints [[buffer(1)]], device float *times [[buffer(2)]], device float *angles [[buffer(3)]], constant float &v0 [[buffer(4)]], constant float &g [[buffer(5)]], constant float &depth [[buffer(6)]], constant uint &n [[buffer(7)]], constant uint &bend [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    float v1=v0+g*depth, x;
    if (bend) {
        x=rays[i];
        float r2=x*x+depth*depth;
        // The circle geometry gives tan(theta) as the ratio of these terms.
        angles[i]=atan2(2.0f*v0*x, 2.0f*v0*depth+g*r2);
    } else {
        float q=sin(rays[i]), s1=q*(v1/v0);
        float c0=sqrt(max(0.0f,1.0f-q*q)), c1=sqrt(max(0.0f,1.0f-s1*s1));
        x=depth*q*((v0+v1)/v0)/(c0+c1);
    }
    float distance=length(float2(x,depth));
    float u=g*distance/(2.0f*sqrt(v0*v1));
    float ratio=abs(u)<1.0e-3f ? 1.0f-u*u/6.0f : asinh(u)/u;
    times[i]=distance/sqrt(v0*v1)*ratio;
    endpoints[2*i]=x; endpoints[2*i+1]=depth;
}
