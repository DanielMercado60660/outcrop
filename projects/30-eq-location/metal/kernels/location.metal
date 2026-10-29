#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline float dist(float x, float y, float z, device const float *s) {
    float dx=x-s[0], dy=y-s[1], dz=z-s[2];
    return sqrt(dx*dx+dy*dy+dz*dz);
}

kernel void misfit(device const float *s [[buffer(0)]], device const float *a [[buffer(1)]],
                   device const float *sig [[buffer(2)]], device const float *m [[buffer(3)]],
                   device float *pred [[buffer(4)]], device float *obj [[buffer(5)]],
                   device float *grad [[buffer(6)]], constant uint &n [[buffer(7)]],
                   constant uint &b [[buffer(8)]], constant float &v [[buffer(9)]], uint e [[thread_position_in_grid]]) {
    if(e>=b) return;
    float x=m[4*e], y=m[4*e+1], z=m[4*e+2], t=m[4*e+3];
    float loss=0.0f, g0=0.0f, g1=0.0f, g2=0.0f, g3=0.0f;
    for(uint i=0;i<n;i++) {
        float dx=x-s[3*i], dy=y-s[3*i+1], dz=z-s[3*i+2];
        float d=sqrt(dx*dx+dy*dy+dz*dz);
        float p=t+d/v, r=p-a[e*n+i], q=r/sig[i];
        pred[e*n+i]=p; loss+=0.5f*q*q;
        float w=q/sig[i], f=w/(v*d);
        g0+=f*dx; g1+=f*dy; g2+=f*dz; g3+=w;
    }
    obj[e]=loss; grad[4*e]=g0; grad[4*e+1]=g1; grad[4*e+2]=g2; grad[4*e+3]=g3;
}

kernel void grid_nodes(device const float *s [[buffer(0)]], device const float *a [[buffer(1)]],
                       device const float *sig [[buffer(2)]], device const float *xs [[buffer(3)]],
                       device const float *ys [[buffer(4)]], device const float *zs [[buffer(5)]],
                       device float *losses [[buffer(6)]], device float *times [[buffer(7)]],
                       constant uint &n [[buffer(8)]], constant uint &b [[buffer(9)]],
                       constant uint &nx [[buffer(10)]], constant uint &ny [[buffer(11)]],
                       constant uint &nz [[buffer(12)]], constant float &v [[buffer(13)]], uint k [[thread_position_in_grid]]) {
    uint nodes=nx*ny*nz; if(k>=nodes*b) return;
    uint e=k/nodes, node=k%nodes;
    float x=xs[node/(ny*nz)], y=ys[(node/nz)%ny], z=zs[node%nz];
    float sw=0.0f, st=0.0f;
    for(uint i=0;i<n;i++) { float w=1.0f/(sig[i]*sig[i]); sw+=w; st+=w*(a[e*n+i]-dist(x,y,z,s+3*i)/v); }
    float t=st/sw, loss=0.0f;
    for(uint i=0;i<n;i++) { float r=(t+dist(x,y,z,s+3*i)/v-a[e*n+i])/sig[i]; loss+=0.5f*r*r; }
    times[k]=t; losses[k]=loss;
}

kernel void grid_reduce(device const float *losses [[buffer(0)]], device const float *times [[buffer(1)]],
                        device const float *xs [[buffer(2)]], device const float *ys [[buffer(3)]],
                        device const float *zs [[buffer(4)]], device float *models [[buffer(5)]],
                        device float *obj [[buffer(6)]], constant uint &b [[buffer(7)]],
                        constant uint &nx [[buffer(8)]], constant uint &ny [[buffer(9)]],
                        constant uint &nz [[buffer(10)]], uint e [[thread_position_in_grid]]) {
    if(e>=b) return; uint nodes=nx*ny*nz, best=0;
    float value=losses[e*nodes];
    for(uint j=1;j<nodes;j++) if(losses[e*nodes+j]<value) {best=j; value=losses[e*nodes+j];}
    models[4*e]=xs[best/(ny*nz)]; models[4*e+1]=ys[(best/nz)%ny]; models[4*e+2]=zs[best%nz];
    models[4*e+3]=times[e*nodes+best]; obj[e]=value;
}

enum LocateStatus {
    locate_ok = 0,
    locate_invalid_geometry = 1,
    locate_singular = 2,
    locate_nonfinite = 3
};

struct CenteredSystem {
    float3 mean_direction;
    float mean_residual;
    float weight_sum;
    float scale[3];
    float upper[3][3];
    float rhs[3];
    float loss;
};

inline float safe_hypot(float a, float b) {
    float hi = max(fabs(a), fabs(b)), lo = min(fabs(a), fabs(b));
    if (hi == 0.0f) return 0.0f;
    float ratio = lo / hi;
    return hi * sqrt(1.0f + ratio * ratio);
}

// Compute u_i-u_0 without subtracting two almost identical unit directions.
// d_i-d_0 = ((p_i+p_0).(s_0-s_i))/(d_i+d_0), where p_i=m-s_i.
// This retains the small depth variation for a deep source over a shallow
// array even when the rounded absolute depth directions would be identical.
inline float3 direction_difference(thread const float m[4], device const float *s,
                                   uint i, float3 reference, float reference_distance,
                                   float3 reference_direction, thread float &distance) {
    float3 p = float3(m[0]-s[3*i], m[1]-s[3*i+1], m[2]-s[3*i+2]);
    distance = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    float3 separation = float3(s[0]-s[3*i], s[1]-s[3*i+1], s[2]-s[3*i+2]);
    float distance_difference = dot(p + reference, separation) / (distance + reference_distance);
    return (separation - reference_direction * distance_difference) / distance;
}

// Weighted centering eliminates the origin-time intercept before any solve.
// QR acts on the centered Jacobian itself, never on J^T W J, so it does not
// square its condition number or subtract nearly equal normal-matrix entries.
inline uint centered_system(thread const float m[4], device const float *s,
                            device const float *a, device const float *sig,
                            uint n, uint e, float v, thread CenteredSystem &system) {
    if (n < 4) return locate_singular;
    float3 reference = float3(m[0]-s[0], m[1]-s[1], m[2]-s[2]);
    float reference_distance = sqrt(reference.x*reference.x + reference.y*reference.y
                                    + reference.z*reference.z);
    if (!(reference_distance > 0.0f) || !isfinite(reference_distance))
        return locate_invalid_geometry;
    float3 reference_direction = reference / reference_distance;
    float4 sums = float4(0.0f), correction = float4(0.0f);
    float sw = 0.0f, weight_correction = 0.0f, loss = 0.0f;
    for (uint i=0; i<n; i++) {
        float distance;
        float3 du = direction_difference(m, s, i, reference, reference_distance,
                                          reference_direction, distance);
        if (!(distance > 0.0f) || !all(isfinite(du))) return locate_invalid_geometry;
        float predicted = m[3] + distance / v;
        float residual = predicted - a[e*n+i];
        float r = residual * v, w = 1.0f / (sig[i]*sig[i]);
        if (!isfinite(r) || !isfinite(w) || !(w > 0.0f)) return locate_nonfinite;
        // Compensated sums of reference-relative directions keep the mean
        // accurate without summing an almost constant depth column.
        float4 term = w * float4(du, r) - correction;
        float4 next = sums + term;
        correction = (next - sums) - term;
        sums = next;
        float weight_term = w - weight_correction;
        float weight_next = sw + weight_term;
        weight_correction = (weight_next - sw) - weight_term;
        sw = weight_next;
        float q = residual / sig[i];
        loss += 0.5f*q*q;
    }
    if (!all(isfinite(sums)) || !isfinite(sw) || !(sw > 0.0f) || !isfinite(loss))
        return locate_nonfinite;
    float4 mean = sums / sw;
    system.mean_direction = reference_direction + mean.xyz;
    system.mean_residual = mean.w;
    system.weight_sum = sw;
    system.loss = loss;

    // Equilibrate each centered column before QR. A small absolute depth
    // column is allowed; rank is judged relative to its own norm.
    for (uint k=0; k<3; k++) {
        system.scale[k] = 0.0f;
        system.rhs[k] = 0.0f;
        for (uint l=0; l<3; l++) system.upper[k][l] = 0.0f;
    }
    for (uint i=0; i<n; i++) {
        float distance;
        float3 du = direction_difference(m, s, i, reference, reference_distance,
                                          reference_direction, distance);
        float3 row = (du - mean.xyz) / sig[i];
        if (!all(isfinite(row))) return locate_nonfinite;
        for (uint k=0; k<3; k++) system.scale[k] = safe_hypot(system.scale[k], row[k]);
    }
    for (uint k=0; k<3; k++) {
        if (!isfinite(system.scale[k])) return locate_nonfinite;
        if (!(system.scale[k] > 0.0f)) return locate_singular;
    }

    // Streaming Givens rotations require only a 3x3 triangular factor and
    // one station row, with no fixed station-count limit or scratch buffer.
    for (uint i=0; i<n; i++) {
        float distance;
        float3 du = direction_difference(m, s, i, reference, reference_distance,
                                          reference_direction, distance);
        float3 centered = (du - mean.xyz) / sig[i];
        float row[3];
        for (uint k=0; k<3; k++) row[k] = centered[k] / system.scale[k];
        float predicted = m[3] + distance / v;
        float residual = (predicted - a[e*n+i]) * v;
        float target = -(residual - mean.w) / sig[i];
        if (!isfinite(target)) return locate_nonfinite;
        for (uint k=0; k<3; k++) {
            if (row[k] == 0.0f) continue;
            float diagonal = safe_hypot(system.upper[k][k], row[k]);
            if (!(diagonal > 0.0f) || !isfinite(diagonal)) return locate_nonfinite;
            float cosine = system.upper[k][k] / diagonal;
            float sine = row[k] / diagonal;
            system.upper[k][k] = diagonal;
            for (uint l=k+1; l<3; l++) {
                float old = system.upper[k][l];
                system.upper[k][l] = cosine*old + sine*row[l];
                row[l] = cosine*row[l] - sine*old;
            }
            float old_rhs = system.rhs[k];
            system.rhs[k] = cosine*old_rhs + sine*target;
            target = cosine*target - sine*old_rhs;
        }
    }
    // Each input column has unit norm. Reject unresolved dependence before
    // dividing by a pivot; this threshold is independent of physical units.
    const float rank_floor = 8.0f * 1.1920928955078125e-7f;
    for (uint k=0; k<3; k++) {
        if (!isfinite(system.rhs[k])) return locate_nonfinite;
        for (uint l=k; l<3; l++)
            if (!isfinite(system.upper[k][l])) return locate_nonfinite;
        if (!(system.upper[k][k] > rank_floor)) return locate_singular;
    }
    return locate_ok;
}

kernel void locate(device const float *s [[buffer(0)]], device const float *a [[buffer(1)]],
                   device const float *sig [[buffer(2)]], device const float *initial [[buffer(3)]],
                   device float *models [[buffer(4)]], device float *obj [[buffer(5)]],
                   device float *cov [[buffer(6)]], device float *ellipse [[buffer(7)]],
                   constant uint &n [[buffer(8)]], constant uint &b [[buffer(9)]],
                   constant float &v [[buffer(10)]], device uint *status [[buffer(11)]],
                   uint e [[thread_position_in_grid]]) {
    if (e>=b) return;
    status[e] = locate_ok;
    float m[4];
    for (uint k=0; k<4; k++) m[k] = initial[4*e+k];
    CenteredSystem system;
    // Retain the previous convergence rule: iterate through tiny spatial
    // corrections to resolve origin time, stopping only when unchanged or at
    // the cap that bounds float32 oscillations near the stationary point.
    for (uint iter=0; iter<64; iter++) {
        uint result = centered_system(m, s, a, sig, n, e, v, system);
        if (result != locate_ok) { status[e] = result; return; }
        float scaled_step[3], step[4];
        for (int k=2; k>=0; k--) {
            float value = system.rhs[k];
            for (uint l=uint(k)+1; l<3; l++) value -= system.upper[k][l]*scaled_step[l];
            scaled_step[k] = value / system.upper[k][k];
            step[k] = scaled_step[k] / system.scale[k];
        }
        // v*dt = -mean(r) - mean(u).dx, with residual r in metres.
        float time_step = -system.mean_residual;
        for (uint k=0; k<3; k++) time_step -= system.mean_direction[k]*step[k];
        step[3] = time_step / v;
        bool unchanged = true;
        for (uint k=0; k<4; k++) {
            float next = m[k] + step[k];
            if (!isfinite(next)) { status[e] = locate_nonfinite; return; }
            unchanged = unchanged && (next == m[k]);
            m[k] = next;
        }
        if (unchanged) break;
    }
    uint result = centered_system(m, s, a, sig, n, e, v, system);
    if (result != locate_ok) { status[e] = result; return; }

    // If A=(u-mean(u))/sigma and A*S^-1=Q*R, then
    // F=S^-1*R^-1 satisfies F*F^T=(A^T*A)^-1. Restore the time
    // intercept by a block factor, including its independent 1/sum(w) term:
    // covariance = [v*F; -mean(u)^T*F] [v*F; -mean(u)^T*F]^T
    //              + diag(0,0,0,1/sum(w)).
    // This is the full undamped known-sigma covariance in (m,m,m,s).
    float inverse[3][3] = {{0.0f}}, factor[4][3];
    for (uint l=0; l<3; l++) {
        for (int k=2; k>=0; k--) {
            float value = (uint(k) == l ? 1.0f : 0.0f);
            for (uint j=uint(k)+1; j<3; j++) value -= system.upper[k][j]*inverse[j][l];
            inverse[k][l] = value / system.upper[k][k];
        }
        float time_factor = 0.0f;
        for (uint k=0; k<3; k++) {
            float spatial_factor = inverse[k][l] / system.scale[k];
            factor[k][l] = v * spatial_factor;
            time_factor -= system.mean_direction[k]*spatial_factor;
        }
        factor[3][l] = time_factor;
    }
    for (uint k=0; k<4; k++) {
        models[4*e+k] = m[k];
        for (uint l=k; l<4; l++) {
            float value = 0.0f;
            for (uint j=0; j<3; j++) value += factor[k][j]*factor[l][j];
            if (k==3 && l==3) value += 1.0f / system.weight_sum;
            if (!isfinite(value) || (k==l && !(value > 0.0f))) {
                status[e] = locate_nonfinite; return;
            }
            cov[16*e+4*k+l] = value;
            cov[16*e+4*l+k] = value;
        }
    }
    // Horizontal block of the FULL inverse: depth and time are marginalized.
    for (uint k=0; k<2; k++) for (uint l=0; l<2; l++) {
        float value = 5.991464547107979f * cov[16*e+4*k+l];
        if (!isfinite(value)) { status[e] = locate_nonfinite; return; }
        ellipse[4*e+2*k+l] = value;
    }
    obj[e] = system.loss;
}
