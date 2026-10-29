#include <metal_stdlib>
using namespace metal;

// Linear u = (eta, q); nonlinear u = (e = h - H_ref, q). The nonlinear
// reference is spatially and temporally constant. Removing g*H_ref^2/2
// analytically leaves the same flux divergence without deep-water
// cancellation. In particular, differences in h use e_r - e_l directly.
inline float2 flux(float2 u, uint mode, float gh, float reference_depth) {
    if (mode == 0) return float2(u.y, gh * u.x);
    float h = reference_depth + u.x;
    return float2(u.y, u.y * u.y / h + 9.81f * u.x * (reference_depth + 0.5f * u.x));
}
inline float speed(float2 u, float reference_depth) {
    float h = reference_depth + u.x;
    return fabs(u.y / h) + sqrt(9.81f * h);
}
inline float2 face(float2 l, float2 r, uint mode, float gh, float reference_depth) {
    float a = mode == 0 ? sqrt(gh) : max(speed(l, reference_depth), speed(r, reference_depth));
    return 0.5f * (flux(l, mode, gh, reference_depth) + flux(r, mode, gh, reference_depth) - a * (r - l));
}
// For Phi = (F(l)+F(r)-a*(r-l))/2, the two Jacobians are
// D_l Phi = (DF(l)+a I-(r-l) outer grad_l(a))/2 and
// D_r Phi = (DF(r)-a I-(r-l) outer grad_r(a))/2.
// Apply their transposes, including the derivative of the selected max speed.
inline float2 face_vjp(float2 l, float2 r, float2 v, bool left, uint mode, float gh, float reference_depth) {
    float2 u = left ? l : r;
    float a = mode == 0 ? sqrt(gh) : max(speed(l, reference_depth), speed(r, reference_depth));
    float2 jf;
    if (mode == 0) {
        jf = float2(gh * v.y, v.x);
    } else {
        float h = reference_depth + u.x;
        float vel = u.y / h;
        // d/de of g*e*(H_ref+e/2) is g*(H_ref+e) = g*h.
        jf = float2((9.81f * h - vel * vel) * v.y, v.x + 2.0f * vel * v.y);
    }
    float2 result = 0.5f * (jf + (left ? a : -a) * v);
    if (mode != 0) {
        float sl = speed(l, reference_depth), sr = speed(r, reference_depth);
        bool selected = left ? sl >= sr : sr > sl;
        if (selected) {
            float h = reference_depth + u.x;
            float vel = u.y / h;
            float signv = vel > 0.0f ? 1.0f : (vel < 0.0f ? -1.0f : 0.0f);
            // a(h,q) = |q/h| + sqrt(g*h), with q held fixed for d/dh:
            // da/dh = -|q/h|/h + g/(2*sqrt(g*h)), da/dq = sign(q/h)/h.
            float2 da = float2(-signv * vel / h + 0.5f * 9.81f / sqrt(9.81f * h), signv / h);
            result -= 0.5f * dot(v, r - l) * da;
        }
    }
    return result;
}
// A cell-local in-place translation is safe: no thread reads a neighbor.
// Use -H_ref once on input and +H_ref once on propagate's output.
kernel void shift_depth(device float2 *state [[buffer(0)]], constant uint &n [[buffer(1)]],
                        constant float &offset [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    state[i].x += offset;
}
kernel void advance(device const float2 *src [[buffer(0)]], device float2 *dst [[buffer(1)]],
                    constant uint &n [[buffer(2)]], constant float &ratio [[buffer(3)]],
                    constant uint &mode [[buffer(4)]], constant float &gh [[buffer(5)]],
                    constant float &reference_depth [[buffer(6)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint prev = i == 0 ? n - 1 : i - 1, next = i + 1 == n ? 0 : i + 1;
    dst[i] = src[i] - ratio * (face(src[i], src[next], mode, gh, reference_depth)
                           - face(src[prev], src[i], mode, gh, reference_depth));
}
kernel void adjoint(device const float2 *state [[buffer(0)]], device const float2 *lambda [[buffer(1)]],
                    device float2 *output [[buffer(2)]], constant uint &n [[buffer(3)]],
                    constant float &ratio [[buffer(4)]], constant uint &mode [[buffer(5)]],
                    constant float &gh [[buffer(6)]], constant float &reference_depth [[buffer(7)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint prev = i == 0 ? n - 1 : i - 1, next = i + 1 == n ? 0 : i + 1;
    float2 right_v = ratio * (lambda[next] - lambda[i]);
    float2 left_v = ratio * (lambda[i] - lambda[prev]);
    // Each face enters its left cell with -ratio and right cell with +ratio.
    // Gather both contributions into one output cell to avoid atomic updates.
    output[i] = lambda[i] + face_vjp(state[i], state[next], right_v, true, mode, gh, reference_depth)
                          + face_vjp(state[prev], state[i], left_v, false, mode, gh, reference_depth);
}
kernel void objective_terms(device const float2 *state [[buffer(0)]],
                            device const float2 *weights [[buffer(1)]],
                            device float *terms [[buffer(2)]], constant uint &n [[buffer(3)]],
                            constant float &reference_depth [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float2 total = state[i];
    total.x += reference_depth;
    terms[i] = dot(total, weights[i]);
}

kernel void reduce_sum(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                       constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint j = 2 * i;
    if (j >= n) return;
    dst[i] = src[j] + (j + 1 < n ? src[j + 1] : 0.0f);
}
