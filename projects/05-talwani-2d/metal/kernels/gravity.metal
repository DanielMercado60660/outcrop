#include <metal_stdlib>
using namespace metal;

// Metal does not provide log1p. The polynomial preserves small increments
// that would otherwise disappear when added to one in float32.
static float log_one_plus(float x) {
    if (abs(x) < 0.01f) {
        return x * (1.0f + x * (-0.5f + x * (0.333333333333f + x * (-0.25f + x * 0.2f))));
    }
    return log(1.0f + x);
}

static float relative_log_radius(float2 p, float2 reference, float reference_r2) {
    float delta = dot(p - reference, p + reference) / reference_r2;
    return 0.5f * (delta < -0.5f ? log(dot(p, p) / reference_r2) : log_one_plus(delta));
}

// The area integral of z/r^2 is the boundary integral of log(r) n_z.
// A signed angle per segment works for concave polygons and either side.
kernel void polygon_gravity(device const float2 *vertices [[buffer(0)]],
                            device const float2 *observations [[buffer(1)]],
                            device float *gz [[buffer(2)]],
                            constant uint &n [[buffer(3)]],
                            constant uint &m [[buffer(4)]],
                            constant float &scale [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= m) return;
    float2 origin = observations[i];
    float2 reference = vertices[0] - origin;
    float reference_r2 = dot(reference, reference);
    // At a vertex the field is finite, but log(0) cannot serve as the
    // subtracted constant. A simple polygon has another nonzero radius.
    if (reference_r2 == 0.0f) {
        for (uint j = 1; j < n; ++j) {
            reference = vertices[j] - origin;
            reference_r2 = dot(reference, reference);
            if (reference_r2 > 0.0f) break;
        }
    }
    float sum = 0.0f;
    float compensation = 0.0f;
    for (uint j = 0; j < n; ++j) {
        float2 a = vertices[j] - origin;
        float2 b = vertices[(j + 1 == n) ? 0 : j + 1] - origin;
        float2 d = vertices[(j + 1 == n) ? 0 : j + 1] - vertices[j];
        float d2 = dot(d, d);
        if (d2 == 0.0f) continue;
        float r2 = dot(a, a);
        float b2 = dot(b, b);
        float ad = dot(a, d);
        float cross = a.x * d.y - a.y * d.x;
        // Subtract a constant log radius, whose closed boundary integral is zero.
        float average_log;
        if (r2 == 0.0f || b2 == 0.0f) {
            // With one endpoint at the observation, r(t) = t * edge_length
            // (or (1-t) * edge_length). Since integral_0^1 log(t) dt = -1,
            // use the finite limit without forming log(0) or 0 * infinity.
            float2 nonzero_endpoint = (r2 == 0.0f) ? b : a;
            average_log = relative_log_radius(nonzero_endpoint, reference, reference_r2) - 1.0f;
        } else if (d2 < 0.01f * min(r2, b2)) {
            // Three-point Gauss-Legendre integration avoids cancellation in
            // the analytic expression when an edge is short relative to r.
            float2 mid = a + 0.5f * d;
            float2 offset = 0.387298334621f * d;
            average_log = (5.0f / 18.0f) * relative_log_radius(mid - offset, reference, reference_r2)
                        + (4.0f / 9.0f) * relative_log_radius(mid, reference, reference_r2)
                        + (5.0f / 18.0f) * relative_log_radius(mid + offset, reference, reference_r2);
        } else {
            float angle = atan2(cross, dot(a, b));
            float log_ref = relative_log_radius(a, reference, reference_r2);
            float delta = (2.0f * ad + d2) / r2;
            float log_ratio = 0.5f * (delta < -0.5f ? log(dot(b, b) / r2) : log_one_plus(delta));
            average_log = log_ref + ((ad + d2) / d2) * log_ratio
                        - 1.0f + (cross / d2) * angle;
        }
        float term = -d.x * average_log;
        float y = term - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    gz[i] = scale * sum;
}
