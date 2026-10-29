#include <metal_stdlib>
using namespace metal;

// Preserve the rounding residuals in compensated source summation, including
// when the runtime compiler's default math mode permits reassociation.
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)

inline void accumulate(float4 value, thread float4 &sum, thread float4 &residual) {
    float4 next = sum + value;
    // Neumaier summation also handles a new term larger than the running sum.
    residual += select((value - next) + sum, (sum - next) + value,
                       abs(sum) >= abs(value));
    sum = next;
}

inline float4 source_field(float3 position, float3 center, float mass, float radius) {
    float3 delta = position - center;
    int delta_shift = 0;
    if (!all(isfinite(delta))) {
        // Finite coordinates can have a difference larger than FLT_MAX.
        // Carry that factor of two in the exponent instead of the vector.
        delta = 0.5f * position - 0.5f * center;
        delta_shift = 1;
    }
    float largest = max(abs(delta.x), max(abs(delta.y), abs(delta.z)));

    // Classify without squaring dimensional lengths. A component at least R
    // guarantees an exterior point; otherwise delta/R is bounded by one.
    // An overflowed difference is necessarily outside any finite radius.
    bool interior = false;
    float scaled_r2 = 0.0f;
    if (delta_shift == 0 && radius > 0.0f && largest < radius) {
        float3 q = precise::divide(delta, float3(radius));
        scaled_r2 = fma(q.x, q.x, fma(q.y, q.y, q.z * q.z));
        interior = scaled_r2 < 1.0f;
    }

    // L = R inside a sphere, r outside. Store L as mantissa * 2^exponent:
    // the exterior norm squares only bounded components, never metres.
    int length_exponent;
    float length_mantissa;
    if (interior) {
        length_mantissa = frexp(radius, length_exponent);
    } else {
        // A point-mass collision is excluded by the API. Sphere centers take
        // the interior branch, so largest is positive here for valid inputs.
        float3 q = precise::divide(delta, float3(largest));
        float norm2 = fma(q.x, q.x, fma(q.y, q.y, q.z * q.z));
        length_mantissa = frexp(largest, length_exponent) * precise::sqrt(norm2);
        length_exponent += delta_shift;
    }

    // Split the mass as well: G*M can underflow even when the final field
    // at a small distance is representable. All mantissa arithmetic below
    // stays in range; ldexp applies the physical scale only to final terms.
    int mass_exponent;
    float gm_mantissa = 6.67430e-11f * frexp(mass, mass_exponent);
    float phi_mantissa = -precise::divide(gm_mantissa, length_mantissa);
    if (interior) {
        // phi = -GM/(2R) * (3 - r^2/R^2), including r = 0.
        phi_mantissa *= fma(-0.5f, scaled_r2, 1.5f);
    }
    float phi = ldexp(phi_mantissa, mass_exponent - length_exponent);

    // a_k = -GM*delta_k/L^3. Cube only L's bounded mantissa, and include
    // delta_k's exponent before rescaling. No tiny inverse-cube factor is
    // ever formed, and small components survive even when delta_k/L would
    // underflow. Zero components (including sphere centers) remain zero.
    int3 delta_exponent;
    float3 delta_mantissa = frexp(delta, delta_exponent);
    float length_cube = (length_mantissa * length_mantissa) * length_mantissa;
    float3 a_mantissa = precise::divide(-gm_mantissa * delta_mantissa, float3(length_cube));
    int3 a_exponent = delta_exponent + int3(mass_exponent + delta_shift - 3 * length_exponent);
    return float4(ldexp(a_mantissa, a_exponent), phi);
}

kernel void gravity(device const float *coordinates [[buffer(0)]],
                    device const float *centers [[buffer(1)]],
                    device const float *masses [[buffer(2)]],
                    device const float *radii [[buffer(3)]],
                    device float *potential [[buffer(4)]],
                    device float *acceleration [[buffer(5)]],
                    constant uint &count [[buffer(6)]],
                    constant uint &sources [[buffer(7)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    // Widen before multiplication: counts fit uint, but three scalars per
    // vector can exceed its range. ulong is the 64-bit Metal integer type.
    ulong observation_offset = 3ul * ulong(i);
    float3 position = float3(coordinates[observation_offset],
                             coordinates[observation_offset + 1],
                             coordinates[observation_offset + 2]);
    // xyz contains acceleration, w contains gravitational potential.
    float4 field = float4(0.0f);
    float4 residual = float4(0.0f);
    for (uint j = 0; j < sources; ++j) {
        ulong source_offset = 3ul * ulong(j);
        float3 center = float3(centers[source_offset], centers[source_offset + 1], centers[source_offset + 2]);
        accumulate(source_field(position, center, masses[j], radii[j]), field, residual);
    }
    field += residual;
    potential[i] = field.w;
    acceleration[observation_offset] = field.x;
    acceleration[observation_offset + 1] = field.y;
    acceleration[observation_offset + 2] = field.z;
}
