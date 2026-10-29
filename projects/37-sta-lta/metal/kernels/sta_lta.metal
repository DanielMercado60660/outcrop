#include <metal_stdlib>
using namespace metal;

// Energy = mantissa * 2^exponent. The exponent is the largest sample's
// squared-amplitude exponent, not an exponent stored in a float. A nonzero
// mantissa lies in [1, 4 * number_of_samples), so neither squaring nor summing
// can overflow/underflow even for FLT_MAX or the smallest float subnormal.
struct Energy {
    float mantissa;
    int exponent;
};

inline Energy add_energy(Energy a, Energy b) {
    if (a.mantissa == 0.0f) return b;
    if (b.mantissa == 0.0f) return a;
    if (a.exponent < b.exponent) {
        Energy tmp = a;
        a = b;
        b = tmp;
    }
    a.mantissa += ldexp(b.mantissa, b.exponent - a.exponent);
    return a;
}

// Read the float's bits, rather than doing floating-point operations on an
// input subnormal (which a GPU may flush to zero). Normalize before squaring.
kernel void energy_leaves(device const uint *trace_bits [[buffer(0)]],
                          device Energy *tree [[buffer(1)]],
                          constant uint &n [[buffer(2)]],
                          constant uint &leaves [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= leaves) return;
    Energy result = {0.0f, 0};
    uint bits = i < n ? trace_bits[i] & 0x7fffffffu : 0u;
    if (bits != 0u) {
        uint fraction = bits & 0x007fffffu;
        int exponent = int(bits >> 23) - 127;
        if ((bits >> 23) == 0u) {
            uint top = 31u - clz(fraction);
            fraction = (fraction << (23u - top)) & 0x007fffffu;
            exponent = int(top) - 149;
        }
        float normalized = as_type<float>(0x3f800000u | fraction);
        result.mantissa = normalized * normalized;
        result.exponent = 2 * exponent;
    }
    tree[leaves + i] = result;
}

// Each launch builds one level of a balanced tree, O(N) total build work.
kernel void energy_reduce(device Energy *tree [[buffer(0)]],
                          constant uint &first [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= first) return;
    uint node = first + i;
    tree[node] = add_energy(tree[2u * node], tree[2u * node + 1u]);
}

// Combine only disjoint, positive subtree sums. There is no subtraction of
// nearly equal prefixes and no rolling residual after a large sample exits.
// A window visits O(1 + log(width)) nodes instead of rescanning width samples.
inline Energy window_energy(device const Energy *tree, uint leaves,
                            uint end, uint width) {
    uint left = leaves + end + 1u - width;
    uint right = leaves + end + 1u;
    Energy sum = {0.0f, 0};
    while (left < right) {
        if (left & 1u) sum = add_energy(sum, tree[left++]);
        if (right & 1u) sum = add_energy(sum, tree[--right]);
        left >>= 1;
        right >>= 1;
    }
    return sum;
}

kernel void classic_sta_lta(device const Energy *tree [[buffer(0)]],
                            device float *characteristic [[buffer(1)]],
                            constant uint &n [[buffer(2)]],
                            constant uint &nsta [[buffer(3)]],
                            constant uint &nlta [[buffer(4)]],
                            constant uint &leaves [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    if (i < nlta - 1u) {
        characteristic[i] = 0.0f;
        return;
    }
    Energy sta = window_energy(tree, leaves, i, nsta);
    if (sta.mantissa == 0.0f) {
        characteristic[i] = 0.0f;
        return;
    }
    Energy lta = window_energy(tree, leaves, i, nlta);
    // Normalize the dimensionless ratio before applying its exponent: dividing
    // the means separately could underflow a valid small STA mean. The LTA is
    // positive by contract, and the result is bounded above by nlta / nsta.
    float ratio = (sta.mantissa / lta.mantissa) * (float(nlta) / float(nsta));
    characteristic[i] = ldexp(ratio, sta.exponent - lta.exponent);
}
