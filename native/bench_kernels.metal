#include <metal_stdlib>
using namespace metal;

constant uint DIM = 6;
constant uint E = 36;

struct RNSCfg {
    uint nOps;
    uint K;
    uint batch;
};

struct MontPrime {
    uint p;
    uint pinv;  // -p^{-1} mod 2^32
    uint rmod;  // 2^32 mod p
    uint pad;
};

inline uint add_mod(uint a, uint b, uint p) {
    ulong s = (ulong)a + (ulong)b;
    return (uint)(s >= (ulong)p ? s - (ulong)p : s);
}

inline uint mont_mul(uint a, uint b, MontPrime pm) {
    ulong t = (ulong)a * (ulong)b;
    uint m = (uint)t * pm.pinv;  // wraps modulo 2^32
    ulong u = (t + (ulong)m * (ulong)pm.p) >> 32;
    uint r = (uint)u;
    return r >= pm.p ? r - pm.p : r;
}

// One thread owns one (prime, repeated trajectory) lane.
// All matrix values stay in Montgomery form through the chain, removing
// integer division/modulo from the hot multiplication loop.
kernel void rns_chain6(
    device const uint *ops [[buffer(0)]],          // Montgomery [K][nOps][36]
    device const MontPrime *meta [[buffer(1)]],   // [K]
    constant RNSCfg &cfg [[buffer(2)]],
    device uint *out [[buffer(3)]],                // canonical residues [K][batch][36]
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)cfg.K * cfg.batch;
    if ((ulong)gid >= total) return;
    uint b = gid % cfg.batch;
    uint k = gid / cfg.batch;
    MontPrime pm = meta[k];

    uint P[E], C[E];
    for (uint i = 0; i < DIM; ++i)
        for (uint j = 0; j < DIM; ++j)
            P[i * DIM + j] = (i == j) ? pm.rmod : 0u; // 1 in Montgomery form

    device const uint *base = ops + (ulong)k * cfg.nOps * E;
    for (uint t = 0; t < cfg.nOps; ++t) {
        device const uint *A = base + (ulong)t * E;
        for (uint i = 0; i < DIM; ++i) {
            for (uint j = 0; j < DIM; ++j) {
                uint acc = 0u;
                for (uint q = 0; q < DIM; ++q)
                    acc = add_mod(acc, mont_mul(P[i * DIM + q], A[q * DIM + j], pm), pm.p);
                C[i * DIM + j] = acc;
            }
        }
        for (uint e = 0; e < E; ++e) P[e] = C[e];
    }

    ulong ob = ((ulong)k * cfg.batch + b) * E;
    for (uint e = 0; e < E; ++e)
        out[ob + e] = mont_mul(P[e], 1u, pm); // leave Montgomery domain
}

struct DF64Cfg {
    uint nOps;
    uint batch;
};

inline float2 df(float x) { return float2(x, 0.0f); }

inline float2 df_renorm(float s, float e) {
    float hi = s + e;
    float lo = e - (hi - s);
    return float2(hi, lo);
}

inline float2 df_add(float2 a, float2 b) {
    float s = a.x + b.x;
    float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v);
    float t = a.y + b.y;
    float w = t - a.y;
    float f = (a.y - (t - w)) + (b.y - w);
    e += t;
    float hi = s + e;
    float lo = e - (hi - s);
    lo += f;
    return df_renorm(hi, lo);
}

inline float2 df_mul(float2 a, float2 b) {
    float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e = fma(a.x, b.y, e);
    e = fma(a.y, b.x, e);
    e = fma(a.y, b.y, e);
    return df_renorm(p, e);
}

inline void normalize_pow2(thread float2 *W, thread int &expSum) {
    float mx = 0.0f;
    for (uint e = 0; e < E; ++e) mx = max(mx, fabs(W[e].x));
    if (mx > 0.0f && isfinite(mx)) {
        int ex;
        frexp(mx, ex);
        float scale = ldexp(1.0f, -ex);
        for (uint e = 0; e < E; ++e) {
            W[e].x *= scale;
            W[e].y *= scale;
        }
        expSum += ex;
    }
}

kernel void df64_chain6(
    device const float2 *ops [[buffer(0)]],    // [nOps][36]
    constant DF64Cfg &cfg [[buffer(1)]],
    device float2 *out [[buffer(2)]],          // [batch][36]
    device int *outExp [[buffer(3)]],          // [batch]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.batch) return;
    float2 P[E], C[E];
    for (uint i = 0; i < DIM; ++i)
        for (uint j = 0; j < DIM; ++j)
            P[i * DIM + j] = df(i == j ? 1.0f : 0.0f);
    int expSum = 0;

    for (uint t = 0; t < cfg.nOps; ++t) {
        device const float2 *A = ops + (ulong)t * E;
        for (uint i = 0; i < DIM; ++i) {
            for (uint j = 0; j < DIM; ++j) {
                float2 acc = df(0.0f);
                for (uint q = 0; q < DIM; ++q)
                    acc = df_add(acc, df_mul(P[i * DIM + q], A[q * DIM + j]));
                C[i * DIM + j] = acc;
            }
        }
        normalize_pow2(C, expSum);
        for (uint e = 0; e < E; ++e) P[e] = C[e];
    }

    ulong ob = (ulong)gid * E;
    for (uint e = 0; e < E; ++e) out[ob + e] = P[e];
    outExp[gid] = expSum;
}
