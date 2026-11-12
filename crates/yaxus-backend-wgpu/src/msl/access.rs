//! The weights of each format as the MSL kernels read them.

use crate::Format;

/// Per-format access to the weights in the decode layout (the values of the WGSL `W_*_16`
/// loaders): `load8(row, k)` decodes weights `k .. k + 8` of `row` (`k` a multiple of 8)
/// with one or two loads.
pub(super) fn access(f: Format) -> Option<&'static str> {
    Some(match f {
        Format::Q8 => {
            "// Scales of a row: `blocks` f16 values, padded when `blocks` is a multiple of 256.
static inline uint sstride(uint blocks) { return blocks + ((blocks % 256u == 0u) ? SPAD : 0u); }
static inline void load8(device const uint* wq, device const uint* wsc, uint k, uint row, uint kk, thread float* w) {
    const float s = float(((device const half*)wsc)[row * sstride(k / 32u) + kk / 32u]);
    const uint2 u = *(device const uint2*)(wq + row * (k / 4u) + kk / 4u);
    const char4 a = as_type<char4>(u.x);
    const char4 b = as_type<char4>(u.y);
    w[0] = float(a.x) * s; w[1] = float(a.y) * s; w[2] = float(a.z) * s; w[3] = float(a.w) * s;
    w[4] = float(b.x) * s; w[5] = float(b.y) * s; w[6] = float(b.z) * s; w[7] = float(b.w) * s;
}
// The 32 values of chunk c, as `load8` gives them, in wide reads.
static inline void load32(device const uint* wq, device const uint* wsc, uint k, uint row, uint c, thread float* w) {
    const float s = float(((device const half*)wsc)[row * sstride(k / 32u) + c]);
    device const uint4* v = (device const uint4*)(wq + row * (k / 4u) + 8u * c);
    const uint4 u0 = v[0];
    const uint4 u1 = v[1];
    const uint us[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
    for (uint i = 0; i < 8; i++) {
        const char4 e = as_type<char4>(us[i]);
        w[4 * i] = float(e.x) * s; w[4 * i + 1] = float(e.y) * s; w[4 * i + 2] = float(e.z) * s; w[4 * i + 3] = float(e.w) * s;
    }
}
"
        }
        Format::Q4K => {
            "// GGUF Q4_K blocks: 36 u32 per 256 values (d and dmin as f16, 12 bytes of packed 6-bit
// scales and mins, then 32 u32 of nibbles in order); `wsc` is the same buffer.
static inline uint hbyte(device const uint* b, uint m, uint i) { return (b[m + 1u + i / 4u] >> (8u * (i % 4u))) & 255u; }
static inline void load8(device const uint* wq, device const uint* wsc, uint k, uint row, uint kk, thread float* w) {
    const uint sub = kk / 32u;
    const uint m = (row * (k / 256u) + sub / 8u) * 36u;
    const half2 dd = as_type<half2>(wsc[m]);
    const uint jj = sub % 8u;
    const bool lo = jj < 4u;
    const uint a = hbyte(wsc, m, lo ? jj : jj - 4u);
    const uint b = hbyte(wsc, m, jj + 4u);
    const uint cc = hbyte(wsc, m, jj);
    const uint sc = lo ? (a & 63u) : ((b & 15u) | ((a >> 6u) << 4u));
    const uint mn = lo ? (b & 63u) : ((b >> 4u) | ((cc >> 6u) << 4u));
    const float d = float(dd.x) * float(sc);
    const float dm = float(dd.y) * float(mn);
    const uint u = wq[m + 4u + (kk % 256u) / 8u];
    for (uint i = 0; i < 8; i++) { w[i] = fma(float((u >> (4u * i)) & 15u), d, -dm); }
}
// The 32 values of chunk c, as `load8` gives them, in wide reads.
static inline void load32(device const uint* wq, device const uint* wsc, uint k, uint row, uint c, thread float* w) {
    const uint m = (row * (k / 256u) + c / 8u) * 36u;
    const uint4 h = *(device const uint4*)(wsc + m);
    const uint4 u = *(device const uint4*)(wq + m + 4u + 4u * (c % 8u));
    const half2 dd = as_type<half2>(h.x);
    const uint hs[3] = { h.y, h.z, h.w };
    const uint jj = c % 8u;
    const bool lo = jj < 4u;
    const uint ia = lo ? jj : jj - 4u;
    const uint a = (hs[ia / 4u] >> (8u * (ia % 4u))) & 255u;
    const uint b = (hs[(jj + 4u) / 4u] >> (8u * ((jj + 4u) % 4u))) & 255u;
    const uint cc = (hs[jj / 4u] >> (8u * (jj % 4u))) & 255u;
    const uint sc = lo ? (a & 63u) : ((b & 15u) | ((a >> 6u) << 4u));
    const uint mn = lo ? (b & 63u) : ((b >> 4u) | ((cc >> 6u) << 4u));
    const float d = float(dd.x) * float(sc);
    const float dm = float(dd.y) * float(mn);
    const uint us[4] = { u.x, u.y, u.z, u.w };
    for (uint i = 0; i < 4; i++) {
        for (uint e = 0; e < 8; e++) { w[8 * i + e] = fma(float((us[i] >> (4u * e)) & 15u), d, -dm); }
    }
}
"
        }
        Format::Q6 => {
            "// 6 u32 per 32 values (4 of low nibbles, 2 of high 2-bit fields); per row K / 256 f16
// `d`, then K / 16 int8 scales (`q6_meta_stride` on the host).
static inline float q6scale(device const uint* wsc, uint k, uint row, uint blk) {
    const uint hb = k / 256u;
    uint rs = (18u * hb + 3u) / 4u;
    rs += ((rs * 4u) % 512u == 0u) ? 4u : 0u;
    const uint base = row * rs;
    const half2 d2 = as_type<half2>(wsc[base + blk / 32u]);
    const float d = ((blk / 16u) & 1u) ? float(d2.y) : float(d2.x);
    const uint off = 2u * hb + blk;
    const char sc = as_type<char4>(wsc[base + off / 4u])[off % 4u];
    return d * float(sc);
}
static inline void load8(device const uint* wq, device const uint* wsc, uint k, uint row, uint kk, thread float* w) {
    const uint j = kk / 16u;
    const uint h8 = (kk % 16u) / 8u;  // which half of the 16
    const uint g = row * (3u * k / 16u) + (j / 2u) * 6u;
    const uint l = wq[g + 2u * (j & 1u) + h8];
    const uint h = wq[g + 4u + (j & 1u)] >> (16u * h8);
    const float s = q6scale(wsc, k, row, j);
    for (uint q = 0; q < 2; q++) {
        const uint lq = l >> (16u * q);
        const uint hq = h >> (8u * q);
        for (uint e = 0; e < 4; e++) {
            const uint v = ((lq >> (4u * e)) & 15u) | (((hq >> (2u * e)) & 3u) << 4u);
            w[4 * q + e] = (float(v) - 32.0f) * s;
        }
    }
}
// The 32 values of chunk c, as `load8` gives them, in wide reads.
static inline void load32(device const uint* wq, device const uint* wsc, uint k, uint row, uint c, thread float* w) {
    device const uint2* g = (device const uint2*)(wq + row * (3u * k / 16u) + 6u * c);
    const uint2 g0 = g[0];
    const uint2 g1 = g[1];
    const uint2 g2 = g[2];
    const uint ls[4] = { g0.x, g0.y, g1.x, g1.y };
    const uint hs[2] = { g2.x, g2.y };
    const float s2[2] = { q6scale(wsc, k, row, 2u * c), q6scale(wsc, k, row, 2u * c + 1u) };
    for (uint i = 0; i < 4; i++) {
        const uint l = ls[i];
        const uint h = hs[i / 2u] >> (16u * (i % 2u));
        const float s = s2[i / 2u];
        for (uint q = 0; q < 2; q++) {
            const uint lq = l >> (16u * q);
            const uint hq = h >> (8u * q);
            for (uint e = 0; e < 4; e++) {
                const uint v = ((lq >> (4u * e)) & 15u) | (((hq >> (2u * e)) & 3u) << 4u);
                w[8 * i + 4 * q + e] = (float(v) - 32.0f) * s;
            }
        }
    }
}
"
        }
        Format::Q3 => {
            "// 3 u32 per 32 values (2 of low 2-bit fields, one per 16; 1 of high bits, 16 per
// half); scales as for Q6: per row K / 256 f16 `d`, then K / 16 int8 scales.
static inline float q3scale(device const uint* wsc, uint k, uint row, uint blk) {
    const uint hb = k / 256u;
    uint rs = (18u * hb + 3u) / 4u;
    rs += ((rs * 4u) % 512u == 0u) ? 4u : 0u;
    const uint base = row * rs;
    const half2 d2 = as_type<half2>(wsc[base + blk / 32u]);
    const float d = ((blk / 16u) & 1u) ? float(d2.y) : float(d2.x);
    const uint off = 2u * hb + blk;
    const char sc = as_type<char4>(wsc[base + off / 4u])[off % 4u];
    return d * float(sc);
}
static inline float q3val(uint lo, uint h, uint e, float s) {
    return (float(((lo >> (2u * e)) & 3u) | (((h >> e) & 1u) << 2u)) - 4.0f) * s;
}
static inline void load8(device const uint* wq, device const uint* wsc, uint k, uint row, uint kk, thread float* w) {
    const uint c = kk / 32u;
    const uint half_ = (kk % 32u) / 16u;
    const uint e0 = kk % 16u;
    const uint g = row * (3u * k / 32u) + 3u * c;
    const uint lo = wq[g + half_];
    const uint h = wq[g + 2u] >> (16u * half_);
    const float s = q3scale(wsc, k, row, 2u * c + half_);
    for (uint i = 0; i < 8; i++) { w[i] = q3val(lo, h, e0 + i, s); }
}
// The 32 values of chunk c, as `load8` gives them.
static inline void load32(device const uint* wq, device const uint* wsc, uint k, uint row, uint c, thread float* w) {
    const uint g = row * (3u * k / 32u) + 3u * c;
    const uint l0 = wq[g];
    const uint l1 = wq[g + 1u];
    const uint h = wq[g + 2u];
    const float s0 = q3scale(wsc, k, row, 2u * c);
    const float s1 = q3scale(wsc, k, row, 2u * c + 1u);
    for (uint e = 0; e < 16; e++) { w[e] = q3val(l0, h, e, s0); }
    for (uint e = 0; e < 16; e++) { w[16 + e] = q3val(l1, h >> 16u, e, s1); }
}
"
        }
        Format::Q2 => {
            "// 16 values per u32; per row K / 256 pairs of f16 (`d`, `dmin`), then K / 16 bytes of
// scale | min << 4. A weight is `q * d * scale - dmin * min`.
static inline float2 q2sm(device const uint* wsc, uint k, uint row, uint blk) {
    const uint hb = k / 256u;
    const uint base = row * 5u * hb;
    const half2 dd = as_type<half2>(wsc[base + blk / 16u]);
    const uint b = wsc[base + hb + blk / 4u] >> (8u * (blk % 4u));
    return float2(float(dd.x) * float(b & 15u), float(dd.y) * float((b >> 4u) & 15u));
}
static inline void load8(device const uint* wq, device const uint* wsc, uint k, uint row, uint kk, thread float* w) {
    const uint j = kk / 16u;
    const uint u = wq[row * (k / 16u) + j] >> (2u * (kk % 16u));
    const float2 sm = q2sm(wsc, k, row, j);
    for (uint i = 0; i < 8; i++) { w[i] = fma(float((u >> (2u * i)) & 3u), sm.x, -sm.y); }
}
// The 32 values of chunk c, as `load8` gives them, in one read.
static inline void load32(device const uint* wq, device const uint* wsc, uint k, uint row, uint c, thread float* w) {
    const uint2 u = *(device const uint2*)(wq + row * (k / 16u) + 2u * c);
    const float2 s0 = q2sm(wsc, k, row, 2u * c);
    const float2 s1 = q2sm(wsc, k, row, 2u * c + 1u);
    for (uint e = 0; e < 16; e++) { w[e] = fma(float((u.x >> (2u * e)) & 3u), s0.x, -s0.y); }
    for (uint e = 0; e < 16; e++) { w[16 + e] = fma(float((u.y >> (2u * e)) & 3u), s1.x, -s1.y); }
}
"
        }
        _ => return None,
    })
}
