//! `rmsnorm`, `add_norm` and `qk_rope` in MSL.

use super::{PRELUDE, header};

/// The small per-row kernels in MSL, after llama.cpp's `kernel_rms_norm` (MIT, the ggml
/// authors): float4 loads, sums by `simd_sum` and a fixed-order pass over the simdgroups'
/// partials, two threadgroup barriers per sum instead of the WGSL kernels' tree. Same
/// bindings and immediates as `RMSNORM`, `ADD_NORM` and `QK_ROPE` (whose fourth immediate,
/// unused there, gives the rope table's length here: MSL has no `arrayLength`).
fn kernel(body: &str, size: u32) -> String {
    let mut s = header(size);
    s.push_str(PRELUDE);
    s.push_str(SUM);
    s.push_str(body);
    s
}

/// `rmsnorm` in MSL (see [`kernel`]).
pub(crate) fn rmsnorm() -> String {
    kernel(RMSNORM, 256)
}

/// `add_norm` in MSL.
pub(crate) fn add_norm() -> String {
    kernel(ADD_NORM, 256)
}

/// `qk_rope` in MSL.
pub(crate) fn qk_rope() -> String {
    kernel(QK_ROPE, 128)
}

const SUM: &str = "
// The sum of `v` over the threadgroup's `n` simdgroups (`red` holds 32 floats), the
// simdgroups' partials added in order. Every thread gets the result.
static inline float tg_sum(float v, threadgroup float* red, uint sg, uint lane, uint n) {
    const float s = simd_sum(v);
    if (lane == 0u) { red[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint i = 0; i < n; i++) { t += red[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}
";

const RMSNORM: &str = "
struct PN { uint cols; float offset; float eps; uint pad; };
kernel void k0(constant PN& p [[buffer(0)]],
               device float* y [[buffer(1)]],
               device const float* x [[buffer(2)]],
               device const float* w [[buffer(3)]],
               uint3 wg [[threadgroup_position_in_grid]],
               uint li [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const uint c4 = p.cols / 4u;
    device const float4* x4 = (device const float4*)(x + wg.x * p.cols);
    device float4* y4 = (device float4*)(y + wg.x * p.cols);
    device const float4* w4 = (device const float4*)w;
    float s = 0.0f;
    for (uint i = li; i < c4; i += 256u) { const float4 v = x4[i]; s += dot(v, v); }
    const float r = 1.0f / sqrt(tg_sum(s, red, sg, lane, 8u) / float(p.cols) + p.eps);
    for (uint i = li; i < c4; i += 256u) { y4[i] = (p.offset + w4[i]) * (x4[i] * r); }
}
";

const ADD_NORM: &str = "
struct PA { uint cols; float o1; float o2; float eps; };
kernel void k0(constant PA& p [[buffer(0)]],
               device float* h [[buffer(1)]],
               device float* x [[buffer(2)]],
               device const float* d [[buffer(3)]],
               device const float* w1 [[buffer(4)]],
               device const float* w2 [[buffer(5)]],
               uint3 wg [[threadgroup_position_in_grid]],
               uint li [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const uint c4 = p.cols / 4u;
    device float4* x4 = (device float4*)(x + wg.x * p.cols);
    device float4* h4 = (device float4*)(h + wg.x * p.cols);
    device const float4* d4 = (device const float4*)(d + wg.x * p.cols);
    device const float4* a4 = (device const float4*)w1;
    device const float4* b4 = (device const float4*)w2;
    // The row in registers when it fits (4 float4 a thread: 4096 columns), else reread.
    float4 dv[4];
    float4 xv[4];
    float s = 0.0f;
    for (uint e = 0; e < 4u; e++) {
        const uint i = li + 256u * e;
        if (i < c4) { dv[e] = d4[i]; xv[e] = x4[i]; s += dot(dv[e], dv[e]); }
    }
    for (uint i = li + 1024u; i < c4; i += 256u) { const float4 v = d4[i]; s += dot(v, v); }
    const float r1 = 1.0f / sqrt(tg_sum(s, red, sg, lane, 8u) / float(p.cols) + p.eps);
    float s2 = 0.0f;
    for (uint e = 0; e < 4u; e++) {
        const uint i = li + 256u * e;
        if (i < c4) {
            const float4 v = xv[e] + (p.o1 + a4[i]) * (dv[e] * r1);
            xv[e] = v;
            x4[i] = v;
            s2 += dot(v, v);
        }
    }
    for (uint i = li + 1024u; i < c4; i += 256u) {
        const float4 v = x4[i] + (p.o1 + a4[i]) * (d4[i] * r1);
        x4[i] = v;
        s2 += dot(v, v);
    }
    const float r2 = 1.0f / sqrt(tg_sum(s2, red, sg, lane, 8u) / float(p.cols) + p.eps);
    for (uint e = 0; e < 4u; e++) {
        const uint i = li + 256u * e;
        if (i < c4) { h4[i] = (p.o2 + b4[i]) * (xv[e] * r2); }
    }
    for (uint i = li + 1024u; i < c4; i += 256u) { h4[i] = (p.o2 + b4[i]) * (x4[i] * r2); }
}
";

const QK_ROPE: &str = "
struct PR { uint hq; uint hkv; float eps; uint ntab; };
// Per-head RMSNorm (offset 1) and half-split RoPE of q and k, q to `qo`, k and v to the
// row's KV slot; one threadgroup per (head, row), a lane per dimension.
kernel void k0(constant PR& p [[buffer(0)]],
               device float* qo [[buffer(1)]],
               device float* kc [[buffer(2)]],
               device float* vc [[buffer(3)]],
               device const float* qkv [[buffer(4)]],
               device const float* qn [[buffer(5)]],
               device const float* kn [[buffer(6)]],
               device const float* cs [[buffer(7)]],
               device const uint4* rmeta [[buffer(8)]],
               uint3 wg [[threadgroup_position_in_grid]],
               uint li [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    threadgroup float nv[128];
    const uint h = wg.x;
    const uint t = wg.y;
    const uint qd = p.hq * 128u;
    const uint kvd = p.hkv * 128u;
    const uint row = t * (qd + 2u * kvd);
    const bool is_q = h < p.hq;
    const uint src = is_q ? row + h * 128u : row + qd + (h - p.hq) * 128u;
    const float val = qkv[src + li];
    const float r = 1.0f / sqrt(tg_sum(val * val, red, sg, lane, 4u) / 128.0f + p.eps);
    const float g = is_q ? qn[li] : kn[li];
    nv[li] = (1.0f + g) * (val * r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint4 m = rmeta[t];
    const uint pos = m.x;
    const uint j = li % 64u;
    const float c = cs[pos * 64u + j];
    const float s = cs[p.ntab + pos * 64u + j];
    const float o = li < 64u ? nv[li] * c - nv[li + 64u] * s : nv[li] * c + nv[li - 64u] * s;
    if (is_q) {
        qo[t * qd + h * 128u + li] = o;
    } else {
        const uint kh = h - p.hq;
        const uint at = m.y * kvd + kh * 128u + li;
        kc[at] = o;
        vc[at] = qkv[row + qd + kvd + kh * 128u + li];
    }
}
";
