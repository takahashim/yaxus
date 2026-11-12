//! Kernels written in Metal Shading Language, compiled through wgpu's passthrough on Metal
//! (`Features::PASSTHROUGH_SHADERS`). wgpu binds the immediates at `buffer(0)` and binding
//! `i` of bind group 0 at `buffer(i + 1)`, and sets no threadgroup memory length, so the
//! kernels declare their threadgroup arrays inside the function. A source starts with the
//! line `// msl workgroup_size=N` (see `kernels::compile_kernel`).
//!
//! Each source starts with `#pragma METAL fp math_mode(safe)`: Metal compiles WGSL with
//! fast math, which wgpu cannot turn off; here the compiler keeps the arithmetic as written,
//! so kernels agree to the bit without the `keep` and `fma` precautions of the WGSL ones.

use std::fmt::Write;

use crate::Format;

/// The marker line of an MSL source and its workgroup size.
pub(crate) fn header(workgroup_size: u32) -> String {
    format!("// msl workgroup_size={workgroup_size}\n")
}

/// The workgroup size of `src` if it is an MSL source.
pub(crate) fn workgroup_size(src: &str) -> Option<u32> {
    src.lines().next()?.strip_prefix("// msl workgroup_size=")?.trim().parse().ok()
}

const PRELUDE: &str = "#include <metal_stdlib>
#pragma METAL fp math_mode(safe)
using namespace metal;
struct P { uint n; uint k; uint t; uint r0; };
";

/// Per-format access to the weights in the decode layout (the values of the WGSL `W_*_16`
/// loaders): `load8(row, k)` decodes weights `k .. k + 8` of `row` (`k` a multiple of 8)
/// with one or two loads.
fn access(f: Format) -> Option<&'static str> {
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
        _ => return None,
    })
}

/// Whether [`gemv_mma`] has a kernel for `f`.
pub(crate) fn gemv_mma_reads(f: Format) -> bool {
    access(f).is_some()
}

/// Outputs (non-GLU) per workgroup of [`gemv_mma`]; with `glu`, intermediate outputs.
pub(crate) fn gemv_mma_outputs(glu: bool) -> usize {
    if glu { 8 } else { 16 }
}

/// `y = x W^T` for `t` <= 16 rows (or `silu(gate) * up` with `glu`) on 8x8 simdgroup
/// matrices, without threadgroup memory in the loop (as llama.cpp's `mul_mv_mma`): each
/// lane fills its two elements of each matrix from registers (lane `l` holds row `fm`,
/// columns `fn` and `fn + 1`). The weights are the A matrices (8 weight rows each), the
/// pass's rows the B ones, padded with zeros to 8 (`tiles` = 1) or 16 (`tiles` = 2).
///
/// K is permuted within each chunk of 32: column `c` of step `st` is value
/// `8 (c / 2) + 2 st + c % 2`, so a lane's A elements over a chunk are 8 consecutive
/// weights of its row, decoded at once (`load8`). The same permutation for A and B leaves
/// the sums as they are.
///
/// A workgroup takes 16 weight rows (with `glu` the 8 gate and the 8 up rows of the same
/// outputs), its 4 simdgroups every fourth chunk, and adds their partial products in a
/// fixed order at the end. A row's products do not depend on the others, so a row gives
/// the same bits in a pass of any size up to 16, and the weights are read once per pass.
///
/// Bindings: 0 `y [T][N]`, 1 `x [T][K]`, 2 and 3 the weights (decode layout). Immediates:
/// N (the intermediate size with `glu`), K, T, first row. Needs K % 32 == 0.
pub(crate) fn gemv_mma(f: Format, tiles: usize, glu: bool) -> Option<String> {
    assert!(tiles == 1 || tiles == 2);
    let acc = access(f)?;
    let rt = tiles;
    let mut s = header(128);
    s.push_str(PRELUDE);
    s.push_str(&acc.replace("SPAD", &format!("{}u", crate::SCALE_PAD)));
    write!(
        s,
        "
kernel void k0(constant P& p [[buffer(0)]],
               device float* y [[buffer(1)]],
               device const float* x [[buffer(2)]],
               device const uint* wq [[buffer(3)]],
               device const uint* wsc [[buffer(4)]],
               uint3 wg [[threadgroup_position_in_grid]],
               uint3 nw [[threadgroups_per_grid]],
               uint li [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {{
    threadgroup float red[4][2 * {rt} * 64];  // per simdgroup: [weight tile][row tile][8][8]
    const uint group = wg.y * nw.x + wg.x;
    // The lane's elements of an 8x8 matrix: row fm, columns fn and fn + 1.
    const uint fm = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    const uint fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
"
    )
    .unwrap();
    // The weight row of the lane's A row in each weight tile (clamped to the last row).
    if glu {
        s.push_str(
            "    const uint u0 = group * 8u;
    const uint wr[2] = { min(u0 + fm, p.n - 1u), p.n + min(u0 + fm, p.n - 1u) };
",
        );
    } else {
        s.push_str(
            "    const uint n0 = group * 16u;
    const uint wr[2] = { min(n0 + fm, p.n - 1u), min(n0 + 8u + fm, p.n - 1u) };
",
        );
    }
    write!(
        s,
        "    // The pass's rows of the lane's B columns, and whether they exist.
    uint xr[{rt}][2];
    bool xok[{rt}][2];
    for (uint r = 0; r < {rt}u; r++) {{
        for (uint e = 0; e < 2u; e++) {{
            const uint row = 8u * r + fn + e;
            xok[r][e] = row < p.t;
            xr[r][e] = (p.r0 + min(row, p.t - 1u)) * p.k;
        }}
    }}
    simdgroup_float8x8 c[2][{rt}];
    for (uint r = 0; r < {rt}u; r++) {{
        c[0][r] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        c[1][r] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }}
    for (uint ch = sg; ch < p.k / 32u; ch += 4u) {{
        // A: 8 consecutive weights of the lane's row per weight tile.
        float w0[8];
        float w1[8];
        load8(wq, wsc, p.k, wr[0], 32u * ch + 8u * (fn / 2u), w0);
        load8(wq, wsc, p.k, wr[1], 32u * ch + 8u * (fn / 2u), w1);
        // B: the value of K row fm of each step, for the lane's two pass rows.
        const uint kb = 32u * ch + 8u * (fm / 2u) + fm % 2u;
        for (uint st = 0; st < 4u; st++) {{
            simdgroup_float8x8 a0;
            simdgroup_float8x8 a1;
            a0.thread_elements()[0] = w0[2u * st];
            a0.thread_elements()[1] = w0[2u * st + 1u];
            a1.thread_elements()[0] = w1[2u * st];
            a1.thread_elements()[1] = w1[2u * st + 1u];
            for (uint r = 0; r < {rt}u; r++) {{
                simdgroup_float8x8 b;
                b.thread_elements()[0] = xok[r][0] ? x[xr[r][0] + kb + 2u * st] : 0.0f;
                b.thread_elements()[1] = xok[r][1] ? x[xr[r][1] + kb + 2u * st] : 0.0f;
                simdgroup_multiply_accumulate(c[0][r], a0, b, c[0][r]);
                simdgroup_multiply_accumulate(c[1][r], a1, b, c[1][r]);
            }}
        }}
    }}
    for (uint r = 0; r < {rt}u; r++) {{
        simdgroup_store(c[0][r], &red[sg][r * 64u], 8);
        simdgroup_store(c[1][r], &red[sg][({rt}u + r) * 64u], 8);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"
    )
    .unwrap();
    // The simdgroups' partial products, added in a fixed order: element (i, j) of tile
    // (wt, r) is weight row 8 wt + i, pass row 8 r + j.
    if glu {
        write!(
            s,
            "    for (uint i = li; i < {rt}u * 64u; i += 128u) {{
        const uint r = i / 64u;
        const uint e = i % 64u;
        const uint row = 8u * r + e % 8u;
        const uint u = u0 + e / 8u;
        const uint ig = r * 64u + e;
        const uint iu = ({rt}u + r) * 64u + e;
        const float g = ((red[0][ig] + red[1][ig]) + red[2][ig]) + red[3][ig];
        const float up = ((red[0][iu] + red[1][iu]) + red[2][iu]) + red[3][iu];
        if (row < p.t && u < p.n) {{ y[(p.r0 + row) * p.n + u] = g / (1.0f + exp(-g)) * up; }}
    }}
}}
"
        )
        .unwrap();
    } else {
        write!(
            s,
            "    for (uint i = li; i < 2u * {rt}u * 64u; i += 128u) {{
        const uint wt = i / ({rt}u * 64u);
        const uint r = (i / 64u) % {rt}u;
        const uint e = i % 64u;
        const uint row = 8u * r + e % 8u;
        const uint n = n0 + 8u * wt + e / 8u;
        const float v = ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
        if (row < p.t && n < p.n) {{ y[(p.r0 + row) * p.n + n] = v; }}
    }}
}}
"
        )
        .unwrap();
    }
    Some(s)
}

/// Per-format access for [`gemv_q`]: `ints16(row, j)` gives the 16 integer values of chunk
/// `j` of `row` (as floats, exact) and `Sc` the chunk's scales; `apply` adds the chunk to
/// `acc` from `sq` (the integers times the activations, summed) and `sx` (the activations
/// summed), as llama.cpp's `mul_mv` kernels do: the weights are never decoded to floats.
fn ints(f: Format) -> Option<&'static str> {
    Some(match f {
        Format::Q8 => {
            "static inline uint sstride(uint blocks) { return blocks + ((blocks % 256u == 0u) ? SPAD : 0u); }
typedef float Sc;
static inline Sc ints16(device const uint* wq, device const uint* wsc, uint k, uint row, uint j, thread float* q) {
    const uint4 u = ((device const uint4*)wq)[row * (k / 16u) + j];
    const uint us[4] = { u.x, u.y, u.z, u.w };
    for (uint i = 0; i < 4; i++) {
        const char4 c = as_type<char4>(us[i]);
        q[4 * i] = float(c.x); q[4 * i + 1] = float(c.y); q[4 * i + 2] = float(c.z); q[4 * i + 3] = float(c.w);
    }
    return float(((device const half*)wsc)[row * sstride(k / 32u) + j / 2u]);
}
static inline float apply(Sc s, float sq, float sx, float acc) { return fma(s, sq, acc); }
"
        }
        Format::Q4K => {
            "static inline uint hbyte(device const uint* b, uint m, uint i) { return (b[m + 1u + i / 4u] >> (8u * (i % 4u))) & 255u; }
typedef float2 Sc;  // (d * scale, dmin * min)
static inline Sc ints16(device const uint* wq, device const uint* wsc, uint k, uint row, uint j, thread float* q) {
    const uint sub = j / 2u;
    const uint m = (row * (k / 256u) + sub / 8u) * 36u;
    const half2 dd = as_type<half2>(wsc[m]);
    const uint jj = sub % 8u;
    const bool lo = jj < 4u;
    const uint a = hbyte(wsc, m, lo ? jj : jj - 4u);
    const uint b = hbyte(wsc, m, jj + 4u);
    const uint c = hbyte(wsc, m, jj);
    const uint sc = lo ? (a & 63u) : ((b & 15u) | ((a >> 6u) << 4u));
    const uint mn = lo ? (b & 63u) : ((b >> 4u) | ((c >> 6u) << 4u));
    const uint2 u = *(device const uint2*)(wq + m + 4u + 2u * (j % 16u));
    for (uint i = 0; i < 8; i++) {
        q[i] = float((u.x >> (4u * i)) & 15u);
        q[8 + i] = float((u.y >> (4u * i)) & 15u);
    }
    return float2(float(dd.x) * float(sc), float(dd.y) * float(mn));
}
static inline float apply(Sc s, float sq, float sx, float acc) { return fma(-s.y, sx, fma(s.x, sq, acc)); }
"
        }
        Format::Q6 => {
            "typedef float Sc;
static inline Sc ints16(device const uint* wq, device const uint* wsc, uint k, uint row, uint j, thread float* q) {
    const uint g = row * (3u * k / 16u) + (j / 2u) * 6u;
    const uint l0 = wq[g + 2u * (j & 1u)];
    const uint l1 = wq[g + 2u * (j & 1u) + 1u];
    const uint h = wq[g + 4u + (j & 1u)];
    const uint los[4] = { l0, l0 >> 16u, l1, l1 >> 16u };
    for (uint i = 0; i < 4; i++) {
        const uint hq = h >> (8u * i);
        for (uint e = 0; e < 4; e++) {
            const uint v = ((los[i] >> (4u * e)) & 15u) | (((hq >> (2u * e)) & 3u) << 4u);
            q[4 * i + e] = float(int(v) - 32);
        }
    }
    const uint hb = k / 256u;
    uint rs = (18u * hb + 3u) / 4u;
    rs += ((rs * 4u) % 512u == 0u) ? 4u : 0u;
    const uint base = row * rs;
    const half2 d2 = as_type<half2>(wsc[base + j / 32u]);
    const float d = ((j / 16u) & 1u) ? float(d2.y) : float(d2.x);
    const uint off = 2u * hb + j;
    const char sc = as_type<char4>(wsc[base + off / 4u])[off % 4u];
    return d * float(sc);
}
static inline float apply(Sc s, float sq, float sx, float acc) { return fma(s, sq, acc); }
"
        }
        _ => return None,
    })
}

/// Whether [`gemv_q`] has a kernel for `f`.
pub(crate) fn gemv_q_reads(f: Format) -> bool {
    ints(f).is_some()
}

/// Whether [`gemv_q`] for `t` rows stages the activations in `max_storage` bytes of
/// threadgroup memory (as `gemv_f32::stages`).
pub(crate) fn gemv_q_stages(t: usize, rw: usize, glu: bool, max_storage: u32) -> bool {
    crate::gemv_f32::stages(t, rw, glu, max_storage)
}

/// `gemv_f32` in MSL for the formats [`ints`] reads: the same lanes, staging and reduction
/// (lane groups of 32, `rw` weight rows each, K 16 values a lane at a time, `t` <= 16 rows),
/// with the weights' integers multiplied by the activations and the scales applied once per
/// 16 values. One source for every row count, with `math_mode(safe)`, so a row gives the
/// same bits in a pass of any size.
pub(crate) fn gemv_q(f: Format, t: usize, rw: usize, glu: bool, stage: bool) -> Option<String> {
    let acc = ints(f)?;
    let mut s = header(256);
    s.push_str(PRELUDE);
    s.push_str(&acc.replace("SPAD", &format!("{}u", crate::SCALE_PAD)));
    write!(
        s,
        "
constant constexpr uint T = {t}u;
constant constexpr uint RW = {rw}u;
kernel void k0(constant P& p [[buffer(0)]],
               device float* y [[buffer(1)]],
               device const float4* x [[buffer(2)]],
               device const uint* wq [[buffer(3)]],
               device const uint* wsc [[buffer(4)]],
               uint3 wg [[threadgroup_position_in_grid]],
               uint3 nw [[threadgroups_per_grid]],
               uint li [[thread_index_in_threadgroup]],
               uint grp [[simdgroup_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {{
"
    )
    .unwrap();
    if stage {
        s.push_str("    threadgroup float4 xs[T * 128];  // [T][4 m][32 lanes]\n");
    }
    if glu {
        s.push_str("    threadgroup float red[8 * RW * T];  // [8 lane groups][RW][T]\n");
    }
    s.push_str(
        "    const uint group = wg.y * nw.x + wg.x;
    const uint k4 = p.k / 4u;
    const uint k16 = p.k / 16u;
",
    );
    if glu {
        s.push_str(
            "    const uint row0 = (group * 4u + (grp % 4u)) * RW;
    uint rws[RW];
    for (uint a = 0; a < RW; a++) { rws[a] = (grp >= 4u ? p.n : 0u) + min(row0 + a, p.n - 1u); }
",
        );
    } else {
        s.push_str(
            "    const uint row0 = (group * 8u + grp) * RW;
    uint rws[RW];
    for (uint a = 0; a < RW; a++) { rws[a] = min(row0 + a, p.n - 1u); }
",
        );
    }
    s.push_str(
        "    float r[RW][T];
    for (uint a = 0; a < RW; a++) { for (uint j = 0; j < T; j++) { r[a][j] = 0.0f; } }
",
    );
    if stage {
        s.push_str(
            "    for (uint c0 = 0; c0 < k16; c0 += 32u) {
        for (uint i = li; i < T * 128u; i += 256u) {
            const uint tt = i / 128u;
            const uint o = i % 128u;
            if (4u * c0 + o < k4) { xs[tt * 128u + (o % 4u) * 32u + o / 4u] = x[(p.r0 + tt) * k4 + 4u * c0 + o]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint c = c0 + lane;
        if (c < k16) {
",
        );
    } else {
        s.push_str("    for (uint c = lane; c < k16; c += 32u) {\n        {\n");
    }
    s.push_str(
        "            float q[RW][16];
            Sc sc[RW];
            for (uint a = 0; a < RW; a++) { sc[a] = ints16(wq, wsc, p.k, rws[a], c, q[a]); }
            for (uint j = 0; j < T; j++) {
                float xv[16];
                for (uint m = 0; m < 4; m++) {
",
    );
    if stage {
        s.push_str("                    const float4 v = xs[j * 128u + m * 32u + lane];\n");
    } else {
        s.push_str("                    const float4 v = x[(p.r0 + j) * k4 + 4u * c + m];\n");
    }
    s.push_str(
        "                    xv[4 * m] = v.x; xv[4 * m + 1] = v.y; xv[4 * m + 2] = v.z; xv[4 * m + 3] = v.w;
                }
                float sx = 0.0f;
                for (uint i = 0; i < 16; i++) { sx += xv[i]; }
                for (uint a = 0; a < RW; a++) {
                    float sq = 0.0f;
                    for (uint i = 0; i < 16; i++) { sq = fma(q[a][i], xv[i], sq); }
                    r[a][j] = apply(sc[a], sq, sx, r[a][j]);
                }
            }
        }
",
    );
    if stage {
        s.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    }
    s.push_str(
        "    }
    for (uint a = 0; a < RW; a++) {
        for (uint j = 0; j < T; j++) {
            float v = r[a][j];
            v += simd_shuffle_xor(v, 16u);
            v += simd_shuffle_xor(v, 8u);
            v += simd_shuffle_xor(v, 4u);
            v += simd_shuffle_xor(v, 2u);
            v += simd_shuffle_xor(v, 1u);
            r[a][j] = v;
        }
    }
",
    );
    if glu {
        s.push_str(
            "    if (lane == 0u) {
        for (uint a = 0; a < RW; a++) { for (uint j = 0; j < T; j++) { red[(grp * RW + a) * T + j] = r[a][j]; } }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (li < 4u) {
        for (uint a = 0; a < RW; a++) {
            const uint o = (group * 4u + li) * RW + a;
            if (o < p.n) {
                for (uint j = 0; j < T; j++) {
                    const float g = red[(li * RW + a) * T + j];
                    y[(p.r0 + j) * p.n + o] = g / (1.0f + exp(-g)) * red[((li + 4u) * RW + a) * T + j];
                }
            }
        }
    }
}
",
        );
    } else {
        s.push_str(
            "    if (lane == 0u) {
        for (uint a = 0; a < RW; a++) {
            if (row0 + a < p.n) { for (uint j = 0; j < T; j++) { y[(p.r0 + j) * p.n + row0 + a] = r[a][j]; } }
        }
    }
}
",
        );
    }
    Some(s)
}

/// The small per-row kernels in MSL, after llama.cpp's `kernel_rms_norm` (MIT, the ggml
/// authors): float4 loads, sums by `simd_sum` and a fixed-order pass over the simdgroups'
/// partials, two threadgroup barriers per sum instead of the WGSL kernels' tree. Same
/// bindings and immediates as `RMSNORM`, `ADD_NORM` and `QK_ROPE` (whose fourth immediate,
/// unused there, gives the rope table's length here: MSL has no `arrayLength`).
pub(crate) fn small_kernels() -> Vec<(&'static str, String, &'static [bool])> {
    let k = |body: &str, size: u32| {
        let mut s = header(size);
        s.push_str(PRELUDE);
        s.push_str(SUM);
        s.push_str(body);
        s
    };
    vec![
        ("rmsnorm_msl", k(RMSNORM, 256), &[false, true, true]),
        ("add_norm_msl", k(ADD_NORM, 256), &[false, false, true, true, true]),
        ("qk_rope_msl", k(QK_ROPE, 128), &[false, false, false, true, true, true, true, true]),
    ]
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

/// Whether [`gemv2`] has a kernel for `f`.
pub(crate) fn gemv2_reads(f: Format) -> bool {
    access(f).is_some()
}

/// Simdgroups per threadgroup of the scalar [`gemv2`] kernel (4 outputs a simdgroup, or 2
/// intermediate outputs with `glu`).
pub(crate) const GEMV2_SGS: usize = 2;

/// Weight tiles of 8 rows a simdgroup of the matrix [`gemv2`] kernel takes for `rt` row
/// tiles (`YAXUS_GEMV2_NT`, default 4, and `YAXUS_GEMV2_NT2`, default 2; even).
pub(crate) fn gemv2_nt(rt: usize) -> usize {
    let (var, default) = if rt == 1 { ("YAXUS_GEMV2_NT", 4) } else { ("YAXUS_GEMV2_NT2", 2) };
    std::env::var(var).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Rows a dispatch of [`gemv2`] takes at most (`YAXUS_GEMV2_SPLIT=1`: 8, the rest in
/// another dispatch).
pub(crate) fn gemv2_max_rows() -> usize {
    if std::env::var("YAXUS_GEMV2_SPLIT").is_ok_and(|v| v == "1") { 8 } else { 16 }
}

/// The most rows [`gemv2`] takes in its scalar kernel (`YAXUS_GEMV2_SCALAR`, default 2);
/// more go to simdgroup matrices (M2, 2B Q4_K_M, 2 rows: scalar 26.2 ms, matrices 34.7).
pub(crate) fn gemv2_scalar_max() -> usize {
    std::env::var("YAXUS_GEMV2_SCALAR").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
}

/// Outputs a threadgroup of [`gemv2`] computes for `t` rows (intermediate outputs with
/// `glu`).
pub(crate) fn gemv2_outputs(t: usize, glu: bool) -> usize {
    let per = if t <= gemv2_scalar_max() { 4 * GEMV2_SGS } else { 8 * gemv2_nt(t.div_ceil(8)) };
    if glu { per / 2 } else { per }
}

/// `y = x W^T` for `t` <= 16 rows (or `silu(gate) * up` with `glu`), in one of two kernels
/// that compute every output with the same arithmetic, so a row's bits do not depend on
/// the pass:
///
/// - K is cut into chunks of 32; chunk `c` belongs to slice `c % 8`. A slice has four
///   accumulators `a0 .. a3`, starting from 0: `a_s` sums, by `fma`, the values `4 k + s`
///   (`k` in 0..8) of each of the slice's chunks in order (the weights decoded to f32 by
///   `load8`). The slice's sum is `(a0 + a1) + (a2 + a3)`. (Four chains, so that a lane of
///   the scalar kernel does not wait on one.)
/// - The 8 slices are added as `((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7))`.
///
/// The scalar kernel gives each slice to a lane (8 lanes a weight row). The matrix kernel
/// gives each slice to a simdgroup and works on 8x8 simdgroup matrices: weights in A (row
/// `fm`, columns `fn` and `fn + 1` of lane `l`), the pass's rows in B, padded to 8 or 16.
/// Column `k` of step `s` is value `4 k + s`, so a lane's A elements over a chunk are 8
/// consecutive weights and its B elements one float4 a row. On the M2 a multiply-
/// accumulate of 8x8 matrices is an `fma` chain over k = 0..7 from the accumulator, bit
/// for bit (DESIGN.md 12.8), which makes the two kernels agree.
///
/// Bindings: 0 `y [T][N]`, 1 `x [T][K]`, 2 and 3 the weights (decode layout). Immediates:
/// N (the intermediate size with `glu`), K, T, first row. Needs K % 32 == 0.
pub(crate) fn gemv2(f: Format, t: usize, glu: bool) -> Option<String> {
    let acc = access(f)?;
    let mut s = if t <= gemv2_scalar_max() { header(32 * GEMV2_SGS as u32) } else { header(256) };
    s.push_str(PRELUDE);
    s.push_str(&acc.replace("SPAD", &format!("{}u", crate::SCALE_PAD)));
    if t <= gemv2_scalar_max() {
        gemv2_scalar(&mut s, t, glu)
    } else {
        gemv2_matrix(&mut s, t.div_ceil(8), glu)
    }
    Some(s)
}

fn gemv2_scalar(s: &mut String, t: usize, glu: bool) {
    write!(
        s,
        "
constant constexpr uint T = {t}u;
kernel void k0(constant P& p [[buffer(0)]],
               device float* y [[buffer(1)]],
               device const float* x [[buffer(2)]],
               device const uint* wq [[buffer(3)]],
               device const uint* wsc [[buffer(4)]],
               uint3 wg [[threadgroup_position_in_grid]],
               uint3 nw [[threadgroups_per_grid]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {{
    const uint group = wg.y * nw.x + wg.x;
    const uint rg = lane / 8u;  // the lane's row of the simdgroup's 4
    const uint l8 = lane % 8u;  // its slice of K
",
    )
    .unwrap();
    if glu {
        write!(
            s,
            "    const uint out = (group * {GEMV2_SGS}u + sg) * 2u + rg % 2u;  // the intermediate output
    const uint row = (rg >= 2u ? p.n : 0u) + min(out, p.n - 1u);
"
        )
        .unwrap();
    } else {
        write!(
            s,
            "    const uint out = (group * {GEMV2_SGS}u + sg) * 4u + rg;
    const uint row = min(out, p.n - 1u);
"
        )
        .unwrap();
    }
    s.push_str(
        "    float4 acc[T];
    for (uint j = 0; j < T; j++) { acc[j] = float4(0.0f); }
    for (uint c = l8; c < p.k / 32u; c += 8u) {
        float w[32];
        load32(wq, wsc, p.k, row, c, w);
        for (uint j = 0; j < T; j++) {
            device const float4* xv = (device const float4*)(x + (p.r0 + j) * p.k + 32u * c);
            float4 v[8];
            for (uint i = 0; i < 8; i++) { v[i] = xv[i]; }
            float4 a = acc[j];
            for (uint kk = 0; kk < 8; kk++) { a = fma(float4(w[4u * kk], w[4u * kk + 1u], w[4u * kk + 2u], w[4u * kk + 3u]), v[kk], a); }
            acc[j] = a;
        }
    }
    float sum[T];
    for (uint j = 0; j < T; j++) {
        float v = (acc[j].x + acc[j].y) + (acc[j].z + acc[j].w);
        v += simd_shuffle_xor(v, 4u);
        v += simd_shuffle_xor(v, 2u);
        v += simd_shuffle_xor(v, 1u);
        sum[j] = v;
    }
",
    );
    if glu {
        s.push_str(
            "    // (lanes of rows 0 and 1 get the up row's sums from 16 lanes on)
    for (uint j = 0; j < T; j++) {
        const float up = simd_shuffle_down(sum[j], 16u);
        if (rg < 2u && l8 == 0u && out < p.n) {
            const float g = sum[j];
            y[(p.r0 + j) * p.n + out] = g / (1.0f + exp(-g)) * up;
        }
    }
}
",
        );
    } else {
        s.push_str(
            "    if (l8 == 0u && out < p.n) {
        for (uint j = 0; j < T; j++) { y[(p.r0 + j) * p.n + out] = sum[j]; }
    }
}
",
        );
    }
}

fn gemv2_matrix(s: &mut String, rt: usize, glu: bool) {
    let nt = gemv2_nt(rt);
    write!(
        s,
        "
#define FOR_UNROLL _Pragma(\"clang loop unroll(full)\") for
constant constexpr uint NT = {nt}u;  // weight tiles of 8 rows
constant constexpr uint RT = {rt}u;  // row tiles of 8
kernel void k0(constant P& p [[buffer(0)]],
               device float* y [[buffer(1)]],
               device const float* x [[buffer(2)]],
               device const uint* wq [[buffer(3)]],
               device const uint* wsc [[buffer(4)]],
               uint3 wg [[threadgroup_position_in_grid]],
               uint3 nw [[threadgroups_per_grid]],
               uint li [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {{
    threadgroup float red[8 * NT * RT * 64];  // [slice][weight tile][row tile][8][8]
    const uint group = wg.y * nw.x + wg.x;
    // The lane's elements of an 8x8 matrix: row fm, columns fn and fn + 1.
    const uint fm = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    const uint fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    uint wr[NT];  // the weight row of the lane's A row in each tile (clamped to the last)
"
    )
    .unwrap();
    if glu {
        s.push_str(
            "    const uint u0 = group * 4u * NT;
    FOR_UNROLL (uint t = 0; t < NT / 2u; t++) {
        wr[t] = min(u0 + 8u * t + fm, p.n - 1u);
        wr[t + NT / 2u] = p.n + wr[t];
    }
",
        );
    } else {
        s.push_str(
            "    const uint n0 = group * 8u * NT;
    FOR_UNROLL (uint t = 0; t < NT; t++) { wr[t] = min(n0 + 8u * t + fm, p.n - 1u); }
",
        );
    }
    s.push_str(
        "    // B: the values 4 fm .. 4 fm + 3 of each chunk for the lane's two pass rows.
    device const float4* xb[RT][2];
    bool xok[RT][2];
    FOR_UNROLL (uint r = 0; r < RT; r++) {
        FOR_UNROLL (uint e = 0; e < 2u; e++) {
            const uint row = 8u * r + fn + e;
            xok[r][e] = row < p.t;
            xb[r][e] = (device const float4*)(x + (p.r0 + min(row, p.t - 1u)) * p.k) + fm;
        }
    }
    simdgroup_float8x8 c[4][NT][RT];  // [step]
    FOR_UNROLL (uint st = 0; st < 4u; st++) {
        FOR_UNROLL (uint t = 0; t < NT; t++) {
            FOR_UNROLL (uint r = 0; r < RT; r++) { c[st][t][r] = make_filled_simdgroup_matrix<float, 8>(0.0f); }
        }
    }
    for (uint ch = sg; ch < p.k / 32u; ch += 8u) {
        float4 b[RT][2];
        FOR_UNROLL (uint r = 0; r < RT; r++) {
            FOR_UNROLL (uint e = 0; e < 2u; e++) { b[r][e] = xok[r][e] ? xb[r][e][8u * ch] : float4(0.0f); }
        }
        float w[NT][8];
        FOR_UNROLL (uint t = 0; t < NT; t++) { load8(wq, wsc, p.k, wr[t], 32u * ch + 4u * fn, w[t]); }
        FOR_UNROLL (uint st = 0; st < 4u; st++) {
            simdgroup_float8x8 mb[RT];
            FOR_UNROLL (uint r = 0; r < RT; r++) {
                mb[r].thread_elements()[0] = b[r][0][st];
                mb[r].thread_elements()[1] = b[r][1][st];
            }
            FOR_UNROLL (uint t = 0; t < NT; t++) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[t][st];
                ma.thread_elements()[1] = w[t][4u + st];
                FOR_UNROLL (uint r = 0; r < RT; r++) { simdgroup_multiply_accumulate(c[st][t][r], ma, mb[r], c[st][t][r]); }
            }
        }
    }
    FOR_UNROLL (uint t = 0; t < NT; t++) {
        FOR_UNROLL (uint r = 0; r < RT; r++) {
            threadgroup float* o = red + ((sg * NT + t) * RT + r) * 64u + 8u * fm + fn;
            FOR_UNROLL (uint e = 0; e < 2u; e++) {
                o[e] = (c[0][t][r].thread_elements()[e] + c[1][t][r].thread_elements()[e])
                     + (c[2][t][r].thread_elements()[e] + c[3][t][r].thread_elements()[e]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Element (i, j) of tile (t, r) is weight row 8 t + i, pass row 8 r + j.
#define SL(q, i) red[(q) * NT * RT * 64u + (i)]
#define SUM(i) (((SL(0, i) + SL(4, i)) + (SL(2, i) + SL(6, i))) + ((SL(1, i) + SL(5, i)) + (SL(3, i) + SL(7, i))))
",
    );
    if glu {
        s.push_str(
            "    for (uint i = li; i < (NT / 2u) * RT * 64u; i += 256u) {
        const uint t = i / (RT * 64u);
        const uint e = i % 64u;
        const uint row = 8u * ((i / 64u) % RT) + e % 8u;
        const uint u = u0 + 8u * t + e / 8u;
        const float g = SUM(i);
        const float up = SUM(i + (NT / 2u) * RT * 64u);
        if (row < p.t && u < p.n) { y[(p.r0 + row) * p.n + u] = g / (1.0f + exp(-g)) * up; }
    }
}
",
        );
    } else {
        s.push_str(
            "    for (uint i = li; i < NT * RT * 64u; i += 256u) {
        const uint t = i / (RT * 64u);
        const uint e = i % 64u;
        const uint row = 8u * ((i / 64u) % RT) + e % 8u;
        const uint n = n0 + 8u * t + e / 8u;
        if (row < p.t && n < p.n) { y[(p.r0 + row) * p.n + n] = SUM(i); }
    }
}
",
        );
    }
}
