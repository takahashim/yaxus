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
    for (uint i = 0; i < 8; i++) { w[i] = float((u >> (4u * i)) & 15u) * d - dm; }
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
