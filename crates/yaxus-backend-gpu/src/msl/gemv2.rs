//! `gemv2`: passes of 1-16 rows.

use std::fmt::Write;

use super::access::access;
use super::{PRELUDE, header};
use crate::Format;
use crate::device::AppleTuning;

/// Whether [`gemv2`] has a kernel for `f`.
pub(crate) fn gemv2_reads(f: Format) -> bool {
    access(f).is_some()
}

/// Simdgroups per threadgroup of the scalar [`gemv2`] kernel (4 outputs a simdgroup, or 2
/// intermediate outputs with `glu`).
const GEMV2_SGS: usize = 2;

/// A [`gemv2`] kernel: `t` rows, with `glu`, in the scalar kernel or on simdgroup matrices
/// of `tiles` weight tiles a simdgroup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gemv2 {
    pub(crate) t: usize,
    pub(crate) glu: bool,
    /// `None`: the scalar kernel.
    pub(crate) tiles: Option<usize>,
}

impl Gemv2 {
    /// The kernel for `t` rows with `apple`'s sizes.
    pub(crate) fn for_rows(t: usize, glu: bool, apple: &AppleTuning) -> Self {
        let tiles = (t > apple.gemv2_scalar_max).then(|| apple.gemv2_tiles[usize::from(t > 8)]);
        debug_assert!(tiles.is_none_or(|nt| nt >= 2 && nt % 2 == 0), "gemv2 weight tiles must be even");
        Gemv2 { t, glu, tiles }
    }

    /// Outputs a threadgroup computes (intermediate outputs with `glu`).
    pub(crate) fn outputs(self) -> usize {
        let per = match self.tiles {
            None => 4 * GEMV2_SGS,
            Some(nt) => 8 * nt,
        };
        if self.glu { per / 2 } else { per }
    }

    pub(crate) fn source(self, f: Format) -> Option<String> {
        gemv2(f, self)
    }
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
/// for bit (HISTORY.md 12.8), which makes the two kernels agree.
///
/// Bindings: 0 `y [T][N]`, 1 `x [T][K]`, 2 and 3 the weights (decode layout). Immediates:
/// N (the intermediate size with `glu`), K, T, first row. Needs K % 32 == 0.
fn gemv2(f: Format, k: Gemv2) -> Option<String> {
    let acc = access(f)?;
    let mut s = header(if k.tiles.is_some() { 256 } else { 32 * GEMV2_SGS as u32 });
    s.push_str(PRELUDE);
    s.push_str(&acc.replace("SPAD", &format!("{}u", crate::SCALE_PAD)));
    match k.tiles {
        None => gemv2_scalar(&mut s, k.t, k.glu),
        Some(nt) => gemv2_matrix(&mut s, k.t.div_ceil(8), nt, k.glu),
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

fn gemv2_matrix(s: &mut String, rt: usize, nt: usize, glu: bool) {
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
