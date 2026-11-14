//! `gemm`: passes of more than 16 rows.

use std::fmt::Write;

use super::access::access;
use super::{PRELUDE, header};
use crate::Format;

/// Whether [`gemm`] has a kernel for `f`.
pub(crate) fn gemm_reads(f: Format) -> bool {
    access(f).is_some()
}

/// A [`gemm`] kernel: tiles of `32 rb` rows of the pass, f16 or f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Gemm {
    pub(crate) f16: bool,
    pub(crate) rb: usize,
}

impl Gemm {
    /// Rows of the pass a tile takes.
    pub(crate) fn rows(self) -> usize {
        32 * self.rb
    }

    pub(crate) fn source(self, f: Format) -> Option<String> {
        gemm(f, self.f16, self.rb)
    }
}

/// `y = x W^T` for passes of more than 16 rows, as llama.cpp's `kernel_mul_mm`: a
/// threadgroup of 128 takes 64 weight rows and `32 rb` of the pass's rows (from the first
/// row immediate on), K in steps of 32.
/// Each step every thread decodes 16 weights (`load8`) and reads 8 activations into
/// threadgroup memory, as 8x8 blocks (`f16`: rounded to half), and each simdgroup
/// multiplies `16 rb` rows by 32 weight rows on 8x8 matrices (f32 sums). Rows of the pass
/// past `t` are computed from row `t - 1` and stored too (the output has room for them).
/// (M2, 2B Q4_K_M, 256 rows: `rb` 1 / 2 588 / 536 ms in f32, 533 / 492 in f16; 128
/// weight rows a tile were slower, 549 / 535, and so was reading the next step ahead.)
///
/// Bindings: 0 `y [T][N]`, 1 `x [T][K]`, 2 and 3 the weights (decode layout). Immediates:
/// N, K, T (the end of the pass), first row. Needs N % 64 == 0, K % 32 == 0.
fn gemm(f: Format, f16: bool, rb: usize) -> Option<String> {
    let acc = access(f)?;
    let (st, mt) = if f16 { ("half", "simdgroup_half8x8") } else { ("float", "simdgroup_float8x8") };
    let mut s = header(128);
    s.push_str(PRELUDE);
    s.push_str(&acc.replace("SPAD", &format!("{}u", crate::SCALE_PAD)));
    write!(
        s,
        "
#define FOR_UNROLL _Pragma(\"clang loop unroll(full)\") for
typedef {st} S;
typedef {mt} M;
constant constexpr uint RB = {rb}u;     // the tile's rows of the pass: 32 RB
constant constexpr uint TB = 4u * RB;  // 8-row blocks of them
constant constexpr uint BV = 8u * RB;  // activations a thread reads per step
kernel void k0(constant P& p [[buffer(0)]],
               device float* y [[buffer(1)]],
               device const float* x [[buffer(2)]],
               device const uint* wq [[buffer(3)]],
               device const uint* wsc [[buffer(4)]],
               uint3 wg [[threadgroup_position_in_grid]],
               ushort li [[thread_index_in_threadgroup]],
               ushort sg [[simdgroup_index_in_threadgroup]]) {{
    threadgroup S sa[64 * 32];  // [k block 4][row block 8][8 k][8 weight rows]
    threadgroup S sb[32 * RB * 32];  // [k block 4][row block TB][8 rows][8 k]
    const uint r0 = wg.y * 64u;  // the tile's weight rows
    const uint r1 = p.r0 + wg.x * 32u * RB;  // the tile's rows of the pass
    const uint lr0 = li / 2u;    // the weight row this thread decodes, 16 values of each step
    const uint il0 = li % 2u;
    const uint lr1 = li / (32u / BV);  // the pass row it reads, BV values of each step
    const uint iy = BV * (li % (32u / BV));
    device const float* xr = x + min(r1 + lr1, p.t - 1u) * p.k + iy;
    M ma[4];
    M mb[2 * RB];
    simdgroup_float8x8 mc[8 * RB];
    FOR_UNROLL (uint i = 0; i < 8 * RB; i++) {{ mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }}
    for (uint kb = 0; kb < p.k; kb += 32u) {{
        float w[16];
        load8(wq, wsc, p.k, r0 + lr0, kb + 16u * il0, w);
        load8(wq, wsc, p.k, r0 + lr0, kb + 16u * il0 + 8u, w + 8);
        float4 xv[BV / 4u];
        FOR_UNROLL (uint i = 0; i < BV / 4u; i++) {{ xv[i] = *(device const float4*)(xr + kb + 4u * i); }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (uint i = 0; i < 16; i++) {{
            const uint ib = 8u * (2u * il0 + i / 8u) + lr0 / 8u;
            sa[64u * ib + 8u * (i % 8u) + lr0 % 8u] = S(w[i]);
        }}
        FOR_UNROLL (uint v = 0; v < BV; v++) {{
            const uint kk = iy + v;
            sb[64u * (TB * (kk / 8u) + lr1 / 8u) + 8u * (lr1 % 8u) + kk % 8u] = S(xv[v / 4u][v % 4u]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const S* lsma = sa + 4u * 64u * (sg % 2u);
        threadgroup const S* lsmb = sb + 2u * RB * 64u * (sg / 2u);
        FOR_UNROLL (uint ik = 0; ik < 4; ik++) {{
            FOR_UNROLL (uint i = 0; i < 4; i++) {{ simdgroup_load(ma[i], lsma + 64u * i, 8); }}
            FOR_UNROLL (uint i = 0; i < 2 * RB; i++) {{ simdgroup_load(mb[i], lsmb + 64u * i, 8); }}
            FOR_UNROLL (uint i = 0; i < 8 * RB; i++) {{ simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i % 4u], mc[i]); }}
            lsma += 8u * 64u;
            lsmb += TB * 64u;
        }}
    }}
    device float* c = y + (r0 + 32u * (sg % 2u)) + (r1 + 16u * RB * (sg / 2u)) * p.n;
    FOR_UNROLL (uint i = 0; i < 8 * RB; i++) {{ simdgroup_store(mc[i], c + 8u * (i % 4u) + 8u * p.n * (i / 4u), p.n); }}
}}
"
    )
    .unwrap();
    Some(s)
}
