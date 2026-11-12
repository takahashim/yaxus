//! f32 GEMV for 1 to 16 activation rows, with or without SwiGLU: generated WGSL.
//!
//! One source for every row count, so that a row's result does not depend on the size
//! of its pass: each lane walks K the same way (16 values at a time, `lane`, `lane + 32`,
//! ...), multiplies them in the same order with explicit `fma`, and the 32 lanes are
//! summed by the same shuffles. The decoded weights go through `keep`: where the compiler
//! uses fast math (Metal) it could otherwise fold a weight's scale into the products in
//! one kernel and not in another.
//!
//! The activations go through workgroup memory 512 values per row at a time, shared by the
//! 8 lane groups, where they fit (see [`stages`]), and are read from the buffer otherwise;
//! either way each lane does the same arithmetic, so the two agree to the bit. Measured on
//! an M2 (2B Q4_K_M, ms per pass of 1 / 2 / 4 / 8 / 12 / 16 rows): staged with 2 weight
//! rows per lane group 23.5 / 28.8 / 42.0 / 60.2 / 79.3 / 131.9; read from the buffer 23.4
//! / 31.1 / 54.5 / 90.2 / 133.7 / 178.1 (with 4 weight rows 25.2 / 27.3 / 41.1 / 84.0 /
//! 154.4 / 226.5); `gemv1` and `GEMV_MULTI` 30.2 / 65 / 96 / 116 / - / 305.

/// Whether the kernel for `t` rows stages the activations: `xs` (and with `glu` the
/// reduction) fits in `max_storage` bytes of workgroup memory.
pub fn stages(t: usize, rw: usize, glu: bool, max_storage: u32) -> bool {
    let bytes = 16 * 128 * t + if glu { 4 * 8 * rw * t } else { 0 };
    bytes <= max_storage as usize
}

use std::fmt::Write;

/// The kernel body for `t` rows, `rw` weight rows per 32-lane group, the activations
/// staged in workgroup memory with `stage`, and with `glu` the
/// fused `silu(gate) * up` (lane groups 0-3 take gate rows, 4-7 the up rows `n + u`, with
/// `p.n` the intermediate size). Goes after `MM_PARAMS` and a format's `W_*_16` loader.
///
/// Immediates: N, K, T (the pass's rows, at least 1: only `keep` reads it), first row.
pub fn source(t: usize, rw: usize, glu: bool, stage: bool) -> String {
    let mut s = String::from(
        "// The decoded weights as computed, whatever their uses (`p.t` is never 0, which
// the compiler cannot know).
fn keep(w: array<vec4<f32>, 4>) -> array<vec4<f32>, 4> {
    let z = vec4<f32>(0.0);
    let never = p.t == 0u;
    return array<vec4<f32>, 4>(select(w[0], z, never), select(w[1], z, never),
                               select(w[2], z, never), select(w[3], z, never));
}
// `acc` plus the 16 products of a chunk, in this order.
fn dot16(w: array<vec4<f32>, 4>, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, acc: f32) -> f32 {
    var s = acc;
    s = fma(w[0].x, x0.x, s); s = fma(w[0].y, x0.y, s); s = fma(w[0].z, x0.z, s); s = fma(w[0].w, x0.w, s);
    s = fma(w[1].x, x1.x, s); s = fma(w[1].y, x1.y, s); s = fma(w[1].z, x1.z, s); s = fma(w[1].w, x1.w, s);
    s = fma(w[2].x, x2.x, s); s = fma(w[2].y, x2.y, s); s = fma(w[2].z, x2.z, s); s = fma(w[2].w, x2.w, s);
    s = fma(w[3].x, x3.x, s); s = fma(w[3].y, x3.y, s); s = fma(w[3].z, x3.z, s); s = fma(w[3].w, x3.w, s);
    return s;
}
",
    );
    let w = &mut s;
    if glu {
        writeln!(w, "var<workgroup> red: array<f32, {}>;  // [8 lane groups][R][T]", 8 * rw * t).unwrap();
    }
    if stage {
        writeln!(w, "var<workgroup> xs: array<vec4<f32>, {}>;  // [T][4 m][32 lanes]", 128 * t).unwrap();
    }
    write!(
        w,
        "
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {{
    let group = wg.y * nw.x + wg.x;
    let grp = li / 32u;
    let lane = li % 32u;
    let k4 = p.k / 4u;
    let k16 = p.k / 16u;
"
    )
    .unwrap();
    // Rows past the end compute a copy of the last row and do not store it.
    if glu {
        writeln!(w, "    let row0 = (group * 4u + (grp % 4u)) * {rw}u;").unwrap();
        for a in 0..rw {
            writeln!(w, "    let rw{a} = select(0u, p.n, grp >= 4u) + min(row0 + {a}u, p.n - 1u);").unwrap();
        }
    } else {
        writeln!(w, "    let row0 = (group * 8u + grp) * {rw}u;").unwrap();
        for a in 0..rw {
            writeln!(w, "    let rw{a} = min(row0 + {a}u, p.n - 1u);").unwrap();
        }
    }
    for a in 0..rw {
        for j in 0..t {
            writeln!(w, "    var r{a}_{j} = 0.0;").unwrap();
        }
    }
    if stage {
        write!(
            w,
            "    for (var c0 = 0u; c0 < k16; c0 += 32u) {{
        for (var i = li; i < {n}u; i += 256u) {{
            let tt = i / 128u;
            let o = i % 128u;
            if (4u * c0 + o < k4) {{ xs[tt * 128u + (o % 4u) * 32u + o / 4u] = x[(p.r0 + tt) * k4 + 4u * c0 + o]; }}
        }}
        workgroupBarrier();
        let c = c0 + lane;
        if (c < k16) {{
",
            n = 128 * t
        )
        .unwrap();
    } else {
        writeln!(w, "    for (var c = lane; c < k16; c += 32u) {{").unwrap();
    }
    for a in 0..rw {
        writeln!(w, "        let w{a} = keep(load16(rw{a}, c));").unwrap();
    }
    for j in 0..t {
        if stage {
            writeln!(w, "        let xb{j} = {}u + lane;", 128 * j).unwrap();
            for a in 0..rw {
                writeln!(
                    w,
                    "        r{a}_{j} = dot16(w{a}, xs[xb{j}], xs[xb{j} + 32u], xs[xb{j} + 64u], xs[xb{j} + 96u], r{a}_{j});"
                )
                .unwrap();
            }
        } else {
            writeln!(w, "        let xb{j} = (p.r0 + {j}u) * k4 + 4u * c;").unwrap();
            for a in 0..rw {
                writeln!(
                    w,
                    "        r{a}_{j} = dot16(w{a}, x[xb{j}], x[xb{j} + 1u], x[xb{j} + 2u], x[xb{j} + 3u], r{a}_{j});"
                )
                .unwrap();
            }
        }
    }
    if stage {
        writeln!(w, "        }}\n        workgroupBarrier();").unwrap();
    }
    writeln!(w, "    }}").unwrap();
    for a in 0..rw {
        for j in 0..t {
            for sh in [16, 8, 4, 2, 1] {
                writeln!(w, "    r{a}_{j} += subgroupShuffleXor(r{a}_{j}, {sh}u);").unwrap();
            }
        }
    }
    if glu {
        writeln!(w, "    if (lane == 0u) {{").unwrap();
        for a in 0..rw {
            for j in 0..t {
                writeln!(w, "        red[(grp * {rw}u + {a}u) * {t}u + {j}u] = r{a}_{j};").unwrap();
            }
        }
        writeln!(w, "    }}\n    workgroupBarrier();").unwrap();
        for a in 0..rw {
            writeln!(w, "    if (li < 4u && (group * 4u + li) * {rw}u + {a}u < p.n) {{").unwrap();
            writeln!(w, "        let o = (group * 4u + li) * {rw}u + {a}u;").unwrap();
            for j in 0..t {
                writeln!(w, "        let g{j} = red[(li * {rw}u + {a}u) * {t}u + {j}u];").unwrap();
                writeln!(
                    w,
                    "        y[(p.r0 + {j}u) * p.n + o] = g{j} / (1.0 + exp(-g{j})) * red[((li + 4u) * {rw}u + {a}u) * {t}u + {j}u];"
                )
                .unwrap();
            }
            writeln!(w, "    }}").unwrap();
        }
    } else {
        for a in 0..rw {
            writeln!(w, "    if (lane == 0u && row0 + {a}u < p.n) {{").unwrap();
            for j in 0..t {
                writeln!(w, "        y[(p.r0 + {j}u) * p.n + row0 + {a}u] = r{a}_{j};").unwrap();
            }
            writeln!(w, "    }}").unwrap();
        }
    }
    s.push_str("}\n");
    s
}
