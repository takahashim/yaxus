//! Matrix products for a few activation rows (decode, speculative verification, short
//! prompts) with packed integer dot products: generated WGSL.
//!
//! As in `gemm_int`, the weights enter as their integer values and the activations as
//! int8 planes per row and block of 32 (`xquant_source` here), and the integer sums are
//! exact. Unlike the GEMM, the weights are read once, straight into registers: 8 weight
//! rows per workgroup, 32 lanes per row, a lane taking one block of 32 weights per step
//! and every activation row against it. A row's result does not depend on the other rows
//! of the pass (the same operations in the same order for every `T`).

use std::fmt::Write;

/// A weight format the integer GEMV reads.
pub struct IntFmt {
    pub name: &'static str,
    /// WGSL declaring `wq` and `wsc` (bindings 2 and 3) and the format's helpers.
    decl: &'static str,
    /// WGSL giving `ea` and `eb`, `vec4<u32>` of int8 x 4: the 8 u32 of weights of block
    /// `kb` in row `r` are `ea.x, eb.x, ea.y, eb.y, ..` (`interleave`) or `ea` then `eb`.
    load: &'static str,
    interleave: bool,
    /// WGSL expression of (scale, min) as `vec2<f32>` for sub-block `h` of block `kb`.
    scale: &'static str,
    /// Scale sub-blocks per block of 32.
    ks: usize,
    /// Whether weights are `q * scale - min` with a nonzero `min`.
    has_min: bool,
    /// Order of the 8 activations of a u32 pair: 0 as stored, 1 even elements then odd.
    pub xmode: u32,
}

const SSTRIDE: &str = "fn sstride(blocks: u32) -> u32 { return blocks + select(0u, {SPAD}u, blocks % 256u == 0u); }
fn f16scale(b: u32) -> f32 {
    let s2 = unpack2x16float(wsc[b / 2u]);
    return select(s2.x, s2.y, (b & 1u) == 1u);
}
";

/// Scales of Q6_K and Q3_K as in the file: per row `K / 256` f16 `d`, then `K / 16` int8
/// (`q6_meta_stride` on the host).
const META_SCALE: &str = "fn mscale(row: u32, blk: u32) -> f32 {
    let hb = p.k / 256u;
    var rs = (18u * hb + 3u) / 4u;
    rs += select(0u, 4u, (rs * 4u) % 512u == 0u);
    let base = row * rs;
    let d = unpack2x16float(wsc[base + blk / 32u])[(blk / 16u) & 1u];
    let off = 2u * hb + blk;
    let sc = extractBits(bitcast<i32>(wsc[base + off / 4u]), 8u * (off % 4u), 8u);
    return d * f32(sc);
}
";

pub const FORMATS: [IntFmt; 6] = [
    IntFmt {
        name: "q4k",
        decl: "@group(0) @binding(2) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Scale and min of 32-block `sub` of a row from its Q4_K header (see `W_Q4K`).
fn hbyte(m: u32, i: u32) -> u32 { return (wsc[m + 1u + i / 4u] >> (8u * (i % 4u))) & 255u; }
fn q4k_sm(row: u32, sub: u32) -> vec2<f32> {
    let m = (row * (p.k / 256u) + sub / 8u) * 36u;
    let dd = unpack2x16float(wsc[m]);
    let j = sub % 8u;
    let lo = j < 4u;
    let a = hbyte(m, select(j - 4u, j, lo));
    let b = hbyte(m, j + 4u);
    let c = hbyte(m, j);
    let sc = select((b & 15u) | ((a >> 6u) << 4u), a & 63u, lo);
    let mn = select((b >> 4u) | ((c >> 6u) << 4u), b & 63u, lo);
    return vec2<f32>(dd.x * f32(sc), dd.y * f32(mn));
}
",
        load: "let v = wq[(r * (p.k / 256u) + kb / 8u) * 9u + 1u + kb % 8u];
        let ea = cen4(v & vec4<u32>(0x0f0f0f0fu), 0x08080808u);
        let eb = cen4((v >> vec4<u32>(4u)) & vec4<u32>(0x0f0f0f0fu), 0x08080808u);",
        interleave: true,
        scale: "center(q4k_sm(r, kb), 8.0)",
        ks: 1,
        has_min: true,
        xmode: 1,
    },
    IntFmt {
        name: "q4",
        decl: "@group(0) @binding(2) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
",
        load: "let v = wq[r * (p.k / 32u) + kb];
        let ea = cen4(v & vec4<u32>(0x0f0f0f0fu), 0x08080808u);
        let eb = cen4((v >> vec4<u32>(4u)) & vec4<u32>(0x0f0f0f0fu), 0x08080808u);",
        interleave: true,
        scale: "vec2<f32>(f16scale(r * sstride(p.k / 32u) + kb), 0.0)",
        ks: 1,
        has_min: false,
        xmode: 1,
    },
    IntFmt {
        name: "q8",
        decl: "@group(0) @binding(2) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
",
        load: "let ea = wq[r * (p.k / 16u) + 2u * kb];
        let eb = wq[r * (p.k / 16u) + 2u * kb + 1u];",
        interleave: false,
        scale: "vec2<f32>(f16scale(r * sstride(p.k / 32u) + kb), 0.0)",
        ks: 1,
        has_min: false,
        xmode: 0,
    },
    // 6 u32 per block: 4 of low nibbles, 2 of high 2-bit fields (see `W_Q6`).
    IntFmt {
        name: "q6",
        decl: "@group(0) @binding(2) var<storage, read> wq: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
",
        load: "let g = r * (3u * p.k / 32u) + kb * 3u;
        let l0 = wq[g];
        let l1 = wq[g + 1u];
        let hh = wq[g + 2u];
        let lo = vec4<u32>(l0, l1);
        let hi = vec4<u32>(hh.x, hh.x >> 16u, hh.y, hh.y >> 16u);
        let ea = cen4((lo & vec4<u32>(0x0f0f0f0fu)) | (spread2e(hi) << vec4<u32>(4u)), 0x20202020u);
        let eb = cen4(((lo >> vec4<u32>(4u)) & vec4<u32>(0x0f0f0f0fu)) | (spread2e(hi >> vec4<u32>(2u)) << vec4<u32>(4u)), 0x20202020u);",
        interleave: true,
        scale: "vec2<f32>(mscale(r, 2u * kb + h), 0.0)",
        ks: 2,
        has_min: false,
        xmode: 1,
    },
    // 3 u32 per block: 2 of low 2-bit fields, 1 of high bits (see `W_Q3`).
    IntFmt {
        name: "q3",
        decl: "@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
",
        load: "let g = r * (3u * p.k / 32u) + kb * 3u;
        let w0 = wq[g];
        let w1 = wq[g + 1u];
        let w2 = wq[g + 2u];
        let lo = vec4<u32>(w0, w0 >> 16u, w1, w1 >> 16u);
        let hi = vec4<u32>(w2, w2 >> 8u, w2 >> 16u, w2 >> 24u);
        let ea = cen4(spread2(lo) | (spread1(hi) << vec4<u32>(2u)), 0x04040404u);
        let eb = cen4(spread2(lo >> vec4<u32>(8u)) | (spread1(hi >> vec4<u32>(4u)) << vec4<u32>(2u)), 0x04040404u);",
        interleave: true,
        scale: "vec2<f32>(mscale(r, 2u * kb + h), 0.0)",
        ks: 2,
        has_min: false,
        xmode: 0,
    },
    // 2 u32 per block, scales and mins as in the file (see `W_Q2_16`).
    IntFmt {
        name: "q2",
        decl: "@group(0) @binding(2) var<storage, read> wq: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
fn q2sm(row: u32, blk: u32) -> vec2<f32> {
    let hb = p.k / 256u;
    let base = row * 5u * hb;
    let dd = unpack2x16float(wsc[base + blk / 16u]);
    let b = wsc[base + hb + blk / 4u] >> (8u * (blk % 4u));
    return dd * vec2<f32>(f32(b & 15u), f32((b >> 4u) & 15u));
}
",
        load: "let w = wq[r * (p.k / 32u) + kb];
        let lo = vec4<u32>(w.x, w.x >> 16u, w.y, w.y >> 16u);
        let ea = cen4(spread2(lo), 0x02020202u);
        let eb = cen4(spread2(lo >> vec4<u32>(8u)), 0x02020202u);",
        interleave: true,
        scale: "center(q2sm(r, 2u * kb + h), 2.0)",
        ks: 2,
        has_min: true,
        xmode: 0,
    },
];

pub fn format(name: &str) -> Option<&'static IntFmt> {
    FORMATS.iter().find(|f| f.name == name)
}

/// Source of the kernel for exactly `t` activation rows of `planes` int8 planes (1: 7 bits
/// and a sign per block of 32; 2: 15 bits, `128 hi + lo`), each 32-lane group taking `rw`
/// weight rows. With `glu`, lane groups 0-3 take gate rows `u` and groups 4-7 the up rows
/// `n + u` (`p.n` the intermediate size), and the kernel stores `silu(gate) * up`.
///
/// Bindings: 0 `y [T][N]` f32, 1 the planes from `xquant`, 2 and 3 the weights, 4 the
/// activation units and sums. Immediates: N, K, T, first row (of `y` and the planes).
///
/// With `inline` (one row only), binding 1 is the f32 row itself and there is no binding
/// 4: each workgroup rounds the 32 blocks of a step into workgroup memory first (a lane
/// per 4 values, the block's maximum and sums by subgroup shuffles), with the operations
/// of `xquant_source`, so the results are the same bits. That saves the `xquant` dispatch,
/// which costs more than its work when a pass is bound by reading the weights.
pub fn source(f: &IntFmt, t: usize, planes: usize, rw: usize, glu: bool, inline: bool) -> String {
    assert!(!inline || t == 1);
    let ks = f.ks;
    let np = 2 * planes; // vec4<u32> per row and block
    let top = if planes == 2 { "16383.0" } else { "127.0" };
    let mut s = String::new();
    let w = &mut s;
    write!(
        w,
        "struct P {{ n: u32, k: u32, t: u32, r0: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
{xdecl}
{decl}{SSTRIDE}{META_SCALE}
// Unsigned bytes minus `c` as signed bytes.
fn cen4(v: vec4<u32>, c: u32) -> vec4<u32> {{
    return ((v | vec4<u32>(0x80808080u)) - vec4<u32>(c)) ^ vec4<u32>(0x80808080u);
}}
// 2-bit fields 0..4 of `v` in the low bits of four bytes; fields 0, 2, 4, 6; bits 0..4.
fn spread2(v: vec4<u32>) -> vec4<u32> {{
    return (v & vec4<u32>(3u)) | ((v & vec4<u32>(0xcu)) << vec4<u32>(6u))
        | ((v & vec4<u32>(0x30u)) << vec4<u32>(12u)) | ((v & vec4<u32>(0xc0u)) << vec4<u32>(18u));
}}
fn spread2e(v: vec4<u32>) -> vec4<u32> {{
    return (v & vec4<u32>(3u)) | ((v & vec4<u32>(0x30u)) << vec4<u32>(4u))
        | ((v & vec4<u32>(0x300u)) << vec4<u32>(8u)) | ((v & vec4<u32>(0x3000u)) << vec4<u32>(12u));
}}
fn spread1(v: vec4<u32>) -> vec4<u32> {{
    return (v & vec4<u32>(1u)) | ((v & vec4<u32>(2u)) << vec4<u32>(7u))
        | ((v & vec4<u32>(4u)) << vec4<u32>(14u)) | ((v & vec4<u32>(8u)) << vec4<u32>(21u));
}}
// (scale, min) of `q * scale - min` for `q - c` in place of `q`.
fn center(sm: vec2<f32>, c: f32) -> vec2<f32> {{ return vec2<f32>(sm.x, sm.y - c * sm.x); }}
fn pk(v: vec4<i32>) -> u32 {{
    let u = bitcast<vec4<u32>>(v) & vec4<u32>(255u);
    return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);
}}
// The weights' (scale, min) as computed, whatever their uses. Metal compiles with fast
// math, which may reassociate `(d * sc) * unit` (`d * sc` is exact) in one kernel and not
// in another, and a row's result would then depend on the size of its pass. A select on
// a value the kernel does not otherwise read (`p.t`, never 0) keeps the product whole.
fn keep(v: vec2<f32>) -> vec2<f32> {{ return select(v, vec2<f32>(0.0), p.t == 0u); }}
",
        decl = f.decl,
        xdecl = if inline {
            format!(
                "@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
var<workgroup> xq: array<vec4<u32>, {}>;  // [block of the step][plane][2]
var<workgroup> xs: array<vec4<f32>, 32>;",
                32 * np
            )
        } else {
            "@group(0) @binding(1) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> xs: array<vec4<f32>>;"
                .to_string()
        },
    )
    .unwrap();
    if glu {
        writeln!(w, "var<workgroup> red: array<f32, {}>;  // [8 lane groups][R][T]", 8 * rw * t).unwrap();
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
    writeln!(w, "    let nb = p.k / 32u;").unwrap();
    if inline {
        // Lanes 8 b .. 8 b + 8 round block b of the step, 4 values each: lane i ends up
        // with the block's u32 number i of each plane.
        write!(
            w,
            "    let qb = li / 8u;
    let qi = li % 8u;
    for (var c0 = 0u; c0 < nb; c0 += 32u) {{
        let v = x[(p.r0 * nb + min(c0 + qb, nb - 1u)) * 8u + qi];
        let av = abs(v);
        var amax = max(max(av.x, av.y), max(av.z, av.w));
        amax = max(amax, subgroupShuffleXor(amax, 1u));
        amax = max(amax, subgroupShuffleXor(amax, 2u));
        amax = max(amax, subgroupShuffleXor(amax, 4u));
        let inv = select(0.0, {top} / amax, amax > 0.0);
        let q = vec4<i32>(round(v * inv));
        var sv = (v.x + v.y) + (v.z + v.w);
        sv += subgroupShuffleXor(sv, 1u);
        sv += subgroupShuffleXor(sv, 2u);
        let so = subgroupShuffleXor(sv, 4u);
"
        )
        .unwrap();
        for pl in 0..planes {
            let e = match (planes, pl) {
                (1, _) => "q",
                (_, 0) => "(q >> vec4<u32>(7u))",
                _ => "(q & vec4<i32>(127))",
            };
            if f.xmode == 1 {
                // Even elements of each 8 (two lanes), then the odd ones.
                write!(
                    w,
                    "        let pa{pl} = pk({e});
        let pb{pl} = subgroupShuffleXor(pa{pl}, 1u);
        let odd{pl} = (qi & 1u) == 1u;
        let fa{pl} = select(pa{pl}, pb{pl}, odd{pl}) >> select(0u, 8u, odd{pl});
        let fb{pl} = select(pb{pl}, pa{pl}, odd{pl}) >> select(0u, 8u, odd{pl});
        let u{pl} = (fa{pl} & 255u) | ((fa{pl} >> 8u) & 0xff00u) | ((fb{pl} & 255u) << 16u) | ((fb{pl} & 0xff0000u) << 8u);
"
                )
                .unwrap();
            } else {
                writeln!(w, "        let u{pl} = pk({e});").unwrap();
            }
            // (lane `qi` holds component `qi % 4` of vector `qi / 4`: the first lane of
            // each 4 gathers and stores the whole vector, since Metal loses concurrent
            // stores to different components of one workgroup vector)
            write!(
                w,
                "        let u{pl}a = subgroupShuffleXor(u{pl}, 1u);
        let u{pl}b = subgroupShuffleXor(u{pl}, 2u);
        let u{pl}c = subgroupShuffleXor(u{pl}, 3u);
        if ((qi & 3u) == 0u) {{ xq[qb * {np}u + {}u + qi / 4u] = vec4<u32>(u{pl}, u{pl}a, u{pl}b, u{pl}c); }}
",
                2 * pl
            )
            .unwrap();
        }
        write!(
            w,
            "        if (qi == 0u) {{ xs[qb] = vec4<f32>(amax / {top}, sv, so, sv + so); }}
        workgroupBarrier();
        let kb = c0 + lane;
        if (kb < nb) {{
"
        )
        .unwrap();
    } else {
        writeln!(w, "    let xb = p.r0 * {np}u * nb;").unwrap();
        writeln!(w, "    for (var kb = lane; kb < nb; kb += 32u) {{").unwrap();
    }
    // Activations: [row][plane][2][block] of vec4<u32>, units and sums [row][block]: the
    // 32 lanes of a row read 512 contiguous bytes per load. (With the block outermost,
    // strides of 256 and 512 bytes between lanes made T = 4, 8 and 16 1.3-1.5x slower.)
    let xload = |w: &mut String, j: usize| {
        for pl in 0..planes {
            for v in 0..2 {
                if inline {
                    writeln!(w, "        let x{j}_{pl}_{v} = xq[lane * {np}u + {}u];", 2 * pl + v).unwrap();
                } else {
                    writeln!(w, "        let x{j}_{pl}_{v} = xq[xb + {}u * nb + kb];", j * np + 2 * pl + v).unwrap();
                }
            }
        }
        if inline {
            writeln!(w, "        let xv{j} = xs[lane];").unwrap();
        } else {
            writeln!(w, "        let xv{j} = xs[(p.r0 + {j}u) * nb + kb];").unwrap();
        }
    };
    // The first rows' loads go out before the weights'.
    let early = t.min(2);
    for j in 0..early {
        xload(w, j);
    }
    for a in 0..rw {
        writeln!(w, "        var ea{a}: vec4<u32>;\n        var eb{a}: vec4<u32>;").unwrap();
        for h in 0..ks {
            writeln!(w, "        var ws{a}_{h}: vec2<f32>;").unwrap();
        }
        writeln!(w, "        {{\n        let r = rw{a};\n        {}", f.load).unwrap();
        writeln!(w, "        ea{a} = ea;\n        eb{a} = eb;").unwrap();
        for h in 0..ks {
            writeln!(w, "        ws{a}_{h} = keep({});", f.scale.replace(" h)", &format!(" {h}u)"))).unwrap();
        }
        writeln!(w, "        }}").unwrap();
    }
    // The 8 u32 of the block, in the order of the activation planes.
    let e = |a: usize, i: usize| -> String {
        let c = ["x", "y", "z", "w"];
        if f.interleave {
            format!("{}{a}.{}", if i.is_multiple_of(2) { "ea" } else { "eb" }, c[i / 2])
        } else {
            format!("{}{a}.{}", if i < 4 { "ea" } else { "eb" }, c[i % 4])
        }
    };
    let steps = 8 / ks;
    for j in 0..t {
        if j >= early {
            xload(w, j);
        }
        for a in 0..rw {
            for h in 0..ks {
                let mut sums = Vec::new();
                for pl in 0..planes {
                    let terms: Vec<String> = (h * steps..(h + 1) * steps)
                        .map(|i| format!("dot4I8Packed({}, x{j}_{pl}_{}[{}])", e(a, i), i / 4, i % 4))
                        .collect();
                    writeln!(w, "        let a{a}_{j}_{h}_{pl} = {};", terms.join(" + ")).unwrap();
                    sums.push(format!("a{a}_{j}_{h}_{pl}"));
                }
                let sum = if planes == 2 {
                    format!("f32({} * 128 + {})", sums[0], sums[1])
                } else {
                    format!("f32({})", sums[0])
                };
                if f.has_min {
                    let xm = if ks == 1 {
                        "w"
                    } else if h == 0 {
                        "y"
                    } else {
                        "z"
                    };
                    writeln!(
                        w,
                        "        r{a}_{j} = fma(-ws{a}_{h}.y, xv{j}.{xm}, fma({sum}, ws{a}_{h}.x * xv{j}.x, r{a}_{j}));"
                    )
                    .unwrap();
                } else {
                    writeln!(w, "        r{a}_{j} = fma({sum}, ws{a}_{h}.x * xv{j}.x, r{a}_{j});").unwrap();
                }
            }
        }
    }
    if inline {
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
    writeln!(w, "}}").unwrap();
    s
}

/// Activation rows as int8 planes for the integer GEMV, rounded as `gemm_int`'s `xquant`
/// does (so a row has the same integers in both). One lane per row and block of 32:
/// `xq [row][plane][2][block]` of `vec4<u32>` (the block's 8 u32 in order, 4 per vector)
/// and `xs [row][block]` of `vec4<f32>` (unit, sum of the first 16 values, of the last 16,
/// of all 32). `mode` 1 stores the even elements of each 8, then the odd ones.
pub fn xquant_source(planes: usize) -> String {
    let top = if planes == 2 { "16383.0" } else { "127.0" };
    let np = 2 * planes;
    let mut s = format!(
        "struct P {{ k: u32, t: u32, mode: u32, _a: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> xq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> xs: array<vec4<f32>>;

fn pk(v: vec4<i32>) -> u32 {{
    let u = bitcast<vec4<u32>>(v) & vec4<u32>(255u);
    return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);
}}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {{
    let idx = (wg.y * nw.x + wg.x) * 64u + li;
    let row = idx / (p.k / 32u);
    if (row >= p.t) {{ return; }}
    let b = idx % (p.k / 32u);
    let src = row * (p.k / 4u) + b * 8u;
    var v: array<vec4<f32>, 8>;
    var sv: array<f32, 8>;
    var m = vec4<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {{
        v[i] = x[src + i];
        m = max(m, abs(v[i]));
        sv[i] = (v[i].x + v[i].y) + (v[i].z + v[i].w);
    }}
    let amax = max(max(m.x, m.y), max(m.z, m.w));
    let inv = select(0.0, {top} / amax, amax > 0.0);
    // (the order of these sums is also that of the GEMV's own rounding, `source`)
    let lo = (sv[0] + sv[1]) + (sv[2] + sv[3]);
    let hi = (sv[4] + sv[5]) + (sv[6] + sv[7]);
    var q: array<u32, {nq}>;
    for (var i = 0u; i < 4u; i++) {{
        let a = vec4<i32>(round(v[2u * i] * inv));
        let c = vec4<i32>(round(v[2u * i + 1u] * inv));
",
        nq = 8 * planes,
    );
    for pl in 0..planes {
        let (ea, ec) = match (planes, pl) {
            (1, _) => ("a", "c"),
            (_, 0) => ("(a >> vec4<u32>(7u))", "(c >> vec4<u32>(7u))"),
            _ => ("(a & vec4<i32>(127))", "(c & vec4<i32>(127))"),
        };
        write!(
            s,
            "        {{
            var u0 = {ea};
            var u1 = {ec};
            if (p.mode == 1u) {{
                u0 = vec4<i32>({ea}.x, {ea}.z, {ec}.x, {ec}.z);
                u1 = vec4<i32>({ea}.y, {ea}.w, {ec}.y, {ec}.w);
            }}
            q[{pl}u * 8u + 2u * i] = pk(u0);
            q[{pl}u * 8u + 2u * i + 1u] = pk(u1);
        }}
"
        )
        .unwrap();
    }
    write!(
        s,
        "    }}
    let nb = p.k / 32u;
    for (var i = 0u; i < {np}u; i++) {{
        xq[(row * {np}u + i) * nb + b] = vec4<u32>(q[4u * i], q[4u * i + 1u], q[4u * i + 2u], q[4u * i + 3u]);
    }}
    xs[row * nb + b] = vec4<f32>(amax / {top}, lo, hi, lo + hi);
}}
"
    )
    .unwrap();
    s
}
