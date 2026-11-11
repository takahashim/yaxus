//! Prefill GEMM with packed integer dot products: generated WGSL.
//!
//! The weights enter as their integer values (int8 per weight, unpacked from the stored
//! bits), the activations as int8 planes made by `xquant` per row and block of 32. One
//! `dot4I8Packed` (a single instruction on GPUs with integer dot products) does four
//! multiply-adds, against one for a float FMA, and the prefill GEMM is bound by
//! arithmetic. Per scale block the integer sums are exact; they are converted once:
//! `y += f32(sum) * w scale * x scale - w min * x sum`.
//!
//! The loops are written out here: drivers do not unroll them (the looped form of the
//! same kernel ran 1.7x slower on RADV).

use std::fmt::Write;

use crate::shaders::{W_Q2, W_Q3, W_Q4, W_Q4K, W_Q6, W_Q8};

/// A weight format the integer GEMM reads.
pub struct IntFmt {
    pub name: &'static str,
    /// WGSL declaring `wq` and `wsc` (bindings 2 and 3) and the format's helpers.
    loader: &'static str,
    /// WGSL giving `e0` and `e1`, int8 x 4 each: weights `8 c .. 8 c + 8` of block `kb` in
    /// row `wrow` (see `xmode` for their order).
    stage: &'static str,
    /// WGSL expression of (scale, min) as `vec2<f32>` for sub-block `h` of block `kb`.
    scale: &'static str,
    /// Scale sub-blocks per block of 32.
    pub ks: usize,
    /// Whether weights are `q * scale - min` with a nonzero `min`.
    has_min: bool,
    /// Order of the 8 activations of a u32 pair: 0 as stored, 1 even elements then odd.
    pub xmode: u32,
}

/// 2-bit fields `0..4` of `v` spread to the low bits of four bytes.
const SPREAD2: &str = "fn spread2(v: u32) -> u32 {
    return (v & 3u) | ((v & 0xcu) << 6u) | ((v & 0x30u) << 12u) | ((v & 0xc0u) << 18u);
}
";

pub const FORMATS: [IntFmt; 6] = [
    // 6-bit values as signed bytes: `((q | 0x80) - 32) ^ 0x80` per byte is `q - 32`. The
    // unsigned formats (`q * scale - min`) are centered the same way, with `min` reduced
    // to match: the rounding error of the activations then multiplies values around zero
    // (about half the error with Q4_K's 0..15).
    IntFmt {
        name: "q4k",
        loader: W_Q4K,
        stage: "let v = wq[(wrow * (p.k / 256u) + kb / 8u) * 36u + 4u + (kb % 8u) * 4u + c];
        let e0 = (((v & 0x0f0f0f0fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
        let e1 = ((((v >> 4u) & 0x0f0f0f0fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;",
        scale: "center(q4k_sm(wr, kb), 8.0)",
        ks: 1,
        has_min: true,
        xmode: 1,
    },
    IntFmt {
        name: "q4",
        loader: W_Q4,
        stage: "let v = wq[wrow * (p.k / 8u) + kb * 4u + c];
        let e0 = (((v & 0x0f0f0f0fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
        let e1 = ((((v >> 4u) & 0x0f0f0f0fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;",
        scale: "f16scale(wr * sstride(p.k / 32u) + kb) * vec2<f32>(1.0, 0.0)",
        ks: 1,
        has_min: false,
        xmode: 1,
    },
    IntFmt {
        name: "q8",
        loader: W_Q8,
        stage: "let o = wrow * (p.k / 4u) + kb * 8u + 2u * c;
        let e0 = wq[o];
        let e1 = wq[o + 1u];",
        scale: "f16scale(wr * sstride(p.k / 32u) + kb) * vec2<f32>(1.0, 0.0)",
        ks: 1,
        has_min: false,
        xmode: 0,
    },
    IntFmt {
        name: "q6",
        loader: W_Q6,
        stage:
            "let g = wrow * (3u * p.k / 16u) + kb * 6u;
        let lo = wq[g + c];
        let hi = wq[g + 4u + c / 2u] >> (16u * (c & 1u));
        let e0 = (((lo & 0x0f0f0f0fu) | (spread2e(hi) << 4u) | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
        let e1 = ((((lo >> 4u) & 0x0f0f0f0fu) | (spread2e(hi >> 2u) << 4u) | 0x80808080u) - 0x20202020u) ^ 0x80808080u;",
        scale: "vec2<f32>(q6scale(wr, 2u * kb + h), 0.0)",
        ks: 2,
        has_min: false,
        xmode: 1,
    },
    IntFmt {
        name: "q3",
        loader: W_Q3,
        stage: "let g = wrow * (3u * p.k / 32u) + kb * 3u;
        let lo = wq[g + c / 2u] >> (16u * (c & 1u));
        let hi = wq[g + 2u] >> (8u * c);
        let e0 = ((spread2(lo) | (spread1(hi) << 2u) | 0x80808080u) - 0x04040404u) ^ 0x80808080u;
        let e1 = ((spread2(lo >> 8u) | (spread1(hi >> 4u) << 2u) | 0x80808080u) - 0x04040404u) ^ 0x80808080u;",
        scale: "vec2<f32>(q3scale(wr, 2u * kb + h), 0.0)",
        ks: 2,
        has_min: false,
        xmode: 0,
    },
    IntFmt {
        name: "q2",
        loader: W_Q2,
        stage: "let lo = wq[wrow * (p.k / 16u) + kb * 2u + c / 2u] >> (16u * (c & 1u));
        let e0 = ((spread2(lo) | 0x80808080u) - 0x02020202u) ^ 0x80808080u;
        let e1 = ((spread2(lo >> 8u) | 0x80808080u) - 0x02020202u) ^ 0x80808080u;",
        scale: "center(unpack2x16float(wsc[wr * sstride(p.k / 16u) + 2u * kb + h]), 2.0)",
        ks: 2,
        has_min: true,
        xmode: 0,
    },
];

pub fn format(name: &str) -> Option<&'static IntFmt> {
    FORMATS.iter().find(|f| f.name == name)
}

/// Source of the kernel for `f` with lanes covering 4 outputs x `tn` tokens (a workgroup:
/// 64 outputs x `16 tn` tokens) and `planes` int8 planes per activation (1: 7 bits and a
/// sign per block of 32; 2: 15 bits, `128 hi + lo`).
///
/// The weights of a block go through workgroup memory (unpacked once, read by the 16
/// lanes of a row group); the activations are read directly, a `vec4` per 4 tokens.
/// Measured on a 780M (2B Q4_K, 20480 x 2560, 256 tokens; the f16 matrix-core GEMM takes
/// 11.4 ms): 4 x 8 lanes 4.9 ms, 4 x 4 6.4 ms, 8 x 8 slower than 4 x 8; of the 4.9 ms the
/// dot products are 2.2, staging the weights 0.4 and the barriers 0.1.
///
/// Bindings: 0 `y [T][N]` f32, 1 the planes from `xquant`, 2 and 3 the weights, 4 the
/// activation scales and sums. Immediates: N, K, T, first token row (a multiple of 4).
pub fn source(f: &IntFmt, tn: usize, planes: usize) -> String {
    assert!(tn == 2 || tn == 4 || tn == 8);
    let nt = 16 * tn; // tokens per workgroup
    let gv = tn.div_ceil(4); // vec4 of activations per lane and u32 of K
    // With 2 tokens per lane a lane uses half of a vec4 of 4 tokens.
    let (xgroup, half) = if tn == 2 {
        ("st / 2u", true)
    } else if tn == 4 {
        ("st", false)
    } else {
        ("st * 2u", false)
    };
    let pick = |e: String| if half { format!("select({e}.xy, {e}.zw, (st & 1u) == 1u)") } else { e };
    let ks = f.ks;
    let mut s = String::new();
    let w = &mut s;
    write!(
        w,
        "struct P {{ n: u32, k: u32, t: u32, r0: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> xs: array<vec4<f32>>;
{loader}
{SPREAD2}
// 2-bit fields 0, 2, 4, 6 of `v` in the low bits of four bytes.
fn spread2e(v: u32) -> u32 {{
    return (v & 3u) | ((v & 0x30u) << 4u) | ((v & 0x300u) << 8u) | ((v & 0x3000u) << 12u);
}}
fn spread1(v: u32) -> u32 {{
    return (v & 1u) | ((v & 2u) << 7u) | ((v & 4u) << 14u) | ((v & 8u) << 21u);
}}
fn f16scale(b: u32) -> f32 {{
    let s2 = unpack2x16float(wsc[b / 2u]);
    return select(s2.x, s2.y, (b & 1u) == 1u);
}}
// (scale, min) of `q * scale - min` for `q - c` in place of `q`.
fn center(sm: vec2<f32>, c: f32) -> vec2<f32> {{ return vec2<f32>(sm.x, sm.y - c * sm.x); }}
var<workgroup> wl: array<vec4<u32>, 128>;  // [8 u32 of the block][16 groups of 4 outputs]
var<workgroup> wsl: array<vec2<f32>, {wslen}>;  // [sub-block][64 outputs]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n0 = wg.x * 64u;
    let t0 = p.r0 + wg.y * {nt}u;
    let sn = li % 16u;  // the lane's 4 outputs
    let st = li / 16u;  // the lane's tokens
    // Staging: 4 lanes per weight row, 8 weights of the block each.
    let row = li / 4u;
    let c = li % 4u;
    let wrow = min(n0 + row, p.n - 1u);
    // Activations: [block][group of 4 tokens][plane][8 u32 of K], a vec4 holding the
    // u32 of the group's 4 tokens; token rows padded to 128 (`xquant`).
    let tp4 = ((p.t + 127u) / 128u) * 32u;
    let xg = t0 / 4u + {xgroup};
",
        loader = f.loader.replace("fn load4", "fn load4_unused"),
        wslen = 64 * ks,
    )
    .unwrap();
    for i in 0..4 {
        for j in 0..tn {
            writeln!(w, "    var r{i}_{j} = 0.0;").unwrap();
        }
    }
    // Per block: the lane's share of the weights and their scales into workgroup memory.
    write!(
        w,
        "    for (var kb = 0u; kb < p.k / 32u; kb++) {{
        {stage}
        wl[(2u * c) * 16u + row / 4u][row % 4u] = e0;
        wl[(2u * c + 1u) * 16u + row / 4u][row % 4u] = e1;
        if (li < {wn}u) {{
            let wr = min(n0 + li % 64u, p.n - 1u);
            let h = li / 64u;
            wsl[li] = {scale};
        }}
        workgroupBarrier();
",
        stage = f.stage,
        wn = 64 * ks,
        scale = f.scale,
    )
    .unwrap();
    // Per block and group of 4 tokens: their units, sums of the first 16 values, of the
    // last 16, of all 32.
    for v in 0..gv {
        writeln!(w, "        let xb{v} = kb * tp4 + xg + {v}u;").unwrap();
        writeln!(w, "        let xu{v} = {};", pick(format!("xs[xb{v} * 4u]"))).unwrap();
        if f.has_min && ks == 1 {
            writeln!(w, "        let xm0_{v} = {};", pick(format!("xs[xb{v} * 4u + 3u]"))).unwrap();
        } else if f.has_min {
            writeln!(w, "        let xm0_{v} = {};", pick(format!("xs[xb{v} * 4u + 1u]"))).unwrap();
            writeln!(w, "        let xm1_{v} = {};", pick(format!("xs[xb{v} * 4u + 2u]"))).unwrap();
        }
    }
    let steps = 8 / ks; // u32 of K per scale sub-block
    for h in 0..ks {
        for pl in 0..planes {
            for i in 0..4 {
                for j in 0..tn {
                    writeln!(w, "        var a{h}{pl}_{i}_{j} = 0;").unwrap();
                }
            }
        }
        for k in h * steps..(h + 1) * steps {
            writeln!(w, "        let w{k} = wl[{}u + sn];", k * 16).unwrap();
            for pl in 0..planes {
                for v in 0..gv {
                    let e = pick(format!("xq[(xb{v} * {planes}u + {pl}u) * 8u + {k}u]"));
                    writeln!(w, "        let x{k}_{pl}_{v} = {e};").unwrap();
                }
                for i in 0..4 {
                    for j in 0..tn {
                        writeln!(
                            w,
                            "        a{h}{pl}_{i}_{j} += dot4I8Packed(w{k}[{i}], x{k}_{pl}_{}[{}]);",
                            j / 4,
                            j % 4
                        )
                        .unwrap();
                    }
                }
            }
        }
        for i in 0..4 {
            writeln!(w, "        let ws{h}_{i} = wsl[{}u + sn * 4u + {i}u];", h * 64).unwrap();
            for j in 0..tn {
                let sum = if planes == 2 {
                    format!("f32(a{h}0_{i}_{j} * 128 + a{h}1_{i}_{j})")
                } else {
                    format!("f32(a{h}0_{i}_{j})")
                };
                let (v, c) = (j / 4, j % 4);
                if f.has_min {
                    writeln!(
                        w,
                        "        r{i}_{j} += {sum} * (ws{h}_{i}.x * xu{v}[{c}]) - ws{h}_{i}.y * xm{h}_{v}[{c}];"
                    )
                    .unwrap();
                } else {
                    writeln!(w, "        r{i}_{j} += {sum} * (ws{h}_{i}.x * xu{v}[{c}]);").unwrap();
                }
            }
        }
    }
    writeln!(w, "        workgroupBarrier();\n    }}").unwrap();
    for j in 0..tn {
        writeln!(w, "    if (t0 + st * {tn}u + {j}u < p.t) {{").unwrap();
        writeln!(w, "        let o = (t0 + st * {tn}u + {j}u) * p.n + n0 + sn * 4u;").unwrap();
        for i in 0..4 {
            writeln!(w, "        if (n0 + sn * 4u + {i}u < p.n) {{ y[o + {i}u] = r{i}_{j}; }}").unwrap();
        }
        writeln!(w, "    }}").unwrap();
    }
    writeln!(w, "}}").unwrap();
    s
}

/// Activation rows as int8 planes for the integer GEMM. Per row and block of 32 values:
/// with one plane, `round(x / (max|x| / 127))`; with two, `round(x / (max|x| / 16383))`
/// split as `128 hi + lo` (`lo` in 0..128), `hi` first. Packed 4 per u32; `mode` 1 stores
/// the even elements of each 8, then the odd ones (as the nibble formats unpack).
///
/// Laid out for the GEMM's loads: `xq [block][group of 4 tokens][plane][8]` of `vec4<u32>`
/// (one u32 per token of the group) and `xs [block][group][4]` of `vec4<f32>` (units, sums
/// of the first 16 values, of the last 16, of all 32), token rows padded to 128. A lane
/// handles one block of one group.
pub fn xquant_source(planes: usize) -> String {
    let top = if planes == 2 { "16383.0" } else { "127.0" };
    let nq = 8 * planes;
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
    let tp4 = ((p.t + 127u) / 128u) * 32u;
    let b = idx / tp4;
    if (b >= p.k / 32u) {{ return; }}
    let tg = idx % tp4;
    var q: array<vec4<u32>, {nq}>;
    var unit = vec4<f32>(0.0);
    var lo = vec4<f32>(0.0);
    var hi = vec4<f32>(0.0);
    for (var r = 0u; r < 4u; r++) {{
        let row = tg * 4u + r;
        if (row >= p.t) {{
            for (var i = 0u; i < {nq}u; i++) {{ q[i][r] = 0u; }}
            continue;
        }}
        let src = row * (p.k / 4u) + b * 8u;
        var v: array<vec4<f32>, 8>;
        var m = vec4<f32>(0.0);
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        for (var i = 0u; i < 4u; i++) {{
            v[i] = x[src + i];
            v[i + 4u] = x[src + i + 4u];
            m = max(m, max(abs(v[i]), abs(v[i + 4u])));
            s0 += v[i];
            s1 += v[i + 4u];
        }}
        let amax = max(max(m.x, m.y), max(m.z, m.w));
        let inv = select(0.0, {top} / amax, amax > 0.0);
        unit[r] = amax / {top};
        lo[r] = s0.x + s0.y + s0.z + s0.w;
        hi[r] = s1.x + s1.y + s1.z + s1.w;
        for (var i = 0u; i < 4u; i++) {{
            let a = vec4<i32>(round(v[2u * i] * inv));
            let c = vec4<i32>(round(v[2u * i + 1u] * inv));
"
    );
    for pl in 0..planes {
        let (ea, ec) = match (planes, pl) {
            (1, _) => ("a", "c"),
            (_, 0) => ("(a >> vec4<u32>(7u))", "(c >> vec4<u32>(7u))"),
            _ => ("(a & vec4<i32>(127))", "(c & vec4<i32>(127))"),
        };
        write!(
            s,
            "            {{
                var u0 = {ea};
                var u1 = {ec};
                if (p.mode == 1u) {{
                    u0 = vec4<i32>({ea}.x, {ea}.z, {ec}.x, {ec}.z);
                    u1 = vec4<i32>({ea}.y, {ea}.w, {ec}.y, {ec}.w);
                }}
                q[{pl}u * 8u + 2u * i][r] = pk(u0);
                q[{pl}u * 8u + 2u * i + 1u][r] = pk(u1);
            }}
"
        )
        .unwrap();
    }
    write!(
        s,
        "        }}
    }}
    let o = b * tp4 + tg;
    for (var i = 0u; i < {nq}u; i++) {{ xq[o * {nq}u + i] = q[i]; }}
    xs[o * 4u] = unit;
    xs[o * 4u + 1u] = lo;
    xs[o * 4u + 2u] = hi;
    xs[o * 4u + 3u] = lo + hi;
}}
"
    )
    .unwrap();
    s
}
