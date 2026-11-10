//! WGSL kernels. Per-dispatch scalars come through `var<immediate>` (Vulkan push constants).
//! Binding 0 is always the output buffer.

/// Weight access for `[N, K]` Q8 matrices: `load4(row, i)` returns weights `4i..4i+4`.
pub const W_Q8: &str = r#"
@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Scales of a row: `blocks` f16 values, padded when `blocks` is a multiple of 256 (see
// `pad_scales` on the host).
fn sstride(blocks: u32) -> u32 { return blocks + select(0u, {SPAD}u, blocks % 256u == 0u); }
fn load4(row: u32, i: u32) -> vec4<f32> {
    let q = bitcast<i32>(wq[row * (p.k / 4u) + i]);
    let b = row * sstride(p.k / 32u) + i / 8u;
    let s2 = unpack2x16float(wsc[b / 2u]);
    let s = select(s2.x, s2.y, (b & 1u) == 1u);
    return vec4<f32>(
        f32(extractBits(q, 0u, 8u)), f32(extractBits(q, 8u, 8u)),
        f32(extractBits(q, 16u, 8u)), f32(extractBits(q, 24u, 8u))) * s;
}
"#;

/// Weight access for `[N, K]` Q4 matrices (8 nibbles per u32, `q + 8`).
pub const W_Q4: &str = r#"
@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Scales of a row: `blocks` f16 values, padded when `blocks` is a multiple of 256 (see
// `pad_scales` on the host).
fn sstride(blocks: u32) -> u32 { return blocks + select(0u, {SPAD}u, blocks % 256u == 0u); }
fn load4(row: u32, i: u32) -> vec4<f32> {
    let u = wq[row * (p.k / 8u) + i / 2u] >> (16u * (i & 1u));
    let b = row * sstride(p.k / 32u) + i / 8u;
    let s2 = unpack2x16float(wsc[b / 2u]);
    let s = select(s2.x, s2.y, (b & 1u) == 1u);
    return (vec4<f32>(f32(u & 15u), f32((u >> 4u) & 15u), f32((u >> 8u) & 15u), f32((u >> 12u) & 15u)) - 8.0) * s;
}
"#;

/// Weight access for `[N, K]` bf16 matrices (two bf16 per u32, little endian).
pub const W_BF16: &str = r#"
@group(0) @binding(2) var<storage, read> wb: array<vec2<u32>>;
fn load4(row: u32, i: u32) -> vec4<f32> {
    let u = wb[row * (p.k / 4u) + i];
    return vec4<f32>(
        bitcast<f32>(u.x << 16u), bitcast<f32>(u.x & 0xffff0000u),
        bitcast<f32>(u.y << 16u), bitcast<f32>(u.y & 0xffff0000u));
}
"#;

pub const MM_PARAMS: &str = r#"
struct P { n: u32, k: u32, t: u32, r0: u32 } // r0: first row (multi-row GEMV only)
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
"#;

/// GEMV weight access, 16 weights per call (one 16-byte load for Q8, two for bf16).
pub const W_Q8_16: &str = r#"
@group(0) @binding(2) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Scales of a row: `blocks` f16 values, padded when `blocks` is a multiple of 256 (see
// `pad_scales` on the host).
fn sstride(blocks: u32) -> u32 { return blocks + select(0u, {SPAD}u, blocks % 256u == 0u); }
fn unpack_q8(u: u32) -> vec4<f32> {
    let q = bitcast<i32>(u);
    return vec4<f32>(f32(extractBits(q, 0u, 8u)), f32(extractBits(q, 8u, 8u)),
                     f32(extractBits(q, 16u, 8u)), f32(extractBits(q, 24u, 8u)));
}
fn load16(row: u32, j: u32) -> array<vec4<f32>, 4> {
    let u = wq[row * (p.k / 16u) + j];
    let b = row * sstride(p.k / 32u) + j / 2u;
    let s2 = unpack2x16float(wsc[b / 2u]);
    let s = select(s2.x, s2.y, (b & 1u) == 1u);
    return array<vec4<f32>, 4>(unpack_q8(u.x) * s, unpack_q8(u.y) * s, unpack_q8(u.z) * s, unpack_q8(u.w) * s);
}
"#;

pub const W_Q4_16: &str = r#"
@group(0) @binding(2) var<storage, read> wq: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Scales of a row: `blocks` f16 values, padded when `blocks` is a multiple of 256 (see
// `pad_scales` on the host).
fn sstride(blocks: u32) -> u32 { return blocks + select(0u, {SPAD}u, blocks % 256u == 0u); }
fn unpack_q4(u: u32, s: f32) -> array<vec4<f32>, 2> {
    let lo = vec4<f32>(f32(u & 15u), f32((u >> 4u) & 15u), f32((u >> 8u) & 15u), f32((u >> 12u) & 15u));
    let hi = vec4<f32>(f32((u >> 16u) & 15u), f32((u >> 20u) & 15u), f32((u >> 24u) & 15u), f32(u >> 28u));
    return array<vec4<f32>, 2>((lo - 8.0) * s, (hi - 8.0) * s);
}
fn load16(row: u32, j: u32) -> array<vec4<f32>, 4> {
    let u = wq[row * (p.k / 16u) + j];
    let b = row * sstride(p.k / 32u) + j / 2u;
    let s2 = unpack2x16float(wsc[b / 2u]);
    let s = select(s2.x, s2.y, (b & 1u) == 1u);
    let a = unpack_q4(u.x, s);
    let c = unpack_q4(u.y, s);
    return array<vec4<f32>, 4>(a[0], a[1], c[0], c[1]);
}
"#;

pub const W_BF16_16: &str = r#"
@group(0) @binding(2) var<storage, read> wb: array<vec4<u32>>;
fn bf2(u: u32) -> vec2<f32> { return vec2<f32>(bitcast<f32>(u << 16u), bitcast<f32>(u & 0xffff0000u)); }
fn load16(row: u32, j: u32) -> array<vec4<f32>, 4> {
    let i = row * (p.k / 8u) + 2u * j;
    let a = wb[i];
    let b = wb[i + 1u];
    return array<vec4<f32>, 4>(vec4<f32>(bf2(a.x), bf2(a.y)), vec4<f32>(bf2(a.z), bf2(a.w)),
                               vec4<f32>(bf2(b.x), bf2(b.y)), vec4<f32>(bf2(b.z), bf2(b.w)));
}
"#;

/// Decode GEMV (one activation row) with subgroups: 8 output rows per workgroup, 32
/// lanes per row reduced with shuffles (no workgroup memory or barriers), the
/// activation loads issued before the weight loads. Faster than `GEMV` for every
/// weight type (2B GGUF o_proj 76 -> 60 us; 0.8B Q4 draft 142 -> 148 tok/s).
pub const GEMV_T1: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let group = wg.y * nw.x + wg.x;
    let row = group * 8u + li / 32u;
    let lane = li % 32u;
    let k16 = p.k / 16u;
    // Rows past the end compute a copy of the last row (uniform control flow for the
    // shuffles) and do not store it.
    let r = min(row, p.n - 1u);
    var acc = 0.0;
    for (var j = lane; j < k16; j += 32u) {
        let xb = 4u * j;
        let x0 = x[xb];
        let x1 = x[xb + 1u];
        let x2 = x[xb + 2u];
        let x3 = x[xb + 3u];
        let w = load16(r, j);
        acc += dot(w[0], x0) + dot(w[1], x1) + dot(w[2], x2) + dot(w[3], x3);
    }
    for (var s = 16u; s > 0u; s >>= 1u) { acc += subgroupShuffleXor(acc, s); }
    if (lane == 0u && row < p.n) { y[row] = acc; }
}
"#;

/// Matrix-vector product for up to MAXT activation rows (substituted: 1 for decode,
/// 8 otherwise). 8 output rows per workgroup, 32 lanes per row, each lane loading 16
/// weights at a time, so every weight is read exactly once with wide loads.
pub const GEMV: &str = r#"
const MAXT: u32 = {MAXT}u;
var<workgroup> red: array<f32, {RED}>; // [MAXT][256]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let group = wg.y * nw.x + wg.x;
    let row = group * 8u + li / 32u;
    let lane = li % 32u;
    let k4 = p.k / 4u;
    let k16 = p.k / 16u;
    var acc: array<f32, MAXT>;
    for (var t = 0u; t < MAXT; t++) { acc[t] = 0.0; }
    if (row < p.n) {
        for (var j = lane; j < k16; j += 32u) {
            let w = load16(row, j);
            for (var t = 0u; t < MAXT; t++) {
                if (t < p.t) {
                    let xb = t * k4 + 4u * j;
                    acc[t] += dot(w[0], x[xb]) + dot(w[1], x[xb + 1u]) + dot(w[2], x[xb + 2u]) + dot(w[3], x[xb + 3u]);
                }
            }
        }
    }
    for (var t = 0u; t < MAXT; t++) { red[t * 256u + li] = acc[t]; }
    workgroupBarrier();
    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            for (var t = 0u; t < MAXT; t++) {
                red[t * 256u + li] += red[t * 256u + li + s];
            }
        }
        workgroupBarrier();
    }
    if (lane == 0u && row < p.n) {
        for (var t = 0u; t < MAXT; t++) {
            if (t < p.t) { y[t * p.n + row] = red[t * 256u + li]; }
        }
    }
}
"#;

/// Matrix product for a few activation rows (speculative verification, tiny prompts).
/// Exactly `{T}` rows. K is walked in chunks of 512: the workgroup stages that slice of all
/// T activation rows in workgroup memory once, and its 8 lane groups x `{R}` weight rows
/// all read it from there, so activations cost neither L2 traffic nor registers. Lanes of
/// a 32-wide group are summed with subgroup shuffles. Needs K % 512 == 0 and SUBGROUP.
pub const GEMV_MULTI: &str = r#"
const T: u32 = {T}u;
const R: u32 = {R}u;
var<workgroup> xs: array<vec4<f32>, {T128}>; // [T][4 m][32 lanes]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let group = wg.y * nw.x + wg.x;
    let grp = li / 32u;
    let lane = li % 32u;
    let row0 = (group * 8u + grp) * R;
    let k4 = p.k / 4u;
    let k16 = p.k / 16u;
    var acc: array<f32, {RT}>;
    for (var i = 0u; i < R * T; i++) { acc[i] = 0.0; }
    for (var c = 0u; c < k16; c += 32u) {
        for (var i = li; i < T * 128u; i += 256u) {
            let t = i / 128u;
            let o = i % 128u; // vec4 index within the chunk: unit o/4, part o%4
            xs[t * 128u + (o % 4u) * 32u + o / 4u] = x[(p.r0 + t) * k4 + c * 4u + o];
        }
        workgroupBarrier();
        var w: array<array<vec4<f32>, 4>, R>;
        for (var r = 0u; r < R; r++) {
            if (row0 + r < p.n) { w[r] = load16(row0 + r, c + lane); }
        }
        for (var t = 0u; t < T; t++) {
            let x0 = xs[t * 128u + lane];
            let x1 = xs[t * 128u + 32u + lane];
            let x2 = xs[t * 128u + 64u + lane];
            let x3 = xs[t * 128u + 96u + lane];
            for (var r = 0u; r < R; r++) {
                acc[r * T + t] += dot(w[r][0], x0) + dot(w[r][1], x1) + dot(w[r][2], x2) + dot(w[r][3], x3);
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < R * T; i++) {
        var v = acc[i];
        v += subgroupShuffleXor(v, 16u);
        v += subgroupShuffleXor(v, 8u);
        v += subgroupShuffleXor(v, 4u);
        v += subgroupShuffleXor(v, 2u);
        v += subgroupShuffleXor(v, 1u);
        acc[i] = v;
    }
    if (lane == 0u) {
        for (var r = 0u; r < R; r++) {
            if (row0 + r < p.n) {
                for (var t = 0u; t < T; t++) { y[(p.r0 + t) * p.n + row0 + r] = acc[r * T + t]; }
            }
        }
    }
}
"#;

/// `(offset + w) * x / rms(x)`, one row per workgroup.
pub const RMSNORM: &str = r#"
struct P { cols: u32, offset: f32, eps: f32, _pad: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let base = wg.x * p.cols;
    var s = 0.0;
    for (var i = li; i < p.cols; i += 256u) { let v = x[base + i]; s += v * v; }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let r = 1.0 / sqrt(red[0] / f32(p.cols) + p.eps);
    for (var i = li; i < p.cols; i += 256u) { y[base + i] = (p.offset + w[i]) * (x[base + i] * r); }
}
"#;

/// Per-head RMSNorm (offset 1) + half-split RoPE for q and k heads; writes q to `qo`
/// and k/v to each row's KV slot. One workgroup per (head, row), 128 lanes = head_dim.
/// `meta[row] = (position, kv slot, page-table offset, 0)`.
pub const QK_ROPE: &str = r#"
struct P { hq: u32, hkv: u32, eps: f32, _a: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> qo: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<f32>;
@group(0) @binding(2) var<storage, read_write> vc: array<f32>;
@group(0) @binding(3) var<storage, read> qkv: array<f32>;
@group(0) @binding(4) var<storage, read> qn: array<f32>;
@group(0) @binding(5) var<storage, read> kn: array<f32>;
@group(0) @binding(6) var<storage, read> cs: array<f32>; // [pos][64] cos, then [pos][64] sin
@group(0) @binding(7) var<storage, read> rmeta: array<vec4<u32>>;
var<workgroup> red: array<f32, 128>;
var<workgroup> v: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let t = wg.y;
    let qd = p.hq * 128u;
    let kvd = p.hkv * 128u;
    let row = t * (qd + 2u * kvd);
    let is_q = h < p.hq;
    var src = row + qd + (h - p.hq) * 128u;
    if (is_q) { src = row + h * 128u; }
    let val = qkv[src + li];
    red[li] = val * val;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st >>= 1u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let r = 1.0 / sqrt(red[0] / 128.0 + p.eps);
    var g = 0.0;
    if (is_q) { g = qn[li]; } else { g = kn[li]; }
    v[li] = (1.0 + g) * (val * r);
    workgroupBarrier();
    let m = rmeta[t];
    let pos = m.x;
    let ntab = arrayLength(&cs) / 2u;
    let j = li % 64u;
    let c = cs[pos * 64u + j];
    let s = cs[ntab + pos * 64u + j];
    var o = 0.0;
    if (li < 64u) { o = v[li] * c - v[li + 64u] * s; } else { o = v[li] * c + v[li - 64u] * s; }
    if (is_q) {
        qo[t * qd + h * 128u + li] = o;
    } else {
        let kh = h - p.hq;
        let at = m.y * kvd + kh * 128u + li;
        kc[at] = o;
        vc[at] = qkv[row + qd + kvd + kh * 128u + li];
    }
}
"#;

/// Causal (optionally sliding-window) attention with online softmax over each row's own
/// sequence: key `j` of a row lives in slot `pages[pt + j / 128] * 128 + j % 128`.
/// One workgroup per (query head, row); 128 lanes = head_dim, 128 keys per step.
pub const ATTENTION: &str = r#"
struct P { hq: u32, hkv: u32, window: u32, scale: f32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read> rmeta: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> pages: array<u32>;
var<workgroup> qs: array<vec4<f32>, 32>;
var<workgroup> sc: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
var<workgroup> kslot: array<u32, 128>; // KV slot of each key in the current step

fn slot(pt: u32, key: u32) -> u32 {
    return pages[pt + key / 128u] * 128u + key % 128u;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let t = wg.y;
    let qd = p.hq * 128u;
    let kvd = p.hkv * 128u;
    let kh = h / (p.hq / p.hkv);
    let m4 = rmeta[t];
    let pos = m4.x;
    let pt = m4.z;
    var lo = 0u;
    if (p.window > 0u && pos + 1u > p.window) { lo = pos + 1u - p.window; }
    if (li < 32u) {
        let qb = t * qd + h * 128u + li * 4u;
        qs[li] = vec4<f32>(q[qb], q[qb + 1u], q[qb + 2u], q[qb + 3u]);
    }
    workgroupBarrier();
    var m = -1e30;
    var l = 0.0;
    var acc = 0.0;
    for (var base = lo; base <= pos; base += 128u) {
        let key = base + li;
        var s = -1e30;
        if (key <= pos) {
            let ks = slot(pt, key);
            kslot[li] = ks;
            let off = (ks * kvd + kh * 128u) / 4u;
            var d = 0.0;
            for (var i = 0u; i < 32u; i++) { d += dot(qs[i], kc[off + i]); }
            s = d * p.scale;
        }
        red[li] = s;
        workgroupBarrier();
        for (var st = 64u; st > 0u; st >>= 1u) {
            if (li < st) { red[li] = max(red[li], red[li + st]); }
            workgroupBarrier();
        }
        let nm = max(m, red[0]);
        var e = 0.0;
        if (key <= pos) { e = exp(s - nm); }
        sc[li] = e;
        workgroupBarrier();
        let corr = exp(m - nm);
        l *= corr;
        acc *= corr;
        let cnt = min(128u, pos + 1u - base);
        for (var j = 0u; j < cnt; j++) {
            let w = sc[j];
            l += w;
            acc += w * vc[kslot[j] * kvd + kh * 128u + li];
        }
        m = nm;
        workgroupBarrier();
    }
    o[t * qd + h * 128u + li] = acc / l;
}
"#;

/// Tiled attention for passes with many rows (prefill): one workgroup of 128 per (tile,
/// key chunk, query head), where a tile is up to 32 consecutive rows of one sequence
/// (built by `begin_pass`). Keys go through workgroup memory in blocks of 32 and serve
/// all rows of the tile, so each key/value is read once per tile instead of once per
/// row. Online softmax per row (4 lanes per row, joined with subgroup shuffles); f32
/// throughout. With one chunk the output is written directly; with `nch > 1` each chunk
/// of `ch` keys writes `ATTN_COMBINE`'s partial results. LDS rows are padded to 33 vec4.
pub const ATTN_TILE: &str = r#"
struct P { hq: u32, hkv: u32, window: u32, scale: f32, nch: u32, ch: u32, _a: u32, _b: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> part: array<vec4<f32>>; // [row][hkv][nch][G][32]
@group(0) @binding(2) var<storage, read_write> ml: array<vec2<f32>>;   // [row][hkv][nch][G]
@group(0) @binding(3) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> vc: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> rmeta: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> pages: array<u32>;
@group(0) @binding(8) var<storage, read> tiles: array<vec4<u32>>;

const BM: u32 = 32u;
const BN: u32 = 32u;
const LD: u32 = 33u;
var<workgroup> qs: array<vec4<f32>, 1056>;  // [BM][LD] queries, pre-scaled
var<workgroup> kv: array<vec4<f32>, 1056>;  // [BN][LD] keys, then values
var<workgroup> ss: array<f32, 1056>;        // [BM][33] scores, then probabilities
var<workgroup> rm: array<f32, 32>;          // running max per row
var<workgroup> rl: array<f32, 32>;          // running sum per row
var<workgroup> rc: array<f32, 32>;          // rescale of the accumulator for this block
var<workgroup> kslot: array<u32, 32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let tile = tiles[wg.x / p.nch];
    let c = wg.x % p.nch;
    let r0 = tile.x;
    let nr = tile.y;
    let h = wg.y;
    let g = p.hq / p.hkv;
    let kh = h / g;
    let qd4 = p.hq * 32u;
    let kvd = p.hkv * 128u;
    let first = rmeta[r0];
    let pt = first.z;
    let pos0 = first.x;
    let last = pos0 + nr - 1u;
    var lo = 0u;
    if (p.window > 0u && pos0 + 1u > p.window) { lo = pos0 + 1u - p.window; }
    let kbeg = lo + c * p.ch;
    let kend = min(last, kbeg + p.ch - 1u);
    // Partial results of row r of the tile (nch > 1).
    let pbase = ((r0 * p.hkv + kh) * p.nch + c) * g + h % g;
    let pstride = p.hkv * p.nch * g;
    if (kbeg > last) {
        // A chunk past the tile's keys (uniform per workgroup).
        if (li < nr) { ml[pbase + li * pstride] = vec2<f32>(-1e30, 0.0); }
        return;
    }
    for (var i = li; i < BM * 32u; i += 128u) {
        let r = i / 32u;
        let cc = i % 32u;
        var v = vec4<f32>(0.0);
        if (r < nr) { v = q[(r0 + r) * qd4 + h * 32u + cc] * p.scale; }
        qs[r * LD + cc] = v;
    }
    if (li < BM) {
        rm[li] = -1e30;
        rl[li] = 0.0;
    }
    // Scores: rows 2*tr, 2*tr+1 and keys tc + 8*j. Softmax: row li/4, keys 8*(li%4)...
    // Output: rows 8*tp .. 8*tp+7, dims 4*td.
    let tc = li % 8u;
    let tr = li / 8u;
    let td = li % 32u;
    let tp = li / 32u;
    let sr = li / 4u;
    let sq = li % 4u;
    var acc: array<vec4<f32>, 8>;
    for (var k = 0u; k < 8u; k++) { acc[k] = vec4<f32>(0.0); }
    for (var base = kbeg; base <= kend; base += BN) {
        workgroupBarrier();
        if (li < BN) {
            let key = base + li;
            var sl = 0u;
            if (key <= kend) { sl = pages[pt + key / 128u] * 128u + key % 128u; }
            kslot[li] = sl;
        }
        workgroupBarrier();
        for (var i = li; i < BN * 32u; i += 128u) {
            let j = i / 32u;
            let cc = i % 32u;
            var v = vec4<f32>(0.0);
            if (base + j <= kend) { v = kc[(kslot[j] * kvd + kh * 128u) / 4u + cc]; }
            kv[j * LD + cc] = v;
        }
        workgroupBarrier();
        var sc: array<f32, 8>;
        for (var k = 0u; k < 8u; k++) { sc[k] = 0.0; }
        for (var cc = 0u; cc < 32u; cc++) {
            let q0 = qs[(2u * tr) * LD + cc];
            let q1 = qs[(2u * tr + 1u) * LD + cc];
            for (var j = 0u; j < 4u; j++) {
                let kk = kv[(tc + 8u * j) * LD + cc];
                sc[j] += dot(q0, kk);
                sc[4u + j] += dot(q1, kk);
            }
        }
        for (var a = 0u; a < 2u; a++) {
            let r = 2u * tr + a;
            let pr = pos0 + r;
            var lr = 0u;
            if (p.window > 0u && pr + 1u > p.window) { lr = pr + 1u - p.window; }
            for (var j = 0u; j < 4u; j++) {
                let key = base + tc + 8u * j;
                let ok = r < nr && key <= pr && key >= lr && key <= kend;
                ss[r * 33u + tc + 8u * j] = select(-1e30, sc[4u * a + j], ok);
            }
        }
        workgroupBarrier();
        var mx = -1e30;
        for (var j = 0u; j < 8u; j++) { mx = max(mx, ss[sr * 33u + 8u * sq + j]); }
        mx = max(mx, subgroupShuffleXor(mx, 1u));
        mx = max(mx, subgroupShuffleXor(mx, 2u));
        let mo = rm[sr];
        let mn = max(mo, mx);
        var l = 0.0;
        for (var j = 0u; j < 8u; j++) {
            let at = sr * 33u + 8u * sq + j;
            let sv = ss[at];
            let e = select(0.0, exp(sv - mn), sv > -1e29);
            ss[at] = e;
            l += e;
        }
        l += subgroupShuffleXor(l, 1u);
        l += subgroupShuffleXor(l, 2u);
        // All lanes of the row have read rm[sr] before it changes.
        workgroupBarrier();
        if (sq == 0u) {
            let corr = exp(mo - mn);
            rm[sr] = mn;
            rl[sr] = rl[sr] * corr + l;
            rc[sr] = corr;
        }
        for (var i = li; i < BN * 32u; i += 128u) {
            let j = i / 32u;
            let cc = i % 32u;
            var v = vec4<f32>(0.0);
            if (base + j <= kend) { v = vc[(kslot[j] * kvd + kh * 128u) / 4u + cc]; }
            kv[j * LD + cc] = v;
        }
        workgroupBarrier();
        for (var k = 0u; k < 8u; k++) { acc[k] *= rc[8u * tp + k]; }
        for (var j = 0u; j < BN; j++) {
            let v = kv[j * LD + td];
            for (var k = 0u; k < 8u; k++) { acc[k] += ss[(8u * tp + k) * 33u + j] * v; }
        }
    }
    for (var k = 0u; k < 8u; k++) {
        let r = 8u * tp + k;
        if (r < nr) {
            if (p.nch == 1u) {
                o[(r0 + r) * qd4 + h * 32u + td] = acc[k] / rl[r];
            } else {
                let pi = pbase + r * pstride;
                part[pi * 32u + td] = acc[k];
                if (td == 0u) { ml[pi] = vec2<f32>(rm[r], rl[r]); }
            }
        }
    }
}
"#;

pub const SWIGLU: &str = r#"
struct P { inter: u32, total: u32, _a: u32, _b: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let idx = gid.y * nw.x * 256u + gid.x;
    if (idx >= p.total) { return; }
    let t = idx / p.inter;
    let j = idx % p.inter;
    let g = x[t * 2u * p.inter + j];
    y[idx] = g / (1.0 + exp(-g)) * x[t * 2u * p.inter + p.inter + j];
}
"#;

pub const ADD: &str = r#"
struct P { total: u32, _a: u32, _b: u32, _c: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let idx = gid.y * nw.x * 256u + gid.x;
    if (idx < p.total) { a[idx] += b[idx]; }
}
"#;

/// Fused residual step: `x += (o1 + w1) * norm(d)`, then `h = (o2 + w2) * norm(x)`.
/// One row per workgroup; each thread only re-reads the elements it wrote itself.
pub const ADD_NORM: &str = r#"
struct P { cols: u32, o1: f32, o2: f32, eps: f32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> d: array<f32>;
@group(0) @binding(3) var<storage, read> w1: array<f32>;
@group(0) @binding(4) var<storage, read> w2: array<f32>;
var<workgroup> red: array<f32, 256>;

fn sum256(li: u32, v: f32) -> f32 {
    red[li] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let base = wg.x * p.cols;
    var s = 0.0;
    for (var i = li; i < p.cols; i += 256u) { let v = d[base + i]; s += v * v; }
    let r1 = 1.0 / sqrt(sum256(li, s) / f32(p.cols) + p.eps);
    var s2 = 0.0;
    for (var i = li; i < p.cols; i += 256u) {
        let v = x[base + i] + (p.o1 + w1[i]) * (d[base + i] * r1);
        x[base + i] = v;
        s2 += v * v;
    }
    let r2 = 1.0 / sqrt(sum256(li, s2) / f32(p.cols) + p.eps);
    for (var i = li; i < p.cols; i += 256u) { h[base + i] = (p.o2 + w2[i]) * (x[base + i] * r2); }
}
"#;

/// Decode-time (T=1) gate/up projection fused with SwiGLU. A workgroup computes 4 outputs:
/// lane groups 0-3 the gate rows `u`, groups 4-7 the matching up rows `n + u`
/// (`p.n` is the intermediate size), then `y[u] = silu(gate) * up`.
pub const GEMV_GLU: &str = r#"
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let group = wg.y * nw.x + wg.x;
    let g = li / 32u;
    let lane = li % 32u;
    let u = group * 4u + (g % 4u);
    let row = select(u, p.n + u, g >= 4u);
    let k4 = p.k / 4u;
    let k16 = p.k / 16u;
    var acc = 0.0;
    if (u < p.n) {
        for (var j = lane; j < k16; j += 32u) {
            let w = load16(row, j);
            let xb = 4u * j;
            acc += dot(w[0], x[xb]) + dot(w[1], x[xb + 1u]) + dot(w[2], x[xb + 2u]) + dot(w[3], x[xb + 3u]);
        }
    }
    red[li] = acc;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) { red[li] += red[li + s]; }
        workgroupBarrier();
    }
    if (li < 4u && group * 4u + li < p.n) {
        let gt = red[li * 32u];
        let up = red[(li + 4u) * 32u];
        y[group * 4u + li] = gt / (1.0 + exp(-gt)) * up;
    }
}
"#;

/// Argmax, stage 1: each workgroup scans one slice of one row and writes
/// (max bits, index of the max, sum of exp(v - max) bits, 0).
pub const ARGMAX1: &str = r#"
struct P { n: u32, slice: u32, _a: u32, _b: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> part: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
var<workgroup> bs: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let row = wg.y;
    let lo = wg.x * p.slice;
    let hi = min(lo + p.slice, p.n);
    var best = -3.0e38;
    var idx = 0xffffffffu;
    var sum = 0.0;
    for (var i = lo + li; i < hi; i += 256u) {
        let v = x[row * p.n + i];
        if (v > best) { sum = sum * exp(best - v) + 1.0; best = v; idx = i; } else { sum += exp(v - best); }
    }
    bv[li] = best;
    bi[li] = idx;
    bs[li] = sum;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) {
            let ov = bv[li + st];
            let oi = bi[li + st];
            let os = bs[li + st];
            let mv = max(bv[li], ov);
            bs[li] = bs[li] * exp(bv[li] - mv) + os * exp(ov - mv);
            if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
        }
        workgroupBarrier();
    }
    if (li == 0u) { part[row * nw.x + wg.x] = vec4<u32>(bitcast<u32>(bv[0]), bi[0], bitcast<u32>(bs[0]), 0u); }
}
"#;

/// Argmax, stage 2: one workgroup per row reduces the per-slice results and writes
/// (index, bits of the softmax probability of that index).
pub const ARGMAX2: &str = r#"
struct P { slices: u32, _a: u32, _b: u32, _c: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read> part: array<vec4<u32>>;
var<workgroup> bv: array<f32, 64>;
var<workgroup> bi: array<u32, 64>;
var<workgroup> bs: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    var best = -3.0e38;
    var idx = 0xffffffffu;
    var sum = 0.0;
    if (li < p.slices) {
        let e = part[wg.x * p.slices + li];
        best = bitcast<f32>(e.x);
        idx = e.y;
        sum = bitcast<f32>(e.z);
    }
    bv[li] = best;
    bi[li] = idx;
    bs[li] = sum;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
        if (li < st) {
            let ov = bv[li + st];
            let oi = bi[li + st];
            let os = bs[li + st];
            let mv = max(bv[li], ov);
            bs[li] = bs[li] * exp(bv[li] - mv) + os * exp(ov - mv);
            if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        out[2u * wg.x] = bi[0];
        out[2u * wg.x + 1u] = bitcast<u32>(1.0 / bs[0]);
    }
}
"#;

/// Prefill GEMM with a 64 (tokens) x 128 (outputs) tile and 4 x 8 outputs per thread,
/// K in steps of 16. Per k a thread reads 3 vec4 from workgroup memory for 32 FMAs
/// (the 64x64 kernel read 2 for 16, and was bound by workgroup-memory bandwidth).
/// f32 throughout; the fallback when cooperative matrices are unavailable (an f16 variant
/// of this kernel, and a dot-product layout, were slower on RADV and are gone).
pub const GEMM2: &str = r#"
var<workgroup> xs: array<vec4<f32>, 256>; // [16 k][16 groups of 4 tokens]
var<workgroup> ws: array<vec4<f32>, 512>; // [16 k][32 groups of 4 outputs]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n0 = wg.x * 128u;
    let t0 = wg.y * 64u;
    let tx = li % 16u;
    let ty = li / 16u;
    let k4 = p.k / 4u;
    var acc: array<vec4<f32>, 8>; // [4 tokens][2 halves]
    for (var i = 0u; i < 8u; i++) { acc[i] = vec4<f32>(0.0); }
    for (var kb = 0u; kb < p.k; kb += 16u) {
        {
            let t = li / 4u;
            let c = li % 4u;
            var v = vec4<f32>(0.0);
            if (t0 + t < p.t) { v = x[(t0 + t) * k4 + kb / 4u + c]; }
            for (var e = 0u; e < 4u; e++) { xs[(c * 4u + e) * 16u + t / 4u][t % 4u] = v[e]; }
        }
        for (var r = 0u; r < 2u; r++) {
            let idx = li + 256u * r;
            let n = idx / 4u;
            let c = idx % 4u;
            var w = vec4<f32>(0.0);
            if (n0 + n < p.n) { w = load4(n0 + n, kb / 4u + c); }
            for (var e = 0u; e < 4u; e++) { ws[(c * 4u + e) * 32u + n / 4u][n % 4u] = w[e]; }
        }
        workgroupBarrier();
        // Accumulating each 16-step slice separately and then adding it is measurably
        // faster on RADV than updating `acc` in place (285 vs 188 tok/s at 512 tokens).
        var part: array<vec4<f32>, 8>;
        for (var i = 0u; i < 8u; i++) { part[i] = vec4<f32>(0.0); }
        for (var kk = 0u; kk < 16u; kk++) {
            let a = xs[kk * 16u + ty];
            let b0 = ws[kk * 32u + tx];
            let b1 = ws[kk * 32u + 16u + tx];
            for (var i = 0u; i < 4u; i++) {
                part[i * 2u] = fma(vec4<f32>(a[i]), b0, part[i * 2u]);
                part[i * 2u + 1u] = fma(vec4<f32>(a[i]), b1, part[i * 2u + 1u]);
            }
        }
        for (var i = 0u; i < 8u; i++) { acc[i] += part[i]; }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i++) {
        let t = t0 + ty * 4u + i;
        if (t < p.t) {
            for (var h = 0u; h < 2u; h++) {
                for (var j = 0u; j < 4u; j++) {
                    let n = n0 + h * 64u + tx * 4u + j;
                    if (n < p.n) { y[t * p.n + n] = acc[i * 2u + h][j]; }
                }
            }
        }
    }
}
"#;

/// Prefill GEMM on the matrix cores (`VK_KHR_cooperative_matrix`, WMMA on RDNA3):
/// f16 operands, f32 accumulation. A workgroup of 128 invocations computes 64 tokens x 64
/// outputs: it stages the 64x32 activation slice and the 64x32 weight slice (dequantized)
/// as f16, and each subgroup multiplies 16-wide output stripes with four 16x16
/// accumulators (64 tokens). With 4 subgroups (wave32) each takes one stripe, with 2
/// (wave64) two. Cooperative ops need uniform control flow, so the subgroup count is
/// broadcast through workgroup memory. Needs N % 64 == 0; `y` needs room for the token
/// count rounded up to 64 rows.
pub const GEMM_COOP: &str = r#"
var<workgroup> xs: array<f16, 2048>;  // [64 tokens][32 k], row-major
var<workgroup> wsm: array<f16, 2048>; // [64 outputs][32 k]
var<workgroup> nsub: u32;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_id) sg: u32, @builtin(num_subgroups) ns: u32) {
    if (li == 0u) { nsub = ns; }
    let two = workgroupUniformLoad(&nsub) == 2u;
    let n0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    let k4 = p.k / 4u;
    let s0 = sg;       // first stripe of this subgroup within the tile
    let s1 = sg + 2u;  // second stripe (only with two subgroups)
    var c0 = coop_mat16x16<f32, C>();
    var c1 = coop_mat16x16<f32, C>();
    var c2 = coop_mat16x16<f32, C>();
    var c3 = coop_mat16x16<f32, C>();
    var d0 = coop_mat16x16<f32, C>();
    var d1 = coop_mat16x16<f32, C>();
    var d2 = coop_mat16x16<f32, C>();
    var d3 = coop_mat16x16<f32, C>();
    for (var kb = 0u; kb < p.k; kb += 32u) {
        for (var r = 0u; r < 4u; r++) {
            let idx = li + 128u * r; // vec4 index in a 64 x 8 slice
            let row = idx / 8u;
            let c = idx % 8u;
            var v = vec4<f32>(0.0);
            if (t0 + row < p.t) { v = x[(t0 + row) * k4 + kb / 4u + c]; }
            let w = load4(n0 + row, kb / 4u + c);
            let o = row * 32u + c * 4u;
            xs[o] = f16(v.x);
            xs[o + 1u] = f16(v.y);
            xs[o + 2u] = f16(v.z);
            xs[o + 3u] = f16(v.w);
            wsm[o] = f16(w.x);
            wsm[o + 1u] = f16(w.y);
            wsm[o + 2u] = f16(w.z);
            wsm[o + 3u] = f16(w.w);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk += 16u) {
            // Indices are bound to `let`s: naga 30 does not emit index expressions written
            // inline in cooperative load/store pointers.
            let i0 = kk;
            let i1 = 512u + kk;
            let i2 = 1024u + kk;
            let i3 = 1536u + kk;
            let ib = s0 * 512u + kk;
            let ib2 = s1 * 512u + kk;
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&xs[i0], 32u);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&xs[i1], 32u);
            let a2 = coopLoadT<coop_mat16x16<f16, A>>(&xs[i2], 32u);
            let a3 = coopLoadT<coop_mat16x16<f16, A>>(&xs[i3], 32u);
            // B[k][n] = W[n][k]: column-major with stride 32.
            let b = coopLoad<coop_mat16x16<f16, B>>(&wsm[ib], 32u);
            c0 = coopMultiplyAdd(a0, b, c0);
            c1 = coopMultiplyAdd(a1, b, c1);
            c2 = coopMultiplyAdd(a2, b, c2);
            c3 = coopMultiplyAdd(a3, b, c3);
            if (two) {
                let b2 = coopLoad<coop_mat16x16<f16, B>>(&wsm[ib2], 32u);
                d0 = coopMultiplyAdd(a0, b2, d0);
                d1 = coopMultiplyAdd(a1, b2, d1);
                d2 = coopMultiplyAdd(a2, b2, d2);
                d3 = coopMultiplyAdd(a3, b2, d3);
            }
        }
        workgroupBarrier();
    }
    // naga 30 does not emit the loads of a `var` or of `p.n` used directly as
    // coopStore arguments ("Expression is not cached"), so bind them to `let`s first.
    let ldy = p.n;
    let col0 = n0 + s0 * 16u;
    let col1 = n0 + s1 * 16u;
    let o0 = t0 * ldy + col0;
    let o1 = (t0 + 16u) * ldy + col0;
    let o2 = (t0 + 32u) * ldy + col0;
    let o3 = (t0 + 48u) * ldy + col0;
    let q0 = t0 * ldy + col1;
    let q1 = (t0 + 16u) * ldy + col1;
    let q2 = (t0 + 32u) * ldy + col1;
    let q3 = (t0 + 48u) * ldy + col1;
    let r0 = c0;
    let r1 = c1;
    let r2 = c2;
    let r3 = c3;
    let e0 = d0;
    let e1 = d1;
    let e2 = d2;
    let e3 = d3;
    coopStoreT(r0, &y[o0], ldy);
    coopStoreT(r1, &y[o1], ldy);
    coopStoreT(r2, &y[o2], ldy);
    coopStoreT(r3, &y[o3], ldy);
    if (two) {
        coopStoreT(e0, &y[q0], ldy);
        coopStoreT(e1, &y[q1], ldy);
        coopStoreT(e2, &y[q2], ldy);
        coopStoreT(e3, &y[q3], ldy);
    }
}
"#;

/// Embedding gather from the bf16 table (two values per u32): one workgroup per row.
pub const EMBED: &str = r#"
struct P { h: u32, _a: u32, _b: u32, _c: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h2 = p.h / 2u;
    let src = ids[wg.x] * h2;
    let dst = wg.x * p.h;
    for (var i = li; i < h2; i += 256u) {
        let u = table[src + i];
        y[dst + 2u * i] = bitcast<f32>(u << 16u);
        y[dst + 2u * i + 1u] = bitcast<f32>(u & 0xffff0000u);
    }
}
"#;


/// Weight access for `[N, K]` Q4K matrices (GGUF Q4_K kept as is): nibbles as in Q4,
/// per-256 headers in `wsc`.
pub const W_Q4K: &str = r#"
@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Scale and min of 32-block `sub` of a row from its Q4_K header (4 u32 per 256: d and
// dmin as f16, then 12 bytes of packed 6-bit scales and mins), as ggml computes them.
fn hbyte(m: u32, i: u32) -> u32 { return (wsc[m + 1u + i / 4u] >> (8u * (i % 4u))) & 255u; }
// Headers of a row, padded by one when their count is a multiple of 32 (`pad_q4k_meta`).
fn q4k_sm(row: u32, sub: u32) -> vec2<f32> {
    let hb = p.k / 256u;
    let m = (row * (hb + select(0u, 1u, hb % 32u == 0u)) + sub / 8u) * 4u;
    let dd = unpack2x16float(wsc[m]);
    let j = sub % 8u;
    let lo = j < 4u;
    let a = hbyte(m, select(j - 4u, j, lo));        // q[j] or q[j - 4]
    let b = hbyte(m, j + 4u);                        // q[j + 4]
    let c = hbyte(m, select(j, j, lo));              // q[j]
    let sc = select((b & 15u) | ((a >> 6u) << 4u), a & 63u, lo);
    let mn = select((b >> 4u) | ((c >> 6u) << 4u), b & 63u, lo);
    return vec2<f32>(dd.x * f32(sc), dd.y * f32(mn));
}
fn load4(row: u32, i: u32) -> vec4<f32> {
    let u = wq[row * (p.k / 8u) + i / 2u] >> (16u * (i & 1u));
    let sm = q4k_sm(row, i / 8u);
    return vec4<f32>(f32(u & 15u), f32((u >> 4u) & 15u), f32((u >> 8u) & 15u), f32((u >> 12u) & 15u)) * sm.x - sm.y;
}
"#;

pub const W_Q4K_16: &str = r#"
@group(0) @binding(2) var<storage, read> wq: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Scale and min of 32-block `sub` of a row from its Q4_K header (4 u32 per 256: d and
// dmin as f16, then 12 bytes of packed 6-bit scales and mins), as ggml computes them.
fn hbyte(m: u32, i: u32) -> u32 { return (wsc[m + 1u + i / 4u] >> (8u * (i % 4u))) & 255u; }
// Headers of a row, padded by one when their count is a multiple of 32 (`pad_q4k_meta`).
fn q4k_sm(row: u32, sub: u32) -> vec2<f32> {
    let hb = p.k / 256u;
    let m = (row * (hb + select(0u, 1u, hb % 32u == 0u)) + sub / 8u) * 4u;
    let dd = unpack2x16float(wsc[m]);
    let j = sub % 8u;
    let lo = j < 4u;
    let a = hbyte(m, select(j - 4u, j, lo));        // q[j] or q[j - 4]
    let b = hbyte(m, j + 4u);                        // q[j + 4]
    let c = hbyte(m, select(j, j, lo));              // q[j]
    let sc = select((b & 15u) | ((a >> 6u) << 4u), a & 63u, lo);
    let mn = select((b >> 4u) | ((c >> 6u) << 4u), b & 63u, lo);
    return vec2<f32>(dd.x * f32(sc), dd.y * f32(mn));
}
fn unpack_q4k(u: u32, sm: vec2<f32>) -> array<vec4<f32>, 2> {
    let lo = vec4<f32>(f32(u & 15u), f32((u >> 4u) & 15u), f32((u >> 8u) & 15u), f32((u >> 12u) & 15u));
    let hi = vec4<f32>(f32((u >> 16u) & 15u), f32((u >> 20u) & 15u), f32((u >> 24u) & 15u), f32(u >> 28u));
    return array<vec4<f32>, 2>(lo * sm.x - sm.y, hi * sm.x - sm.y);
}
fn load16(row: u32, j: u32) -> array<vec4<f32>, 4> {
    let u = wq[row * (p.k / 16u) + j];
    let sm = q4k_sm(row, j / 2u);
    let a = unpack_q4k(u.x, sm);
    let c = unpack_q4k(u.y, sm);
    return array<vec4<f32>, 4>(a[0], a[1], c[0], c[1]);
}
"#;

/// Weight access for `[N, K]` Q6 matrices (GGUF Q6_K): 6 u32 per 32 values (4 of low
/// nibbles, 2 of high 2-bit fields); one f16 scale per 16 values.
pub const W_Q6: &str = r#"
@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Per-row scale blocks, padded when their count is a multiple of 256 (`pad_scales`).
fn sstride(blocks: u32) -> u32 { return blocks + select(0u, {SPAD}u, blocks % 256u == 0u); }
fn q6scale(row: u32, blk: u32) -> f32 {
    let b = row * sstride(p.k / 16u) + blk;
    let s2 = unpack2x16float(wsc[b / 2u]);
    return select(s2.x, s2.y, (b & 1u) == 1u);
}
fn q6x4(lo: u32, hi: u32) -> vec4<f32> {
    let q = (vec4<u32>(lo, lo >> 4u, lo >> 8u, lo >> 12u) & vec4<u32>(15u))
          | ((vec4<u32>(hi, hi >> 2u, hi >> 4u, hi >> 6u) & vec4<u32>(3u)) << vec4<u32>(4u));
    return vec4<f32>(q) - 32.0;
}
fn load4(row: u32, i: u32) -> vec4<f32> {
    let g = row * (3u * p.k / 16u) + (i / 8u) * 6u;
    let lo = wq[g + (i % 8u) / 2u] >> (16u * (i & 1u));
    let hi = wq[g + 4u + (i % 8u) / 4u] >> (8u * (i & 3u));
    return q6x4(lo, hi) * q6scale(row, i / 4u);
}
"#;

pub const W_Q6_16: &str = r#"
@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> wsc: array<u32>;
// Per-row scale blocks, padded when their count is a multiple of 256 (`pad_scales`).
fn sstride(blocks: u32) -> u32 { return blocks + select(0u, {SPAD}u, blocks % 256u == 0u); }
fn q6scale(row: u32, blk: u32) -> f32 {
    let b = row * sstride(p.k / 16u) + blk;
    let s2 = unpack2x16float(wsc[b / 2u]);
    return select(s2.x, s2.y, (b & 1u) == 1u);
}
fn q6x4(lo: u32, hi: u32) -> vec4<f32> {
    let q = (vec4<u32>(lo, lo >> 4u, lo >> 8u, lo >> 12u) & vec4<u32>(15u))
          | ((vec4<u32>(hi, hi >> 2u, hi >> 4u, hi >> 6u) & vec4<u32>(3u)) << vec4<u32>(4u));
    return vec4<f32>(q) - 32.0;
}
fn load16(row: u32, j: u32) -> array<vec4<f32>, 4> {
    let g = row * (3u * p.k / 16u) + (j / 2u) * 6u;
    let l0 = wq[g + 2u * (j & 1u)];
    let l1 = wq[g + 2u * (j & 1u) + 1u];
    let h = wq[g + 4u + (j & 1u)];
    let s = q6scale(row, j);
    return array<vec4<f32>, 4>(q6x4(l0, h) * s, q6x4(l0 >> 16u, h >> 8u) * s, q6x4(l1, h >> 16u) * s, q6x4(l1 >> 16u, h >> 24u) * s);
}
"#;


/// Split-key attention for passes with few rows (decode, verification), part 1 of 2.
/// One workgroup of 128 per (row, KV head, chunk of 64 keys): it handles all
/// `G = hq / hkv` query heads sharing the KV head, so each key/value is read once, and
/// writes per head the chunk's max score, sum of exp and unnormalized output for
/// `ATTN_COMBINE`. Scores: two lanes per key (half the dimensions each, joined with a
/// subgroup shuffle); values: one lane per dimension accumulating every head. `G <= 16`.
pub const ATTN_SPLIT: &str = r#"
struct P { hq: u32, hkv: u32, window: u32, scale: f32, nch: u32, _a: u32, _b: u32, _c: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> part: array<f32>;  // [row][hkv][nch][G][128]
@group(0) @binding(1) var<storage, read_write> ml: array<vec2<f32>>; // [row][hkv][nch][G]
@group(0) @binding(2) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> vc: array<f32>;
@group(0) @binding(5) var<storage, read> rmeta: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> pages: array<u32>;

const G: u32 = {G}u;
const CH: u32 = {CH}u;          // keys per chunk
const LPK: u32 = 128u / CH;     // lanes per key (scores)
const DV4: u32 = 32u / LPK;     // vec4 of the key per lane

var<workgroup> qs: array<vec4<f32>, 512>;  // [G][32]
var<workgroup> sc: array<f32, {GCH}>;      // [G][CH] scores, then exp(score - max)
var<workgroup> kslot: array<u32, CH>;
var<workgroup> hm: array<f32, 16>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let g = G;
    let kh = wg.x % p.hkv;
    let c = wg.x / p.hkv;
    let t = wg.y;
    let kvd = p.hkv * 128u;
    let m4 = rmeta[t];
    let pos = m4.x;
    let pt = m4.z;
    var lo = 0u;
    if (p.window > 0u && pos + 1u > p.window) { lo = pos + 1u - p.window; }
    let start = lo + c * CH;
    let base = (((t * p.hkv + kh) * p.nch) + c) * g;
    // Uniform per workgroup: chunks past this row's context only mark themselves empty.
    if (start > pos) {
        if (li < g) { ml[base + li] = vec2<f32>(-1e30, 0.0); }
        return;
    }
    for (var i = li; i < g * 32u; i += 128u) {
        qs[i] = q[(t * p.hq + kh * g) * 32u + i];
    }
    workgroupBarrier();
    let j = li / LPK;
    let sub = li % LPK;
    let key = start + j;
    let live = key <= pos;
    var d: array<f32, G>;
    for (var h = 0u; h < G; h++) { d[h] = 0.0; }
    var ks = 0u;
    if (live) {
        ks = pages[pt + key / 128u] * 128u + key % 128u;
        let off = (ks * kvd + kh * 128u) / 4u + sub * DV4;
        for (var i = 0u; i < DV4; i++) {
            let kv = kc[off + i];
            for (var h = 0u; h < G; h++) { d[h] += dot(qs[h * 32u + sub * DV4 + i], kv); }
        }
    }
    for (var h = 0u; h < G; h++) {
        var full = d[h];
        for (var s = 1u; s < LPK; s <<= 1u) { full += subgroupShuffleXor(full, s); }
        if (sub == 0u) { sc[h * CH + j] = select(-1e30, full * p.scale, live); }
    }
    if (sub == 0u) { kslot[j] = ks; }
    workgroupBarrier();
    let n = min(CH, pos + 1u - start);
    if (li < g) {
        var mx = -1e30;
        for (var k = 0u; k < n; k++) { mx = max(mx, sc[li * CH + k]); }
        hm[li] = mx;
    }
    workgroupBarrier();
    for (var i = li; i < g * CH; i += 128u) {
        sc[i] = select(0.0, exp(sc[i] - hm[i / CH]), i % CH < n);
    }
    workgroupBarrier();
    if (li < g) {
        var l = 0.0;
        for (var k = 0u; k < n; k++) { l += sc[li * CH + k]; }
        ml[base + li] = vec2<f32>(hm[li], l);
    }
    var acc: array<f32, G>;
    for (var h = 0u; h < G; h++) { acc[h] = 0.0; }
    let vo = kh * 128u + li;
    var k = 0u;
    // Four independent loads in flight per step.
    for (; k + 4u <= n; k += 4u) {
        let v0 = vc[kslot[k] * kvd + vo];
        let v1 = vc[kslot[k + 1u] * kvd + vo];
        let v2 = vc[kslot[k + 2u] * kvd + vo];
        let v3 = vc[kslot[k + 3u] * kvd + vo];
        for (var h = 0u; h < G; h++) {
            let w = h * CH + k;
            acc[h] += sc[w] * v0 + sc[w + 1u] * v1 + sc[w + 2u] * v2 + sc[w + 3u] * v3;
        }
    }
    for (; k < n; k++) {
        let v = vc[kslot[k] * kvd + vo];
        for (var h = 0u; h < G; h++) { acc[h] += sc[h * CH + k] * v; }
    }
    for (var h = 0u; h < G; h++) { part[(base + h) * 128u + li] = acc[h]; }
}
"#;

/// Split-key attention, part 2: merges the chunks of each (row, query head).
pub const ATTN_COMBINE: &str = r#"
struct P { hq: u32, hkv: u32, nch: u32, _a: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@group(0) @binding(1) var<storage, read> part: array<f32>;
@group(0) @binding(2) var<storage, read> ml: array<vec2<f32>>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let t = wg.y;
    let g = p.hq / p.hkv;
    let kh = h / g;
    let hi = h % g;
    // One pass with a running maximum; four chunks per step keep loads in flight.
    // Empty chunks have (max, sum) = (-1e30, 0) and never wrote `part`: they are
    // dropped by value, not just by weight.
    let b0 = ((t * p.hkv + kh) * p.nch) * g + hi;
    var mx = -1e30;
    var l = 0.0;
    var acc = 0.0;
    var c = 0u;
    for (; c + 4u <= p.nch; c += 4u) {
        let i0 = b0 + c * g;
        let e0 = ml[i0];
        let e1 = ml[i0 + g];
        let e2 = ml[i0 + 2u * g];
        let e3 = ml[i0 + 3u * g];
        let a0 = part[i0 * 128u + li];
        let a1 = part[(i0 + g) * 128u + li];
        let a2 = part[(i0 + 2u * g) * 128u + li];
        let a3 = part[(i0 + 3u * g) * 128u + li];
        let nm = max(max(mx, max(e0.x, e1.x)), max(e2.x, e3.x));
        let r = exp(mx - nm);
        let w0 = select(0.0, exp(e0.x - nm), e0.y > 0.0);
        let w1 = select(0.0, exp(e1.x - nm), e1.y > 0.0);
        let w2 = select(0.0, exp(e2.x - nm), e2.y > 0.0);
        let w3 = select(0.0, exp(e3.x - nm), e3.y > 0.0);
        l = l * r + e0.y * w0 + e1.y * w1 + e2.y * w2 + e3.y * w3;
        acc = acc * r + select(0.0, a0 * w0, e0.y > 0.0) + select(0.0, a1 * w1, e1.y > 0.0)
            + select(0.0, a2 * w2, e2.y > 0.0) + select(0.0, a3 * w3, e3.y > 0.0);
        mx = nm;
    }
    for (; c < p.nch; c++) {
        let i = b0 + c * g;
        let e = ml[i];
        let nm = max(mx, e.x);
        let r = exp(mx - nm);
        let w = select(0.0, exp(e.x - nm), e.y > 0.0);
        l = l * r + e.y * w;
        acc = acc * r + select(0.0, part[i * 128u + li] * w, e.y > 0.0);
        mx = nm;
    }
    o[(t * p.hq + h) * 128u + li] = acc / l;
}
"#;
