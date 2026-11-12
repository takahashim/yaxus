//! Compiling the kernels: the base set when the device opens, the integer kernels of a
//! model's formats ahead of use, and the rest on first use.

use crate::dev::Kernel;
use crate::device::{DeviceCaps, Tuning};
use crate::shaders;
use crate::{Format, GEMV_MAX_T, GEMVM_MAX_T, GemmKind, Gemv, Gpu, SCALE_PAD, gemm_int, gemv_f32, gemv_int, plan};

/// Kernel names, the one place each family's name is spelt: `build_kernels` and
/// `precompile` compile under them, the ops dispatch by them.
pub(crate) mod name {
    use crate::Format;

    /// The single-row f32 GEMV.
    pub(crate) fn gemv1(f: Format) -> String {
        format!("gemv1_{}", f.suffix())
    }
    /// The f32 GEMV of up to `GEMV_MAX_T` rows (without subgroups).
    pub(crate) fn gemv8(f: Format) -> String {
        format!("gemv8_{}", f.suffix())
    }
    /// The multi-row f32 GEMV of exactly `t` rows.
    pub(crate) fn gemvm(t: usize, f: Format) -> String {
        format!("gemvm{t}_{}", f.suffix())
    }
    /// `gemv_f32` for `t` rows with `rw` weight rows per lane group.
    pub(crate) fn gemv_f32(t: usize, rw: usize, glu: bool, f: Format) -> String {
        format!("gemvf{t}x{rw}{}_{}", if glu { "g" } else { "" }, f.suffix())
    }
    /// `msl::gemv2` for `t` rows.
    pub(crate) fn gemv2(t: usize, glu: bool, f: Format) -> String {
        format!("gemv2_{t}{}_{}", if glu { "g" } else { "" }, f.suffix())
    }
    /// `msl::gemv_q` for `t` rows with `rw` weight rows per lane group.
    pub(crate) fn gemv_q(t: usize, rw: usize, glu: bool, f: Format) -> String {
        format!("gemvq{t}x{rw}{}_{}", if glu { "g" } else { "" }, f.suffix())
    }
    /// `msl::gemv_mma` with `tiles` 8-row tiles.
    pub(crate) fn gemv_mma(tiles: usize, glu: bool, f: Format) -> String {
        format!("gemvmma{tiles}{}_{}", if glu { "g" } else { "" }, f.suffix())
    }
    /// The single-row f32 GEMV with SwiGLU.
    pub(crate) fn glu(f: Format) -> String {
        format!("glu_{}", f.suffix())
    }
    pub(crate) fn gemm_coop(f: Format) -> String {
        format!("gemmcoop_{}", f.suffix())
    }
    pub(crate) fn gemm_f32(f: Format) -> String {
        format!("gemm2f32_{}", f.suffix())
    }
    /// The split attention for `g` query heads per KV head, `hg` of them per workgroup
    /// and chunks of `ch` keys.
    pub(crate) fn attn_split(g: usize, ch: usize, hg: usize) -> String {
        format!("attn_split{g}_{ch}_{hg}")
    }
    /// The rounding of activation rows to `planes` int8 planes for the integer GEMM.
    pub(crate) fn xquant(planes: usize) -> String {
        format!("xquant{planes}")
    }
    /// The same for the integer GEMVs.
    pub(crate) fn xquant_rows(planes: usize) -> String {
        format!("xquantr{planes}")
    }
    /// The integer GEMM tile of `tn` lanes for the format named `fmt` (`IntFmt::name`).
    pub(crate) fn gemm_int(planes: usize, tn: usize, fmt: &str) -> String {
        format!("gemmint{planes}_{tn}_{fmt}")
    }
    /// The integer GEMV of `t` rows with `rw` weight rows per lane group (`inline`: the
    /// single-row one that rounds its row).
    pub(crate) fn gemv_int(planes: usize, t: usize, rw: usize, glu: bool, inline: bool, fmt: &str) -> String {
        let g = if glu { "g" } else { "" };
        if inline { format!("gemvi{planes}_1ix{rw}{g}_{fmt}") } else { format!("gemvi{planes}_{t}x{rw}{g}_{fmt}") }
    }
}

/// A kernel compiled ahead of use: (name, read-only bindings, source).
pub(crate) type Job = (String, Vec<bool>, Box<dyn Fn() -> String + Send + Sync>);

/// The kernels compiled ahead of use for a model with weights of `formats`, with `gemv`
/// and `gemm`: the integer ones, and `gemv_f32` where the tuning takes it.
pub(crate) fn int_jobs(caps: &DeviceCaps, tuning: &Tuning, gemm: GemmKind, formats: &[Format], gemv: Gemv) -> Vec<Job> {
    let rw = tuning.gemvi_rows;
    let mut jobs: Vec<Job> = Vec::new();
    // (`gemv_f32` in place of the f32 GEMVs: the base set does not hold it, since a
    // model compiles its formats only)
    if gemv.planes.is_none() && tuning.gemv_f32 {
        let (rf, wgmem) = (tuning.gemv_f32_rows, caps.max_workgroup_storage);
        for &f in formats {
            // (`msl::gemv_mma` for the formats it reads; K is known only at use, so both
            // families of a format are listed)
            if tuning.gemm_msl && caps.msl && crate::msl::gemm_reads(f) {
                let f16 = gemm == GemmKind::CoopF16;
                for rb in [1, 2] {
                    jobs.push((
                        format!("gemm{}{rb}_{}", if f16 { "h" } else { "f" }, f.suffix()),
                        decode_layout(f).to_vec(),
                        Box::new(move || crate::msl::gemm(f, f16, rb).unwrap()),
                    ));
                }
            }
            if tuning.gemv2 && caps.msl && crate::msl::gemv2_reads(f) {
                for glu in [false, true] {
                    for t in 1..=GEMVM_MAX_T {
                        jobs.push((
                            name::gemv2(t, glu, f),
                            decode_layout(f).to_vec(),
                            Box::new(move || crate::msl::gemv2(f, t, glu).unwrap()),
                        ));
                    }
                }
                continue;
            }
            if tuning.gemv_q && caps.msl && crate::msl::gemv_q_reads(f) {
                for glu in [false, true] {
                    for t in 1..=GEMVM_MAX_T {
                        let stage = crate::msl::gemv_q_stages(t, rf, glu, wgmem);
                        jobs.push((
                            name::gemv_q(t, rf, glu, f),
                            decode_layout(f).to_vec(),
                            Box::new(move || crate::msl::gemv_q(f, t, rf, glu, stage).unwrap()),
                        ));
                    }
                }
                continue;
            }
            if tuning.gemv_mma && caps.msl && crate::msl::gemv_mma_reads(f) {
                for glu in [false, true] {
                    for tiles in [1, 2] {
                        jobs.push((
                            name::gemv_mma(tiles, glu, f),
                            decode_layout(f).to_vec(),
                            Box::new(move || crate::msl::gemv_mma(f, tiles, glu).unwrap()),
                        ));
                    }
                }
            }
            for glu in [false, true] {
                for t in 1..=GEMVM_MAX_T {
                    jobs.push((
                        name::gemv_f32(t, rf, glu, f),
                        decode_layout(f).to_vec(),
                        Box::new(move || gemv_f32_source(f, t, rf, glu, wgmem)),
                    ));
                }
            }
        }
    }
    if let Some(planes) = gemv.planes {
        jobs.push((
            name::xquant_rows(planes),
            vec![false, true, false],
            Box::new(move || gemv_int::xquant_source(planes)),
        ));
        for f in formats.iter().filter_map(|f| f.gemv_int()) {
            for glu in [false, true] {
                // (the row counts and the rounding as `MatmulCase::plan` picks them)
                for t in (1..=GEMVM_MAX_T).filter(|&t| gemv.planes(t).is_some()) {
                    let inline = plan::gemv_int_inline(caps, t);
                    let bindings =
                        if inline { vec![false, true, true, true] } else { vec![false, true, true, true, true] };
                    jobs.push((
                        name::gemv_int(planes, t, rw, glu, inline, f.name),
                        bindings,
                        Box::new(move || gemv_int::source(f, t, planes, rw, glu, inline)),
                    ));
                }
            }
        }
    }
    if let Some(planes) = gemm.planes() {
        jobs.push((name::xquant(planes), vec![false, true, false], Box::new(move || gemm_int::xquant_source(planes))));
        for f in formats.iter().filter_map(|f| f.gemm_int()) {
            for tn in [2, 4, 8] {
                jobs.push((
                    name::gemm_int(planes, tn, f.name),
                    vec![false, true, true, true, true],
                    Box::new(move || gemm_int::source(f, tn, planes)),
                ));
            }
        }
    }
    jobs
}

/// The kernels every model uses, compiled when the device is opened: (name, source,
/// read-only bindings) for a device with `caps`.
pub(crate) fn base_sources(caps: &DeviceCaps, tuning: &Tuning) -> Vec<(String, String, &'static [bool])> {
    use shaders::*;
    let mut out = Vec::new();
    let mut add = |name: String, src: &str, read_only: &'static [bool]| out.push((name, src.to_string(), read_only));
    let sg = caps.subgroups;
    // Decode GEMV: the subgroup kernel where available.
    let gemv1 = move |w16: &str| -> String {
        if sg {
            format!("{MM_PARAMS}{w16}{GEMV_T1}")
        } else {
            format!("{MM_PARAMS}{w16}{}", GEMV.replace("{MAXT}", "1").replace("{RED}", "256"))
        }
    };
    let gemv = |t: usize| GEMV.replace("{MAXT}", &t.to_string()).replace("{RED}", &(t * 256).to_string());
    // (the weight bindings: values and scales, or the bf16 matrix alone)
    let layout = |f: Format| -> &'static [bool] {
        if f == Format::Bf16 { &[false, true, true] } else { &[false, true, true, true] }
    };
    // The decode kernels read the weights 16 at a time (`W_*_16`), the prefill GEMMs
    // through `load4` (`W_*`).
    let decode = [(Format::Q8, W_Q8_16), (Format::Q4, W_Q4_16), (Format::Bf16, W_BF16_16), (Format::Q4K, W_Q4K_16)]
        .into_iter()
        .chain([(Format::Q6, W_Q6_16), (Format::Q3, W_Q3_16), (Format::Q2, W_Q2_16)]);
    for (f, w16) in decode {
        add(name::gemv1(f), &gemv1(w16), layout(f));
        add(name::gemv8(f), &format!("{MM_PARAMS}{w16}{}", gemv(GEMV_MAX_T)), layout(f));
        add(name::glu(f), &format!("{MM_PARAMS}{w16}{GEMV_GLU}"), layout(f));
        if caps.subgroups {
            for t in 2..=GEMVM_MAX_T {
                let r = tuning.gemvm_rows;
                let body = GEMV_MULTI
                    .replace("{T}", &t.to_string())
                    .replace("{R}", &r.to_string())
                    .replace("{RT}", &(r * t).to_string())
                    .replace("{T128}", &(128 * t).to_string());
                add(name::gemvm(t, f), &format!("{MM_PARAMS}{w16}{body}"), layout(f));
            }
        }
    }
    let prefill = [(Format::Q8, W_Q8), (Format::Q4, W_Q4), (Format::Bf16, W_BF16), (Format::Q4K, W_Q4K)]
        .into_iter()
        .chain([(Format::Q6, W_Q6), (Format::Q3, W_Q3), (Format::Q2, W_Q2)]);
    for (f, w4) in prefill {
        if caps.coop_size == 16 {
            let c = format!("enable f16;\nenable wgpu_cooperative_matrix;\n{MM_PARAMS}{w4}{GEMM_COOP}");
            add(name::gemm_coop(f), &c, layout(f));
        } else if caps.coop_size == 8 {
            // (the decode loader: `gemm_coop8` decodes 16 weights at a time)
            let c = format!("enable wgpu_cooperative_matrix;\n{MM_PARAMS}{}{}", w16(f), gemm_coop8());
            add(name::gemm_coop(f), &c, layout(f));
        }
        add(name::gemm_f32(f), &format!("{MM_PARAMS}{w4}{GEMM2}"), layout(f));
    }
    add("rmsnorm".into(), RMSNORM, &[false, true, true]);
    add("add_norm".into(), ADD_NORM, &[false, false, true, true, true]);
    if sg {
        add("add_norm_sg".into(), ADD_NORM_SG, &[false, false, true, true, true]);
    }
    add("argmax1".into(), ARGMAX1, &[false, true]);
    add("argmax2".into(), ARGMAX2, &[false, true]);
    add("qk_rope".into(), QK_ROPE, &[false, false, false, true, true, true, true, true]);
    add("attention".into(), ATTENTION, &[false, true, true, true, true, true]);
    add("embed".into(), EMBED, &[false, true, true]);
    // (the split attention reduces with subgroup shuffles; without them the per-row
    // kernel takes every pass)
    let sg = caps.subgroups;
    for g in plan::SPLIT_GROUPS.into_iter().filter(|_| sg) {
        for ch in plan::SPLIT_CHUNKS {
            for hg in plan::SPLIT_HEAD_GROUPS.into_iter().filter(|h| g % h == 0) {
                let src = ATTN_SPLIT
                    .replace("{GF}", &g.to_string())
                    .replace("{G}", &hg.to_string())
                    .replace("{CH}", &ch.to_string())
                    .replace("{GCH}", &(16 * ch).to_string());
                add(name::attn_split(g, ch, hg), &src, &[false, false, true, true, true, true, true]);
            }
        }
    }
    add("attn_combine".into(), ATTN_COMBINE, &[false, true, true]);
    // Without subgroups, prefill attention takes the per-row kernel.
    if plan::tile_compiled(caps) {
        let src = attn_tile(plan::tile_stages_queries(caps));
        add("attn_tile".into(), &src, &[false, false, false, true, true, true, true, true, true]);
    }
    // (the MSL ones beside them, chosen at dispatch: `Tuning::msl_small`)
    if caps.msl {
        for (name, src, read_only) in crate::msl::small_kernels() {
            add(name.to_string(), &src, read_only);
        }
    }
    add("swiglu".into(), SWIGLU, &[false, true]);
    add("add".into(), ADD, &[false, true]);
    out
}

/// The decode kernels' weight access for `f` (16 values at a time).
pub(crate) fn w16(f: Format) -> &'static str {
    use shaders::*;
    match f {
        Format::Q8 => W_Q8_16,
        Format::Q4 => W_Q4_16,
        Format::Bf16 => W_BF16_16,
        Format::Q4K => W_Q4K_16,
        Format::Q6 => W_Q6_16,
        Format::Q3 => W_Q3_16,
        Format::Q2 => W_Q2_16,
    }
}

/// The weight bindings of the decode kernels for `f` (values and scales, or the bf16
/// matrix alone), after the output and the activations.
pub(crate) fn decode_layout(f: Format) -> &'static [bool] {
    if f == Format::Bf16 { &[false, true, true] } else { &[false, true, true, true] }
}

/// The source of `gemv_f32` for `f`, on a device with `max_storage` bytes of workgroup
/// memory.
pub(crate) fn gemv_f32_source(f: Format, t: usize, rw: usize, glu: bool, max_storage: u32) -> String {
    let stage = gemv_f32::stages(t, rw, glu, max_storage);
    format!("{}{}{}", shaders::MM_PARAMS, w16(f), gemv_f32::source(t, rw, glu, stage))
}

impl Gpu {
    pub(crate) fn add_kernel(&mut self, name: String, src: &str, read_only: &[bool]) {
        let src = src.replace("{SPAD}", &SCALE_PAD.to_string());
        let k = self.dev.compiler(self.compile).compile(&name, &src, read_only);
        self.insert_kernel(name, k);
    }

    fn insert_kernel(&mut self, name: String, mut k: Kernel) {
        k.id = self.kernels.len();
        self.kernels.insert(name, k);
    }

    /// Compiles the integer kernels a model with weights of `formats` can use, ahead of
    /// use and on all cores: the GEMV for every row count up to 16, with and without
    /// SwiGLU, and the GEMM tiles. Their sources are long (written out, not looped) and
    /// take up to 40 ms each on a 780M; compiled on first use, they stalled the first
    /// requests (57 kernels and 1 s within the 19-sentence set).
    pub(crate) fn precompile(&mut self, formats: &[Format], gemv: Gemv) {
        if !self.caps.subgroups {
            return;
        }
        let mut jobs = int_jobs(&self.caps, &self.tuning, self.gemm, formats, gemv);
        jobs.retain(|(name, ..)| !self.kernels.contains_key(name.as_str()));
        if jobs.is_empty() {
            return;
        }
        let t0 = std::time::Instant::now();
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(jobs.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (compiler, jobs) = (self.dev.compiler(self.compile), &jobs);
        let built: Vec<(String, Kernel)> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some((name, read_only, source)) = jobs.get(i) else { break };
                            let src = source().replace("{SPAD}", &SCALE_PAD.to_string());
                            out.push((name.clone(), compiler.compile(name, &src, read_only)));
                        }
                        out
                    })
                })
                .collect();
            workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
        });
        if self.diag.compile_log {
            eprintln!("[compile] {} kernels ahead of use, {:.0} ms", built.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
        for (name, k) in built {
            self.insert_kernel(name, k);
        }
    }

    pub(crate) fn build_kernels(&mut self) {
        let t0 = std::time::Instant::now();
        for (name, src, read_only) in base_sources(&self.caps, &self.tuning) {
            self.add_kernel(name, &src, read_only);
        }
        if self.diag.compile_log {
            eprintln!("[compile] {} base kernels, {:.0} ms", self.kernels.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
    }

    /// The kernel `name` (an integer kernel), compiled on first use if `precompile` did
    /// not (their sources are long, and a model uses two or three of the formats).
    pub(crate) fn int_kernel(&mut self, name: String, read_only: &[bool], source: impl FnOnce() -> String) -> String {
        if self.kernels.contains_key(&name) {
            return name;
        }
        let t0 = std::time::Instant::now();
        self.add_kernel(name.clone(), &source(), read_only);
        // (`precompile` should have covered it; a line here means a stall mid-request)
        if self.diag.compile_log {
            eprintln!("[compile] {name} on first use, {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Vendor;

    /// Why `src` may hold a loop that does not end, if it may: the kernels are compiled
    /// without naga's loop counters (`CompileSettings::loop_bounding`), and wgpu then
    /// requires that none loops forever. Every loop must be a `for` whose variable only
    /// its header changes, either counting up (`i++`, `i += C` with a positive constant
    /// `C`, to a bound) or halving to zero (`s >>= 1u` while `s > 0u`), or doubling to a
    /// constant (`s <<= 1u` from a positive start while `s < C`).
    fn unbounded_loop(src: &str) -> Option<String> {
        let code: String = src.lines().map(|l| l.split("//").next().unwrap()).collect::<Vec<_>>().join("\n");
        let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let words = || code.split(|c: char| !is_word(c));
        if let Some(w) = words().find(|&w| w == "loop" || w == "while") {
            return Some(format!("a `{w}` loop"));
        }
        // A positive constant: a literal, or a `const` of the module set to one.
        let positive = |c: &str| {
            let lit = |v: &str| v.trim_end_matches('u').parse::<u64>().is_ok_and(|n| n > 0);
            lit(c)
                || code.lines().any(|l| {
                    let l = l.trim();
                    l.strip_prefix("const ").and_then(|r| r.split_once('=')).is_some_and(|(name, v)| {
                        name.split(':').next().unwrap().trim() == c && lit(v.trim().trim_end_matches(';').trim())
                    })
                })
        };
        // A module-scope `const` (a value fixed when the source is generated).
        let constant = |c: &str| {
            code.lines().any(|l| {
                l.trim().strip_prefix("const ").is_some_and(|r| r.split([':', '=']).next().unwrap().trim() == c)
            })
        };
        let mut rest = code.as_str();
        while let Some(i) = rest.find("for (") {
            let after = &rest[i + "for (".len()..];
            let header = &after[..after.find(')')?];
            // The body: from the `{` after the header to its matching `}`.
            let body = after[header.len() + 1..].trim_start();
            if !body.starts_with('{') {
                return Some(format!("`for ({header})` without a block"));
            }
            let mut depth = 0;
            let end = body.char_indices().find_map(|(j, c)| {
                depth += (c == '{') as i32 - (c == '}') as i32;
                (depth == 0).then_some(j)
            })?;
            let body = &body[..end];
            let parts: Vec<&str> = header.split(';').map(str::trim).collect();
            let fail = || Some(format!("`for ({header})`"));
            let [init, cond, step] = parts[..] else { return fail() };
            // (the variable is declared in the header, or before the loop when the header
            // leaves it out: `for (; k < n; k++)`)
            // (`var` in WGSL; `uint` and `int` in the MSL kernels)
            let v = match init.strip_prefix("var ").or(init.strip_prefix("uint ")).or(init.strip_prefix("int ")) {
                Some(r) => r.split('=').next().unwrap().trim(),
                None if init.is_empty() => step.split([' ', '+']).next().unwrap(),
                None => return fail(),
            };
            // `v < e`, `v <= e`, or `v + c <= e` with a positive constant `c`.
            let up = cond.starts_with(&format!("{v} <"))
                || cond
                    .strip_prefix(&format!("{v} + "))
                    .and_then(|r| r.split_once(" <"))
                    .is_some_and(|(c, _)| positive(c));
            let bounded = if step == format!("{v}++") {
                up
            } else if let Some(c) = step.strip_prefix(&format!("{v} += ")) {
                positive(c) && up
            } else if step == format!("{v} >>= 1u") {
                cond == format!("{v} > 0u")
            } else if step == format!("{v} <<= 1u") {
                // (from a positive start to a small constant: all of them are lane counts)
                let start = init.split('=').nth(1).map(str::trim).is_some_and(positive);
                start && cond.strip_prefix(&format!("{v} < ")).is_some_and(|c| positive(c) || constant(c))
            } else {
                false
            };
            // (the body leaves the variable alone)
            let ops = ["=", "+=", "-=", "*=", "/=", ">>=", "<<=", "++", "--"];
            let assigned = body.match_indices(v).any(|(j, _)| {
                let (head, tail) = (&body[..j], &body[j + v.len()..]);
                let word_start = !head.ends_with(is_word) && !head.ends_with("let ") && !head.ends_with("var ");
                let t = tail.trim_start();
                word_start
                    && !tail.starts_with(is_word)
                    && ops.iter().any(|op| t.starts_with(op))
                    && !t.starts_with("==")
            });
            if !bounded || assigned {
                return fail();
            }
            rest = after;
        }
        None
    }

    #[test]
    fn the_checker_finds_unbounded_loops() {
        let ok = "const BN: u32 = 32u;\nfn main() {\n    for (var i = 0u; i < p.n; i += BN) { x[i] = 1.0; }\n    for (var s = 16u; s > 0u; s >>= 1u) { }\n    var k = 0u;\n    for (; k + 4u <= p.n; k += 4u) { }\n}";
        assert_eq!(unbounded_loop(ok), None);
        for bad in [
            "fn main() {\n    loop { if (x[0] > 0.0) { break; } }\n}",
            "fn main() {\n    while (x[0] > 0.0) { }\n}",
            "fn main() {\n    for (var i = 0u; i < p.n; i += p.step) { }\n}",
            "fn main() {\n    for (var i = 0u; i != p.n; i += 2u) { }\n}",
            "fn main() {\n    for (var s = 16u; s > 0u; s <<= 1u) { }\n}",
            "fn main() {\n    for (var i = 0u; i < p.n; i++) { i = 0u; }\n}",
            "fn main() {\n    var k = 0u;\n    for (; k + 4u <= n; k += 0u) { }\n}",
        ] {
            assert!(unbounded_loop(bad).is_some(), "{bad}");
        }
        // (comments and words that merely contain the keywords do not count)
        assert_eq!(unbounded_loop("// loop over the keys\nfn main() { let loops = 1u; }"), None);
    }

    /// Every kernel source, for every device and every option that changes the source.
    #[test]
    fn every_kernel_loop_ends() {
        let caps = DeviceCaps {
            name: "test".into(),
            backend: wgpu::Backend::Vulkan,
            vendor: Vendor::Amd,
            device_type: wgpu::DeviceType::IntegratedGpu,
            f16: true,
            subgroups: true,
            coop: true,
            coop_size: 16,
            fast_math: false,
            msl: true,
            timestamps: false,
            max_workgroup_storage: 65536,
        };
        let formats = [Format::Q8, Format::Q4, Format::Q4K, Format::Q6, Format::Q3, Format::Q2, Format::Bf16];
        let gemvs = [Gemv::INT16, Gemv { planes: Some(1), single_f32: false }, Gemv::INT16.for_draft()];
        let mut sources = Vec::new();
        // (Vulkan with matrix cores, Metal with its 8x8 matrices, no subgroups)
        for (subgroups, fast_math, coop_size) in [(true, false, 16), (true, true, 8), (false, false, 0)] {
            // (Apple's 32 KiB of workgroup memory: `gemv_f32` both staged and not)
            let max_workgroup_storage = if coop_size == 8 { 32768 } else { 65536 };
            let caps = DeviceCaps {
                subgroups,
                coop: coop_size != 0,
                coop_size,
                fast_math,
                max_workgroup_storage,
                ..caps.clone()
            };
            // (the MSL GEMVs, opt-in, each in place of `gemv_f32` for the formats it reads)
            for (rows, gemv_q, gemv_mma) in [(1, false, false), (4, false, false), (2, true, false), (2, false, true)] {
                let tuning = Tuning {
                    gemvm_rows: rows,
                    gemvi_rows: rows,
                    gemv_f32: true,
                    gemv_f32_rows: rows,
                    gemv_q,
                    gemv_mma,
                    ..Tuning::for_device(&caps)
                };
                sources.extend(base_sources(&caps, &tuning).into_iter().map(|(name, src, _)| (name, src)));
                for gemm in [GemmKind::Int16, GemmKind::Int8] {
                    for gemv in gemvs.into_iter().chain([Gemv::F32]) {
                        let jobs = int_jobs(&caps, &tuning, gemm, &formats, gemv);
                        sources.extend(jobs.into_iter().map(|(name, _, source)| (name, source())));
                    }
                }
            }
        }
        // (the same source comes from many of the combinations)
        let mut seen = std::collections::HashSet::new();
        sources.retain(|(_, src)| seen.insert(src.clone()));
        assert!(sources.len() > 500, "{} sources", sources.len());
        for (name, src) in &sources {
            if let Some(why) = unbounded_loop(src) {
                panic!("{name}: {why} may not end (the kernels are compiled without loop counters)");
            }
        }
    }
}
