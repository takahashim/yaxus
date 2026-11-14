//! The kernels: what identifies one ([`KernelId`]) and, in one place per family, its
//! name, storage bindings and source; the base set compiled when the device opens; the
//! set a model's matrices take, compiled ahead of use; and [`KernelCache`], which compiles
//! them and holds them.

use std::collections::{HashMap, HashSet};

use crate::dev::{Dev, Kernel};
use crate::device::{CompileSettings, DeviceCaps, Tuning};
use crate::plan::{self, MatmulCase};
use crate::{
    Format, GEMV_MAX_T, GEMVM_MAX_T, GemmKind, Gemv, SCALE_PAD, gemm_int, gemv_f32, gemv_int, msl, shaders, steps,
};

/// Which `add_norm` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AddNormKind {
    Plain,
    /// The subgroup kernel, which holds a row in registers (`ops::ADD_NORM_SG_MAX_COLS`).
    Subgroup,
    Msl,
}

/// A kernel: its family and what its source is generated for. The ops dispatch by it,
/// `Gpu::precompile` compiles by it, and its name, bindings and source are given here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum KernelId {
    /// The single-row f32 GEMV.
    Gemv1(Format),
    /// The f32 GEMV of up to `GEMV_MAX_T` rows (without subgroups).
    Gemv8(Format),
    /// The multi-row f32 GEMV of exactly `t` rows.
    Gemvm {
        t: usize,
        fmt: Format,
    },
    /// The single-row f32 GEMV with SwiGLU.
    Glu1(Format),
    /// The matrix-core GEMM (16x16 f16 matrices on Vulkan, Metal's 8x8 ones in f32).
    GemmCoop(Format),
    /// The f32 GEMM.
    GemmF32(Format),
    /// `gemv_f32` for `t` rows with `rw` weight rows per lane group.
    GemvF32 {
        fmt: Format,
        t: usize,
        rw: usize,
        glu: bool,
    },
    /// An `msl::gemv2` kernel.
    Gemv2 {
        fmt: Format,
        k: msl::Gemv2,
    },
    /// An `msl::gemm` kernel.
    GemmMsl {
        fmt: Format,
        k: msl::Gemm,
    },
    /// The rounding of activation rows to `planes` int8 planes for the integer GEMM.
    Xquant {
        planes: usize,
    },
    /// The same for the integer GEMVs.
    XquantRows {
        planes: usize,
    },
    /// The integer GEMM tile of `tn` lanes.
    GemmInt {
        fmt: Format,
        planes: usize,
        tn: usize,
    },
    /// The integer GEMV of `t` rows with `rw` weight rows per lane group (`inline`: the
    /// single-row one that rounds its row).
    GemvInt {
        fmt: Format,
        planes: usize,
        t: usize,
        rw: usize,
        glu: bool,
        inline: bool,
    },
    RmsNorm {
        msl: bool,
    },
    AddNorm(AddNormKind),
    QkRope {
        msl: bool,
    },
    /// The per-row attention.
    Attention,
    /// The split attention for `g` query heads per KV head, `hg` of them per workgroup and
    /// chunks of `ch` keys.
    AttnSplit {
        g: usize,
        ch: usize,
        hg: usize,
    },
    AttnCombine,
    AttnTile,
    Embed,
    EmbedQ4k,
    Argmax1,
    Argmax2,
    Swiglu,
    Add,
}

impl KernelId {
    /// Its name, for the compile log, the profile and the device's labels.
    pub(crate) fn name(self) -> String {
        use KernelId::*;
        let gl = |glu: bool| if glu { "g" } else { "" };
        match self {
            Gemv1(f) => format!("gemv1_{}", f.suffix()),
            Gemv8(f) => format!("gemv8_{}", f.suffix()),
            Gemvm { t, fmt } => format!("gemvm{t}_{}", fmt.suffix()),
            Glu1(f) => format!("glu_{}", f.suffix()),
            GemmCoop(f) => format!("gemmcoop_{}", f.suffix()),
            GemmF32(f) => format!("gemm2f32_{}", f.suffix()),
            GemvF32 { fmt, t, rw, glu } => format!("gemvf{t}x{rw}{}_{}", gl(glu), fmt.suffix()),
            Gemv2 { fmt, k } => match k.tiles {
                None => format!("gemv2_{}{}_{}", k.t, gl(k.glu), fmt.suffix()),
                Some(nt) => format!("gemv2m{nt}_{}{}_{}", k.t, gl(k.glu), fmt.suffix()),
            },
            GemmMsl { fmt, k } => format!("gemm{}{}_{}", if k.f16 { "h" } else { "f" }, k.rb, fmt.suffix()),
            Xquant { planes } => format!("xquant{planes}"),
            XquantRows { planes } => format!("xquantr{planes}"),
            GemmInt { fmt, planes, tn } => format!("gemmint{planes}_{tn}_{}", fmt.suffix()),
            GemvInt { fmt, planes, t, rw, glu, inline } => {
                let rows = if inline { "1i".to_string() } else { t.to_string() };
                format!("gemvi{planes}_{rows}x{rw}{}_{}", gl(glu), fmt.suffix())
            }
            RmsNorm { msl } => if msl { "rmsnorm_msl" } else { "rmsnorm" }.into(),
            AddNorm(AddNormKind::Plain) => "add_norm".into(),
            AddNorm(AddNormKind::Subgroup) => "add_norm_sg".into(),
            AddNorm(AddNormKind::Msl) => "add_norm_msl".into(),
            QkRope { msl } => if msl { "qk_rope_msl" } else { "qk_rope" }.into(),
            Attention => "attention".into(),
            AttnSplit { g, ch, hg } => format!("attn_split{g}_{ch}_{hg}"),
            AttnCombine => "attn_combine".into(),
            AttnTile => "attn_tile".into(),
            Embed => "embed".into(),
            EmbedQ4k => "embed_q4k".into(),
            Argmax1 => "argmax1".into(),
            Argmax2 => "argmax2".into(),
            Swiglu => "swiglu".into(),
            Add => "add".into(),
        }
    }

    /// Whether each of its storage bindings is read-only.
    pub(crate) fn read_only(self) -> &'static [bool] {
        use KernelId::*;
        // (the output, the planes from `xquant`, the weights, the rows' scales and sums)
        const PLANES: &[bool] = &[false, true, true, true, true];
        match self {
            Gemv1(f) | Gemv8(f) | Glu1(f) | GemmCoop(f) | GemmF32(f) => weight_bindings(f),
            Gemvm { fmt, .. } | GemvF32 { fmt, .. } | Gemv2 { fmt, .. } | GemmMsl { fmt, .. } => weight_bindings(fmt),
            Xquant { .. } | XquantRows { .. } => &[false, true, false],
            GemmInt { .. } | GemvInt { inline: false, .. } => PLANES,
            // (the f32 row in place of the planes, and no scales of its own)
            GemvInt { inline: true, .. } => &[false, true, true, true],
            RmsNorm { .. } | Embed | AttnCombine => &[false, true, true],
            AddNorm(_) => &[false, false, true, true, true],
            QkRope { .. } => &[false, false, false, true, true, true, true, true],
            Attention => &[false, true, true, true, true, true],
            AttnSplit { .. } => &[false, false, true, true, true, true, true],
            AttnTile => &[false, false, false, true, true, true, true, true, true],
            EmbedQ4k => &[false, true, true, true],
            Argmax1 | Argmax2 | Swiglu | Add => &[false, true],
        }
    }

    /// Its source (WGSL, or MSL marked by `msl::header`) for a device with `caps`.
    pub(crate) fn source(self, caps: &DeviceCaps, tuning: &Tuning) -> String {
        use KernelId::*;
        use shaders::*;
        let gemv = |t: usize| GEMV.replace("{MAXT}", &t.to_string()).replace("{RED}", &(t * 256).to_string());
        let src = match self {
            // Decode GEMV: the subgroup kernel where available.
            Gemv1(f) if caps.subgroups => format!("{MM_PARAMS}{}{GEMV_T1}", w16(f)),
            Gemv1(f) => format!("{MM_PARAMS}{}{}", w16(f), gemv(1)),
            Gemv8(f) => format!("{MM_PARAMS}{}{}", w16(f), gemv(GEMV_MAX_T)),
            Glu1(f) => format!("{MM_PARAMS}{}{GEMV_GLU}", w16(f)),
            Gemvm { t, fmt } => {
                let r = tuning.gemvm_rows;
                let body = GEMV_MULTI
                    .replace("{T}", &t.to_string())
                    .replace("{R}", &r.to_string())
                    .replace("{RT}", &(r * t).to_string())
                    .replace("{T128}", &(128 * t).to_string());
                format!("{MM_PARAMS}{}{body}", w16(fmt))
            }
            // The decode kernels read the weights 16 at a time (`W_*_16`), the prefill
            // GEMMs through `load4` (`W_*`), except `gemm_coop8`, which decodes 16 at a time.
            GemmCoop(f) => match caps.coop_size {
                16 => format!("enable f16;\nenable wgpu_cooperative_matrix;\n{MM_PARAMS}{}{GEMM_COOP}", w4(f)),
                8 => format!("enable wgpu_cooperative_matrix;\n{MM_PARAMS}{}{}", w16(f), gemm_coop8()),
                n => panic!("no matrix-core GEMM on {n}x{n} matrices"),
            },
            GemmF32(f) => format!("{MM_PARAMS}{}{GEMM2}", w4(f)),
            GemvF32 { fmt, t, rw, glu } => {
                let stage = gemv_f32::stages(t, rw, glu, caps.max_workgroup_storage);
                format!("{MM_PARAMS}{}{}", w16(fmt), gemv_f32::source(t, rw, glu, stage))
            }
            Gemv2 { fmt, k } => k.source(fmt).expect("the plan takes gemv2 only for its formats"),
            GemmMsl { fmt, k } => k.source(fmt).expect("the plan takes msl::gemm only for its formats"),
            Xquant { planes } => gemm_int::xquant_source(planes),
            XquantRows { planes } => gemv_int::xquant_source(planes),
            GemmInt { fmt, planes, tn } => {
                gemm_int::source(fmt.gemm_int().expect("an integer GEMM format"), tn, planes)
            }
            GemvInt { fmt, planes, t, rw, glu, inline } => {
                gemv_int::source(fmt.gemv_int().expect("an integer GEMV format"), t, planes, rw, glu, inline)
            }
            RmsNorm { msl: false } => RMSNORM.into(),
            RmsNorm { msl: true } => msl::rmsnorm_source(),
            AddNorm(AddNormKind::Plain) => ADD_NORM.into(),
            AddNorm(AddNormKind::Subgroup) => ADD_NORM_SG.into(),
            AddNorm(AddNormKind::Msl) => msl::add_norm_source(),
            QkRope { msl: false } => QK_ROPE.into(),
            QkRope { msl: true } => msl::qk_rope_source(),
            Attention => ATTENTION.into(),
            AttnSplit { g, ch, hg } => ATTN_SPLIT
                .replace("{GF}", &g.to_string())
                .replace("{G}", &hg.to_string())
                .replace("{CH}", &ch.to_string())
                .replace("{GCH}", &(16 * ch).to_string()),
            AttnCombine => ATTN_COMBINE.into(),
            AttnTile => attn_tile(plan::tile_stages_queries(caps)),
            Embed => EMBED.into(),
            EmbedQ4k => EMBED_Q4K.replace("{LOADER}", W_Q4K),
            Argmax1 => ARGMAX1.into(),
            Argmax2 => ARGMAX2.into(),
            Swiglu => SWIGLU.into(),
            Add => ADD.into(),
        };
        src.replace("{SPAD}", &SCALE_PAD.to_string())
    }
}

/// The bindings of a matrix product over a matrix of `f`: the output, the activations,
/// then the weights (values and scales, or the bf16 matrix alone).
fn weight_bindings(f: Format) -> &'static [bool] {
    if f == Format::Bf16 { &[false, true, true] } else { &[false, true, true, true] }
}

/// The decode kernels' weight access for `f` (16 values at a time).
fn w16(f: Format) -> &'static str {
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

/// The prefill GEMMs' weight access for `f` (`load4`).
fn w4(f: Format) -> &'static str {
    use shaders::*;
    match f {
        Format::Q8 => W_Q8,
        Format::Q4 => W_Q4,
        Format::Bf16 => W_BF16,
        Format::Q4K => W_Q4K,
        Format::Q6 => W_Q6,
        Format::Q3 => W_Q3,
        Format::Q2 => W_Q2,
    }
}

/// The kernels every model uses, compiled when the device is opened.
pub(crate) fn base_kernels(caps: &DeviceCaps) -> Vec<KernelId> {
    use KernelId::*;
    let sg = caps.subgroups;
    let mut out = Vec::new();
    for f in Format::ALL {
        out.extend([Gemv1(f), Gemv8(f), Glu1(f)]);
        if sg {
            out.extend((2..=GEMVM_MAX_T).map(|t| Gemvm { t, fmt: f }));
        }
        if matches!(caps.coop_size, 8 | 16) {
            out.push(GemmCoop(f));
        }
        out.push(GemmF32(f));
    }
    out.extend([RmsNorm { msl: false }, AddNorm(AddNormKind::Plain)]);
    if sg {
        out.push(AddNorm(AddNormKind::Subgroup));
    }
    out.extend([Argmax1, Argmax2, QkRope { msl: false }, Attention, Embed, EmbedQ4k]);
    // (the split attention reduces with subgroup shuffles; without them the per-row
    // kernel takes every pass)
    if sg {
        for g in plan::SPLIT_GROUPS {
            for ch in plan::SPLIT_CHUNKS {
                out.extend(plan::SPLIT_HEAD_GROUPS.into_iter().filter(|h| g % h == 0).map(|hg| AttnSplit {
                    g,
                    ch,
                    hg,
                }));
            }
        }
    }
    out.push(AttnCombine);
    // Without subgroups, prefill attention takes the per-row kernel.
    if plan::tile_compiled(caps) {
        out.push(AttnTile);
    }
    // (the MSL ones beside them, chosen at dispatch: `plan::msl_small`)
    if caps.msl {
        out.extend([RmsNorm { msl: true }, AddNorm(AddNormKind::Msl), QkRope { msl: true }]);
    }
    out.extend([Swiglu, Add]);
    out
}

/// A matrix a model's matrix products read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    pub fmt: Format,
    pub n: usize,
    pub k: usize,
}

/// Passes of up to this many rows have their kernels compiled ahead of use. Longer ones
/// take the kernels of a shorter pass: the GEMMs' tiles and rests repeat every 128 rows.
const AHEAD_ROWS: usize = 256;

/// The kernels the matrix products over `shapes` take (their [`steps`]), as plain
/// products and as GLUs, in passes of any size, with `gemm` and `gemv`.
pub(crate) fn ahead_kernels(
    caps: &DeviceCaps,
    tuning: &Tuning,
    gemm: GemmKind,
    gemv: Gemv,
    shapes: &[Shape],
) -> Vec<KernelId> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &Shape { fmt, n, k } in shapes {
        for t in 1..=AHEAD_ROWS {
            let case = MatmulCase { t, n, k, fmt, gemm, gemv };
            let mut ids: Vec<KernelId> = steps::matmul(&case, caps, tuning).kernels().collect();
            match steps::glu(&case, caps, tuning) {
                Some(s) => ids.extend(s.kernels()),
                None => ids.push(KernelId::Swiglu),
            }
            out.extend(ids.into_iter().filter(|id| seen.insert(*id)));
        }
    }
    out
}

/// The compiled kernels of a device.
pub(crate) struct KernelCache {
    kernels: HashMap<KernelId, Kernel>,
    settings: CompileSettings,
    /// `YAXUS_COMPILE_LOG`: compilations and their times go to stderr.
    log: bool,
}

impl KernelCache {
    pub(crate) fn new(settings: CompileSettings, log: bool) -> Self {
        KernelCache { kernels: HashMap::new(), settings, log }
    }

    /// Compiles those of `ids` that are not yet, on all cores (a cold shader cache makes a
    /// kernel cost a compiler run: on CI's virtual Apple GPUs, ~0.2 s each). `what` names
    /// them in the compile log.
    pub(crate) fn compile_ahead(
        &mut self,
        dev: &Dev,
        caps: &DeviceCaps,
        tuning: &Tuning,
        ids: &[KernelId],
        what: &str,
    ) {
        let ids: Vec<KernelId> = ids.iter().copied().filter(|id| !self.kernels.contains_key(id)).collect();
        if ids.is_empty() {
            return;
        }
        let t0 = web_time::Instant::now();
        let compiler = dev.compiler(self.settings);
        let compile = |id: KernelId| compiler.compile(&id.name(), &id.source(caps, tuning), id.read_only());
        for (id, k) in ids.iter().copied().zip(compile_all(&ids, compile)) {
            self.insert(id, k);
        }
        if self.log {
            eprintln!("[compile] {} {what}, {:.0} ms", ids.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
    }

    /// The kernel `id`, compiled now if it was not compiled ahead (a line in the compile
    /// log then means a stall mid-request).
    pub(crate) fn get(&mut self, dev: &Dev, caps: &DeviceCaps, tuning: &Tuning, id: KernelId) -> &Kernel {
        if !self.kernels.contains_key(&id) {
            let t0 = web_time::Instant::now();
            let name = id.name();
            let k = dev.compiler(self.settings).compile(&name, &id.source(caps, tuning), id.read_only());
            self.insert(id, k);
            if self.log {
                eprintln!("[compile] {name} on first use, {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        &self.kernels[&id]
    }

    fn insert(&mut self, id: KernelId, mut k: Kernel) {
        // (its number in the set: bind groups are cached by it)
        k.id = self.kernels.len();
        self.kernels.insert(id, k);
    }
}

/// `compile` of every one of `ids`, in their order: on the threads of all cores natively,
/// in turn in the browser (WebGPU's device is not `Send`, and the page has no worker
/// threads here).
#[cfg(target_arch = "wasm32")]
fn compile_all(ids: &[KernelId], compile: impl Fn(KernelId) -> Kernel) -> Vec<Kernel> {
    ids.iter().map(|&id| compile(id)).collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn compile_all(ids: &[KernelId], compile: impl Fn(KernelId) -> Kernel + Sync) -> Vec<Kernel> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(ids.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut built: Vec<(usize, Kernel)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&id) = ids.get(i) else { break };
                        out.push((i, compile(id)));
                    }
                    out
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
    });
    built.sort_by_key(|(i, _)| *i);
    built.into_iter().map(|(_, k)| k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Api, DeviceKind, Vendor};

    /// Why `src` may hold a loop that does not end, if it may: the kernels are compiled
    /// without naga's loop counters (`CompileSettings::loop_bounding`), and wgpu then
    /// requires that none loops forever. Every loop must be a `for` whose variable only
    /// its header changes, either counting up (`i++`, `i += C` with a positive constant
    /// `C`, to a bound) or halving to zero (`s >>= 1u` while `s > 0u`), or doubling to a
    /// constant (`s <<= 1u` from a positive start while `s < C`).
    fn unbounded_loop(src: &str) -> Option<String> {
        let code: String = src.lines().map(|l| l.split("//").next().unwrap()).collect::<Vec<_>>().join("\n");
        // (MSL: no string literals, which hold only `_Pragma` arguments; the unrolled loops
        // are loops too; `constant constexpr uint` is a module-scope constant)
        let code: String = code.split('"').step_by(2).collect();
        let code = code.replace("FOR_UNROLL (", "for (").replace("constant constexpr uint ", "const ");
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
            // (to the `)` that closes the header: conditions may hold parentheses)
            let mut open = 1;
            let close = after.char_indices().find_map(|(j, c)| {
                open += (c == '(') as i32 - (c == ')') as i32;
                (open == 0).then_some(j)
            })?;
            let header = &after[..close];
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

    fn vulkan() -> DeviceCaps {
        DeviceCaps {
            name: "test".into(),
            api: Api::Vulkan,
            vendor: Vendor::Amd,
            kind: DeviceKind::Integrated,
            f16: true,
            subgroups: true,
            coop: true,
            coop_size: 16,
            fast_math: false,
            msl: true,
            timestamps: false,
            max_workgroup_storage: 65536,
        }
    }

    /// Shapes of the 2B's matrices, and odd ones that change the plans (K not a multiple
    /// of 256 or 512, N not of 64).
    const SHAPES: [(usize, usize); 4] = [(2048, 2048), (16384, 2048), (2048, 8192), (1000, 1536)];

    fn shapes() -> Vec<Shape> {
        Format::ALL.iter().flat_map(|&fmt| SHAPES.map(|(n, k)| Shape { fmt, n, k })).collect()
    }

    /// Every kernel source, for every device and every option that changes the source.
    #[test]
    fn every_kernel_loop_ends() {
        let gemvs = [Gemv::INT16, Gemv { planes: Some(1), single_f32: false }, Gemv::INT16.for_draft(), Gemv::F32];
        let mut sources = Vec::new();
        // (Vulkan with matrix cores, Metal with its 8x8 matrices, no subgroups)
        for (subgroups, fast_math, coop_size) in [(true, false, 16), (true, true, 8), (false, false, 0)] {
            // (Apple's 32 KiB of workgroup memory: `gemv_f32` both staged and not)
            let max_workgroup_storage = if coop_size == 8 { 32768 } else { 65536 };
            let caps =
                DeviceCaps { subgroups, coop: coop_size != 0, coop_size, fast_math, max_workgroup_storage, ..vulkan() };
            // (with and without the MSL kernels where the device takes them)
            for (rows, msl) in [(1, false), (4, false), (2, true)] {
                let tuning = Tuning {
                    gemvm_rows: rows,
                    gemvi_rows: rows,
                    gemv_f32: true,
                    gemv_f32_rows: rows,
                    apple: crate::device::AppleTuning { gemv2: msl, gemm_msl: msl, ..Tuning::for_device(&caps).apple },
                    ..Tuning::for_device(&caps)
                };
                let mut ids = base_kernels(&caps);
                for gemm in [GemmKind::Int16, GemmKind::Int8, GemmKind::Coop, GemmKind::F32] {
                    for gemv in gemvs {
                        ids.extend(ahead_kernels(&caps, &tuning, gemm, gemv, &shapes()));
                    }
                }
                sources.extend(ids.into_iter().map(|id| (id.name(), id.source(&caps, &tuning))));
            }
        }
        // (the same source comes from many of the combinations)
        let mut seen = HashSet::new();
        sources.retain(|(_, src)| seen.insert(src.clone()));
        assert!(sources.len() > 500, "{} sources", sources.len());
        for (name, src) in &sources {
            if let Some(why) = unbounded_loop(src) {
                panic!("{name}: {why} may not end (the kernels are compiled without loop counters)");
            }
        }
    }

    /// Passes longer than [`AHEAD_ROWS`] take no kernel the ahead set lacks.
    #[test]
    fn the_ahead_set_covers_longer_passes() {
        let apple = DeviceCaps {
            vendor: Vendor::Apple,
            api: Api::Metal,
            coop_size: 8,
            fast_math: true,
            max_workgroup_storage: 32768,
            ..vulkan()
        };
        for caps in [vulkan(), apple] {
            let tuning = Tuning::for_device(&caps);
            for (gemm, gemv) in
                [(tuning.gemm, tuning.gemv), (GemmKind::Int8, Gemv::INT16), (GemmKind::CoopF16, Gemv::F32)]
            {
                let ahead: HashSet<KernelId> =
                    ahead_kernels(&caps, &tuning, gemm, gemv, &shapes()).into_iter().collect();
                for Shape { fmt, n, k } in shapes() {
                    for t in AHEAD_ROWS + 1..=3 * AHEAD_ROWS {
                        let case = MatmulCase { t, n, k, fmt, gemm, gemv };
                        for id in steps::matmul(&case, &caps, &tuning).kernels() {
                            assert!(ahead.contains(&id), "{} for {t} rows of {fmt:?} {n}x{k}", id.name());
                        }
                    }
                }
            }
        }
    }

    /// No two kernels share a name (the compile log and the profile go by it).
    #[test]
    fn kernel_names_are_unique() {
        let caps = vulkan();
        let tuning = Tuning::for_device(&caps);
        let mut ids = base_kernels(&caps);
        for gemm in [GemmKind::Int16, GemmKind::Int8, GemmKind::CoopF16] {
            for gemv in [Gemv::INT16, Gemv::F32, Gemv::INT16.for_draft()] {
                ids.extend(ahead_kernels(&caps, &Tuning { gemv_f32: true, ..tuning.clone() }, gemm, gemv, &shapes()));
            }
        }
        let ids: HashSet<KernelId> = ids.into_iter().collect();
        let names: HashSet<String> = ids.iter().map(|id| id.name()).collect();
        assert_eq!(names.len(), ids.len());
    }

    /// The workgroup memory `ATTN_TILE_SHARED_BYTES` counts is that of the kernel's
    /// `var<workgroup>` arrays, in both of its variants.
    #[test]
    fn the_tile_attention_memory_matches_its_source() {
        let bytes = |src: &str| {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
            let mut layouter = naga::proc::Layouter::default();
            layouter.update(module.to_ctx()).unwrap();
            module
                .global_variables
                .iter()
                .filter(|(_, g)| g.space == naga::AddressSpace::WorkGroup)
                .map(|(_, g)| layouter[g.ty].size)
                .sum::<u32>()
        };
        assert_eq!(bytes(&shaders::attn_tile(true)), crate::ATTN_TILE_SHARED_BYTES);
        assert_eq!(bytes(&shaders::attn_tile(false)), crate::ATTN_TILE_SHARED_BYTES_NO_QS);
    }
}

#[cfg(test)]
mod webgpu {
    use super::*;
    use crate::device::{Api, DeviceCaps, DeviceKind, Tuning, Vendor};

    /// The browser's variant of every kernel (push constants as a uniform binding) has
    /// to parse and pass naga's validator, including the uniform layout rules. This is
    /// the part that cannot be run without a browser.
    #[test]
    fn transformed_kernels_parse_and_validate() {
        let caps = DeviceCaps {
            name: "webgpu".into(),
            api: Api::WebGpu,
            vendor: Vendor::Other,
            kind: DeviceKind::Other,
            f16: true,
            subgroups: false,
            coop: false,
            fast_math: false,
            msl: false,
            coop_size: 0,
            timestamps: false,
            max_workgroup_storage: 32768,
        };
        let tuning = Tuning::for_device(&caps);
        assert_eq!(tuning.gemm, GemmKind::F32);
        for id in base_kernels(&caps) {
            let name = id.name();
            let src = crate::dev::wgpu::webgpu_source(&id.source(&caps, &tuning));
            assert!(
                src.contains("@group(0) @binding(15) var<uniform> p: P;"),
                "{name}: the per-dispatch struct was not turned into a uniform"
            );
            let module =
                naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
            );
            validator.validate(&module).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
