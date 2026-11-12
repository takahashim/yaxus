//! Compiling the kernels: the base set when the device opens, the integer kernels of a
//! model's formats ahead of use, and the rest on first use.

use crate::shaders;
use crate::{Format, GEMV_MAX_T, GEMVM_MAX_T, Gemv, Gpu, SCALE_PAD, gemm_int, gemv_int, plan};

pub(crate) struct Kernel {
    /// Its number in the device's kernel set (bind groups are cached by it).
    pub(crate) id: usize,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
}

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

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn compile_kernel(device: &wgpu::Device, name: &str, src: &str, read_only: &[bool]) -> Kernel {
    let src = &src.replace("{SPAD}", &SCALE_PAD.to_string());
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let entries: Vec<_> = read_only.iter().enumerate().map(|(i, &r)| storage_entry(i as u32, r)).collect();
    let layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(name), entries: &entries });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(name),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 32,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { id: 0, pipeline, layout }
}

impl Gpu {
    pub(crate) fn add_kernel(&mut self, name: String, src: &str, read_only: &[bool]) {
        let k = compile_kernel(&self.device, &name, src, read_only);
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
        let gemvi_rows = self.tuning.gemvi_rows;
        // (name, bindings, source)
        type Job = (String, Vec<bool>, Box<dyn Fn() -> String + Send + Sync>);
        let mut jobs: Vec<Job> = Vec::new();
        if let Some(planes) = gemv.planes {
            jobs.push((
                name::xquant_rows(planes),
                vec![false, true, false],
                Box::new(move || gemv_int::xquant_source(planes)),
            ));
            for f in formats.iter().filter_map(|f| f.gemv_int()) {
                for glu in [false, true] {
                    if !gemv.single_f32 {
                        let rw = gemvi_rows;
                        jobs.push((
                            name::gemv_int(planes, 1, rw, glu, true, f.name),
                            vec![false, true, true, true],
                            Box::new(move || gemv_int::source(f, 1, planes, rw, glu, true)),
                        ));
                    }
                    for t in 2..=GEMVM_MAX_T {
                        let rw = gemvi_rows;
                        jobs.push((
                            name::gemv_int(planes, t, rw, glu, false, f.name),
                            vec![false, true, true, true, true],
                            Box::new(move || gemv_int::source(f, t, planes, rw, glu, false)),
                        ));
                    }
                }
            }
        }
        if let Some(planes) = self.gemm.planes() {
            jobs.push((
                name::xquant(planes),
                vec![false, true, false],
                Box::new(move || gemm_int::xquant_source(planes)),
            ));
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
        jobs.retain(|(name, ..)| !self.kernels.contains_key(name.as_str()));
        if jobs.is_empty() {
            return;
        }
        let t0 = std::time::Instant::now();
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(jobs.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (device, jobs) = (&self.device, &jobs);
        let built: Vec<(String, Kernel)> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some((name, read_only, source)) = jobs.get(i) else { break };
                            out.push((name.clone(), compile_kernel(device, name, &source(), read_only)));
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
        use shaders::*;
        let t0 = std::time::Instant::now();
        let sg = self.caps.subgroups;
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
            self.add_kernel(name::gemv1(f), &gemv1(w16), layout(f));
            self.add_kernel(name::gemv8(f), &format!("{MM_PARAMS}{w16}{}", gemv(GEMV_MAX_T)), layout(f));
            self.add_kernel(name::glu(f), &format!("{MM_PARAMS}{w16}{GEMV_GLU}"), layout(f));
            if self.caps.subgroups {
                for t in 2..=GEMVM_MAX_T {
                    let r = self.tuning.gemvm_rows;
                    let body = GEMV_MULTI
                        .replace("{T}", &t.to_string())
                        .replace("{R}", &r.to_string())
                        .replace("{RT}", &(r * t).to_string())
                        .replace("{T128}", &(128 * t).to_string());
                    self.add_kernel(name::gemvm(t, f), &format!("{MM_PARAMS}{w16}{body}"), layout(f));
                }
            }
        }
        let prefill = [(Format::Q8, W_Q8), (Format::Q4, W_Q4), (Format::Bf16, W_BF16), (Format::Q4K, W_Q4K)]
            .into_iter()
            .chain([(Format::Q6, W_Q6), (Format::Q3, W_Q3), (Format::Q2, W_Q2)]);
        for (f, w4) in prefill {
            if self.caps.coop {
                let c = format!("enable f16;\nenable wgpu_cooperative_matrix;\n{MM_PARAMS}{w4}{GEMM_COOP}");
                self.add_kernel(name::gemm_coop(f), &c, layout(f));
            }
            self.add_kernel(name::gemm_f32(f), &format!("{MM_PARAMS}{w4}{GEMM2}"), layout(f));
        }
        self.add_kernel("rmsnorm".into(), RMSNORM, &[false, true, true]);
        self.add_kernel("add_norm".into(), ADD_NORM, &[false, false, true, true, true]);
        if sg {
            self.add_kernel("add_norm_sg".into(), ADD_NORM_SG, &[false, false, true, true, true]);
        }
        self.add_kernel("argmax1".into(), ARGMAX1, &[false, true]);
        self.add_kernel("argmax2".into(), ARGMAX2, &[false, true]);
        self.add_kernel("qk_rope".into(), QK_ROPE, &[false, false, false, true, true, true, true, true]);
        self.add_kernel("attention".into(), ATTENTION, &[false, true, true, true, true, true]);
        self.add_kernel("embed".into(), EMBED, &[false, true, true]);
        // (the split attention reduces with subgroup shuffles; without them the per-row
        // kernel takes every pass)
        let sg = self.caps.subgroups;
        for g in plan::SPLIT_GROUPS.into_iter().filter(|_| sg) {
            for ch in plan::SPLIT_CHUNKS {
                for hg in plan::SPLIT_HEAD_GROUPS.into_iter().filter(|h| g % h == 0) {
                    let src = ATTN_SPLIT
                        .replace("{GF}", &g.to_string())
                        .replace("{G}", &hg.to_string())
                        .replace("{CH}", &ch.to_string())
                        .replace("{GCH}", &(16 * ch).to_string());
                    self.add_kernel(name::attn_split(g, ch, hg), &src, &[false, false, true, true, true, true, true]);
                }
            }
        }
        self.add_kernel("attn_combine".into(), ATTN_COMBINE, &[false, true, true]);
        // The tiled kernel needs more workgroup memory than some devices (32 KiB) have;
        // without it, prefill attention takes the per-row kernel.
        if plan::tile_compiled(&self.caps) {
            self.add_kernel("attn_tile".into(), ATTN_TILE, &[false, false, false, true, true, true, true, true, true]);
        }
        self.add_kernel("swiglu".into(), SWIGLU, &[false, true]);
        self.add_kernel("add".into(), ADD, &[false, true]);
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
