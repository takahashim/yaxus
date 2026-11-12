//! Compiling the kernels: the base set when the device opens, the integer kernels of a
//! model's formats ahead of use, and the rest on first use.

use crate::shaders;
use crate::{Format, GEMV_MAX_T, GEMVM_MAX_T, Gemv, Gpu, SCALE_PAD, gemm_int, gemv_int, plan};

pub(crate) struct Kernel {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
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

/// The name of a `gemv_int` kernel (`inline`: the single-row one that rounds its row).
pub(crate) fn gemvi_name(planes: usize, t: usize, rw: usize, glu: bool, inline: bool, format: &str) -> String {
    let g = if glu { "g" } else { "" };
    if inline { format!("gemvi{planes}_1ix{rw}{g}_{format}") } else { format!("gemvi{planes}_{t}x{rw}{g}_{format}") }
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
    Kernel { pipeline, layout }
}

impl Gpu {
    pub(crate) fn add_kernel(&mut self, name: &'static str, src: &str, read_only: &[bool]) {
        let k = compile_kernel(&self.device, name, src, read_only);
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
                format!("xquantr{planes}"),
                vec![false, true, false],
                Box::new(move || gemv_int::xquant_source(planes)),
            ));
            for f in formats.iter().filter_map(|f| f.gemv_int()) {
                for glu in [false, true] {
                    if !gemv.single_f32 {
                        let rw = gemvi_rows;
                        jobs.push((
                            gemvi_name(planes, 1, rw, glu, true, f.name),
                            vec![false, true, true, true],
                            Box::new(move || gemv_int::source(f, 1, planes, rw, glu, true)),
                        ));
                    }
                    for t in 2..=GEMVM_MAX_T {
                        let rw = gemvi_rows;
                        jobs.push((
                            gemvi_name(planes, t, rw, glu, false, f.name),
                            vec![false, true, true, true, true],
                            Box::new(move || gemv_int::source(f, t, planes, rw, glu, false)),
                        ));
                    }
                }
            }
        }
        if let Some(planes) = self.gemm.planes() {
            jobs.push((
                format!("xquant{planes}"),
                vec![false, true, false],
                Box::new(move || gemm_int::xquant_source(planes)),
            ));
            for f in formats.iter().filter_map(|f| f.gemm_int()) {
                for tn in [2, 4, 8] {
                    jobs.push((
                        format!("gemmint{planes}_{tn}_{}", f.name),
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
        let built: Vec<(&'static str, Kernel)> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some((name, read_only, source)) = jobs.get(i) else { break };
                            let name: &'static str = Box::leak(name.clone().into_boxed_str());
                            out.push((name, compile_kernel(device, name, &source(), read_only)));
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
        self.kernels.extend(built);
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
        let q8 = [false, true, true, true];
        let bf = [false, true, true];
        self.add_kernel("gemv1_q8", &gemv1(W_Q8_16), &q8);
        self.add_kernel("gemv1_q4", &gemv1(W_Q4_16), &q8);
        self.add_kernel("gemv8_q4", &format!("{MM_PARAMS}{W_Q4_16}{}", gemv(GEMV_MAX_T)), &q8);
        self.add_kernel("glu_q4", &format!("{MM_PARAMS}{W_Q4_16}{GEMV_GLU}"), &q8);
        self.add_kernel("gemv8_q8", &format!("{MM_PARAMS}{W_Q8_16}{}", gemv(GEMV_MAX_T)), &q8);
        if self.caps.subgroups {
            for t in 2..=GEMVM_MAX_T {
                let r = self.tuning.gemvm_rows;
                let body = GEMV_MULTI
                    .replace("{T}", &t.to_string())
                    .replace("{R}", &r.to_string())
                    .replace("{RT}", &(r * t).to_string())
                    .replace("{T128}", &(128 * t).to_string());
                let q8_name: &'static str = Box::leak(format!("gemvm{t}_q8").into_boxed_str());
                let bf_name: &'static str = Box::leak(format!("gemvm{t}_bf16").into_boxed_str());
                let q4_name: &'static str = Box::leak(format!("gemvm{t}_q4").into_boxed_str());
                self.add_kernel(q4_name, &format!("{MM_PARAMS}{W_Q4_16}{body}"), &q8);
                self.add_kernel(q8_name, &format!("{MM_PARAMS}{W_Q8_16}{body}"), &q8);
                self.add_kernel(bf_name, &format!("{MM_PARAMS}{W_BF16_16}{body}"), &bf);
            }
        }
        for (w, wsrc, layout) in [("q8", W_Q8, &q8[..]), ("q4", W_Q4, &q8[..]), ("bf16", W_BF16, &bf[..])] {
            if self.caps.coop {
                let c = format!("enable f16;\nenable wgpu_cooperative_matrix;\n{MM_PARAMS}{wsrc}{GEMM_COOP}");
                self.add_kernel(Box::leak(format!("gemmcoop_{w}").into_boxed_str()), &c, layout);
            }
            let f32src = format!("{MM_PARAMS}{wsrc}{GEMM2}");
            self.add_kernel(Box::leak(format!("gemm2f32_{w}").into_boxed_str()), &f32src, layout);
        }
        // GGUF K-quants at their own bit widths.
        for (w, w4, w16) in
            [("q4k", W_Q4K, W_Q4K_16), ("q6", W_Q6, W_Q6_16), ("q3", W_Q3, W_Q3_16), ("q2", W_Q2, W_Q2_16)]
        {
            let name = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
            self.add_kernel(name(format!("gemv1_{w}")), &gemv1(w16), &q8);
            self.add_kernel(name(format!("gemv8_{w}")), &format!("{MM_PARAMS}{w16}{}", gemv(GEMV_MAX_T)), &q8);
            self.add_kernel(name(format!("glu_{w}")), &format!("{MM_PARAMS}{w16}{GEMV_GLU}"), &q8);
            if self.caps.subgroups {
                for t in 2..=GEMVM_MAX_T {
                    let r = self.tuning.gemvm_rows;
                    let body = GEMV_MULTI
                        .replace("{T}", &t.to_string())
                        .replace("{R}", &r.to_string())
                        .replace("{RT}", &(r * t).to_string())
                        .replace("{T128}", &(128 * t).to_string());
                    self.add_kernel(name(format!("gemvm{t}_{w}")), &format!("{MM_PARAMS}{w16}{body}"), &q8);
                }
            }
            if self.caps.coop {
                let c = format!("enable f16;\nenable wgpu_cooperative_matrix;\n{MM_PARAMS}{w4}{GEMM_COOP}");
                self.add_kernel(name(format!("gemmcoop_{w}")), &c, &q8);
            }
            self.add_kernel(name(format!("gemm2f32_{w}")), &format!("{MM_PARAMS}{w4}{GEMM2}"), &q8);
        }
        self.add_kernel("gemv1_bf16", &gemv1(W_BF16_16), &bf);
        self.add_kernel("gemv8_bf16", &format!("{MM_PARAMS}{W_BF16_16}{}", gemv(GEMV_MAX_T)), &bf);
        self.add_kernel("rmsnorm", RMSNORM, &[false, true, true]);
        self.add_kernel("add_norm", ADD_NORM, &[false, false, true, true, true]);
        if sg {
            self.add_kernel("add_norm_sg", ADD_NORM_SG, &[false, false, true, true, true]);
        }
        self.add_kernel("glu_q8", &format!("{MM_PARAMS}{W_Q8_16}{GEMV_GLU}"), &q8);
        self.add_kernel("glu_bf16", &format!("{MM_PARAMS}{W_BF16_16}{GEMV_GLU}"), &bf);
        self.add_kernel("argmax1", ARGMAX1, &[false, true]);
        self.add_kernel("argmax2", ARGMAX2, &[false, true]);
        self.add_kernel("qk_rope", QK_ROPE, &[false, false, false, true, true, true, true, true]);
        self.add_kernel("attention", ATTENTION, &[false, true, true, true, true, true]);
        self.add_kernel("embed", EMBED, &[false, true, true]);
        // (the split attention reduces with subgroup shuffles; without them the per-row
        // kernel takes every pass)
        let sg = self.caps.subgroups;
        for g in plan::SPLIT_GROUPS.into_iter().filter(|_| sg) {
            for ch in plan::SPLIT_CHUNKS {
                for hg in plan::SPLIT_HEAD_GROUPS.into_iter().filter(|h| g % h == 0) {
                    let name: &'static str = Box::leak(format!("attn_split{g}_{ch}_{hg}").into_boxed_str());
                    let src = ATTN_SPLIT
                        .replace("{GF}", &g.to_string())
                        .replace("{G}", &hg.to_string())
                        .replace("{CH}", &ch.to_string())
                        .replace("{GCH}", &(16 * ch).to_string());
                    self.add_kernel(name, &src, &[false, false, true, true, true, true, true]);
                }
            }
        }
        self.add_kernel("attn_combine", ATTN_COMBINE, &[false, true, true]);
        // The tiled kernel needs more workgroup memory than some devices (32 KiB) have;
        // without it, prefill attention takes the per-row kernel.
        if plan::tile_compiled(&self.caps) {
            self.add_kernel("attn_tile", ATTN_TILE, &[false, false, false, true, true, true, true, true, true]);
        }
        self.add_kernel("swiglu", SWIGLU, &[false, true]);
        self.add_kernel("add", ADD, &[false, true]);
        if self.diag.compile_log {
            eprintln!("[compile] {} base kernels, {:.0} ms", self.kernels.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
    }

    /// The name of a `gemm_int` kernel, compiled on first use (their sources are long, and
    /// a model uses two or three of the formats).
    pub(crate) fn int_kernel(
        &mut self,
        name: String,
        read_only: &[bool],
        source: impl FnOnce() -> String,
    ) -> &'static str {
        if let Some((k, _)) = self.kernels.get_key_value(name.as_str()) {
            return k;
        }
        let name: &'static str = Box::leak(name.into_boxed_str());
        let t0 = std::time::Instant::now();
        self.add_kernel(name, &source(), read_only);
        // (`precompile` should have covered it; a line here means a stall mid-request)
        if self.diag.compile_log {
            eprintln!("[compile] {name} on first use, {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        name
    }
}
