//! Vulkan backend (wgpu + WGSL). Weights stay packed (Q8 or bf16) and are decoded in the
//! kernels; activations and the KV cache are f32 so results track the CPU reference.
//!
//! All ops are recorded into one command encoder and submitted at sync points
//! (`argmax` / `to_host`), so a decode step is a single queue submission.

mod shaders;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use wgpu::util::DeviceExt;
use yakusel_model::{Backend, Config, LayerMeta, Quant, TensorRef, bf16_to_f32};

/// Activation rows up to which the GEMV kernel (weights read once) is used.
const GEMV_MAX_T: usize = 8;

/// Matmul work after which recorded commands are submitted. On APUs compute jobs run on
/// the graphics ring, where the kernel resets the GPU if one job runs for ~2 s (observed
/// with a 448-token prefill in one submission) and long jobs also stall the desktop.
/// 32 GFLOP is roughly 40 ms on a Radeon 780M; a 2B decode step is ~5 GFLOP.
const SUBMIT_FLOPS: f64 = 32e9;

type Pool = Arc<Mutex<HashMap<u64, Vec<wgpu::Buffer>>>>;

/// A `[rows, cols]` f32 tensor on the GPU. Its buffer returns to the pool on drop.
pub struct Tensor {
    buf: Option<wgpu::Buffer>,
    pub rows: usize,
    pub cols: usize,
    pool: Pool,
}

impl Tensor {
    fn buf(&self) -> &wgpu::Buffer {
        self.buf.as_ref().unwrap()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.pool.lock().unwrap().entry(b.size()).or_default().push(b);
        }
    }
}

pub enum Weight {
    Q8 { q: wgpu::Buffer, s: wgpu::Buffer, n: usize, k: usize },
    Q4 { q: wgpu::Buffer, s: wgpu::Buffer, n: usize, k: usize },
    Bf16 { w: wgpu::Buffer, n: usize, k: usize },
    /// 1-D weights (norms), converted to f32.
    Vector(wgpu::Buffer),
    /// Embedding table: rows are gathered on the CPU from the mapped file.
    Embedding(TensorRef),
}

struct LayerCache {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    cap: usize,
}

pub struct Cache {
    layers: Vec<LayerCache>,
    /// cos/sin tables `[2][max_len][head_dim/2]`, for sliding and full layers.
    rope_sliding: wgpu::Buffer,
    rope_full: wgpu::Buffer,
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    kernels: HashMap<&'static str, Kernel>,
    encoder: Option<wgpu::CommandEncoder>,
    /// The compute pass being recorded into `encoder`, if any.
    pass: Option<wgpu::ComputePass<'static>>,
    pool: Pool,
    /// Total bytes of uploaded weights.
    pub weight_bytes: usize,
    readback: wgpu::Buffer,
    argmax_out: wgpu::Buffer,
    /// Per-slice argmax partials `[rows][ARGMAX_SLICES]` of (value bits, index).
    argmax_part: wgpu::Buffer,
    /// Bind groups by (kernel, buffers). Pooled buffers recur every token, so most
    /// dispatches reuse one instead of creating it.
    bind_groups: HashMap<(String, Vec<wgpu::Buffer>), wgpu::BindGroup>,
    /// Matmul FLOPs recorded since the last submission.
    pending_flops: f64,
    /// Subgroup shuffles available (multi-row GEMV kernels).
    subgroups: bool,
    /// Cooperative matrices (16x16x16, f16 x f16 -> f32) available.
    coop: bool,
    /// Which prefill GEMM to use.
    pub gemm: GemmKind,
    kernels_profile: Option<KernelProfile>,
    /// Extra detail (matrix shape) for the next profiled dispatch.
    prof_detail: Option<String>,
    /// `YAKUSEL_PROFILE=1`: time spent recording vs. waiting for the GPU.
    profile: Option<Profile>,
}

/// Per-kernel GPU time from timestamp queries (`YAKUSEL_PROFILE=2`).
struct KernelProfile {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    pending: Vec<String>,
    totals: HashMap<String, (f64, usize)>,
    period_ns: f64,
}

const MAX_TIMED: usize = 2048;

#[derive(Default)]
struct Profile {
    last_sync: Option<std::time::Instant>,
    record: std::time::Duration,
    wait: std::time::Duration,
    dispatches: usize,
    syncs: usize,
}

impl Drop for WgpuBackend {
    fn drop(&mut self) {
        // End the open pass before the encoder it was recorded into goes away.
        self.pass = None;
        if let Some(k) = &self.kernels_profile
            && !k.totals.is_empty()
        {
            let mut v: Vec<_> = k.totals.iter().collect();
            v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            let total: f64 = v.iter().map(|x| x.1.0).sum();
            eprintln!("[wgpu kernels] total {:.1} ms", total / 1e6);
            for (name, (ns, n)) in v {
                eprintln!("  {name:<26} {:>9.2} ms {:>5.1}%  {n:>6} calls  {:>8.1} us/call",
                    ns / 1e6, 100.0 * ns / total, ns / 1e3 / *n as f64);
            }
        }
        if let Some(p) = &self.profile
            && p.syncs > 0
        {
            eprintln!(
                "[wgpu profile] {} syncs: record {:.2?}/sync, GPU wait {:.2?}/sync, {} dispatches/sync",
                p.syncs,
                p.record / p.syncs as u32,
                p.wait / p.syncs as u32,
                p.dispatches / p.syncs
            );
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

/// Prefill GEMM variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmKind {
    /// 64x128 tiles in f32 (CUDA-core style).
    F32,
    /// Matrix cores (cooperative matrices): f16 operands, f32 accumulation. Logits move by
    /// ~5e-3 instead of ~5e-5 against the CPU; prefill is ~1.35x faster on a 780M.
    Coop,
}

impl GemmKind {
    /// The default: matrix cores when available. `YAKUSEL_GEMM=f32|coop` overrides.
    fn from_env(coop: bool) -> Self {
        match std::env::var("YAKUSEL_GEMM").as_deref() {
            Ok("f32") => GemmKind::F32,
            _ if coop => GemmKind::Coop,
            _ => GemmKind::F32,
        }
    }
}

/// Workgroups per row in the first argmax stage (stage 2 is one 64-wide workgroup).
const ARGMAX_SLICES: usize = 64;

/// Weight rows per 32-lane group in the multi-row GEMV for `t` activation rows.
fn gemvm_rows(_t: usize) -> usize {
    std::env::var("YAKUSEL_GEMVM_R").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
}

/// Splits a 1-D workgroup count into a 2-D grid within the 65535 per-dimension limit.
fn grid(groups: u32) -> (u32, u32) {
    if groups <= 65535 { (groups.max(1), 1) } else { (65535, groups.div_ceil(65535)) }
}

impl WgpuBackend {
    /// Opens the first non-CPU Vulkan adapter.
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let adapter = adapters
            .into_iter()
            .find(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .context(
                "no Vulkan GPU found (only a software renderer, if any). Check that the user is in \
                 the `render` group and that the session has it (re-login, or run under `sg render`)",
            )?;
        let info = adapter.get_info();
        let alimits = adapter.limits();
        if !adapter.features().contains(wgpu::Features::IMMEDIATES) {
            bail!("{} does not support push constants (wgpu IMMEDIATES)", info.name);
        }
        let f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let timed = std::env::var("YAKUSEL_PROFILE").is_ok_and(|v| v == "2")
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        // The multi-row GEMV reduces 32-lane groups with subgroup shuffles.
        let subgroups = adapter.features().contains(wgpu::Features::SUBGROUP) && info.subgroup_min_size >= 32;
        // Matrix cores: 16x16x16 with f16 operands and f32 accumulators, via subgroups.
        let coop = f16
            && subgroups
            && adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            && adapter.cooperative_matrix_properties().iter().any(|c| {
                (c.m_size, c.n_size, c.k_size) == (16, 16, 16)
                    && c.ab_type == wgpu::CooperativeScalarType::F16
                    && c.cr_type == wgpu::CooperativeScalarType::F32
            });
        let limits = wgpu::Limits {
            max_buffer_size: alimits.max_buffer_size,
            max_storage_buffer_binding_size: alimits.max_storage_buffer_binding_size,
            max_storage_buffers_per_shader_stage: 8,
            max_immediate_size: 32,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("yakusel"),
            required_features: {
                let mut f = wgpu::Features::IMMEDIATES;
                if f16 {
                    f |= wgpu::Features::SHADER_F16;
                }
                if coop {
                    f |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
                }
                if subgroups {
                    f |= wgpu::Features::SUBGROUP;
                }
                if timed {
                    f |= wgpu::Features::TIMESTAMP_QUERY;
                }
                f
            },
            required_limits: limits,
            // SAFETY: only the cooperative-matrix feature is experimental here; its kernel is
            // checked against the f32 GEMM and the CPU reference by the test suite.
            experimental_features: if coop {
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            },
            ..Default::default()
        }))
        .context("creating the wgpu device")?;

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let argmax_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("argmax"),
            size: 512,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let kernels_profile = timed.then(|| {
            let size = (MAX_TIMED * 2 * 8) as u64;
            KernelProfile {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: (MAX_TIMED * 2) as u32,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                staging: device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                pending: Vec::new(),
                totals: HashMap::new(),
                period_ns: queue.get_timestamp_period() as f64,
            }
        });
        let argmax_part = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("argmax-part"),
            size: (64 * ARGMAX_SLICES * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut b = WgpuBackend {
            argmax_part,
            bind_groups: HashMap::new(),
            kernels_profile,
            prof_detail: None,
            device,
            queue,
            adapter_name: format!("{} ({:?})", info.name, info.backend),
            kernels: HashMap::new(),
            encoder: None,
            pass: None,
            pool: Arc::new(Mutex::new(HashMap::new())),
            weight_bytes: 0,
            readback,
            argmax_out,
            pending_flops: 0.0,
            subgroups,
            coop,
            gemm: GemmKind::from_env(coop),
            profile: std::env::var_os("YAKUSEL_PROFILE").map(|_| Profile::default()),
        };
        b.build_kernels();
        Ok(b)
    }

    fn add_kernel(&mut self, name: &'static str, src: &str, read_only: &[bool]) {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let entries: Vec<_> = read_only.iter().enumerate().map(|(i, &r)| storage_entry(i as u32, r)).collect();
        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(name),
            entries: &entries,
        });
        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 32,
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.kernels.insert(name, Kernel { pipeline, layout });
    }

    fn build_kernels(&mut self) {
        use shaders::*;
        let gemv = |t: usize| GEMV.replace("{MAXT}", &t.to_string()).replace("{RED}", &(t * 256).to_string());
        let q8 = [false, true, true, true];
        let bf = [false, true, true];
        self.add_kernel("gemv1_q8", &format!("{MM_PARAMS}{W_Q8_16}{}", gemv(1)), &q8);
        self.add_kernel("gemv1_q4", &format!("{MM_PARAMS}{W_Q4_16}{}", gemv(1)), &q8);
        self.add_kernel("gemv8_q4", &format!("{MM_PARAMS}{W_Q4_16}{}", gemv(GEMV_MAX_T)), &q8);
        self.add_kernel("glu_q4", &format!("{MM_PARAMS}{W_Q4_16}{GEMV_GLU}"), &q8);
        self.add_kernel("gemv8_q8", &format!("{MM_PARAMS}{W_Q8_16}{}", gemv(GEMV_MAX_T)), &q8);
        if self.subgroups {
            for t in 2..=GEMV_MAX_T {
                let r = gemvm_rows(t);
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
            if self.coop {
                let c = format!("enable f16;\nenable wgpu_cooperative_matrix;\n{MM_PARAMS}{wsrc}{GEMM_COOP}");
                self.add_kernel(Box::leak(format!("gemmcoop_{w}").into_boxed_str()), &c, layout);
            }
            let f32src = format!("{MM_PARAMS}{wsrc}{GEMM2}");
            self.add_kernel(Box::leak(format!("gemm2f32_{w}").into_boxed_str()), &f32src, layout);
        }
        self.add_kernel("gemv1_bf16", &format!("{MM_PARAMS}{W_BF16_16}{}", gemv(1)), &bf);
        self.add_kernel("gemv8_bf16", &format!("{MM_PARAMS}{W_BF16_16}{}", gemv(GEMV_MAX_T)), &bf);
        self.add_kernel("rmsnorm", RMSNORM, &[false, true, true]);
        self.add_kernel("add_norm", ADD_NORM, &[false, false, true, true, true]);
        self.add_kernel("glu_q8", &format!("{MM_PARAMS}{W_Q8_16}{GEMV_GLU}"), &q8);
        self.add_kernel("glu_bf16", &format!("{MM_PARAMS}{W_BF16_16}{GEMV_GLU}"), &bf);
        self.add_kernel("argmax1", ARGMAX1, &[false, true]);
        self.add_kernel("argmax2", ARGMAX2, &[false, true]);
        self.add_kernel("qk_rope", QK_ROPE, &[false, false, false, true, true, true, true]);
        self.add_kernel("attention", ATTENTION, &[false, true, true, true]);
        self.add_kernel("swiglu", SWIGLU, &[false, true]);
        self.add_kernel("add", ADD, &[false, true]);
    }

    fn tensor(&mut self, rows: usize, cols: usize) -> Tensor {
        self.tensor_padded(rows, cols, rows)
    }

    /// A tensor whose buffer has room for `cap_rows >= rows` rows.
    fn tensor_padded(&mut self, rows: usize, cols: usize, cap_rows: usize) -> Tensor {
        let size = ((cap_rows * cols * 4) as u64).next_power_of_two().max(256);
        let buf = self.pool.lock().unwrap().get_mut(&size).and_then(Vec::pop).unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Tensor { buf: Some(buf), rows, cols, pool: self.pool.clone() }
    }

    /// The encoder for non-dispatch commands; ends the open compute pass first.
    fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.pass = None;
        let device = &self.device;
        self.encoder.get_or_insert_with(|| device.create_command_encoder(&Default::default()))
    }

    fn dispatch(&mut self, name: &str, bufs: &[&wgpu::Buffer], params: &[u32], groups: (u32, u32)) {
        let k = &self.kernels[name];
        let key = (name.to_string(), bufs.iter().map(|b| (*b).clone()).collect::<Vec<_>>());
        let bg = match self.bind_groups.get(&key) {
            Some(bg) => bg.clone(),
            None => {
                let entries: Vec<_> = bufs
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                    .collect();
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &k.layout,
                    entries: &entries,
                });
                // Per-request buffers (KV cache, rope tables) make old entries stale.
                if self.bind_groups.len() >= 4096 {
                    self.bind_groups.clear();
                }
                self.bind_groups.insert(key, bg.clone());
                bg
            }
        };
        let pipeline = k.pipeline.clone();
        let mut imm = [0u32; 8];
        imm[..params.len()].copy_from_slice(params);
        if let Some(p) = &mut self.profile {
            p.dispatches += 1;
        }
        let timing = self.kernels_profile.as_mut().and_then(|k| {
            (k.pending.len() < MAX_TIMED).then(|| {
                let i = k.pending.len() as u32;
                let key = match self.prof_detail.take() {
                    Some(d) => format!("{name} {d}"),
                    None => name.to_string(),
                };
                k.pending.push(key);
                (k.queries.clone(), i)
            })
        });
        // Consecutive dispatches share one compute pass (wgpu still inserts the barriers
        // between dependent dispatches). Timestamp profiling needs a pass per dispatch.
        if timing.is_some() || self.kernels_profile.is_some() {
            self.pass = None;
        }
        if self.pass.is_none() {
            let device = &self.device;
            let enc = self.encoder.get_or_insert_with(|| device.create_command_encoder(&Default::default()));
            let pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: timing.as_ref().map(|(q, i)| wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(2 * i),
                    end_of_pass_write_index: Some(2 * i + 1),
                }),
            });
            self.pass = Some(pass.forget_lifetime());
        }
        let pass = self.pass.as_mut().unwrap();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_immediates(0, bytemuck::cast_slice(&imm));
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    fn submit(&mut self) {
        self.pass = None;
        self.pending_flops = 0.0;
        if let Some(enc) = self.encoder.take() {
            self.queue.submit([enc.finish()]);
        }
    }

    fn read(&mut self, src: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        let staging = if size <= self.readback.size() {
            self.readback.clone()
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback-large"),
                size: size.next_multiple_of(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        self.encoder().copy_buffer_to_buffer(src, offset, &staging, 0, size);
        let timed_n = self.kernels_profile.as_ref().map_or(0, |k| k.pending.len());
        if timed_n > 0 {
            let k = self.kernels_profile.as_ref().unwrap();
            let (q, r, s) = (k.queries.clone(), k.resolve.clone(), k.staging.clone());
            let enc = self.encoder();
            enc.resolve_query_set(&q, 0..(2 * timed_n) as u32, &r, 0);
            enc.copy_buffer_to_buffer(&r, 0, &s, 0, (timed_n * 16) as u64);
        }
        let t_rec = std::time::Instant::now();
        self.submit();
        staging.map_async(wgpu::MapMode::Read, ..size, |r| r.expect("map failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        if let Some(p) = &mut self.profile {
            let now = std::time::Instant::now();
            if let Some(last) = p.last_sync {
                p.record += t_rec - last;
                p.wait += now - t_rec;
                p.syncs += 1;
            }
            p.last_sync = Some(now);
        }
        let out = staging.get_mapped_range(..size).expect("mapped").to_vec();
        staging.unmap();
        if timed_n > 0 {
            let k = self.kernels_profile.as_mut().unwrap();
            let bytes = (timed_n * 16) as u64;
            k.staging.map_async(wgpu::MapMode::Read, ..bytes, |r| r.expect("map failed"));
            self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
            let ts: Vec<u64> = bytemuck::cast_slice(&k.staging.get_mapped_range(..bytes).expect("mapped")).to_vec();
            k.staging.unmap();
            for (i, name) in k.pending.drain(..).enumerate() {
                let ns = ts[2 * i + 1].saturating_sub(ts[2 * i]) as f64 * k.period_ns;
                let e = k.totals.entry(name).or_default();
                e.0 += ns;
                e.1 += 1;
            }
        }
        out
    }

    fn upload(&mut self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        self.weight_bytes += bytes.len();
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn rope_table(&self, meta: &LayerMeta, max_len: usize, d: usize) -> wgpu::Buffer {
        let half = d / 2;
        let mut tab = vec![0f32; 2 * max_len * half];
        let (cos, sin) = tab.split_at_mut(max_len * half);
        for pos in 0..max_len {
            meta.rope.cos_sin(pos, &mut cos[pos * half..(pos + 1) * half], &mut sin[pos * half..(pos + 1) * half]);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope"),
            contents: bytemuck::cast_slice(&tab),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }
}

impl Backend for WgpuBackend {
    type Tensor = Tensor;
    type Weight = Weight;
    type Cache = Cache;

    fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    fn name(&self) -> String {
        format!("vulkan: {}", self.adapter_name)
    }

    fn load_embedding(&mut self, t: &TensorRef) -> Result<Weight> {
        Ok(Weight::Embedding(t.clone()))
    }

    fn load_weight(&mut self, t: &TensorRef, quant: Quant) -> Result<Weight> {
        if t.shape.len() == 1 {
            let v = t.to_f32_vec();
            return Ok(Weight::Vector(self.upload("vector", bytemuck::cast_slice(&v))));
        }
        let (n, k) = (t.shape[0], t.shape[1]);
        let raw;
        let bf16: &[u16] = match t.as_bf16() {
            Some(b) => b,
            None => {
                raw = t.to_bf16_vec();
                &raw
            }
        };
        self.load_matrix(bf16, n, k, quant)
    }

    fn new_cache(&mut self, cfg: &Config, metas: &[LayerMeta], max_len: usize, chunk: usize) -> Result<Cache> {
        anyhow::ensure!(cfg.head_dim == 128, "the WGSL kernels assume head_dim = 128");
        let kvd = cfg.kv_dim();
        let mut layers = Vec::new();
        for m in metas {
            let cap = match m.window {
                Some(w) => max_len.min(w + chunk),
                None => max_len,
            };
            let mk = || {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("kv"),
                    size: (cap * kvd * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            layers.push(LayerCache { k: mk(), v: mk(), cap });
        }
        let sliding = metas.iter().find(|m| !m.full_attn).context("no sliding layer")?;
        let full = metas.iter().find(|m| m.full_attn).unwrap_or(sliding);
        Ok(Cache {
            layers,
            rope_sliding: self.rope_table(sliding, max_len, cfg.head_dim),
            rope_full: self.rope_table(full, max_len, cfg.head_dim),
        })
    }

    fn embed(&mut self, table: &Weight, ids: &[u32]) -> Tensor {
        let Weight::Embedding(t) = table else { panic!("embed needs the embedding table") };
        let h = t.shape[1];
        let bytes = t.bytes();
        let mut rows = vec![0f32; ids.len() * h];
        for (r, &id) in ids.iter().enumerate() {
            let src = &bytes[id as usize * h * 2..(id as usize + 1) * h * 2];
            for (o, b) in rows[r * h..(r + 1) * h].iter_mut().zip(src.chunks_exact(2)) {
                *o = bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        // `write_buffer` lands before the commands of the next submission, so submit what
        // is recorded first: a pooled buffer may still be used by those commands.
        self.submit();
        let y = self.tensor(ids.len(), h);
        self.queue.write_buffer(y.buf(), 0, bytemuck::cast_slice(&rows));
        y
    }

    fn rmsnorm(&mut self, x: &Tensor, w: &Weight, offset: f32, eps: f32) -> Tensor {
        let Weight::Vector(wb) = w else { panic!("rmsnorm weight must be a vector") };
        let y = self.tensor(x.rows, x.cols);
        self.dispatch(
            "rmsnorm",
            &[y.buf(), x.buf(), wb],
            &[x.cols as u32, offset.to_bits(), eps.to_bits()],
            (x.rows as u32, 1),
        );
        y
    }

    fn matmul(&mut self, x: &Tensor, w: &Weight) -> Tensor {
        let (n, k, suffix, bufs): (usize, usize, &str, Vec<&wgpu::Buffer>) = match w {
            Weight::Q8 { q, s, n, k } => (*n, *k, "q8", vec![q, s]),
            Weight::Q4 { q, s, n, k } => (*n, *k, "q4", vec![q, s]),
            Weight::Bf16 { w, n, k } => (*n, *k, "bf16", vec![w]),
            _ => panic!("matmul needs a matrix"),
        };
        assert_eq!(x.cols, k, "matmul inner dim");
        let t = x.rows;
        // The matrix-core GEMM stores whole 64-token tiles.
        let y = if t > GEMV_MAX_T && self.gemm == GemmKind::Coop {
            self.tensor_padded(t, n, t.div_ceil(64) * 64)
        } else {
            self.tensor(t, n)
        };
        let mut all = vec![y.buf(), x.buf()];
        all.extend(bufs);
        let params = [n as u32, k as u32, t as u32];
        let (name, groups) = if t == 1 {
            (format!("gemv1_{suffix}"), grid(n.div_ceil(8) as u32))
        } else if t <= GEMV_MAX_T && self.subgroups && k % 512 == 0 {
            (format!("gemvm{t}_{suffix}"), grid(n.div_ceil(8 * gemvm_rows(t)) as u32))
        } else if t <= GEMV_MAX_T {
            (format!("gemv8_{suffix}"), grid(n.div_ceil(8) as u32))
        } else {
            match self.gemm {
                GemmKind::Coop if n % 64 == 0 => {
                    (format!("gemmcoop_{suffix}"), ((n / 64) as u32, t.div_ceil(64) as u32))
                }
                _ => (format!("gemm2f32_{suffix}"), (n.div_ceil(128) as u32, t.div_ceil(64) as u32)),
            }
        };
        // `all` borrows `y`; dispatch only needs the buffers for the bind group.
        let all: Vec<wgpu::Buffer> = all.into_iter().cloned().collect();
        let refs: Vec<&wgpu::Buffer> = all.iter().collect();
        if self.kernels_profile.is_some() {
            self.prof_detail = Some(format!("{n}x{k}"));
        }
        self.dispatch(&name, &refs, &params, groups);
        self.pending_flops += 2.0 * (t * n * k) as f64;
        if self.pending_flops >= SUBMIT_FLOPS {
            self.submit();
        }
        y
    }

    fn attention(
        &mut self,
        qkv: &Tensor,
        q_norm: &Weight,
        k_norm: &Weight,
        cache: &mut Cache,
        cfg: &Config,
        layer: &LayerMeta,
        pos0: usize,
    ) -> Tensor {
        let (Weight::Vector(qn), Weight::Vector(kn)) = (q_norm, k_norm) else { panic!("q/k norm") };
        let (hq, hkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let t = qkv.rows;
        let lc = &cache.layers[layer.index];
        assert!(t <= lc.cap, "chunk larger than cache capacity");
        let (kc, vc, cap) = (lc.k.clone(), lc.v.clone(), lc.cap as u32);
        let rope = if layer.full_attn { cache.rope_full.clone() } else { cache.rope_sliding.clone() };
        assert!((pos0 + t) as u64 * 64 * 4 * 2 <= rope.size(), "position beyond the rope table");
        let q = self.tensor(t, hq * 128);
        self.dispatch(
            "qk_rope",
            &[q.buf(), &kc, &vc, qkv.buf(), qn, kn, &rope],
            &[hq as u32, hkv as u32, pos0 as u32, cap, cfg.rms_norm_eps.to_bits()],
            ((hq + hkv) as u32, t as u32),
        );
        let o = self.tensor(t, hq * 128);
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        self.dispatch(
            "attention",
            &[o.buf(), q.buf(), &kc, &vc],
            &[hq as u32, hkv as u32, pos0 as u32, cap, layer.window.unwrap_or(0) as u32, scale.to_bits()],
            (hq as u32, t as u32),
        );
        o
    }

    fn add_norm(
        &mut self,
        x: &mut Tensor,
        delta: &Tensor,
        post_w: &Weight,
        post_offset: f32,
        pre_w: &Weight,
        pre_offset: f32,
        eps: f32,
    ) -> Tensor {
        let (Weight::Vector(w1), Weight::Vector(w2)) = (post_w, pre_w) else { panic!("norm weights") };
        assert_eq!((x.rows, x.cols), (delta.rows, delta.cols));
        let h = self.tensor(x.rows, x.cols);
        let xb = x.buf().clone();
        self.dispatch(
            "add_norm",
            &[h.buf(), &xb, delta.buf(), w1, w2],
            &[x.cols as u32, post_offset.to_bits(), pre_offset.to_bits(), eps.to_bits()],
            (x.rows as u32, 1),
        );
        h
    }

    fn matmul_glu(&mut self, x: &Tensor, w: &Weight) -> Tensor {
        if x.rows != 1 {
            let gu = self.matmul(x, w);
            return self.swiglu(&gu);
        }
        let (n2, k, name, bufs): (usize, usize, &str, Vec<wgpu::Buffer>) = match w {
            Weight::Q8 { q, s, n, k } => (*n, *k, "glu_q8", vec![q.clone(), s.clone()]),
            Weight::Q4 { q, s, n, k } => (*n, *k, "glu_q4", vec![q.clone(), s.clone()]),
            Weight::Bf16 { w, n, k } => (*n, *k, "glu_bf16", vec![w.clone()]),
            _ => panic!("matmul_glu needs a matrix"),
        };
        assert_eq!(x.cols, k);
        let inter = n2 / 2;
        let y = self.tensor(1, inter);
        let mut all = vec![y.buf().clone(), x.buf().clone()];
        all.extend(bufs);
        let refs: Vec<&wgpu::Buffer> = all.iter().collect();
        if self.kernels_profile.is_some() {
            self.prof_detail = Some(format!("{n2}x{k}"));
        }
        self.dispatch(name, &refs, &[inter as u32, k as u32, 1], grid(inter.div_ceil(4) as u32));
        self.pending_flops += 2.0 * (n2 * k) as f64;
        y
    }

    fn swiglu(&mut self, x: &Tensor) -> Tensor {
        let inter = x.cols / 2;
        let y = self.tensor(x.rows, inter);
        let total = x.rows * inter;
        self.dispatch("swiglu", &[y.buf(), x.buf()], &[inter as u32, total as u32], grid(total.div_ceil(256) as u32));
        y
    }

    fn add_assign(&mut self, a: &mut Tensor, b: &Tensor) {
        let total = a.rows * a.cols;
        assert_eq!(total, b.rows * b.cols);
        let (ab, bb) = (a.buf().clone(), b.buf().clone());
        self.dispatch("add", &[&ab, &bb], &[total as u32], grid(total.div_ceil(256) as u32));
    }

    fn last_row(&mut self, x: &Tensor) -> Tensor {
        let y = self.tensor(1, x.cols);
        let bytes = (x.cols * 4) as u64;
        let (src, dst) = (x.buf().clone(), y.buf().clone());
        self.encoder().copy_buffer_to_buffer(&src, (x.rows as u64 - 1) * bytes, &dst, 0, bytes);
        y
    }

    fn argmax(&mut self, logits: &Tensor) -> u32 {
        // Only the last row matters; the rows are independent, so take them all and pick.
        *self.argmax_rows(logits).last().unwrap()
    }

    fn argmax_prob(&mut self, logits: &Tensor) -> (u32, f32) {
        *self.argmax_prob_rows(logits).last().unwrap()
    }

    fn argmax_rows(&mut self, logits: &Tensor) -> Vec<u32> {
        self.argmax_prob_rows(logits).into_iter().map(|(i, _)| i).collect()
    }

    fn to_host(&mut self, x: &Tensor) -> Vec<f32> {
        let size = (x.rows * x.cols * 4) as u64;
        let b = self.read(&x.buf().clone(), 0, size);
        bytemuck::cast_slice(&b).to_vec()
    }
}

impl WgpuBackend {
    /// Greedy token and its softmax probability for every row (two-stage reduction).
    fn argmax_prob_rows(&mut self, logits: &Tensor) -> Vec<(u32, f32)> {
        assert!(logits.rows <= 64, "argmax_rows supports up to 64 rows");
        let (out, part) = (self.argmax_out.clone(), self.argmax_part.clone());
        let slice = logits.cols.div_ceil(ARGMAX_SLICES);
        self.dispatch(
            "argmax1",
            &[&part, logits.buf()],
            &[logits.cols as u32, slice as u32],
            (ARGMAX_SLICES as u32, logits.rows as u32),
        );
        self.dispatch("argmax2", &[&out, &part], &[ARGMAX_SLICES as u32], (logits.rows as u32, 1));
        let b = self.read(&out, 0, logits.rows as u64 * 8);
        let v: &[u32] = bytemuck::cast_slice(&b);
        v.chunks_exact(2).map(|c| (c[0], f32::from_bits(c[1]))).collect()
    }
}

impl WgpuBackend {
    fn load_matrix(&mut self, bf16: &[u16], n: usize, k: usize, quant: Quant) -> Result<Weight> {
        anyhow::ensure!(k % 32 == 0, "K={k} must be a multiple of 32");
        Ok(match quant {
            Quant::Q8 => {
                let q = yakusel_model::quant::quantize_q8(bf16, k);
                let qb = self.upload("q8", bytemuck::cast_slice(&q.qs));
                let mut sc: Vec<u16> = q.scales.iter().map(|s| s.to_bits()).collect();
                if sc.len() % 2 == 1 {
                    sc.push(0);
                }
                let sb = self.upload("q8-scales", bytemuck::cast_slice(&sc));
                Weight::Q8 { q: qb, s: sb, n, k }
            }
            Quant::Q4 => {
                let q = yakusel_model::quant::quantize_q4(bf16, k);
                let qb = self.upload("q4", &q.qs);
                let mut sc: Vec<u16> = q.scales.iter().map(|s| s.to_bits()).collect();
                if sc.len() % 2 == 1 {
                    sc.push(0);
                }
                let sb = self.upload("q4-scales", bytemuck::cast_slice(&sc));
                Weight::Q4 { q: qb, s: sb, n, k }
            }
            Quant::Bf16 => Weight::Bf16 { w: self.upload("bf16", bytemuck::cast_slice(bf16)), n, k },
        })
    }
}

impl WgpuBackend {
    /// Whether the prefill GEMM rounds its operands to f16.
    #[doc(hidden)]
    pub fn gemm_uses_f16(&self) -> bool {
        self.gemm == GemmKind::Coop
    }

    /// Test hook: `x [t, k] · W^T` (or `swiglu` of it with `glu`) for a bf16 matrix `W [n, k]`
    /// stored as `quant`, through the same kernels the model uses.
    #[doc(hidden)]
    pub fn debug_matmul(&mut self, x: &[f32], t: usize, w: &[u16], n: usize, k: usize, quant: Quant, glu: bool) -> Vec<f32> {
        let w = self.load_matrix(w, n, k, quant).unwrap();
        let xt = self.tensor(t, k);
        self.queue.write_buffer(xt.buf(), 0, bytemuck::cast_slice(x));
        let y = if glu { self.matmul_glu(&xt, &w) } else { self.matmul(&xt, &w) };
        self.to_host(&y)
    }
}
