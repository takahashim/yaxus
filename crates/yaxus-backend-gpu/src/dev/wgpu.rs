//! The device through wgpu: Vulkan, or Metal on macOS.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use wgpu::util::DeviceExt;

use super::Buf;
use crate::device::{Api, CompileSettings, DeviceCaps, DeviceKind, Diagnostics, Vendor};
use crate::profile::{KernelProfile, MAX_TIMED};

/// Weight bytes uploaded between flushes of the staging memory.
const UPLOAD_FLUSH_BYTES: usize = 512 << 20;

pub(crate) struct WgpuKernel {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
}

pub(crate) struct WgpuDev {
    pub(crate) device: wgpu::Device,
    queue: wgpu::Queue,
    encoder: Option<wgpu::CommandEncoder>,
    /// The compute pass being recorded into `encoder`, if any.
    pass: Option<wgpu::ComputePass<'static>>,
    /// Bind groups by (kernel, buffers). Pooled buffers recur every token, so most
    /// dispatches reuse one instead of creating it.
    bind_groups: HashMap<(usize, Vec<wgpu::Buffer>), wgpu::BindGroup>,
    readback: wgpu::Buffer,
    /// Bytes uploaded since the staging memory was last released.
    unflushed_upload: usize,
    pub(crate) kernels_profile: Option<KernelProfile>,
    /// WebGPU only: there are no push constants (`immediates`), so the per-dispatch
    /// scalars go into a slot of this ring uniform buffer, one slot per dispatch.
    #[cfg(target_arch = "wasm32")]
    immediates: wgpu::Buffer,
    #[cfg(target_arch = "wasm32")]
    imm_i: usize,
}

/// Uniform ring slots: one per dispatch recorded between submissions (reset at each).
/// A slot is 256 bytes apart, the minimum uniform buffer offset alignment.
#[cfg(target_arch = "wasm32")]
const IMMEDIATE_SLOTS: usize = 4096;

impl Drop for WgpuDev {
    fn drop(&mut self) {
        // End the open pass before the encoder it was recorded into goes away.
        self.pass = None;
        if let Some(k) = &self.kernels_profile {
            k.report();
        }
    }
}

fn wb(b: &Buf) -> &wgpu::Buffer {
    match b {
        Buf::Wgpu(b) => b,
        #[allow(unreachable_patterns)]
        _ => panic!("a buffer of another device"),
    }
}

/// Allocation and validation errors captured around a closure (see `Gpu::checked`).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct ErrorScope {
    oom: wgpu::ErrorScopeGuard,
    invalid: wgpu::ErrorScopeGuard,
}

#[cfg(not(target_arch = "wasm32"))]
impl ErrorScope {
    pub(crate) fn finish(self) -> Option<String> {
        let invalid = pollster::block_on(self.invalid.pop());
        let oom = pollster::block_on(self.oom.pop());
        oom.or(invalid).map(|e| e.to_string())
    }
}

impl WgpuDev {
    /// Opens the first GPU of the platform's API (Vulkan; Metal on macOS).
    /// `YAXUS_SOFTWARE_GPU=1` accepts a software adapter too (lavapipe), so the kernel tests
    /// can run on machines without a GPU (CI).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn open(diag: &Diagnostics) -> Result<(WgpuDev, DeviceCaps)> {
        let backends = platform_backends();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(backends));
        let adapter = adapters
            .into_iter()
            .find(|a| diag.software || a.get_info().device_type != wgpu::DeviceType::Cpu)
            .with_context(|| {
                let hint = if cfg!(target_os = "linux") {
                    ". Check that the user is in the `render` group and that the session has it \
                     (re-login, or run under `sg render`)"
                } else {
                    ""
                };
                format!("no {backends:?} GPU found (only a software renderer, if any){hint}")
            })?;
        let alimits = adapter.limits();
        let caps = detect(&adapter);
        if !adapter.features().contains(wgpu::Features::IMMEDIATES) {
            bail!("{} does not support push constants (wgpu IMMEDIATES)", caps.name);
        }
        let timed = diag.profile == Some(2) && caps.timestamps;
        let (f16, coop, subgroups) = (caps.f16, caps.coop, caps.subgroups);
        let limits = wgpu::Limits {
            max_buffer_size: alimits.max_buffer_size,
            max_storage_buffer_binding_size: alimits.max_storage_buffer_binding_size,
            max_storage_buffers_per_shader_stage: 10,
            max_compute_workgroup_storage_size: caps.max_workgroup_storage,
            max_immediate_size: 32,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("yaxus"),
            required_features: {
                let mut f = wgpu::Features::IMMEDIATES;
                if caps.msl {
                    f |= wgpu::Features::PASSTHROUGH_SHADERS;
                }
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
        let d = WgpuDev {
            device,
            queue,
            encoder: None,
            pass: None,
            bind_groups: HashMap::new(),
            readback,
            unflushed_upload: 0,
            kernels_profile,
        };
        Ok((d, caps))
    }

    /// The encoder for non-dispatch commands; ends the open compute pass first.
    fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.pass = None;
        let device = &self.device;
        self.encoder.get_or_insert_with(|| device.create_command_encoder(&Default::default()))
    }

    pub(crate) fn storage(&self, label: &str, size: u64) -> Buf {
        Buf::Wgpu(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    pub(crate) fn upload(&mut self, label: &str, bytes: &[u8]) -> Buf {
        // Not create_buffer_init: when the allocation fails (out of memory, above
        // max_buffer_size) it panics mapping the invalid buffer, before an error scope
        // can report it. A plain buffer and a queued write report the error instead.
        let size = (bytes.len().max(4) as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[cfg(not(target_arch = "wasm32"))]
        if bytes.len() as u64 == size {
            self.queue.write_buffer(&buf, 0, bytes);
        } else {
            let mut padded = bytes.to_vec();
            padded.resize(size as usize, 0);
            self.queue.write_buffer(&buf, 0, &padded);
        }
        // WebGPU: write in chunks, so the browser does not hold a second copy of a
        // whole tensor (the largest is 136 MB) in its staging memory at once.
        #[cfg(target_arch = "wasm32")]
        {
            const CHUNK: usize = 16 << 20;
            let mut off = 0usize;
            while off < bytes.len() {
                let end = (off + CHUNK).min(bytes.len());
                self.queue.write_buffer(&buf, off as u64, &bytes[off..end]);
                off = end;
            }
        }
        // Initial contents go through staging memory that is only released after the
        // next submission; flush regularly so loading does not hold a second copy of
        // the weights (an 8B model in bf16 would exceed the GTT limit).
        self.unflushed_upload += bytes.len();
        if self.unflushed_upload >= UPLOAD_FLUSH_BYTES {
            self.unflushed_upload = 0;
            self.queue.submit([]);
            #[cfg(not(target_arch = "wasm32"))]
            self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
            // (on WebGPU, wait acts as poll; the staging memory is released when the
            // queue drains, which the page's event loop turns take care of)
            #[cfg(target_arch = "wasm32")]
            self.device.poll(wgpu::PollType::Poll).expect("device lost");
        }
        Buf::Wgpu(buf)
    }

    pub(crate) fn write(&mut self, dst: &Buf, offset: u64, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let staging = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("write-staging"),
            contents: bytes,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let dst = wb(dst).clone();
        self.encoder().copy_buffer_to_buffer(&staging, 0, &dst, offset, bytes.len() as u64);
    }

    /// The pass metadata: one buffer per slot, grown when too small, written through the
    /// command stream (so after the previous pass's commands that read it).
    pub(crate) fn upload_meta(&mut self, slot: &mut Option<Buf>, data: &[u8]) {
        let len = (data.len() as u64).max(4).next_multiple_of(4);
        if slot.as_ref().is_none_or(|b| b.size() < len) {
            let size = len.next_power_of_two().max(4096);
            *slot = Some(self.storage("pass-meta", size));
        }
        let dst = slot.clone().unwrap();
        self.write(&dst, 0, data);
    }

    pub(crate) fn copy(&mut self, src: &Buf, src_off: u64, dst: &Buf, dst_off: u64, size: u64) {
        let (src, dst) = (wb(src).clone(), wb(dst).clone());
        self.encoder().copy_buffer_to_buffer(&src, src_off, &dst, dst_off, size);
    }

    pub(crate) fn dispatch(
        &mut self,
        id: usize,
        k: &WgpuKernel,
        bufs: &[&Buf],
        imm: &[u32; 8],
        groups: (u32, u32),
        label: Option<String>,
    ) {
        let key = (id, bufs.iter().map(|b| wb(b).clone()).collect::<Vec<_>>());
        let bg = match self.bind_groups.get(&key) {
            Some(bg) => bg.clone(),
            None => {
                let entries: Vec<_> = key
                    .1
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                    .collect();
                #[cfg(target_arch = "wasm32")]
                let entries = {
                    let mut e = entries;
                    e.push(wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.immediates,
                            offset: 0,
                            size: Some(std::num::NonZeroU64::new(32).unwrap()),
                        }),
                    });
                    e
                };
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
        let timing = self.kernels_profile.as_mut().and_then(|kp| {
            (kp.pending.len() < MAX_TIMED).then(|| {
                let i = kp.pending.len() as u32;
                kp.pending.push(label.unwrap_or_default());
                (kp.queries.clone(), i)
            })
        });
        // Consecutive dispatches share one compute pass (wgpu still inserts the barriers
        // between dependent dispatches). Timestamp profiling needs a pass per dispatch.
        if self.kernels_profile.is_some() {
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
        pass.set_pipeline(&k.pipeline);
        #[cfg(not(target_arch = "wasm32"))]
        {
            pass.set_bind_group(0, &bg, &[]);
            pass.set_immediates(0, bytemuck::cast_slice(imm));
        }
        // WebGPU has no push constants: the scalars go through the ring's next slot
        // (the write is applied in queue order, before this submission's commands).
        #[cfg(target_arch = "wasm32")]
        {
            let slot = (self.imm_i * 256) as u64;
            self.imm_i += 1;
            assert!(self.imm_i <= IMMEDIATE_SLOTS, "the immediates ring is exhausted in one submission");
            self.queue.write_buffer(&self.immediates, slot, bytemuck::cast_slice(imm));
            pass.set_bind_group(0, &bg, &[slot as u32]);
        }
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    pub(crate) fn submit(&mut self) {
        self.pass = None;
        #[cfg(target_arch = "wasm32")]
        {
            self.imm_i = 0;
        }
        if let Some(enc) = self.encoder.take() {
            self.queue.submit([enc.finish()]);
        }
    }

    /// The async counterpart of `read` for WebGPU: the map callback comes from the
    /// event loop, so the read-back awaits event-loop turns while polling. (Kernel
    /// timing does not exist there.)
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn read_async(&mut self, src: &Buf, offset: u64, size: u64) -> Vec<u8> {
        let src = wb(src).clone();
        let staging = self.readback_staging(size);
        self.encoder().copy_buffer_to_buffer(&src, offset, &staging, 0, size);
        self.submit();
        let mapped = std::rc::Rc::new(std::cell::Cell::new(false));
        {
            let mapped = mapped.clone();
            staging.map_async(wgpu::MapMode::Read, ..size, move |r| {
                if r.is_err() {
                    web_sys::console::error_1(&"map_async failed".into());
                }
                mapped.set(true);
            });
        }
        for _ in 0..600 {
            self.device.poll(wgpu::PollType::Poll).expect("device lost");
            if mapped.get() {
                break;
            }
            event_loop_turn().await;
        }
        assert!(mapped.get(), "read-back did not map in time");
        let out = staging.get_mapped_range(..size).expect("mapped").to_vec();
        staging.unmap();
        out
    }

    /// The MAP_READ buffer for reading `size` bytes: the pooled small one, or a fresh
    /// large one.
    fn readback_staging(&mut self, size: u64) -> wgpu::Buffer {
        if size <= self.readback.size() {
            self.readback.clone()
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback-large"),
                size: size.next_multiple_of(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn read(&mut self, src: &Buf, offset: u64, size: u64) -> Vec<u8> {
        let src = wb(src).clone();
        let staging = self.readback_staging(size);
        self.encoder().copy_buffer_to_buffer(&src, offset, &staging, 0, size);
        let timed_n = self.kernels_profile.as_ref().map_or(0, |k| k.pending.len());
        if timed_n > 0 {
            let k = self.kernels_profile.as_ref().unwrap();
            let (q, r, s) = (k.queries.clone(), k.resolve.clone(), k.staging.clone());
            let enc = self.encoder();
            enc.resolve_query_set(&q, 0..(2 * timed_n) as u32, &r, 0);
            enc.copy_buffer_to_buffer(&r, 0, &s, 0, (timed_n * 16) as u64);
        }
        self.submit();
        staging.map_async(wgpu::MapMode::Read, ..size, |r| r.expect("map failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
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

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn error_scope(&self) -> ErrorScope {
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let invalid = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        ErrorScope { oom, invalid }
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

pub(crate) fn compile(
    device: &wgpu::Device,
    settings: CompileSettings,
    name: &str,
    src: &str,
    read_only: &[bool],
) -> WgpuKernel {
    // WebGPU: port the push constants to a uniform buffer at binding 9 (every kernel
    // has at most 9 buffer bindings), and add the `enable` directives Chrome requires
    // (native wgpu enables f16 and subgroups from the device features).
    #[cfg(target_arch = "wasm32")]
    let src = {
        let s = src.replace("var<immediate> p: P;", "@group(0) @binding(9) var<uniform> p: P;");
        if s.starts_with("enable ") {
            s
        } else {
            format!("enable subgroups;\nenable f16;\n{s}")
        }
    };
    let (module, entry) = if let Some(size) = crate::msl::workgroup_size(&src) {
        // SAFETY: the source is ours (`msl`), its loops end (`every_kernel_loop_ends`) and
        // its bindings are those of `read_only` at the buffer indices wgpu gives them.
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(name),
                entry_points: std::borrow::Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                    name: "k0".into(),
                    workgroup_size: (size, 1, 1),
                }]),
                msl: Some(src.into()),
                ..Default::default()
            })
        };
        (module, "k0")
    } else {
        let desc = wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) };
        let checks = wgpu::ShaderRuntimeChecks {
            force_loop_bounding: settings.loop_bounding,
            ..wgpu::ShaderRuntimeChecks::checked()
        };
        // SAFETY: without loop counters wgpu requires that no loop runs forever. The
        // kernels' loops count to shapes or halve to zero, which `every_kernel_loop_ends`
        // checks for every source; bounds checks stay on.
        (unsafe { device.create_shader_module_trusted(desc, checks) }, "main")
    };
    let entries: Vec<_> = read_only.iter().enumerate().map(|(i, &r)| storage_entry(i as u32, r)).collect();
    #[cfg(target_arch = "wasm32")]
    let entries = {
        let mut e = entries;
        e.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        });
        e
    };
    let layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(name), entries: &entries });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(name),
        bind_group_layouts: &[Some(&layout)],
        #[cfg(not(target_arch = "wasm32"))]
        immediate_size: 32,
        #[cfg(target_arch = "wasm32")]
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: settings.zero_workgroup_memory,
            ..Default::default()
        },
        cache: None,
    });
    WgpuKernel { pipeline, layout }
}

/// The wgpu backends of the platform: Metal on macOS, Vulkan elsewhere.
#[cfg(not(target_arch = "wasm32"))]
fn platform_backends() -> wgpu::Backends {
    if cfg!(target_os = "macos") { wgpu::Backends::METAL } else { wgpu::Backends::VULKAN }
}

/// Opens the browser's WebGPU device (wasm only). The futures are JS promises, so
/// this must be awaited (`Gpu::new_web`); the scalars of the kernels go through a
/// uniform ring buffer instead of push constants, and the WGSL sources carry the
/// `enable` directives and the `var<uniform>` port (see `compile`).
#[cfg(target_arch = "wasm32")]
pub(crate) async fn open_web(_diag: &Diagnostics) -> Result<(WgpuDev, DeviceCaps)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut adapter = None;
    for fallback in [false, true] {
        let opts = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: fallback,
            ..Default::default()
        };
        match instance.request_adapter(&opts).await {
            Ok(a) => {
                adapter = Some(a);
                break;
            }
            Err(e) if fallback => return Err(e).context("no WebGPU adapter found"),
            Err(_) => continue,
        }
    }
    let adapter = adapter.context("no WebGPU adapter found")?;
    // The kernels are written with f16 scalars and subgroup shuffles behind `enable`
    // directives, which Chrome requires explicitly (native wgpu enables them from the
    // device features). Chrome 129 and later have both.
    if !adapter.features().contains(wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16) {
        bail!("this WebGPU adapter lacks subgroups or f16 (a newer Chrome may help)");
    }
    let alimits = adapter.limits();
    let caps = detect(&adapter);
    let limits = wgpu::Limits {
        max_buffer_size: alimits.max_buffer_size,
        max_storage_buffer_binding_size: alimits.max_storage_buffer_binding_size,
        max_storage_buffers_per_shader_stage: 10,
        max_compute_workgroup_storage_size: caps.max_workgroup_storage,
        // The kernels run 128- and 256-lane workgroups; the WebGPU defaults are 64.
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_size_x: 256,
        ..wgpu::Limits::default()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("yaxus"),
            required_features: wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            ..Default::default()
        })
        .await
        .context("creating the WebGPU device")?;
    // Validation errors come back through the console; a failed allocation or kernel
    // otherwise surfaces as a lost device at the next read-back.
    device.on_uncaptured_error(std::sync::Arc::new(|e| web_sys::console::error_1(&format!("WebGPU: {e}").into())));
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let immediates = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("immediates"),
        size: (IMMEDIATE_SLOTS * 256) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let d = WgpuDev {
        device,
        queue,
        encoder: None,
        pass: None,
        bind_groups: HashMap::new(),
        readback,
        unflushed_upload: 0,
        kernels_profile: None,
        immediates,
        imm_i: 0,
    };
    Ok((d, caps))
}

/// One event-loop turn (a macrotask), so browser callbacks (a buffer map, device
/// loss) can run.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn event_loop_turn() {
    use wasm_bindgen::JsCast;
    let p: js_sys::Promise =
        js_sys::eval("new Promise(r => setTimeout(r, 0))").expect("eval").unchecked_into();
    let _ = wasm_bindgen_futures::JsFuture::from(p).await;
}

/// The vendor of `info`'s adapter (PCI vendor ids; Apple's on Metal).
fn vendor_of(info: &wgpu::AdapterInfo) -> Vendor {
    match info.vendor {
        0x1002 => Vendor::Amd,
        0x106b => Vendor::Apple,
        0x8086 => Vendor::Intel,
        0x10de => Vendor::Nvidia,
        _ if info.backend == wgpu::Backend::Metal => Vendor::Apple,
        _ => Vendor::Other,
    }
}

/// What the adapter can do.
pub(crate) fn detect(adapter: &wgpu::Adapter) -> DeviceCaps {
    let info = adapter.get_info();
    let features = adapter.features();
    let vendor = vendor_of(&info);
    let f16 = features.contains(wgpu::Features::SHADER_F16);
    // Metal reports subgroup sizes of 4..=64 whatever the GPU, but Apple GPUs run
    // SIMD-groups of 32 threads, as the kernels need.
    let lanes32 = info.subgroup_min_size >= 32 || (info.backend == wgpu::Backend::Metal && vendor == Vendor::Apple);
    let subgroups = features.contains(wgpu::Features::SUBGROUP) && lanes32;
    let props = adapter.cooperative_matrix_properties();
    let has = |size: u32, ab: wgpu::CooperativeScalarType| {
        props.iter().any(|c| {
            (c.m_size, c.n_size, c.k_size) == (size, size, size)
                && c.ab_type == ab
                && c.cr_type == wgpu::CooperativeScalarType::F32
        })
    };
    let matrices = subgroups && features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    let coop_size = if matrices && f16 && has(16, wgpu::CooperativeScalarType::F16) {
        16
    } else if matrices && info.backend == wgpu::Backend::Metal && has(8, wgpu::CooperativeScalarType::F32) {
        8
    } else {
        0
    };
    let coop = coop_size != 0;
    DeviceCaps {
        name: info.name.clone(),
        api: match info.backend {
            wgpu::Backend::Vulkan => Api::Vulkan,
            wgpu::Backend::Metal => Api::Metal,
            _ => Api::Other,
        },
        vendor,
        kind: match info.device_type {
            wgpu::DeviceType::IntegratedGpu => DeviceKind::Integrated,
            wgpu::DeviceType::DiscreteGpu => DeviceKind::Discrete,
            wgpu::DeviceType::Cpu => DeviceKind::Software,
            _ => DeviceKind::Other,
        },
        f16,
        subgroups,
        coop,
        fast_math: info.backend == wgpu::Backend::Metal,
        msl: info.backend == wgpu::Backend::Metal && features.contains(wgpu::Features::PASSTHROUGH_SHADERS),
        coop_size,
        timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        max_workgroup_storage: adapter.limits().max_compute_workgroup_storage_size.min(65536),
    }
}
