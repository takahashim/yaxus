//! The device through wgpu: Vulkan, or Metal on macOS.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use wgpu::util::DeviceExt;

use super::Buf;
use crate::device::{self, CompileSettings, DeviceCaps, Diagnostics};
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
}

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
pub(crate) struct ErrorScope {
    oom: wgpu::ErrorScopeGuard,
    invalid: wgpu::ErrorScopeGuard,
}

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
    pub(crate) fn open(diag: &Diagnostics) -> Result<(WgpuDev, DeviceCaps)> {
        let backends = device::platform_backends();
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
        let caps = DeviceCaps::detect(&adapter);
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
        if bytes.len() as u64 == size {
            self.queue.write_buffer(&buf, 0, bytes);
        } else {
            let mut padded = bytes.to_vec();
            padded.resize(size as usize, 0);
            self.queue.write_buffer(&buf, 0, &padded);
        }
        // Initial contents go through staging memory that is only released after the
        // next submission; flush regularly so loading does not hold a second copy of
        // the weights (an 8B model in bf16 would exceed the GTT limit).
        self.unflushed_upload += bytes.len();
        if self.unflushed_upload >= UPLOAD_FLUSH_BYTES {
            self.unflushed_upload = 0;
            self.queue.submit([]);
            self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
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
        pass.set_bind_group(0, &bg, &[]);
        pass.set_immediates(0, bytemuck::cast_slice(imm));
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    pub(crate) fn submit(&mut self) {
        self.pass = None;
        if let Some(enc) = self.encoder.take() {
            self.queue.submit([enc.finish()]);
        }
    }

    pub(crate) fn read(&mut self, src: &Buf, offset: u64, size: u64) -> Vec<u8> {
        let src = wb(src).clone();
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
    let (module, entry) = if let Some(size) = crate::msl::workgroup_size(src) {
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
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: settings.zero_workgroup_memory,
            ..Default::default()
        },
        cache: None,
    });
    WgpuKernel { pipeline, layout }
}
