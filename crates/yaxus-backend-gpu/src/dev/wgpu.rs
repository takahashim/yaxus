//! The device through wgpu: Vulkan, or Metal on macOS.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use wgpu::util::DeviceExt;

use super::Buf;
use crate::device::{Api, CompileSettings, DeviceCaps, DeviceKind, Diagnostics, Vendor};
use crate::profile::{KernelProfile, MAX_TIMED};

/// Weight bytes uploaded between flushes of the staging memory. Smaller in the browser,
/// where a flush does not wait for the queue and the GPU process holds the copies.
#[cfg(not(target_arch = "wasm32"))]
const UPLOAD_FLUSH_BYTES: usize = 512 << 20;
#[cfg(target_arch = "wasm32")]
const UPLOAD_FLUSH_BYTES: usize = 64 << 20;

/// On wasm, per-dispatch scalars go through a uniform buffer at this binding (WebGPU has
/// no push constants). It is above every storage binding the kernels use.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) const PARAM_BINDING: u32 = 15;
/// The uniform ring: 4096 slots of 256 bytes (the usual offset alignment).
#[cfg(target_arch = "wasm32")]
const PARAM_BYTES: u64 = 1 << 20;

pub(crate) struct WgpuKernel {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
    /// Whether the source takes the per-dispatch struct (a uniform binding on wasm).
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(crate) has_params: bool,
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
    /// The uniform buffer carrying the per-dispatch scalars, and the ring cursor.
    #[cfg(target_arch = "wasm32")]
    params: wgpu::Buffer,
    #[cfg(target_arch = "wasm32")]
    param_offset: u64,
    #[cfg(target_arch = "wasm32")]
    param_stride: u64,
    /// The first error WebGPU reported, or the device's loss (see `Fault`).
    #[cfg(target_arch = "wasm32")]
    pub(crate) fault: Fault,
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

/// The first error of a WebGPU device. Error scopes resolve asynchronously in the browser,
/// so the device reports every error to its uncaptured-error hook instead, and the loss
/// of the device to its lost callback; both land here, and the next read-back (the point
/// where the browser path awaits the GPU anyway) returns it. A device that failed once is
/// not trusted again: the slot is never cleared.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Default)]
pub(crate) struct Fault(std::sync::Arc<std::sync::Mutex<Option<String>>>);

#[cfg(target_arch = "wasm32")]
impl Fault {
    fn record(&self, what: String) {
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(what);
        }
    }

    pub(crate) fn get(&self) -> Option<String> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn watch(&self, device: &wgpu::Device) {
        let fault = self.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| fault.record(format!("WebGPU: {e}"))));
        let fault = self.clone();
        device.set_device_lost_callback(move |reason, message| {
            fault.record(format!("the GPU device was lost ({reason:?}): {message}"));
        });
    }
}

/// Error scopes pushed around a stretch of work in the browser (see
/// `WgpuDev::async_error_scope`).
#[cfg(target_arch = "wasm32")]
pub(crate) struct AsyncErrorScope {
    scopes: [wgpu::ErrorScopeGuard; 3],
}

#[cfg(target_arch = "wasm32")]
impl AsyncErrorScope {
    /// Pops the scopes (innermost first) and returns the first error they captured.
    pub(crate) async fn finish(self) -> Option<String> {
        let [oom, invalid, internal] = self.scopes;
        let popped = [internal.pop(), invalid.pop(), oom.pop()];
        let mut first = None;
        for p in popped {
            if let Some(e) = p.await {
                first.get_or_insert_with(|| format!("WebGPU: {e}"));
            }
        }
        first
    }
}

/// Allocation and validation errors captured around a closure (see `Gpu::checked`).
pub(crate) struct ErrorScope {
    #[cfg(not(target_arch = "wasm32"))]
    oom: wgpu::ErrorScopeGuard,
    #[cfg(not(target_arch = "wasm32"))]
    invalid: wgpu::ErrorScopeGuard,
}

impl ErrorScope {
    pub(crate) fn finish(self) -> Option<String> {
        // Popping an error scope is async in WebGPU, and `Gpu::checked` runs while the
        // model loads, synchronously. The errors go to `Fault` instead, which
        // `take_failure` and every read-back report.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let invalid = pollster::block_on(self.invalid.pop());
            let oom = pollster::block_on(self.oom.pop());
            oom.or(invalid).map(|e| e.to_string())
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
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
        Ok((Self::from_device(device, queue, timed), caps))
    }

    /// Opens WebGPU (wasm32) and requests room for the model's largest tensor. WebGPU
    /// reports a conservative set of limits; asking for more than the adapter supports
    /// makes `requestDevice` fail, so a smaller request is retried.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn open_async(
        diag: &Diagnostics,
        need_binding: u64,
        need_buffer: u64,
    ) -> Result<(WgpuDev, DeviceCaps)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .context("WebGPU is not available (requestAdapter failed)")?;
        let info = adapter.get_info();
        if !diag.software && info.device_type == wgpu::DeviceType::Cpu {
            bail!("only a software WebGPU adapter is available");
        }
        let mut caps = detect(&adapter);
        // The subgroup kernels assume 32 lanes, which WebGPU does not report; keep to
        // the f32 kernels tested on lavapipe. Cooperative matrices are not in WebGPU.
        caps.subgroups = false;
        caps.coop = false;
        caps.coop_size = 0;
        caps.msl = false;
        let alimits = adapter.limits();
        let timed = false;
        // Ask for what the model needs, not for the adapter's maximum (Deno's WebGPU on
        // an M2 reports a 9.5 GB `maxBufferSize`, for example).
        let limits = wgpu::Limits {
            max_buffer_size: alimits.max_buffer_size.min(need_buffer.max(256 << 20)),
            max_storage_buffer_binding_size: alimits.max_storage_buffer_binding_size.min(need_binding.max(128 << 20)),
            max_storage_buffers_per_shader_stage: alimits.max_storage_buffers_per_shader_stage.clamp(6, 10),
            max_compute_workgroup_storage_size: caps.max_workgroup_storage,
            max_compute_invocations_per_workgroup: 256,
            max_uniform_buffer_binding_size: 64 << 10,
            max_bind_groups: 4,
            ..wgpu::Limits::default()
        };
        let desc = wgpu::DeviceDescriptor {
            label: Some("yaxus"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            ..Default::default()
        };
        let (device, queue) = match adapter.request_device(&desc).await {
            Ok(x) => x,
            Err(e) => {
                // Some adapters advertise more than `requestDevice` grants. Fall back to the
                // buffer sizes every WebGPU device has (256 MB, a 128 MB binding); a matrix
                // that does not fit one binding then loads as Q4 (`Gpu::load_weight`).
                let base = wgpu::Limits::default();
                let limits = wgpu::Limits {
                    max_buffer_size: base.max_buffer_size.min(alimits.max_buffer_size),
                    max_storage_buffer_binding_size: base
                        .max_storage_buffer_binding_size
                        .min(alimits.max_storage_buffer_binding_size),
                    ..limits
                };
                let desc = wgpu::DeviceDescriptor { required_limits: limits, ..desc };
                adapter.request_device(&desc).await.with_context(|| {
                    format!(
                        "creating the WebGPU device ({e}); the model needs a {need_buffer} byte buffer \
                         and a {need_binding} byte storage binding"
                    )
                })?
            }
        };
        Ok((Self::from_device(device, queue, timed), caps))
    }

    fn from_device(device: wgpu::Device, queue: wgpu::Queue, timed: bool) -> WgpuDev {
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
        #[cfg(target_arch = "wasm32")]
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[cfg(target_arch = "wasm32")]
        let param_stride = device.limits().min_uniform_buffer_offset_alignment.max(32) as u64;
        #[cfg(target_arch = "wasm32")]
        let fault = {
            let fault = Fault::default();
            fault.watch(&device);
            fault
        };
        WgpuDev {
            device,
            queue,
            encoder: None,
            pass: None,
            bind_groups: HashMap::new(),
            readback,
            unflushed_upload: 0,
            #[cfg(target_arch = "wasm32")]
            params,
            #[cfg(target_arch = "wasm32")]
            param_offset: 0,
            #[cfg(target_arch = "wasm32")]
            param_stride,
            #[cfg(target_arch = "wasm32")]
            fault,
            kernels_profile,
        }
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
        // WebGPU: put the scalars in a uniform slot and pass its offset dynamically.
        #[cfg(target_arch = "wasm32")]
        let param_offset = if k.has_params { self.next_param(imm) } else { 0 };
        let key = (id, bufs.iter().map(|b| wb(b).clone()).collect::<Vec<_>>());
        let bg = match self.bind_groups.get(&key) {
            Some(bg) => bg.clone(),
            None => {
                #[allow(unused_mut)]
                let mut entries: Vec<_> = key
                    .1
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                    .collect();
                #[cfg(target_arch = "wasm32")]
                if k.has_params {
                    entries.push(wgpu::BindGroupEntry {
                        binding: PARAM_BINDING,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.params,
                            offset: 0,
                            size: std::num::NonZeroU64::new(32),
                        }),
                    });
                }
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
        #[cfg(target_arch = "wasm32")]
        if k.has_params {
            pass.set_bind_group(0, &bg, &[param_offset]);
        } else {
            pass.set_bind_group(0, &bg, &[]);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            pass.set_bind_group(0, &bg, &[]);
            pass.set_immediates(0, bytemuck::cast_slice(imm));
        }
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    /// Writes the scalars of one dispatch into the next uniform slot and returns its
    /// dynamic offset. Slots are reused only after the submission that reads them.
    #[cfg(target_arch = "wasm32")]
    fn next_param(&mut self, imm: &[u32; 8]) -> u32 {
        if self.param_offset + self.param_stride > self.params.size() {
            self.submit();
        }
        let offset = self.param_offset;
        self.queue.write_buffer(&self.params, offset, bytemuck::cast_slice(imm));
        self.param_offset += self.param_stride;
        offset as u32
    }

    pub(crate) fn submit(&mut self) {
        self.pass = None;
        if let Some(enc) = self.encoder.take() {
            self.queue.submit([enc.finish()]);
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.param_offset = 0;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
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

    /// Submits, waits for the map callback, and returns `size` bytes at `offset`.
    /// WebGPU polls the device itself; `Device::poll` is a no-op there, so the read
    /// must be awaited rather than blocked on. An error the device reported by then (the
    /// work before the read included) fails the read.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn read_async(&mut self, src: &Buf, offset: u64, size: u64) -> Result<Vec<u8>> {
        if let Some(e) = self.fault.get() {
            bail!("{e}");
        }
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
        self.submit();
        let (tx, rx) = futures_channel::oneshot::channel();
        staging.map_async(wgpu::MapMode::Read, ..size, move |r| {
            let _ = tx.send(r);
        });
        let mapped = rx.await;
        if let Some(e) = self.fault.get() {
            bail!("{e}");
        }
        match mapped {
            Ok(Ok(())) => {}
            Ok(Err(e)) => bail!("reading back from the GPU: {e}"),
            Err(_) => bail!("reading back from the GPU: the map request was dropped"),
        }
        let out = staging.get_mapped_range(..size).context("reading back from the GPU")?.to_vec();
        staging.unmap();
        Ok(out)
    }

    /// Captures every error of the device until [`AsyncErrorScope::finish`], which
    /// resolves only after the errors of all earlier work are known (unlike `Fault`, which
    /// an uncaptured-error event may reach after a read-back has already completed).
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn async_error_scope(&self) -> AsyncErrorScope {
        let scopes = [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::Internal]
            .map(|f| self.device.push_error_scope(f));
        AsyncErrorScope { scopes }
    }

    pub(crate) fn error_scope(&self) -> ErrorScope {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let invalid = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            ErrorScope { oom, invalid }
        }
        #[cfg(target_arch = "wasm32")]
        {
            ErrorScope {}
        }
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
    let _has_params = src.contains("var<immediate> p: P;");
    let (module, entry) = compile_module(device, settings, name, src);
    #[allow(unused_mut)]
    let mut entries: Vec<_> = read_only.iter().enumerate().map(|(i, &r)| storage_entry(i as u32, r)).collect();
    #[cfg(target_arch = "wasm32")]
    if _has_params {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: PARAM_BINDING,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(32),
            },
            count: None,
        });
    }
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
    WgpuKernel { pipeline, layout, has_params: _has_params }
}

#[cfg(not(target_arch = "wasm32"))]
fn compile_module(
    device: &wgpu::Device,
    settings: CompileSettings,
    name: &str,
    src: &str,
) -> (wgpu::ShaderModule, &'static str) {
    if let Some(size) = crate::msl::workgroup_size(src) {
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
    }
}

/// The WGSL a WebGPU kernel gets: push constants become a uniform binding. Shared with
/// the test that parses every kernel with naga.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn webgpu_source(src: &str) -> String {
    src.replace("var<immediate> p: P;", &format!("@group(0) @binding({PARAM_BINDING}) var<uniform> p: P;"))
}

/// WebGPU has no push constants: the per-dispatch struct becomes a uniform binding.
#[cfg(target_arch = "wasm32")]
fn compile_module(
    device: &wgpu::Device,
    _settings: CompileSettings,
    name: &str,
    src: &str,
) -> (wgpu::ShaderModule, &'static str) {
    let src = webgpu_source(src);
    let desc = wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) };
    (device.create_shader_module(desc), "main")
}

/// The wgpu backends of the platform: Metal on macOS, Vulkan elsewhere.
#[cfg(not(target_arch = "wasm32"))]
fn platform_backends() -> wgpu::Backends {
    if cfg!(target_os = "macos") { wgpu::Backends::METAL } else { wgpu::Backends::VULKAN }
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
            wgpu::Backend::BrowserWebGpu => Api::WebGpu,
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
