//! WebGPU in the browser (wasm32): what it does differently from wgpu on native.
//!
//! - No push constants: the per-dispatch scalars go through a uniform ring
//!   ([`ParamRing`]) at `PARAM_BINDING`, passed by dynamic offset.
//! - No blocking: the device opens and reads back with `await` (`open_async`,
//!   `read_async`).
//! - Errors arrive asynchronously: the device records its first one in [`Fault`], and
//!   [`AsyncErrorScope`] captures those of a stretch of work for certain.

use anyhow::{Context, Result, bail};

use super::{Buf, PARAM_BINDING, WgpuDev, detect, wb};
use crate::device::{CompileSettings, DeviceCaps, Diagnostics};

/// The uniform ring: 4096 slots of 256 bytes (the usual offset alignment).
const PARAM_BYTES: u64 = 1 << 20;

/// What only the WebGPU device has.
pub(crate) struct WebState {
    pub(super) params: ParamRing,
    /// The first error WebGPU reported, or the device's loss.
    pub(crate) fault: Fault,
}

impl WebState {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let fault = Fault::default();
        fault.watch(device);
        WebState { params: ParamRing::new(device), fault }
    }
}

/// The uniform buffer carrying the per-dispatch scalars, and its cursor. A slot is
/// reused only after the submission that reads it.
pub(super) struct ParamRing {
    buf: wgpu::Buffer,
    offset: u64,
    stride: u64,
}

impl ParamRing {
    fn new(device: &wgpu::Device) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stride = device.limits().min_uniform_buffer_offset_alignment.max(32) as u64;
        ParamRing { buf, offset: 0, stride }
    }

    /// Whether every slot is taken since the last submission.
    fn is_full(&self) -> bool {
        self.offset + self.stride > self.buf.size()
    }

    /// Writes `imm` into the next slot and returns its dynamic offset.
    fn push(&mut self, queue: &wgpu::Queue, imm: &[u32; 8]) -> u32 {
        let offset = self.offset;
        queue.write_buffer(&self.buf, offset, bytemuck::cast_slice(imm));
        self.offset += self.stride;
        offset as u32
    }

    /// After a submission: every slot may be written again.
    pub(super) fn reset(&mut self) {
        self.offset = 0;
    }

    /// The ring's entry in a kernel's bind group (one slot, at the dynamic offset).
    pub(super) fn bind_entry(&self) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: PARAM_BINDING,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.buf,
                offset: 0,
                size: std::num::NonZeroU64::new(32),
            }),
        }
    }
}

/// The ring's entry in the bind group layout of a kernel that takes scalars.
pub(super) fn param_layout_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: PARAM_BINDING,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: std::num::NonZeroU64::new(32),
        },
        count: None,
    }
}

/// WebGPU has no push constants: the per-dispatch struct becomes a uniform binding.
pub(super) fn compile_module(
    device: &wgpu::Device,
    _settings: CompileSettings,
    name: &str,
    src: &str,
) -> (wgpu::ShaderModule, &'static str) {
    let src = super::webgpu_source(src);
    let desc = wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) };
    (device.create_shader_module(desc), "main")
}

/// The first error of a WebGPU device. Error scopes resolve asynchronously in the browser,
/// so the device reports every error to its uncaptured-error hook instead, and the loss
/// of the device to its lost callback; both land here, and the next read-back (the point
/// where the browser path awaits the GPU anyway) returns it. A device that failed once is
/// not trusted again: the slot is never cleared.
#[derive(Clone, Default)]
pub(crate) struct Fault(std::sync::Arc<std::sync::Mutex<Option<String>>>);

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
pub(crate) struct AsyncErrorScope {
    scopes: [wgpu::ErrorScopeGuard; 3],
}

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

impl WgpuDev {
    /// Opens WebGPU (wasm32) and requests room for the model's largest tensor. WebGPU
    /// reports a conservative set of limits; asking for more than the adapter supports
    /// makes `requestDevice` fail, so a smaller request is retried.
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

    /// Writes the scalars of one dispatch into the next uniform slot and returns its
    /// dynamic offset, submitting first when every slot is taken.
    pub(super) fn next_param(&mut self, imm: &[u32; 8]) -> u32 {
        if self.web.params.is_full() {
            self.submit();
        }
        self.web.params.push(&self.queue, imm)
    }

    /// Submits, waits for the map callback, and returns `size` bytes at `offset`.
    /// WebGPU polls the device itself; `Device::poll` is a no-op there, so the read
    /// must be awaited rather than blocked on. An error the device reported by then (the
    /// work before the read included) fails the read.
    pub(crate) async fn read_async(&mut self, src: &Buf, offset: u64, size: u64) -> Result<Vec<u8>> {
        if let Some(e) = self.web.fault.get() {
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
        if let Some(e) = self.web.fault.get() {
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
    pub(crate) fn async_error_scope(&self) -> AsyncErrorScope {
        let scopes = [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::Internal]
            .map(|f| self.device.push_error_scope(f));
        AsyncErrorScope { scopes }
    }
}
