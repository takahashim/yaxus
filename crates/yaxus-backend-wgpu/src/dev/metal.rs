//! The device through Metal directly (macOS), without wgpu: shared-storage buffers the CPU
//! reads and writes in place, one command buffer and compute encoder at a time, and the
//! kernels compiled with `MTLCompileOptions` this crate chooses (no fast math). WGSL
//! kernels are translated to MSL by naga with the bindings at the indices the MSL kernels
//! of `msl` use (immediates at `buffer(0)`, binding `i` at `buffer(i + 1)`); MSL kernels
//! are compiled as they are.

use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::{Context, Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSize,
};

use super::Buf;
use crate::device::{CompileSettings, DeviceCaps, Vendor};

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = ProtocolObject<dyn MTLBuffer>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// A Metal buffer, compared and hashed by identity.
#[derive(Clone)]
pub struct MBuf(Retained<Buffer>);

// SAFETY: Metal buffers may be used from any thread; the crate orders their use through
// the `Gpu` mutex.
unsafe impl Send for MBuf {}
unsafe impl Sync for MBuf {}

impl PartialEq for MBuf {
    fn eq(&self, o: &Self) -> bool {
        Retained::as_ptr(&self.0) == Retained::as_ptr(&o.0)
    }
}
impl Eq for MBuf {}
impl std::hash::Hash for MBuf {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        Retained::as_ptr(&self.0).hash(h)
    }
}
impl std::fmt::Debug for MBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MBuf({} bytes)", self.0.length())
    }
}

impl MBuf {
    pub(crate) fn size(&self) -> u64 {
        self.0.length() as u64
    }
}

fn mb(b: &Buf) -> &Retained<Buffer> {
    match b {
        Buf::Metal(b) => &b.0,
        #[allow(unreachable_patterns)]
        _ => panic!("a buffer of another device"),
    }
}

/// A compiled kernel: its pipeline, threads per threadgroup and threadgroup memory.
pub(crate) struct MetalKernel {
    pipeline: Retained<Pipeline>,
    threads: usize,
    /// Byte lengths of the threadgroup arguments of a naga-translated kernel, by index.
    tg_memory: Vec<usize>,
    /// The bindings whose byte lengths naga's `_mslBufferSizes` holds, in order (at the
    /// buffer index after the bindings).
    sized: Vec<usize>,
}

// SAFETY: pipeline states are immutable and may be used from any thread.
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

/// What compiles kernels, from any thread.
#[derive(Clone)]
pub(crate) struct MetalCompiler {
    device: Retained<Device>,
    settings: CompileSettings,
}

// SAFETY: MTLDevice's compilation methods are thread-safe.
unsafe impl Send for MetalCompiler {}
unsafe impl Sync for MetalCompiler {}

impl MetalCompiler {
    pub(crate) fn compile(&self, name: &str, src: &str, read_only: &[bool]) -> MetalKernel {
        let (msl, entry, threads, tg_memory, sized) = match crate::msl::workgroup_size(src) {
            Some(size) => (src.to_string(), "k0".to_string(), size as usize, Vec::new(), Vec::new()),
            None => translate(src, read_only, self.settings).unwrap_or_else(|e| panic!("{name}: {e:#}")),
        };
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_1);
        // Arithmetic as written: kernels that must agree to the bit rely on it.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&msl), Some(&options))
            .unwrap_or_else(|e| panic!("{name}: {e}\n{msl}"));
        let function = library
            .newFunctionWithName(&NSString::from_str(&entry))
            .unwrap_or_else(|| panic!("{name}: no function {entry}"));
        let pipeline =
            self.device.newComputePipelineStateWithFunction_error(&function).unwrap_or_else(|e| panic!("{name}: {e}"));
        MetalKernel { pipeline, threads, tg_memory, sized }
    }
}

/// The MSL of a WGSL kernel, its entry point's name, threads per threadgroup, the lengths of
/// its threadgroup arguments and the bindings whose lengths it is passed.
type Translated = (String, String, usize, Vec<usize>, Vec<usize>);

/// WGSL to MSL with naga.
fn translate(src: &str, read_only: &[bool], settings: CompileSettings) -> Result<Translated> {
    use naga::back::msl;
    let module = naga::front::wgsl::parse_str(src).map_err(|e| anyhow!("WGSL: {}", e.emit_to_string(src)))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| anyhow!("validation: {e:?}"))?;
    let ep = module.entry_points.first().context("no entry point")?;
    let mut resources = msl::BindingMap::default();
    for (i, &ro) in read_only.iter().enumerate() {
        resources.insert(
            naga::ResourceBinding { group: 0, binding: i as u32 },
            msl::BindTarget { buffer: Some(i as u8 + 1), mutable: !ro, ..Default::default() },
        );
    }
    let ep_res = msl::EntryPointResources {
        resources,
        immediates_buffer: Some(0),
        sizes_buffer: Some(read_only.len() as u8 + 1),
    };
    let options = msl::Options {
        lang_version: (3, 1),
        per_entry_point_map: msl::EntryPointResourceMap::from([(ep.name.clone(), ep_res)]),
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        zero_initialize_workgroup_memory: settings.zero_workgroup_memory,
        force_loop_bounding: settings.loop_bounding,
        emit_int_div_checks: false,
        ..Default::default()
    };
    let pipeline =
        msl::PipelineOptions { entry_point: Some((naga::ShaderStage::Compute, ep.name.clone())), ..Default::default() };
    let (msl_src, tr) = msl::write_string(&module, &info, &options, &pipeline).map_err(|e| anyhow!("MSL: {e:?}"))?;
    let entry = tr.entry_point_names[0].as_ref().map_err(|e| anyhow!("{e}"))?.clone();
    let ep_info = info.get_entry_point(0);
    let tg_memory = module
        .global_variables
        .iter()
        .filter(|(h, v)| v.space == naga::AddressSpace::WorkGroup && !ep_info[*h].is_empty())
        .map(|(_, v)| (module.types[v.ty].inner.size(module.to_ctx()) as usize).next_multiple_of(16))
        .collect();
    // (as naga's `_mslBufferSizes`: every global holding a runtime-sized array, in order)
    let sized = module
        .global_variables
        .iter()
        .filter(|(_, v)| module.types[v.ty].inner.needs_host_buffer_byte_size(&module.types))
        .map(|(_, v)| v.binding.as_ref().map_or(0, |b| b.binding as usize))
        .collect();
    let threads = ep.workgroup_size.iter().product::<u32>() as usize;
    Ok((msl_src, entry, threads, tg_memory, sized))
}

pub(crate) struct MetalDev {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The command buffer being recorded, and its open compute encoder.
    cmd: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    enc: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// The last committed command buffer (earlier ones complete before it).
    last: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// An allocation that failed since `take_failure` (a small buffer stood in for it).
    failed: std::cell::Cell<Option<u64>>,
}

// SAFETY: the device, queue and command objects are only used under the `Gpu` mutex.
unsafe impl Send for MetalDev {}

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

impl MetalDev {
    pub(crate) fn open() -> Result<(MetalDev, DeviceCaps)> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("creating a Metal command queue")?;
        let caps = DeviceCaps {
            name: device.name().to_string(),
            backend: wgpu::Backend::Metal,
            vendor: Vendor::Apple,
            device_type: wgpu::DeviceType::IntegratedGpu,
            f16: true,
            subgroups: true,
            coop: true,
            coop_size: 8,
            // (`MTLCompileOptions` without fast math)
            fast_math: false,
            msl: true,
            timestamps: false,
            max_workgroup_storage: (device.maxThreadgroupMemoryLength() as u32).min(65536),
        };
        Ok((MetalDev { device, queue, cmd: None, enc: None, last: None, failed: Default::default() }, caps))
    }

    pub(crate) fn compiler(&self, settings: CompileSettings) -> MetalCompiler {
        MetalCompiler { device: self.device.clone(), settings }
    }

    pub(crate) fn max_buffer(&self) -> u64 {
        self.device.maxBufferLength() as u64
    }

    /// A buffer of `size` bytes, or a small one in its place when Metal cannot allocate it
    /// (reported by `take_failure`, so `Gpu::checked` returns an error).
    fn alloc(&self, size: u64) -> Retained<Buffer> {
        let b =
            (size <= self.max_buffer()).then(|| self.device.newBufferWithLength_options(size.max(16) as usize, SHARED));
        match b.flatten() {
            Some(b) => b,
            None => {
                self.failed.set(Some(size));
                self.device.newBufferWithLength_options(16, SHARED).expect("allocating 16 bytes")
            }
        }
    }

    /// The size of an allocation that failed since the last call.
    pub(crate) fn take_failure(&self) -> Option<String> {
        self.failed.take().map(|size| format!("Metal could not allocate a buffer of {size} bytes"))
    }

    pub(crate) fn storage(&self, _label: &str, size: u64) -> Buf {
        let b = self.alloc(size);
        // SAFETY: a shared buffer of `length` bytes, not yet used by the GPU.
        unsafe { std::ptr::write_bytes(b.contents().as_ptr().cast::<u8>(), 0, b.length()) };
        Buf::Metal(MBuf(b))
    }

    fn with_bytes(&self, bytes: &[u8]) -> Retained<Buffer> {
        let len = bytes.len().max(16);
        if bytes.len() < len {
            let mut padded = bytes.to_vec();
            padded.resize(len, 0);
            return self.with_bytes(&padded);
        }
        let b = self.alloc(len as u64);
        if b.length() >= len {
            // SAFETY: a new shared buffer of at least `len` bytes, not yet used by the GPU.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr().cast::<u8>(), len) };
        }
        b
    }

    pub(crate) fn upload(&mut self, _label: &str, bytes: &[u8]) -> Buf {
        Buf::Metal(MBuf(self.with_bytes(bytes)))
    }

    /// Waits for everything submitted so far.
    fn wait(&mut self) {
        if let Some(c) = self.last.take() {
            c.waitUntilCompleted();
            if c.status() == MTLCommandBufferStatus::Error {
                panic!("Metal command buffer failed: {:?}", c.error());
            }
        }
    }

    pub(crate) fn write(&mut self, dst: &Buf, offset: u64, bytes: &[u8]) {
        // (rare: test data; in place, after the GPU is done with everything recorded)
        self.submit();
        self.wait();
        let b = mb(dst);
        assert!(offset as usize + bytes.len() <= b.length());
        // SAFETY: in bounds, and the GPU is idle.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                b.contents().as_ptr().cast::<u8>().add(offset as usize),
                bytes.len(),
            )
        };
    }

    /// A new buffer per pass: the command buffers that read the previous one retain it.
    pub(crate) fn upload_meta(&mut self, slot: &mut Option<Buf>, data: &[u8]) {
        *slot = Some(Buf::Metal(MBuf(self.with_bytes(data))));
    }

    fn cmd(&mut self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        let q = &self.queue;
        self.cmd.get_or_insert_with(|| q.commandBuffer().expect("a command buffer"))
    }

    fn end_compute(&mut self) {
        if let Some(e) = self.enc.take() {
            e.endEncoding();
        }
    }

    pub(crate) fn copy(&mut self, src: &Buf, src_off: u64, dst: &Buf, dst_off: u64, size: u64) {
        self.end_compute();
        let blit = self.cmd().blitCommandEncoder().expect("a blit encoder");
        // SAFETY: both ranges are within their buffers (the callers' offsets).
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                mb(src),
                src_off as usize,
                mb(dst),
                dst_off as usize,
                size as usize,
            )
        };
        blit.endEncoding();
    }

    pub(crate) fn dispatch(&mut self, k: &MetalKernel, bufs: &[&Buf], imm: &[u32; 8], groups: (u32, u32)) {
        if self.enc.is_none() {
            let e = self.cmd().computeCommandEncoder().expect("a compute encoder");
            self.enc = Some(e);
        }
        let e = self.enc.as_ref().unwrap();
        e.setComputePipelineState(&k.pipeline);
        // SAFETY: the buffers are the kernel's bindings, the immediates 32 bytes.
        unsafe {
            e.setBytes_length_atIndex(NonNull::new(imm.as_ptr() as *mut c_void).unwrap(), 32, 0);
            for (i, b) in bufs.iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(mb(b)), 0, i + 1);
            }
            for (i, &len) in k.tg_memory.iter().enumerate() {
                e.setThreadgroupMemoryLength_atIndex(len, i);
            }
            if !k.sized.is_empty() {
                let sizes: Vec<u32> = k.sized.iter().map(|&i| bufs[i].size() as u32).collect();
                let ptr = NonNull::new(sizes.as_ptr() as *mut c_void).unwrap();
                e.setBytes_length_atIndex(ptr, 4 * sizes.len(), bufs.len() + 1);
            }
        }
        let grid = MTLSize { width: groups.0 as usize, height: groups.1 as usize, depth: 1 };
        let tg = MTLSize { width: k.threads, height: 1, depth: 1 };
        e.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    pub(crate) fn submit(&mut self) {
        self.end_compute();
        if let Some(c) = self.cmd.take() {
            c.commit();
            self.last = Some(c);
        }
    }

    pub(crate) fn read(&mut self, src: &Buf, offset: u64, size: u64) -> Vec<u8> {
        self.submit();
        self.wait();
        let b = mb(src);
        assert!((offset + size) as usize <= b.length());
        // SAFETY: in bounds, and the GPU is idle.
        unsafe {
            std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(offset as usize), size as usize).to_vec()
        }
    }
}

impl Drop for MetalDev {
    fn drop(&mut self) {
        self.submit();
        self.wait();
    }
}
