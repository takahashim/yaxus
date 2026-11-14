//! The device through Metal directly (macOS), without wgpu: shared-storage buffers the CPU
//! reads and writes in place, one command buffer and compute encoder at a time, and the
//! kernels compiled with `MTLCompileOptions` this crate chooses (no fast math). WGSL
//! kernels are translated to MSL by naga with the bindings at the indices the MSL kernels
//! of `msl` use (immediates at `buffer(0)`, binding `i` at `buffer(i + 1)`); MSL kernels
//! are compiled as they are.

use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::{Context, Result};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSize,
};

mod translate;

use self::translate::{Translated, translate};
use super::{Buf, BufImpl};
use crate::device::{Api, CompileSettings, DeviceCaps, DeviceKind, Vendor};

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = ProtocolObject<dyn MTLBuffer>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// A Metal buffer, compared and hashed by identity.
#[derive(Clone)]
pub(crate) struct MBuf(Retained<Buffer>);

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
    match &b.0 {
        BufImpl::Metal(b) => &b.0,
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
        // (the compiler's autoreleased objects go with this pool, on whichever thread)
        autoreleasepool(|_| self.compile_pooled(name, src, read_only))
    }

    fn compile_pooled(&self, name: &str, src: &str, read_only: &[bool]) -> MetalKernel {
        let Translated { msl, entry, threads, tg_memory, sized } = match crate::msl::workgroup_size(src) {
            Some(size) => Translated::msl(src, size),
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
        let width = pipeline.threadExecutionWidth();
        assert_eq!(width, 32, "{name}: SIMD-groups of {width} threads; the kernels are written for 32");
        MetalKernel { pipeline, threads, tg_memory, sized }
    }
}

pub(crate) struct MetalDev {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The command buffer being recorded, and its open compute encoder.
    cmd: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    enc: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// The command buffers committed since the last wait, in order.
    committed: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// Inside `Gpu::checked` (`capture` to `take_failure`, through `Dev::finish_errors`): a failed allocation is reported
    /// there, with a small buffer standing in for it. Elsewhere it panics, as wgpu's
    /// uncaptured errors do.
    capturing: std::cell::Cell<bool>,
    /// An allocation that failed while capturing.
    failed: std::cell::Cell<Option<u64>>,
    /// `YAXUS_PROFILE=2`: each dispatch in a command buffer of its own, timed by the GPU.
    pub(crate) timed: Option<Timed>,
}

/// Per-kernel GPU time: dispatches waiting to be read, and the totals by label.
#[derive(Default)]
pub(crate) struct Timed {
    pending: Vec<(String, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>,
    totals: std::collections::HashMap<String, (f64, usize)>,
}

impl Timed {
    fn collect(&mut self) {
        for (label, c) in self.pending.drain(..) {
            let e = self.totals.entry(label).or_default();
            e.0 += c.GPUEndTime() - c.GPUStartTime();
            e.1 += 1;
        }
    }

    fn report(&self) {
        if self.totals.is_empty() {
            return;
        }
        let mut v: Vec<_> = self.totals.iter().collect();
        v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let total: f64 = v.iter().map(|x| x.1.0).sum();
        eprintln!("[metal kernels] total {:.1} ms (GPU time of each dispatch alone)", total * 1e3);
        for (name, (s, n)) in v {
            eprintln!(
                "  {name:<34} {:>9.2} ms {:>5.1}%  {n:>6} calls  {:>8.1} us/call",
                s * 1e3,
                100.0 * s / total,
                s * 1e6 / *n as f64
            );
        }
    }
}

// SAFETY: the device, queue and command objects are only used under the `Gpu` mutex.
unsafe impl Send for MetalDev {}

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

impl MetalDev {
    pub(crate) fn open(timed: bool) -> Result<(MetalDev, DeviceCaps)> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        // The kernels need Apple's 8x8 simdgroup matrices and 32-lane simdgroups: an Apple7
        // GPU (M1 and later), or Mac2 on Apple silicon, as a virtual machine's paravirtual
        // GPU is (the Mac2 family also holds Intel and AMD GPUs, of other widths; each
        // pipeline's width is checked as it compiles).
        let apple_silicon = cfg!(target_arch = "aarch64");
        anyhow::ensure!(
            device.supportsFamily(MTLGPUFamily::Apple7) || (apple_silicon && device.supportsFamily(MTLGPUFamily::Mac2)),
            "YAXUS_DEVICE=metal needs an Apple silicon GPU (M1 and later, or its virtual machines'), not {}",
            device.name()
        );
        let queue = device.newCommandQueue().context("creating a Metal command queue")?;
        let caps = DeviceCaps {
            name: device.name().to_string(),
            api: Api::MetalDirect,
            vendor: Vendor::Apple,
            kind: DeviceKind::Integrated,
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
        let timed = timed.then(Timed::default);
        let dev = MetalDev {
            device,
            queue,
            cmd: None,
            enc: None,
            committed: Vec::new(),
            capturing: Default::default(),
            failed: Default::default(),
            timed,
        };
        Ok((dev, caps))
    }

    pub(crate) fn compiler(&self, settings: CompileSettings) -> MetalCompiler {
        MetalCompiler { device: self.device.clone(), settings }
    }

    pub(crate) fn max_buffer(&self) -> u64 {
        self.device.maxBufferLength() as u64
    }

    /// The largest buffer a kernel may bind: naga's `_mslBufferSizes` holds u32 lengths,
    /// so a larger binding would be clamped to a wrong length rather than rejected.
    pub(crate) fn max_binding(&self) -> u64 {
        self.max_buffer().min(u32::MAX as u64)
    }

    /// A buffer of `size` bytes, or a small one in its place when Metal cannot allocate it
    /// (reported by `take_failure`, so `Gpu::checked` returns an error).
    fn alloc(&self, size: u64) -> Retained<Buffer> {
        let b =
            (size <= self.max_buffer()).then(|| self.device.newBufferWithLength_options(size.max(16) as usize, SHARED));
        match b.flatten() {
            Some(b) => b,
            None if self.capturing.get() => {
                self.failed.set(Some(size));
                self.device.newBufferWithLength_options(16, SHARED).expect("allocating 16 bytes")
            }
            None => panic!("Metal could not allocate a buffer of {size} bytes"),
        }
    }

    /// From now until `take_failure`, a failed allocation is reported rather than fatal.
    pub(crate) fn capture(&self) {
        self.capturing.set(true);
    }

    /// The size of an allocation that failed since `capture`, which this ends.
    pub(crate) fn take_failure(&self) -> Option<String> {
        self.capturing.set(false);
        self.failed.take().map(|size| format!("Metal could not allocate a buffer of {size} bytes"))
    }

    /// A zeroed buffer of `size` bytes: filled by the GPU, in order with what is recorded
    /// (the CPU need not touch every page of a KV arena).
    pub(crate) fn storage(&mut self, _label: &str, size: u64) -> Buf {
        let b = self.alloc(size);
        autoreleasepool(|_| {
            self.end_compute();
            let blit = self.cmd().blitCommandEncoder().expect("a blit encoder");
            blit.fillBuffer_range_value(&b, NSRange { location: 0, length: b.length() }, 0);
            blit.endEncoding();
        });
        Buf(BufImpl::Metal(MBuf(b)))
    }

    fn with_bytes(&self, bytes: &[u8]) -> Retained<Buffer> {
        // (a whole number of u32s, as wgpu's buffers: kernels see the same array lengths,
        // except under 16 bytes, where wgpu's buffers are as short as 4)
        let len = bytes.len().max(16).next_multiple_of(4);
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
        Buf(BufImpl::Metal(MBuf(self.with_bytes(bytes))))
    }

    /// Waits for everything submitted so far; a command buffer that failed is fatal (as
    /// wgpu's device loss).
    fn wait(&mut self) {
        for c in self.committed.drain(..) {
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
        *slot = Some(Buf(BufImpl::Metal(MBuf(self.with_bytes(data)))));
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
        // (wgpu rejects a copy out of range or off 4-byte alignment; Metal would write past
        // the buffer)
        assert!(src_off + size <= src.size() && dst_off + size <= dst.size(), "a copy out of range");
        assert!((src_off | dst_off | size).is_multiple_of(4), "a copy off 4-byte alignment");
        autoreleasepool(|_| {
            self.end_compute();
            let blit = self.cmd().blitCommandEncoder().expect("a blit encoder");
            // SAFETY: both ranges are within their buffers (checked above).
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
        });
    }

    pub(crate) fn dispatch(
        &mut self,
        k: &MetalKernel,
        bufs: &[&Buf],
        imm: &[u32; 8],
        groups: (u32, u32),
        label: Option<String>,
    ) {
        // A new command buffer comes autoreleased (on the M2, 168 bytes each; encoders and
        // the calls on them leave nothing): a thread that never ends (a server's worker)
        // never drains its implicit pool, so each dispatch has its own.
        autoreleasepool(|_| self.dispatch_pooled(k, bufs, imm, groups, label))
    }

    fn dispatch_pooled(
        &mut self,
        k: &MetalKernel,
        bufs: &[&Buf],
        imm: &[u32; 8],
        groups: (u32, u32),
        label: Option<String>,
    ) {
        // (timed: the work so far in its own command buffer, then this dispatch alone)
        if self.timed.is_some() {
            self.submit();
        }
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
                // (`max_binding` keeps weights and KV within u32; this catches the rest)
                let sizes: Vec<u32> = k
                    .sized
                    .iter()
                    .map(|&i| u32::try_from(bufs[i].size()).expect("a binding over u32::MAX bytes"))
                    .collect();
                let ptr = NonNull::new(sizes.as_ptr() as *mut c_void).unwrap();
                e.setBytes_length_atIndex(ptr, 4 * sizes.len(), bufs.len() + 1);
            }
        }
        let grid = MTLSize { width: groups.0 as usize, height: groups.1 as usize, depth: 1 };
        let tg = MTLSize { width: k.threads, height: 1, depth: 1 };
        e.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        if self.timed.is_some() {
            self.end_compute();
            let c = self.cmd.take().unwrap();
            c.commit();
            self.committed.push(c.clone());
            self.timed.as_mut().unwrap().pending.push((label.unwrap_or_default(), c));
        }
    }

    pub(crate) fn submit(&mut self) {
        self.end_compute();
        if let Some(c) = self.cmd.take() {
            c.commit();
            self.committed.push(c);
        }
    }

    pub(crate) fn read(&mut self, src: &Buf, offset: u64, size: u64) -> Vec<u8> {
        self.submit();
        self.wait();
        if let Some(t) = &mut self.timed {
            t.collect();
        }
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
        if let Some(t) = &mut self.timed {
            t.collect();
            t.report();
        }
    }
}
