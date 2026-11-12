//! The device layer: buffers, uploads, copies, dispatches, submissions and read-backs,
//! and kernel compilation, behind one interface for the two ways of reaching the GPU:
//! wgpu (`wgpu`: Vulkan, or Metal through wgpu) and Metal directly (`metal`, macOS).
//! Everything above it (the ops, the plans, the weights) is the same for both.

pub(crate) mod wgpu;

use anyhow::Result;

use crate::device::{CompileSettings, DeviceCaps, Diagnostics};

/// A GPU buffer of either device (opaque outside the crate).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Buf {
    #[doc(hidden)]
    Wgpu(::wgpu::Buffer),
}

impl Buf {
    pub(crate) fn size(&self) -> u64 {
        match self {
            Buf::Wgpu(b) => b.size(),
        }
    }
}

/// A compiled kernel of either device, with its number in the device's kernel set.
pub(crate) struct Kernel {
    /// Its number in the device's kernel set (bind groups are cached by it).
    pub(crate) id: usize,
    pub(crate) imp: KernelImpl,
}

pub(crate) enum KernelImpl {
    Wgpu(wgpu::WgpuKernel),
}

/// What compiles kernels, shareable with the threads of `Gpu::precompile`.
#[derive(Clone)]
pub(crate) enum Compiler {
    Wgpu(::wgpu::Device, CompileSettings),
}

impl Compiler {
    /// Compiles `src` (WGSL, or MSL marked by `msl::header`) with the storage bindings
    /// `read_only` (binding `i` read-only when `read_only[i]`).
    pub(crate) fn compile(&self, name: &str, src: &str, read_only: &[bool]) -> Kernel {
        let imp = match self {
            Compiler::Wgpu(device, settings) => {
                KernelImpl::Wgpu(wgpu::compile(device, *settings, name, src, read_only))
            }
        };
        Kernel { id: 0, imp }
    }
}

/// The device: one of the two implementations.
pub(crate) enum Dev {
    Wgpu(wgpu::WgpuDev),
}

impl Dev {
    /// Opens the device the environment asks for (see `Gpu::new`).
    pub(crate) fn open(diag: &Diagnostics) -> Result<(Dev, DeviceCaps)> {
        let (d, caps) = wgpu::WgpuDev::open(diag)?;
        Ok((Dev::Wgpu(d), caps))
    }

    pub(crate) fn compiler(&self, settings: CompileSettings) -> Compiler {
        match self {
            Dev::Wgpu(d) => Compiler::Wgpu(d.device.clone(), settings),
        }
    }

    /// The largest buffer a kernel may bind.
    pub(crate) fn max_binding(&self) -> u64 {
        match self {
            Dev::Wgpu(d) => d.device.limits().max_storage_buffer_binding_size,
        }
    }

    /// The largest buffer the device allocates.
    pub(crate) fn max_buffer(&self) -> u64 {
        match self {
            Dev::Wgpu(d) => d.device.limits().max_buffer_size,
        }
    }

    /// A zeroed storage buffer of `size` bytes that kernels read and write and copies
    /// read from and write to.
    pub(crate) fn storage(&self, label: &str, size: u64) -> Buf {
        match self {
            Dev::Wgpu(d) => d.storage(label, size),
        }
    }

    /// A buffer holding `bytes` (weights, tables) for every command recorded from now on.
    pub(crate) fn upload(&mut self, label: &str, bytes: &[u8]) -> Buf {
        match self {
            Dev::Wgpu(d) => d.upload(label, bytes),
        }
    }

    /// Writes `bytes` into `dst` at `offset`, after everything recorded so far.
    pub(crate) fn write(&mut self, dst: &Buf, offset: u64, bytes: &[u8]) {
        match self {
            Dev::Wgpu(d) => d.write(dst, offset, bytes),
        }
    }

    /// Puts `data` in `slot` for the commands recorded from now on (the pass metadata),
    /// growing or replacing the buffer as the device needs.
    pub(crate) fn upload_meta(&mut self, slot: &mut Option<Buf>, data: &[u8]) {
        match self {
            Dev::Wgpu(d) => d.upload_meta(slot, data),
        }
    }

    /// Copies `size` bytes from `src` at `src_off` to `dst` at `dst_off`, in order.
    pub(crate) fn copy(&mut self, src: &Buf, src_off: u64, dst: &Buf, dst_off: u64, size: u64) {
        match self {
            Dev::Wgpu(d) => d.copy(src, src_off, dst, dst_off, size),
        }
    }

    /// Records a dispatch of `k` with `bufs` at its bindings and the immediates `imm`;
    /// `label` names it for the per-kernel profile when `timing()`.
    pub(crate) fn dispatch(
        &mut self,
        k: &Kernel,
        bufs: &[&Buf],
        imm: &[u32; 8],
        groups: (u32, u32),
        label: Option<String>,
    ) {
        match (self, &k.imp) {
            (Dev::Wgpu(d), KernelImpl::Wgpu(w)) => d.dispatch(k.id, w, bufs, imm, groups, label),
        }
    }

    /// Submits what has been recorded.
    pub(crate) fn submit(&mut self) {
        match self {
            Dev::Wgpu(d) => d.submit(),
        }
    }

    /// Submits, waits, and returns `size` bytes of `src` from `offset`.
    pub(crate) fn read(&mut self, src: &Buf, offset: u64, size: u64) -> Vec<u8> {
        match self {
            Dev::Wgpu(d) => d.read(src, offset, size),
        }
    }

    /// Whether dispatches are timed (`YAXUS_PROFILE=2`).
    pub(crate) fn timing(&self) -> bool {
        match self {
            Dev::Wgpu(d) => d.kernels_profile.is_some(),
        }
    }

    /// Captures allocation and validation errors until `ErrorScope::finish` (see
    /// `Gpu::checked`).
    pub(crate) fn error_scope(&self) -> ErrorScope {
        match self {
            Dev::Wgpu(d) => ErrorScope::Wgpu(d.error_scope()),
        }
    }
}

/// Errors captured since `Dev::error_scope`.
pub(crate) enum ErrorScope {
    Wgpu(wgpu::ErrorScope),
}

impl ErrorScope {
    pub(crate) fn finish(self) -> Option<String> {
        match self {
            ErrorScope::Wgpu(s) => s.finish(),
        }
    }
}
