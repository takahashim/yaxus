//! Recording and submitting commands: [`Recorder`] holds the device and decides when what
//! is recorded is submitted; `Gpu`'s methods here dispatch kernels by [`KernelId`], hand
//! out the pooled tensors and capture the device's errors.

#[cfg(target_arch = "wasm32")]
use anyhow::Result;

use crate::dev::{Buf, Dev, Kernel};
use crate::device::Tuning;
use crate::kernels::KernelId;
use crate::profile::Profile;
use crate::{Gpu, MetaBuf, Tensor};

/// Splits a 1-D workgroup count into a 2-D grid within the 65535 per-dimension limit.
pub(crate) fn grid(groups: u32) -> (u32, u32) {
    if groups <= 65535 { (groups.max(1), 1) } else { (65535, groups.div_ceil(65535)) }
}

/// When recorded commands are submitted (see the fields of [`Tuning`] they come from).
#[derive(Debug, Clone, Copy)]
struct SubmitPolicy {
    /// Matmul FLOPs recorded before a submission (`Tuning::submit_flops`).
    submit_flops: f64,
    /// Dispatches of a small pass before its first submission
    /// (`Tuning::early_submit_dispatches`).
    early_dispatches: usize,
    /// Rows up to which a pass is small (`Tuning::early_submit_max_rows`).
    small_rows: usize,
}

/// The device, and the commands recorded on it since the last submission.
pub(crate) struct Recorder {
    pub(crate) dev: Dev,
    policy: SubmitPolicy,
    /// Matmul FLOPs recorded since the last submission.
    pending_flops: f64,
    /// Dispatches recorded since the pass began, and whether it is small enough to be
    /// submitted early.
    pass_dispatches: usize,
    small_pass: bool,
    /// `YAXUS_PROFILE=1`: time spent recording vs. waiting for the GPU.
    profile: Option<Profile>,
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if let Some(p) = &self.profile {
            p.report();
        }
    }
}

impl Recorder {
    pub(crate) fn new(dev: Dev, tuning: &Tuning, profile: bool) -> Self {
        let policy = SubmitPolicy {
            submit_flops: tuning.submit_flops,
            early_dispatches: tuning.early_submit_dispatches,
            small_rows: tuning.early_submit_max_rows,
        };
        Recorder {
            dev,
            policy,
            pending_flops: 0.0,
            pass_dispatches: 0,
            small_pass: false,
            profile: profile.then(Profile::default),
        }
    }

    /// A forward pass of `rows` rows begins.
    pub(crate) fn begin_pass(&mut self, rows: usize) {
        self.pass_dispatches = 0;
        self.small_pass = rows <= self.policy.small_rows;
    }

    /// Records a dispatch of `k`; `label` names it for the per-kernel profile.
    pub(crate) fn dispatch(
        &mut self,
        k: &Kernel,
        bufs: &[&Buf],
        params: &[u32],
        groups: (u32, u32),
        label: impl FnOnce() -> String,
    ) {
        let mut imm = [0u32; 8];
        imm[..params.len()].copy_from_slice(params);
        if let Some(p) = &mut self.profile {
            p.dispatches += 1;
        }
        let timing = self.dev.timing();
        self.dev.dispatch(k, bufs, &imm, groups, timing.then(label));
        // Small passes (decode, verification) are recorded while the GPU already runs their
        // first layer: one early submission. More submissions cost more than they hide.
        self.pass_dispatches += 1;
        if self.pass_dispatches == self.policy.early_dispatches && self.small_pass && !timing {
            self.submit();
        }
    }

    /// Records that the commands just recorded do `flops` of matrix work, and submits
    /// them once `Tuning::submit_flops` have accumulated (so that no submission runs long
    /// enough for the driver to reset the GPU).
    pub(crate) fn account(&mut self, flops: f64) {
        self.pending_flops += flops;
        if self.pending_flops >= self.policy.submit_flops {
            self.submit();
        }
    }

    pub(crate) fn submit(&mut self) {
        self.pending_flops = 0.0;
        self.dev.submit();
    }

    /// Submits, waits, and returns `size` bytes of `src` from `offset`.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn read(&mut self, src: &Buf, offset: u64, size: u64) -> Vec<u8> {
        let t_rec = web_time::Instant::now();
        self.pending_flops = 0.0;
        let out = self.dev.read(src, offset, size);
        if let Some(p) = &mut self.profile {
            let now = web_time::Instant::now();
            if let Some(last) = p.last_sync {
                p.record += t_rec - last;
                p.wait += now - t_rec;
                p.syncs += 1;
            }
            p.last_sync = Some(now);
        }
        out
    }

    /// The browser's read-back: a synchronous one would never return (WebGPU polls
    /// itself), so there is none on wasm, and `SyncReadback` is native only.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn read_async(&mut self, src: &Buf, offset: u64, size: u64) -> Result<Vec<u8>> {
        self.pending_flops = 0.0;
        self.dev.read_async(src, offset, size).await
    }
}

impl Gpu {
    pub(crate) fn tensor(&mut self, rows: usize, cols: usize) -> Tensor {
        self.tensor_padded(rows, cols, rows)
    }

    /// A tensor whose buffer has room for `cap_rows >= rows` rows.
    pub(crate) fn tensor_padded(&mut self, rows: usize, cols: usize, cap_rows: usize) -> Tensor {
        let size = ((cap_rows * cols * 4) as u64).next_power_of_two().max(256);
        let pooled = self.pool.lock().unwrap().get_mut(&size).and_then(Vec::pop);
        let buf = pooled.unwrap_or_else(|| self.rec.dev.storage("tensor", size));
        Tensor { buf: Some(buf), rows, cols, pool: self.pool.clone() }
    }

    pub(crate) fn dispatch(&mut self, id: KernelId, bufs: &[&Buf], params: &[u32], groups: (u32, u32)) {
        let k = self.kernels.get(&self.rec.dev, &self.caps, &self.tuning, id);
        self.rec.dispatch(k, bufs, params, groups, || id.name());
    }

    /// [`Gpu::dispatch`] of a matrix product over an `[n, k]` weight: the per-kernel
    /// profile (`YAXUS_PROFILE=2`) labels it with that shape.
    pub(crate) fn dispatch_matmul(
        &mut self,
        id: KernelId,
        bufs: &[&Buf],
        params: &[u32],
        groups: (u32, u32),
        (n, k): (usize, usize),
    ) {
        let kernel = self.kernels.get(&self.rec.dev, &self.caps, &self.tuning, id);
        self.rec.dispatch(kernel, bufs, params, groups, || format!("{} {n}x{k}", id.name()));
    }

    /// Puts `data` in the pass metadata `slot` for the commands recorded from now on.
    pub(crate) fn upload_ordered(&mut self, slot: MetaBuf, data: &[u8]) {
        self.rec.dev.upload_meta(&mut self.pass.meta[slot as usize], data);
    }

    pub(crate) fn meta_buf(&self, slot: MetaBuf) -> Buf {
        self.pass.meta[slot as usize].clone().expect("begin_pass first")
    }

    /// Runs `f` with the device's out-of-memory and validation errors captured. Returns
    /// its result and the first error, so that a buffer that does not fit comes back as
    /// an error instead of panicking in the device's default error handler.
    pub(crate) fn captured<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> (T, Option<String>) {
        let scope = self.rec.dev.error_scope();
        let r = f(self);
        (r, self.rec.dev.finish_errors(scope))
    }

    /// [`Gpu::captured`], with the device's error as the result's.
    pub(crate) fn checked<T>(
        &mut self,
        what: &str,
        f: impl FnOnce(&mut Self) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        match self.captured(f) {
            (_, Some(e)) => anyhow::bail!("{what}: {e}"),
            (r, None) => r,
        }
    }
}
