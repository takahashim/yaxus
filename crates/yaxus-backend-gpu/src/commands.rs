//! Recording and submitting commands, uploads and read-backs, and the pooled tensors,
//! through the device layer (`dev`).

use anyhow::Result;

use crate::dev::Buf;
use crate::{Gpu, MetaBuf, Tensor};

impl Gpu {
    pub(crate) fn tensor(&mut self, rows: usize, cols: usize) -> Tensor {
        self.tensor_padded(rows, cols, rows)
    }

    /// A tensor whose buffer has room for `cap_rows >= rows` rows.
    pub(crate) fn tensor_padded(&mut self, rows: usize, cols: usize, cap_rows: usize) -> Tensor {
        let size = ((cap_rows * cols * 4) as u64).next_power_of_two().max(256);
        let pooled = self.pool.lock().unwrap().get_mut(&size).and_then(Vec::pop);
        let buf = pooled.unwrap_or_else(|| self.dev.storage("tensor", size));
        Tensor { buf: Some(buf), rows, cols, pool: self.pool.clone() }
    }

    pub(crate) fn dispatch(&mut self, name: &str, bufs: &[&Buf], params: &[u32], groups: (u32, u32)) {
        self.dispatch_shaped(name, bufs, params, groups, None);
    }

    /// [`Gpu::dispatch`] of a matrix product over an `[n, k]` weight: the per-kernel
    /// profile (`YAXUS_PROFILE=2`) labels it with that shape.
    pub(crate) fn dispatch_matmul(
        &mut self,
        name: &str,
        bufs: &[&Buf],
        params: &[u32],
        groups: (u32, u32),
        shape: (usize, usize),
    ) {
        self.dispatch_shaped(name, bufs, params, groups, Some(shape));
    }

    fn dispatch_shaped(
        &mut self,
        name: &str,
        bufs: &[&Buf],
        params: &[u32],
        groups: (u32, u32),
        shape: Option<(usize, usize)>,
    ) {
        let k = &self.kernels[name];
        let mut imm = [0u32; 8];
        imm[..params.len()].copy_from_slice(params);
        if let Some(p) = &mut self.profile {
            p.dispatches += 1;
        }
        let timing = self.dev.timing();
        let label = timing.then(|| match shape {
            Some((n, k)) => format!("{name} {n}x{k}"),
            None => name.to_string(),
        });
        self.dev.dispatch(k, bufs, &imm, groups, label);
        // Small passes (decode, verification) are recorded while the GPU already runs their
        // first layer: one early submission. More submissions cost more than they hide.
        self.pass.dispatches += 1;
        if self.pass.dispatches == self.tuning.early_submit_dispatches
            && self.pass.rows <= self.tuning.split_attn_max_rows
            && !timing
        {
            self.submit();
        }
    }

    /// Records that the commands just recorded do `flops` of matrix work, and submits
    /// them once `Tuning::submit_flops` have accumulated (so that no submission runs long enough
    /// for the driver to reset the GPU).
    pub(crate) fn account(&mut self, flops: f64) {
        self.pending_flops += flops;
        if self.pending_flops >= self.tuning.submit_flops {
            self.submit();
        }
    }

    pub(crate) fn submit(&mut self) {
        self.pending_flops = 0.0;
        self.dev.submit();
    }

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

    pub(crate) fn upload(&mut self, label: &str, bytes: &[u8]) -> Buf {
        self.loaded.weight_bytes += bytes.len();
        // Binding it would fail at the first dispatch, long after loading.
        if bytes.len() as u64 > self.dev.max_binding() {
            self.loaded.oversized.get_or_insert((label.to_string(), bytes.len()));
        }
        self.dev.upload(label, bytes)
    }

    /// Puts `data` in the pass metadata `slot` for the commands recorded from now on.
    pub(crate) fn upload_ordered(&mut self, slot: MetaBuf, data: &[u8]) {
        self.dev.upload_meta(&mut self.pass.meta[slot as usize], data);
    }

    pub(crate) fn meta_buf(&self, slot: MetaBuf) -> Buf {
        self.pass.meta[slot as usize].clone().expect("begin_pass first")
    }

    /// Runs `f` with the device's out-of-memory and validation errors captured, so that
    /// a weight or KV arena that does not fit comes back as `Err` instead of panicking
    /// in the device's default error handler.
    pub(crate) fn checked<T>(&mut self, what: &str, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        // Only report a buffer from this call (uploads outside `checked` also set it).
        self.loaded.oversized = None;
        let scope = self.dev.error_scope();
        let r = f(self);
        if let Some(e) = scope.finish().or_else(|| self.dev.take_failure()) {
            anyhow::bail!("{what}: {e}");
        }
        if let Some((label, bytes)) = self.loaded.oversized.take() {
            anyhow::bail!(
                "{what}: a {label} buffer of {bytes} bytes exceeds the device's storage binding limit of {}",
                self.dev.max_binding()
            );
        }
        r
    }
}
