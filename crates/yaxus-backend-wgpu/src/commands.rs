//! Recording and submitting commands, uploads and read-backs, and the pooled tensors.

use anyhow::Result;
use wgpu::util::DeviceExt;

use crate::profile::MAX_TIMED;
use crate::{Gpu, MetaBuf, Tensor};

/// Weight bytes uploaded between flushes of the staging memory.
pub(crate) const UPLOAD_FLUSH_BYTES: usize = 512 << 20;

impl Gpu {
    pub(crate) fn tensor(&mut self, rows: usize, cols: usize) -> Tensor {
        self.tensor_padded(rows, cols, rows)
    }

    /// A tensor whose buffer has room for `cap_rows >= rows` rows.
    pub(crate) fn tensor_padded(&mut self, rows: usize, cols: usize, cap_rows: usize) -> Tensor {
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
    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.pass = None;
        let device = &self.device;
        self.encoder.get_or_insert_with(|| device.create_command_encoder(&Default::default()))
    }

    pub(crate) fn dispatch(&mut self, name: &str, bufs: &[&wgpu::Buffer], params: &[u32], groups: (u32, u32)) {
        let k = &self.kernels[name];
        let key = (k.id, bufs.iter().map(|b| (*b).clone()).collect::<Vec<_>>());
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
        // Small passes (decode, verification) are recorded while the GPU already runs their
        // first layer: one early submission. More submissions cost more than they hide.
        self.pass_dispatches += 1;
        if self.pass_dispatches == self.tuning.early_submit_dispatches
            && self.pass_rows <= self.tuning.split_attn_max_rows
            && self.kernels_profile.is_none()
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
        self.pass = None;
        self.pending_flops = 0.0;
        if let Some(enc) = self.encoder.take() {
            self.queue.submit([enc.finish()]);
        }
    }

    pub(crate) fn read(&mut self, src: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
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

    pub(crate) fn upload(&mut self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        self.weight_bytes += bytes.len();
        // Binding it would fail at the first dispatch, long after loading.
        if bytes.len() as u64 > self.device.limits().max_storage_buffer_binding_size {
            self.oversized.get_or_insert((label.to_string(), bytes.len()));
        }
        // Not create_buffer_init: when the allocation fails (out of memory, above
        // max_buffer_size) it panics mapping the invalid buffer, before an error scope
        // can report it. A plain buffer and a queued write report the error instead.
        let size = (bytes.len().max(4) as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        buf
    }

    /// Copies `data` into `dst` (growing it if needed) through the command stream, so the
    /// upload is ordered after everything recorded so far.
    pub(crate) fn upload_ordered(&mut self, slot: MetaBuf, data: &[u8]) {
        let len = (data.len() as u64).max(4).next_multiple_of(4);
        let cur = &self.meta_bufs[slot as usize];
        if cur.as_ref().is_none_or(|b| b.size() < len) {
            let size = len.next_power_of_two().max(4096);
            self.meta_bufs[slot as usize] = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pass-meta"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if data.is_empty() {
            return;
        }
        let staging = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pass-meta-upload"),
            contents: data,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let dst = self.meta_bufs[slot as usize].clone().unwrap();
        self.encoder().copy_buffer_to_buffer(&staging, 0, &dst, 0, data.len() as u64);
    }

    pub(crate) fn meta_buf(&self, slot: MetaBuf) -> wgpu::Buffer {
        self.meta_bufs[slot as usize].clone().expect("begin_pass first")
    }

    /// Runs `f` with the device's out-of-memory and validation errors captured, so that
    /// a weight or KV arena that does not fit comes back as `Err` instead of panicking
    /// in wgpu's default error handler.
    pub(crate) fn checked<T>(&mut self, what: &str, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        // Only report a buffer from this call (uploads outside `checked` also set it).
        self.oversized = None;
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let invalid = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let r = f(self);
        let invalid = pollster::block_on(invalid.pop());
        let oom = pollster::block_on(oom.pop());
        if let Some(e) = oom.or(invalid) {
            anyhow::bail!("{what}: {e}");
        }
        if let Some((label, bytes)) = self.oversized.take() {
            anyhow::bail!(
                "{what}: a {label} buffer of {bytes} bytes exceeds the device's storage binding limit of {}",
                self.device.limits().max_storage_buffer_binding_size
            );
        }
        r
    }
}
