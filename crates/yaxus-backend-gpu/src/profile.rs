//! Profiling (`YAXUS_PROFILE`): time spent recording vs. waiting for the GPU, and the GPU
//! time of each kernel from timestamp queries.

use std::collections::HashMap;

/// Per-kernel GPU time from timestamp queries (`YAXUS_PROFILE=2`).
pub(crate) struct KernelProfile {
    pub(crate) queries: wgpu::QuerySet,
    pub(crate) resolve: wgpu::Buffer,
    pub(crate) staging: wgpu::Buffer,
    pub(crate) pending: Vec<String>,
    pub(crate) totals: HashMap<String, (f64, usize)>,
    pub(crate) period_ns: f64,
}

pub(crate) const MAX_TIMED: usize = 2048;

#[derive(Default)]
pub(crate) struct Profile {
    pub(crate) last_sync: Option<std::time::Instant>,
    pub(crate) record: std::time::Duration,
    pub(crate) wait: std::time::Duration,
    pub(crate) dispatches: usize,
    pub(crate) syncs: usize,
}

impl KernelProfile {
    /// The GPU time of each kernel, longest first, on stderr.
    pub(crate) fn report(&self) {
        if self.totals.is_empty() {
            return;
        }
        let mut v: Vec<_> = self.totals.iter().collect();
        v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let total: f64 = v.iter().map(|x| x.1.0).sum();
        eprintln!("[wgpu kernels] total {:.1} ms", total / 1e6);
        for (name, (ns, n)) in v {
            eprintln!(
                "  {name:<26} {:>9.2} ms {:>5.1}%  {n:>6} calls  {:>8.1} us/call",
                ns / 1e6,
                100.0 * ns / total,
                ns / 1e3 / *n as f64
            );
        }
    }
}

impl Profile {
    /// Recording and waiting time per synchronization, on stderr.
    pub(crate) fn report(&self) {
        if self.syncs == 0 {
            return;
        }
        eprintln!(
            "[wgpu profile] {} syncs: record {:.2?}/sync, GPU wait {:.2?}/sync, {} dispatches/sync",
            self.syncs,
            self.record / self.syncs as u32,
            self.wait / self.syncs as u32,
            self.dispatches / self.syncs
        );
    }
}
