//! The rayon thread pool the CPU backend runs on.

/// Number of physical cores (SMT siblings counted once), if it can be determined.
fn physical_cores() -> Option<usize> {
    let mut seen = std::collections::HashSet::new();
    for e in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("cpu")
            && name[3..].chars().all(|c| c.is_ascii_digit())
            && let Ok(s) = std::fs::read_to_string(e.path().join("topology/thread_siblings_list"))
        {
            seen.insert(s.trim().to_string());
        }
    }
    (!seen.is_empty()).then_some(seen.len())
}

/// Sets up the global rayon pool. Decoding is memory-bound, and one thread per physical
/// core is slightly faster than using SMT siblings. `RAYON_NUM_THREADS` overrides this.
pub fn init_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    if let Some(n) = physical_cores() {
        // The host's cores are not all ours under a cgroup CPU quota (containers) or an
        // affinity mask (taskset); available_parallelism accounts for both.
        let n = std::thread::available_parallelism().map_or(n, |a| n.min(a.get()));
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
}
