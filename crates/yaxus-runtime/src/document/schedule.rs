//! The order segments are translated in: chains of consecutive segments, each translated
//! in order, one segment at a time; the chains run side by side.

struct Chain {
    /// Segment indices of the chain, in order.
    segs: Vec<usize>,
    /// Position in `segs` of the first segment not finished.
    next: usize,
    /// `segs[next]` has been handed out and is not finished.
    busy: bool,
}

/// Which segment of each chain may run, knowing only segment indices.
pub(crate) struct ChainSchedule {
    chains: Vec<Chain>,
    /// Chain of each segment.
    chain_of: Vec<usize>,
}

impl ChainSchedule {
    /// `segments` split into `chains` contiguous, nearly equal chains (at least 1, at most
    /// one per segment), so that each chain's history is local context.
    pub(crate) fn new(segments: usize, chains: usize) -> Self {
        let k = chains.clamp(1, segments.max(1));
        let mut out = Vec::with_capacity(k);
        let mut chain_of = vec![0; segments];
        let mut at = 0;
        for c in 0..k {
            let len = segments / k + usize::from(c < segments % k);
            let segs: Vec<usize> = (at..at + len).collect();
            for &s in &segs {
                chain_of[s] = c;
            }
            at += len;
            out.push(Chain { segs, next: 0, busy: false });
        }
        ChainSchedule { chains: out, chain_of }
    }

    pub(crate) fn chains(&self) -> usize {
        self.chains.len()
    }

    pub(crate) fn chain_of(&self, segment: usize) -> usize {
        self.chain_of[segment]
    }

    /// The next segment of every idle chain, which is then busy. A segment for which
    /// `skip` returns true is finished on the spot and the chain moves on.
    pub(crate) fn take_ready(&mut self, mut skip: impl FnMut(usize) -> bool) -> Vec<usize> {
        let mut out = Vec::new();
        for chain in self.chains.iter_mut().filter(|c| !c.busy) {
            while let Some(&i) = chain.segs.get(chain.next) {
                if skip(i) {
                    chain.next += 1;
                    continue;
                }
                chain.busy = true;
                out.push(i);
                break;
            }
        }
        out
    }

    /// Finishes `segment` if it is the busy one of its chain, which may then continue.
    pub(crate) fn finish(&mut self, segment: usize) {
        let chain = &mut self.chains[self.chain_of[segment]];
        if chain.busy && chain.segs.get(chain.next) == Some(&segment) {
            chain.busy = false;
            chain.next += 1;
        }
    }

    /// Every segment is finished.
    pub(crate) fn is_done(&self) -> bool {
        self.chains.iter().all(|c| c.next == c.segs.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_split_contiguously_and_advance_in_order() {
        let mut s = ChainSchedule::new(10, 3);
        assert_eq!(s.chains(), 3);
        assert_eq!(s.take_ready(|_| false), [0, 4, 7]); // chains of 4, 3, 3
        assert!(s.take_ready(|_| false).is_empty(), "busy chains offer nothing");
        s.finish(4);
        assert_eq!(s.take_ready(|_| false), [5]);
        // A segment that is not its chain's busy one is ignored.
        s.finish(9);
        assert!(s.take_ready(|_| false).is_empty());
    }

    #[test]
    fn skipped_segments_finish_at_once() {
        let mut s = ChainSchedule::new(4, 1);
        assert_eq!(s.take_ready(|i| i < 2), [2]);
        s.finish(2);
        assert_eq!(s.take_ready(|_| true), Vec::<usize>::new());
        assert!(s.is_done());
    }

    #[test]
    fn chain_counts_are_clamped() {
        assert_eq!(ChainSchedule::new(3, 8).chains(), 3);
        assert_eq!(ChainSchedule::new(0, 0).chains(), 1);
        assert!(ChainSchedule::new(0, 0).is_done());
    }
}
