//! Paged KV bookkeeping: which arena pages each sequence owns, prefixes shared between
//! sequences, and the finished sequences kept for reuse.
//!
//! Host-side only: the page contents live in the backend's `Kv`. An operation that needs
//! pages copied pushes them to its `copies` argument, and the caller makes the copies
//! (in order, before the next pass) whether or not the operation then succeeds: the page
//! tables already point at the copies.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};

use crate::PassMeta;

/// KV cache page size in tokens. Sequences own lists of pages; a page can be shared by
/// several sequences (common prefixes) and is copied before it is written while shared.
pub const PAGE: usize = 128;

/// Shortest common prefix worth reusing (shorter ones are recomputed).
pub const MIN_REUSE: usize = 16;
/// Finished sequences kept for reuse at most.
pub const MAX_CACHED: usize = 32;

/// Identifies a sequence in a model's KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(u64);

/// Page `src` is to be copied to page `dst`, in every layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageCopy {
    pub src: u32,
    pub dst: u32,
}

#[derive(Debug, Clone, Default)]
struct Seq {
    pages: Vec<u32>,
    len: usize,
    /// The tokens consumed so far (`tokens.len() == len`), for prefix matching.
    tokens: Vec<u32>,
}

/// Page allocation with reference counts (shared prefixes).
struct PagePool {
    refs: Vec<u32>,
    free: Vec<u32>,
}

impl PagePool {
    fn new(pages: usize) -> Self {
        PagePool { refs: vec![0; pages], free: (0..pages as u32).rev().collect() }
    }
    fn alloc(&mut self) -> Option<u32> {
        let p = self.free.pop()?;
        self.refs[p as usize] = 1;
        Some(p)
    }
    fn retain(&mut self, p: u32) {
        self.refs[p as usize] += 1;
    }
    fn release(&mut self, p: u32) {
        let r = &mut self.refs[p as usize];
        *r -= 1;
        if *r == 0 {
            self.free.push(p);
        }
    }
    fn is_shared(&self, p: u32) -> bool {
        self.refs[p as usize] > 1
    }
}

/// Finished sequences kept for prefix reuse, by last use.
#[derive(Default)]
struct PrefixCache {
    entries: Vec<(SeqId, u64)>,
    clock: u64,
}

impl PrefixCache {
    /// Makes `id` the most recently used entry (adding it if needed).
    fn touch(&mut self, id: SeqId) {
        self.remove(id);
        self.clock += 1;
        self.entries.push((id, self.clock));
    }

    /// Whether `id` was an entry.
    fn remove(&mut self, id: SeqId) -> bool {
        let before = self.entries.len();
        self.entries.retain(|(c, _)| *c != id);
        self.entries.len() != before
    }

    fn least_recent(&self) -> Option<SeqId> {
        self.entries.iter().min_by_key(|(_, t)| *t).map(|(id, _)| *id)
    }

    fn ids(&self) -> impl Iterator<Item = SeqId> + '_ {
        self.entries.iter().map(|(id, _)| *id)
    }
}

/// The sequences of one model's KV arena.
pub struct KvCache {
    pool: PagePool,
    seqs: HashMap<SeqId, Seq>,
    next_id: u64,
    max_pos: usize,
    prefix: PrefixCache,
    /// Keep finished sequences and reuse their KV for prompts sharing a prefix.
    reuse: bool,
}

/// The rows of a pass: its metadata, and the row of each item's last token.
pub(crate) struct PassRows {
    pub meta: PassMeta,
    pub last: Vec<usize>,
}

impl KvCache {
    /// An arena of `pages` pages for positions `< max_pos`, with prefix reuse.
    pub fn new(pages: usize, max_pos: usize) -> Self {
        KvCache {
            pool: PagePool::new(pages),
            seqs: HashMap::new(),
            next_id: 0,
            max_pos,
            prefix: PrefixCache::default(),
            reuse: true,
        }
    }

    /// Turns prefix reuse on or off (off also drops the sequences kept for it).
    pub fn set_prefix_reuse(&mut self, on: bool) {
        self.reuse = on;
        if !on {
            self.clear_prefix_cache();
        }
    }

    fn insert(&mut self, seq: Seq) -> SeqId {
        let id = SeqId(self.next_id);
        self.next_id += 1;
        self.seqs.insert(id, seq);
        id
    }

    pub fn new_seq(&mut self) -> SeqId {
        self.insert(Seq::default())
    }

    pub fn free_seq(&mut self, id: SeqId) {
        self.prefix.remove(id);
        if let Some(s) = self.seqs.remove(&id) {
            for p in s.pages {
                self.pool.release(p);
            }
        }
    }

    /// Frees the least recently used kept sequence; `false` when none is kept.
    fn evict_one(&mut self) -> bool {
        let Some(id) = self.prefix.least_recent() else { return false };
        self.free_seq(id);
        true
    }

    /// Evicts kept sequences until `n` pages are free (or none is kept). Pages shared
    /// with live sequences are freed only when their last owner goes.
    fn ensure_free(&mut self, n: usize) {
        while self.pool.free.len() < n && self.evict_one() {}
    }

    /// Drops every kept sequence.
    pub fn clear_prefix_cache(&mut self) {
        while self.evict_one() {}
    }

    /// Creates a sequence for `prompt` without processing it: with prefix reuse, the
    /// sequence shares the KV of the longest common prefix with a kept sequence (always
    /// leaving at least the last prompt token to compute). Returns the sequence and the
    /// number of prompt tokens already in it.
    pub fn begin(&mut self, prompt: &[u32], copies: &mut Vec<PageCopy>) -> Result<(SeqId, usize)> {
        ensure!(!prompt.is_empty(), "empty prompt");
        let Some((src, mut keep)) = self.best_prefix(prompt) else { return Ok((self.new_seq(), 0)) };
        // Whole pages are shared; only a partly used last page is copied, which needs a
        // free page. Make room without evicting `src` itself (with the arena full of live
        // sequences it would go last otherwise), and when no page can be had, reuse just
        // the whole pages.
        let kept = self.prefix.remove(src);
        if !keep.is_multiple_of(PAGE) {
            self.ensure_free(1);
            if self.pool.free.is_empty() {
                keep = keep / PAGE * PAGE;
            }
        }
        if kept {
            self.prefix.touch(src);
        }
        if keep >= MIN_REUSE
            && let Ok(s) = self.fork(src, keep, copies)
        {
            return Ok((s, keep));
        }
        Ok((self.new_seq(), 0))
    }

    /// The kept sequence sharing the longest prefix with `prompt` (the first of equals),
    /// and how many tokens of it to reuse, when that is at least `MIN_REUSE`.
    fn best_prefix(&self, prompt: &[u32]) -> Option<(SeqId, usize)> {
        if !self.reuse {
            return None;
        }
        let mut best: Option<(SeqId, usize)> = None;
        for id in self.prefix.ids() {
            let lcp = self.seqs[&id].tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
            if best.is_none_or(|(_, b)| lcp > b) {
                best = Some((id, lcp));
            }
        }
        best.map(|(id, lcp)| (id, lcp.min(prompt.len() - 1))).filter(|&(_, keep)| keep >= MIN_REUSE)
    }

    /// Ends a sequence: keeps it for prefix reuse (LRU, at most `MAX_CACHED`) or frees it.
    pub fn finish(&mut self, id: SeqId) {
        if !self.reuse || self.seq_len(id) < MIN_REUSE {
            self.free_seq(id);
            return;
        }
        self.prefix.touch(id);
        while self.prefix.entries.len() > MAX_CACHED && self.evict_one() {}
    }

    pub fn seq_len(&self, id: SeqId) -> usize {
        self.seqs.get(&id).map_or(0, |s| s.len)
    }

    /// The tokens a sequence has consumed.
    pub fn seq_tokens(&self, id: SeqId) -> &[u32] {
        self.seqs.get(&id).map_or(&[], |s| &s.tokens)
    }

    /// Free pages left in the arena.
    pub fn free_pages(&self) -> usize {
        self.pool.free.len()
    }

    /// Forgets the tokens at positions `>= len` (their pages beyond `len` are released).
    pub fn truncate(&mut self, id: SeqId, len: usize) {
        let Some(s) = self.seqs.get_mut(&id) else { return };
        assert!(len <= s.len, "truncate beyond the end");
        s.len = len;
        s.tokens.truncate(len);
        for p in s.pages.drain(len.div_ceil(PAGE)..) {
            self.pool.release(p);
        }
    }

    /// A new sequence sharing the first `len` tokens of `id` (whole pages are shared; a
    /// partially used last page is copied).
    pub fn fork(&mut self, id: SeqId, len: usize, copies: &mut Vec<PageCopy>) -> Result<SeqId> {
        let src = self.seqs.get(&id).context("unknown sequence")?;
        ensure!(len <= src.len, "fork beyond the end");
        let full = len / PAGE;
        let partial = match len.is_multiple_of(PAGE) {
            true => None,
            false => Some(self.pool.alloc().context("KV cache is full")?),
        };
        let mut pages = src.pages[..full].to_vec();
        for &p in &pages {
            self.pool.retain(p);
        }
        if let Some(dst) = partial {
            copies.push(PageCopy { src: src.pages[full], dst });
            pages.push(dst);
        }
        let tokens = src.tokens[..len].to_vec();
        Ok(self.insert(Seq { pages, len, tokens }))
    }

    /// Makes positions `[len, len + n)` of `id` writable: allocates missing pages and
    /// copies shared ones (copy on write).
    fn reserve(&mut self, id: SeqId, n: usize, copies: &mut Vec<PageCopy>) -> Result<()> {
        // Pages that will be allocated: new ones plus shared ones to copy.
        let need = {
            let s = self.seqs.get(&id).context("unknown sequence")?;
            let end = s.len + n;
            // Before evicting anything: a pass that cannot run should not flush the cache.
            ensure!(end <= self.max_pos, "context length {end} exceeds the limit {}", self.max_pos);
            (s.len / PAGE..end.div_ceil(PAGE))
                .filter(|&pi| pi >= s.pages.len() || self.pool.is_shared(s.pages[pi]))
                .count()
        };
        self.ensure_free(need);
        let s = self.seqs.get_mut(&id).context("unknown sequence")?;
        for pi in s.len / PAGE..(s.len + n).div_ceil(PAGE) {
            if pi < s.pages.len() {
                let p = s.pages[pi];
                if self.pool.is_shared(p) {
                    let q = self.pool.alloc().context("KV cache is full")?;
                    copies.push(PageCopy { src: p, dst: q });
                    self.pool.release(p);
                    s.pages[pi] = q;
                }
            } else {
                s.pages.push(self.pool.alloc().context("KV cache is full")?);
            }
        }
        Ok(())
    }

    /// Appends `items[i] = (seq, tokens)` to their sequences, making room for them, and
    /// returns the rows of the pass that computes them.
    pub(crate) fn append(&mut self, items: &[(SeqId, &[u32])], copies: &mut Vec<PageCopy>) -> Result<PassRows> {
        for (id, t) in items {
            self.reserve(*id, t.len(), copies)?;
        }
        let mut meta = PassMeta::default();
        let mut last = Vec::with_capacity(items.len());
        for (id, toks) in items {
            let s = self.seqs.get_mut(id).unwrap();
            let off = meta.pages.len() as u32;
            meta.pages.extend_from_slice(&s.pages);
            for (j, &t) in toks.iter().enumerate() {
                let pos = s.len + j;
                meta.ids.push(t);
                meta.pos.push(pos as u32);
                meta.kv_slot.push(s.pages[pos / PAGE] * PAGE as u32 + (pos % PAGE) as u32);
                meta.pt_off.push(off);
            }
            s.len += toks.len();
            s.tokens.extend_from_slice(toks);
            last.push(meta.ids.len() - 1);
        }
        Ok(PassRows { meta, last })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(kv: &mut KvCache, id: SeqId, toks: &[u32]) -> Vec<PageCopy> {
        let mut copies = Vec::new();
        kv.append(&[(id, toks)], &mut copies).unwrap();
        copies
    }

    #[test]
    fn a_shared_page_is_copied_before_it_is_written() {
        let mut kv = KvCache::new(4, 1024);
        let a = kv.new_seq();
        let toks: Vec<u32> = (0..150).collect();
        assert!(run(&mut kv, a, &toks).is_empty());
        let mut copies = Vec::new();
        // Sharing 140 tokens: page 0 shared, page 1 (partly used) copied.
        let b = kv.fork(a, 140, &mut copies).unwrap();
        assert_eq!(copies.len(), 1);
        assert_eq!((kv.seq_len(b), kv.free_pages()), (140, 1));
        // Writing into the shared first page of a third sequence copies it.
        let c = kv.fork(a, PAGE, &mut Vec::new()).unwrap();
        kv.truncate(c, 100);
        let copies = run(&mut kv, c, &[7]);
        assert_eq!(copies.len(), 1);
        assert_ne!(copies[0].src, copies[0].dst);
        assert_eq!(kv.seq_tokens(c).len(), 101);
    }

    #[test]
    fn kept_sequences_are_evicted_least_recent_first() {
        let mut kv = KvCache::new(3, 1024);
        let (a, b) = (kv.new_seq(), kv.new_seq());
        run(&mut kv, a, &[1; 20]);
        run(&mut kv, b, &[2; 20]);
        kv.finish(a);
        kv.finish(b);
        assert_eq!(kv.free_pages(), 1);
        // A prompt sharing `a`'s prefix reuses it and makes it the most recent.
        let (c, reused) = kv.begin(&[1; 30], &mut Vec::new()).unwrap();
        assert_eq!(reused, 20);
        kv.free_seq(c);
        // Needing two pages evicts `b` first, then `a`.
        let d = kv.new_seq();
        run(&mut kv, d, &[3; 2 * PAGE]);
        assert_eq!(kv.seq_len(b), 0);
        assert_eq!(kv.seq_len(a), 20);
    }
}
