//! Translation memory: the latest output tokens per source text, given to the engine as
//! the request's hint when the same text is translated again with a different prompt
//! (for example after an earlier paragraph of a document was edited, which changes the
//! history). Identical prompts are answered by the engine's response cache instead.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use yaxus_pfml::ValidRequest;

/// Entries kept by default.
pub const DEFAULT_ENTRIES: usize = 4096;

pub struct Memory {
    cap: usize,
    map: HashMap<MemoryKey, Vec<u32>>,
    /// Least recently used first.
    order: VecDeque<MemoryKey>,
}

/// The key under which the memory keeps a translation (see [`key`]).
///
/// It is a hash, so two different requests may share one. That costs speed only: the
/// entry is a hint, whose tokens the engine verifies before it emits them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryKey(u64);

impl MemoryKey {
    /// A key of tests' own choosing.
    #[cfg(test)]
    pub(crate) const fn raw(n: u64) -> Self {
        MemoryKey(n)
    }
}

/// What a translation mostly depends on: the text, the languages and the form (not the
/// history, context or glossary, which a hint may safely ignore).
pub fn key(req: &ValidRequest) -> MemoryKey {
    let mut h = DefaultHasher::new();
    (req.text(), req.target_lang(), req.source_lang(), req.form()).hash(&mut h);
    MemoryKey(h.finish())
}

impl Memory {
    pub fn new(cap: usize) -> Self {
        Memory { cap, map: HashMap::new(), order: VecDeque::new() }
    }

    pub fn get(&self, key: MemoryKey) -> Vec<u32> {
        self.map.get(&key).cloned().unwrap_or_default()
    }

    pub fn put(&mut self, key: MemoryKey, tokens: Vec<u32>) {
        if self.cap == 0 || tokens.is_empty() {
            return;
        }
        // Only a key already present needs its old place in the order removed.
        if self.map.insert(key, tokens).is_some() {
            self.order.retain(|&k| k != key);
        }
        self.order.push_back(key);
        while self.map.len() > self.cap
            && let Some(old) = self.order.pop_front()
        {
            self.map.remove(&old);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_recently_put_goes_first() {
        let k = MemoryKey::raw;
        let mut m = Memory::new(2);
        m.put(k(1), vec![1]);
        m.put(k(2), vec![2]);
        m.put(k(1), vec![10]); // 1 is now the newest
        m.put(k(3), vec![3]);
        assert_eq!((m.get(k(1)), m.get(k(2)), m.get(k(3))), (vec![10], vec![], vec![3]));
        assert_eq!(m.order.len(), 2);
        m.put(k(4), vec![]); // empty outputs are not kept
        assert_eq!(m.get(k(4)), Vec::<u32>::new());
    }
}
