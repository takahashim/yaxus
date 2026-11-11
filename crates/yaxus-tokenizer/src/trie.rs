//! Character trie over the vocabulary, walked forward from each position by the
//! Viterbi DP.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Hasher for the `(node << 32) | codepoint` edge keys. The default SipHash dominates
/// the Viterbi inner loop; one 128-bit multiply mixes both halves into every bit.
#[derive(Default)]
struct EdgeHasher(u64);

impl Hasher for EdgeHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys are hashed");
    }

    fn write_u64(&mut self, x: u64) {
        let p = (x as u128) * 0x9e37_79b9_7f4a_7c15;
        self.0 = (p as u64) ^ ((p >> 64) as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Trie {
    /// Per node: (token id, integer score) when the path from the root spells a token.
    /// Node 0 is the root.
    tokens: Vec<Option<(u32, i64)>>,
    edges: HashMap<u64, u32, BuildHasherDefault<EdgeHasher>>,
}

#[inline]
fn edge(node: u32, c: char) -> u64 {
    ((node as u64) << 32) | c as u64
}

impl Trie {
    pub(crate) fn new() -> Self {
        Trie { tokens: vec![None], edges: Default::default() }
    }

    /// Adds `token` (non-empty), replacing an earlier token with the same text.
    pub(crate) fn insert(&mut self, token: &str, id: u32, score: i64) {
        debug_assert!(!token.is_empty());
        let mut node = 0u32;
        for c in token.chars() {
            node = match self.edges.get(&edge(node, c)) {
                Some(&next) => next,
                None => {
                    let next = self.tokens.len() as u32;
                    self.tokens.push(None);
                    self.edges.insert(edge(node, c), next);
                    next
                }
            };
        }
        self.tokens[node as usize] = Some((id, score));
    }

    /// The tokens that are prefixes of `data`, shortest first, as `(length, id, score)`.
    #[inline]
    pub(crate) fn prefixes<'a>(&'a self, data: &'a [char]) -> Prefixes<'a> {
        Prefixes { trie: self, data, node: 0, len: 0 }
    }
}

pub(crate) struct Prefixes<'a> {
    trie: &'a Trie,
    data: &'a [char],
    node: u32,
    len: usize,
}

impl Iterator for Prefixes<'_> {
    type Item = (usize, u32, i64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(&c) = self.data.get(self.len) {
            self.node = *self.trie.edges.get(&edge(self.node, c))?;
            self.len += 1;
            if let Some((id, score)) = self.trie.tokens[self.node as usize] {
                return Some((self.len, id, score));
            }
        }
        None
    }
}
