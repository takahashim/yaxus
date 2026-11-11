//! Translation memory: the latest output tokens per source text, given to the engine as
//! the request's hint when the same text is translated again with a different prompt
//! (for example after an earlier paragraph of a document was edited, which changes the
//! history). Identical prompts are answered by the engine's response cache instead.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use yaxus_tokenizer::TranslationRequest;

/// Entries kept by default.
pub const DEFAULT_ENTRIES: usize = 4096;

pub struct Memory {
    cap: usize,
    map: HashMap<u64, Vec<u32>>,
    /// Least recently used first.
    order: VecDeque<u64>,
}

/// What a translation mostly depends on: the text, the languages and the form (not the
/// history, context or glossary, which a hint may safely ignore).
pub fn key(req: &TranslationRequest) -> u64 {
    let mut h = DefaultHasher::new();
    (&req.text, &req.target_lang, &req.source_lang, &req.form).hash(&mut h);
    h.finish()
}

impl Memory {
    pub fn new(cap: usize) -> Self {
        Memory { cap, map: HashMap::new(), order: VecDeque::new() }
    }

    pub fn get(&self, key: u64) -> Vec<u32> {
        self.map.get(&key).cloned().unwrap_or_default()
    }

    pub fn put(&mut self, key: u64, tokens: Vec<u32>) {
        if self.cap == 0 || tokens.is_empty() {
            return;
        }
        self.map.insert(key, tokens);
        self.order.retain(|&k| k != key);
        self.order.push_back(key);
        while self.map.len() > self.cap
            && let Some(old) = self.order.pop_front()
        {
            self.map.remove(&old);
        }
    }
}
