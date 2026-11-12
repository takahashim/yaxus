//! Outputs of finished requests, to answer a repeated request without any pass.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use super::{Finish, Request};

struct Cached {
    prompt: Vec<u32>,
    stop: Vec<u32>,
    out: Vec<u32>,
    finish: Finish,
}

/// Outputs of finished requests by prompt (least recently used first in `order`).
#[derive(Default)]
pub(super) struct ResponseCache {
    map: HashMap<u64, Cached>,
    order: VecDeque<u64>,
}

fn cache_key(prompt: &[u32], stop: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    prompt.hash(&mut h);
    stop.hash(&mut h);
    h.finish()
}

impl ResponseCache {
    /// The cached output for `req`, if it is the full answer to it.
    pub(super) fn get(&mut self, req: &Request) -> Option<(Vec<u32>, Finish)> {
        let key = cache_key(&req.prompt, &req.stop);
        let c = self.map.get(&key).filter(|c| c.prompt == req.prompt && c.stop == req.stop)?;
        let hit = if c.out.len() >= req.max_new_tokens {
            Some((c.out[..req.max_new_tokens].to_vec(), Finish::Length))
        } else if c.finish == Finish::Stop {
            Some((c.out.clone(), Finish::Stop))
        } else {
            None // cut by a smaller limit
        };
        if hit.is_some() {
            self.touch(key);
        }
        hit
    }

    pub(super) fn put(&mut self, cap: usize, prompt: Vec<u32>, stop: Vec<u32>, out: Vec<u32>, finish: Finish) {
        let key = cache_key(&prompt, &stop);
        self.map.insert(key, Cached { prompt, stop, out, finish });
        self.touch(key);
        while self.map.len() > cap
            && let Some(old) = self.order.pop_front()
        {
            self.map.remove(&old);
        }
    }

    fn touch(&mut self, key: u64) {
        self.order.retain(|&k| k != key);
        self.order.push_back(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_cache_answers_only_complete_outputs() {
        let mut c = ResponseCache::default();
        let req = |max| Request::new(0, vec![1, 2, 3], max, vec![7]);
        c.put(2, vec![1, 2, 3], vec![7], vec![4, 5], Finish::Stop);
        assert_eq!(c.get(&req(10)), Some((vec![4, 5], Finish::Stop)));
        assert_eq!(c.get(&req(1)), Some((vec![4], Finish::Length)));
        // Another stop list is another request.
        assert_eq!(c.get(&Request::new(0, vec![1, 2, 3], 10, vec![8])), None);
        // An output cut at the limit does not answer a request with a larger limit.
        c.put(2, vec![9], vec![7], vec![4, 5], Finish::Length);
        assert_eq!(c.get(&Request::new(0, vec![9], 3, vec![7])), None);
        assert_eq!(c.get(&Request::new(0, vec![9], 2, vec![7])), Some((vec![4, 5], Finish::Length)));
        // Least recently used entries go first.
        c.put(2, vec![8], vec![7], vec![1], Finish::Stop);
        assert_eq!(c.get(&req(10)), None);
    }
}
