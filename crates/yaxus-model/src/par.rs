//! Parallel iteration over chunks: rayon on native, sequential on wasm (there are
//! no threads in a web page's main thread). The two shims return iterators with
//! the same `zip` / `enumerate` / `for_each` surface, so the call sites below
//! read identically; only `with_min_len` (a rayon scheduling hint) is native-only.

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use rayon::prelude::*;

    pub use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    pub fn chunks<T: Send + Sync>(v: &[T], n: usize) -> rayon::slice::Chunks<'_, T> {
        v.par_chunks(n)
    }
    pub fn chunks_mut<T: Send>(v: &mut [T], n: usize) -> rayon::slice::ChunksMut<'_, T> {
        v.par_chunks_mut(n)
    }
    pub fn iter_mut<T: Send>(v: &mut [T]) -> rayon::slice::IterMut<'_, T> {
        v.par_iter_mut()
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    pub fn chunks<T>(v: &[T], n: usize) -> std::slice::Chunks<'_, T> {
        v.chunks(n)
    }
    pub fn chunks_mut<T>(v: &mut [T], n: usize) -> std::slice::ChunksMut<'_, T> {
        v.chunks_mut(n)
    }
    pub fn iter_mut<T>(v: &mut [T]) -> std::slice::IterMut<'_, T> {
        v.iter_mut()
    }
}

pub(crate) use imp::*;
