//! Incremental detokenization for streaming output.

use crate::Tokenizer;

/// Accumulates token bytes and yields text only once it forms complete UTF-8,
/// so multi-byte characters split across byte tokens are never emitted half-way.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one token and returns the newly completed text (possibly empty).
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        let mut out = String::new();
        self.push_into(tok, id, &mut out);
        out
    }

    /// Feeds one token and appends the newly completed text to `out`, without
    /// allocating when `out` has room. The new text is `out[len_before..]`.
    pub fn push_into(&mut self, tok: &Tokenizer, id: u32, out: &mut String) {
        self.pending.extend_from_slice(tok.token_bytes(id));
        let mut done = 0;
        loop {
            match std::str::from_utf8(&self.pending[done..]) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[done..done + valid]).unwrap());
                    match e.error_len() {
                        // Incomplete sequence at the end: wait for more bytes.
                        None => {
                            self.pending.drain(..done + valid);
                            return;
                        }
                        Some(len) => {
                            out.push('\u{FFFD}');
                            done += valid + len;
                        }
                    }
                }
            }
        }
    }

    /// Flushes any trailing incomplete bytes as U+FFFD.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}
