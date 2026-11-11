//! Small vocabularies with the PLaMo special-token layout, for tests (here and, with the
//! `test-util` feature, in other crates).

use crate::{BOS_ID, TAG_ID, TokenKind, Tokenizer};

/// Control tokens at ids 0..=16 (`<|plamo:bos|>` at 1, `<|plamo:tag|>` at 16), the 256
/// byte tokens, then `normal` with their scores, as parallel arrays.
pub fn vocab(normal: &[(&str, f64)]) -> (Vec<String>, Vec<TokenKind>, Vec<f64>) {
    let mut tokens: Vec<String> = (0..=TAG_ID)
        .map(|id| match id {
            BOS_ID => "<|plamo:bos|>".to_string(),
            TAG_ID => "<|plamo:tag|>".to_string(),
            _ => format!("<|plamo:c{id}|>"),
        })
        .collect();
    let mut kinds = vec![TokenKind::Control; tokens.len()];
    tokens.extend((0..=255u8).map(|b| format!("<0x{b:02X}>")));
    kinds.extend([TokenKind::Byte; 256]);
    let mut scores = vec![0.0; tokens.len()];
    for &(t, score) in normal {
        tokens.push(t.to_string());
        kinds.push(TokenKind::Normal);
        scores.push(score);
    }
    (tokens, kinds, scores)
}

/// The tokenizer of [`vocab`]`(normal)`. The normal tokens get ids from 273 on, in order.
pub fn plamo_like(normal: &[(&str, f64)]) -> Tokenizer {
    let (tokens, kinds, scores) = vocab(normal);
    Tokenizer::from_vocab(tokens, kinds, scores).unwrap()
}
