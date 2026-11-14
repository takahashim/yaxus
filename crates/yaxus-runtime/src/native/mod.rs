//! What puts a model on the native backends: finding it on disk, the options of loading
//! it, and loading it. The browser has no files, backends or options to choose: it loads
//! its own model (`yaxus-web`), so none of this is built for wasm32.

mod load;
mod options;
mod resolve;

pub use load::{load_generator, load_tokenizer};
pub use options::{BackendKind, Decode, DraftOptions, LoadOptions, Prefill, Scheduling, default_decode_rows};
pub use resolve::{DraftSpec, ResolvedModel, quant_label, resolve_draft, resolve_model};
