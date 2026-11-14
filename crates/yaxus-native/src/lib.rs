//! What puts a model on the native backends (CPU, Vulkan, Metal): finding it on disk, the
//! options of loading it, and loading it. The browser has no files, backends or options to
//! choose: `yaxus-browser` loads its own model from the bytes of a file the page read.
//! What runs a translation on the loaded engine is `yaxus-runtime`, for both.

mod load;
mod options;
mod resolve;

pub use load::{load_generator, load_tokenizer};
pub use options::{BackendKind, Decode, DraftOptions, LoadOptions, Prefill, Scheduling, default_decode_rows};
pub use resolve::{DraftSpec, ResolvedModel, quant_label, resolve_draft, resolve_model};
