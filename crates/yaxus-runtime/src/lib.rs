//! What runs a translation, independent of how it is requested: finding and loading the
//! model ([`resolve_model`], [`load_generator`]), API requests as prompts
//! ([`prompt_request`], [`Prompt`]), the requests in flight on an engine with their
//! decoding and translation memory ([`Session`]), documents as chains of segments
//! ([`document`]) and single requests run to completion ([`run_request`]). The HTTPS
//! server (`yaxus-server`) and the local commands of the CLI are built on it.

pub mod document;
mod load;
pub mod memory;
mod options;
mod prompt;
mod resolve;
mod run;
mod session;

pub use load::{load_generator, load_tokenizer};
pub use options::{BackendKind, Decode, DraftOptions, LoadOptions, Prefill, Scheduling, default_decode_rows};
pub use prompt::{Prompt, prompt_request};
pub use resolve::{DraftSpec, ResolvedModel, quant_label, resolve_draft, resolve_model};
pub use run::{RunStats, run_request};
pub use session::{Finished, STOP_TOKENS, Session, Update};
