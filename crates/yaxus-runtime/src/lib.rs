//! What runs a translation, independent of how it is requested: finding and loading the
//! model ([`resolve_model`], [`load_generator`]), API requests as prompts
//! ([`prompt_request`], [`Prompt`]), the requests in flight on an engine with their
//! decoding and translation memory ([`Session`]), documents as chains of segments
//! ([`document`]) and single requests run to completion ([`run_request`]). The HTTPS
//! server (`yaxus-server`) and the local commands of the CLI are built on it.

pub mod document;
mod load;
pub mod memory;
mod prompt;
mod resolve;
mod run;
mod session;

pub use load::{
    BackendKind, Decode, DraftOptions, LoadOptions, Prefill, Scheduling, load_generator, load_tokenizer, quant_label,
};
pub use prompt::{Prompt, prompt_request};
pub use resolve::{ResolvedModel, resolve_draft, resolve_model};
pub use run::{RunStats, run_request};
pub use session::{Finished, Session, Update};
