//! What runs a translation, independent of how it is requested: finding and loading the
//! model ([`resolve_model`], [`load_generator`]), API requests as prompts
//! ([`prompt_request`], [`Prompt`]), the requests in flight on an engine with their
//! decoding and translation memory ([`Session`]), documents as chains of segments
//! ([`document`]) and single requests run to completion ([`run_request`]). The HTTPS
//! server (`yaxus-server`) and the local commands of the CLI are built on it.

pub mod document;
pub mod memory;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod prompt;
mod run;
pub mod scheduler;
pub mod score;
mod session;

#[cfg(not(target_arch = "wasm32"))]
pub use native::{
    BackendKind, Decode, DraftOptions, DraftSpec, LoadOptions, Prefill, ResolvedModel, Scheduling, default_decode_rows,
    load_generator, load_tokenizer, quant_label, resolve_draft, resolve_model,
};
pub use prompt::{Prompt, prompt_request};
pub use run::{RunStats, run_request, run_request_async};
pub use session::{Finished, STOP_TOKENS, Session, Update};
