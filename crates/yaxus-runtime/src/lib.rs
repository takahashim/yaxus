//! What runs a translation, independent of how it is requested: finding and loading the
//! model ([`resolve_model`], [`load_generator`]), turning API requests into prompts
//! ([`prompt_request`]), documents as chains of segments ([`document`]) and the
//! translation memory that feeds earlier outputs back as hints ([`memory`]). The HTTPS
//! server (`yaxus-server`) and the local commands of the CLI are built on it.

pub mod document;
mod load;
pub mod memory;
mod request;
mod resolve;

pub use load::{BackendKind, Decode, DraftOptions, LoadOptions, Prefill, load_generator, load_tokenizer, quant_label};
pub use request::{RunStats, prompt_request, run_request};
pub use resolve::{hf_snapshot, resolve_draft, resolve_model};
