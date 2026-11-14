//! What runs a translation on a loaded engine, independent of how it is requested: API
//! requests as prompts ([`prompt_request`], [`Prompt`]), the requests in flight on an
//! engine with their decoding and translation memory ([`Session`]), documents as chains of
//! segments ([`document`]) and single requests run to completion ([`run_request`]). The
//! HTTPS server (`yaxus-server`), the local commands of the CLI and the browser build
//! (`yaxus-browser`) are built on it. Loading a model is not here: `yaxus-native` does it
//! on the native backends, and the browser loads its own.

pub mod document;
pub mod memory;
mod prompt;
mod run;
pub mod scheduler;
pub mod score;
mod session;

pub use prompt::{Prompt, prompt_request};
pub use run::{RunStats, run_request, run_request_async};
pub use session::{Finished, STOP_TOKENS, Session, Update};
