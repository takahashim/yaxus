//! `yaxus serve`: HTTPS + bearer-token API, SSE streaming and an embedded web UI.

mod auth;
mod queue;
mod secret_file;
mod tls;
mod worker;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use yaxus_api::{
    ApiError, DocumentEvent, DocumentRequest, DocumentResponse, ErrorCode, ErrorResponse, Health, HealthStatus,
    LANGUAGES, Language, StreamEvent, TranslateRequest, TranslateResponse,
};
use yaxus_pfml::{PromptError, PromptErrorKind};
use yaxus_runtime::document::{self, DocError};
use yaxus_runtime::{LoadOptions, Prompt, load_generator, load_tokenizer, quant_label};
use yaxus_tokenizer::Tokenizer;

use auth::AccessToken;
use queue::{JobQueue, Rejected, job_queue};
pub use tls::default_sans;
use worker::{Job, Work};

const INDEX_HTML: &str = include_str!("../../../web/index.html");

pub struct ServeOptions {
    pub model_dir: PathBuf,
    pub load: LoadOptions,
    pub model_name: String,
    pub listen: SocketAddr,
    /// Bearer token; generated and stored in `config_dir/token` when `None`.
    pub token: Option<String>,
    pub cert: Option<PathBuf>,
    pub key: Option<PathBuf>,
    /// Extra names/IPs for the self-signed certificate.
    pub extra_sans: Vec<String>,
    pub config_dir: PathBuf,
    pub queue_limit: usize,
    /// Maximum prompt + generated tokens per request.
    pub max_context: usize,
}

struct AppState {
    tok: Arc<Tokenizer>,
    jobs: JobQueue,
    health: Health,
    max_context: usize,
}

type Shared = Arc<AppState>;

/// The HTTP status of each error code.
fn status_of(code: ErrorCode) -> StatusCode {
    match code {
        ErrorCode::InvalidRequest | ErrorCode::PromptTooLong => StatusCode::BAD_REQUEST,
        ErrorCode::Unauthorized => StatusCode::UNAUTHORIZED,
        ErrorCode::QueueFull => StatusCode::TOO_MANY_REQUESTS,
        ErrorCode::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        ErrorCode::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn error_response(error: ApiError) -> Response {
    (status_of(error.code), Json(ErrorResponse { error })).into_response()
}

fn error(code: ErrorCode, msg: impl Into<String>) -> Response {
    error_response(ApiError::new(code, msg))
}

/// A request whose prompt cannot be built, as the API reports it.
pub(crate) fn prompt_error(e: PromptError) -> ApiError {
    let code = match e.kind {
        PromptErrorKind::Invalid => ErrorCode::InvalidRequest,
        PromptErrorKind::TooLong => ErrorCode::PromptTooLong,
    };
    ApiError::new(code, e.message)
}

pub(crate) fn segment_error(e: document::SegmentError) -> ApiError {
    prompt_error(e.error).in_segment(e.segment)
}

async fn index() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-cache"), (header::X_CONTENT_TYPE_OPTIONS, "nosniff")], Html(INDEX_HTML))
}

async fn healthz(State(st): State<Shared>) -> Response {
    let mut h = st.health.clone();
    h.queue = st.jobs.len();
    // The inference thread has stopped (after a panic): report it instead of "ok".
    if st.jobs.is_closed() {
        h.status = HealthStatus::Stopped;
        return (StatusCode::SERVICE_UNAVAILABLE, Json(h)).into_response();
    }
    Json(h).into_response()
}

async fn languages() -> Json<Vec<Language>> {
    Json(LANGUAGES.iter().map(|(c, n)| Language { code: c.to_string(), name: n.to_string() }).collect())
}

/// `source->target` for the log line.
fn label(source_lang: Option<&str>, target_lang: &str) -> String {
    format!("{}->{}", source_lang.unwrap_or("auto"), target_lang)
}

/// The events of a job as an SSE stream, one JSON object per event.
fn sse_response<E: Serialize + Send + 'static>(rx: mpsc::UnboundedReceiver<E>) -> Response {
    let stream = UnboundedReceiverStream::new(rx)
        .map(|ev| Ok::<_, Infallible>(SseEvent::default().data(serde_json::to_string(&ev).unwrap())));
    let mut r = Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    // Keep reverse proxies from buffering the stream.
    r.headers_mut().insert("x-accel-buffering", HeaderValue::from_static("no"));
    r
}

/// The worker dropped a job's channel without a final event.
fn worker_stopped() -> Response {
    error(ErrorCode::Unavailable, "inference worker stopped")
}

async fn translate(
    State(st): State<Shared>,
    body: Result<Json<TranslateRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return error(ErrorCode::InvalidRequest, e.body_text()),
    };
    let (prompt, max_new_tokens) = match Prompt::for_request(&st.tok, &req, st.max_context) {
        Ok(p) => p,
        Err(e) => return error_response(prompt_error(e)),
    };
    let (events, mut rx) = mpsc::unbounded_channel();
    let job = Job {
        work: Work::Single { prompt, max_new_tokens, events },
        enqueued: Instant::now(),
        label: label(req.source_lang.as_deref(), &req.target_lang),
    };
    if let Err(r) = enqueue(&st, job) {
        return r;
    }
    if req.stream {
        return sse_response(rx);
    }
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Delta { text: t } => text.push_str(&t),
            StreamEvent::Done { usage } => return Json(TranslateResponse { text, usage }).into_response(),
            StreamEvent::Error { error } => return error_response(error),
        }
    }
    worker_stopped()
}

// (the error is the response to send; boxing it buys nothing on this path)
#[allow(clippy::result_large_err)]
fn enqueue(st: &AppState, job: Job) -> std::result::Result<(), Response> {
    st.jobs.push(job).map_err(|e| match e {
        Rejected::Full => error(ErrorCode::QueueFull, "translation queue is full; retry later"),
        Rejected::Stopped => error(ErrorCode::Unavailable, "inference worker stopped"),
    })
}

/// Most segments accepted in one document request.
const MAX_SEGMENTS: usize = 2000;

async fn translate_document(
    State(st): State<Shared>,
    body: Result<Json<DocumentRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return error(ErrorCode::InvalidRequest, e.body_text()),
    };
    if req.segments.is_empty() || req.segments.len() > MAX_SEGMENTS {
        return error(ErrorCode::InvalidRequest, format!("segments must contain 1..={MAX_SEGMENTS} items"));
    }
    let (n, stream, label) = (req.segments.len(), req.stream, label(req.source_lang.as_deref(), &req.target_lang));
    let doc = document::DocJob::from_request(req, st.max_context);
    // Options and every segment's prompt without history are checked up front, against the
    // context too (blank segments come back empty); history that does not fit is dropped
    // when the segment is submitted.
    if let Err(e) = doc.validate(&st.tok) {
        return error_response(match e {
            DocError::Options(e) => prompt_error(e),
            DocError::Segment(e) => segment_error(e),
        });
    }
    let (events, mut rx) = mpsc::unbounded_channel();
    let job = Job { work: Work::Document { job: Box::new(doc), events }, enqueued: Instant::now(), label };
    if let Err(r) = enqueue(&st, job) {
        return r;
    }
    if stream {
        return sse_response(rx);
    }
    let mut out = vec![String::new(); n];
    while let Some(ev) = rx.recv().await {
        match ev {
            DocumentEvent::Delta { segment, text } => out[segment].push_str(&text),
            DocumentEvent::SegmentDone { .. } => {}
            DocumentEvent::Done { usage } => {
                return Json(DocumentResponse { segments: out, usage }).into_response();
            }
            DocumentEvent::Error { error } => return error_response(error),
        }
    }
    worker_stopped()
}

fn router(st: Shared, token: Arc<AccessToken>) -> Router {
    let api = Router::new()
        .route("/v1/translate", post(translate))
        .route("/v1/translate/document", post(translate_document))
        .route_layer(middleware::from_fn_with_state(token, auth::require));
    Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/v1/languages", get(languages))
        .merge(api)
        .with_state(st)
}

/// Loads the model, starts the inference thread and serves until Ctrl-C.
pub fn serve(opts: ServeOptions) -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    opts.check()?;
    let (token, created) = match opts.token.as_deref() {
        Some(t) => (AccessToken::new(t)?, false),
        None => AccessToken::load_or_create(&opts.config_dir)?,
    };
    let tls = opts.tls_files()?;
    let (st, failed) = start_inference(&opts)?;
    let token = Arc::new(token);
    let app = router(Arc::new(st), token.clone());

    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    rt.block_on(async move {
        let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&tls.cert, &tls.key)
            .await
            .context("loading TLS certificate")?;
        let handle = axum_server::Handle::new();
        let inference_failed = shut_down_on(handle.clone(), failed);
        print_banner(&opts, &tls, &token, created);
        axum_server::bind_rustls(opts.listen, config)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .context("server error")?;
        anyhow::ensure!(!inference_failed.load(Ordering::SeqCst), "the inference thread failed (see the log above)");
        Ok(())
    })
}

impl ServeOptions {
    fn check(&self) -> Result<()> {
        // The KV's RoPE tables cover DEFAULT_MAX_POS positions; longer requests would pass
        // the checks of the handlers and then fail in the engine.
        anyhow::ensure!(
            // (at least 2: documents give each segment up to half of it to generate)
            (2..=yaxus_model::DEFAULT_MAX_POS).contains(&self.max_context),
            "max context must be 2..={} tokens",
            yaxus_model::DEFAULT_MAX_POS
        );
        Ok(())
    }

    /// The user's certificate, or the self-signed one in `config_dir` (created if needed).
    fn tls_files(&self) -> Result<tls::TlsFiles> {
        match (&self.cert, &self.key) {
            (Some(c), Some(k)) => Ok(tls::TlsFiles { cert: c.clone(), key: k.clone(), ca: None, sans: vec![] }),
            (None, None) => {
                let mut sans = default_sans();
                sans.extend(self.extra_sans.iter().cloned());
                tls::ensure_self_signed(&self.config_dir, sans)
            }
            _ => anyhow::bail!("--cert and --key must be given together"),
        }
    }
}

/// Loads the model and starts the inference thread on it. Returns the handlers' state,
/// and the channel on which the thread reports a fatal failure.
fn start_inference(opts: &ServeOptions) -> Result<(AppState, tokio::sync::oneshot::Receiver<()>)> {
    let tok = Arc::new(load_tokenizer(&opts.model_dir)?);
    let t0 = Instant::now();
    let model = load_generator(&opts.model_dir, &opts.load)
        .with_context(|| format!("loading model from {}", opts.model_dir.display()))?;
    let quant = quant_label(&opts.model_dir, opts.load.quant);
    tracing::info!(
        "loaded {} ({quant}, {:.2} GB) on {} in {:.2?}",
        opts.model_name,
        model.weight_bytes() as f64 / 1e9,
        model.backend_name(),
        t0.elapsed()
    );
    let health = Health {
        status: HealthStatus::Ok,
        model: opts.model_name.clone(),
        backend: model.backend_name(),
        quant,
        queue: 0,
    };
    let (jobs, queue) = job_queue(opts.queue_limit);
    let (failed_tx, failed_rx) = tokio::sync::oneshot::channel();
    let worker_tok = tok.clone();
    std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || worker::run(model, worker_tok, queue, failed_tx))?;
    Ok((AppState { tok, jobs, health, max_context: opts.max_context }, failed_rx))
}

/// Shuts the server down gracefully on Ctrl-C, or when the inference thread reports a
/// fatal failure on `failed` (then `serve` returns an error, so that a supervisor can
/// restart the process). Returns the flag set in that case.
fn shut_down_on(handle: axum_server::Handle, failed: tokio::sync::oneshot::Receiver<()>) -> Arc<AtomicBool> {
    const GRACE: Duration = Duration::from_secs(5);
    let h = handle.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
        h.graceful_shutdown(Some(GRACE));
    });
    let flag = Arc::new(AtomicBool::new(false));
    let f = flag.clone();
    tokio::spawn(async move {
        // (an Err means the thread ended normally and dropped the sender)
        if failed.await.is_ok() {
            tracing::error!("the inference thread stopped after an internal error; shutting down");
            f.store(true, Ordering::SeqCst);
            handle.graceful_shutdown(Some(GRACE));
        }
    });
    flag
}

/// Where to reach the server, and how to trust it and authenticate.
fn print_banner(opts: &ServeOptions, tls: &tls::TlsFiles, token: &AccessToken, created: bool) {
    let port = opts.listen.port();
    eprintln!();
    eprintln!("yaxus listening on https://{}", opts.listen);
    for san in tls.sans.iter().filter(|s| !s.contains(':')) {
        eprintln!("  https://{san}:{port}/");
    }
    if let Some(ca) = &tls.ca {
        eprintln!("CA certificate (for `yaxus client --ca` or to import into a browser): {}", ca.display());
    }
    if created {
        eprintln!("Generated access token (saved to {}):", opts.config_dir.join("token").display());
        eprintln!("  {}", token.as_str());
    } else if opts.token.is_none() {
        eprintln!("Access token: {}", opts.config_dir.join("token").display());
    }
    eprintln!();
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    use super::*;

    const TOKEN: &str = "0123456789abcdef";

    /// The router on a tiny tokenizer, with no inference thread (`queue` is the receiving
    /// end: drop it to mimic a stopped one) and a context that fits the prompt of "x" with
    /// room to spare, but not that of a hundred "x".
    fn app(limit: usize) -> (Router, queue::Jobs) {
        let tok = Arc::new(yaxus_tokenizer::testing::plamo_like(&[("x", 0.0)]));
        let x = TranslateRequest { text: "x".into(), target_lang: "en".into(), ..Default::default() };
        let max_context = Prompt::for_request(&tok, &x, usize::MAX).unwrap().0.len() + 20;
        let (jobs, queue) = job_queue(limit);
        let health =
            Health { status: HealthStatus::Ok, model: "m".into(), backend: "b".into(), quant: "q".into(), queue: 0 };
        let st = Arc::new(AppState { tok, jobs, health, max_context });
        (router(st, Arc::new(AccessToken::new(TOKEN).unwrap())), queue)
    }

    fn request(uri: &str, body: Option<serde_json::Value>, token: bool) -> Request<Body> {
        let mut req = Request::builder().uri(uri).method(if body.is_some() { "POST" } else { "GET" });
        if token {
            req = req.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"));
        }
        match body {
            Some(b) => req.header(header::CONTENT_TYPE, "application/json").body(Body::from(b.to_string())),
            None => req.body(Body::empty()),
        }
        .unwrap()
    }

    async fn call(
        app: &Router,
        uri: &str,
        body: Option<serde_json::Value>,
        token: bool,
    ) -> (StatusCode, serde_json::Value) {
        let res = app.clone().oneshot(request(uri, body, token)).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    fn code(body: &serde_json::Value) -> &str {
        body["error"]["code"].as_str().unwrap_or_else(|| panic!("no error code in {body}"))
    }

    #[tokio::test]
    async fn errors_carry_a_code_matching_the_status() {
        let (app, _queue) = app(1);
        let one = serde_json::json!({"text": "x", "target_lang": "en"});

        let (s, b) = call(&app, "/v1/translate", Some(one.clone()), false).await;
        assert_eq!((s, code(&b)), (StatusCode::UNAUTHORIZED, "unauthorized"));

        // A form outside the enum is rejected when the request is parsed.
        let casual = serde_json::json!({"text": "x", "target_lang": "ja", "form": "casual"});
        let (s, b) = call(&app, "/v1/translate", Some(casual), true).await;
        assert_eq!((s, code(&b)), (StatusCode::BAD_REQUEST, "invalid_request"));
        let (s, b) = call(&app, "/v1/translate", Some(serde_json::json!({"text": "x", "target_lang": ""})), true).await;
        assert_eq!((s, code(&b)), (StatusCode::BAD_REQUEST, "invalid_request"));

        let long = serde_json::json!({"text": "x".repeat(100), "target_lang": "en"});
        let (s, b) = call(&app, "/v1/translate", Some(long), true).await;
        assert_eq!((s, code(&b)), (StatusCode::BAD_REQUEST, "prompt_too_long"));

        // A document names the segment.
        let doc = |second: &str| serde_json::json!({"segments": ["x", second], "target_lang": "en", "stream": true});
        let (s, b) = call(&app, "/v1/translate/document", Some(doc(&"x".repeat(100))), true).await;
        assert_eq!((s, code(&b), &b["error"]["segment"]), (StatusCode::BAD_REQUEST, "prompt_too_long", &1.into()));
        // A streamed document takes the one place in the queue (nothing runs it, and its
        // body is not read); the next request finds the queue full.
        let res = app.clone().oneshot(request("/v1/translate/document", Some(doc("x")), true)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let (s, b) = call(&app, "/v1/translate/document", Some(doc("x")), true).await;
        assert_eq!((s, code(&b)), (StatusCode::TOO_MANY_REQUESTS, "queue_full"));
    }

    #[tokio::test]
    async fn a_stopped_worker_is_reported() {
        let (app, queue) = app(4);
        let (s, b) = call(&app, "/healthz", None, false).await;
        assert_eq!((s, b["status"].as_str()), (StatusCode::OK, Some("ok")));
        drop(queue);
        let (s, b) = call(&app, "/healthz", None, false).await;
        assert_eq!((s, b["status"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("stopped")));
        let one = serde_json::json!({"text": "x", "target_lang": "en"});
        let (s, b) = call(&app, "/v1/translate", Some(one), true).await;
        assert_eq!((s, code(&b)), (StatusCode::SERVICE_UNAVAILABLE, "unavailable"));
    }
}
