//! `yaxus serve`: HTTPS + bearer-token API, SSE streaming and an embedded web UI.

mod auth;
mod error;
mod queue;
mod secret_file;
mod tls;
mod worker;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
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
    Collector, DocumentCollector, DocumentRequest, ErrorCode, Health, HealthStatus, LANGUAGES, Language,
    TranslateCollector, TranslateRequest,
};
use yaxus_runtime::document::DocJob;
use yaxus_runtime::scheduler::{Job, Sink, Work, prompt_error};
use yaxus_runtime::{LoadOptions, Prompt, load_generator, load_tokenizer, quant_label};
use yaxus_tokenizer::Tokenizer;

use auth::{AccessToken, TokenSource};
use error::{ApiFailure, ApiJson};
use queue::JobQueue;
use tls::TlsFiles;
pub use tls::default_sans;
use worker::{Events, Failure};

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
    /// What `/healthz` reports besides the state of the queue.
    health: Health,
    max_context: usize,
}

type Shared = Arc<AppState>;

impl AppState {
    /// The health report, with the queue as it is now.
    fn health(&self) -> Health {
        let stopped = self.jobs.is_closed();
        Health {
            queue: self.jobs.len(),
            // The inference thread has stopped (after a panic): report it instead of "ok".
            status: if stopped { HealthStatus::Stopped } else { self.health.status },
            ..self.health.clone()
        }
    }

    /// Queues the job of `work` (given its events' channel) and answers with its events:
    /// as an SSE stream, or as one response that `collect` builds from them.
    async fn run<C>(
        &self,
        label: String,
        stream: bool,
        mut collect: C,
        work: impl FnOnce(Box<dyn Sink<C::Event>>) -> Work,
    ) -> Result<Response, ApiFailure>
    where
        C: Collector,
        C::Event: Serialize + Send + 'static,
        C::Response: Serialize,
    {
        let (events, mut rx) = mpsc::unbounded_channel();
        self.jobs.push(Job::new(work(Box::new(Events(events))), label))?;
        if stream {
            return Ok(sse_response(rx));
        }
        while let Some(ev) = rx.recv().await {
            if let Some(r) = collect.push(ev) {
                return Ok(Json(r?).into_response());
            }
        }
        // The worker dropped the channel without a final event.
        Err(ApiFailure::worker_stopped())
    }
}

async fn index() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-cache"), (header::X_CONTENT_TYPE_OPTIONS, "nosniff")], Html(INDEX_HTML))
}

async fn healthz(State(st): State<Shared>) -> Response {
    let h = st.health();
    let status = if h.status == HealthStatus::Ok { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (status, Json(h)).into_response()
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

async fn translate(State(st): State<Shared>, ApiJson(req): ApiJson<TranslateRequest>) -> Result<Response, ApiFailure> {
    let (prompt, max_new_tokens) = Prompt::for_request(&st.tok, &req, st.max_context).map_err(prompt_error)?;
    let label = label(req.source_lang.as_deref(), &req.target_lang);
    st.run(label, req.stream, TranslateCollector::default(), |events| Work::Single { prompt, max_new_tokens, events })
        .await
}

/// Most segments accepted in one document request (a limit of this server: the browser
/// build has its own).
const MAX_SEGMENTS: usize = 2000;

async fn translate_document(
    State(st): State<Shared>,
    ApiJson(req): ApiJson<DocumentRequest>,
) -> Result<Response, ApiFailure> {
    if req.segments.is_empty() || req.segments.len() > MAX_SEGMENTS {
        return Err(ApiFailure::new(
            ErrorCode::InvalidRequest,
            format!("segments must contain 1..={MAX_SEGMENTS} items"),
        ));
    }
    let (n, stream, label) = (req.segments.len(), req.stream, label(req.source_lang.as_deref(), &req.target_lang));
    let doc = DocJob::from_request(req, st.max_context);
    // Options and every segment's prompt without history are checked up front, against the
    // context too (blank segments come back empty); history that does not fit is dropped
    // when the segment is submitted.
    doc.validate(&st.tok).map_err(yaxus_api::ApiError::from)?;
    st.run(label, stream, DocumentCollector::new(n), |events| Work::Document { job: Box::new(doc), events }).await
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
///
/// Everything that can fail on the options alone (the token, the certificate) is done
/// before the model is loaded.
pub fn serve(opts: ServeOptions) -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    opts.check()?;
    let (token, token_source) = AccessToken::resolve(opts.token.as_deref(), &opts.config_dir)?;
    let tls = TlsFiles::resolve(opts.cert.as_deref(), opts.key.as_deref(), &opts.extra_sans, &opts.config_dir)?;
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    let config = rt.block_on(tls.load())?;

    let (tok, engine, health) = load_model(&opts)?;
    let (jobs, failure) = worker::spawn(engine, tok.clone(), opts.queue_limit)?;
    let st = AppState { tok, jobs, health, max_context: opts.max_context };
    let token = Arc::new(token);
    let app = router(Arc::new(st), token.clone());

    rt.block_on(async move {
        let handle = axum_server::Handle::new();
        let failed = failure.reported();
        shut_down_on(handle.clone(), failure);
        print_banner(&opts, &tls, &token, &token_source);
        axum_server::bind_rustls(opts.listen, config)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .context("server error")?;
        anyhow::ensure!(!failed.load(Ordering::SeqCst), "the inference thread failed (see the log above)");
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
}

/// Loads the tokenizer and the model. Returns them and what `/healthz` says about them.
fn load_model(opts: &ServeOptions) -> Result<(Arc<Tokenizer>, Box<dyn yaxus_model::Generator + Send>, Health)> {
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
    Ok((tok, model, health))
}

/// Shuts the server down gracefully on Ctrl-C, or when the inference thread reports a
/// fatal failure (then `serve` returns an error, so that a supervisor can restart the
/// process).
fn shut_down_on(handle: axum_server::Handle, failure: Failure) {
    const GRACE: Duration = Duration::from_secs(5);
    let h = handle.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
        h.graceful_shutdown(Some(GRACE));
    });
    tokio::spawn(async move {
        if failure.wait().await {
            tracing::error!("the inference thread stopped after an internal error; shutting down");
            handle.graceful_shutdown(Some(GRACE));
        }
    });
}

/// Where to reach the server, and how to trust it and authenticate. Printed to the
/// terminal rather than logged, since it may show the token.
fn print_banner(opts: &ServeOptions, tls: &TlsFiles, token: &AccessToken, source: &TokenSource) {
    let port = opts.listen.port();
    eprintln!();
    eprintln!("yaxus listening on https://{}", opts.listen);
    for san in &tls.sans {
        // (an IPv6 address goes in brackets)
        if san.contains(':') {
            eprintln!("  https://[{san}]:{port}/");
        } else {
            eprintln!("  https://{san}:{port}/");
        }
    }
    if let Some(ca) = &tls.ca {
        eprintln!("CA certificate (for `yaxus client --ca` or to import into a browser): {}", ca.display());
    }
    match source {
        TokenSource::Given => {}
        TokenSource::Stored(path) => eprintln!("Access token: {}", path.display()),
        TokenSource::Created(path) => {
            eprintln!("Generated access token (saved to {}):", path.display());
            eprintln!("  {}", token.as_str());
        }
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
        let (jobs, queue) = queue::job_queue(limit);
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
