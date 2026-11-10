use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use yakusel_api::{DEFAULT_MAX_NEW_TOKENS, Pair, StreamEvent, TranslateRequest, TranslateResponse};

#[cfg(feature = "server")]
mod local;

#[derive(Parser)]
#[command(name = "yakusel", version, about = "PLaMo 3 Translate in pure Rust")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Translation options shared by `translate` and `client`.
#[derive(Args)]
struct LangArgs {
    /// Target language code (en, ja, zh-hans, ja-easy, ...).
    #[arg(long)]
    to: String,
    /// Source language code (detected automatically if omitted).
    #[arg(long)]
    from: Option<String>,
    /// polite or plain (only with --to ja).
    #[arg(long)]
    form: Option<String>,
    /// Reference context for the whole text (situation, surrounding text).
    #[arg(long)]
    context: Option<String>,
    /// Glossary entry "source=target" (repeatable).
    #[arg(long = "glossary", short = 'g', value_name = "SRC=DST")]
    glossary: Vec<String>,
    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_new_tokens: usize,
    /// History budget in characters: earlier paragraphs and their translations are sent
    /// with each paragraph until this is exceeded (then the older half is dropped). 0 = none.
    #[arg(long, default_value_t = yakusel_api::DEFAULT_HISTORY_CHARS)]
    history_chars: usize,
    /// Text to translate (reads stdin if omitted). Paragraphs separated by blank lines
    /// are translated one by one, with earlier ones as history (see --history-chars).
    text: Option<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the HTTPS translation server.
    #[cfg(feature = "server")]
    Serve(local::ServeArgs),
    /// Translate locally without a server.
    #[cfg(feature = "server")]
    Translate {
        #[command(flatten)]
        model: local::ModelArgs,
        #[command(flatten)]
        lang: LangArgs,
        /// Print token counts and speed to stderr.
        #[arg(long, short)]
        verbose: bool,
    },
    /// Measure prefill and decode speed.
    #[cfg(feature = "server")]
    Bench(local::BenchArgs),
    /// Translate through a remote yakusel server.
    Client {
        /// Server URL.
        #[arg(long, env = "YAKUSEL_SERVER", default_value = "https://localhost:8443")]
        server: String,
        /// Access token (default: $YAKUSEL_TOKEN, then ~/.config/yakusel/token).
        #[arg(long, env = "YAKUSEL_TOKEN", hide_env_values = true)]
        token: Option<String>,
        /// CA certificate to trust (the server's ca.pem).
        #[arg(long, env = "YAKUSEL_CA")]
        ca: Option<PathBuf>,
        /// Do not verify the server certificate.
        #[arg(long)]
        insecure: bool,
        /// Wait for each paragraph instead of streaming tokens.
        #[arg(long)]
        no_stream: bool,
        #[command(flatten)]
        lang: LangArgs,
    },
}

pub fn config_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(x).join("yakusel");
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".config/yakusel")
}

fn read_input(text: Option<String>) -> Result<String> {
    Ok(match text {
        Some(t) => t,
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        }
    })
}

fn parse_glossary(entries: &[String]) -> Result<Vec<Pair>> {
    entries
        .iter()
        .map(|e| {
            let (s, t) = e
                .split_once('=')
                .with_context(|| format!("glossary entry {e:?} must be SRC=DST"))?;
            Ok(Pair {
                source: s.trim().to_string(),
                target: t.trim().to_string(),
            })
        })
        .collect()
}

impl LangArgs {
    /// The request for one paragraph (without history and text).
    fn base_request(&self) -> Result<TranslateRequest> {
        Ok(TranslateRequest {
            text: String::new(),
            target_lang: self.to.clone(),
            source_lang: self.from.clone(),
            form: self.form.clone(),
            context: self.context.clone(),
            glossary: parse_glossary(&self.glossary)?,
            history: vec![],
            max_new_tokens: Some(self.max_new_tokens),
            stream: false,
        })
    }
}

/// Translates paragraph by paragraph, printing as it goes. `one` translates a single
/// request, calling the callback with streamed pieces, and returns the full translation.
fn translate_paragraphs(
    lang: &LangArgs,
    mut one: impl FnMut(&TranslateRequest, &mut dyn FnMut(&str)) -> Result<String>,
) -> Result<()> {
    let input = read_input(lang.text.clone())?;
    let paras = yakusel_api::split_paragraphs(&input);
    if paras.is_empty() {
        bail!("no text to translate");
    }
    let base = lang.base_request()?;
    let mut history: Vec<Pair> = Vec::new();
    let mut window = yakusel_api::HistoryWindow::new(lang.history_chars);
    let mut stdout = std::io::stdout().lock();
    for (i, p) in paras.iter().enumerate() {
        if i > 0 {
            writeln!(stdout)?;
        }
        let mut req = base.clone();
        req.text = p.clone();
        req.history = window.select(&history).to_vec();
        let out = one(&req, &mut |s| {
            let _ = stdout.write_all(s.as_bytes());
            let _ = stdout.flush();
        })?;
        writeln!(stdout)?;
        history.push(Pair {
            source: p.clone(),
            target: out,
        });
    }
    Ok(())
}

fn run_client(
    server: String,
    token: Option<String>,
    ca: Option<PathBuf>,
    insecure: bool,
    no_stream: bool,
    lang: LangArgs,
) -> Result<()> {
    let token = match token {
        Some(t) => t,
        None => std::fs::read_to_string(config_dir().join("token"))
            .map(|t| t.trim().to_string())
            .context("no token: pass --token or set YAKUSEL_TOKEN")?,
    };
    let mut builder = reqwest::blocking::Client::builder()
        .timeout(None)
        .use_rustls_tls();
    if let Some(ca) = &ca {
        let pem = std::fs::read(ca).with_context(|| format!("reading {}", ca.display()))?;
        builder = builder.add_root_certificate(reqwest::Certificate::from_pem(&pem)?);
    }
    if insecure {
        builder = builder.danger_accept_invalid_certs(true);
    }
    let http = builder.build()?;
    let url = format!("{}/v1/translate", server.trim_end_matches('/'));

    translate_paragraphs(&lang, |req, emit| {
        let mut req = req.clone();
        req.stream = !no_stream;
        let res = http.post(&url).bearer_auth(&token).json(&req).send().map_err(|e| {
            let mut msg = format!("request to {url} failed: {e}");
            let mut src: Option<&dyn std::error::Error> = std::error::Error::source(&e);
            while let Some(s) = src {
                msg += &format!(": {s}");
                src = s.source();
            }
            if msg.contains("certificate") || msg.contains("UnknownIssuer") {
                msg += "\nhint: pass the server's ca.pem with --ca (or --insecure to skip verification)";
            }
            anyhow::anyhow!(msg)
        })?;
        let status = res.status();
        if !status.is_success() {
            let body = res.text().unwrap_or_default();
            let msg = serde_json::from_str::<yakusel_api::ErrorResponse>(&body)
                .map(|e| e.error)
                .unwrap_or(body);
            bail!("server returned {status}: {msg}");
        }
        if no_stream {
            let r: TranslateResponse = res.json()?;
            emit(&r.text);
            return Ok(r.text);
        }
        let mut out = String::new();
        for line in BufReader::new(res).lines() {
            let line = line?;
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            match serde_json::from_str::<StreamEvent>(data.trim_start())? {
                StreamEvent::Delta { delta } => {
                    emit(&delta);
                    out += &delta;
                }
                StreamEvent::Done { .. } => return Ok(out),
                StreamEvent::Error { error } => bail!("server error: {error}"),
            }
        }
        bail!("stream ended unexpectedly")
    })
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        #[cfg(feature = "server")]
        Cmd::Serve(args) => local::serve(args),
        #[cfg(feature = "server")]
        Cmd::Bench(args) => local::bench(args),
        #[cfg(feature = "server")]
        Cmd::Translate {
            model,
            lang,
            verbose,
        } => local::translate(model, lang, verbose),
        Cmd::Client {
            server,
            token,
            ca,
            insecure,
            no_stream,
            lang,
        } => run_client(server, token, ca, insecure, no_stream, lang),
    }
}
