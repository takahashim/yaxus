use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use yaxus_api::{DEFAULT_MAX_NEW_TOKENS, DocumentEvent, DocumentRequest, DocumentResponse, Pair};

#[cfg(feature = "server")]
mod local;

#[derive(Parser)]
#[command(name = "yaxus", version, about = "PLaMo 3 Translate in pure Rust")]
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
    /// History budget per chain in characters: earlier paragraphs of the chain and their
    /// translations are sent with each paragraph until this is exceeded. 0 = none.
    #[arg(long, default_value_t = yaxus_api::DEFAULT_HISTORY_CHARS)]
    history_chars: usize,
    /// Chains of consecutive paragraphs translated in parallel (0: one per 4 paragraphs,
    /// at most 8; 1: strictly in order, the most consistent).
    #[arg(long, default_value_t = 0)]
    chains: usize,
    /// Pass the neighbouring source paragraphs as context (useful with --history-chars 0).
    #[arg(long)]
    neighbor_context: bool,
    /// Text to translate (reads stdin if omitted). Paragraphs are separated by blank lines.
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
    /// Translate through a remote yaxus server.
    Client {
        /// Server URL.
        #[arg(long, env = "YAXUS_SERVER", default_value = "https://localhost:8443")]
        server: String,
        /// Access token (default: $YAXUS_TOKEN, then ~/.config/yaxus/token).
        #[arg(long, env = "YAXUS_TOKEN", hide_env_values = true)]
        token: Option<String>,
        /// CA certificate to trust (the server's ca.pem).
        #[arg(long, env = "YAXUS_CA")]
        ca: Option<PathBuf>,
        /// Do not verify the server certificate.
        #[arg(long)]
        insecure: bool,
        /// Wait for the whole result instead of streaming tokens.
        #[arg(long)]
        no_stream: bool,
        #[command(flatten)]
        lang: LangArgs,
    },
}

pub fn config_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(x).join("yaxus");
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".config/yaxus")
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
    /// The document request for the input text.
    fn document(&self, stream: bool) -> Result<DocumentRequest> {
        let input = read_input(self.text.clone())?;
        let segments = yaxus_api::split_paragraphs(&input);
        if segments.is_empty() {
            bail!("no text to translate");
        }
        Ok(DocumentRequest {
            segments,
            target_lang: self.to.clone(),
            source_lang: self.from.clone(),
            form: self.form.clone(),
            context: self.context.clone(),
            glossary: parse_glossary(&self.glossary)?,
            chains: Some(self.chains),
            history_chars: Some(self.history_chars),
            neighbor_context: self.neighbor_context,
            max_new_tokens: Some(self.max_new_tokens),
            stream,
        })
    }
}

/// Prints segments in order while they are produced in parallel: the first unfinished
/// segment streams live, later ones are buffered until their turn.
pub struct OrderedPrinter<W: Write> {
    out: W,
    cursor: usize,
    buf: Vec<String>,
    ended: Vec<bool>,
}

impl<W: Write> OrderedPrinter<W> {
    pub fn new(out: W, segments: usize) -> Self {
        OrderedPrinter { out, cursor: 0, buf: vec![String::new(); segments], ended: vec![false; segments] }
    }

    pub fn delta(&mut self, segment: usize, text: &str) -> Result<()> {
        if segment == self.cursor {
            self.out.write_all(text.as_bytes())?;
            self.out.flush()?;
        } else if segment < self.buf.len() {
            self.buf[segment].push_str(text);
        }
        Ok(())
    }

    pub fn end(&mut self, segment: usize) -> Result<()> {
        if segment >= self.ended.len() {
            return Ok(());
        }
        self.ended[segment] = true;
        while self.cursor < self.ended.len() && self.ended[self.cursor] {
            writeln!(self.out)?;
            self.cursor += 1;
            if self.cursor < self.buf.len() {
                writeln!(self.out)?;
                let b = std::mem::take(&mut self.buf[self.cursor]);
                self.out.write_all(b.as_bytes())?;
            }
        }
        self.out.flush()?;
        Ok(())
    }
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
            .context("no token: pass --token or set YAXUS_TOKEN")?,
    };
    let mut builder = reqwest::blocking::Client::builder().timeout(None).use_rustls_tls();
    if let Some(ca) = &ca {
        let pem = std::fs::read(ca).with_context(|| format!("reading {}", ca.display()))?;
        builder = builder.add_root_certificate(reqwest::Certificate::from_pem(&pem)?);
    }
    if insecure {
        builder = builder.danger_accept_invalid_certs(true);
    }
    let http = builder.build()?;
    let url = format!("{}/v1/translate/document", server.trim_end_matches('/'));
    let req = lang.document(!no_stream)?;
    let n = req.segments.len();
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
        let msg = serde_json::from_str::<yaxus_api::ErrorResponse>(&body).map(|e| e.error).unwrap_or(body);
        bail!("server returned {status}: {msg}");
    }
    let mut out = OrderedPrinter::new(std::io::stdout().lock(), n);
    if no_stream {
        let r: DocumentResponse = res.json()?;
        for (i, s) in r.segments.iter().enumerate() {
            out.delta(i, s)?;
            out.end(i)?;
        }
        return Ok(());
    }
    for line in BufReader::new(res).lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data:") else { continue };
        match serde_json::from_str::<DocumentEvent>(data.trim_start())? {
            DocumentEvent::Delta { segment, delta } => out.delta(segment, &delta)?,
            DocumentEvent::SegmentEnd { segment, .. } => out.end(segment)?,
            DocumentEvent::Done { .. } => return Ok(()),
            DocumentEvent::Error { error } => bail!("server error: {error}"),
        }
    }
    bail!("stream ended unexpectedly")
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
