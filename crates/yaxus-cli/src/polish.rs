//! `yaxus polish`: proofreading help for Japanese. Each paragraph is translated there and
//! back through another language (English by default; the pivot can also come from a
//! command, a general LLM that decides what is said) or directly (ja to ja), and what
//! changed is shown for a person to judge. The rewrite is a suggestion: a round trip can drift in meaning or register, so
//! the politeness of the source is carried over (`--form auto`) and the words a rewrite
//! must keep (numbers, Latin words, URLs, quotations) are checked. `yaxus polish mark`
//! rewrites nothing: it marks where the model would have written otherwise.

use std::io::Write;
use std::str::FromStr;

use anyhow::{Result, bail};
use clap::{Args, ValueEnum};
use yaxus_api::{DocumentRequest, Form};
pub use yaxus_runtime::score::Choice;

/// The language of the text: `polish` is for Japanese.
pub(crate) const LANG: &str = "ja";

/// How a paragraph gets back to Japanese.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Via {
    /// Translated into Japanese at once.
    Direct,
    /// Translated into this language and back.
    Lang(String),
}

impl FromStr for Via {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "" => Err("an empty pivot".into()),
            "direct" => Ok(Via::Direct),
            LANG => Err(format!("{LANG} is the text's own language: use direct")),
            l => {
                // (checked here rather than after the model has loaded)
                let req = yaxus_pfml::TranslationRequest { target_lang: l.to_string(), ..Default::default() };
                match req.validate_options() {
                    Ok(()) => Ok(Via::Lang(l.to_string())),
                    Err(_) => Err(format!("{l:?} is not a language code (e.g. en, zh-hans) or direct")),
                }
            }
        }
    }
}

impl std::fmt::Display for Via {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Via::Direct => f.write_str("direct"),
            Via::Lang(l) => f.write_str(l),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FormChoice {
    /// The source's politeness, from its sentence endings (none if it has no clear one).
    Auto,
    Polite,
    Plain,
    /// Let the model choose.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// The source with the changes marked: [-removed-]{+added+}.
    Diff,
    /// The rewritten text alone.
    Text,
    /// Every paragraph's candidates, pivots, changes and checks.
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MarkFormat {
    /// The source with the marks: [?piece→the model's choice?].
    Diff,
    /// Every paragraph's marks, and each piece's probability and the model's choice.
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ColorChoice {
    Auto,
    Always,
    Never,
}

/// The options `polish` and `polish mark` share.
#[derive(Args)]
#[command(next_help_heading = "Polish")]
pub struct Shared {
    /// Politeness of the rewrite.
    #[arg(long, value_enum, default_value_t = FormChoice::Auto)]
    form: FormChoice,
    /// Also print the text in the pivot language (diff output; JSON always has it).
    #[arg(long)]
    show_pivot: bool,
    /// Colour the output (auto: on a terminal, unless NO_COLOR is set).
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto)]
    color: ColorChoice,
    /// Most tokens generated per paragraph (`mark --via direct` generates nothing).
    #[arg(long, default_value_t = yaxus_api::DEFAULT_MAX_NEW_TOKENS, help_heading = "Translation")]
    max_new_tokens: usize,
    /// History budget per chain in characters (see `translate --history-chars`; the
    /// scores of `mark` take none).
    #[arg(long, default_value_t = yaxus_api::DEFAULT_HISTORY_CHARS, help_heading = "Translation")]
    history_chars: usize,
    /// Chains of consecutive paragraphs run in parallel (see `translate --chains`).
    #[arg(long, default_value_t = 0, help_heading = "Translation")]
    chains: usize,
    /// Japanese text (reads stdin if omitted). Paragraphs are separated by blank lines.
    text: Option<String>,
}

/// `yaxus polish`.
#[derive(Args)]
#[command(next_help_heading = "Polish")]
pub struct RewriteArgs {
    /// How each paragraph is rewritten: a language code to go there and back (en, the
    /// default; zh-hans, ...: the model writes the Japanese anew, in its own words) or
    /// `direct` (into Japanese at once: the closest to the source, it fixes errors but
    /// keeps the wording). Several, comma-separated, give one candidate each.
    #[arg(long, value_delimiter = ',', default_value = "en")]
    via: Vec<Via>,
    /// Make the pivot with this shell command instead of the model: it reads the Japanese
    /// paragraphs on stdin and writes as many in the --via language on stdout, separated
    /// by blank lines; the model writes the Japanese from them. With a general LLM, it
    /// decides what is said (no emphasis, stock phrases or needless summaries) and the
    /// model how it is said. The text goes wherever the command sends it.
    #[arg(long, value_name = "CMD")]
    pivot_cmd: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Diff)]
    format: OutputFormat,
    #[command(flatten)]
    shared: Shared,
}

/// `yaxus polish mark`.
#[derive(Args)]
#[command(next_help_heading = "Polish")]
pub struct MarkArgs {
    /// What each paragraph is scored as: `direct`, the model's translation of itself (one
    /// pass, no generation; finds errors), or a language code, the translation of the
    /// paragraph in that language (en: the content is fixed, so only the wording counts),
    /// with each paragraph's log-probability per character next to the model's own
    /// rewrite. A high probability means typical wording, not necessarily human wording.
    #[arg(long, default_value = "direct")]
    via: Via,
    /// Mark pieces the model gives less than this probability (the best F1 of the 2B and
    /// the 8B on `yaxus bench polish`, with --via direct).
    #[arg(long, default_value_t = 0.05)]
    threshold: f32,
    #[arg(long, value_enum, default_value_t = MarkFormat::Diff)]
    format: MarkFormat,
    #[command(flatten)]
    shared: Shared,
}

/// A piece of a scored paragraph (see `yaxus_runtime::score::Piece`).
pub struct Scored {
    /// Byte range in the paragraph.
    pub start: usize,
    pub end: usize,
    pub logprob: f32,
    /// The model's own choice where the piece starts.
    pub best: Choice,
    pub best_logprob: f32,
}

/// What is scored: `output` as the translation of `input` from `from` into `to`.
pub struct ScoreRequest<'a> {
    pub input: &'a str,
    pub from: &'a str,
    pub output: &'a str,
    pub to: &'a str,
    pub form: Option<Form>,
}

/// The model `polish` runs on.
pub trait Polisher {
    /// Translates a document: `stage` names the step (`ja→en`) for progress messages.
    fn translate(&mut self, stage: &str, req: DocumentRequest) -> Result<Vec<String>>;
    /// The pieces of `r.output` with their log-probabilities (see `Generator::score`).
    fn score(&mut self, r: &ScoreRequest) -> Result<Vec<Scored>>;
}

impl RewriteArgs {
    /// The options this rewrite was given that differ from their defaults: before `mark`
    /// they would be the rewrite's, and not apply.
    pub fn given(&self) -> Vec<&'static str> {
        let d = &self.shared;
        [
            ("--via", self.via != [Via::Lang("en".into())]),
            ("--pivot-cmd", self.pivot_cmd.is_some()),
            ("--format", self.format != OutputFormat::Diff),
            ("--form", d.form != FormChoice::Auto),
            ("--show-pivot", d.show_pivot),
            ("--color", d.color != ColorChoice::Auto),
            ("--max-new-tokens", d.max_new_tokens != yaxus_api::DEFAULT_MAX_NEW_TOKENS),
            ("--history-chars", d.history_chars != yaxus_api::DEFAULT_HISTORY_CHARS),
            ("--chains", d.chains != 0),
            ("TEXT", d.text.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect()
    }
}

impl MarkArgs {
    pub fn via(&self) -> &Via {
        &self.via
    }
}

/// The options of the translations a polish run makes.
pub(crate) struct Opts {
    pub max_new_tokens: usize,
    pub history_chars: usize,
    pub chains: usize,
}

impl Shared {
    fn opts(&self) -> Opts {
        Opts { max_new_tokens: self.max_new_tokens, history_chars: self.history_chars, chains: self.chains }
    }

    /// The paragraphs of the input.
    fn segments(&self) -> Result<Vec<String>> {
        let segments = yaxus_api::split_paragraphs(&crate::read_input(self.text.clone())?);
        if segments.is_empty() {
            bail!("no text to polish");
        }
        Ok(segments)
    }
}

fn request(o: &Opts, segments: Vec<String>, from: &str, to: &str, form: Option<Form>) -> DocumentRequest {
    DocumentRequest {
        segments,
        target_lang: to.to_string(),
        source_lang: Some(from.to_string()),
        form,
        context: None,
        glossary: Vec::new(),
        chains: std::num::NonZeroUsize::new(o.chains),
        history_chars: Some(o.history_chars),
        neighbor_context: false,
        max_new_tokens: Some(o.max_new_tokens),
        stream: false,
    }
}

/// `texts` from `from` into the text's language: one request per form, each with its
/// paragraphs in order.
fn home(
    p: &mut dyn Polisher,
    o: &Opts,
    stage: &str,
    from: &str,
    texts: &[String],
    forms: &[Option<Form>],
) -> Result<Vec<String>> {
    let mut out = vec![String::new(); texts.len()];
    let mut groups: Vec<Option<Form>> = Vec::new();
    for f in forms {
        if !groups.contains(f) {
            groups.push(*f);
        }
    }
    for g in groups {
        let idx: Vec<usize> = (0..texts.len()).filter(|&i| forms[i] == g).collect();
        let part = idx.iter().map(|&i| texts[i].clone()).collect();
        let label = match g {
            Some(f) if forms.iter().any(|x| *x != Some(f)) => format!("{stage}, {}", f.as_str()),
            _ => stage.to_string(),
        };
        let got = p.translate(&label, request(o, part, from, LANG, g))?;
        if got.len() != idx.len() {
            bail!("{} paragraphs came back for {}", got.len(), idx.len());
        }
        for (i, t) in idx.into_iter().zip(got) {
            out[i] = t;
        }
    }
    Ok(out)
}

/// The rewrite of `segments` through `via`.
/// The rewrite of `segments` through `via`, the pivot made by the model or by `pivot_cmd`.
fn route(
    p: &mut dyn Polisher,
    o: &Opts,
    via: &Via,
    segments: &[String],
    forms: &[Option<Form>],
    pivot_cmd: Option<&str>,
) -> Result<Candidate> {
    let lang = LANG;
    let c = match via {
        Via::Direct => {
            let output = home(p, o, &format!("{lang}→{lang}"), lang, segments, forms)?;
            Candidate { via: via.clone(), pivot: None, output }
        }
        Via::Lang(v) => {
            let there = match pivot_cmd {
                Some(cmd) => pivot_by_command(cmd, segments)?,
                None => p.translate(&format!("{lang}→{v}"), request(o, segments.to_vec(), lang, v, None))?,
            };
            if there.len() != segments.len() {
                bail!("{} paragraphs came back for {}", there.len(), segments.len());
            }
            let output = home(p, o, &format!("{v}→{lang}"), v, &there, forms)?;
            Candidate { via: via.clone(), pivot: Some(there), output }
        }
    };
    if c.output.len() != segments.len() {
        bail!("{} paragraphs came back for {}", c.output.len(), segments.len());
    }
    Ok(c)
}

/// The pivot of `segments` from the shell command `cmd`: the paragraphs on its stdin, as
/// many back on its stdout, both separated by blank lines.
fn pivot_by_command(cmd: &str, segments: &[String]) -> Result<Vec<String>> {
    use std::process::{Command, Stdio};
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("--pivot-cmd: cannot run sh: {e}"))?;
    let input = segments.join("\n\n") + "\n";
    let mut stdin = child.stdin.take().expect("piped stdin");
    // (written apart from reading: a command that answers as it reads would block)
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let out = child.wait_with_output()?;
    // A command that exits without reading all of its input closes the pipe: its status
    // says whether that went wrong.
    let written = writer.join().expect("the writer thread");
    if !out.status.success() {
        bail!("--pivot-cmd failed ({})", out.status);
    }
    // (a command that succeeded without reading all of it is answered as it is)
    if let Err(e) = written
        && e.kind() != std::io::ErrorKind::BrokenPipe
    {
        bail!("--pivot-cmd: writing its input: {e}");
    }
    let text = String::from_utf8(out.stdout).map_err(|_| anyhow::anyhow!("--pivot-cmd wrote invalid UTF-8"))?;
    let pivot = yaxus_api::split_paragraphs(&text);
    if pivot.len() != segments.len() {
        bail!(
            "--pivot-cmd wrote {} paragraphs for {}: it must keep the paragraphs, separated by blank lines",
            pivot.len(),
            segments.len()
        );
    }
    Ok(pivot)
}

/// How likely the model finds a paragraph as its rewrite: how typical its wording is.
pub(crate) struct Natural {
    pub pieces: Vec<Scored>,
    /// Log-probability per character; none when the paragraph could not be scored (its
    /// pivot came back empty).
    pub per_char: Option<f64>,
    /// Through a pivot: the pivot, and the model's own rewrite with its log-probability per
    /// character (the most it would give a text of that content; none if it came back
    /// empty).
    pub pivot: Option<String>,
    pub own: Option<(String, f64)>,
}

pub(crate) fn per_char(text: &str, pieces: &[Scored]) -> f64 {
    pieces.iter().map(|p| p.logprob as f64).sum::<f64>() / text.chars().count().max(1) as f64
}

/// Each paragraph scored as the model's rewrite of it: directly (as the translation of
/// itself), or through `via` as the translation of its pivot, the content in another
/// language, so that only the wording counts.
pub(crate) fn naturalness(
    p: &mut dyn Polisher,
    o: &Opts,
    via: &Via,
    segments: &[String],
    forms: &[Option<Form>],
) -> Result<Vec<Natural>> {
    let lang = LANG;
    let routed = match via {
        Via::Direct => None,
        Via::Lang(_) => Some(route(p, o, via, segments, forms, None)?),
    };
    let mut out = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let (input, from) = match (&routed, via) {
            (Some(c), Via::Lang(v)) => (c.pivot.as_ref().unwrap()[i].as_str(), v.as_str()),
            _ => (seg.as_str(), lang),
        };
        let pivot = routed.as_ref().map(|c| c.pivot.as_ref().unwrap()[i].clone());
        // An empty pivot is no request (and a paragraph of the input is never empty).
        if input.trim().is_empty() {
            out.push(Natural { pieces: Vec::new(), per_char: None, pivot, own: None });
            continue;
        }
        let r = |output| ScoreRequest { input, from, output, to: lang, form: forms[i] };
        let pieces = p.score(&r(seg.as_str()))?;
        let own = match &routed {
            // (an empty rewrite would score 0, the most there is)
            Some(c) if !c.output[i].trim().is_empty() => {
                let o = &c.output[i];
                let lp = per_char(o, &p.score(&r(o.as_str()))?);
                Some((o.clone(), lp))
            }
            _ => None,
        };
        let per = per_char(seg, &pieces);
        out.push(Natural { pieces, per_char: Some(per), pivot, own });
    }
    Ok(out)
}

/// `polish mark`: each paragraph with the pieces the model finds unlikely marked; through a
/// pivot, also how natural the paragraph is next to the model's own rewrite.
pub fn mark(a: &MarkArgs, p: &mut dyn Polisher) -> Result<()> {
    anyhow::ensure!(a.threshold > 0.0 && a.threshold < 1.0, "--threshold must be between 0 and 1");
    let segments = a.shared.segments()?;
    let via = &a.via;
    let forms = forms(a.shared.form, &segments);
    let limit = a.threshold.ln();
    let color = use_color(a.shared.color);
    let paint = |code: &str, s: String| if color { format!("\x1b[{code}m{s}\x1b[0m") } else { s };
    let nat = naturalness(p, &a.shared.opts(), via, &segments, &forms)?;
    let mut out = std::io::stdout().lock();
    let mut json = Vec::new();
    let (mut places, mut marked) = (0usize, 0usize);
    for (i, (seg, n)) in segments.iter().zip(&nat).enumerate() {
        let pieces = &n.pieces;
        let spans = unlikely(seg, pieces, limit);
        places += spans.len();
        marked += usize::from(!spans.is_empty());
        if a.format == MarkFormat::Json {
            let ps: Vec<_> = pieces
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "text": &seg[p.start..p.end],
                        "prob": p.logprob.exp(),
                        "best": p.best.text(),
                        "best_ends": p.best == Choice::End,
                        "best_prob": p.best_logprob.exp(),
                    })
                })
                .collect();
            let ms: Vec<_> = spans
                .iter()
                .map(|&(s, e, k)| {
                    serde_json::json!({"start": s, "end": e, "text": &seg[s..e], "best": pieces[k].best.text(),
                        "best_ends": pieces[k].best == Choice::End, "prob": pieces[k].logprob.exp()})
                })
                .collect();
            json.push(serde_json::json!({
                "source": seg,
                "form": forms[i].map(Form::as_str),
                "via": via.to_string(),
                "pivot": n.pivot,
                "logprob_per_char": n.per_char,
                "own": n.own.as_ref().map(|(t, lp)| serde_json::json!({"text": t, "logprob_per_char": lp})),
                "marks": ms,
                "pieces": ps,
            }));
            continue;
        }
        if i > 0 {
            writeln!(out)?;
        }
        let mut line = String::new();
        let mut at = 0;
        for &(s, e, k) in &spans {
            line += &seg[at..s];
            line += &paint("33;1", format!("[?{}→{}?]", &seg[s..e], pieces[k].best));
            at = e;
        }
        line += &seg[at..];
        writeln!(out, "{line}")?;
        match (n.per_char, &n.own) {
            (None, _) => writeln!(out, "{}", paint("33", format!("  ! not scored: the {via} came back empty")))?,
            (Some(lp), Some((_, own))) => {
                let note = format!("  {lp:.3} nats/char; the model's own rewrite {own:.3} (gap {:.3})", own - lp);
                writeln!(out, "{}", paint("2", note))?;
            }
            (Some(lp), None) if n.pivot.is_some() => {
                let note = format!("  {lp:.3} nats/char; the model's own rewrite came back empty");
                writeln!(out, "{}", paint("2", note))?;
            }
            _ => {}
        }
        if a.shared.show_pivot
            && let Some(pv) = &n.pivot
        {
            writeln!(out, "{}", paint("2", format!("  via {via}: {}", pv.replace('\n', " "))))?;
        }
    }
    if a.format == MarkFormat::Json {
        writeln!(out, "{}", serde_json::to_string_pretty(&serde_json::json!({ "paragraphs": json }))?)?;
    }
    out.flush()?;
    eprintln!(
        "[mark via {via}: {places} places in {marked} of {} paragraphs, below p={}]",
        segments.len(),
        a.threshold
    );
    Ok(())
}

/// Runs of consecutive pieces below `limit` (a log-probability) where the model would
/// have written otherwise, as (start, end, first piece).
pub(crate) fn unlikely(text: &str, pieces: &[Scored], limit: f32) -> Vec<(usize, usize, usize)> {
    spans_of(pieces, |p| p.logprob < limit && !same_text(text, p))
}

/// Whether the model's choice is a longer token that the text goes on with: the text
/// was split otherwise (`な` `る` for its `なる`), but the model would write the same.
fn same_text(text: &str, p: &Scored) -> bool {
    p.best.text().is_some_and(|b| b.len() > p.end - p.start && text[p.start..].starts_with(b))
}

/// Runs of consecutive pieces that `flag` marks, as (start, end, first piece).
fn spans_of(pieces: &[Scored], flag: impl Fn(&Scored) -> bool) -> Vec<(usize, usize, usize)> {
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    for (k, p) in pieces.iter().enumerate() {
        if !flag(p) {
            continue;
        }
        match spans.last_mut() {
            // (a span ends where its last piece does: the previous piece was marked)
            Some(last) if last.1 == p.start => last.1 = p.end,
            _ => spans.push((p.start, p.end, k)),
        }
    }
    spans
}

fn use_color(c: ColorChoice) -> bool {
    match c {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => {
            std::io::IsTerminal::is_terminal(&std::io::stdout()) && std::env::var_os("NO_COLOR").is_none()
        }
    }
}

/// One way of rewriting the whole document.
struct Candidate {
    via: Via,
    /// The text in the pivot language, paragraph by paragraph.
    pivot: Option<Vec<String>>,
    output: Vec<String>,
}

/// What a rewrite of one paragraph did.
struct Review {
    diff: Vec<(Op, String)>,
    /// Protected words of the source the rewrite lost.
    missing: Vec<String>,
    /// The rewrite came back empty (the source is kept).
    empty: bool,
}

impl Review {
    fn of(source: &str, output: &str) -> Self {
        let empty = !source.trim().is_empty() && output.trim().is_empty();
        let output = if empty { source } else { output };
        Review { diff: diff(source, output), missing: missing(source, output), empty }
    }

    fn changed(&self) -> bool {
        self.diff.iter().any(|(op, _)| *op != Op::Equal)
    }

    fn to_check(&self) -> bool {
        self.empty || !self.missing.is_empty()
    }
}

pub fn run(a: &RewriteArgs, p: &mut dyn Polisher) -> Result<()> {
    if a.via.is_empty() {
        bail!("--via needs at least one of direct or a language code");
    }
    let segments = a.shared.segments()?;
    let forms = forms(a.shared.form, &segments);
    let o = a.shared.opts();
    let mut candidates = Vec::new();
    let pivots = a.via.iter().filter(|v| **v != Via::Direct).count();
    if a.pivot_cmd.is_some() && pivots != 1 {
        bail!("--pivot-cmd makes one pivot: give --via one language (and direct, if wanted)");
    }
    for via in &a.via {
        candidates.push(route(p, &o, via, &segments, &forms, a.pivot_cmd.as_deref())?);
    }
    let reviews: Vec<Vec<Review>> =
        candidates.iter().map(|c| segments.iter().zip(&c.output).map(|(s, o)| Review::of(s, o)).collect()).collect();
    let color = use_color(a.shared.color);
    let mut out = std::io::stdout().lock();
    match a.format {
        OutputFormat::Text => print_text(&mut out, &segments, &candidates, &reviews)?,
        OutputFormat::Diff => print_diff(&mut out, &segments, &candidates, &reviews, a.shared.show_pivot, color)?,
        OutputFormat::Json => print_json(&mut out, &segments, &forms, &candidates, &reviews)?,
    }
    out.flush()?;
    let count = |f: Form| forms.iter().filter(|x| **x == Some(f)).count();
    let form_note = match (count(Form::Polite), count(Form::Plain)) {
        (0, 0) => String::new(),
        (p, 0) if p == segments.len() => "; polite".into(),
        (0, p) if p == segments.len() => "; plain".into(),
        (p, q) => format!("; {p} polite, {q} plain"),
    };
    for (c, r) in candidates.iter().zip(&reviews) {
        let changed = r.iter().filter(|r| r.changed()).count();
        let check = r.iter().filter(|r| r.to_check()).count();
        eprintln!("[{}: {changed} of {} paragraphs changed, {check} to check{form_note}]", c.via, segments.len());
    }
    Ok(())
}

/// Phrases common in machine-written Japanese prose: a rough measure of it, by count.
const STOCK_PHRASES: [&str; 30] = [
    "と言えるだろう",
    "と言えるでしょう",
    "と言える",
    "と考えられる",
    "と考えられます",
    "さまざまな",
    "様々な",
    "において",
    "することが重要",
    "が重要である",
    "が重要です",
    "不可欠",
    "多角的",
    "観点から",
    "を踏まえ",
    "しかしながら",
    "結論として",
    "といった",
    "することができる",
    "することが可能",
    "ポイント",
    "に留意",
    "が求められ",
    "と言っても過言ではない",
    "を通じて",
    "に寄与",
    "メカニズム",
    "側面",
    "非常に",
    "その結果として",
];

/// How many stock phrases `text` has (each occurrence; a phrase inside a longer one, as
/// と言える in と言えるだろう, is not counted again).
pub(crate) fn stock_phrases(text: &str) -> usize {
    let mut taken = vec![false; text.len()];
    let mut by_len: Vec<&str> = STOCK_PHRASES.to_vec();
    by_len.sort_by_key(|p| std::cmp::Reverse(p.len()));
    let mut n = 0;
    for p in by_len {
        for (i, _) in text.match_indices(p) {
            if !taken[i..i + p.len()].iter().any(|&t| t) {
                taken[i..i + p.len()].iter_mut().for_each(|t| *t = true);
                n += 1;
            }
        }
    }
    n
}

/// The form each paragraph is rewritten in. `auto` follows each paragraph's own endings,
/// and the document's for a paragraph without sentences (a heading, a list).
pub(crate) fn forms(choice: FormChoice, segments: &[String]) -> Vec<Option<Form>> {
    let all = |f| vec![f; segments.len()];
    match choice {
        FormChoice::Polite => all(Some(Form::Polite)),
        FormChoice::Plain => all(Some(Form::Plain)),
        FormChoice::None => all(None),
        FormChoice::Auto => {
            let doc = detect_form(segments);
            segments.iter().map(|s| detect_form(std::slice::from_ref(s)).or(doc)).collect()
        }
    }
}

fn print_text(out: &mut impl Write, segments: &[String], cands: &[Candidate], reviews: &[Vec<Review>]) -> Result<()> {
    for (k, (c, r)) in cands.iter().zip(reviews).enumerate() {
        if cands.len() > 1 {
            if k > 0 {
                writeln!(out)?;
            }
            writeln!(out, "# via {}\n", c.via)?;
        }
        for (i, (o, r)) in c.output.iter().zip(r).enumerate() {
            if i > 0 {
                writeln!(out)?;
            }
            writeln!(out, "{}", if r.empty { &segments[i] } else { o })?;
        }
    }
    Ok(())
}

fn print_diff(
    out: &mut impl Write,
    segments: &[String],
    cands: &[Candidate],
    reviews: &[Vec<Review>],
    show_pivot: bool,
    color: bool,
) -> Result<()> {
    let paint = |code: &str, s: String| if color { format!("\x1b[{code}m{s}\x1b[0m") } else { s };
    for i in 0..segments.len() {
        if i > 0 {
            writeln!(out)?;
        }
        for (c, r) in cands.iter().zip(reviews) {
            let r = &r[i];
            let mut line = String::new();
            if cands.len() > 1 {
                line += &paint("1", format!("[{}] ", c.via));
            }
            for (op, text) in &r.diff {
                line += &match op {
                    Op::Equal => text.clone(),
                    Op::Delete => paint("31", format!("[-{text}-]")),
                    Op::Insert => paint("32", format!("{{+{text}+}}")),
                };
            }
            writeln!(out, "{line}")?;
            if r.empty {
                writeln!(out, "{}", paint("33", "  ! the rewrite came back empty; the source is kept".into()))?;
            }
            if !r.missing.is_empty() {
                writeln!(out, "{}", paint("33", format!("  ! check: lost {}", r.missing.join(", "))))?;
            }
            if show_pivot && let Some(p) = &c.pivot {
                writeln!(out, "{}", paint("2", format!("  via {}: {}", c.via, p[i].replace('\n', " "))))?;
            }
        }
    }
    Ok(())
}

fn print_json(
    out: &mut impl Write,
    segments: &[String],
    forms: &[Option<Form>],
    cands: &[Candidate],
    reviews: &[Vec<Review>],
) -> Result<()> {
    let paragraphs: Vec<_> = segments
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let candidates: Vec<_> = cands
                .iter()
                .zip(reviews)
                .map(|(c, r)| {
                    let r = &r[i];
                    let diff: Vec<_> =
                        r.diff.iter().map(|(op, t)| serde_json::json!({"op": op.as_str(), "text": t})).collect();
                    serde_json::json!({
                        "via": c.via.to_string(),
                        "pivot": c.pivot.as_ref().map(|p| &p[i]),
                        "output": if r.empty { s } else { &c.output[i] },
                        "changed": r.changed(),
                        "empty": r.empty,
                        "missing": r.missing,
                        "diff": diff,
                    })
                })
                .collect();
            serde_json::json!({"source": s, "form": forms[i].map(Form::as_str), "candidates": candidates})
        })
        .collect();
    let doc = serde_json::json!({ "paragraphs": paragraphs });
    writeln!(out, "{}", serde_json::to_string_pretty(&doc)?)?;
    Ok(())
}

/// The politeness of Japanese text from its sentence endings: the more common of
/// polite (です, ます, ...) and plain, or none when there are no sentences or as many of
/// each. Sentences end in 。！？; a line that ends without one counts only when polite
/// (`向上します✨`: a heading such as `概要` is no plain sentence). Headings (`#`) and list
/// items are left out, and so are emoji and other symbols after an ending.
pub fn detect_form(segments: &[String]) -> Option<Form> {
    const POLITE: [&str; 9] =
        ["です", "ます", "でした", "ました", "ません", "ください", "でしょう", "ましょう", "ませ"];
    // Polite, plain, or no sentence (no Japanese).
    let judge = |sentence: &str| -> Option<bool> {
        let s = sentence.trim_end_matches(|c: char| !is_japanese(c) && !c.is_alphanumeric());
        if !s.chars().any(is_japanese) {
            return None;
        }
        // (final particles: ですか, ますね, ですよね)
        let s = s.trim_end_matches(['か', 'ね', 'よ', 'な']);
        Some(POLITE.iter().any(|e| s.ends_with(e)))
    };
    let (mut polite, mut plain) = (0usize, 0usize);
    for line in segments.iter().flat_map(|s| s.lines()) {
        let line = line.trim();
        if line.starts_with('#') || is_list_item(line) {
            continue;
        }
        let mut sentence = String::new();
        for c in line.chars() {
            if matches!(c, '。' | '！' | '？' | '!' | '?') {
                match judge(&sentence) {
                    Some(true) => polite += 1,
                    Some(false) => plain += 1,
                    None => {}
                }
                sentence.clear();
            } else {
                sentence.push(c);
            }
        }
        if judge(&sentence) == Some(true) {
            polite += 1;
        }
    }
    match polite.cmp(&plain) {
        _ if polite + plain == 0 => None,
        std::cmp::Ordering::Greater => Some(Form::Polite),
        std::cmp::Ordering::Less => Some(Form::Plain),
        std::cmp::Ordering::Equal => None,
    }
}

/// A Markdown list item: `- `, `* `, `+ `, `・`, `1. `, `1) `.
fn is_list_item(line: &str) -> bool {
    if line.starts_with("- ") || line.starts_with("* ") || line.starts_with("+ ") || line.starts_with('・') {
        return true;
    }
    let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
    digits > 0 && (line[digits..].starts_with(". ") || line[digits..].starts_with(") "))
}

fn is_japanese(c: char) -> bool {
    matches!(c, '\u{3040}'..='\u{30ff}' | '\u{4e00}'..='\u{9fff}' | '\u{3400}'..='\u{4dbf}')
}

/// Full-width ASCII and the ideographic space as ASCII, so that `２０２６` and `2026`
/// are the same word.
fn normalize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{ff01}'..='\u{ff5e}' => char::from_u32(c as u32 - 0xfee0).unwrap_or(c),
            '\u{3000}' => ' ',
            c => c,
        })
        .collect()
}

/// Words a rewrite must not lose: numbers and Latin words (with `.`, `_`, `-`, `/`, `:`
/// inside, so URLs and versions stay whole), and the contents of 「」 and 『』.
fn protected(s: &str) -> (Vec<String>, Vec<String>) {
    let mut words = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, words: &mut Vec<String>| {
        let w = cur.trim_end_matches(['.', '-', '/', ':', '_', ',']).to_string();
        if w.chars().any(|c| c.is_ascii_digit()) || w.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 2 {
            words.push(w);
        }
        cur.clear();
    };
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || (!cur.is_empty() && matches!(c, '.' | '_' | '-' | '/' | ':' | ',' | '%')) {
            cur.push(c);
        } else {
            flush(&mut cur, &mut words);
        }
    }
    flush(&mut cur, &mut words);
    let mut quotes = Vec::new();
    for (open, close) in [('「', '」'), ('『', '』')] {
        let mut rest = s;
        while let Some(a) = rest.find(open) {
            let after = &rest[a + open.len_utf8()..];
            let Some(b) = after.find(close) else { break };
            let q = &after[..b];
            if !q.trim().is_empty() && q.chars().count() <= 40 {
                quotes.push(q.to_string());
            }
            rest = &after[b + close.len_utf8()..];
        }
    }
    (words, quotes)
}

/// The protected words of `source` that `output` has fewer of, each once.
fn missing(source: &str, output: &str) -> Vec<String> {
    let (source, output) = (normalize(source), normalize(output));
    let (words, quotes) = protected(&source);
    let (out_words, _) = protected(&output);
    let mut lost = Vec::new();
    let mut add = |w: &String| {
        if !lost.contains(w) {
            lost.push(w.clone());
        }
    };
    for w in &words {
        let n = |ws: &[String]| ws.iter().filter(|x| *x == w).count();
        if n(&out_words) < n(&words) {
            add(w);
        }
    }
    for q in &quotes {
        let n = source.matches(q.as_str()).count();
        if n > 0 && output.matches(q.as_str()).count() < n {
            add(q);
        }
    }
    lost
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Equal,
    Delete,
    Insert,
}

impl Op {
    fn as_str(self) -> &'static str {
        match self {
            Op::Equal => "equal",
            Op::Delete => "delete",
            Op::Insert => "insert",
        }
    }
}

/// Characters past which the middle of two texts is replaced whole instead of diffed
/// (the table is the product of their lengths).
const DIFF_CELLS: usize = 4_000_000;

/// A character diff of `a` into `b`: runs of equal, deleted and inserted text. An equal
/// run of one character between changes joins them (`[-を行う-]{+する+}` rather than a
/// change broken by a shared kana), and each change lists its deletion first.
pub fn diff(a: &str, b: &str) -> Vec<(Op, String)> {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let pre = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suf = a[pre..].iter().rev().zip(b[pre..].iter().rev()).take_while(|(x, y)| x == y).count();
    let (ma, mb) = (&a[pre..a.len() - suf], &b[pre..b.len() - suf]);
    let mut ops: Vec<(Op, char)> = a[..pre].iter().map(|&c| (Op::Equal, c)).collect();
    if ma.len() * mb.len() > DIFF_CELLS {
        ops.extend(ma.iter().map(|&c| (Op::Delete, c)));
        ops.extend(mb.iter().map(|&c| (Op::Insert, c)));
    } else {
        ops.extend(lcs_ops(ma, mb));
    }
    ops.extend(a[a.len() - suf..].iter().map(|&c| (Op::Equal, c)));
    let mut runs: Vec<(Op, String)> = Vec::new();
    for (op, c) in ops {
        match runs.last_mut() {
            Some((o, s)) if *o == op => s.push(c),
            _ => runs.push((op, c.to_string())),
        }
    }
    // An equal run of one character between changes joins them.
    let n = runs.len();
    let absorbed: Vec<bool> =
        (0..n).map(|i| runs[i].0 == Op::Equal && i > 0 && i + 1 < n && runs[i].1.chars().count() == 1).collect();
    let mut out: Vec<(Op, String)> = Vec::new();
    let (mut del, mut ins) = (String::new(), String::new());
    let flush = |out: &mut Vec<(Op, String)>, del: &mut String, ins: &mut String| {
        if !del.is_empty() {
            out.push((Op::Delete, std::mem::take(del)));
        }
        if !ins.is_empty() {
            out.push((Op::Insert, std::mem::take(ins)));
        }
    };
    for (i, (op, s)) in runs.into_iter().enumerate() {
        match op {
            Op::Equal if absorbed[i] => {
                del.push_str(&s);
                ins.push_str(&s);
            }
            Op::Equal => {
                flush(&mut out, &mut del, &mut ins);
                out.push((Op::Equal, s));
            }
            Op::Delete => del.push_str(&s),
            Op::Insert => ins.push_str(&s),
        }
    }
    flush(&mut out, &mut del, &mut ins);
    out
}

/// The edit of `a` into `b` along a longest common subsequence.
fn lcs_ops(a: &[char], b: &[char]) -> Vec<(Op, char)> {
    let (n, m) = (a.len(), b.len());
    // l[i][j]: the LCS of a[i..] and b[j..].
    let w = m + 1;
    let mut l = vec![0u32; (n + 1) * w];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            l[i * w + j] =
                if a[i] == b[j] { l[(i + 1) * w + j + 1] + 1 } else { l[(i + 1) * w + j].max(l[i * w + j + 1]) };
        }
    }
    let mut ops = Vec::with_capacity(n + m);
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            ops.push((Op::Equal, a[i]));
            i += 1;
            j += 1;
        } else if j < m && (i == n || l[i * w + j + 1] >= l[(i + 1) * w + j]) {
            ops.push((Op::Insert, b[j]));
            j += 1;
        } else {
            ops.push((Op::Delete, a[i]));
            i += 1;
        }
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(d: &[(Op, String)]) -> String {
        d.iter()
            .map(|(op, t)| match op {
                Op::Equal => t.clone(),
                Op::Delete => format!("[-{t}-]"),
                Op::Insert => format!("{{+{t}+}}"),
            })
            .collect()
    }

    fn apply(d: &[(Op, String)], side: Op) -> String {
        d.iter().filter(|(op, _)| *op == Op::Equal || *op == side).map(|(_, t)| t.as_str()).collect()
    }

    #[test]
    fn diff_rebuilds_both_texts() {
        let pairs = [
            ("この機能は実装が行われました。", "この機能は実装されました。"),
            (
                "彼は会議において、その件についての説明を行うことをしなかった。",
                "彼は会議でその件について説明しなかった。",
            ),
            ("", "新しい"),
            ("古い", ""),
            ("同じ", "同じ"),
            ("abc", "xyz"),
        ];
        for (a, b) in pairs {
            let d = diff(a, b);
            assert_eq!(apply(&d, Op::Delete), a, "{}", render(&d));
            assert_eq!(apply(&d, Op::Insert), b, "{}", render(&d));
        }
    }

    #[test]
    fn diff_marks_the_change() {
        assert_eq!(render(&diff("実装が行われました。", "実装されました。")), "実装[-が行わ-]{+さ+}れました。");
        assert_eq!(render(&diff("同じ", "同じ")), "同じ");
    }

    #[test]
    fn a_shared_character_joins_two_changes() {
        // "x" alone between two changes is not left as an equal run.
        let d = diff("AxB", "CxD");
        assert_eq!(render(&d), "[-AxB-]{+CxD+}");
    }

    #[test]
    fn long_texts_are_replaced_whole() {
        let a = "あ".repeat(3000);
        let b = "い".repeat(3000);
        let d = diff(&a, &b);
        assert_eq!(d.len(), 2);
        assert_eq!(apply(&d, Op::Insert), b);
    }

    #[test]
    fn form_follows_the_sentence_endings() {
        let s = |t: &str| vec![t.to_string()];
        assert_eq!(detect_form(&s("これはペンです。あれは本だ。それも本です。")), Some(Form::Polite));
        assert_eq!(detect_form(&s("説明しなかった。会議は長かった。")), Some(Form::Plain));
        assert_eq!(detect_form(&s("行きましょう！どうですか？")), Some(Form::Polite));
        assert_eq!(detect_form(&s("「はい」と言いました。")), Some(Form::Polite));
        assert_eq!(detect_form(&s("見出しだけ")), None);
        assert_eq!(detect_form(&s("Hello. World.")), None);
        assert_eq!(detect_form(&s("本です。本だ。")), None);
        // Several final particles.
        assert_eq!(detect_form(&s("そうですよね。便利ですよね。本だ。")), Some(Form::Polite));
        assert_eq!(detect_form(&s("どうですかね。いいですね。")), Some(Form::Polite));
        // Emoji after an ending; a polite line without 。; list items and headings left out.
        assert_eq!(detect_form(&s("生産性は向上します✨\nこれが鍵なのです🚀")), Some(Form::Polite));
        assert_eq!(detect_form(&s("### ポイント\n- 実施する\n- 徹底する！\n1. 確認する。")), None);
        assert_eq!(detect_form(&s("説明しなかった。\n概要")), Some(Form::Plain));
    }

    #[test]
    fn each_paragraph_keeps_its_own_form() {
        let segs: Vec<String> = ["説明します。", "説明しなかった。", "# 見出し"].map(String::from).into();
        // (the heading takes the document's: one polite, one plain is no majority)
        assert_eq!(forms(FormChoice::Auto, &segs), vec![Some(Form::Polite), Some(Form::Plain), None]);
        assert_eq!(forms(FormChoice::Plain, &segs), vec![Some(Form::Plain); 3]);
    }

    #[test]
    fn lost_words_are_found() {
        let m = missing("2026年にAPIを公開し、「設定」画面を変えた。", "今年、APIの設定画面を変えた。");
        assert_eq!(m, vec!["2026"]);
        assert_eq!(missing("「設定」画面", "「構成」画面"), vec!["設定"]);
        // Full-width and half-width are the same word; a reordering loses nothing.
        assert!(missing("ＡＰＩは３つ", "3つのAPI").is_empty());
        assert!(missing("https://example.com/a を見る", "https://example.com/a を参照").is_empty());
        assert_eq!(missing("v1.2と1,000件", "v1.3と1000件"), vec!["v1.2", "1,000"]);
    }

    #[test]
    fn an_empty_rewrite_keeps_the_source() {
        let r = Review::of("本文です。", "");
        assert!(r.empty && r.to_check() && !r.changed());
    }

    #[test]
    fn unlikely_pieces_join_into_spans() {
        let p = |start, end, prob: f32| Scored {
            start,
            end,
            logprob: prob.ln(),
            best: Choice::Text("x".into()),
            best_logprob: 0.0,
        };
        let pieces = [p(0, 3, 0.9), p(3, 6, 0.01), p(6, 9, 0.05), p(9, 12, 0.9), p(12, 15, 0.02)];
        let text = "abcdefghijklmno";
        assert_eq!(unlikely(text, &pieces, (0.1f32).ln()), vec![(3, 9, 1), (12, 15, 4)]);
        assert!(unlikely(text, &pieces, (0.001f32).ln()).is_empty());
    }

    #[test]
    fn a_longer_token_for_the_same_text_is_no_mark() {
        // 速く|な|る: the model's なる is the text's な and る.
        let text = "速くなる";
        let p = |start, end, best: &str| Scored {
            start,
            end,
            logprob: -9.0,
            best: Choice::Text(best.into()),
            best_logprob: 0.0,
        };
        let likely = Scored { logprob: -0.01, ..p(9, 12, "る") };
        assert!(unlikely(text, &[p(6, 9, "なる"), likely], -2.0).is_empty());
        // A shorter choice (時間 for 時間ほど) or another word is a mark.
        let text = "時間ほど";
        assert_eq!(unlikely(text, &[p(0, 12, "時間")], -2.0), vec![(0, 12, 0)]);
        assert_eq!(unlikely("速くなる", &[p(6, 9, "する")], -2.0), vec![(6, 9, 0)]);
    }

    #[test]
    fn stock_phrases_count_once() {
        assert_eq!(stock_phrases("重要であると言えるだろう。さまざまな観点から"), 3);
        assert_eq!(stock_phrases("川が流れる。"), 0);
    }

    /// Translates by a table (`from→to` of each paragraph) and scores -1 per character.
    struct Fake(Vec<(String, String)>);

    impl Polisher for Fake {
        fn translate(&mut self, _: &str, req: DocumentRequest) -> Result<Vec<String>> {
            let to = &req.target_lang;
            Ok(req
                .segments
                .iter()
                .map(|t| self.0.iter().find(|(s, _)| s == t).map_or_else(|| format!("{t}@{to}"), |(_, o)| o.clone()))
                .collect())
        }
        fn score(&mut self, r: &ScoreRequest) -> Result<Vec<Scored>> {
            assert!(!r.input.trim().is_empty(), "an empty input reached score");
            let n = r.output.len();
            Ok(if n == 0 {
                Vec::new()
            } else {
                let chars = r.output.chars().count() as f32;
                vec![Scored { start: 0, end: n, logprob: -chars, best: Choice::End, best_logprob: 0.0 }]
            })
        }
    }

    #[test]
    fn an_empty_pivot_or_rewrite_is_not_scored() {
        let segs: Vec<String> = ["一つ目です。", "二つ目です。", "三つ目です。"].map(String::from).into();
        let o = Opts { max_new_tokens: 8, history_chars: 0, chains: 0 };
        // The second paragraph's English comes back empty, the third's Japanese rewrite.
        let mut p = Fake(vec![("二つ目です。".into(), "".into()), ("三つ目です。@en".into(), "".into())]);
        let forms = forms(FormChoice::Auto, &segs);
        let n = naturalness(&mut p, &o, &Via::Lang("en".into()), &segs, &forms).unwrap();
        assert_eq!(n[0].per_char, Some(-1.0));
        assert!(n[0].own.is_some());
        assert_eq!(n[1].per_char, None);
        assert!(n[1].pieces.is_empty() && n[1].own.is_none());
        // (an empty rewrite would have scored 0, better than any text)
        assert_eq!(n[2].per_char, Some(-1.0));
        assert!(n[2].own.is_none());
    }

    #[test]
    fn a_pivot_comes_from_a_command() {
        let segs: Vec<String> = ["一つ目。", "二つ目。"].map(String::from).into();
        assert_eq!(pivot_by_command("sed '/./s/^/E:/'", &segs).unwrap(), vec!["E:一つ目。", "E:二つ目。"]);
        let e = pivot_by_command("echo one", &segs).unwrap_err().to_string();
        assert!(e.contains("1 paragraphs for 2"), "{e}");
        let e = pivot_by_command("exit 3", &segs).unwrap_err().to_string();
        assert!(e.contains("failed"), "{e}");
        // A route through it: the model writes the Japanese from the command's pivot.
        let mut p = Fake(Vec::new());
        let o = Opts { max_new_tokens: 8, history_chars: 0, chains: 0 };
        let forms = forms(FormChoice::Auto, &segs);
        let c = route(&mut p, &o, &Via::Lang("en".into()), &segs, &forms, Some("sed '/./s/^/E:/'")).unwrap();
        assert_eq!(c.pivot.unwrap(), vec!["E:一つ目。", "E:二つ目。"]);
        assert_eq!(c.output, vec!["E:一つ目。@ja", "E:二つ目。@ja"]);
    }

    #[test]
    fn a_special_choice_is_no_text() {
        let p = Scored { start: 0, end: 3, logprob: -9.0, best: Choice::End, best_logprob: 0.0 };
        assert!(!same_text("あいう", &p));
        assert_eq!(Choice::End.to_string(), "(end)");
        assert_eq!(Choice::Other.to_string(), "?");
    }

    #[test]
    fn via_parses() {
        assert_eq!("direct".parse::<Via>(), Ok(Via::Direct));
        assert_eq!("zh-hans".parse::<Via>(), Ok(Via::Lang("zh-hans".into())));
        assert!("".parse::<Via>().is_err());
        assert!("ja".parse::<Via>().is_err());
        // Language codes are checked when parsed, not after the model has loaded.
        assert!("EN".parse::<Via>().is_err());
        assert!("english".parse::<Via>().is_err());
        assert_eq!("ja-easy".parse::<Via>(), Ok(Via::Lang("ja-easy".into())));
    }
}
