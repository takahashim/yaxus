//! The bearer token every `/v1/translate*` request must carry.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::Response;

use crate::secret_file::write_private;

/// Shortest token accepted.
const MIN_LEN: usize = 16;

pub struct AccessToken(String);

impl AccessToken {
    /// A token given by the user. Clients' tokens are trimmed before the comparison, so
    /// this one is too.
    pub fn new(token: &str) -> Result<Self> {
        let t = token.trim();
        anyhow::ensure!(t.len() >= MIN_LEN, "token must be at least {MIN_LEN} characters");
        Ok(AccessToken(t.to_string()))
    }

    /// The token stored in `dir/token`, or a new random one saved there. Returns it and
    /// whether it was created.
    pub fn load_or_create(dir: &Path) -> Result<(Self, bool)> {
        let path = dir.join("token");
        match std::fs::read_to_string(&path) {
            Ok(t) if !t.trim().is_empty() => return Ok((AccessToken::new(&t)?, false)),
            // Missing or empty: create one below.
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            // Replacing an unreadable token would lock every client out without saying why.
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        }
        std::fs::create_dir_all(dir)?;
        let mut buf = [0u8; 24];
        getrandom::fill(&mut buf).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
        let t: String = buf.iter().map(|b| format!("{b:02x}")).collect();
        write_private(&path, format!("{t}\n").as_bytes())?;
        Ok((AccessToken(t), true))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether an `Authorization` header carries this token.
    fn accepts(&self, authorization: Option<&HeaderValue>) -> bool {
        authorization
            .and_then(|v| v.to_str().ok())
            .and_then(bearer_token)
            .is_some_and(|t| constant_time_eq(t.as_bytes(), self.0.as_bytes()))
    }
}

/// Middleware: answers 401 to a request without the token.
pub async fn require(State(token): State<Arc<AccessToken>>, req: Request, next: Next) -> Response {
    if !token.accepts(req.headers().get(header::AUTHORIZATION)) {
        let mut r = crate::error(StatusCode::UNAUTHORIZED, "missing or invalid bearer token");
        r.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        return r;
    }
    next.run(req).await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The token of an `Authorization: Bearer <token>` header (the scheme is case-insensitive).
fn bearer_token(value: &str) -> Option<&str> {
    let (scheme, token) = value.trim().split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_scheme_is_case_insensitive() {
        assert_eq!(bearer_token("Bearer abc"), Some("abc"));
        assert_eq!(bearer_token("bearer  abc "), Some("abc"));
        assert_eq!(bearer_token("BEARER abc"), Some("abc"));
        assert_eq!(bearer_token("Basic abc"), None);
        assert_eq!(bearer_token("Bearerabc"), None);
    }

    #[test]
    fn a_token_accepts_only_itself() {
        let t = AccessToken::new(" 0123456789abcdef\n").unwrap();
        let h = |v: &str| Some(HeaderValue::from_str(v).unwrap());
        assert!(t.accepts(h("Bearer 0123456789abcdef").as_ref()));
        assert!(!t.accepts(h("Bearer 0123456789abcdeg").as_ref()));
        assert!(!t.accepts(h("Bearer 0123456789abcdef0").as_ref()));
        assert!(!t.accepts(None));
        assert!(AccessToken::new("short").is_err());
    }

    #[test]
    fn token_file_is_created_reused_and_never_silently_replaced() {
        let dir = std::env::temp_dir().join(format!("yaxus-token-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (t, created) = AccessToken::load_or_create(&dir).unwrap();
        assert!(created && t.as_str().len() == 48);
        let (again, created) = AccessToken::load_or_create(&dir).unwrap();
        assert!(!created && again.as_str() == t.as_str());
        // An empty file gets a new token.
        std::fs::write(dir.join("token"), "\n").unwrap();
        let (t2, created) = AccessToken::load_or_create(&dir).unwrap();
        assert!(created && t2.as_str() != t.as_str());
        // A file that cannot be read is an error, not a new token.
        std::fs::write(dir.join("token"), [0xff, 0xfe]).unwrap();
        assert!(AccessToken::load_or_create(&dir).is_err());
        assert_eq!(std::fs::read(dir.join("token")).unwrap(), [0xff, 0xfe]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
