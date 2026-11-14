//! Self-managed TLS: a local CA (created once) signs the server certificate.
//! Clients pin `ca.pem` (`yaxus client --ca`), browsers can import it.

use std::fs;
use std::net::{IpAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use axum_server::tls_rustls::RustlsConfig;
use rcgen::{BasicConstraints, Certificate, CertificateParams, DnType, IsCa, KeyPair, KeyUsagePurpose};

use crate::secret_file::write_private;

pub struct TlsFiles {
    pub cert: PathBuf,
    pub key: PathBuf,
    /// The CA certificate to give to clients (None when the user supplied a certificate).
    pub ca: Option<PathBuf>,
    pub sans: Vec<String>,
}

impl TlsFiles {
    /// The user's certificate, or else the self-signed one in `dir` (created or renewed
    /// if needed) for the [`default_sans`] and `extra_sans`.
    pub fn resolve(cert: Option<&Path>, key: Option<&Path>, extra_sans: &[String], dir: &Path) -> Result<Self> {
        match (cert, key) {
            (Some(c), Some(k)) => Ok(TlsFiles { cert: c.into(), key: k.into(), ca: None, sans: vec![] }),
            (None, None) => {
                let mut sans = default_sans();
                sans.extend(extra_sans.iter().cloned());
                ensure_self_signed(dir, sans)
            }
            _ => anyhow::bail!("--cert and --key must be given together"),
        }
    }

    /// Reads the certificate and the key.
    pub async fn load(&self) -> Result<RustlsConfig> {
        RustlsConfig::from_pem_file(&self.cert, &self.key)
            .await
            .with_context(|| format!("loading TLS certificate {}", self.cert.display()))
    }
}

/// Names the server is likely reached by: localhost, the host name and the primary IP.
pub fn default_sans() -> Vec<String> {
    let mut sans = vec!["localhost".to_string(), "127.0.0.1".to_string(), "::1".to_string()];
    if let Some(h) = host_name() {
        sans.push(format!("{h}.local"));
        sans.push(h);
    }
    // Connecting a UDP socket sends nothing; it only selects the outbound interface.
    if let Ok(s) = UdpSocket::bind("0.0.0.0:0")
        && s.connect("192.0.2.1:9").is_ok()
        && let Ok(a) = s.local_addr()
        && !a.ip().is_unspecified()
    {
        sans.push(a.ip().to_string());
    }
    sans
}

/// The host name without `.local` (macOS gives it with one), if there is one.
fn host_name() -> Option<String> {
    let h = fs::read_to_string("/proc/sys/kernel/hostname").ok().or_else(|| {
        // (not Linux: the command is everywhere else the server runs)
        let out = std::process::Command::new("hostname").output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    })?;
    let h = h.trim();
    let h = h.strip_suffix(".local").unwrap_or(h);
    (!h.is_empty()).then(|| h.to_string())
}

fn normalize(sans: &mut Vec<String>) {
    for s in sans.iter_mut() {
        if let Ok(ip) = s.parse::<IpAddr>() {
            *s = ip.to_string();
        }
    }
    sans.sort();
    sans.dedup();
}

/// Creates (or reuses) the CA and a server certificate covering `sans`.
fn ensure_self_signed(dir: &Path, mut sans: Vec<String>) -> Result<TlsFiles> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    normalize(&mut sans);
    let (ca, created) = LocalCa::load_or_create(dir)?;
    let server = ServerCert::in_dir(dir);
    // (any old server certificate was signed by a different CA)
    if created || !server.is_valid_for(&sans, SystemTime::now()) {
        server.issue(&ca, &sans)?;
    }
    Ok(TlsFiles { cert: server.cert, key: server.key, ca: Some(ca.cert_path), sans })
}

/// The CA that signs the server certificate; clients pin its certificate.
struct LocalCa {
    cert_path: PathBuf,
    cert: Certificate,
    key: KeyPair,
}

impl LocalCa {
    /// The CA stored in `dir`, or a new one saved there. Returns it and whether it was
    /// created.
    fn load_or_create(dir: &Path) -> Result<(Self, bool)> {
        let cert_path = dir.join("ca.pem");
        let key_path = dir.join("ca-key.pem");
        if cert_path.exists() && key_path.exists() {
            let key = KeyPair::from_pem(&fs::read_to_string(&key_path)?)?;
            let params = CertificateParams::from_ca_cert_pem(&fs::read_to_string(&cert_path)?)?;
            // (signed again only to sign with: the issuer is the same name and key)
            let cert = params.self_signed(&key)?;
            return Ok((LocalCa { cert_path, cert, key }, false));
        }
        let mut params = CertificateParams::new(Vec::<String>::new())?;
        params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        params.distinguished_name.push(DnType::CommonName, "yaxus local CA");
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        params.not_before = rcgen::date_time_ymd(2025, 1, 1);
        params.not_after = rcgen::date_time_ymd(2045, 1, 1);
        let key = KeyPair::generate()?;
        let cert = params.self_signed(&key)?;
        write_private(&key_path, key.serialize_pem().as_bytes())?;
        fs::write(&cert_path, cert.pem())?;
        tracing::info!("created local CA {}", cert_path.display());
        Ok((LocalCa { cert_path, cert, key }, true))
    }
}

/// How long a server certificate is valid: Apple's platforms refuse longer ones, even
/// under a CA the user added.
const SERVER_VALIDITY: Duration = Duration::from_secs(397 * 86400);
/// How long before it expires a server certificate is renewed.
const RENEW_BEFORE: Duration = Duration::from_secs(30 * 86400);

/// The server certificate's files in the config directory.
struct ServerCert {
    cert: PathBuf,
    key: PathBuf,
    /// The names it was issued for, one per line.
    sans: PathBuf,
}

impl ServerCert {
    fn in_dir(dir: &Path) -> Self {
        ServerCert { cert: dir.join("server.pem"), key: dir.join("server-key.pem"), sans: dir.join("server-sans.txt") }
    }

    /// The certificate exists, covers `sans` and stays valid long enough after `now`
    /// (one issued for longer than [`SERVER_VALIDITY`], as earlier versions did, is not).
    fn is_valid_for(&self, sans: &[String], now: SystemTime) -> bool {
        let issued_for = fs::read_to_string(&self.sans).unwrap_or_default();
        if issued_for != sans.join("\n") || !self.key.exists() {
            return false;
        }
        let Some(not_after) = fs::read_to_string(&self.cert)
            .ok()
            .and_then(|pem| CertificateParams::from_ca_cert_pem(&pem).ok())
            .and_then(|p| u64::try_from(p.not_after.unix_timestamp()).ok())
        else {
            return false;
        };
        let now = unix_secs(now);
        let day = 86400;
        now + RENEW_BEFORE.as_secs() < not_after && not_after <= now + SERVER_VALIDITY.as_secs() + day
    }

    fn issue(&self, ca: &LocalCa, sans: &[String]) -> Result<()> {
        let now = rcgen::date_time_ymd(1970, 1, 1) + Duration::from_secs(unix_secs(SystemTime::now()));
        let mut params = CertificateParams::new(sans.to_vec())?;
        params.distinguished_name.push(DnType::CommonName, "yaxus");
        // (a day early, for clients whose clocks are behind)
        params.not_before = now - Duration::from_secs(86400);
        params.not_after = now + SERVER_VALIDITY;
        let key = KeyPair::generate()?;
        let cert = params.signed_by(&key, &ca.cert, &ca.key)?;
        write_private(&self.key, key.serialize_pem().as_bytes())?;
        fs::write(&self.cert, cert.pem())?;
        fs::write(&self.sans, sans.join("\n"))?;
        tracing::info!("issued server certificate for {}", sans.join(", "));
        Ok(())
    }
}

fn unix_secs(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_server_certificate_is_renewed_before_it_expires() {
        let dir = std::env::temp_dir().join(format!("yaxus-tls-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let sans = vec!["localhost".to_string()];
        let files = ensure_self_signed(&dir, sans.clone()).unwrap();
        let server = ServerCert::in_dir(&dir);
        let now = SystemTime::now();
        assert!(server.is_valid_for(&sans, now));
        assert!(!server.is_valid_for(&["other".to_string()], now));
        assert!(!server.is_valid_for(&sans, now + SERVER_VALIDITY - RENEW_BEFORE));

        // Kept while valid, with the same CA.
        let pem = fs::read_to_string(&files.cert).unwrap();
        let again = ensure_self_signed(&dir, sans).unwrap();
        assert_eq!(fs::read_to_string(&again.cert).unwrap(), pem);
        assert_eq!(again.ca, files.ca);
        fs::remove_dir_all(&dir).unwrap();
    }
}
