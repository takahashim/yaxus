//! Self-managed TLS: a local CA (created once) signs the server certificate.
//! Clients pin `ca.pem` (`yaxus client --ca`), browsers can import it.

use std::fs;
use std::io::Write;
use std::net::{IpAddr, UdpSocket};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, KeyPair, KeyUsagePurpose};

pub struct TlsFiles {
    pub cert: PathBuf,
    pub key: PathBuf,
    /// The CA certificate to give to clients (None when the user supplied a certificate).
    pub ca: Option<PathBuf>,
    pub sans: Vec<String>,
}

pub fn write_private(path: &Path, data: &[u8]) -> Result<()> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path).with_context(|| format!("writing {}", path.display()))?;
    f.write_all(data)?;
    Ok(())
}

/// Names the server is likely reached by: localhost, the host name and the primary IP.
pub fn default_sans() -> Vec<String> {
    let mut sans = vec!["localhost".to_string(), "127.0.0.1".to_string(), "::1".to_string()];
    if let Ok(h) = fs::read_to_string("/proc/sys/kernel/hostname") {
        let h = h.trim();
        if !h.is_empty() {
            sans.push(h.to_string());
            sans.push(format!("{h}.local"));
        }
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
pub fn ensure_self_signed(dir: &Path, mut sans: Vec<String>) -> Result<TlsFiles> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    normalize(&mut sans);
    let ca_cert_path = dir.join("ca.pem");
    let ca_key_path = dir.join("ca-key.pem");
    let cert_path = dir.join("server.pem");
    let key_path = dir.join("server-key.pem");
    let sans_path = dir.join("server-sans.txt");

    let (ca_params, ca_key) = if ca_cert_path.exists() && ca_key_path.exists() {
        let key = KeyPair::from_pem(&fs::read_to_string(&ca_key_path)?)?;
        let params = CertificateParams::from_ca_cert_pem(&fs::read_to_string(&ca_cert_path)?)?;
        (params, key)
    } else {
        let mut params = CertificateParams::new(Vec::<String>::new())?;
        params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        params.distinguished_name.push(DnType::CommonName, "yaxus local CA");
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        params.not_before = rcgen::date_time_ymd(2025, 1, 1);
        params.not_after = rcgen::date_time_ymd(2045, 1, 1);
        let key = KeyPair::generate()?;
        let cert = params.clone().self_signed(&key)?;
        write_private(&ca_key_path, key.serialize_pem().as_bytes())?;
        fs::write(&ca_cert_path, cert.pem())?;
        tracing::info!("created local CA {}", ca_cert_path.display());
        // Any old server certificate was signed by a different CA.
        let _ = fs::remove_file(&sans_path);
        (params, key)
    };
    let ca_cert = ca_params.self_signed(&ca_key)?;

    let current = fs::read_to_string(&sans_path).unwrap_or_default();
    let wanted = sans.join("\n");
    if current != wanted || !cert_path.exists() || !key_path.exists() {
        let mut params = CertificateParams::new(sans.clone())?;
        params.distinguished_name.push(DnType::CommonName, "yaxus");
        params.not_before = rcgen::date_time_ymd(2025, 1, 1);
        params.not_after = rcgen::date_time_ymd(2035, 1, 1);
        let key = KeyPair::generate()?;
        let cert = params.signed_by(&key, &ca_cert, &ca_key)?;
        write_private(&key_path, key.serialize_pem().as_bytes())?;
        fs::write(&cert_path, cert.pem())?;
        fs::write(&sans_path, &wanted)?;
        tracing::info!("issued server certificate for {}", sans.join(", "));
    }
    Ok(TlsFiles { cert: cert_path, key: key_path, ca: Some(ca_cert_path), sans })
}
