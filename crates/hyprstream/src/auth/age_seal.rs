//! Shared `age` seal/open boundary for deployment secrets.
//!
//! Deployment trust minting and relational UserStore DEK wrapping deliberately
//! use this one subprocess boundary. Keeping recipient/identity validation and
//! partial-plaintext zeroization here prevents the two production paths from
//! drifting into subtly different `age` invocations.

use anyhow::{anyhow, ensure, Context, Result};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{Seek as _, SeekFrom, Write as _},
    os::fd::{AsRawFd as _, FromRawFd as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use zeroize::Zeroizing;

/// A validated, non-empty set of deployment `age` recipients.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AgeRecipients(Vec<String>);

impl AgeRecipients {
    pub(crate) fn new(recipients: Vec<String>) -> Result<Self> {
        let mut distinct = BTreeSet::new();
        for recipient in recipients {
            let recipient = recipient.trim();
            ensure!(!recipient.is_empty(), "age recipient is empty");
            ensure!(
                recipient.is_ascii() && !recipient.contains(['\n', '\r', '\0']),
                "age recipient contains invalid characters"
            );
            distinct.insert(recipient.to_owned());
        }
        ensure!(!distinct.is_empty(), "no age recipients supplied");
        Ok(Self(distinct.into_iter().collect()))
    }

    /// Encrypt bytes through the deployment trust-mint `age` seam.
    pub(crate) fn seal(&self, plaintext: &[u8], max_ciphertext_bytes: usize) -> Result<Vec<u8>> {
        let mut command = age_command(false);
        for recipient in &self.0 {
            command.arg("--recipient").arg(recipient);
        }
        command.arg("-");
        let output = run_with_stdin(command, plaintext, "encryption")?;
        ensure!(output.status.success(), "age encryption failed");
        ensure!(
            !output.stdout.is_empty() && output.stdout.len() <= max_ciphertext_bytes,
            "age encryption returned invalid ciphertext size"
        );
        Ok(output.stdout.to_vec())
    }
}

/// Where an `age` identity comes from: an on-disk file, or in-memory bytes
/// staged into an anonymous memfd for the `age` child so the inherited-FD
/// credential interface (`mint-registry-jwt --identity-fd`) never writes
/// plaintext identity material to a filesystem path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AgeIdentitySource {
    Path(PathBuf),
    InMemory(Zeroizing<Vec<u8>>),
}

/// A validated, non-empty set of deployment `age` identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AgeIdentities(Vec<AgeIdentitySource>);

impl AgeIdentities {
    pub(crate) fn new(identities: Vec<PathBuf>) -> Result<Self> {
        Self::from_sources(
            identities
                .into_iter()
                .map(AgeIdentitySource::Path)
                .collect(),
        )
    }

    /// In-memory identities read from inherited file descriptors. The bytes
    /// are handed to the `age` child through an anonymous memfd, never a
    /// filesystem path.
    pub(crate) fn new_in_memory(identities: Vec<Zeroizing<Vec<u8>>>) -> Result<Self> {
        Self::from_sources(
            identities
                .into_iter()
                .map(AgeIdentitySource::InMemory)
                .collect(),
        )
    }

    pub(crate) fn from_sources(identities: Vec<AgeIdentitySource>) -> Result<Self> {
        ensure!(!identities.is_empty(), "no age identities supplied");
        for identity in &identities {
            match identity {
                AgeIdentitySource::Path(path) => ensure!(
                    path.is_file(),
                    "age identity file does not exist: {}",
                    path.display()
                ),
                AgeIdentitySource::InMemory(bytes) => {
                    ensure!(!bytes.is_empty(), "in-memory age identity is empty");
                }
            }
        }
        Ok(Self(identities))
    }

    /// Decrypt an in-memory ciphertext, zeroizing partial output on every
    /// status and size error path.
    pub(crate) fn open(
        &self,
        ciphertext: &[u8],
        max_plaintext_bytes: usize,
    ) -> Result<Zeroizing<Vec<u8>>> {
        ensure!(!ciphertext.is_empty(), "age ciphertext is empty");
        let (mut command, _guards) = self.decrypt_command()?;
        command.arg("-");
        let output = run_with_stdin(command, ciphertext, "decryption")?;
        checked_plaintext(output, max_plaintext_bytes)
    }

    /// Decrypt a file without copying its ciphertext into process memory.
    pub(crate) fn open_file(
        &self,
        path: &Path,
        max_plaintext_bytes: usize,
    ) -> Result<Zeroizing<Vec<u8>>> {
        ensure!(
            path.is_file(),
            "age ciphertext file does not exist: {}",
            path.display()
        );
        let (mut command, _guards) = self.decrypt_command()?;
        command.stdin(Stdio::inherit()).arg(path);
        let output = command.output().context("launch age decryption")?;
        checked_plaintext(
            GuardedOutput {
                status: output.status,
                stdout: Zeroizing::new(output.stdout),
            },
            max_plaintext_bytes,
        )
    }

    /// Build the `age --decrypt` command. Returned `File` guards keep the
    /// memfd descriptors backing in-memory identities alive (and inheritable
    /// by the child) until the command has been awaited.
    fn decrypt_command(&self) -> Result<(Command, Vec<File>)> {
        let mut command = age_command(true);
        let mut guards = Vec::new();
        for identity in &self.0 {
            command.arg("--identity");
            match identity {
                AgeIdentitySource::Path(path) => {
                    command.arg(path);
                }
                AgeIdentitySource::InMemory(bytes) => {
                    let memfd = memfd_identity(bytes)?;
                    command.arg(format!("/proc/self/fd/{}", memfd.as_raw_fd()));
                    guards.push(memfd);
                }
            }
        }
        Ok((command, guards))
    }
}

/// Stage identity bytes in an anonymous memfd the `age` child inherits.
///
/// The memfd is created without `MFD_CLOEXEC` so the child spawned by
/// [`Command`] inherits it; `Command` only closes descriptors flagged
/// close-on-exec. The caller keeps the returned `File` alive until the child
/// has exited.
fn memfd_identity(bytes: &[u8]) -> Result<File> {
    let name = c"hyprstream-age-identity";
    let fd = unsafe { libc::memfd_create(name.as_ptr(), 0) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error()).context("create memfd for age identity");
    }
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(bytes)
        .context("write age identity to memfd")?;
    file.seek(SeekFrom::Start(0))
        .context("rewind age identity memfd")?;
    Ok(file)
}

fn age_command(decrypt: bool) -> Command {
    let mut command = Command::new("age");
    command
        .arg(if decrypt { "--decrypt" } else { "--encrypt" })
        .arg("--output")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    command
}

fn run_with_stdin(
    mut command: Command,
    input: &[u8],
    operation: &'static str,
) -> Result<GuardedOutput> {
    let mut child = command
        .spawn()
        .with_context(|| format!("launch age {operation}"))?;
    let write_result = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("age stdin unavailable"))?
        .write_all(input)
        .with_context(|| format!("write age {operation} input"));
    let output = child
        .wait_with_output()
        .with_context(|| format!("wait for age {operation}"))?;
    let output = GuardedOutput {
        status: output.status,
        // Encryption produces ciphertext, while decryption can produce partial
        // plaintext before failure. Guard both so a stdin-write error is also
        // unable to bypass zeroization.
        stdout: Zeroizing::new(output.stdout),
    };
    write_result?;
    Ok(output)
}

struct GuardedOutput {
    status: std::process::ExitStatus,
    stdout: Zeroizing<Vec<u8>>,
}

fn checked_plaintext(
    output: GuardedOutput,
    max_plaintext_bytes: usize,
) -> Result<Zeroizing<Vec<u8>>> {
    ensure!(output.status.success(), "age decryption failed");
    ensure!(
        !output.stdout.is_empty() && output.stdout.len() <= max_plaintext_bytes,
        "age decryption returned invalid plaintext size"
    );
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipient_validation_is_nonempty_distinct_and_control_free() {
        assert!(AgeRecipients::new(Vec::new()).is_err());
        assert!(AgeRecipients::new(vec!["age1ok\nage1injected".to_owned()]).is_err());
        assert_eq!(
            AgeRecipients::new(vec![" age1same ".to_owned(), "age1same".to_owned()])
                .unwrap_or_else(|error| panic!("{error}"))
                .0,
            vec!["age1same"]
        );
    }

    #[test]
    fn identity_validation_is_fail_closed() {
        assert!(AgeIdentities::new(Vec::new()).is_err());
        assert!(AgeIdentities::new(vec![PathBuf::from("/definitely/not/an/identity")]).is_err());
    }

    #[test]
    fn in_memory_identity_validation_is_fail_closed() {
        assert!(AgeIdentities::new_in_memory(Vec::new()).is_err());
        assert!(AgeIdentities::new_in_memory(vec![Zeroizing::new(Vec::new())]).is_err());
        assert!(AgeIdentities::new_in_memory(vec![Zeroizing::new(
            b"AGE-SECRET-KEY-1TEST".to_vec()
        )])
        .is_ok());
    }
}
