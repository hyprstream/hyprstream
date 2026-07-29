//! Shared `age` seal/open boundary for deployment secrets.
//!
//! Deployment trust minting and relational UserStore DEK wrapping deliberately
//! use this one subprocess boundary. Keeping recipient/identity validation and
//! partial-plaintext zeroization here prevents the two production paths from
//! drifting into subtly different `age` invocations.

use anyhow::{anyhow, ensure, Context, Result};
use std::{
    collections::BTreeSet,
    io::Write as _,
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

/// A validated, non-empty set of deployment `age` identity files.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AgeIdentities(Vec<PathBuf>);

impl AgeIdentities {
    pub(crate) fn new(identities: Vec<PathBuf>) -> Result<Self> {
        ensure!(!identities.is_empty(), "no age identities supplied");
        for identity in &identities {
            ensure!(
                identity.is_file(),
                "age identity file does not exist: {}",
                identity.display()
            );
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
        let mut command = self.decrypt_command();
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
        let mut command = self.decrypt_command();
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

    fn decrypt_command(&self) -> Command {
        let mut command = age_command(true);
        for identity in &self.0 {
            command.arg("--identity").arg(identity);
        }
        command
    }
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
}
