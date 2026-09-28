//! Wire-compatible bytes whose diagnostic representation never exposes contents.

/// Serialization intentionally preserves the bytes for the authenticated wire
/// protocol. Debug is redacted; callers must not log serialized values either.
#[derive(Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct SensitiveBytes(Vec<u8>);

/// Domain types over wire `Data` enforce their own byte budget at decode,
/// before any owned copy is materialized by generated readers.
pub trait BoundedWireBytes: Sized {
    fn from_wire_bytes(bytes: &[u8]) -> anyhow::Result<Self>;
}

impl SensitiveBytes {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

}

impl BoundedWireBytes for SensitiveBytes {
    /// Bounded wire decode: reject before any owned copy when the peer sent
    /// more than the mediated-evidence cap. The outer envelope frame has its
    /// own (larger) bound; this enforces the stated evidence budget at the
    /// field boundary so generated dispatch never materializes an oversized
    /// buffer ahead of verification.
    fn from_wire_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        use crate::envelope::MAX_MEDIATED_EVIDENCE_BYTES;
        anyhow::ensure!(
            bytes.len() <= MAX_MEDIATED_EVIDENCE_BYTES,
            "sensitive wire bytes exceed the mediated evidence cap"
        );
        Ok(Self(bytes.to_vec()))
    }
}

impl From<Vec<u8>> for SensitiveBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self::new(bytes)
    }
}

impl std::ops::Deref for SensitiveBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for SensitiveBytes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SensitiveBytes([REDACTED])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_contents_without_changing_wire_bytes() {
        let value = SensitiveBytes::new(b"secret-credential".to_vec());
        assert_eq!(format!("{value:?}"), "SensitiveBytes([REDACTED])");
        assert_eq!(&*value, b"secret-credential");
    }

    #[test]
    fn wire_decode_rejects_oversized_before_copy() {
        use crate::envelope::MAX_MEDIATED_EVIDENCE_BYTES;
        use super::BoundedWireBytes;
        let oversized = vec![0u8; MAX_MEDIATED_EVIDENCE_BYTES + 1];
        assert!(SensitiveBytes::from_wire_bytes(&oversized).is_err());
        let exact = vec![0u8; MAX_MEDIATED_EVIDENCE_BYTES];
        assert!(SensitiveBytes::from_wire_bytes(&exact).is_ok());
    }
}
