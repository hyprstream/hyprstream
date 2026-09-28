//! Wire-compatible bytes whose diagnostic representation never exposes contents.

/// Serialization intentionally preserves the bytes for the authenticated wire
/// protocol. Debug is redacted; callers must not log serialized values either.
#[derive(Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct SensitiveBytes(Vec<u8>);

impl SensitiveBytes {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
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
}
