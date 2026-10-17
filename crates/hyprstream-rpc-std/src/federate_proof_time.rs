//! Unit boundary between millisecond RPC envelopes and second-based proof CWTs.

pub(crate) fn proof_timestamps(envelope_iat_ms: i64) -> anyhow::Result<(u64, u64)> {
    // Reject negative values before division: -1ms must not truncate to zero.
    let issued_at_ms = u64::try_from(envelope_iat_ms)
        .map_err(|_| anyhow::anyhow!("Federate proof requires a nonnegative request timestamp"))?;
    let issued_at = issued_at_ms / 1000;
    let expires_at = issued_at
        .checked_add(30)
        .ok_or_else(|| anyhow::anyhow!("Federate proof timestamp overflow"))?;
    Ok((issued_at, expires_at))
}

#[cfg(test)]
mod tests {
    use super::proof_timestamps;

    #[test]
    fn federate_proof_time_converts_epoch_milliseconds_to_seconds() {
        assert_eq!(
            proof_timestamps(1_792_210_000_000).unwrap(),
            (1_792_210_000, 1_792_210_030)
        );
    }

    #[test]
    fn federate_proof_time_floors_subseconds_and_rejects_negative_input() {
        for (milliseconds, seconds) in [(0, 0), (999, 0), (1000, 1), (1999, 1)] {
            assert_eq!(
                proof_timestamps(milliseconds).unwrap(),
                (seconds, seconds + 30)
            );
        }
        for milliseconds in [-1, -999, -1000, i64::MIN] {
            assert!(proof_timestamps(milliseconds).is_err());
        }
    }

    #[test]
    fn federate_proof_time_maximum_input_cannot_overflow_expiry() {
        let seconds = (i64::MAX as u64) / 1000;
        assert_eq!(proof_timestamps(i64::MAX).unwrap(), (seconds, seconds + 30));
    }
}
