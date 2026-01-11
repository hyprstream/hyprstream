//! Service notification helpers
//!
//! Wraps systemd sd_notify protocol for service lifecycle notifications.
//! When the `systemd` feature is disabled, all functions are no-ops.

use anyhow::Result;

/// Notify systemd that service is ready
///
/// Sends `READY=1` to the notification socket.
/// This should be called after the service has completed initialization
/// and is ready to handle requests.
#[cfg(feature = "systemd")]
pub fn ready() -> Result<()> {
    let state = [("READY", "1")];
    systemd::daemon::notify(false, state.iter())?;
    Ok(())
}

#[cfg(not(feature = "systemd"))]
pub fn ready() -> Result<()> {
    Ok(())
}

/// Notify systemd that service is stopping
///
/// Sends `STOPPING=1` to the notification socket.
/// This should be called when the service begins its shutdown sequence.
#[cfg(feature = "systemd")]
pub fn stopping() -> Result<()> {
    let state = [("STOPPING", "1")];
    systemd::daemon::notify(false, state.iter())?;
    Ok(())
}

#[cfg(not(feature = "systemd"))]
pub fn stopping() -> Result<()> {
    Ok(())
}

/// Notify systemd of current status
///
/// Sends `STATUS={msg}` to the notification socket.
/// The status message is displayed in `systemctl status`.
#[cfg(feature = "systemd")]
pub fn status(msg: &str) -> Result<()> {
    let state = [("STATUS", msg)];
    systemd::daemon::notify(false, state.iter())?;
    Ok(())
}

#[cfg(not(feature = "systemd"))]
pub fn status(_msg: &str) -> Result<()> {
    Ok(())
}

/// Ping the watchdog
///
/// Sends `WATCHDOG=1` to the notification socket.
/// Required for services with `WatchdogSec=` configured.
#[cfg(feature = "systemd")]
pub fn watchdog() -> Result<()> {
    let state = [("WATCHDOG", "1")];
    systemd::daemon::notify(false, state.iter())?;
    Ok(())
}

#[cfg(not(feature = "systemd"))]
pub fn watchdog() -> Result<()> {
    Ok(())
}
