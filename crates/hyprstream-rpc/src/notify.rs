//! Service notification helpers
//!
//! Wraps the systemd sd_notify protocol for service lifecycle notifications.
//! The sd_notify transport is a plain datagram to `NOTIFY_SOCKET`, so it is
//! supervisor-agnostic: `ready()` delivers `READY=1` in every build (through
//! the `systemd` crate where that feature is on, and through a `nix` datagram
//! send otherwise), and stays a no-op only when `NOTIFY_SOCKET` is unset.

use anyhow::Result;

/// Notify systemd that service is ready
///
/// Sends `READY=1` to the notification socket.
/// This should be called after the service has completed initialization
/// and is ready to handle requests.
///
/// The sd_notify protocol is supervisor-agnostic: the message is a datagram to
/// whatever `NOTIFY_SOCKET` points at, so a direct-launch supervisor
/// (`--daemon` / no-systemd fallback, #1585) that provisions a per-child
/// notification endpoint gets the same post-initialization signal systemd
/// receives. With `NOTIFY_SOCKET` unset the call stays a no-op in every build.
#[cfg(feature = "systemd")]
pub fn ready() -> Result<()> {
    let state = [("READY", "1")];
    systemd::daemon::notify(false, state.iter())?;
    Ok(())
}

/// Feature-less builds send the same datagram through `nix` so required-native
/// readiness does not silently degrade to a no-op (and a launcher notification
/// wait to a guaranteed timeout) on no-systemd builds. Abstract (`@`-prefixed)
/// socket paths — which real systemd hands out — are addressed through
/// `UnixAddr::new_abstract` (the sd_notify leading-`@` convention); every
/// other value is a pathname address.
#[cfg(not(feature = "systemd"))]
pub fn ready() -> Result<()> {
    use nix::sys::socket::{
        sendto, socket, AddressFamily, MsgFlags, SockFlag, SockType, UnixAddr,
    };
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;

    let Some(socket_path) = std::env::var_os("NOTIFY_SOCKET") else {
        return Ok(());
    };
    let bytes = socket_path.as_bytes();
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let address = {
        if bytes.first() == Some(&b'@') {
            UnixAddr::new_abstract(&bytes[1..])?
        } else {
            UnixAddr::new(std::path::Path::new(&socket_path))?
        }
    };
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let address = UnixAddr::new(std::path::Path::new(&socket_path))?;
    let fd = socket(
        AddressFamily::Unix,
        SockType::Datagram,
        SockFlag::empty(),
        None,
    )?;
    sendto(fd.as_raw_fd(), b"READY=1", &address, MsgFlags::empty())?;
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// `NOTIFY_SOCKET` is process-global state: serialize the send regressions.
    static NOTIFY_ENV_LOCK: Mutex<()> = Mutex::new(());

    const READY_PAYLOAD: &[u8] = b"READY=1";

    #[test]
    fn ready_sends_datagram_to_pathname_socket() -> anyhow::Result<()> {
        let _guard = NOTIFY_ENV_LOCK.lock();
        let path = std::env::temp_dir().join(format!(
            "hyprstream-notify-pathname-{}-{}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let receiver = std::os::unix::net::UnixDatagram::bind(&path)?;
        receiver.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;

        std::env::set_var("NOTIFY_SOCKET", &path);
        let outcome = ready();
        std::env::remove_var("NOTIFY_SOCKET");
        outcome?;

        let mut buffer = [0u8; 16];
        let (received, _) = receiver.recv_from(&mut buffer)?;
        assert_eq!(&buffer[..received], READY_PAYLOAD);
        Ok(())
    }

    /// Real systemd hands out abstract `@`-prefixed endpoints; the send side
    /// must address them with the sd_notify leading-NUL convention.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn ready_sends_datagram_to_abstract_socket() -> anyhow::Result<()> {
        use nix::cmsg_space;
        use nix::sys::socket::{
            bind, recvmsg, socket, AddressFamily, MsgFlags, SockFlag, SockType, UnixAddr,
            UnixCredentials,
        };
        use std::io::IoSliceMut;
        use std::os::fd::AsRawFd;

        let _guard = NOTIFY_ENV_LOCK.lock();
        let name = format!(
            "hyprstream-notify-abstract-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        let fd = socket(
            AddressFamily::Unix,
            SockType::Datagram,
            SockFlag::SOCK_NONBLOCK | SockFlag::SOCK_CLOEXEC,
            None,
        )?;
        bind(fd.as_raw_fd(), &UnixAddr::new_abstract(name.as_bytes())?)?;

        std::env::set_var("NOTIFY_SOCKET", format!("@{name}"));
        let outcome = ready();
        std::env::remove_var("NOTIFY_SOCKET");
        outcome?;

        let mut buffer = [0u8; 16];
        let received = {
            let mut iov = [IoSliceMut::new(&mut buffer)];
            let mut cmsg = cmsg_space!(UnixCredentials);
            let message = recvmsg::<UnixAddr>(
                fd.as_raw_fd(),
                &mut iov,
                Some(&mut cmsg),
                MsgFlags::empty(),
            )?;
            message.bytes
        };
        assert!(received > 0);
        assert_eq!(&buffer[..received], READY_PAYLOAD);
        Ok(())
    }
}
