//! End-to-end integration test: the translator serving a real TCP connection.
//!
//! Spins up the translator on an ephemeral loopback port with an in-memory
//! backend, then drives it from a raw `TcpStream` using the 9P codec directly
//! (version → attach → walk → lopen → read → clunk). This proves the accept
//! loop, length-prefix framing, fid table, and dispatch all work together.

use std::sync::Arc;
use std::time::Duration;

use hyprstream_9p::memory::MemoryBackend;
use hyprstream_9p::msg::{self, Response};
use hyprstream_9p::Translator;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Read one complete 9P message (length-prefixed) from `stream`.
async fn recv_message(stream: &mut TcpStream) -> Vec<u8> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).await.unwrap();
    let total = u32::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; total];
    buf[..4].copy_from_slice(&len);
    stream.read_exact(&mut buf[4..]).await.unwrap();
    buf
}

async fn rpc(stream: &mut TcpStream, req: Vec<u8>) -> Response {
    stream.write_all(&req).await.unwrap();
    let resp = recv_message(stream).await;
    let (_, parsed) = msg::parse_response(&resp).unwrap();
    parsed
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tcp_full_session() {
    let backend = MemoryBackend::default();
    backend.add_file("/greeting", b"hello from kata-translator");

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let translator = Translator::new(Arc::new(backend));
    // Spawn the accept loop; it returns on listener close, which we never
    // trigger, so unwrap it onto a background task.
    let server = tokio::spawn(async move {
        let _ = translator.serve(listener).await;
    });

    // Give the listener a beat to start accepting.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = TcpStream::connect(addr).await.unwrap();

    // Version negotiation.
    match rpc(&mut client, msg::tversion(1, 4096, "9P2000.L")).await {
        Response::Version { version, .. } => assert_eq!(version, "9P2000.L"),
        other => panic!("expected Rversion, got {other:?}"),
    }

    // Attach: fid 0 = root.
    match rpc(&mut client, msg::tattach(2, 0, u32::MAX, "user", "/")).await {
        Response::Attach { qid } => assert!(qid.is_dir(), "root qid should be a dir"),
        other => panic!("expected Rattach, got {other:?}"),
    }

    // Walk root → fid 1 (the file).
    match rpc(&mut client, msg::twalk(3, 0, 1, &["greeting"])).await {
        Response::Walk { qids } => assert_eq!(qids.len(), 1),
        other => panic!("expected Rwalk, got {other:?}"),
    }

    // Open fid 1.
    match rpc(&mut client, msg::tlopen(4, 1, 0)).await {
        Response::Lopen { iounit, .. } => assert!(iounit > 0),
        other => panic!("expected Rlopen, got {other:?}"),
    }

    // Read.
    match rpc(&mut client, msg::tread(5, 1, 0, 256)).await {
        Response::Read { data } => {
            assert_eq!(&data, b"hello from kata-translator");
        }
        other => panic!("expected Rread, got {other:?}"),
    }

    // Stat (getattr).
    match rpc(&mut client, msg::tgetattr(6, 1, 0x7ff)).await {
        Response::Getattr { size, .. } => assert_eq!(size, b"hello from kata-translator".len() as u64),
        other => panic!("expected Rgetattr, got {other:?}"),
    }

    // Clunk.
    match rpc(&mut client, msg::tclunk(7, 1)).await {
        Response::Clunk => {}
        other => panic!("expected Rclunk, got {other:?}"),
    }

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tcp_write_then_read() {
    let backend = MemoryBackend::default();
    backend.add_file("/pipe", b"");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let translator = Translator::new(Arc::new(backend));
    let server = tokio::spawn(async move {
        let _ = translator.serve(listener).await;
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = TcpStream::connect(addr).await.unwrap();
    rpc(&mut client, msg::tversion(1, 4096, "9P2000.L")).await;
    rpc(&mut client, msg::tattach(2, 0, u32::MAX, "user", "/")).await;
    rpc(&mut client, msg::twalk(3, 0, 1, &["pipe"])).await;
    rpc(&mut client, msg::tlopen(4, 1, 1 /* OWRITE */)).await;

    // Write, then clunk+reopen to read back.
    match rpc(&mut client, msg::twrite(5, 1, 0, b"kata-9p-writes-work")).await {
        Response::Write { count } => assert_eq!(count as usize, b"kata-9p-writes-work".len()),
        other => panic!("expected Rwrite, got {other:?}"),
    }
    rpc(&mut client, msg::tclunk(6, 1)).await;
    rpc(&mut client, msg::twalk(7, 0, 2, &["pipe"])).await;
    rpc(&mut client, msg::tlopen(8, 2, 0 /* OREAD */)).await;
    match rpc(&mut client, msg::tread(9, 2, 0, 256)).await {
        Response::Read { data } => assert_eq!(&data, b"kata-9p-writes-work"),
        other => panic!("expected Rread after write, got {other:?}"),
    }

    server.abort();
}
