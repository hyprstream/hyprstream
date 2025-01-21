use hyprstream_core::{
    cli::{commands::sql::SqlCommand, handlers::execute_sql},
    config::{self, set_tls_data, get_tls_config},
    service::FlightSqlServer,
    storage::{duckdb::DuckDbBackend, StorageBackendType},
};
use ::config::Config;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tonic::transport::Server;

const TEST_CERT: &[u8] = b"-----BEGIN CERTIFICATE-----
MIIFJTCCAw2gAwIBAgIUEFC993+G2Z0JZHLRI+ZhIiFQpMkwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI1MDExOTIxNDIzNVoXDTI2MDEx
OTIxNDIzNVowFDESMBAGA1UEAwwJbG9jYWxob3N0MIICIjANBgkqhkiG9w0BAQEF
AAOCAg8AMIICCgKCAgEA+ie6b0rBo9vahyyUVfPxKzzmQ3LMe99Fk1EZShFRLmk0
DtvdFeJTdyDn+TYvC/g1qZLc+4XPFY8m4RidqL9tMlM7wzJ4SaFI8/l0mmKDgFrz
m6QnX/CMhC/zkn9esjhbIjsSob4m7kRoFmoHzU4Lkf4TKjfZD7e0v8xPk2rpkxTV
h3sGqvehNpPZeBiy4zFT+Y49OdGmoyhleQ4c+msthCcZ0HdLUbjdXFPtABX7cxKb
ClSUWTf3QSuHN+l2fGJvc7KjAWqIbI+0nwI2B+okZvYJfNJYUPHuJeYzDPH/TT9I
EsKO6xOjxCu4ZsLrjBzt6/wtMtvPrmIDFlGf6fcNw2S2ikWZHZ+MWrRYTZ2QsjgV
QrDx0Olm98R7XkB/waWIMYZWaWfiDXRMwrjJGC5TNrJARQH4DScwLstXlXiV11Wk
NkbwBtE6HPIMGArYl/5I95/I2lqD1pvcKqtXAMbGxWFkKC4ZMHd6157kLvro4VXP
VPGtQiquaB6YQgvo8Zz75+2zmyhUcjYXUJreGqMIS63TsRFXXCTxaHVLmKiElhrK
lokHBkO4OqrkSGXsaBFdcNfyvzRGKBpmuEO3Tqmjkt9Znk+u5ztPBi+YKSia2Svp
+8VH6UUuvFD2Sk4mosjvcegjQOZZBVsctQlp6KMlozPEEJXl1VqIRIhP8PMwYjkC
AwEAAaNvMG0wHQYDVR0OBBYEFOLSoBz+VI1O12X9ZRfJfRUXACz+MB8GA1UdIwQY
MBaAFOLSoBz+VI1O12X9ZRfJfRUXACz+MA8GA1UdEwEB/wQFMAMBAf8wGgYDVR0R
BBMwEYIJbG9jYWxob3N0hwR/AAABMA0GCSqGSIb3DQEBCwUAA4ICAQDZ6e9FEzoT
nNMUQPX3GtuTrin3jQs1KZ9QlPS2eEUw3NQQ5rlJiEgfTmmk24F1lvuEQFIlv5tN
eBHKXHDqseTJDyp2TQFuXIkkqSC387qExY3vID/LVlkSRndXdWoMMA0vle6C+9GO
LVUdBOaiErS9A3wSaGLfXk8t26FRTBLJKcAGlZhPBnsKnb+dN3ecVvaXFmNICWDd
Gq1jujsOnry6zavF+FZxt4LPY9Pegv8o2YQZINdFImDhQFeqOEJgcgDOYiGAYnQx
6YqdKktiemGGv6EaJNyK/1srRBXiJG354U5iP4z04qBo9TYdNTzbMOhA6tMC/MGJ
mnMXzkkrM5bPfE9GK+BJmckho3krtNBE5Z7W6T93Adf+zahqty8MyDw70uswNXqc
XTMN8UBJVVEzyqhY2zePxGi0zC5H5VlzbFLpv/X+H5iT0JtEVqJlJThHbYpMpCr1
etWmT9AROz35qSW2GY3vhRreGVEf+6vCnEBDDkRnmcAiEL32X+J4uB7u/8d67IKW
hJC4imcViBHS2pIJSdxltOyakTYiTbYI2f7eKtwVEG/u+9Q4Exr+HjuRv83NOGqk
bF5jiPTeiA1Wy9Eu6R0+gR49yX8MF+EkuC8dpG03Ygzy5u+F9HibTkqRhCcapDjv
HMBZMYuRLIe0guDxZlW1vGepf/J5Mice/A==
-----END CERTIFICATE-----";

const TEST_KEY: &[u8] = b"-----BEGIN PRIVATE KEY-----
MIIJQgIBADANBgkqhkiG9w0BAQEFAASCCSwwggkoAgEAAoICAQD6J7pvSsGj29qH
LJRV8/ErPOZDcsx730WTURlKEVEuaTQO290V4lN3IOf5Ni8L+DWpktz7hc8Vjybh
GJ2ov20yUzvDMnhJoUjz+XSaYoOAWvObpCdf8IyEL/OSf16yOFsiOxKhvibuRGgW
agfNTguR/hMqN9kPt7S/zE+TaumTFNWHewaq96E2k9l4GLLjMVP5jj050aajKGV5
Dhz6ay2EJxnQd0tRuN1cU+0AFftzEpsKVJRZN/dBK4c36XZ8Ym9zsqMBaohsj7Sf
AjYH6iRm9gl80lhQ8e4l5jMM8f9NP0gSwo7rE6PEK7hmwuuMHO3r/C0y28+uYgMW
UZ/p9w3DZLaKRZkdn4xatFhNnZCyOBVCsPHQ6Wb3xHteQH/BpYgxhlZpZ+INdEzC
uMkYLlM2skBFAfgNJzAuy1eVeJXXVaQ2RvAG0Toc8gwYCtiX/kj3n8jaWoPWm9wq
q1cAxsbFYWQoLhkwd3rXnuQu+ujhVc9U8a1CKq5oHphCC+jxnPvn7bObKFRyNhdQ
mt4aowhLrdOxEVdcJPFodUuYqISWGsqWiQcGQ7g6quRIZexoEV1w1/K/NEYoGma4
Q7dOqaOS31meT67nO08GL5gpKJrZK+n7xUfpRS68UPZKTiaiyO9x6CNA5lkFWxy1
CWnooyWjM8QQleXVWohEiE/w8zBiOQIDAQABAoICABM0bVtZtIXPTUa7KQyzwJBC
mcV0GOeKJ7nkfCHr9CzxWfQplD63vFNtIO4Ip0I+nSUOf71mI5S6w5/8ny77OkeG
rRQCagpyEcscO8PV/BVEtkbcyoKSscD8uvEEawlY6wM04IxvECdS9GBDJePwwdHk
nQVM2hLbNkrSxUmyp6m5a969ttB1mCh735JZKBOp3/Hs5f2sPyQvx+GMMDSX+Zu3
kkNnQy6sF/98eHmdFmu6UhGQFn8GfUqhLD2CRIzOVFLgNCQ50O0vt6zM80e2hbKr
YSVWcz4Mos1BT+pGomRkbzS0f9sji/s1pY+rF4EPYAMx3ij1R+uR/f1uyQ2B4GoY
1Bcm9bCE3l3QsJQBlOAVbdMKzwGz6X1j6rSHhemHknNNXc+76wqxanqDHSbiKF47
cgITZpT8WHr/tSast/AACgxAvf81AKciPJuCO9kHZRgBfzMiuQfSWBOljudt+BxV
vkW5pribINd1pd10yAorRpLSfbcHJOsWtKu3Wxqz8u6i7GtPHOT5oHNJ+BBQAgC1
V0XAtBp1ARzmLp5I+vbejIIW36tWay4DckVJFTXzgkgyp5g0rcxBfos+z9LUJTGW
dRIjNLClj5Gb2aHR4k+zbtoqHIUn61F1fsrEgWdKoV5QzRkz0aK6vd8u57JqrBpR
JZrm0xJraL+CBP5Up8DBAoIBAQD+TK0vD949DFM+/OAt8O41cBMblt95Fotz8KXc
lJot8tBR4yw+bAhqQ31QMIfuukz60uD3P5k5vRFPq7e8XxFRlu2zv3wq0lVh37de
9JwJDO97hB5dKN3j+jP4yIDwo/EUFzyht/kVOQMF6uF4N7zA9Ax60FsFg9TcfrFQ
sfIXY74KpXhiMGLTyE+3YkRcMnlpvjA+HbBhbWmX5aLhSVR/B6beaZoR+fnwOCCw
aYwfJ5/vOMziG1nj65lmTJkL+1JCS++3FbykkIB72BGDJB3sAJmoL6xOKL6j/u9R
BJ3UguEOOZCoHqE/QMVvoGV+lxBUYy4uV6AwY7EZ4kijENXBAoIBAQD70/UQIhnJ
jVgbnYY4M7Fl36saMisCDvziXHWv0QY0Y+KxY2qPsro6RM4js5Mtaergjvozrydf
87yg4GmwdG4IEat46oUD151jgCJGObezqx9IC9fLJy946qVJrrn/5PShoTsntwhd
brjMUkgQsANpB5HbQUokCxZYmFNwERhpnAa102a63ETvVa3zZdeb0mHb+6arOkX6
TEEYtis5Mq2E+R+dZfIOmK/vcL8WLan9Ogw2uw9xT/iJfYTTEmLVoZnBVGbB7Bci
HvollW7WBDNzBMVtpgd0R8RN7Wa24iLe9P4VYLEhn03ccaF/WRZxWqGkBhDx44jy
cgVFDK4Zddp5AoIBAALnBSMAX1z7Awg5AqYDlfRuLwmlky9innzYRkxaNdhIaTBG
E38y5HWyB4Aeza5f2fkS5xZrV2hdTBFIuHQh8aSowFXI3bXvaKIRV5px2EYSK7mR
LHeLu9yaQnWYdEBK3rmH+l0uKF2hpPMwVxp0KGdbYbkVH7TUaF2L5KIzJbw2mzir
4s/cFYStSJujN3yF5vTaAtryo8y43veo208O8zPv9mubcPK7k6q2OUlKKxs/7Idi
cpQyE7iSO9H7FdQZLjsrerTwPpLyQ0UmliyVAPJsn1RYFvNda6+bfUfDcbm3NLJg
3dHNZ7G9H4PCpOXo+3q7Fw/YWC+1M5REDOgvjQECggEBAOi8/ttXOM/+8rQrBLYC
iGxnqAHA5eC0K2GlJBtGql5XBlb9U6nU+6oIlx+FwnsRTcMWQQTtVw2l/OoOHX+4
S0znz7sju6VOa6Ze8M5IX5AMkg+K6nhWEdjFu9b6RerLFpAeq8ZLsc5wGxiy3umV
UsGJ/nJNyBDBsnhU56BGHHLWgZkf9Oyz0H4FiIvPztGzQUAHNwU/CReHzA3jptTp
Elc3ytE0O97jnI5FfEUqFNX1BP68KUyHJWMkf1J3xqI8BRcZQxLseIDPck6z6cif
/1DI0xJAhNkhzrpaszhIjQPUFtN5FpvFWDdpSWGh200N/x/Rf22e5Z10ZYxoaKsd
MbkCggEAApwJeXTtd6GiZ4bcKN35vfgdYvKMgTViJywL2K2/9Q61qZo7RfbW7Y75
jl1JZI8SLATlkza/wWi11K2Utdv8ofMNQ47dt6wjgMlWZ1edo5f00nYvnRl9DoH/
MKr5K/o/GK5GSOW9t7VdKf56pAv4EOsnTzfMyBvPuTt6auf/Q7dauNWNg5hig+Pk
MDqSnJK5Njm1GC1dqrd7M6d4eM4gqMhTAUM/Kaho5h8d3DC7lUJNZafg4YB6DsU6
l3V+gb6WUXdee1JtKDHTByBVo57fBK+6xDTfGPQSbg5j1bJgOVf06wco+O4L9RSM
3OCL1SU32TOBWmM+5fFyYagvKl+IMg==
-----END PRIVATE KEY-----";

async fn start_test_server() -> (tokio::task::JoinHandle<()>, std::net::SocketAddr) {
    // Install the default crypto provider
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Start a test server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Create a test database
    let backend = StorageBackendType::DuckDb(DuckDbBackend::new_in_memory().unwrap());
    let service = FlightSqlServer::new(backend);

    // Run the server in the background
    let server_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    (server_handle, addr)
}

async fn start_tls_test_server() -> (tokio::task::JoinHandle<()>, std::net::SocketAddr) {
    // Install the default crypto provider
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Start a test server with TLS
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Create a test database
    let backend = StorageBackendType::DuckDb(DuckDbBackend::new_in_memory().unwrap());
    let service = FlightSqlServer::new(backend);

    // Create config with TLS settings
    let config = Config::builder();
    let config = set_tls_data(
        config,
        TEST_CERT,
        TEST_KEY,
        Some(TEST_CERT), // Use same cert as CA for testing
    )
    .unwrap()
    .build()
    .unwrap();

    // Get TLS config from Config
    let (identity, ca_cert) = get_tls_config(&config).unwrap();

    // Run the server in the background with TLS
    let server_handle = tokio::spawn(async move {
        println!("Starting TLS server...");
        let tls_config = tonic::transport::ServerTlsConfig::new()
            .identity(identity)
            .client_ca_root(ca_cert.unwrap());

        let server = Server::builder()
            .tls_config(tls_config)
            .unwrap()
            .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

        println!("Server built, starting to serve...");
        match server.await {
            Ok(_) => println!("Server finished successfully"),
            Err(e) => println!("Server error: {}", e),
        }
    });

    // Give the server more time to start and initialize TLS
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    // Print server address for debugging
    println!("TLS test server started on {}", addr);

    (server_handle, addr)
}

#[tokio::test]
async fn test_sql_command_basic() {
    let (server_handle, addr) = start_test_server().await;

    // Test basic SQL query
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), addr.port())),
        "CREATE TABLE test (id INTEGER);".to_string(),
        None,
        false,
    )
    .await;
    assert!(result.is_ok(), "Basic SQL query failed: {:?}", result.err());

    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_sql_command_tls() {
    // Start TLS server
    let (server_handle, addr) = start_tls_test_server().await;

    // Create config with TLS settings
    let config = Config::builder();
    let config = set_tls_data(
        config,
        TEST_CERT,
        TEST_KEY,
        Some(TEST_CERT), // Use same cert as CA for testing
    )
    .unwrap()
    .build()
    .unwrap();

    // Print TLS configuration for debugging
    println!("Testing TLS connection to {}", addr);
    println!("TLS enabled: {}", config.get_bool("tls.enabled").unwrap_or(false));
    println!("Certificate data length: {}", config.get::<Vec<u8>>("tls.cert_data").map(|d| d.len()).unwrap_or(0));
    println!("Key data length: {}", config.get::<Vec<u8>>("tls.key_data").map(|d| d.len()).unwrap_or(0));

    // Test TLS connection
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), addr.port())),
        "CREATE TABLE test (id INTEGER);".to_string(),
        Some(&config),
        true, // Enable verbose mode for more debug output
    )
    .await;
    assert!(result.is_ok(), "TLS connection failed: {:?}", result.err());

    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_sql_command_timeout() {
    // Start test server
    let (server_handle, addr) = start_test_server().await;

    // Test connection timeout (wrong port)
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), addr.port() + 1)), // Wrong port
        "SELECT 1;".to_string(),
        None,
        false,
    )
    .await;
    assert!(result.is_err(), "Expected timeout error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("timed out") || err.contains("connection refused") || err.contains("Timeout expired"),
        "Expected timeout error but got: {}",
        err
    );

    // Test query timeout
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), addr.port())),
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t WHERE n < 1000000) SELECT COUNT(*) FROM t;".to_string(), // Query that will take too long
        None,
        false,
    )
    .await;
    assert!(result.is_err(), "Expected query timeout");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("timed out"),
        "Expected timeout error but got: {}",
        err
    );

    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_sql_command_args() {
    // Create temporary files with test certificates
    let temp_dir = TempDir::new().unwrap();
    let cert_path = temp_dir.path().join("cert.pem");
    let key_path = temp_dir.path().join("key.pem");
    let ca_path = temp_dir.path().join("ca.pem");

    // Write test certificates to files
    std::fs::write(&cert_path, TEST_CERT).unwrap();
    std::fs::write(&key_path, TEST_KEY).unwrap();
    std::fs::write(&ca_path, TEST_CERT).unwrap(); // Use same cert as CA for testing

    // Test command line argument parsing
    let cmd = SqlCommand {
        host: Some("localhost:8080".to_string()),
        query: "SELECT 1".to_string(),
        tls_cert: Some(cert_path.clone()),
        tls_key: Some(key_path.clone()),
        tls_ca: Some(ca_path.clone()),
        tls_skip_verify: false,
        verbose: true,
        help: None,
    };

    // Verify command line arguments
    assert_eq!(cmd.host.as_deref(), Some("localhost:8080"));
    assert_eq!(cmd.query, "SELECT 1");
    assert!(cmd.tls_cert.is_some());
    assert!(cmd.tls_key.is_some());
    assert!(cmd.tls_ca.is_some());
    assert!(!cmd.tls_skip_verify);
    assert!(cmd.verbose);

    // Test that we can create a Config with TLS settings from command args
    let config = Config::builder();
    let config = set_tls_data(
        config,
        &std::fs::read(&cert_path).unwrap(),
        &std::fs::read(&key_path).unwrap(),
        Some(&std::fs::read(&ca_path).unwrap()),
    )
    .unwrap()
    .build()
    .unwrap();

    // Verify we can get TLS config from the Config
    let (identity, ca_cert) = get_tls_config(&config).unwrap();
    assert!(ca_cert.is_some());

    // Test that we can use the Config with execute_sql
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080)),
        cmd.query.clone(),
        Some(&config),
        cmd.verbose,
    ).await;

    // The connection will fail (no server) but we just want to verify the Config works
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Connection timed out"));
}
