//! `hyprstream-decision-stub` — serve the System One P0.7 stub.
//!
//! Usage:
//!   hyprstream-decision-stub [--listen ADDR] [--flight-listen ADDR]
//!
//! Defaults: facade on loopback port 8080, Flight SQL on loopback port 8081. Point the
//! stock SDKs at the facade with `TYPESAFE_BASE_URL=http://<facade address>` (and any
//! `TYPESAFE_API_KEY`).

use std::net::TcpListener;

// The bound-address stdout lines are the machine-read contract for scripts/sdk_smoke.sh.
#[allow(clippy::print_stdout)]
mod addresses {
    pub fn facade(addr: &std::net::SocketAddr) {
        println!("facade http://{addr}");
    }
    pub fn flight(addr: &std::net::SocketAddr) {
        println!("flight grpc://{addr}");
    }
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let facade_addr = arg_value(&args, "--listen")
        .unwrap_or_else(|| std::net::SocketAddr::from(([127, 0, 0, 1], 8080)).to_string());
    let flight_addr = arg_value(&args, "--flight-listen")
        .unwrap_or_else(|| std::net::SocketAddr::from(([127, 0, 0, 1], 8081)).to_string());

    let facade_listener = TcpListener::bind(&facade_addr).unwrap_or_else(|error| panic!("bind facade: {error}"));
    let (facade_addr, facade) =
        hyprstream_decision_stub::facade::serve(facade_listener)
            .await
            .unwrap_or_else(|error| panic!("serve facade: {error}"));
    addresses::facade(&facade_addr);

    let flight_listener = TcpListener::bind(&flight_addr).unwrap_or_else(|error| panic!("bind flight: {error}"));
    let (flight_addr, flight) = hyprstream_decision_stub::flight::serve(flight_listener)
        .await
        .unwrap_or_else(|error| panic!("serve flight: {error}"));
    addresses::flight(&flight_addr);

    tokio::select! {
        _ = facade => {},
        _ = flight => {},
        _ = tokio::signal::ctrl_c() => {},
    }
}
