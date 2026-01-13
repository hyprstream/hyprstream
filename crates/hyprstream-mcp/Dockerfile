FROM rust:1.70 as builder
WORKDIR /app
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/tcl-mcp-server /usr/bin/
ENTRYPOINT ["/usr/bin/tcl-mcp-server"]