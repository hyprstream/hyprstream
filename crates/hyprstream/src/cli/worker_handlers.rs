//! CLI handlers for worker management commands
//!
//! Provides handler functions for sandbox (Kata VM), container, and image operations.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{bail, Result};
use std::io::{self, Write};
use tracing::info;

use crate::services::worker::{
    ContainerConfig, ContainerFilter, ContainerStatsFilter,
    ContainerState, ContainerStats, KeyValue, Timestamp,
    ImageSpec, PodSandboxConfig, PodSandboxFilter,
    PodSandboxState, PodSandboxStats,
    // Request types for scoped client calls
    PodSandboxStatusRequest, CreateContainerRequest, StopContainerRequest,
    ContainerStatusRequest, ExecSyncRequest,
    ImageFilter, ImageStatusRequest, PullImageRequest,
};
use hyprstream_workers::runtime::ContainerMetadata;

use crate::services::WorkerZmqClient;

// ─────────────────────────────────────────────────────────────────────────────
// List Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker list` command
pub async fn handle_worker_list(
    client: &WorkerZmqClient,
    sandbox_filter: Option<String>,
    container_filter: Option<String>,
    containers_only: bool,
    sandboxes_only: bool,
    _state_filter: Option<String>,
    _verbose: bool,
) -> Result<()> {
    if !containers_only {
        // List sandboxes
        let filter = PodSandboxFilter {
            id: sandbox_filter.clone().unwrap_or_default(),
            ..Default::default()
        };

        let sandboxes = client.sandbox().list(&filter).await?;

        if !sandboxes.is_empty() {
            println!("SANDBOX ID                              STATE           CREATED");
            for sb in &sandboxes {
                let state = match sb.state {
                    PodSandboxState::SandboxReady => "Ready",
                    PodSandboxState::SandboxNotReady => "NotReady",
                };
                let id_short = truncate_id(&sb.id, 36);
                println!("{:<40}{:<16}{}", id_short, state, format_timestamp(sb.created_at.clone()));
            }
        } else if sandboxes_only {
            println!("No sandboxes found.");
        }
    }

    if !sandboxes_only {
        // List containers
        let filter = ContainerFilter {
            id: container_filter.unwrap_or_default(),
            pod_sandbox_id: sandbox_filter.clone().unwrap_or_default(),
            ..Default::default()
        };

        let containers = client.container().list(&filter).await?;

        if !containers.is_empty() {
            if !containers_only {
                println!(); // Separator
            }
            println!(
                "CONTAINER ID                            IMAGE                           STATE           SANDBOX"
            );
            for c in &containers {
                let state = match c.state {
                    ContainerState::ContainerCreated => "Created",
                    ContainerState::ContainerRunning => "Running",
                    ContainerState::ContainerExited => "Exited",
                    ContainerState::ContainerUnknown => "Unknown",
                };
                let id_short = truncate_id(&c.id, 36);
                let image = if c.image.image.is_empty() { "-" } else { &c.image.image };
                let image_short = truncate_id(image, 30);
                let sandbox_short = truncate_id(&c.pod_sandbox_id, 12);
                println!(
                    "{id_short:<40}{image_short:<32}{state:<16}{sandbox_short}"
                );
            }
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Run Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker run` command
#[allow(clippy::too_many_arguments)]
pub async fn handle_worker_run(
    client: &WorkerZmqClient,
    image: &str,
    command: Vec<String>,
    sandbox_id: Option<String>,
    name: Option<String>,
    env: Vec<String>,
    workdir: Option<String>,
    detach: bool,
    auto_remove: bool,
) -> Result<()> {
    info!(image = %image, "Running container");

    // 1. Ensure image is pulled
    let image_spec = ImageSpec {
        image: image.to_owned(),
        annotations: vec![],
        runtime_handler: String::new(),
    };

    // Check if image exists locally, pull if not
    if client.image().status(&ImageStatusRequest {
        image: image_spec.clone(),
        verbose: false,
    }).await.is_err() {
        println!("Pulling image {image}...");
        client.image().pull(&PullImageRequest {
            image: image_spec.clone(),
            auth: Default::default(),
            sandbox_config: PodSandboxConfig::default(),
        }).await?;
        println!("Image pulled");
    }

    // 2. Create or reuse sandbox
    let sandbox_id = if let Some(id) = sandbox_id {
        id
    } else {
        let config = PodSandboxConfig::default();
        let id = client.sandbox().run(&config).await?;
        println!("Created sandbox {}", truncate_id(&id, 12));
        id
    };

    // 3. Create container config
    let container_config = ContainerConfig {
        metadata: ContainerMetadata {
            name: name.unwrap_or_else(generate_name),
            attempt: 0,
        },
        image: ImageSpec {
            image: image_spec.image.clone(),
            annotations: Default::default(),
            runtime_handler: image_spec.runtime_handler.clone(),
        },
        command: if command.is_empty() {
            vec![]
        } else {
            command.clone()
        },
        args: vec![],
        working_dir: workdir.unwrap_or_default(),
        envs: parse_env_vars(&env),
        ..Default::default()
    };

    // 4. Create and start container
    let container_id = client
        .container().create(&CreateContainerRequest {
            pod_sandbox_id: sandbox_id.clone(),
            config: container_config,
            sandbox_config: PodSandboxConfig::default(),
        })
        .await?;

    client.container().start(&container_id).await?;

    if detach {
        println!("{container_id}");
    } else {
        println!("Container {} started", truncate_id(&container_id, 12));

        // Wait for container to exit
        loop {
            let status = client.container().status(&ContainerStatusRequest {
                container_id: container_id.clone(),
                verbose: false,
            }).await?;
            if status.status.state == ContainerState::ContainerExited {
                println!(
                    "Container exited with code {:?}",
                    status.status.exit_code
                );
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        if auto_remove {
            client.container().remove(&container_id).await?;
            println!("Container removed");
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Stop Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker stop` command
pub async fn handle_worker_stop(
    client: &WorkerZmqClient,
    id: &str,
    timeout: i64,
    force: bool,
) -> Result<()> {
    info!(id = %id, timeout = %timeout, force = %force, "Stopping");

    // Try as container first
    if client.container().status(&ContainerStatusRequest {
        container_id: id.to_owned(),
        verbose: false,
    }).await.is_ok() {
        let actual_timeout = if force { 0 } else { timeout };
        client.container().stop(&StopContainerRequest {
            container_id: id.to_owned(),
            timeout: actual_timeout,
        }).await?;
        println!("{id}");
        return Ok(());
    }

    // Try as sandbox
    client.sandbox().stop(id).await?;
    println!("{id}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Start Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker start` command
pub async fn handle_worker_start(client: &WorkerZmqClient, container_id: &str) -> Result<()> {
    info!(container_id = %container_id, "Starting container");

    client.container().start(container_id).await?;
    println!("{container_id}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Restart Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker restart` command
pub async fn handle_worker_restart(client: &WorkerZmqClient, id: &str, timeout: i64) -> Result<()> {
    info!(id = %id, "Restarting");

    // Stop then start
    handle_worker_stop(client, id, timeout, false).await?;

    // For containers, we can restart
    if client.container().status(&ContainerStatusRequest {
        container_id: id.to_owned(),
        verbose: false,
    }).await.is_ok() {
        client.container().start(id).await?;
    } else {
        // For sandboxes, need to recreate
        bail!("Sandbox restart not supported - use rm + run");
    }

    println!("{id}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Remove Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker rm` command
pub async fn handle_worker_rm(
    client: &WorkerZmqClient,
    ids: Vec<String>,
    force: bool,
) -> Result<()> {
    for id in ids {
        // Try as container first
        if let Ok(status) = client.container().status(&ContainerStatusRequest {
            container_id: id.clone(),
            verbose: false,
        }).await {
            if status.status.state == ContainerState::ContainerRunning {
                if force {
                    client.container().stop(&StopContainerRequest {
                        container_id: id.clone(),
                        timeout: 0,
                    }).await?;
                } else {
                    bail!("Container {} is running. Use -f to force.", id);
                }
            }
            client.container().remove(&id).await?;
            println!("{id}");
            continue;
        }

        // Try as sandbox
        if client.sandbox().status(&PodSandboxStatusRequest {
            pod_sandbox_id: id.clone(),
            verbose: false,
        }).await.is_ok() {
            if force {
                let _ = client.sandbox().stop(&id).await;
            }
            client.sandbox().remove(&id).await?;
            println!("{id}");
            continue;
        }

        bail!("No such container or sandbox: {}", id);
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Status Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker status` command
pub async fn handle_worker_status(client: &WorkerZmqClient, id: &str, verbose: bool) -> Result<()> {
    // Try as container
    if let Ok(status) = client.container().status(&ContainerStatusRequest {
        container_id: id.to_owned(),
        verbose,
    }).await {
        println!("Type: Container");
        println!("ID: {}", status.status.id);
        println!("State: {:?}", status.status.state);
        let image_str = if status.status.image.image.is_empty() { "-" } else { &status.status.image.image };
        println!("Image: {image_str}");
        println!("Created: {}", format_timestamp(status.status.created_at));
        if status.status.state == ContainerState::ContainerExited {
            println!("Exit Code: {}", status.status.exit_code);
        }
        if verbose {
            println!("\nInfo:");
            print_kv_info(&status.info);
        }
        return Ok(());
    }

    // Try as sandbox
    let status = client.sandbox().status(&PodSandboxStatusRequest {
        pod_sandbox_id: id.to_owned(),
        verbose,
    }).await?;
    println!("Type: Sandbox");
    println!("ID: {}", status.status.id);
    println!("State: {:?}", status.status.state);
    println!("Created: {}", format_timestamp(status.status.created_at));
    if verbose {
        println!("\nInfo:");
        print_kv_info(&status.info);
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker stats` command
pub async fn handle_worker_stats(
    client: &WorkerZmqClient,
    ids: Vec<String>,
    no_header: bool,
) -> Result<()> {
    if !no_header {
        println!(
            "{:<40}{:<12}{:<24}{:<16}",
            "ID", "CPU %", "MEM USAGE / LIMIT", "NET I/O"
        );
    }

    if ids.is_empty() {
        // Get all container stats
        let stats = client.container().list_stats(&ContainerStatsFilter::default()).await?;
        for s in stats {
            print_container_stats(&s.attributes.id, &s);
        }
    } else {
        for id in ids {
            if let Ok(s) = client.container().stats(&id).await {
                print_container_stats(&id, &s);
            } else if let Ok(s) = client.sandbox().stats(&id).await {
                print_sandbox_stats(&id, &s);
            }
        }
    }

    Ok(())
}

fn print_container_stats(id: &str, stats: &ContainerStats) {
    let cpu_pct = format!("{:.2}%", stats.cpu.usage_nano_cores as f64 / 1e9 * 100.0);
    let mem = format!(
        "{} / {}",
        format_size(stats.memory.usage_bytes),
        format_size(stats.memory.available_bytes),
    );

    println!(
        "{:<40}{:<12}{:<24}{:<16}",
        truncate_id(id, 36),
        cpu_pct,
        mem,
        "-"
    );
}

fn print_sandbox_stats(id: &str, stats: &PodSandboxStats) {
    let cpu_pct = format!(
        "{:.2}%",
        stats.linux.cpu.usage_nano_cores as f64 / 1e9 * 100.0,
    );
    let mem = format!(
        "{} / {}",
        format_size(stats.linux.memory.usage_bytes),
        format_size(stats.linux.memory.available_bytes),
    );
    let net = format!(
        "{} / {}",
        format_size(stats.linux.network.default_interface.rx_bytes),
        format_size(stats.linux.network.default_interface.tx_bytes),
    );

    println!(
        "{:<40}{:<12}{:<24}{:<16}",
        truncate_id(id, 36),
        cpu_pct,
        mem,
        net
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Exec Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker exec` command
pub async fn handle_worker_exec(
    client: &WorkerZmqClient,
    container_id: &str,
    command: Vec<String>,
    timeout: i64,
) -> Result<()> {
    if command.is_empty() {
        bail!("No command specified");
    }

    let result = client.container().exec(&ExecSyncRequest {
        container_id: container_id.to_owned(),
        cmd: command,
        timeout,
    }).await?;

    // Print stdout
    if !result.stdout.is_empty() {
        io::stdout().write_all(&result.stdout)?;
    }

    // Print stderr
    if !result.stderr.is_empty() {
        io::stderr().write_all(&result.stderr)?;
    }

    if result.exit_code != 0 {
        std::process::exit(result.exit_code);
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Terminal Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker terminal` command - attach to container I/O streams
///
/// This provides tmux-like terminal streaming with DH-authenticated E2E security:
/// - StreamHandle encapsulates DH key exchange, subscription, and HMAC verification
/// - Handles detach sequence (default: Ctrl-])
pub async fn handle_worker_terminal(
    client: &WorkerZmqClient,
    container_id: &str,
    detach_keys: &str,
) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tokio::signal;
    use hyprstream_rpc::streaming::StreamPayload;

    info!(container_id = %container_id, "Attaching to container terminal");

    // Attach RPC — DH keypair, SUB socket, and HMAC verification all managed internally.
    let mut stream_handle = client.attach(container_id).await?;

    println!(
        "Attached to container {}\n\
         Stream ID: {}\n\
         Detach with: {}",
        truncate_id(container_id, 12),
        truncate_id(stream_handle.stream_id(), 16),
        detach_keys,
    );

    // 4. Set up terminal control
    let running = Arc::new(AtomicBool::new(true));
    let cancel = stream_handle.cancel_token().clone();

    // Spawn task to handle Ctrl-C
    let running_signal = running.clone();
    let cancel_signal = cancel.clone();
    tokio::spawn(async move {
        let _ = signal::ctrl_c().await;
        running_signal.store(false, Ordering::SeqCst);
        cancel_signal.cancel();
    });

    // Parse detach key sequence
    let detach_byte = parse_detach_keys(detach_keys);

    println!("\n--- Terminal attached. Press {detach_keys} to detach ---\n");

    // 5. Main I/O loop — StreamHandle handles HMAC-verified receive
    let running_recv = running.clone();
    let cancel_recv = cancel.clone();
    let recv_handle = std::thread::spawn(move || {
        // Drop guard ensures `running` is set to false on all exit paths (normal, break, panic)
        struct RunGuard(Arc<AtomicBool>);
        impl Drop for RunGuard {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = RunGuard(running_recv);

        // Mini tokio runtime for async TMQ subscriber on this dedicated thread.
        // We use std::thread (not tokio::spawn) because the outer function blocks
        // on stdin which would starve the tokio runtime.
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("failed to build stream recv runtime: {e}");
                return;
            }
        };
        rt.block_on(async move {
            use futures::StreamExt;
            while let Some(payload) = stream_handle.next().await {
                match payload {
                    Ok(StreamPayload::Data(data)) => {
                        // Worker FD data is raw bytes (terminal output)
                        print!("{}", String::from_utf8_lossy(&data));
                        let _ = io::stdout().flush();
                    }
                    Ok(StreamPayload::Complete(_)) => {
                        println!("\n--- Stream complete ---");
                        break;
                    }
                    Ok(StreamPayload::Error(message)) => {
                        eprintln!("\n--- Stream error: {message} ---");
                        break;
                    }
                    Err(e) => {
                        tracing::warn!("Stream receive error: {}", e);
                        break;
                    }
                }
            }
        });
        cancel_recv.cancel();  // notify server promptly on all exit paths
        // _guard drops here, storing false into running
    });

    // Read stdin (stdin streaming via StreamBuilder would be implemented here)
    // For now, just wait for detach or stream completion
    use std::io::BufRead;
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        if let Ok(line) = line {
            // Check for detach sequence
            if line.as_bytes().first() == Some(&detach_byte) {
                println!("\n--- Detached ---");
                break;
            }

            // TODO: Send stdin via authenticated StreamBuilder
            // For now, stdin is not implemented (read-only terminal)
            tracing::debug!("Stdin input (not sent): {}", line);
        }
    }

    cancel.cancel();              // fires token → wakes poll_next → returns None → loop exits
    running.store(false, Ordering::SeqCst);
    let _ = recv_handle.join();

    println!("Terminal session ended.");
    Ok(())
}

/// Parse detach key sequence string to byte
fn parse_detach_keys(keys: &str) -> u8 {
    // Default detach key is Ctrl-] (0x1D, ASCII GS)
    const DEFAULT_DETACH_KEY: u8 = 0x1D;

    match keys.to_lowercase().as_str() {
        "ctrl-a" | "ctrl+a" => 0x01,
        "ctrl-b" | "ctrl+b" => 0x02,
        "ctrl-c" | "ctrl+c" => 0x03,
        "ctrl-d" | "ctrl+d" => 0x04,
        "ctrl-q" | "ctrl+q" => 0x11,
        // "ctrl-]", "ctrl+]", and unrecognized keys default to Ctrl-]
        _ => DEFAULT_DETACH_KEY,
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Image Commands
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker images list` command
pub async fn handle_images_list(client: &WorkerZmqClient, _verbose: bool) -> Result<()> {
    let images = client.image().list(&ImageFilter { image: ImageSpec::default() }).await?;

    println!(
        "{:<48}{:<16}{:<16}{:<12}",
        "REPOSITORY", "TAG", "IMAGE ID", "SIZE"
    );
    for img in images {
        for tag in &img.repo_tags {
            let (repo, tag_part) = tag.rsplit_once(':').unwrap_or((tag, "latest"));
            let id_short = truncate_id(&img.id, 12);
            let size = format_size(img.size);
            println!("{repo:<48}{tag_part:<16}{id_short:<16}{size:<12}");
        }
    }

    Ok(())
}

/// Handle `worker images pull` command
pub async fn handle_images_pull(
    client: &WorkerZmqClient,
    image: &str,
    username: Option<String>,
    password: Option<String>,
) -> Result<()> {
    println!("Pulling {image}...");

    let image_spec = ImageSpec {
        image: image.to_owned(),
        annotations: vec![],
        runtime_handler: String::new(),
    };

    let auth = match (username, password) {
        (Some(u), Some(p)) => crate::services::worker::AuthConfig {
            username: u,
            password: p,
            auth: String::new(),
            server_address: String::new(),
            identity_token: String::new(),
            registry_token: String::new(),
        },
        _ => Default::default(),
    };

    let id = client.image().pull(&PullImageRequest {
        image: image_spec,
        auth,
        sandbox_config: PodSandboxConfig::default(),
    }).await?;
    println!("Pulled {} ({})", image, truncate_id(&id, 12));

    Ok(())
}

/// Handle `worker images rm` command
pub async fn handle_images_rm(
    client: &WorkerZmqClient,
    images: Vec<String>,
    _force: bool,
) -> Result<()> {
    for img in images {
        let image_spec = ImageSpec {
            image: img.clone(),
            annotations: vec![],
            runtime_handler: String::new(),
        };

        client.image().remove(&image_spec).await?;
        println!("Deleted: {img}");
    }

    Ok(())
}

/// Handle `worker images df` command
pub async fn handle_images_df(client: &WorkerZmqClient) -> Result<()> {
    let usage = client.image().fs_info().await?;

    println!(
        "{:<16}{:<16}{:<16}{:<16}",
        "TYPE", "TOTAL", "USED", "AVAILABLE"
    );

    if usage.is_empty() {
        println!("{:<16}{:<16}{:<16}{:<16}", "Images", "-", "0B", "-");
    } else {
        for fs in &usage {
            let used = format_size(fs.used_bytes);
            println!("{:<16}{:<16}{:<16}{:<16}", "Images", "-", used, "-");
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

fn format_timestamp(ts: Timestamp) -> String {
    use chrono::{TimeZone, Utc};
    if ts.seconds == 0 && ts.nanos == 0 {
        return "-".to_owned();
    }
    Utc.timestamp_opt(ts.seconds, ts.nanos as u32)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "-".to_owned())
}

fn print_kv_info(info: &[KeyValue]) {
    for kv in info {
        println!("  {}: {}", kv.key, kv.value);
    }
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
    }
}

fn truncate_id(id: &str, max_len: usize) -> &str {
    if id.len() <= max_len {
        id
    } else {
        &id[..max_len]
    }
}

fn generate_name() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("container-{}", timestamp % 100000)
}

fn parse_env_vars(env: &[String]) -> Vec<KeyValue> {
    env.iter()
        .filter_map(|e| {
            let parts: Vec<&str> = e.splitn(2, '=').collect();
            if parts.len() == 2 {
                Some(KeyValue {
                    key: parts[0].to_owned(),
                    value: parts[1].to_owned(),
                })
            } else {
                None
            }
        })
        .collect()
}
