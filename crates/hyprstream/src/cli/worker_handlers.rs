//! CLI handlers for worker management commands
//!
//! Provides handler functions for sandbox (Kata VM), container, and image operations.

use anyhow::{bail, Result};
use std::io::{self, Write};
use tracing::info;

use hyprstream_workers::image::{AuthConfig, ImageClient, ImageSpec};
use hyprstream_workers::runtime::{
    ContainerConfig, ContainerFilter, ContainerImageSpec, ContainerState, ContainerStats,
    PodSandboxConfig, PodSandboxFilter, PodSandboxState, PodSandboxStats,
    RuntimeClient,
};

use crate::services::WorkerClient;

// ─────────────────────────────────────────────────────────────────────────────
// List Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker list` command
pub async fn handle_worker_list(
    client: &WorkerClient,
    sandbox_filter: Option<String>,
    containers_only: bool,
    sandboxes_only: bool,
    state_filter: Option<String>,
    _verbose: bool,
) -> Result<()> {
    if !containers_only {
        // List sandboxes
        let filter = sandbox_filter.as_ref().map(|id| PodSandboxFilter {
            id: Some(id.clone()),
            state: parse_sandbox_state(&state_filter),
            label_selector: Default::default(),
        });

        let sandboxes = client.list_pod_sandbox(filter.as_ref()).await?;

        if !sandboxes.is_empty() || sandboxes_only {
            println!("SANDBOX ID                              STATE           CREATED");
            for sb in &sandboxes {
                let state = match sb.state {
                    PodSandboxState::SandboxReady => "Ready",
                    PodSandboxState::SandboxNotReady => "NotReady",
                };
                let id_short = truncate_id(&sb.id, 36);
                println!("{:<40}{:<16}{}", id_short, state, sb.created_at);
            }
        }
    }

    if !sandboxes_only {
        // List containers
        let filter = sandbox_filter.as_ref().map(|id| ContainerFilter {
            id: None,
            pod_sandbox_id: Some(id.clone()),
            state: parse_container_state(&state_filter),
            label_selector: Default::default(),
        });

        let containers = client.list_containers(filter.as_ref()).await?;

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
                    "{:<40}{:<32}{:<16}{}",
                    id_short, image_short, state, sandbox_short
                );
            }
        }
    }

    if sandboxes_only {
        let sandboxes = client.list_pod_sandbox(None).await?;
        if sandboxes.is_empty() {
            println!("No sandboxes found.");
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Run Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker run` command
pub async fn handle_worker_run(
    client: &WorkerClient,
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
        image: image.to_string(),
        annotations: Default::default(),
        runtime_handler: String::new(),
    };

    // Check if image exists locally, pull if not
    if client.image_status(&image_spec, false).await.is_err() {
        println!("Pulling image {}...", image);
        client.pull_image(&image_spec, None).await?;
        println!("Image pulled");
    }

    // 2. Create or reuse sandbox
    let sandbox_id = if let Some(id) = sandbox_id {
        id
    } else {
        let config = PodSandboxConfig::default();
        let id = client.run_pod_sandbox(&config).await?;
        println!("Created sandbox {}", truncate_id(&id, 12));
        id
    };

    // 3. Create container config
    // Convert image ImageSpec to runtime ImageSpec
    let runtime_image_spec = ContainerImageSpec {
        image: image_spec.image.clone(),
        annotations: image_spec.annotations.clone(),
        runtime_handler: image_spec.runtime_handler.clone(),
    };
    let container_config = ContainerConfig {
        metadata: hyprstream_workers::runtime::ContainerMetadata {
            name: name.unwrap_or_else(generate_name),
            attempt: 0,
        },
        image: runtime_image_spec,
        command: if command.is_empty() {
            vec![]
        } else {
            command.clone()
        },
        args: vec![],
        working_dir: workdir.unwrap_or_default(),
        envs: parse_env_vars(&env),
        mounts: vec![],
        devices: vec![],
        labels: Default::default(),
        annotations: Default::default(),
        log_path: String::new(),
        stdin: false,
        stdin_once: false,
        tty: false,
        linux: None,
    };

    // 4. Create and start container
    let container_id = client
        .create_container(&sandbox_id, &container_config, &PodSandboxConfig::default())
        .await?;

    client.start_container(&container_id).await?;

    if detach {
        println!("{}", container_id);
    } else {
        println!("Container {} started", truncate_id(&container_id, 12));

        // Wait for container to exit
        loop {
            let status = client.container_status(&container_id, false).await?;
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
            client.remove_container(&container_id).await?;
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
    client: &WorkerClient,
    id: &str,
    timeout: i64,
    force: bool,
) -> Result<()> {
    info!(id = %id, timeout = %timeout, force = %force, "Stopping");

    // Try as container first
    if client.container_status(id, false).await.is_ok() {
        let actual_timeout = if force { 0 } else { timeout };
        client.stop_container(id, actual_timeout).await?;
        println!("{}", id);
        return Ok(());
    }

    // Try as sandbox
    client.stop_pod_sandbox(id).await?;
    println!("{}", id);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Start Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker start` command
pub async fn handle_worker_start(client: &WorkerClient, container_id: &str) -> Result<()> {
    info!(container_id = %container_id, "Starting container");

    client.start_container(container_id).await?;
    println!("{}", container_id);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Restart Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker restart` command
pub async fn handle_worker_restart(client: &WorkerClient, id: &str, timeout: i64) -> Result<()> {
    info!(id = %id, "Restarting");

    // Stop then start
    handle_worker_stop(client, id, timeout, false).await?;

    // For containers, we can restart
    if client.container_status(id, false).await.is_ok() {
        client.start_container(id).await?;
    } else {
        // For sandboxes, need to recreate
        bail!("Sandbox restart not supported - use rm + run");
    }

    println!("{}", id);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Remove Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker rm` command
pub async fn handle_worker_rm(
    client: &WorkerClient,
    ids: Vec<String>,
    force: bool,
) -> Result<()> {
    for id in ids {
        // Try as container first
        if let Ok(status) = client.container_status(&id, false).await {
            if status.status.state == ContainerState::ContainerRunning {
                if force {
                    client.stop_container(&id, 0).await?;
                } else {
                    bail!("Container {} is running. Use -f to force.", id);
                }
            }
            client.remove_container(&id).await?;
            println!("{}", id);
            continue;
        }

        // Try as sandbox
        if client.pod_sandbox_status(&id, false).await.is_ok() {
            if force {
                let _ = client.stop_pod_sandbox(&id).await;
            }
            client.remove_pod_sandbox(&id).await?;
            println!("{}", id);
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
pub async fn handle_worker_status(client: &WorkerClient, id: &str, verbose: bool) -> Result<()> {
    // Try as container
    if let Ok(status) = client.container_status(id, verbose).await {
        println!("Type: Container");
        println!("ID: {}", status.status.id);
        println!("State: {:?}", status.status.state);
        let image_str = if status.status.image.image.is_empty() { "-" } else { &status.status.image.image };
        println!("Image: {}", image_str);
        println!("Created: {}", status.status.created_at);
        if status.status.state == ContainerState::ContainerExited {
            println!("Exit Code: {}", status.status.exit_code);
        }
        if verbose {
            println!("\nInfo:");
            for (k, v) in &status.info {
                println!("  {}: {}", k, v);
            }
        }
        return Ok(());
    }

    // Try as sandbox
    let status = client.pod_sandbox_status(id, verbose).await?;
    println!("Type: Sandbox");
    println!("ID: {}", status.status.id);
    println!("State: {:?}", status.status.state);
    println!("Created: {}", status.status.created_at);
    if verbose {
        println!("\nInfo:");
        for (k, v) in &status.info {
            println!("  {}: {}", k, v);
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats Command
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker stats` command
pub async fn handle_worker_stats(
    client: &WorkerClient,
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
        let stats = client.list_container_stats(None).await?;
        for s in stats {
            print_container_stats(&s.attributes.id, &s);
        }
    } else {
        for id in ids {
            if let Ok(s) = client.container_stats(&id).await {
                print_container_stats(&id, &s);
            } else if let Ok(s) = client.pod_sandbox_stats(&id).await {
                print_sandbox_stats(&id, &s);
            }
        }
    }

    Ok(())
}

fn print_container_stats(id: &str, stats: &ContainerStats) {
    let cpu_pct = stats
        .cpu
        .as_ref()
        .map(|c| format!("{:.2}%", c.usage_nano_cores as f64 / 1e9 * 100.0))
        .unwrap_or_else(|| "-".to_string());

    let mem = stats
        .memory
        .as_ref()
        .map(|m| format!("{} / {}", format_size(m.usage_bytes), format_size(m.available_bytes)))
        .unwrap_or_else(|| "-".to_string());

    println!(
        "{:<40}{:<12}{:<24}{:<16}",
        truncate_id(id, 36),
        cpu_pct,
        mem,
        "-"
    );
}

fn print_sandbox_stats(id: &str, stats: &PodSandboxStats) {
    let cpu_pct = stats
        .linux
        .as_ref()
        .and_then(|l| l.cpu.as_ref())
        .map(|c| format!("{:.2}%", c.usage_nano_cores as f64 / 1e9 * 100.0))
        .unwrap_or_else(|| "-".to_string());

    let mem = stats
        .linux
        .as_ref()
        .and_then(|l| l.memory.as_ref())
        .map(|m| format!("{} / {}", format_size(m.usage_bytes), format_size(m.available_bytes)))
        .unwrap_or_else(|| "-".to_string());

    let net = stats
        .linux
        .as_ref()
        .and_then(|l| l.network.as_ref())
        .and_then(|n| n.default_interface.as_ref())
        .map(|i| format!("{} / {}", format_size(i.rx_bytes), format_size(i.tx_bytes)))
        .unwrap_or_else(|| "-".to_string());

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
    client: &WorkerClient,
    container_id: &str,
    command: Vec<String>,
    timeout: i64,
) -> Result<()> {
    if command.is_empty() {
        bail!("No command specified");
    }

    let result = client.exec_sync(container_id, &command, timeout).await?;

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
/// This provides tmux-like terminal streaming:
/// - Subscribes to stdout/stderr topics via ZMQ SUB
/// - Pushes stdin to the container via ZMQ PUSH
/// - Handles detach sequence (default: Ctrl-])
pub async fn handle_worker_terminal(
    client: &WorkerClient,
    container_id: &str,
    detach_keys: &str,
) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tokio::signal;

    info!(container_id = %container_id, "Attaching to container terminal");

    // 1. Call Attach RPC to get topics
    let attach_response = client.attach(container_id).await?;

    println!(
        "Attached to container {}\n\
         Stream endpoint: {}\n\
         stdout topic: {}\n\
         stderr topic: {}\n\
         stdin topic: {}\n\
         Detach with: {}",
        truncate_id(&attach_response.container_id, 12),
        attach_response.stream_endpoint,
        attach_response.stdout_topic,
        attach_response.stderr_topic,
        attach_response.stdin_topic,
        detach_keys,
    );

    // 2. Set up ZMQ sockets
    let context = zmq::Context::new();

    // SUB socket for stdout/stderr
    let sub_socket = context.socket(zmq::SUB)?;
    sub_socket.connect(&attach_response.stream_endpoint)?;
    sub_socket.set_subscribe(attach_response.stdout_topic.as_bytes())?;
    sub_socket.set_subscribe(attach_response.stderr_topic.as_bytes())?;

    // PUSH socket for stdin (to StreamService PULL)
    // The push endpoint is typically the same host but different port
    let push_endpoint = attach_response
        .stream_endpoint
        .replace(":5560", ":5559"); // XPUB is 5560, PULL is 5559
    let push_socket = context.socket(zmq::PUSH)?;
    push_socket.connect(&push_endpoint)?;

    // 3. Set up terminal control
    let running = Arc::new(AtomicBool::new(true));
    let running_signal = running.clone();

    // Spawn task to handle Ctrl-C
    tokio::spawn(async move {
        let _ = signal::ctrl_c().await;
        running_signal.store(false, Ordering::SeqCst);
    });

    // Parse detach key sequence
    let detach_byte = parse_detach_keys(detach_keys);

    println!("\n--- Terminal attached. Press {} to detach ---\n", detach_keys);

    // 4. Main I/O loop
    // For a proper implementation, we'd use crossterm for raw mode
    // and tokio for async I/O. This is a simplified version.

    // Spawn receiver thread for stdout/stderr
    let stdout_topic = attach_response.stdout_topic.clone();
    let stderr_topic = attach_response.stderr_topic.clone();
    let running_recv = running.clone();

    let recv_handle = std::thread::spawn(move || {
        while running_recv.load(Ordering::SeqCst) {
            // Poll with 100ms timeout
            if sub_socket.poll(zmq::POLLIN, 100).unwrap_or(0) > 0 {
                if let Ok(msg) = sub_socket.recv_bytes(0) {
                    // Message format: {topic}{streaming_capnp::StreamChunk}
                    // For now, just print the raw data after topic
                    let topic_len = if msg.starts_with(stdout_topic.as_bytes()) {
                        stdout_topic.len()
                    } else if msg.starts_with(stderr_topic.as_bytes()) {
                        stderr_topic.len()
                    } else {
                        continue;
                    };

                    // Extract payload (skip topic prefix and capnp overhead)
                    // In a full implementation, we'd parse the wire format StreamChunk
                    if msg.len() > topic_len {
                        let payload = &msg[topic_len..];
                        // Try to extract data from wire format
                        if let Some(text) = extract_stream_chunk_data(payload) {
                            print!("{}", String::from_utf8_lossy(&text));
                            let _ = io::stdout().flush();
                        }
                    }
                }
            }
        }
    });

    // Read stdin and send to container
    // In a full implementation, we'd use raw mode for character-by-character input
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

            // Send line + newline as stdin chunk
            let mut data = line.into_bytes();
            data.push(b'\n');

            // Build wire format message and send
            // For now, send raw data (full implementation would build capnp)
            let mut msg = attach_response.stdin_topic.as_bytes().to_vec();
            msg.extend_from_slice(&data);

            if let Err(e) = push_socket.send(&msg, 0) {
                eprintln!("Failed to send stdin: {}", e);
                break;
            }
        }
    }

    running.store(false, Ordering::SeqCst);
    let _ = recv_handle.join();

    println!("Terminal session ended.");
    Ok(())
}

/// Parse detach key sequence string to byte
fn parse_detach_keys(keys: &str) -> u8 {
    match keys.to_lowercase().as_str() {
        "ctrl-]" | "ctrl+]" => 0x1D, // ASCII GS (Group Separator)
        "ctrl-a" | "ctrl+a" => 0x01,
        "ctrl-b" | "ctrl+b" => 0x02,
        "ctrl-c" | "ctrl+c" => 0x03,
        "ctrl-d" | "ctrl+d" => 0x04,
        "ctrl-q" | "ctrl+q" => 0x11,
        _ => 0x1D, // Default to Ctrl-]
    }
}

/// Extract data from streaming_capnp::StreamChunk wire format message
fn extract_stream_chunk_data(payload: &[u8]) -> Option<Vec<u8>> {
    use capnp::serialize;

    let _reader = serialize::read_message(
        &mut std::io::Cursor::new(payload),
        capnp::message::ReaderOptions::default(),
    ).ok()?;

    // Try to read as streaming_capnp::StreamChunk
    // Note: This requires the capnp module to be accessible
    // For now, return the raw payload and let the caller handle it
    // In a full implementation, we'd properly deserialize the StreamChunk

    // Fallback: return raw payload if capnp parsing fails
    Some(payload.to_vec())
}

// ─────────────────────────────────────────────────────────────────────────────
// Image Commands
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `worker images list` command
pub async fn handle_images_list(client: &WorkerClient, _verbose: bool) -> Result<()> {
    let images = client.list_images(None).await?;

    println!(
        "{:<48}{:<16}{:<16}{:<12}",
        "REPOSITORY", "TAG", "IMAGE ID", "SIZE"
    );
    for img in images {
        for tag in &img.repo_tags {
            let (repo, tag_part) = tag.rsplit_once(':').unwrap_or((tag, "latest"));
            let id_short = truncate_id(&img.id, 12);
            let size = format_size(img.size);
            println!("{:<48}{:<16}{:<16}{:<12}", repo, tag_part, id_short, size);
        }
    }

    Ok(())
}

/// Handle `worker images pull` command
pub async fn handle_images_pull(
    client: &WorkerClient,
    image: &str,
    username: Option<String>,
    password: Option<String>,
) -> Result<()> {
    println!("Pulling {}...", image);

    let image_spec = ImageSpec {
        image: image.to_string(),
        annotations: Default::default(),
        runtime_handler: String::new(),
    };

    let auth = match (username, password) {
        (Some(u), Some(p)) => Some(AuthConfig {
            username: u,
            password: p,
            ..Default::default()
        }),
        _ => None,
    };

    let id = client.pull_image(&image_spec, auth.as_ref()).await?;
    println!("Pulled {} ({})", image, truncate_id(&id, 12));

    Ok(())
}

/// Handle `worker images rm` command
pub async fn handle_images_rm(
    client: &WorkerClient,
    images: Vec<String>,
    _force: bool,
) -> Result<()> {
    for image in images {
        let image_spec = ImageSpec {
            image: image.clone(),
            annotations: Default::default(),
            runtime_handler: String::new(),
        };

        client.remove_image(&image_spec).await?;
        println!("Deleted: {}", image);
    }

    Ok(())
}

/// Handle `worker images df` command
pub async fn handle_images_df(client: &WorkerClient) -> Result<()> {
    let usage = client.image_fs_info().await?;

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

fn parse_sandbox_state(filter: &Option<String>) -> Option<PodSandboxState> {
    filter.as_ref().and_then(|s| {
        match s.to_lowercase().as_str() {
            "ready" => Some(PodSandboxState::SandboxReady),
            "not-ready" | "notready" => Some(PodSandboxState::SandboxNotReady),
            _ => None,
        }
    })
}

fn parse_container_state(filter: &Option<String>) -> Option<ContainerState> {
    filter.as_ref().and_then(|s| {
        match s.to_lowercase().as_str() {
            "created" => Some(ContainerState::ContainerCreated),
            "running" => Some(ContainerState::ContainerRunning),
            "exited" => Some(ContainerState::ContainerExited),
            _ => None,
        }
    })
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
        format!("{}B", bytes)
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

fn parse_env_vars(env: &[String]) -> Vec<hyprstream_workers::runtime::KeyValue> {
    env.iter()
        .filter_map(|e| {
            let parts: Vec<&str> = e.splitn(2, '=').collect();
            if parts.len() == 2 {
                Some(hyprstream_workers::runtime::KeyValue {
                    key: parts[0].to_string(),
                    value: parts[1].to_string(),
                })
            } else {
                None
            }
        })
        .collect()
}
