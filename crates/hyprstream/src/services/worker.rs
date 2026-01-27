//! WorkerClient - ZMQ client for WorkerService
//!
//! Provides high-level async API for sandbox/container/image operations.
//! Implements both RuntimeClient and ImageClient via ZMQ REQ/REP.
//!
//! # Architecture
//!
//! Uses extension traits on `ZmqClient` following the `RegistryOps` pattern:
//! - `RuntimeOps` - CRI RuntimeClient operations
//! - `ImageOps` - CRI ImageClient operations
//!
//! All requests are Cap'n Proto serialized and signed via `ZmqClient::call()`.

use async_trait::async_trait;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::serialize_message;
use anyhow::{anyhow, Result};
use capnp::message::ReaderOptions;
use capnp::serialize;
use std::sync::Arc;

use super::{CallOptions, ZmqClient};
use hyprstream_workers::workers_capnp;

// Re-export types from hyprstream-workers for convenience
pub use hyprstream_workers::image::{
    AuthConfig, Image, ImageFilter, ImageClient, ImageSpec,
};
pub use hyprstream_workers::runtime::{
    AttachResponse, Container, ContainerAttributes, ContainerConfig, ContainerFilter,
    ContainerMetadata, ContainerState, ContainerStats, ContainerStatsFilter, ContainerStatus,
    ContainerStatusResponse, CpuUsage, ExecSyncResponse, FdStreamAuthResponse, FilesystemIdentifier,
    FilesystemUsage, KeyValue, LinuxPodSandboxStats, MemoryUsage, NetworkInterfaceUsage,
    NetworkUsage, PodIP, PodSandbox, PodSandboxAttributes, PodSandboxConfig, PodSandboxFilter,
    PodSandboxMetadata, PodSandboxNetworkStatus, PodSandboxState, PodSandboxStats,
    PodSandboxStatsFilter, PodSandboxStatus, PodSandboxStatusResponse, ProcessUsage,
    RuntimeCondition, RuntimeClient, RuntimeStatus, StatusResponse, VersionResponse,
    ContainerImageSpec,
};

/// Service name for endpoint registry
const SERVICE_NAME: &str = "worker";

// ============================================================================
// RuntimeOps Extension Trait
// ============================================================================

/// Runtime operations extension trait for `ZmqClient`.
///
/// Provides CRI RuntimeClient methods when `ZmqClient` is connected to WorkerService.
/// All requests are automatically signed via `ZmqClient::call()`.
#[async_trait]
pub trait RuntimeOps {
    /// Get runtime version information.
    async fn runtime_version(&self, version: &str) -> Result<VersionResponse>;

    /// Get runtime status.
    async fn runtime_status(&self, verbose: bool) -> Result<StatusResponse>;

    /// Create and start a pod sandbox (Kata VM).
    async fn run_pod_sandbox(&self, config: &PodSandboxConfig) -> Result<String>;

    /// Stop a pod sandbox.
    async fn stop_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()>;

    /// Remove a pod sandbox.
    async fn remove_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()>;

    /// Get pod sandbox status.
    async fn pod_sandbox_status(
        &self,
        pod_sandbox_id: &str,
        verbose: bool,
    ) -> Result<PodSandboxStatusResponse>;

    /// List pod sandboxes.
    async fn list_pod_sandbox(&self, filter: Option<&PodSandboxFilter>) -> Result<Vec<PodSandbox>>;

    /// Create a container within a pod sandbox.
    async fn create_container(
        &self,
        pod_sandbox_id: &str,
        config: &ContainerConfig,
        sandbox_config: &PodSandboxConfig,
    ) -> Result<String>;

    /// Start a container.
    async fn start_container(&self, container_id: &str) -> Result<()>;

    /// Stop a container.
    async fn stop_container(&self, container_id: &str, timeout: i64) -> Result<()>;

    /// Remove a container.
    async fn remove_container(&self, container_id: &str) -> Result<()>;

    /// Get container status.
    async fn container_status(
        &self,
        container_id: &str,
        verbose: bool,
    ) -> Result<ContainerStatusResponse>;

    /// List containers.
    async fn list_containers(&self, filter: Option<&ContainerFilter>) -> Result<Vec<Container>>;

    /// Execute a command synchronously in a container.
    async fn exec_sync(
        &self,
        container_id: &str,
        cmd: &[String],
        timeout: i64,
    ) -> Result<ExecSyncResponse>;

    /// Get pod sandbox stats.
    async fn pod_sandbox_stats(&self, pod_sandbox_id: &str) -> Result<PodSandboxStats>;

    /// List pod sandbox stats.
    async fn list_pod_sandbox_stats(
        &self,
        filter: Option<&PodSandboxStatsFilter>,
    ) -> Result<Vec<PodSandboxStats>>;

    /// Get container stats.
    async fn container_stats(&self, container_id: &str) -> Result<ContainerStats>;

    /// List container stats.
    async fn list_container_stats(
        &self,
        filter: Option<&ContainerStatsFilter>,
    ) -> Result<Vec<ContainerStats>>;

    /// Attach to container I/O streams.
    async fn attach(&self, container_id: &str) -> Result<AttachResponse>;

    /// Start FD stream (DH handshake)
    async fn start_fd_stream(&self, stream_id: &str, client_pubkey: &[u8; 32]) -> Result<FdStreamAuthResponse>;

    /// Detach from container I/O streams.
    async fn detach(&self, container_id: &str) -> Result<()>;
}

// ============================================================================
// ImageOps Extension Trait
// ============================================================================

/// Image operations extension trait for `ZmqClient`.
///
/// Provides CRI ImageClient methods when `ZmqClient` is connected to WorkerService.
#[async_trait]
pub trait ImageOps {
    /// List images.
    async fn list_images(&self, filter: Option<&ImageFilter>) -> Result<Vec<Image>>;

    /// Get image status.
    async fn image_status(
        &self,
        image: &ImageSpec,
        verbose: bool,
    ) -> Result<hyprstream_workers::image::ImageStatusResponse>;

    /// Pull an image.
    async fn pull_image(&self, image: &ImageSpec, auth: Option<&AuthConfig>) -> Result<String>;

    /// Remove an image.
    async fn remove_image(&self, image: &ImageSpec) -> Result<()>;

    /// Get filesystem info.
    async fn image_fs_info(&self) -> Result<Vec<hyprstream_workers::image::FilesystemUsage>>;
}

// ============================================================================
// RuntimeOps Implementation
// ============================================================================

#[async_trait]
impl RuntimeOps for ZmqClient {
    async fn runtime_version(&self, version: &str) -> Result<VersionResponse> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            req.set_version(version);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_version_response(&response)
    }

    async fn runtime_status(&self, verbose: bool) -> Result<StatusResponse> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let mut status = req.init_status();
            status.set_verbose(verbose);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_status_response(&response)
    }

    async fn run_pod_sandbox(&self, config: &PodSandboxConfig) -> Result<String> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let sandbox_config = req.init_run_pod_sandbox();
            build_pod_sandbox_config(sandbox_config, config);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_sandbox_id_response(&response)
    }

    async fn stop_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            req.set_stop_pod_sandbox(pod_sandbox_id);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_success_response(&response)
    }

    async fn remove_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            req.set_remove_pod_sandbox(pod_sandbox_id);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_success_response(&response)
    }

    async fn pod_sandbox_status(
        &self,
        pod_sandbox_id: &str,
        verbose: bool,
    ) -> Result<PodSandboxStatusResponse> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let mut status_req = req.init_pod_sandbox_status();
            status_req.set_pod_sandbox_id(pod_sandbox_id);
            status_req.set_verbose(verbose);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_sandbox_status_response(&response)
    }

    async fn list_pod_sandbox(&self, filter: Option<&PodSandboxFilter>) -> Result<Vec<PodSandbox>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let filter_builder = req.init_list_pod_sandbox();
            if let Some(f) = filter {
                build_pod_sandbox_filter(filter_builder, f);
            }
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_sandboxes_response(&response)
    }

    async fn create_container(
        &self,
        pod_sandbox_id: &str,
        config: &ContainerConfig,
        sandbox_config: &PodSandboxConfig,
    ) -> Result<String> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let mut create_req = req.init_create_container();
            create_req.set_pod_sandbox_id(pod_sandbox_id);
            build_container_config(create_req.reborrow().init_config(), config);
            build_pod_sandbox_config(create_req.init_sandbox_config(), sandbox_config);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_container_id_response(&response)
    }

    async fn start_container(&self, container_id: &str) -> Result<()> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            req.set_start_container(container_id);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_success_response(&response)
    }

    async fn stop_container(&self, container_id: &str, timeout: i64) -> Result<()> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let mut stop_req = req.init_stop_container();
            stop_req.set_container_id(container_id);
            stop_req.set_timeout(timeout);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_success_response(&response)
    }

    async fn remove_container(&self, container_id: &str) -> Result<()> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            req.set_remove_container(container_id);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_success_response(&response)
    }

    async fn container_status(
        &self,
        container_id: &str,
        verbose: bool,
    ) -> Result<ContainerStatusResponse> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let mut status_req = req.init_container_status();
            status_req.set_container_id(container_id);
            status_req.set_verbose(verbose);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_container_status_response(&response)
    }

    async fn list_containers(&self, filter: Option<&ContainerFilter>) -> Result<Vec<Container>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let filter_builder = req.init_list_containers();
            if let Some(f) = filter {
                build_container_filter(filter_builder, f);
            }
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_containers_response(&response)
    }

    async fn exec_sync(
        &self,
        container_id: &str,
        cmd: &[String],
        timeout: i64,
    ) -> Result<ExecSyncResponse> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let mut exec_req = req.init_exec_sync();
            exec_req.set_container_id(container_id);
            let mut cmd_builder = exec_req.reborrow().init_cmd(cmd.len() as u32);
            for (i, c) in cmd.iter().enumerate() {
                cmd_builder.set(i as u32, c);
            }
            exec_req.set_timeout(timeout);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_exec_response(&response)
    }

    async fn pod_sandbox_stats(&self, pod_sandbox_id: &str) -> Result<PodSandboxStats> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            req.set_pod_sandbox_stats(pod_sandbox_id);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_sandbox_stats_response(&response)
    }

    async fn list_pod_sandbox_stats(
        &self,
        filter: Option<&PodSandboxStatsFilter>,
    ) -> Result<Vec<PodSandboxStats>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let filter_builder = req.init_list_pod_sandbox_stats();
            if let Some(f) = filter {
                build_pod_sandbox_stats_filter(filter_builder, f);
            }
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_sandbox_stats_list_response(&response)
    }

    async fn container_stats(&self, container_id: &str) -> Result<ContainerStats> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            req.set_container_stats(container_id);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_container_stats_response(&response)
    }

    async fn list_container_stats(
        &self,
        filter: Option<&ContainerStatsFilter>,
    ) -> Result<Vec<ContainerStats>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let filter_builder = req.init_list_container_stats();
            if let Some(f) = filter {
                build_container_stats_filter(filter_builder, f);
            }
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_container_stats_list_response(&response)
    }

    async fn attach(&self, container_id: &str) -> Result<AttachResponse> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let mut attach_req = req.init_attach();
            attach_req.set_container_id(container_id);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_attach_response(&response)
    }

    async fn start_fd_stream(&self, stream_id: &str, client_pubkey: &[u8; 32]) -> Result<FdStreamAuthResponse> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            let mut start_req = req.init_start_fd_stream();
            start_req.set_stream_id(stream_id);
            start_req.set_client_pubkey(client_pubkey);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_fd_stream_auth_response(&response)
    }

    async fn detach(&self, container_id: &str) -> Result<()> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::runtime_request::Builder>();
            req.set_id(id);
            req.set_detach(container_id);
        })?;
        let response = self.call(payload, CallOptions::default()).await?;
        parse_success_response(&response)
    }
}

// ============================================================================
// ImageOps Implementation
// ============================================================================

/// Prefix for ImageRequest messages (to distinguish from RuntimeRequest)
const IMAGE_REQUEST_PREFIX: u8 = 0x01;

/// Wrap payload with image request prefix
fn prefix_image_request(payload: Vec<u8>) -> Vec<u8> {
    let mut prefixed = Vec::with_capacity(1 + payload.len());
    prefixed.push(IMAGE_REQUEST_PREFIX);
    prefixed.extend(payload);
    prefixed
}

#[async_trait]
impl ImageOps for ZmqClient {
    async fn list_images(&self, filter: Option<&ImageFilter>) -> Result<Vec<Image>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::image_request::Builder>();
            req.set_id(id);
            let filter_builder = req.init_list_images();
            if let Some(f) = filter {
                build_image_filter(filter_builder, f);
            }
        })?;
        let response = self.call(prefix_image_request(payload), CallOptions::default()).await?;
        parse_images_response(&response)
    }

    async fn image_status(
        &self,
        image: &ImageSpec,
        verbose: bool,
    ) -> Result<hyprstream_workers::image::ImageStatusResponse> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::image_request::Builder>();
            req.set_id(id);
            let mut status_req = req.init_image_status();
            build_image_spec(status_req.reborrow().init_image(), image);
            status_req.set_verbose(verbose);
        })?;
        let response = self.call(prefix_image_request(payload), CallOptions::default()).await?;
        parse_image_status_response(&response)
    }

    async fn pull_image(&self, image: &ImageSpec, auth: Option<&AuthConfig>) -> Result<String> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::image_request::Builder>();
            req.set_id(id);
            let mut pull_req = req.init_pull_image();
            build_image_spec(pull_req.reborrow().init_image(), image);
            if let Some(a) = auth {
                build_auth_config(pull_req.init_auth(), a);
            }
        })?;
        let response = self.call(prefix_image_request(payload), CallOptions::default()).await?;
        parse_image_ref_response(&response)
    }

    async fn remove_image(&self, image: &ImageSpec) -> Result<()> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::image_request::Builder>();
            req.set_id(id);
            let image_spec = req.init_remove_image();
            build_image_spec(image_spec, image);
        })?;
        let response = self.call(prefix_image_request(payload), CallOptions::default()).await?;
        parse_image_success_response(&response)
    }

    async fn image_fs_info(&self) -> Result<Vec<hyprstream_workers::image::FilesystemUsage>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<workers_capnp::image_request::Builder>();
            req.set_id(id);
            req.set_image_fs_info(());
        })?;
        let response = self.call(prefix_image_request(payload), CallOptions::default()).await?;
        parse_fs_info_response(&response)
    }
}

// ============================================================================
// Response Parsing Helpers - Runtime
// ============================================================================

/// Generic response parser for RuntimeResponse.
fn parse_runtime_response<T, F>(response: &[u8], extractor: F) -> Result<T>
where
    F: FnOnce(workers_capnp::runtime_response::Reader) -> Result<T>,
{
    let reader = serialize::read_message(response, ReaderOptions::new())?;
    let resp = reader.get_root::<workers_capnp::runtime_response::Reader>()?;

    use workers_capnp::runtime_response::Which;
    if let Which::Error(err) = resp.which()? {
        let err = err?;
        return Err(anyhow!("{}", err.get_message()?.to_str()?));
    }

    extractor(resp)
}

fn parse_version_response(response: &[u8]) -> Result<VersionResponse> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::Version(v) => {
                let v = v?;
                Ok(VersionResponse {
                    version: v.get_version()?.to_str()?.to_owned(),
                    runtime_name: v.get_runtime_name()?.to_str()?.to_owned(),
                    runtime_version: v.get_runtime_version()?.to_str()?.to_owned(),
                    runtime_api_version: v.get_runtime_api_version()?.to_str()?.to_owned(),
                })
            }
            _ => Err(anyhow!("Expected version response")),
        }
    })
}

fn parse_status_response(response: &[u8]) -> Result<StatusResponse> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::Status(s) => {
                let s = s?;
                let conditions = s.get_conditions()?;
                let mut result_conditions = Vec::with_capacity(conditions.len() as usize);
                for c in conditions.iter() {
                    result_conditions.push(RuntimeCondition {
                        condition_type: c.get_condition_type()?.to_str()?.to_owned(),
                        status: c.get_status(),
                        reason: c.get_reason()?.to_str()?.to_owned(),
                        message: c.get_message()?.to_str()?.to_owned(),
                    });
                }
                Ok(StatusResponse {
                    status: RuntimeStatus {
                        conditions: result_conditions,
                    },
                    info: std::collections::HashMap::new(),
                })
            }
            _ => Err(anyhow!("Expected status response")),
        }
    })
}

fn parse_success_response(response: &[u8]) -> Result<()> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::Success(()) => Ok(()),
            _ => Err(anyhow!("Expected success response")),
        }
    })
}

fn parse_attach_response(response: &[u8]) -> Result<AttachResponse> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::AttachResponse(attach) => {
                let attach = attach?;
                let server_pubkey_data = attach.get_server_pubkey()?;
                let mut server_pubkey = [0u8; 32];
                if server_pubkey_data.len() == 32 {
                    server_pubkey.copy_from_slice(server_pubkey_data);
                }
                Ok(AttachResponse {
                    container_id: attach.get_container_id()?.to_str()?.to_owned(),
                    stream_id: attach.get_stream_id()?.to_str()?.to_owned(),
                    stream_endpoint: attach.get_stream_endpoint()?.to_str()?.to_owned(),
                    server_pubkey,
                })
            }
            _ => Err(anyhow!("Expected attach_response")),
        }
    })
}

fn parse_fd_stream_auth_response(response: &[u8]) -> Result<FdStreamAuthResponse> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::FdStreamAuthorized(auth) => {
                let auth = auth?;
                Ok(FdStreamAuthResponse {
                    stream_id: auth.get_stream_id()?.to_str()?.to_owned(),
                    authorized: auth.get_authorized(),
                })
            }
            _ => Err(anyhow!("Expected fd_stream_authorized response")),
        }
    })
}

fn parse_sandbox_id_response(response: &[u8]) -> Result<String> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::SandboxId(id) => Ok(id?.to_str()?.to_owned()),
            _ => Err(anyhow!("Expected sandbox_id response")),
        }
    })
}

fn parse_container_id_response(response: &[u8]) -> Result<String> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::ContainerId(id) => Ok(id?.to_str()?.to_owned()),
            _ => Err(anyhow!("Expected container_id response")),
        }
    })
}

fn parse_sandbox_status_response(response: &[u8]) -> Result<PodSandboxStatusResponse> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::SandboxStatus(s) => {
                let s = s?;
                let status = s.get_status()?;
                Ok(PodSandboxStatusResponse {
                    status: parse_pod_sandbox_status(status)?,
                    info: parse_key_values_to_map(s.get_info()?),
                })
            }
            _ => Err(anyhow!("Expected sandbox_status response")),
        }
    })
}

fn parse_sandboxes_response(response: &[u8]) -> Result<Vec<PodSandbox>> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::Sandboxes(sandboxes) => {
                let sandboxes = sandboxes?;
                let mut result = Vec::with_capacity(sandboxes.len() as usize);
                for sb in sandboxes.iter() {
                    result.push(parse_pod_sandbox_info(sb)?);
                }
                Ok(result)
            }
            _ => Err(anyhow!("Expected sandboxes response")),
        }
    })
}

fn parse_container_status_response(response: &[u8]) -> Result<ContainerStatusResponse> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::ContainerStatus(s) => {
                let s = s?;
                Ok(ContainerStatusResponse {
                    status: parse_container_status(s.get_status()?)?,
                    info: parse_key_values_to_map(s.get_info()?),
                })
            }
            _ => Err(anyhow!("Expected container_status response")),
        }
    })
}

fn parse_containers_response(response: &[u8]) -> Result<Vec<Container>> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::Containers(containers) => {
                let containers = containers?;
                let mut result = Vec::with_capacity(containers.len() as usize);
                for c in containers.iter() {
                    result.push(parse_container_info(c)?);
                }
                Ok(result)
            }
            _ => Err(anyhow!("Expected containers response")),
        }
    })
}

fn parse_exec_response(response: &[u8]) -> Result<ExecSyncResponse> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::ExecResult(r) => {
                let r = r?;
                Ok(ExecSyncResponse {
                    stdout: r.get_stdout()?.to_vec(),
                    stderr: r.get_stderr()?.to_vec(),
                    exit_code: r.get_exit_code(),
                })
            }
            _ => Err(anyhow!("Expected exec_result response")),
        }
    })
}

fn parse_sandbox_stats_response(response: &[u8]) -> Result<PodSandboxStats> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::SandboxStats(s) => parse_pod_sandbox_stats(s?),
            _ => Err(anyhow!("Expected sandbox_stats response")),
        }
    })
}

fn parse_sandbox_stats_list_response(response: &[u8]) -> Result<Vec<PodSandboxStats>> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::SandboxStatsList(list) => {
                let list = list?;
                let mut result = Vec::with_capacity(list.len() as usize);
                for s in list.iter() {
                    result.push(parse_pod_sandbox_stats(s)?);
                }
                Ok(result)
            }
            _ => Err(anyhow!("Expected sandbox_stats_list response")),
        }
    })
}

fn parse_container_stats_response(response: &[u8]) -> Result<ContainerStats> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::ContainerStatsResult(s) => parse_container_stats(s?),
            _ => Err(anyhow!("Expected container_stats response")),
        }
    })
}

fn parse_container_stats_list_response(response: &[u8]) -> Result<Vec<ContainerStats>> {
    parse_runtime_response(response, |resp| {
        use workers_capnp::runtime_response::Which;
        match resp.which()? {
            Which::ContainerStatsList(list) => {
                let list = list?;
                let mut result = Vec::with_capacity(list.len() as usize);
                for s in list.iter() {
                    result.push(parse_container_stats(s)?);
                }
                Ok(result)
            }
            _ => Err(anyhow!("Expected container_stats_list response")),
        }
    })
}

// ============================================================================
// Response Parsing Helpers - Image
// ============================================================================

/// Generic response parser for ImageResponse.
fn parse_image_response<T, F>(response: &[u8], extractor: F) -> Result<T>
where
    F: FnOnce(workers_capnp::image_response::Reader) -> Result<T>,
{
    let reader = serialize::read_message(response, ReaderOptions::new())?;
    let resp = reader.get_root::<workers_capnp::image_response::Reader>()?;

    use workers_capnp::image_response::Which;
    if let Which::Error(err) = resp.which()? {
        let err = err?;
        return Err(anyhow!("{}", err.get_message()?.to_str()?));
    }

    extractor(resp)
}

fn parse_images_response(response: &[u8]) -> Result<Vec<Image>> {
    parse_image_response(response, |resp| {
        use workers_capnp::image_response::Which;
        match resp.which()? {
            Which::Images(images) => {
                let images = images?;
                let mut result = Vec::with_capacity(images.len() as usize);
                for img in images.iter() {
                    result.push(parse_image_info(img)?);
                }
                Ok(result)
            }
            _ => Err(anyhow!("Expected images response")),
        }
    })
}

fn parse_image_status_response(
    response: &[u8],
) -> Result<hyprstream_workers::image::ImageStatusResponse> {
    parse_image_response(response, |resp| {
        use workers_capnp::image_response::Which;
        match resp.which()? {
            Which::ImageStatus(s) => {
                let s = s?;
                let image = if s.has_image() {
                    Some(parse_image_info(s.get_image()?)?)
                } else {
                    None
                };
                Ok(hyprstream_workers::image::ImageStatusResponse {
                    image,
                    info: parse_key_values_to_map(s.get_info()?),
                })
            }
            _ => Err(anyhow!("Expected image_status response")),
        }
    })
}

fn parse_image_ref_response(response: &[u8]) -> Result<String> {
    parse_image_response(response, |resp| {
        use workers_capnp::image_response::Which;
        match resp.which()? {
            Which::ImageRef(r) => Ok(r?.to_str()?.to_owned()),
            _ => Err(anyhow!("Expected image_ref response")),
        }
    })
}

fn parse_image_success_response(response: &[u8]) -> Result<()> {
    parse_image_response(response, |resp| {
        use workers_capnp::image_response::Which;
        match resp.which()? {
            Which::Success(()) => Ok(()),
            _ => Err(anyhow!("Expected success response")),
        }
    })
}

fn parse_fs_info_response(response: &[u8]) -> Result<Vec<hyprstream_workers::image::FilesystemUsage>> {
    parse_image_response(response, |resp| {
        use workers_capnp::image_response::Which;
        match resp.which()? {
            Which::FsInfo(info) => {
                let info = info?;
                let mut result = Vec::with_capacity(info.len() as usize);
                for fs in info.iter() {
                    result.push(hyprstream_workers::image::FilesystemUsage {
                        timestamp: fs.get_timestamp(),
                        fs_id: hyprstream_workers::image::FilesystemIdentifier {
                            mountpoint: fs.get_fs_id()?.get_mountpoint()?.to_str()?.to_owned(),
                        },
                        used_bytes: fs.get_used_bytes(),
                        inodes_used: fs.get_inodes_used(),
                    });
                }
                Ok(result)
            }
            _ => Err(anyhow!("Expected fs_info response")),
        }
    })
}

// ============================================================================
// Type Parsing Helpers
// ============================================================================

fn parse_key_values_to_map(
    kv_list: capnp::struct_list::Reader<workers_capnp::key_value::Owned>,
) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for kv in kv_list.iter() {
        if let (Ok(k), Ok(v)) = (kv.get_key(), kv.get_value()) {
            if let (Ok(key), Ok(val)) = (k.to_str(), v.to_str()) {
                map.insert(key.to_owned(), val.to_owned());
            }
        }
    }
    map
}

fn parse_pod_sandbox_state(state: workers_capnp::PodSandboxState) -> PodSandboxState {
    match state {
        workers_capnp::PodSandboxState::SandboxReady => PodSandboxState::SandboxReady,
        workers_capnp::PodSandboxState::SandboxNotReady => PodSandboxState::SandboxNotReady,
    }
}

fn parse_container_state(state: workers_capnp::ContainerState) -> ContainerState {
    match state {
        workers_capnp::ContainerState::ContainerCreated => ContainerState::ContainerCreated,
        workers_capnp::ContainerState::ContainerRunning => ContainerState::ContainerRunning,
        workers_capnp::ContainerState::ContainerExited => ContainerState::ContainerExited,
        workers_capnp::ContainerState::ContainerUnknown => ContainerState::ContainerUnknown,
    }
}

fn parse_pod_sandbox_metadata(
    meta: workers_capnp::pod_sandbox_metadata::Reader,
) -> Result<PodSandboxMetadata> {
    Ok(PodSandboxMetadata {
        name: meta.get_name()?.to_str()?.to_owned(),
        uid: meta.get_uid()?.to_str()?.to_owned(),
        namespace: meta.get_namespace()?.to_str()?.to_owned(),
        attempt: meta.get_attempt(),
    })
}

fn parse_pod_sandbox_status(
    status: workers_capnp::pod_sandbox_status::Reader,
) -> Result<PodSandboxStatus> {
    use chrono::{TimeZone, Utc};

    Ok(PodSandboxStatus {
        id: status.get_id()?.to_str()?.to_owned(),
        metadata: parse_pod_sandbox_metadata(status.get_metadata()?)?,
        state: parse_pod_sandbox_state(status.get_state()?),
        created_at: Utc.timestamp_nanos(status.get_created_at()),
        network: if status.has_network() {
            let net = status.get_network()?;
            Some(PodSandboxNetworkStatus {
                ip: net.get_ip()?.to_str()?.to_owned(),
                additional_ips: {
                    let ips = net.get_additional_ips()?;
                    let mut result = Vec::with_capacity(ips.len() as usize);
                    for ip in ips.iter() {
                        result.push(PodIP {
                            ip: ip.get_ip()?.to_str()?.to_owned(),
                        });
                    }
                    result
                },
            })
        } else {
            None
        },
        linux: None,
        labels: parse_key_values_to_map(status.get_labels()?),
        annotations: parse_key_values_to_map(status.get_annotations()?),
        runtime_handler: status.get_runtime_handler()?.to_str()?.to_owned(),
    })
}

fn parse_pod_sandbox_info(
    info: workers_capnp::pod_sandbox_info::Reader,
) -> Result<PodSandbox> {
    use chrono::{TimeZone, Utc};

    Ok(PodSandbox::from_info(
        info.get_id()?.to_str()?.to_owned(),
        parse_pod_sandbox_metadata(info.get_metadata()?)?,
        parse_pod_sandbox_state(info.get_state()?),
        Utc.timestamp_nanos(info.get_created_at()),
        parse_key_values_to_map(info.get_labels()?),
        parse_key_values_to_map(info.get_annotations()?),
        info.get_runtime_handler()?.to_str()?.to_owned(),
    ))
}

fn parse_container_metadata(
    meta: workers_capnp::container_metadata::Reader,
) -> Result<ContainerMetadata> {
    Ok(ContainerMetadata {
        name: meta.get_name()?.to_str()?.to_owned(),
        attempt: meta.get_attempt(),
    })
}

fn parse_image_spec(spec: workers_capnp::image_spec::Reader) -> Result<ImageSpec> {
    Ok(ImageSpec {
        image: spec.get_image()?.to_str()?.to_owned(),
        annotations: parse_key_values_to_map(spec.get_annotations()?),
        runtime_handler: spec.get_runtime_handler()?.to_str()?.to_owned(),
    })
}

fn parse_container_image_spec(spec: workers_capnp::image_spec::Reader) -> Result<ContainerImageSpec> {
    Ok(ContainerImageSpec {
        image: spec.get_image()?.to_str()?.to_owned(),
        annotations: parse_key_values_to_map(spec.get_annotations()?),
        runtime_handler: spec.get_runtime_handler()?.to_str()?.to_owned(),
    })
}

fn parse_container_status(
    status: workers_capnp::container_status::Reader,
) -> Result<ContainerStatus> {
    use chrono::{TimeZone, Utc};

    let started_at_ns = status.get_started_at();
    let finished_at_ns = status.get_finished_at();

    Ok(ContainerStatus {
        id: status.get_id()?.to_str()?.to_owned(),
        metadata: parse_container_metadata(status.get_metadata()?)?,
        state: parse_container_state(status.get_state()?),
        created_at: Utc.timestamp_nanos(status.get_created_at()),
        started_at: if started_at_ns > 0 {
            Some(Utc.timestamp_nanos(started_at_ns))
        } else {
            None
        },
        finished_at: if finished_at_ns > 0 {
            Some(Utc.timestamp_nanos(finished_at_ns))
        } else {
            None
        },
        exit_code: status.get_exit_code(),
        image: parse_container_image_spec(status.get_image()?)?,
        image_ref: status.get_image_ref()?.to_str()?.to_owned(),
        reason: status.get_reason()?.to_str()?.to_owned(),
        message: status.get_message()?.to_str()?.to_owned(),
        labels: parse_key_values_to_map(status.get_labels()?),
        annotations: parse_key_values_to_map(status.get_annotations()?),
        mounts: vec![], // TODO: parse mounts if needed
        log_path: status.get_log_path()?.to_str()?.to_owned(),
    })
}

fn parse_container_info(info: workers_capnp::container_info::Reader) -> Result<Container> {
    use chrono::{TimeZone, Utc};

    Ok(Container::from_info(
        info.get_id()?.to_str()?.to_owned(),
        info.get_pod_sandbox_id()?.to_str()?.to_owned(),
        parse_container_metadata(info.get_metadata()?)?,
        parse_container_image_spec(info.get_image()?)?,
        parse_container_state(info.get_state()?),
        Utc.timestamp_nanos(info.get_created_at()),
        parse_key_values_to_map(info.get_labels()?),
        parse_key_values_to_map(info.get_annotations()?),
    ))
}

fn parse_pod_sandbox_stats(stats: workers_capnp::pod_sandbox_stats::Reader) -> Result<PodSandboxStats> {
    let attrs = stats.get_attributes()?;
    Ok(PodSandboxStats {
        attributes: PodSandboxAttributes {
            id: attrs.get_id()?.to_str()?.to_owned(),
            metadata: parse_pod_sandbox_metadata(attrs.get_metadata()?)?,
            labels: parse_key_values_to_map(attrs.get_labels()?),
            annotations: parse_key_values_to_map(attrs.get_annotations()?),
        },
        linux: if stats.has_linux() {
            let linux = stats.get_linux()?;
            Some(LinuxPodSandboxStats {
                cpu: Some(parse_cpu_usage(linux.get_cpu()?)?),
                memory: Some(parse_memory_usage(linux.get_memory()?)?),
                network: Some(parse_network_usage(linux.get_network()?)?),
                process: Some(parse_process_usage(linux.get_process()?)?),
            })
        } else {
            None
        },
    })
}

fn parse_container_stats(stats: workers_capnp::container_stats::Reader) -> Result<ContainerStats> {
    let attrs = stats.get_attributes()?;
    Ok(ContainerStats {
        attributes: ContainerAttributes {
            id: attrs.get_id()?.to_str()?.to_owned(),
            metadata: parse_container_metadata(attrs.get_metadata()?)?,
            labels: parse_key_values_to_map(attrs.get_labels()?),
            annotations: parse_key_values_to_map(attrs.get_annotations()?),
        },
        cpu: Some(parse_cpu_usage(stats.get_cpu()?)?),
        memory: Some(parse_memory_usage(stats.get_memory()?)?),
        writable_layer: Some(parse_filesystem_usage(stats.get_writable_layer()?)?),
    })
}

fn parse_cpu_usage(cpu: workers_capnp::cpu_usage::Reader) -> Result<CpuUsage> {
    Ok(CpuUsage {
        timestamp: cpu.get_timestamp(),
        usage_core_nano_seconds: cpu.get_usage_core_nano_seconds(),
        usage_nano_cores: cpu.get_usage_nano_cores(),
    })
}

fn parse_memory_usage(mem: workers_capnp::memory_usage::Reader) -> Result<MemoryUsage> {
    Ok(MemoryUsage {
        timestamp: mem.get_timestamp(),
        working_set_bytes: mem.get_working_set_bytes(),
        available_bytes: mem.get_available_bytes(),
        usage_bytes: mem.get_usage_bytes(),
        rss_bytes: mem.get_rss_bytes(),
        page_faults: mem.get_page_faults(),
        major_page_faults: mem.get_major_page_faults(),
    })
}

fn parse_network_usage(net: workers_capnp::network_usage::Reader) -> Result<NetworkUsage> {
    Ok(NetworkUsage {
        timestamp: net.get_timestamp(),
        default_interface: Some(parse_network_interface_usage(net.get_default_interface()?)?),
        interfaces: {
            let ifaces = net.get_interfaces()?;
            let mut result = Vec::with_capacity(ifaces.len() as usize);
            for iface in ifaces.iter() {
                result.push(parse_network_interface_usage(iface)?);
            }
            result
        },
    })
}

fn parse_network_interface_usage(
    iface: workers_capnp::network_interface_usage::Reader,
) -> Result<NetworkInterfaceUsage> {
    Ok(NetworkInterfaceUsage {
        name: iface.get_name()?.to_str()?.to_owned(),
        rx_bytes: iface.get_rx_bytes(),
        tx_bytes: iface.get_tx_bytes(),
        rx_errors: iface.get_rx_errors(),
        tx_errors: iface.get_tx_errors(),
    })
}

fn parse_process_usage(proc: workers_capnp::process_usage::Reader) -> Result<ProcessUsage> {
    Ok(ProcessUsage {
        timestamp: proc.get_timestamp(),
        process_count: proc.get_process_count(),
    })
}

fn parse_filesystem_usage(fs: workers_capnp::filesystem_usage::Reader) -> Result<FilesystemUsage> {
    Ok(FilesystemUsage {
        timestamp: fs.get_timestamp(),
        fs_id: Some(FilesystemIdentifier {
            mountpoint: fs.get_fs_id()?.get_mountpoint()?.to_str()?.to_owned(),
        }),
        used_bytes: fs.get_used_bytes(),
        inodes_used: fs.get_inodes_used(),
    })
}

fn parse_image_info(img: workers_capnp::image_info::Reader) -> Result<Image> {
    Ok(Image {
        id: img.get_id()?.to_str()?.to_owned(),
        repo_tags: {
            let tags = img.get_repo_tags()?;
            let mut result = Vec::with_capacity(tags.len() as usize);
            for t in tags.iter() {
                result.push(t?.to_str()?.to_owned());
            }
            result
        },
        repo_digests: {
            let digests = img.get_repo_digests()?;
            let mut result = Vec::with_capacity(digests.len() as usize);
            for d in digests.iter() {
                result.push(d?.to_str()?.to_owned());
            }
            result
        },
        size: img.get_size(),
        uid: if img.get_uid() >= 0 {
            Some(img.get_uid())
        } else {
            None
        },
        username: img.get_username()?.to_str()?.to_owned(),
        spec: if img.has_spec() {
            Some(parse_image_spec(img.get_spec()?)?)
        } else {
            None
        },
        pinned: img.get_pinned(),
    })
}

// ============================================================================
// Request Building Helpers
// ============================================================================

fn build_key_values(
    mut builder: capnp::struct_list::Builder<workers_capnp::key_value::Owned>,
    map: &std::collections::HashMap<String, String>,
) {
    for (i, (k, v)) in map.iter().enumerate() {
        let mut kv = builder.reborrow().get(i as u32);
        kv.set_key(k);
        kv.set_value(v);
    }
}

fn build_pod_sandbox_config(
    mut builder: workers_capnp::pod_sandbox_config::Builder,
    config: &PodSandboxConfig,
) {
    // Metadata
    let mut meta = builder.reborrow().init_metadata();
    meta.set_name(&config.metadata.name);
    meta.set_uid(&config.metadata.uid);
    meta.set_namespace(&config.metadata.namespace);
    meta.set_attempt(config.metadata.attempt);

    builder.set_hostname(&config.hostname);
    builder.set_log_directory(&config.log_directory);

    // DNS config
    if let Some(dns) = &config.dns_config {
        let mut dns_builder = builder.reborrow().init_dns_config();
        let mut servers = dns_builder.reborrow().init_servers(dns.servers.len() as u32);
        for (i, s) in dns.servers.iter().enumerate() {
            servers.set(i as u32, s);
        }
        let mut searches = dns_builder.reborrow().init_searches(dns.searches.len() as u32);
        for (i, s) in dns.searches.iter().enumerate() {
            searches.set(i as u32, s);
        }
        let mut options = dns_builder.init_options(dns.options.len() as u32);
        for (i, o) in dns.options.iter().enumerate() {
            options.set(i as u32, o);
        }
    }

    // Port mappings (Note: port_mappings is Vec<PortMapping> which is not currently in PodSandboxConfig)
    // Skip port mappings for now - they're not commonly used and can be added later

    // Labels
    let labels = builder.reborrow().init_labels(config.labels.len() as u32);
    build_key_values(labels, &config.labels);

    // Annotations
    let annotations = builder.init_annotations(config.annotations.len() as u32);
    build_key_values(annotations, &config.annotations);
}

fn build_pod_sandbox_filter(
    mut builder: workers_capnp::pod_sandbox_filter::Builder,
    filter: &PodSandboxFilter,
) {
    if let Some(id) = &filter.id {
        builder.set_id(id);
    }
    if let Some(state) = &filter.state {
        builder.set_state(match state {
            PodSandboxState::SandboxReady => workers_capnp::PodSandboxState::SandboxReady,
            PodSandboxState::SandboxNotReady => workers_capnp::PodSandboxState::SandboxNotReady,
        });
    }
    let labels = builder.init_label_selector(filter.label_selector.len() as u32);
    build_key_values(labels, &filter.label_selector);
}

fn build_container_config(
    mut builder: workers_capnp::container_config::Builder,
    config: &ContainerConfig,
) {
    // Metadata
    let mut meta = builder.reborrow().init_metadata();
    meta.set_name(&config.metadata.name);
    meta.set_attempt(config.metadata.attempt);

    // Image (use ContainerImageSpec variant)
    build_container_image_spec(builder.reborrow().init_image(), &config.image);

    // Command
    let mut cmd = builder.reborrow().init_command(config.command.len() as u32);
    for (i, c) in config.command.iter().enumerate() {
        cmd.set(i as u32, c);
    }

    // Args
    let mut args = builder.reborrow().init_args(config.args.len() as u32);
    for (i, a) in config.args.iter().enumerate() {
        args.set(i as u32, a);
    }

    builder.set_working_dir(&config.working_dir);

    // Envs (Vec<KeyValue>)
    let mut envs = builder.reborrow().init_envs(config.envs.len() as u32);
    for (i, kv) in config.envs.iter().enumerate() {
        let mut env = envs.reborrow().get(i as u32);
        env.set_key(&kv.key);
        env.set_value(&kv.value);
    }

    // Labels
    let labels = builder.reborrow().init_labels(config.labels.len() as u32);
    build_key_values(labels, &config.labels);

    // Annotations
    let annotations = builder.reborrow().init_annotations(config.annotations.len() as u32);
    build_key_values(annotations, &config.annotations);

    builder.set_log_path(&config.log_path);
    builder.set_stdin(config.stdin);
    builder.set_stdin_once(config.stdin_once);
    builder.set_tty(config.tty);
}

fn build_container_filter(
    mut builder: workers_capnp::container_filter::Builder,
    filter: &ContainerFilter,
) {
    if let Some(id) = &filter.id {
        builder.set_id(id);
    }
    if let Some(sandbox_id) = &filter.pod_sandbox_id {
        builder.set_pod_sandbox_id(sandbox_id);
    }
    if let Some(state) = &filter.state {
        builder.set_state(match state {
            ContainerState::ContainerCreated => workers_capnp::ContainerState::ContainerCreated,
            ContainerState::ContainerRunning => workers_capnp::ContainerState::ContainerRunning,
            ContainerState::ContainerExited => workers_capnp::ContainerState::ContainerExited,
            ContainerState::ContainerUnknown => workers_capnp::ContainerState::ContainerUnknown,
        });
    }
    let labels = builder.init_label_selector(filter.label_selector.len() as u32);
    build_key_values(labels, &filter.label_selector);
}

fn build_pod_sandbox_stats_filter(
    mut builder: workers_capnp::pod_sandbox_stats_filter::Builder,
    filter: &PodSandboxStatsFilter,
) {
    if let Some(id) = &filter.id {
        builder.set_id(id);
    }
    let labels = builder.init_label_selector(filter.label_selector.len() as u32);
    build_key_values(labels, &filter.label_selector);
}

fn build_container_stats_filter(
    mut builder: workers_capnp::container_stats_filter::Builder,
    filter: &ContainerStatsFilter,
) {
    if let Some(id) = &filter.id {
        builder.set_id(id);
    }
    if let Some(sandbox_id) = &filter.pod_sandbox_id {
        builder.set_pod_sandbox_id(sandbox_id);
    }
    let labels = builder.init_label_selector(filter.label_selector.len() as u32);
    build_key_values(labels, &filter.label_selector);
}

fn build_image_spec(mut builder: workers_capnp::image_spec::Builder, spec: &ImageSpec) {
    builder.set_image(&spec.image);
    let annotations = builder.reborrow().init_annotations(spec.annotations.len() as u32);
    build_key_values(annotations, &spec.annotations);
    builder.set_runtime_handler(&spec.runtime_handler);
}

fn build_container_image_spec(mut builder: workers_capnp::image_spec::Builder, spec: &ContainerImageSpec) {
    builder.set_image(&spec.image);
    let annotations = builder.reborrow().init_annotations(spec.annotations.len() as u32);
    build_key_values(annotations, &spec.annotations);
    builder.set_runtime_handler(&spec.runtime_handler);
}

fn build_image_filter(builder: workers_capnp::image_filter::Builder, filter: &ImageFilter) {
    if let Some(spec) = &filter.image {
        build_image_spec(builder.init_image(), spec);
    }
}

fn build_auth_config(mut builder: workers_capnp::auth_config::Builder, auth: &AuthConfig) {
    builder.set_username(&auth.username);
    builder.set_password(&auth.password);
    builder.set_auth(&auth.auth);
    builder.set_server_address(&auth.server_address);
    builder.set_identity_token(&auth.identity_token);
    builder.set_registry_token(&auth.registry_token);
}

// ============================================================================
// WorkerClient
// ============================================================================

/// WorkerClient wraps ZMQ communication with WorkerService
///
/// Implements both RuntimeClient (sandbox/container lifecycle) and
/// ImageClient (image management) via ZMQ REQ/REP.
pub struct WorkerClient {
    client: Arc<ZmqClient>,
}

impl WorkerClient {
    /// Create a new WorkerClient with the given signing key and identity
    ///
    /// # Note
    /// Uses the same signing key for both request signing and response verification.
    /// This is appropriate for internal communication where client and server share keys.
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        let server_verifying_key = signing_key.verifying_key();
        Self {
            client: Arc::new(ZmqClient::new(&endpoint, signing_key, server_verifying_key, identity)),
        }
    }

    /// Create a WorkerClient connected to a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let server_verifying_key = signing_key.verifying_key();
        Self {
            client: Arc::new(ZmqClient::new(endpoint, signing_key, server_verifying_key, identity)),
        }
    }

    /// Create from existing ZmqClient
    pub fn from_client(client: Arc<ZmqClient>) -> Self {
        Self { client }
    }

    /// Get the underlying ZMQ client
    pub fn zmq_client(&self) -> &Arc<ZmqClient> {
        &self.client
    }

    /// Attach to container I/O streams
    pub async fn attach(&self, container_id: &str) -> anyhow::Result<AttachResponse> {
        self.client.attach(container_id).await
    }

    /// Start FD stream (complete DH handshake)
    pub async fn start_fd_stream(&self, stream_id: &str, client_pubkey: &[u8; 32]) -> anyhow::Result<FdStreamAuthResponse> {
        self.client.start_fd_stream(stream_id, client_pubkey).await
    }

    /// Detach from container I/O streams
    pub async fn detach(&self, container_id: &str) -> anyhow::Result<()> {
        self.client.detach(container_id).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RuntimeClient Implementation (delegates to RuntimeOps)
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
impl RuntimeClient for WorkerClient {
    async fn version(&self, version: &str) -> hyprstream_workers::error::Result<VersionResponse> {
        self.client
            .runtime_version(version)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn status(&self, verbose: bool) -> hyprstream_workers::error::Result<StatusResponse> {
        self.client
            .runtime_status(verbose)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn run_pod_sandbox(
        &self,
        config: &PodSandboxConfig,
    ) -> hyprstream_workers::error::Result<String> {
        self.client
            .run_pod_sandbox(config)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn stop_pod_sandbox(
        &self,
        pod_sandbox_id: &str,
    ) -> hyprstream_workers::error::Result<()> {
        self.client
            .stop_pod_sandbox(pod_sandbox_id)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn remove_pod_sandbox(
        &self,
        pod_sandbox_id: &str,
    ) -> hyprstream_workers::error::Result<()> {
        self.client
            .remove_pod_sandbox(pod_sandbox_id)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn pod_sandbox_status(
        &self,
        pod_sandbox_id: &str,
        verbose: bool,
    ) -> hyprstream_workers::error::Result<PodSandboxStatusResponse> {
        self.client
            .pod_sandbox_status(pod_sandbox_id, verbose)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn list_pod_sandbox(
        &self,
        filter: Option<&PodSandboxFilter>,
    ) -> hyprstream_workers::error::Result<Vec<PodSandbox>> {
        self.client
            .list_pod_sandbox(filter)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn create_container(
        &self,
        pod_sandbox_id: &str,
        config: &ContainerConfig,
        sandbox_config: &PodSandboxConfig,
    ) -> hyprstream_workers::error::Result<String> {
        self.client
            .create_container(pod_sandbox_id, config, sandbox_config)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn start_container(
        &self,
        container_id: &str,
    ) -> hyprstream_workers::error::Result<()> {
        self.client
            .start_container(container_id)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn stop_container(
        &self,
        container_id: &str,
        timeout: i64,
    ) -> hyprstream_workers::error::Result<()> {
        self.client
            .stop_container(container_id, timeout)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn remove_container(
        &self,
        container_id: &str,
    ) -> hyprstream_workers::error::Result<()> {
        self.client
            .remove_container(container_id)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn container_status(
        &self,
        container_id: &str,
        verbose: bool,
    ) -> hyprstream_workers::error::Result<ContainerStatusResponse> {
        self.client
            .container_status(container_id, verbose)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn list_containers(
        &self,
        filter: Option<&ContainerFilter>,
    ) -> hyprstream_workers::error::Result<Vec<Container>> {
        self.client
            .list_containers(filter)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn exec_sync(
        &self,
        container_id: &str,
        cmd: &[String],
        timeout: i64,
    ) -> hyprstream_workers::error::Result<ExecSyncResponse> {
        self.client
            .exec_sync(container_id, cmd, timeout)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn pod_sandbox_stats(
        &self,
        pod_sandbox_id: &str,
    ) -> hyprstream_workers::error::Result<PodSandboxStats> {
        self.client
            .pod_sandbox_stats(pod_sandbox_id)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn list_pod_sandbox_stats(
        &self,
        filter: Option<&PodSandboxStatsFilter>,
    ) -> hyprstream_workers::error::Result<Vec<PodSandboxStats>> {
        self.client
            .list_pod_sandbox_stats(filter)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn container_stats(
        &self,
        container_id: &str,
    ) -> hyprstream_workers::error::Result<ContainerStats> {
        self.client
            .container_stats(container_id)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn list_container_stats(
        &self,
        filter: Option<&ContainerStatsFilter>,
    ) -> hyprstream_workers::error::Result<Vec<ContainerStats>> {
        self.client
            .list_container_stats(filter)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ImageClient Implementation (delegates to ImageOps)
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
impl ImageClient for WorkerClient {
    async fn list_images(
        &self,
        filter: Option<&ImageFilter>,
    ) -> hyprstream_workers::error::Result<Vec<Image>> {
        self.client
            .list_images(filter)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn image_status(
        &self,
        image: &ImageSpec,
        verbose: bool,
    ) -> hyprstream_workers::error::Result<hyprstream_workers::image::ImageStatusResponse> {
        self.client
            .image_status(image, verbose)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::ImageNotFound(e.to_string()))
    }

    async fn pull_image(
        &self,
        image: &ImageSpec,
        auth: Option<&AuthConfig>,
    ) -> hyprstream_workers::error::Result<String> {
        self.client
            .pull_image(image, auth)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn remove_image(
        &self,
        image: &ImageSpec,
    ) -> hyprstream_workers::error::Result<()> {
        self.client
            .remove_image(image)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }

    async fn image_fs_info(
        &self,
    ) -> hyprstream_workers::error::Result<Vec<hyprstream_workers::image::FilesystemUsage>> {
        self.client
            .image_fs_info()
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::Internal(e.to_string()))
    }
}
