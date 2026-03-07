//! GPU detection for the setup wizard.
//!
//! Probes the host for NVIDIA (nvidia-smi) and AMD (rocminfo) GPUs,
//! maps driver versions to CUDA/ROCm compatibility, and detects the
//! current run mode (AppImage, bare binary, development).

use hyprstream_tui::wizard::backend::{
    EnvironmentInfo, GpuKind, LibtorchVariant, RunMode,
};
use std::path::Path;
use tracing::debug;

/// Detect GPU hardware on the host.
pub fn detect_gpu() -> GpuKind {
    // NVIDIA: parse nvidia-smi
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !version.is_empty() {
                let cuda_compat = driver_to_cuda_compat(&version);
                debug!("Detected NVIDIA GPU: driver={version}, cuda_compat={cuda_compat}");
                return GpuKind::Nvidia {
                    driver_version: version,
                    cuda_compat,
                };
            }
        }
    }

    // AMD: parse rocminfo
    if let Ok(output) = std::process::Command::new("rocminfo").output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            let version = parse_rocm_version(&text);
            debug!("Detected AMD GPU: rocm_version={version:?}");
            return GpuKind::Amd {
                rocm_version: version,
            };
        }
    }

    debug!("No GPU detected");
    GpuKind::None
}

/// Map NVIDIA driver version to highest compatible CUDA version.
///
/// Based on NVIDIA's CUDA compatibility matrix:
/// <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>
fn driver_to_cuda_compat(driver_version: &str) -> String {
    let mut parts = driver_version.split('.');
    let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    if major >= 555 {
        "cuda130".to_owned()
    } else if major >= 535 {
        "cuda128".to_owned()
    } else {
        // Older drivers: CPU fallback
        "cpu".to_owned()
    }
}

/// Extract ROCm version from rocminfo output.
fn parse_rocm_version(rocminfo_output: &str) -> Option<String> {
    // rocminfo prints "ROCm Runtime Version: X.Y.Z"
    for line in rocminfo_output.lines() {
        if let Some(rest) = line.strip_prefix("ROCm Runtime Version:") {
            let version = rest.trim().to_owned();
            if !version.is_empty() {
                return Some(version);
            }
        }
    }
    // Also check for "HSA Runtime Version" as fallback
    for line in rocminfo_output.lines() {
        if line.contains("HSA Runtime Version") {
            if let Some(v) = line.split(':').nth(1) {
                let version = v.trim().to_owned();
                if !version.is_empty() {
                    return Some(version);
                }
            }
        }
    }
    None
}

/// Detect how hyprstream is currently running.
pub fn detect_run_mode() -> RunMode {
    if std::env::var("APPIMAGE").is_ok() {
        RunMode::AppImage
    } else if std::env::var("CARGO_MANIFEST_DIR").is_ok()
        || std::env::var("HYPRSTREAM_LIBTORCH_CONFIGURED").is_ok()
    {
        RunMode::Development
    } else {
        RunMode::BareBinary
    }
}

/// Check if a backend variant is already installed in the data directory.
pub fn detect_installed_variant(data_dir: &Path) -> Option<LibtorchVariant> {
    let active = std::fs::read_to_string(data_dir.join("active-backend")).ok()?;
    variant_from_id(active.trim())
}

/// Read the active version string from the data directory.
///
/// Validates that the version string contains only safe characters
/// (alphanumeric, dots, hyphens) to prevent path traversal attacks.
pub fn detect_active_version(data_dir: &Path) -> Option<String> {
    std::fs::read_to_string(data_dir.join("active-version"))
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .filter(|s| is_safe_version_string(s))
}

/// Check that a version string contains only safe path characters.
fn is_safe_version_string(v: &str) -> bool {
    !v.is_empty()
        && !v.contains("..")
        && v.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
}

/// Map a variant ID string to a `LibtorchVariant`.
pub fn variant_from_id(id: &str) -> Option<LibtorchVariant> {
    match id {
        "cpu" => Some(LibtorchVariant::Cpu),
        "cuda128" => Some(LibtorchVariant::Cuda128),
        "cuda130" => Some(LibtorchVariant::Cuda130),
        "rocm71" => Some(LibtorchVariant::Rocm71),
        _ => None,
    }
}

/// Map GPU kind to recommended variant.
pub fn recommend_variant(gpu: &GpuKind) -> LibtorchVariant {
    match gpu {
        GpuKind::Nvidia { cuda_compat, .. } => match cuda_compat.as_str() {
            "cuda130" => LibtorchVariant::Cuda130,
            "cuda128" => LibtorchVariant::Cuda128,
            _ => LibtorchVariant::Cpu,
        },
        GpuKind::Amd {
            rocm_version: Some(_),
        } => LibtorchVariant::Rocm71,
        _ => LibtorchVariant::Cpu,
    }
}

/// Build full environment info snapshot.
pub fn detect_environment(data_dir: &Path) -> EnvironmentInfo {
    let gpu = detect_gpu();
    let run_mode = detect_run_mode();
    let recommended = recommend_variant(&gpu);
    let installed = detect_installed_variant(data_dir);

    EnvironmentInfo {
        run_mode,
        gpu,
        libtorch_path: std::env::var("LIBTORCH").ok(),
        current_variant: LibtorchVariant::from_compiled(),
        recommended_variant: recommended,
        installed_variant: installed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_to_cuda_compat() {
        assert_eq!(driver_to_cuda_compat("555.42"), "cuda130");
        assert_eq!(driver_to_cuda_compat("560.0"), "cuda130");
        assert_eq!(driver_to_cuda_compat("535.104"), "cuda128");
        assert_eq!(driver_to_cuda_compat("535.0"), "cuda128");
        assert_eq!(driver_to_cuda_compat("530.0"), "cpu");
        assert_eq!(driver_to_cuda_compat("470.82"), "cpu");
    }

    #[test]
    fn test_parse_rocm_version() {
        assert_eq!(
            parse_rocm_version("ROCm Runtime Version: 7.1.0"),
            Some("7.1.0".to_owned())
        );
        assert_eq!(
            parse_rocm_version("some junk\nROCm Runtime Version: 6.2.0\nmore"),
            Some("6.2.0".to_owned())
        );
        assert_eq!(parse_rocm_version("no version here"), None);
    }

    #[test]
    fn test_variant_from_id() {
        assert_eq!(variant_from_id("cpu"), Some(LibtorchVariant::Cpu));
        assert_eq!(variant_from_id("cuda130"), Some(LibtorchVariant::Cuda130));
        assert_eq!(variant_from_id("unknown"), None);
    }

    #[test]
    fn test_recommend_variant() {
        let nvidia = GpuKind::Nvidia {
            driver_version: "555.0".to_owned(),
            cuda_compat: "cuda130".to_owned(),
        };
        assert_eq!(recommend_variant(&nvidia), LibtorchVariant::Cuda130);

        let amd = GpuKind::Amd {
            rocm_version: Some("7.1.0".to_owned()),
        };
        assert_eq!(recommend_variant(&amd), LibtorchVariant::Rocm71);

        assert_eq!(recommend_variant(&GpuKind::None), LibtorchVariant::Cpu);
    }
}
