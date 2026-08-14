/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

/// Target GPU backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TargetBackend {
    /// NVIDIA CUDA (NVVM/PTX) — the default.
    #[default]
    Cuda,
    /// MetaX GPU (MXMACA).
    Maca,
}

/// Compiler control for materializing semantic IKET annotations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IketInstrumentation {
    /// Erase annotations before ordinary code generation.
    Disabled,
    /// Prefer NativeDump and switch to ExtendedNativeDump above 30 names.
    Auto,
    /// Require NativeDump.
    NativeDump,
    /// Require ExtendedNativeDump.
    ExtendedNativeDump,
    /// Preserve an invalid environment value for a pipeline diagnostic.
    Invalid(String),
}

/// Explicit backend knobs; replaces every `CUDA_OXIDE_*` env read inside the
/// backend. `run_pipeline` (mir-importer) builds one from the environment at
/// its own boundary. The experimental API builds one from typed compile
/// options without reading the environment.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BackendOptions {
    /// IKET physical instrumentation policy.
    pub iket: IketInstrumentation,
    /// Hard target override (`llc -mcpu=`), e.g. `"sm_120"`.
    pub target_arch: Option<String>,
    /// Human-readable name for whatever set `target_arch`, used only to
    /// describe target provenance in diagnostics and errors (e.g.
    /// `"CUDA_OXIDE_TARGET"` for the env-driven rustc pipeline, or a
    /// caller-facing description for the standalone experimental API).
    ///
    /// Keep this in step with `target_arch`: whoever writes one writes the
    /// other, or a target error names a source the caller never used.
    pub target_arch_source: &'static str,
    /// Advisory local-GPU arch; used only when it satisfies detected features.
    pub device_arch_hint: Option<String>,
    /// Skip the `opt -O2` middle-end.
    pub no_opt: bool,
    /// Suppress `llc -fp-contract=fast` (fmul+fadd fusion to fma).
    pub no_fma: bool,
    /// Print progress and tool-selection notes to stderr.
    pub verbose: bool,
    /// Explicit `llc` binary (was `CUDA_OXIDE_LLC`).
    pub llc_override: Option<PathBuf>,
    /// Explicit `opt` binary (was `CUDA_OXIDE_OPT`).
    pub opt_override: Option<PathBuf>,
    /// Explicit `mxcc` binary used for MXMACA device-image generation.
    pub mxcc_override: Option<PathBuf>,
    /// MXMACA SDK root passed to `mxcc` via `--maca-path`.
    pub maca_path: Option<PathBuf>,
    /// Target GPU backend (CUDA or MXMACA).
    pub backend: TargetBackend,
    /// Optional staged dialect-mir pass pipeline (`CUDA_OXIDE_MIR_PASSES`).
    ///
    /// Empty or `None` preserves the default pipeline. The available names
    /// are defined by the cuda-oxide-owned optimization registry. Each entry
    /// declares whether it runs before or after standard MIR preparation.
    pub mir_pass_pipeline: Option<String>,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            iket: IketInstrumentation::Auto,
            target_arch: None,
            target_arch_source: "CUDA_OXIDE_TARGET",
            device_arch_hint: None,
            no_opt: false,
            no_fma: false,
            verbose: false,
            llc_override: None,
            opt_override: None,
            mxcc_override: None,
            maca_path: None,
            backend: TargetBackend::Cuda,
            mir_pass_pipeline: None,
        }
    }
}

impl BackendOptions {
    /// Reads the historical `CUDA_OXIDE_*` variables; called by rustc-pipeline
    /// hosts, never by the backend itself. The only other env access in this
    /// crate is `CUDA_OXIDE_LLVM_LINK` in `llvm_tools::resolve_sibling_tool`
    /// (a per-toolchain tool override, not a compile option).
    pub fn from_env() -> Self {
        Self::try_from_env().unwrap_or_else(|message| panic!("{message}"))
    }

    /// Reads compatibility options from the environment and rejects an unknown
    /// target backend instead of silently compiling for CUDA.
    pub fn try_from_env() -> Result<Self, String> {
        let backend =
            target_backend_from_value(std::env::var("CUDA_OXIDE_TARGET_BACKEND").ok().as_deref())?;
        let iket = match std::env::var("CUDA_OXIDE_IKET") {
            Err(std::env::VarError::NotPresent) => IketInstrumentation::Auto,
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "" | "1" | "on" | "true" | "auto" => IketInstrumentation::Auto,
                "native" | "native_dump" | "nativedump" => IketInstrumentation::NativeDump,
                "extended" | "extended_native_dump" | "extendednativedump" => {
                    IketInstrumentation::ExtendedNativeDump
                }
                "0" | "off" | "false" => IketInstrumentation::Disabled,
                _ => IketInstrumentation::Invalid(value),
            },
            Err(std::env::VarError::NotUnicode(value)) => {
                IketInstrumentation::Invalid(value.to_string_lossy().into_owned())
            }
        };
        Ok(Self {
            iket,
            target_arch: std::env::var("CUDA_OXIDE_TARGET").ok(),
            target_arch_source: "CUDA_OXIDE_TARGET",
            device_arch_hint: std::env::var("CUDA_OXIDE_DEVICE_ARCH").ok(),
            no_opt: std::env::var("CUDA_OXIDE_NO_OPT").is_ok(),
            no_fma: std::env::var("CUDA_OXIDE_NO_FMA").is_ok(),
            verbose: std::env::var("CUDA_OXIDE_VERBOSE").is_ok(),
            llc_override: std::env::var("CUDA_OXIDE_LLC").ok().map(PathBuf::from),
            opt_override: std::env::var("CUDA_OXIDE_OPT").ok().map(PathBuf::from),
            mxcc_override: std::env::var("CUDA_OXIDE_MXCC").ok().map(PathBuf::from),
            maca_path: std::env::var("MACA_PATH").ok().map(PathBuf::from),
            backend,
            mir_pass_pipeline: std::env::var("CUDA_OXIDE_MIR_PASSES").ok(),
        })
    }
}

fn target_backend_from_value(value: Option<&str>) -> Result<TargetBackend, String> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("cuda") => Ok(TargetBackend::Cuda),
        Some("maca") => Ok(TargetBackend::Maca),
        Some(_) => Err(format!(
            "invalid CUDA_OXIDE_TARGET_BACKEND value `{}`; expected `cuda` or `maca`",
            value.unwrap_or_default()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_backend_uses_the_dedicated_environment_value() {
        assert_eq!(target_backend_from_value(None), Ok(TargetBackend::Cuda));
        assert_eq!(
            target_backend_from_value(Some("cuda")),
            Ok(TargetBackend::Cuda)
        );
        assert_eq!(
            target_backend_from_value(Some("MACA")),
            Ok(TargetBackend::Maca)
        );
        assert!(target_backend_from_value(Some("macaa")).is_err());
        assert!(target_backend_from_value(Some("")).is_err());
        assert!(target_backend_from_value(Some("/tmp/librustc_codegen_cuda.so")).is_err());
    }
}
