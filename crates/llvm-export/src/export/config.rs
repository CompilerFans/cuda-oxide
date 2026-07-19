/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend configuration traits and built-in implementations.

/// Minimal data layout for PTX mode (default behavior).
pub(super) const NVPTX_DATALAYOUT_PTX: &str = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

/// Full NVPTX data layout for libNVVM/LTOIR mode (Blackwell+, modern dialect).
///
/// This matches nvcc's output for sm_100+ and is required for full NVVM compatibility.
pub(super) const NVPTX_DATALAYOUT_FULL: &str = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-\
    i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-\
    v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8";

/// The only supported 64-bit data layout for the legacy LLVM 7 NVVM dialect.
///
/// Keep this in sync with NVIDIA's NVVM IR specification. In particular, the
/// legacy parser does not accept the modern per-address-space layout entries.
pub(super) const NVPTX_DATALAYOUT_LEGACY: &str = "e-p:64:64:64-i1:8:8-i8:8:8-\
    i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-\
    v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

/// Textual LLVM dialect selected by libNVVM for a concrete GPU target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NvvmIrDialect {
    /// LLVM 7 syntax, including typed pointers. Used for pre-Blackwell targets.
    LegacyLlvm7,
    /// The current toolkit's modern opaque-pointer syntax. Used for Blackwell+.
    #[default]
    Modern,
}

impl NvvmIrDialect {
    pub fn uses_typed_pointers(self) -> bool {
        matches!(self, Self::LegacyLlvm7)
    }
}

/// Device debug metadata to emit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugKind {
    /// Do not emit LLVM debug metadata.
    #[default]
    Off,
    /// Emit enough metadata for source-line breakpoints and stepping.
    LineTables,
    /// Emit line tables plus the supported local-variable/type metadata.
    Full,
}

/// Calling convention emitted for functions marked as GPU kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KernelCallingConvention {
    /// Ordinary C calling convention. Kernel identity is carried by metadata.
    #[default]
    C,
    /// NVPTX kernel calling convention used by the direct PTX pipeline.
    PtxKernel,
    /// MXMACA kernel calling convention consumed by `mxcc`.
    MetaxGpuKernel,
}

impl DebugKind {
    pub fn line_tables_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub fn variables_enabled(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Configuration trait for export backends (PTX, LTOIR, MXMACA, etc.).
///
/// This trait allows different backends to customize IR generation without
/// exposing backend-specific details in the public API.
pub trait ExportBackendConfig {
    /// Data layout string for the target.
    fn datalayout(&self) -> &str;

    /// Target triple for the LLVM module (e.g., "nvptx64-nvidia-cuda", "mxc-metax-macahca").
    fn target_triple(&self) -> &str;

    /// Whether to emit `@llvm.used` for kernel functions.
    /// This prevents the optimizer from removing "unused" kernels.
    fn emit_llvm_used(&self) -> bool;

    /// Whether to emit `!nvvmir.version` metadata.
    fn emit_nvvmir_version(&self) -> bool;

    /// The version tuple for `!nvvmir.version` metadata.
    /// Format: [major, minor, debug_major, debug_minor]
    fn nvvmir_version(&self) -> [i32; 4];

    /// Whether to emit `!nvvm.annotations` for ALL kernels.
    /// When false, only kernels with special attributes get annotations.
    fn emit_all_kernel_annotations(&self) -> bool;

    /// Whether kernel definitions should use the `ptx_kernel` calling convention.
    ///
    /// This is the original backend hook and remains the compatibility surface
    /// for external configurations. New backends with a non-PTX convention
    /// should override [`Self::kernel_calling_convention`] as well.
    fn emit_ptx_kernel_keyword(&self) -> bool {
        false
    }

    /// Calling convention to emit for GPU kernel declarations and definitions.
    fn kernel_calling_convention(&self) -> KernelCallingConvention {
        if self.emit_ptx_kernel_keyword() {
            KernelCallingConvention::PtxKernel
        } else {
            KernelCallingConvention::C
        }
    }

    /// NVVM input dialect, when this is an NVVM export.
    fn nvvm_ir_dialect(&self) -> Option<NvvmIrDialect> {
        None
    }

    /// Address space in which `alloca` results are materialized.
    ///
    /// Most targets use the generic address space (0) directly. Targets whose
    /// backend cannot select a frame index in the generic space (e.g. MXMACA,
    /// where local memory lives in address space 5) place allocas in a
    /// private space and `addrspacecast` the result back to a generic pointer,
    /// mirroring what the target's own clang emits.
    fn alloca_address_space(&self) -> u32 {
        0
    }

    /// Which device debug metadata tier to emit.
    fn debug_kind(&self) -> DebugKind {
        DebugKind::Off
    }
}

/// Default PTX export configuration.
///
/// Uses minimal settings appropriate for standard PTX generation via llc.
#[derive(Clone, Debug, Default)]
pub struct PtxExportConfig;

impl ExportBackendConfig for PtxExportConfig {
    fn datalayout(&self) -> &str {
        NVPTX_DATALAYOUT_PTX
    }

    fn target_triple(&self) -> &str {
        "nvptx64-nvidia-cuda"
    }

    fn emit_llvm_used(&self) -> bool {
        false
    }

    fn emit_nvvmir_version(&self) -> bool {
        false
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        [0, 0, 0, 0] // Not used in PTX mode
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        false
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        true
    }
}

/// Export configuration for NVVM IR output.
///
/// Emits LLVM IR with full NVVM compatibility:
/// - Full NVPTX datalayout string
/// - `@llvm.used` to prevent kernel optimization
/// - `!nvvm.annotations` for all kernels
/// - `!nvvmir.version` metadata
///
/// This produces IR suitable for consumption by libNVVM (e.g., `nvvmCompileProgram -gen-lto`)
/// or other NVVM-compatible tools.
///
/// Supports both libNVVM input dialects: LLVM 7 typed-pointer IR for
/// pre-Blackwell targets and the current toolkit's opaque-pointer IR for
/// Blackwell and newer targets.
#[derive(Clone, Debug, Default)]
pub struct NvvmExportConfig {
    dialect: NvvmIrDialect,
}

impl NvvmExportConfig {
    pub fn new(dialect: NvvmIrDialect) -> Self {
        Self { dialect }
    }

    pub fn dialect(&self) -> NvvmIrDialect {
        self.dialect
    }
}

impl ExportBackendConfig for NvvmExportConfig {
    fn datalayout(&self) -> &str {
        match self.dialect {
            NvvmIrDialect::LegacyLlvm7 => NVPTX_DATALAYOUT_LEGACY,
            NvvmIrDialect::Modern => NVPTX_DATALAYOUT_FULL,
        }
    }

    fn target_triple(&self) -> &str {
        "nvptx64-nvidia-cuda"
    }

    fn emit_llvm_used(&self) -> bool {
        true
    }

    fn emit_nvvmir_version(&self) -> bool {
        true
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        match self.dialect {
            NvvmIrDialect::LegacyLlvm7 => [2, 0, 3, 1],
            NvvmIrDialect::Modern => [2, 0, 3, 2],
        }
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        true // Emit annotations for all kernels
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        false
    }

    fn nvvm_ir_dialect(&self) -> Option<NvvmIrDialect> {
        Some(self.dialect)
    }
}

// ============================================================================
// MXMACA (MetaX) backend configuration
// ============================================================================

/// MXMACA data layout for MetaX GPU targets (xcore1000).
///
/// Obtained from `mxcc -S -emit-llvm` output on a simple .maca file.
pub(super) const MACA_DATALAYOUT: &str = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-\
    p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-\
    v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7";

/// MXMACA target triple.
pub(super) const MACA_TARGET_TRIPLE: &str = "mxc-metax-macahca";

/// Export configuration for MXMACA (MetaX GPU) backend.
///
/// Uses MXMACA-specific settings:
/// - MXMACA data layout (xcore1000)
/// - `mxc-metax-macahca` target triple
/// - No NVVM metadata (not NVIDIA)
/// - No PTX kernel keyword (uses `metaxgpu_kernel` calling convention)
#[derive(Clone, Debug, Default)]
pub struct MacaExportConfig;

impl ExportBackendConfig for MacaExportConfig {
    fn datalayout(&self) -> &str {
        MACA_DATALAYOUT
    }

    fn target_triple(&self) -> &str {
        MACA_TARGET_TRIPLE
    }

    fn emit_llvm_used(&self) -> bool {
        false
    }

    fn emit_nvvmir_version(&self) -> bool {
        false
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        [0, 0, 0, 0]
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        false
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        false
    }

    fn kernel_calling_convention(&self) -> KernelCallingConvention {
        KernelCallingConvention::MetaxGpuKernel
    }

    /// MXMACA local memory lives in address space 5; the backend cannot
    /// select a frame index in the generic address space (matches the
    /// `alloca ..., addrspace(5)` emitted by MetaX clang).
    fn alloca_address_space(&self) -> u32 {
        5
    }
}
