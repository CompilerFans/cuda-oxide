/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Low-level FFI to the CUDA Driver API (`cuda.h`).
//!
//! Bindings are generated at build time by [`bindgen`](https://docs.rs/bindgen) from `wrapper.h`,
//! which includes the toolkit `cuda.h`. The build script passes `-I$CUDA_TOOLKIT_PATH/include` to
//! Clang, emits `cargo:rustc-link-search` for discovered library directories, and links
//! `libcuda` (`dylib=cuda`). Generated Rust lives under `OUT_DIR` as `bindings.rs` and is pulled in
//! via [`include!`].
//!
//! **Toolkit path:** set `CUDA_TOOLKIT_PATH` (or, failing that, `CUDA_HOME`) to the root of your
//! CUDA installation (the directory that contains `include/cuda.h`). If neither is set, the build
//! script and [`cuda_toolkit_dir`] both use `/usr/local/cuda`. Changing either variable or
//! `wrapper.h` triggers a rebuild.
//!
//! Types and functions in the generated module are `unsafe` where required by Rust; each carries
//! the usual CUDA API preconditions (valid handles, device state, stream ordering, etc.).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// The generated bindings carry CUDA's C doxygen comments verbatim. Those contain
// `[...]` spans, bare URLs, HTML-ish tables, and `\brief`-style code that rustdoc
// flags as broken intra-doc links, bare URLs, unclosed HTML tags, and unparseable
// Rust code blocks. We keep the comments (they are useful API docs) but silence
// these lints for this generated FFI crate; its doctests are excluded from the
// `--doc` gate in CI.
#![allow(rustdoc::broken_intra_doc_links)]
#![allow(rustdoc::bare_urls)]
#![allow(rustdoc::invalid_html_tags)]
#![allow(rustdoc::invalid_rust_codeblocks)]
// cuda.h transitively declares libc's `malloc`/`realloc` with C's `unsigned
// long`, which bindgen renders as `c_ulong` (`u64`) while rustc 1.100+ expects
// the runtime symbols spelled with `usize` and warns
// (`suspicious_runtime_symbol_definitions`). These are plain EXTERN
// declarations of the very allocator the platform already provides (same ABI
// on every 64-bit target we build for), not redefinitions, so the lint does
// not indicate a real hazard here.
#![allow(suspicious_runtime_symbol_definitions)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::env;

#[cfg(cuda_oxide_cu_bridge)]
mod cu_bridge_compat;
#[cfg(cuda_oxide_cu_bridge)]
pub use cu_bridge_compat::*;

/// Driver implementation selected when this crate was compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaDriverBackend {
    /// NVIDIA's native CUDA Driver API.
    NativeCuda,
    /// MXMACA's CUDA-compatible cu-bridge Driver API.
    MacaCuBridge,
}

/// Driver implementation selected from the CUDA headers used by `build.rs`.
#[cfg(cuda_oxide_cu_bridge)]
pub const CUDA_DRIVER_BACKEND: CudaDriverBackend = CudaDriverBackend::MacaCuBridge;

/// Driver implementation selected from the CUDA headers used by `build.rs`.
#[cfg(not(cuda_oxide_cu_bridge))]
pub const CUDA_DRIVER_BACKEND: CudaDriverBackend = CudaDriverBackend::NativeCuda;

/// Returns whether `error` is CUDA's invalid-cluster-size status.
///
/// MXMACA 3.7 has no equivalent cu-bridge status, so that branch returns
/// `false` instead of assigning a guessed error value.
pub fn is_invalid_cluster_size(error: CUresult) -> bool {
    #[cfg(cuda_oxide_cu_bridge)]
    {
        let _ = error;
        false
    }
    #[cfg(not(cuda_oxide_cu_bridge))]
    {
        error == cudaError_enum_CUDA_ERROR_INVALID_CLUSTER_SIZE
    }
}

/// Returns whether `error` denotes PTX unsupported by the CUDA driver.
///
/// MACA consumes LLVM IR rather than PTX and has no semantically equivalent
/// status, so this predicate is always false in cu-bridge builds.
pub fn is_unsupported_ptx_version(error: CUresult) -> bool {
    #[cfg(cuda_oxide_cu_bridge)]
    {
        let _ = error;
        false
    }
    #[cfg(not(cuda_oxide_cu_bridge))]
    {
        error == cudaError_enum_CUDA_ERROR_UNSUPPORTED_PTX_VERSION
    }
}

/// Returns CUDA's unsupported-PTX status when the selected backend defines it.
#[doc(hidden)]
pub fn unsupported_ptx_version_error() -> Option<CUresult> {
    #[cfg(cuda_oxide_cu_bridge)]
    {
        None
    }
    #[cfg(not(cuda_oxide_cu_bridge))]
    {
        Some(cudaError_enum_CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
    }
}

/// Queries a function's compile-time required cluster dimensions.
///
/// # Safety
///
/// `width`, `height`, and `depth` must each be valid for an `i32` write, and
/// `function` must be a live function in the current context.
pub unsafe fn cu_function_required_cluster_dimensions(
    width: *mut std::ffi::c_int,
    height: *mut std::ffi::c_int,
    depth: *mut std::ffi::c_int,
    function: CUfunction,
) -> CUresult {
    #[cfg(cuda_oxide_cu_bridge)]
    {
        let _ = (width, height, depth, function);
        cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
    }
    #[cfg(not(cuda_oxide_cu_bridge))]
    {
        let result = unsafe {
            cuFuncGetAttribute(
                width,
                CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
                function,
            )
        };
        if result != cudaError_enum_CUDA_SUCCESS {
            return result;
        }
        let result = unsafe {
            cuFuncGetAttribute(
                height,
                CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,
                function,
            )
        };
        if result != cudaError_enum_CUDA_SUCCESS {
            return result;
        }
        unsafe {
            cuFuncGetAttribute(
                depth,
                CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
                function,
            )
        }
    }
}

/// Reports the elapsed time between two recorded events, dispatching to the
/// event elapsed-time driver entry point declared by this build's toolkit
/// headers.
///
/// # Safety
///
/// Same contract as the underlying driver call: `elapsed_ms` must be valid
/// for a `f32` write, and `start`/`end` must be valid event handles recorded
/// in the current context.
pub unsafe fn cu_event_elapsed_time(
    elapsed_ms: *mut f32,
    start: CUevent,
    end: CUevent,
) -> CUresult {
    #[cfg(cuda_oxide_cu_bridge)]
    {
        unsafe { wcuEventElapsedTime(elapsed_ms, start, end) }
    }
    #[cfg(all(not(cuda_oxide_cu_bridge), cuda_has_cuEventElapsedTime_v2))]
    {
        unsafe { cuEventElapsedTime_v2(elapsed_ms, start, end) }
    }
    #[cfg(all(not(cuda_oxide_cu_bridge), not(cuda_has_cuEventElapsedTime_v2)))]
    {
        unsafe { cuEventElapsedTime(elapsed_ms, start, end) }
    }
}

/// Root directory of the CUDA toolkit used for this build, for host code that must agree with
/// compile-time include and link paths (e.g. loading companion libraries or probing layout).
///
/// Resolution matches `build.rs`: the first set variable among `CUDA_TOOLKIT_PATH` and
/// `CUDA_HOME` (taken verbatim); when neither is present (or the value is not Unicode),
/// returns `/usr/local/cuda`.
pub fn cuda_toolkit_dir() -> String {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .find_map(|var| env::var(var).ok())
        .unwrap_or_else(|| "/usr/local/cuda".to_string())
}
