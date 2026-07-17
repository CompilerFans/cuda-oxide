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

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::env;

// Type aliases to map cu-bridge types to CUDA types for compatibility
pub type CUresult = mcDrvError_enum;
pub type CUevent = mcDrvEvent_t;
pub type CUdevice = mcDrvDevice_t;
pub type CUcontext = mcDrvContext_t;
pub type CUmodule = mcDrvModule_t;
pub type CUfunction = mcDrvFunction_t;
pub type CUstream = mcDrvStream_t;
pub type CUdeviceptr = mcDrvDeviceptr_t;

// Error code aliases
pub const CUDA_SUCCESS: mcDrvError_enum = mcDrvError_enum_MC_SUCCESS;
pub const CUDA_ERROR_INVALID_VALUE: mcDrvError_enum = mcDrvError_enum_MC_ERROR_INVALID_VALUE;
pub const CUDA_ERROR_OUT_OF_MEMORY: mcDrvError_enum = mcDrvError_enum_MC_ERROR_OUT_OF_MEMORY;
pub const CUDA_ERROR_NOT_INITIALIZED: mcDrvError_enum = mcDrvError_enum_MC_ERROR_NOT_INITIALIZED;
pub const CUDA_ERROR_DEINITIALIZED: mcDrvError_enum = mcDrvError_enum_MC_ERROR_DEINITIALIZED;
pub const CUDA_ERROR_NO_DEVICE: mcDrvError_enum = mcDrvError_enum_MC_ERROR_NO_DEVICE;
pub const CUDA_ERROR_INVALID_DEVICE: mcDrvError_enum = mcDrvError_enum_MC_ERROR_INVALID_DEVICE;
pub const CUDA_ERROR_INVALID_IMAGE: mcDrvError_enum = mcDrvError_enum_MC_ERROR_INVALID_IMAGE;
pub const CUDA_ERROR_INVALID_CONTEXT: mcDrvError_enum = mcDrvError_enum_MC_ERROR_INVALID_CONTEXT;
pub const CUDA_ERROR_INVALID_HANDLE: mcDrvError_enum = mcDrvError_enum_MC_ERROR_INVALID_HANDLE;
pub const CUDA_ERROR_NOT_FOUND: mcDrvError_enum = mcDrvError_enum_MC_ERROR_NOT_FOUND;
pub const CUDA_ERROR_NOT_READY: mcDrvError_enum = mcDrvError_enum_MC_ERROR_NOT_READY;
pub const CUDA_ERROR_ILLEGAL_ADDRESS: mcDrvError_enum = mcDrvError_enum_MC_ERROR_ILLEGAL_ADDRESS;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: mcDrvError_enum = mcDrvError_enum_MC_ERROR_LAUNCH_OUT_OF_RESOURCES;
pub const CUDA_ERROR_LAUNCH_TIMEOUT: mcDrvError_enum = mcDrvError_enum_MC_ERROR_LAUNCH_TIMEOUT;
pub const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: mcDrvError_enum = mcDrvError_enum_MC_ERROR_PEER_ACCESS_ALREADY_ENABLED;
pub const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: mcDrvError_enum = mcDrvError_enum_MC_ERROR_PEER_ACCESS_NOT_ENABLED;
pub const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: mcDrvError_enum = mcDrvError_enum_MC_ERROR_PRIMARY_CONTEXT_ACTIVE;
pub const CUDA_ERROR_CONTEXT_IS_DESTROYED: mcDrvError_enum = mcDrvError_enum_MC_ERROR_CONTEXT_IS_DESTROYED;
pub const CUDA_ERROR_NOT_SUPPORTED: mcDrvError_enum = mcDrvError_enum_MC_ERROR_NOT_SUPPORTED;
pub const CUDA_ERROR_UNKNOWN: mcDrvError_enum = mcDrvError_enum_MC_ERROR_UNKNOWN;

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
    start: mcDrvEvent_t,
    end: mcDrvEvent_t,
) -> mcDrvError_enum {
    unsafe { wcuEventElapsedTime(elapsed_ms, start, end) }
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
