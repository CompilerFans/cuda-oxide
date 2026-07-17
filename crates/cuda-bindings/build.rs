/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{env, error::Error, path::Path, path::PathBuf, process::exit};

/// Environment variables consulted (in order) to locate the CUDA toolkit root.
/// `CUDA_HOME` is the conventional name used by nvcc wrappers and CI images.
const TOOLKIT_ENV_VARS: [&str; 2] = ["CUDA_TOOLKIT_PATH", "CUDA_HOME"];

/// Toolkit root fallback when none of [`TOOLKIT_ENV_VARS`] is set.
const DEFAULT_TOOLKIT_DIR: &str = "/usr/local/cuda";

/// Returns the CUDA toolkit install root: the first set variable among
/// [`TOOLKIT_ENV_VARS`], otherwise [`DEFAULT_TOOLKIT_DIR`]. Used for include
/// paths, library search paths, and bindgen’s Clang configuration.
fn cuda_toolkit_dir() -> String {
    TOOLKIT_ENV_VARS
        .iter()
        .find_map(|var| env::var(var).ok())
        .unwrap_or_else(|| DEFAULT_TOOLKIT_DIR.to_string())
}

/// Runs [`run`]; on error, prints the message and exits with status 1.
fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

/// Configures the crate build: declares rerun triggers, discovers the CUDA
/// include directory, adds native link search paths for `libcuda`, links
/// `cuda`, and invokes bindgen on `wrapper.h` with the discovered include
/// directory, writing `bindings.rs` into `OUT_DIR`.
fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=wrapper.h");
    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-env-changed=MACA_PATH");
    println!("cargo::rustc-check-cfg=cfg(cuda_has_cuEventElapsedTime_v2)");
    println!("cargo::rustc-check-cfg=cfg(cuda_oxide_cu_bridge)");

    let toolkit = cuda_toolkit_dir();
    let include_dir = find_cuda_include_dir(&toolkit)?;
    let using_cu_bridge = is_cu_bridge(&include_dir);
    if using_cu_bridge {
        println!("cargo:rustc-cfg=cuda_oxide_cu_bridge");
    }
    probe_event_elapsed_time_v2(&include_dir.join("cuda.h"));

    for path in collect_lib_paths(&toolkit, using_cu_bridge) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
    if using_cu_bridge {
        let runtime_lib_dir = cu_bridge_runtime_lib_dir(&toolkit).ok_or_else(|| {
            "cuda-bindings: could not find cu-bridge libruntime_cu.so; set MACA_PATH to the MXMACA SDK root"
        })?;
        println!(
            "cargo:rustc-link-search=native={}",
            runtime_lib_dir.display()
        );
        println!("cargo:rustc-link-lib=dylib=runtime_cu");
    }

    bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // CUDA 13.2+ adds types to CUlaunchAttributeValue that bindgen/libclang
        // cannot translate, collapsing the struct to a 1-byte opaque blob while the
        // size assertion still expects the real C size. Making both the struct and its
        // inner union opaque produces correctly-sized byte blobs across CUDA versions.
        // launch_kernel_ex in cuda-core constructs this struct via raw pointer writes.
        .opaque_type("CUlaunchAttribute_st")
        .opaque_type("CUlaunchAttributeValue_union")
        // Block duplicate constants from cu-bridge headers
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_NAN")
        .generate()
        .map_err(|error| format!("cuda-bindings: failed to generate CUDA bindings: {error}"))?
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

/// CUDA toolkit `targets/` directory name for cargo's build `TARGET`, when
/// the toolkit ships one for that architecture. CUDA names these layouts
/// after the GPU platform, not the Rust triple: x86_64 hosts use
/// `x86_64-linux` and aarch64 servers use `sbsa-linux`.
fn toolkit_target_dir() -> Option<&'static str> {
    let target = env::var("TARGET").ok()?;
    match target.split('-').next()? {
        "x86_64" => Some("x86_64-linux"),
        "aarch64" => Some("sbsa-linux"),
        _ => None,
    }
}

/// Returns the include directory containing `cuda.h`: `{toolkit}/include`
/// for standard installs, or `{toolkit}/targets/<dir>/include` for
/// redistributable layouts that have no top-level `include/`.
///
/// Only the single `targets/` directory matching cargo's build `TARGET` is
/// probed. Globbing all of `targets/*` would let another architecture's
/// headers and stubs shadow the right ones on multi-target installs
/// (`sbsa-linux` sorts before `x86_64-linux`).
///
/// A missing `cuda.h` is a hard error here: bindgen cannot run without it,
/// and failing now produces one clean message instead of raw clang
/// diagnostics.
fn find_cuda_include_dir(toolkit: &str) -> Result<PathBuf, String> {
    let base = Path::new(toolkit);
    let mut candidates = vec![base.join("include")];
    if let Some(target_dir) = toolkit_target_dir() {
        candidates.push(base.join("targets").join(target_dir).join("include"));
    }

    if let Some(dir) = candidates.iter().find(|dir| dir.join("cuda.h").is_file()) {
        return Ok(dir.clone());
    }

    let probed: Vec<String> = candidates
        .iter()
        .map(|dir| format!("  {}", dir.join("cuda.h").display()))
        .collect();
    Err(format!(
        "cuda-bindings: could not find cuda.h in the CUDA toolkit at `{toolkit}`.\n\
         Probed:\n\
         {}\n\
         Set CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA Toolkit install root; \
         when neither is set, {DEFAULT_TOOLKIT_DIR} is used.",
        probed.join("\n")
    ))
}

/// Returns whether the discovered headers belong to MXMACA cu-bridge.
///
/// cu-bridge ships its Driver API declarations under `bridge/runtime`; a
/// native CUDA toolkit does not have that directory. Detecting the header
/// layout keeps the public Rust API source-compatible without relying on the
/// installation path containing a particular directory name.
fn is_cu_bridge(include_dir: &Path) -> bool {
    let driver_wrapper = include_dir.join("bridge/runtime/cuda_driver_wrapper.h");
    println!("cargo:rerun-if-changed={}", driver_wrapper.display());
    driver_wrapper.is_file()
}

/// Finds the MXMACA library directory containing cu-bridge's `wcu*` symbols.
///
/// Prefer an explicit `MACA_PATH`. As a fallback, resolve cu-bridge's
/// `libcuda.so` symlink, which points into the SDK library directory.
fn cu_bridge_runtime_lib_dir(toolkit: &str) -> Option<PathBuf> {
    if let Ok(maca_path) = env::var("MACA_PATH") {
        let lib_dir = PathBuf::from(maca_path).join("lib");
        if lib_dir.join("libruntime_cu.so").is_file() {
            return Some(lib_dir);
        }
    }

    let cuda_library = Path::new(toolkit).join("lib/libcuda.so");
    let lib_dir = std::fs::canonicalize(cuda_library)
        .ok()?
        .parent()?
        .to_path_buf();
    lib_dir
        .join("libruntime_cu.so")
        .is_file()
        .then_some(lib_dir)
}

/// Probes the discovered `cuda.h` for `cuEventElapsedTime_v2` and emits the
/// `cuda_has_cuEventElapsedTime_v2` cfg when present.
///
/// CUDA 12.8 renamed the event elapsed-time driver entry point to
/// `cuEventElapsedTime_v2`; earlier toolkits only declare
/// `cuEventElapsedTime`. The cfg lets `src/lib.rs` dispatch to whichever
/// symbol the headers used for this build actually declare.
///
/// A missing `cuda.h` is already a hard error in [`find_cuda_include_dir`];
/// a present but unreadable `cuda.h` stays a warning here (treated as the
/// pre-12.8 spelling) because bindgen reports the authoritative failure
/// right after.
fn probe_event_elapsed_time_v2(cuda_h: &Path) {
    println!("cargo:rerun-if-changed={}", cuda_h.display());
    match std::fs::read_to_string(cuda_h) {
        Ok(header) => {
            if header.contains("cuEventElapsedTime_v2") {
                println!("cargo:rustc-cfg=cuda_has_cuEventElapsedTime_v2");
            }
        }
        Err(error) => {
            println!(
                "cargo:warning=cuda-bindings: failed to probe {}: {error}",
                cuda_h.display()
            );
        }
    }
}

/// Candidate directories for `rustc-link-search=native` when linking against the driver library.
///
/// Adds `{toolkit}/lib64` and `{toolkit}/lib64/stubs` when `lib64` exists. If the build
/// target's `{toolkit}/targets/<dir>/include/cuda.h` exists (redistributable / cross-layout
/// install), also adds that target's `lib` and `lib/stubs`. Only the single `targets/`
/// directory matching cargo's build `TARGET` is considered, never all of `targets/*`.
/// Order is preserved; duplicates are not filtered.
fn collect_lib_paths(toolkit: &str, using_cu_bridge: bool) -> Vec<PathBuf> {
    let base = PathBuf::from(toolkit);
    let mut paths = vec![];

    if using_cu_bridge {
        // cu-bridge installs libcuda.so directly under `<root>/lib`.
        let lib = base.join("lib");
        if lib.is_dir() {
            paths.push(lib);
        }
    }

    let lib64 = base.join("lib64");
    if lib64.is_dir() {
        paths.push(lib64.clone());
        paths.push(lib64.join("stubs"));
    }

    if let Some(target_dir) = toolkit_target_dir() {
        let target_root = base.join("targets").join(target_dir);
        if target_root.join("include/cuda.h").is_file() {
            paths.push(target_root.join("lib"));
            paths.push(target_root.join("lib/stubs"));
        }
    }

    paths
}
