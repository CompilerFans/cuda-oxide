# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

cuda-oxide is a custom rustc backend that compiles GPU kernels written in pure Rust to CUDA PTX. Host and device code live in the same file, built with one `cargo oxide build`. The compilation pipeline is: Rust source → Rust MIR → `dialect-mir` (Pliron IR) → LLVM IR → PTX.

## Build & Test Commands

```bash
# Build and run an example (primary dev workflow)
cargo oxide run vecadd
cargo oxide build vecadd

# Show the full compilation pipeline
cargo oxide pipeline vecadd

# Run under NVIDIA Compute Sanitizer
cargo oxide sanitize vecadd

# Unit tests (workspace crates only, no GPU needed)
cargo test -p cuda-host -p cuda-macros -p llvm-export -p dialect-mir -p dialect-nvvm --lib --tests

# Single crate test
cargo test -p dialect-mir --lib

# Clippy (workspace crates)
cargo clippy --workspace -- -D warnings

# Clippy (codegen backend — separate workspace)
cd crates/rustc-codegen-cuda && cargo clippy --all-targets -- -D warnings

# Format check
cargo oxide fmt --check

# Format
cargo oxide fmt

# Doc build + doctest check (mirrors CI)
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace
cargo test --doc --workspace --exclude cuda-bindings

# All checks (fmt + clippy + test + docs) via just
just check

# Environment validation
cargo oxide doctor
```

The `rustc-codegen-cuda` crate and each example under `crates/rustc-codegen-cuda/examples/` are their own standalone workspaces (not part of the root workspace) because they require `rustc_private` features.

## Architecture

### Compilation Pipeline

The compiler pipeline flows through these crates:

1. **`rustc-codegen-cuda`** — Custom rustc codegen backend (dylib). Loads `#[kernel]` functions from Rust MIR. Contains three large files: `lib.rs` (rustc interface), `collector.rs` (MIR visitor/kernel extraction), `device_codegen.rs` (PTX emission).

2. **`mir-importer`** — Translates Rust MIR into `dialect-mir` operations via Pliron. The `pipeline.rs` file orchestrates the full MIR→PTX pipeline.

3. **`dialect-mir`** — Pliron dialect that models Rust MIR operations, types, and attributes. The `ops/` subdirectory has per-operation modules.

4. **`dialect-nvvm`** — Pliron dialect for NVVM intrinsics (GPU-specific ops like thread indexing, shared memory, barriers).

5. **`mir-lower`** — Lowers `dialect-mir` → LLVM dialect. The `convert/` subdirectory has the DRR-based lowering logic.

6. **`llvm-export`** — Exports Pliron LLVM dialect to textual `.ll` and drives `llc` to produce PTX.

7. **`cuda-oxide-codegen`** — Experimental rustc-independent PTX backend. Accepts `dialect-mir`/`dialect-nvvm` modules directly (no rustc linkage needed).

### User-Facing Crates

- **`cuda-device`** — Device-side intrinsics: `thread::*`, `warp::*`, shared memory, barriers, TMA, clusters, atomics, cooperative groups, tcgen05 (Blackwell tensor cores).
- **`cuda-macros`** — Proc macros: `#[cuda_module]` (embeds device artifact, generates typed launch methods), `#[kernel]` (marks device functions), `gpu_printf!`, inline PTX.
- **`cuda-host`** — Typed module loading, launch helpers, LTOIR loader.
- **`cuda-core`** — Safe RAII wrappers: `CudaContext`, `CudaStream`, `DeviceBuffer<T>`.
- **`cuda-async`** — Async execution: `DeviceOperation`, `DeviceFuture`, `DeviceBox<T>`.
- **`cuda-bindings`** — Raw `bindgen` FFI bindings to `cuda.h`.

### Build Tooling

- **`cargo-oxide`** — Cargo subcommand driving the build. `cargo oxide run|build|pipeline|sanitize|debug|new|fmt|doctor|setup`. Workspace alias in `.cargo/config.toml` routes `cargo oxide` here.

### Key Dependency

Pliron (MLIR-like IR framework in Rust) is pinned to a specific git rev. Both the root workspace and `rustc-codegen-cuda` MUST use the same Pliron rev or types stop unifying.

## Conventions

- **Edition 2024** workspace-wide. Nightly toolchain pinned in `rust-toolchain.toml` (currently `nightly-2026-04-03`).
- **DCO sign-off required** on all commits: `git commit -s`.
- `cargo deny` checks licenses and advisories (`deny.toml`).
- CI runs clippy with `-D warnings` on workspace crates, the codegen backend, and all examples.
- Crates that depend on `cuda-bindings` (transitively) need the CUDA toolkit installed for build-time bindgen.
- `llc` from LLVM 21+ is required for Hopper/Blackwell PTX features. Set `CUDA_OXIDE_LLC` to pin a specific binary.
- The `Nix` flake provides a reproducible dev shell with all dependencies.
