/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]
#![no_std]

/// Wave size (threads per wave/warp).
///
/// - NVIDIA CUDA: 32 threads per warp
/// - MetaX MXMACA: 64 threads per wave
///
/// This constant is used by `warp::warp_id()` and cooperative groups to
/// compute wave-level indices. `cargo oxide` defines the
/// `cuda_oxide_target_maca` cfg when building device code for the MetaX
/// backend, selecting the hardware-correct value at compile time.
#[cfg(cuda_oxide_target_maca)]
pub const WAVE_SIZE: u32 = 64;

/// Wave size (threads per wave/warp); see the MACA variant above.
#[cfg(not(cuda_oxide_target_maca))]
pub const WAVE_SIZE: u32 = 32;

/// Participation mask for one hardware wave.
///
/// 64 bits wide on every target so kernel source is portable: on NVIDIA
/// (wave = 32) only the low 32 bits are meaningful and the lowering
/// truncates to the hardware's 32-bit mask registers; on MetaX C500 all
/// 64 bits are used.
pub type WaveMask = u64;

pub use cuda_macros::{
    cluster_launch, constant, convergent, cooperative_launch, cuda_module, device, gpu_printf,
    kernel, launch_bounds, launch_contract, ptx_asm, pure, readonly,
};

// Re-export for convenience
pub mod async_copy;
pub mod atomic;
pub mod barrier;
pub mod bf16x2;
pub mod clc;
pub mod cluster;
pub mod constant;
pub mod convert;
pub mod cooperative_groups;
pub mod cusimd;
pub mod debug;
pub mod disjoint;
pub mod dotprod;
pub mod fence;
pub mod grid;
pub mod ptx;
pub mod shared;
pub mod tcgen05;
pub mod thread;
pub mod tma;
pub mod warp;
pub mod wgmma;
pub mod wmma;

pub use barrier::{
    // Core type
    Barrier,
    BarrierToken,
    GeneralBarrier,
    Invalidated,
    // Typestate managed barrier
    ManagedBarrier,
    MmaBarrier,
    MmaBarrierHandle,
    Ready,
    // Kind markers
    TmaBarrier,
    TmaBarrier0,
    TmaBarrier1,
    // Type aliases
    TmaBarrierHandle,
    // State markers
    Uninit,
};
pub use constant::{ConstantMemory, ConstantMemoryValue};
pub use cusimd::{CuSimd, Float2, Float4, TmemRegs4, TmemRegs32};
#[doc(hidden)]
pub use disjoint::__LaunchContractDisjointSlice;
pub use disjoint::DisjointSlice;
pub use fence::*;
pub use shared::{DynamicSharedArray, SharedArray};
pub use tcgen05::{
    TensorMemoryHandle, TmemAddress, TmemDeallocated, TmemF32x4, TmemF32x32, TmemGuard, TmemReady,
    TmemUninit,
};
pub use thread::*;
pub use tma::TmaDescriptor;
