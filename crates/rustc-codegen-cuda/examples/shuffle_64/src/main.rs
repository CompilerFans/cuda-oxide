/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! 64-bit warp shuffles (`u64` / `f64`).
//!
//! PTX `shfl.sync` only moves 32-bit registers — there is no `shfl.sync.*.b64`
//! instruction and no 64-bit shuffle intrinsic in LLVM. cuda-oxide therefore
//! lowers each 64-bit shuffle to one convergent inline-PTX block that splits the
//! value into low/high halves (`mov.b64 {lo, hi}, x`), shuffles each half, and
//! reassembles the result. `f64` shuffles reuse the `u64` path via a bitcast.
//!
//! Three kernels, one per shuffle mode family:
//!   1. `shuffle_u64_broadcast` — `idx`: broadcast one lane's `u64` to the warp.
//!      The value has distinct high/low 32-bit halves so a reassembly bug shows.
//!   2. `shuffle_f64_butterfly_sum` — `bfly`: full-warp `f64` reduction; every
//!      lane ends with the warp sum.
//!   3. `shuffle_u64_neighbor` — `down`/`up`: read the next / previous lane,
//!      with edge lanes keeping their own value (the PTX out-of-range rule).
//!
//! Build and run with:
//!   cargo oxide run shuffle_64

use cuda_device::{DisjointSlice, WAVE_SIZE, kernel, warp};
use cuda_host::cuda_module;

/// Lane whose value is broadcast in kernel 1.
const SRC_LANE: u32 = 5;
/// High-word tag for the broadcast value (distinct from the low word).
const TAG: u64 = 0xABCD_0000;
/// High half for the neighbor-exchange value.
const NEIGHBOR_HI: u64 = 0xDEAD_0000_0000_0000;
/// High half for the masked half-warp value.
const HALF_HI: u64 = 0xF00D_0000_0000_0000;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// `idx` mode: every lane builds a `u64` with *different* high and low words
    /// (`(lane << 32) | (TAG + lane)`) then reads lane `SRC_LANE`'s value. If the
    /// lo/hi split-and-reassemble were wrong, the halves would not match.
    ///
    /// Expected (single warp): every entry equals
    /// `((SRC_LANE as u64) << 32) | (TAG + SRC_LANE)`.
    #[kernel]
    pub fn shuffle_u64_broadcast(mut out: DisjointSlice<u64>) {
        let lane = warp::lane_id();
        let val: u64 = ((lane as u64) << 32) | (TAG + lane as u64);
        let got = warp::shuffle_u64(val, SRC_LANE);
        unsafe {
            *out.get_unchecked_mut(lane as usize) = got;
        }
    }

    /// `bfly` mode on `f64`: a classic butterfly reduction. Each lane seeds
    /// `lane + 1` and XOR-shuffles with strides `WAVE_SIZE/2 .. 1`, so every
    /// lane ends with the full-warp sum `1 + 2 + ... + WAVE_SIZE`.
    #[kernel]
    pub fn shuffle_f64_butterfly_sum(mut out: DisjointSlice<f64>) {
        let lane = warp::lane_id();
        let mut v = (lane as f64) + 1.0;
        let mut stride = WAVE_SIZE / 2;
        while stride >= 1 {
            v += warp::shuffle_xor_f64(v, stride);
            stride >>= 1;
        }
        unsafe {
            *out.get_unchecked_mut(lane as usize) = v;
        }
    }

    /// `down`/`up` modes: each lane holds `NEIGHBOR_HI | lane` and reads its
    /// upper (`down`, lane+1) and lower (`up`, lane-1) neighbor. The PTX
    /// out-of-range rule means lane 31's `down` and lane 0's `up` keep their own
    /// value.
    #[kernel]
    pub fn shuffle_u64_neighbor(mut down_out: DisjointSlice<u64>, mut up_out: DisjointSlice<u64>) {
        let lane = warp::lane_id();
        let val: u64 = NEIGHBOR_HI | (lane as u64);
        let down = warp::shuffle_down_u64(val, 1);
        let up = warp::shuffle_up_u64(val, 1);
        unsafe {
            *down_out.get_unchecked_mut(lane as usize) = down;
            *up_out.get_unchecked_mut(lane as usize) = up;
        }
    }

    /// Masked `idx`: the warp splits into two halves that each shuffle
    /// *independently*. In each divergent branch `active_mask()` yields just that
    /// half's membermask, and every member broadcasts its leader's value — lane 0
    /// for the lower half, lane `WAVE_SIZE/2` for the upper half (src_lane is
    /// absolute, not relative to the mask). If the membermask operand were
    /// mis-wired, the two halves would bleed into each other.
    ///
    /// Expected: `out[0..H] == HALF_HI | 0` and `out[H..WAVE_SIZE] == HALF_HI | H`
    /// where `H = WAVE_SIZE / 2`.
    #[kernel]
    pub fn shuffle_u64_halfwarp(mut out: DisjointSlice<u64>) {
        let lane = warp::lane_id();
        let val: u64 = HALF_HI | (lane as u64);
        let half = WAVE_SIZE / 2;
        let got = if lane < half {
            // Lower half: broadcast lane 0 within this subset.
            let mask = warp::active_mask();
            warp::shuffle_u64_sync(mask, val, 0)
        } else {
            // Upper half: broadcast the half's leader within this subset.
            let mask = warp::active_mask();
            warp::shuffle_u64_sync(mask, val, half)
        };
        unsafe {
            *out.get_unchecked_mut(lane as usize) = got;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

const WARP: usize = WAVE_SIZE as usize;

fn main() {
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    println!("=== 64-bit warp shuffle (u64 / f64) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    let module = ctx
        .load_module_from_file("shuffle_64.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    // A single warp is enough to demonstrate the shuffle semantics.
    let cfg = LaunchConfig {
        block_dim: (WARP as u32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut failed = false;

    // ===== Test 1: u64 broadcast (idx) =====
    println!("\n--- Test 1: shuffle_u64 (idx broadcast of lane {SRC_LANE}) ---");
    let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, WARP).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.shuffle_u64_broadcast((stream).as_ref(), cfg, &mut out_dev) }
        .expect("Kernel launch failed");
    let out = out_dev.to_host_vec(&stream).unwrap();

    let want = ((SRC_LANE as u64) << 32) | (TAG + SRC_LANE as u64);
    println!("out[0]   = {:#018x} (expected {:#018x})", out[0], want);
    println!("out[{}]  = {:#018x}", WARP - 1, out[WARP - 1]);
    if out.iter().all(|&v| v == want) {
        println!("✓ every lane received lane {SRC_LANE}'s full 64-bit value");
    } else {
        println!("✗ broadcast mismatch (high/low half reassembly bug?)");
        failed = true;
    }

    // ===== Test 2: f64 butterfly reduction (bfly) =====
    println!("\n--- Test 2: shuffle_xor_f64 (butterfly sum) ---");
    let mut sum_dev = DeviceBuffer::<f64>::zeroed(&stream, WARP).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.shuffle_f64_butterfly_sum((stream).as_ref(), cfg, &mut sum_dev) }
        .expect("Kernel launch failed");
    let sums = sum_dev.to_host_vec(&stream).unwrap();

    let want_sum = (1..=WARP).sum::<usize>() as f64;
    println!("sum[0]   = {} (expected {})", sums[0], want_sum);
    if sums.iter().all(|&v| (v - want_sum).abs() < 1e-9) {
        println!("✓ every lane holds the full-warp f64 sum");
    } else {
        println!("✗ butterfly reduction mismatch!");
        failed = true;
    }

    // ===== Test 3: u64 neighbor exchange (down / up) =====
    println!("\n--- Test 3: shuffle_down_u64 / shuffle_up_u64 (delta 1) ---");
    let mut down_dev = DeviceBuffer::<u64>::zeroed(&stream, WARP).unwrap();
    let mut up_dev = DeviceBuffer::<u64>::zeroed(&stream, WARP).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.shuffle_u64_neighbor((stream).as_ref(), cfg, &mut down_dev, &mut up_dev) }
        .expect("Kernel launch failed");
    let down = down_dev.to_host_vec(&stream).unwrap();
    let up = up_dev.to_host_vec(&stream).unwrap();

    // down[L] reads lane L+1; the top lane is out of range and keeps its own value.
    let want_down: Vec<u64> = (0..WARP)
        .map(|l| NEIGHBOR_HI | (if l + 1 < WARP { l + 1 } else { l }) as u64)
        .collect();
    // up[L] reads lane L-1; lane 0 is out of range and keeps its own value.
    let want_up: Vec<u64> = (0..WARP)
        .map(|l| NEIGHBOR_HI | (if l >= 1 { l - 1 } else { l }) as u64)
        .collect();

    println!(
        "down[0]  = {:#018x} (expected {:#018x})",
        down[0], want_down[0]
    );
    println!(
        "down[{}] = {:#018x} (expected {:#018x}, edge keeps own)",
        WARP - 1,
        down[WARP - 1],
        want_down[WARP - 1]
    );
    println!(
        "up[0]    = {:#018x} (expected {:#018x}, edge keeps own)",
        up[0], want_up[0]
    );
    println!(
        "up[{}]   = {:#018x} (expected {:#018x})",
        WARP - 1,
        up[WARP - 1],
        want_up[WARP - 1]
    );
    if down == want_down && up == want_up {
        println!("✓ neighbor exchange shifted 64-bit values correctly");
    } else {
        println!("✗ neighbor exchange mismatch!");
        failed = true;
    }

    // ===== Test 4: masked half-warp broadcast (subset membermask) =====
    let h = WARP / 2;
    println!("\n--- Test 4: shuffle_u64_sync (two independent {h}-lane halves) ---");
    let mut half_dev = DeviceBuffer::<u64>::zeroed(&stream, WARP).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.shuffle_u64_halfwarp((stream).as_ref(), cfg, &mut half_dev) }
        .expect("Kernel launch failed");
    let half = half_dev.to_host_vec(&stream).unwrap();

    // Each half broadcasts its leader: lanes 0..h -> lane 0, lanes h..WARP -> lane h.
    let want_half: Vec<u64> = (0..WARP)
        .map(|l| HALF_HI | (if l < h { 0 } else { h }) as u64)
        .collect();
    println!(
        "half[0]  = {:#018x} (expected {:#018x})",
        half[0], want_half[0]
    );
    println!(
        "half[{h}] = {:#018x} (expected {:#018x})",
        half[h], want_half[h]
    );
    if half == want_half {
        println!("✓ each half shuffled within its own membermask (no cross-talk)");
    } else {
        println!("✗ masked half-warp broadcast mismatch (membermask mis-wired?)");
        failed = true;
    }

    if failed {
        std::process::exit(1);
    }
    println!("\nSUCCESS: 64-bit warp shuffles produced correct results");
}
