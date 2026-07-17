/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! C500 Wave64 shuffle/vote and atomic smoke test.
//!
//! Build and run with:
//!   cargo oxide run maca_primitives_smoke --target maca --release

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::cooperative_groups::{block_reduce, block_scan, ops::Sum, this_thread_block};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn wave64_primitives(
        mut broadcast: DisjointSlice<u32>,
        mut xor32: DisjointSlice<u32>,
        mut reduction: DisjointSlice<u32>,
        mut ballot: DisjointSlice<u64>,
        mut active: DisjointSlice<u64>,
        mut lane_masks: DisjointSlice<u64>,
    ) {
        let lane = warp::lane_id();
        let value = lane + 1;

        let broadcast_value = warp::shuffle(value, 7);
        let xor_value = warp::shuffle_xor(lane, 32);
        let mut sum = value;
        sum += warp::shuffle_down(sum, 32);
        sum += warp::shuffle_down(sum, 16);
        sum += warp::shuffle_down(sum, 8);
        sum += warp::shuffle_down(sum, 4);
        sum += warp::shuffle_down(sum, 2);
        sum += warp::shuffle_down(sum, 1);
        let ballot_value = warp::ballot((lane & 1) == 0);
        let active_value = warp::active_mask();
        let lane_masks_value = warp::lanemask_lt()
            ^ warp::lanemask_le()
            ^ warp::lanemask_eq()
            ^ warp::lanemask_ge()
            ^ warp::lanemask_gt();

        if let Some(out) = broadcast.get_mut(thread::index_1d()) {
            *out = broadcast_value;
        }
        if let Some(out) = xor32.get_mut(thread::index_1d()) {
            *out = xor_value;
        }
        if let Some(out) = reduction.get_mut(thread::index_1d()) {
            *out = sum;
        }
        if let Some(out) = ballot.get_mut(thread::index_1d()) {
            *out = ballot_value;
        }
        if let Some(out) = active.get_mut(thread::index_1d()) {
            *out = active_value;
        }
        if let Some(out) = lane_masks.get_mut(thread::index_1d()) {
            *out = lane_masks_value;
        }
    }

    #[kernel]
    pub fn atomic_counter(counter: &[u32], mut old_values: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };
        let old = counter.fetch_add(1, AtomicOrdering::Relaxed);
        if let Some(out) = old_values.get_mut(index) {
            *out = old;
        }
    }

    #[kernel]
    pub fn block_collectives(mut reduction: DisjointSlice<u32>, mut scan: DisjointSlice<u32>) {
        static mut REDUCE_SMEM: SharedArray<u32, 2> = SharedArray::UNINIT;
        static mut SCAN_SMEM: SharedArray<u32, 2> = SharedArray::UNINIT;
        let block = this_thread_block();
        let total = block_reduce::<u32, Sum, 2>(&block, 1, &raw mut REDUCE_SMEM);
        let prefix = block_scan::<u32, Sum, 2>(&block, 1, &raw mut SCAN_SMEM);
        if let Some(out) = reduction.get_mut(thread::index_1d()) {
            *out = total;
        }
        if let Some(out) = scan.get_mut(thread::index_1d()) {
            *out = prefix;
        }
    }
}

fn main() {
    const WAVE: usize = 64;
    const ATOMIC_THREADS: usize = 256;
    const BLOCK_THREADS: usize = 128;

    let context = CudaContext::new(0).expect("failed to create C500 context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("failed to load embedded MACA module");

    let mut broadcast = DeviceBuffer::<u32>::zeroed(&stream, WAVE).unwrap();
    let mut xor32 = DeviceBuffer::<u32>::zeroed(&stream, WAVE).unwrap();
    let mut reduction = DeviceBuffer::<u32>::zeroed(&stream, WAVE).unwrap();
    let mut ballot = DeviceBuffer::<u64>::zeroed(&stream, WAVE).unwrap();
    let mut active = DeviceBuffer::<u64>::zeroed(&stream, WAVE).unwrap();
    let mut lane_masks = DeviceBuffer::<u64>::zeroed(&stream, WAVE).unwrap();
    let wave_launch = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (WAVE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.wave64_primitives(
            &stream,
            wave_launch,
            &mut broadcast,
            &mut xor32,
            &mut reduction,
            &mut ballot,
            &mut active,
            &mut lane_masks,
        )
    }
    .expect("wave64_primitives launch failed");

    let broadcast = broadcast.to_host_vec(&stream).unwrap();
    let xor32 = xor32.to_host_vec(&stream).unwrap();
    let reduction = reduction.to_host_vec(&stream).unwrap();
    let ballot = ballot.to_host_vec(&stream).unwrap();
    let active = active.to_host_vec(&stream).unwrap();
    let lane_masks = lane_masks.to_host_vec(&stream).unwrap();
    for lane in 0..WAVE {
        assert_eq!(broadcast[lane], 8, "broadcast lane {lane}");
        assert_eq!(xor32[lane], (lane as u32) ^ 32, "xor lane {lane}");
        assert_eq!(ballot[lane], 0x5555_5555_5555_5555, "ballot lane {lane}");
        assert_eq!(active[lane], u64::MAX, "active mask lane {lane}");
        let eq = 1u64 << lane;
        let lt = eq - 1;
        let le = lt | eq;
        let ge = !lt;
        let gt = !le;
        assert_eq!(
            lane_masks[lane],
            lt ^ le ^ eq ^ ge ^ gt,
            "lane masks {lane}"
        );
    }
    assert_eq!(reduction[0], 2080, "Wave64 reduction");

    let counter = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    let mut old_values = DeviceBuffer::<u32>::zeroed(&stream, ATOMIC_THREADS).unwrap();
    let atomic_launch = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (ATOMIC_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.atomic_counter(&stream, atomic_launch, &counter, &mut old_values) }
        .expect("atomic_counter launch failed");
    let counter_value = counter.to_host_vec(&stream).unwrap()[0];
    let mut old_values = old_values.to_host_vec(&stream).unwrap();
    old_values.sort_unstable();
    assert_eq!(counter_value, ATOMIC_THREADS as u32);
    assert_eq!(old_values, (0..ATOMIC_THREADS as u32).collect::<Vec<_>>());

    let mut block_reduction = DeviceBuffer::<u32>::zeroed(&stream, BLOCK_THREADS).unwrap();
    let mut block_scan = DeviceBuffer::<u32>::zeroed(&stream, BLOCK_THREADS).unwrap();
    let block_launch = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.block_collectives(&stream, block_launch, &mut block_reduction, &mut block_scan)
    }
    .expect("block_collectives launch failed");
    let block_reduction = block_reduction.to_host_vec(&stream).unwrap();
    let block_scan = block_scan.to_host_vec(&stream).unwrap();
    for thread in 0..BLOCK_THREADS {
        assert_eq!(block_reduction[thread], BLOCK_THREADS as u32);
        assert_eq!(block_scan[thread], thread as u32 + 1);
    }

    println!(
        "PASS: Wave64 primitives, {} atomic increments, and {}-thread block reduce/scan",
        ATOMIC_THREADS, BLOCK_THREADS
    );
}
