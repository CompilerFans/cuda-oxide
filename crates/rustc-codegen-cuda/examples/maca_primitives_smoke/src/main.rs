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
        mut wide_shuffle: DisjointSlice<u64>,
        mut match32: DisjointSlice<u64>,
        mut match64: DisjointSlice<u64>,
        mut match_all: DisjointSlice<u64>,
        mut redux: DisjointSlice<u32>,
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
        let wide_value = ((lane as u64) << 40) | (lane as u64 + 0x1234);
        let wide_broadcast = warp::shuffle_u64(wide_value, 7);
        let double_xor = warp::shuffle_xor_f64(lane as f64 + 0.5, 32);
        let match32_value = warp::match_any_sync(u64::MAX, lane & 3);
        let match64_value = warp::match_any_i64_sync(u64::MAX, (lane & 7) as u64);
        let match_all_value =
            warp::match_all_sync(u64::MAX, 17) ^ warp::match_all_i64_sync(u64::MAX, lane as u64);
        let redux_values = [
            warp::redux_sync_add(u64::MAX, value),
            warp::redux_sync_min_u32(u64::MAX, 100 - lane),
            warp::redux_sync_min_i32(u64::MAX, lane as i32 - 40) as u32,
            warp::redux_sync_max_u32(u64::MAX, lane),
            warp::redux_sync_max_i32(u64::MAX, lane as i32 - 40) as u32,
            warp::redux_sync_and(u64::MAX, !(1u32 << (lane & 31))),
            warp::redux_sync_or(u64::MAX, 1u32 << (lane & 31)),
            warp::redux_sync_xor(u64::MAX, lane),
            warp::redux_sync_add(0x5555_5555_5555_5555, value),
        ];

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
        if let Some(out) = wide_shuffle.get_mut(thread::index_1d()) {
            *out = wide_broadcast ^ double_xor.to_bits();
        }
        if let Some(out) = match32.get_mut(thread::index_1d()) {
            *out = match32_value;
        }
        if let Some(out) = match64.get_mut(thread::index_1d()) {
            *out = match64_value;
        }
        if let Some(out) = match_all.get_mut(thread::index_1d()) {
            *out = match_all_value;
        }
        for i in 0..redux_values.len() {
            unsafe {
                *redux
                    .as_mut_ptr()
                    .add(thread::index_1d().get() * redux_values.len() + i) = redux_values[i];
            }
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
    let mut wide_shuffle = DeviceBuffer::<u64>::zeroed(&stream, WAVE).unwrap();
    let mut match32 = DeviceBuffer::<u64>::zeroed(&stream, WAVE).unwrap();
    let mut match64 = DeviceBuffer::<u64>::zeroed(&stream, WAVE).unwrap();
    let mut match_all = DeviceBuffer::<u64>::zeroed(&stream, WAVE).unwrap();
    let mut redux = DeviceBuffer::<u32>::zeroed(&stream, WAVE * 9).unwrap();
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
            &mut wide_shuffle,
            &mut match32,
            &mut match64,
            &mut match_all,
            &mut redux,
        )
    }
    .expect("wave64_primitives launch failed");

    let broadcast = broadcast.to_host_vec(&stream).unwrap();
    let xor32 = xor32.to_host_vec(&stream).unwrap();
    let reduction = reduction.to_host_vec(&stream).unwrap();
    let ballot = ballot.to_host_vec(&stream).unwrap();
    let active = active.to_host_vec(&stream).unwrap();
    let lane_masks = lane_masks.to_host_vec(&stream).unwrap();
    let wide_shuffle = wide_shuffle.to_host_vec(&stream).unwrap();
    let match32 = match32.to_host_vec(&stream).unwrap();
    let match64 = match64.to_host_vec(&stream).unwrap();
    let match_all = match_all.to_host_vec(&stream).unwrap();
    let redux = redux.to_host_vec(&stream).unwrap();
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
        let wide_from_lane7 = (7u64 << 40) | (7 + 0x1234);
        let double_from_xor_lane = ((lane ^ 32) as f64 + 0.5).to_bits();
        assert_eq!(
            wide_shuffle[lane],
            wide_from_lane7 ^ double_from_xor_lane,
            "u64/f64 shuffle lane {lane}"
        );
        assert_eq!(
            match32[lane],
            0x1111_1111_1111_1111u64 << (lane & 3),
            "match-any i32 lane {lane}"
        );
        assert_eq!(
            match64[lane],
            0x0101_0101_0101_0101u64 << (lane & 7),
            "match-any i64 lane {lane}"
        );
        assert_eq!(match_all[lane], u64::MAX, "match-all lane {lane}");
        let expected = [
            2080,
            37,
            (-40i32) as u32,
            63,
            23,
            0,
            u32::MAX,
            0,
            if lane & 1 == 0 { 1024 } else { lane as u32 + 1 },
        ];
        assert_eq!(
            &redux[lane * 9..lane * 9 + 9],
            &expected,
            "redux lane {lane}"
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
        "PASS: Wave64 shuffle/vote/match/redux, {} atomic increments, and {}-thread block reduce/scan",
        ATOMIC_THREADS, BLOCK_THREADS
    );
}
