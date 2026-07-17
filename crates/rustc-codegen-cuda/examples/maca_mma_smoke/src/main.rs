/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! C500-native Wave64 16x16x16 FP16 MMA smoke test.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, warp, wmma};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn mma_ones(a: &[u16], b: &[u16], mut output: DisjointSlice<f32>) {
        let lane = warp::lane_id() as usize;
        let row_or_col = lane & 15;
        let group = lane >> 4;
        let a_base = row_or_col * 16 + group * 4;
        let b_base = row_or_col * 16 + group * 4;
        let a_fragment = [
            a[a_base] as u32 | ((a[a_base + 1] as u32) << 16),
            a[a_base + 2] as u32 | ((a[a_base + 3] as u32) << 16),
        ];
        let b_fragment = [
            b[b_base] as u32 | ((b[b_base + 1] as u32) << 16),
            b[b_base + 2] as u32 | ((b[b_base + 3] as u32) << 16),
        ];
        let d = unsafe { wmma::mma_m16n16k16_f32_f16([0.0; 4], a_fragment, b_fragment) };
        for i in 0..4 {
            let row = group * 4 + i;
            unsafe {
                *output.as_mut_ptr().add(row * 16 + row_or_col) = d[i];
            }
        }
    }
}

fn main() {
    const ELEMENTS: usize = 16 * 16;
    const FP16_ONE: u16 = 0x3c00;

    let context = CudaContext::new(0).expect("failed to create C500 context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("failed to load embedded MACA module");
    let a = DeviceBuffer::from_host(&stream, &[FP16_ONE; ELEMENTS]).unwrap();
    let b = DeviceBuffer::from_host(&stream, &[FP16_ONE; ELEMENTS]).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, ELEMENTS).unwrap();
    let launch = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.mma_ones(&stream, launch, &a, &b, &mut output) }
        .expect("mma_ones launch failed");
    let output = output.to_host_vec(&stream).unwrap();
    for (index, value) in output.iter().enumerate() {
        assert_eq!(*value, 16.0, "MMA output element {index}");
    }
    println!("PASS: C500 Wave64 m16n16k16 FP16 MMA produced 16x16 matrix of 16.0");
}
