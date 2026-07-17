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
    pub fn mma_ones(
        f16_values: &[u16],
        bf16_values: &[u16],
        i8_values: &[u8],
        mut f16_output: DisjointSlice<f32>,
        mut bf16_output: DisjointSlice<f32>,
        mut i8_output: DisjointSlice<i32>,
    ) {
        let lane = warp::lane_id() as usize;
        let row_or_col = lane & 15;
        let group = lane >> 4;
        let a_base = row_or_col * 16 + group * 4;
        let f16_fragment = [
            f16_values[a_base] as u32 | ((f16_values[a_base + 1] as u32) << 16),
            f16_values[a_base + 2] as u32 | ((f16_values[a_base + 3] as u32) << 16),
        ];
        let bf16_fragment = [
            bf16_values[a_base] as u32 | ((bf16_values[a_base + 1] as u32) << 16),
            bf16_values[a_base + 2] as u32 | ((bf16_values[a_base + 3] as u32) << 16),
        ];
        let i8_fragment = i8_values[a_base] as u32
            | ((i8_values[a_base + 1] as u32) << 8)
            | ((i8_values[a_base + 2] as u32) << 16)
            | ((i8_values[a_base + 3] as u32) << 24);
        let f16_d = unsafe { wmma::mma_m16n16k16_f32_f16([0.0; 4], f16_fragment, f16_fragment) };
        let bf16_d =
            unsafe { wmma::mma_m16n16k16_f32_bf16([0.0; 4], bf16_fragment, bf16_fragment) };
        let i8_d = unsafe { wmma::mma_m16n16k16_i32_i8([0; 4], i8_fragment, i8_fragment) };
        for i in 0..4 {
            let row = group * 4 + i;
            let output_index = row * 16 + row_or_col;
            unsafe {
                *f16_output.as_mut_ptr().add(output_index) = f16_d[i];
                *bf16_output.as_mut_ptr().add(output_index) = bf16_d[i];
                *i8_output.as_mut_ptr().add(output_index) = i8_d[i];
            }
        }
    }
}

fn main() {
    const ELEMENTS: usize = 16 * 16;
    const FP16_ONE: u16 = 0x3c00;
    const BF16_ONE: u16 = 0x3f80;

    let context = CudaContext::new(0).expect("failed to create C500 context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("failed to load embedded MACA module");
    let f16_values = DeviceBuffer::from_host(&stream, &[FP16_ONE; ELEMENTS]).unwrap();
    let bf16_values = DeviceBuffer::from_host(&stream, &[BF16_ONE; ELEMENTS]).unwrap();
    let i8_values = DeviceBuffer::from_host(&stream, &[1u8; ELEMENTS]).unwrap();
    let mut f16_output = DeviceBuffer::<f32>::zeroed(&stream, ELEMENTS).unwrap();
    let mut bf16_output = DeviceBuffer::<f32>::zeroed(&stream, ELEMENTS).unwrap();
    let mut i8_output = DeviceBuffer::<i32>::zeroed(&stream, ELEMENTS).unwrap();
    let launch = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.mma_ones(
            &stream,
            launch,
            &f16_values,
            &bf16_values,
            &i8_values,
            &mut f16_output,
            &mut bf16_output,
            &mut i8_output,
        )
    }
    .expect("mma_ones launch failed");
    let f16_output = f16_output.to_host_vec(&stream).unwrap();
    let bf16_output = bf16_output.to_host_vec(&stream).unwrap();
    let i8_output = i8_output.to_host_vec(&stream).unwrap();
    for index in 0..ELEMENTS {
        assert_eq!(f16_output[index], 16.0, "FP16 MMA output element {index}");
        assert_eq!(bf16_output[index], 16.0, "BF16 MMA output element {index}");
        assert_eq!(i8_output[index], 16, "INT8 MMA output element {index}");
    }
    println!("PASS: C500 Wave64 m16n16k16 FP16/BF16/INT8 MMA produced 16x16 matrices of 16");
}
