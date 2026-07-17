/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! C500 FP16 tiled GEMM using the native Wave64 m16n16k16 MMA.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp, wmma};
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn load_f16x4(values: &[u16], base: usize) -> [u32; 2] {
        [
            values[base] as u32 | ((values[base + 1] as u32) << 16),
            values[base + 2] as u32 | ((values[base + 3] as u32) << 16),
        ]
    }

    /// Computes row-major A times column-major B into row-major C.
    /// M, N, and K must be positive multiples of 16.
    #[kernel]
    pub fn gemm_f16(a: &[u16], b_col_major: &[u16], mut c: DisjointSlice<f32>, n: u32, k: u32) {
        let lane = warp::lane_id();
        let lane_row_or_col = lane & 15;
        let lane_group = lane >> 4;
        let tiles_n = n / 16;
        let tile = thread::blockIdx_x();
        let tile_m = tile / tiles_n;
        let tile_n = tile - tile_m * tiles_n;
        let row = tile_m * 16 + lane_row_or_col;
        let col = tile_n * 16 + lane_row_or_col;
        let mut accumulator = [0.0f32; 4];
        let mut k_base = 0u32;

        while k_base < k {
            let a_base = row * k + k_base + lane_group * 4;
            let b_base = col * k + k_base + lane_group * 4;
            let a_base = a_base as usize;
            let b_base = b_base as usize;
            let a_fragment = [
                a[a_base] as u32 | ((a[a_base + 1] as u32) << 16),
                a[a_base + 2] as u32 | ((a[a_base + 3] as u32) << 16),
            ];
            let b_fragment = [
                b_col_major[b_base] as u32 | ((b_col_major[b_base + 1] as u32) << 16),
                b_col_major[b_base + 2] as u32 | ((b_col_major[b_base + 3] as u32) << 16),
            ];
            accumulator =
                unsafe { wmma::mma_m16n16k16_f32_f16(accumulator, a_fragment, b_fragment) };
            k_base += 16;
        }

        for i in 0..4 {
            let output_row = tile_m * 16 + lane_group * 4 + i;
            let output_index = (output_row * n + col) as usize;
            unsafe {
                *c.as_mut_ptr().add(output_index) = accumulator[i as usize];
            }
        }
    }

    /// Computes a 2x2 group of 16x16 tiles per wave so A/B fragments are
    /// each reused by two native MMA operations.
    #[kernel]
    pub fn gemm_f16_2x2(a: &[u16], b_col_major: &[u16], mut c: DisjointSlice<f32>, n: u32, k: u32) {
        let lane = warp::lane_id();
        let lane_row_or_col = lane & 15;
        let lane_group = lane >> 4;
        let tile_groups_n = n / 32;
        let tile_group = thread::blockIdx_x();
        let group_m = tile_group / tile_groups_n;
        let group_n = tile_group - group_m * tile_groups_n;
        let row0 = group_m * 32 + lane_row_or_col;
        let row1 = row0 + 16;
        let col0 = group_n * 32 + lane_row_or_col;
        let col1 = col0 + 16;
        let mut c00 = [0.0f32; 4];
        let mut c01 = [0.0f32; 4];
        let mut c10 = [0.0f32; 4];
        let mut c11 = [0.0f32; 4];
        let mut k_base = 0u32;

        while k_base < k {
            let offset = k_base + lane_group * 4;
            let a0 = load_f16x4(a, (row0 * k + offset) as usize);
            let a1 = load_f16x4(a, (row1 * k + offset) as usize);
            let b0 = load_f16x4(b_col_major, (col0 * k + offset) as usize);
            let b1 = load_f16x4(b_col_major, (col1 * k + offset) as usize);
            unsafe {
                c00 = wmma::mma_m16n16k16_f32_f16(c00, a0, b0);
                c01 = wmma::mma_m16n16k16_f32_f16(c01, a0, b1);
                c10 = wmma::mma_m16n16k16_f32_f16(c10, a1, b0);
                c11 = wmma::mma_m16n16k16_f32_f16(c11, a1, b1);
            }
            k_base += 16;
        }

        for i in 0..4 {
            let row_offset = lane_group * 4 + i;
            let output_row0 = group_m * 32 + row_offset;
            let output_row1 = output_row0 + 16;
            unsafe {
                *c.as_mut_ptr().add((output_row0 * n + col0) as usize) = c00[i as usize];
                *c.as_mut_ptr().add((output_row0 * n + col1) as usize) = c01[i as usize];
                *c.as_mut_ptr().add((output_row1 * n + col0) as usize) = c10[i as usize];
                *c.as_mut_ptr().add((output_row1 * n + col1) as usize) = c11[i as usize];
            }
        }
    }
}

fn main() {
    const M: usize = 1024;
    const N: usize = 1024;
    const K: usize = 1024;
    const WARMUP: usize = 5;
    const ITERATIONS: usize = 20;
    const FP16_ONE: u16 = 0x3c00;
    const FP16_TWO: u16 = 0x4000;

    let context = CudaContext::new(0).expect("failed to create C500 context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("failed to load embedded MACA module");
    let mut a_host = vec![0u16; M * K];
    for row in 0..M {
        a_host[row * K + row] = FP16_ONE;
    }
    let mut b_host = vec![0u16; K * N];
    for col in 0..N {
        for row in 0..K {
            b_host[col * K + row] = if (row + col) & 1 == 0 {
                FP16_ONE
            } else {
                FP16_TWO
            };
        }
    }
    let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, M * N).unwrap();
    let launch = LaunchConfig {
        grid_dim: (((M / 32) * (N / 32)) as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    for _ in 0..WARMUP {
        unsafe { module.gemm_f16_2x2(&stream, launch, &a, &b, &mut c, N as u32, K as u32) }
            .expect("GEMM warmup launch failed");
    }
    stream.synchronize().unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe { module.gemm_f16_2x2(&stream, launch, &a, &b, &mut c, N as u32, K as u32) }
            .expect("GEMM launch failed");
    }
    stream.synchronize().unwrap();
    let seconds = start.elapsed().as_secs_f64() / ITERATIONS as f64;

    let output = c.to_host_vec(&stream).unwrap();
    for (index, value) in output.iter().enumerate() {
        let row = index / N;
        let col = index % N;
        let expected = if (row + col) & 1 == 0 { 1.0 } else { 2.0 };
        assert_eq!(*value, expected, "GEMM output element {index}");
    }
    let tflops = (2.0 * M as f64 * N as f64 * K as f64) / seconds / 1.0e12;
    println!("PASS: {M}x{N}x{K} FP16 GEMM, {seconds:.6} s, {tflops:.3} TFLOP/s");
}
