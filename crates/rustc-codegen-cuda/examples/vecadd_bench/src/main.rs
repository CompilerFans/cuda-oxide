/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! C500 bandwidth benchmark for Oxide FP32 vector addition.
//!
//! The timed region contains kernel launches only. Effective bandwidth counts
//! two FP32 reads and one FP32 write: `12 * elements` bytes per launch.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::error::Error;

const DEFAULT_MAX_ELEMENTS: usize = 1 << 26;
const DEFAULT_WARMUP: usize = 10;
const DEFAULT_BASE_ITERS: usize = 100;
const DEFAULT_SAMPLES: usize = 5;
const TARGET_BYTES_PER_SAMPLE: u64 = 32 * 1024 * 1024 * 1024;
const C500_NOMINAL_GBPS: f64 = 1832.0;
const MCPYTORCH_REFERENCE_GBPS: f64 = 1489.8;

#[cuda_module]
mod kernels {
    use super::*;

    /// Report CUDA launch dimensions so the benchmark also validates the
    /// MACA dispatch-packet lowering used by grid-stride kernels.
    #[kernel]
    pub fn report_launch_dimensions(mut out: DisjointSlice<u32>) {
        if thread::threadIdx_x() == 0
            && thread::threadIdx_y() == 0
            && thread::threadIdx_z() == 0
            && thread::blockIdx_x() == 0
            && thread::blockIdx_y() == 0
            && thread::blockIdx_z() == 0
        {
            let out = out.as_mut_ptr();
            // SAFETY: exactly one thread enters this branch and the host
            // supplies a six-element output buffer.
            unsafe {
                out.write(thread::gridDim_x());
                out.add(1).write(thread::gridDim_y());
                out.add(2).write(thread::gridDim_z());
                out.add(3).write(thread::blockDim_x());
                out.add(4).write(thread::blockDim_y());
                out.add(5).write(thread::blockDim_z());
            }
        }
    }

    /// The unchanged one-thread-per-element algorithm used by the official
    /// vecadd example. This is the Oxide scalar baseline.
    #[kernel]
    pub fn vecadd_direct(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }

    /// Fixed-grid scalar kernel. Each thread walks a disjoint grid-stride
    /// sequence, avoiding short-lived blocks. The induction variable stays
    /// 32-bit to match C500's native launch-index width.
    #[kernel]
    pub fn vecadd_scalar_grid(
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
        n: u32,
        stride: u32,
    ) {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();
        let mut i = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());

        while i < n {
            let offset = i as usize;
            // SAFETY: i < n, host guarantees n <= every buffer length, and
            // grid-stride sequences are disjoint for distinct thread IDs.
            unsafe {
                c_ptr
                    .add(offset)
                    .write(*a_ptr.add(offset) + *b_ptr.add(offset));
            }
            i = i.wrapping_add(stride);
        }
    }

    /// The same fixed-grid scalar path with four independent operations per
    /// loop body to expose memory-level parallelism and reduce loop overhead.
    #[kernel]
    pub fn vecadd_scalar_grid_u4(
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
        n: u32,
        stride: u32,
    ) {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();
        let step = stride.wrapping_mul(4);
        let mut i = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());

        while i < n {
            let offset = i as usize;
            // SAFETY: each lane below is bounds checked; adding a multiple of
            // the global thread count preserves disjointness across threads.
            unsafe {
                c_ptr
                    .add(offset)
                    .write(*a_ptr.add(offset) + *b_ptr.add(offset));
                let i1 = i.wrapping_add(stride);
                if i1 < n {
                    let offset = i1 as usize;
                    c_ptr
                        .add(offset)
                        .write(*a_ptr.add(offset) + *b_ptr.add(offset));
                }
                let i2 = i1.wrapping_add(stride);
                if i2 < n {
                    let offset = i2 as usize;
                    c_ptr
                        .add(offset)
                        .write(*a_ptr.add(offset) + *b_ptr.add(offset));
                }
                let i3 = i2.wrapping_add(stride);
                if i3 < n {
                    let offset = i3 as usize;
                    c_ptr
                        .add(offset)
                        .write(*a_ptr.add(offset) + *b_ptr.add(offset));
                }
            }
            i = i.wrapping_add(step);
        }
    }

    /// Explicit 128-bit path over the original FP32 buffers. Device allocations
    /// are at least 16-byte aligned and every chunk begins at four FP32 values.
    #[kernel]
    pub fn vecadd_packed128_grid(
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
        n: usize,
        chunk_stride: usize,
    ) {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();
        let tid = thread::index_1d().get();
        let chunks = n / 4;
        let mut chunk = tid;

        while chunk < chunks {
            // SAFETY: allocation bases are 16-byte aligned, chunk offsets are
            // multiples of 16 bytes, and grid-stride sequences are disjoint.
            unsafe {
                add128(a_ptr, b_ptr, c_ptr, chunk);
            }
            chunk += chunk_stride;
        }

        let tail = chunks * 4 + tid;
        if tail < n {
            // SAFETY: tail is in bounds and only one thread owns this index.
            unsafe {
                c_ptr.add(tail).write(*a_ptr.add(tail) + *b_ptr.add(tail));
            }
        }
    }

    /// Four-chunk unrolled variant of the explicit 128-bit path.
    #[kernel]
    pub fn vecadd_packed128_grid_u4(
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
        n: usize,
        chunk_stride: usize,
    ) {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();
        let tid = thread::index_1d().get();
        let chunks = n / 4;
        let step = chunk_stride * 4;
        let mut chunk = tid;

        while chunk < chunks {
            // SAFETY: every chunk index is checked and each thread owns a
            // unique grid-stride sequence.
            unsafe {
                add128(a_ptr, b_ptr, c_ptr, chunk);
                let i1 = chunk + chunk_stride;
                if i1 < chunks {
                    add128(a_ptr, b_ptr, c_ptr, i1);
                }
                let i2 = i1 + chunk_stride;
                if i2 < chunks {
                    add128(a_ptr, b_ptr, c_ptr, i2);
                }
                let i3 = i2 + chunk_stride;
                if i3 < chunks {
                    add128(a_ptr, b_ptr, c_ptr, i3);
                }
            }
            chunk += step;
        }

        let tail = chunks * 4 + tid;
        if tail < n {
            // SAFETY: tail is in bounds and only one thread owns this index.
            unsafe {
                c_ptr.add(tail).write(*a_ptr.add(tail) + *b_ptr.add(tail));
            }
        }
    }

    #[inline(always)]
    unsafe fn add128(a: *const f32, b: *const f32, c: *mut f32, chunk: usize) {
        let base = chunk * 4;
        unsafe {
            let av = (a.add(base) as *const u128).read();
            let bv = (b.add(base) as *const u128).read();
            let r0 = f32::from_bits(av as u32) + f32::from_bits(bv as u32);
            let r1 = f32::from_bits((av >> 32) as u32) + f32::from_bits((bv >> 32) as u32);
            let r2 = f32::from_bits((av >> 64) as u32) + f32::from_bits((bv >> 64) as u32);
            let r3 = f32::from_bits((av >> 96) as u32) + f32::from_bits((bv >> 96) as u32);
            let packed = u128::from(r0.to_bits())
                | (u128::from(r1.to_bits()) << 32)
                | (u128::from(r2.to_bits()) << 64)
                | (u128::from(r3.to_bits()) << 96);
            (c.add(base) as *mut u128).write(packed);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GridConfig {
    blocks: u32,
    threads: u32,
}

impl GridConfig {
    fn launch(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.blocks, 1, 1),
            block_dim: (self.threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

#[derive(Clone, Debug)]
struct Timing {
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    iterations: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let device_id = env_usize("VECADD_DEVICE", 0);
    let max_elements = env_usize("VECADD_MAX_ELEMENTS", DEFAULT_MAX_ELEMENTS);
    let warmup = env_usize("VECADD_WARMUP", DEFAULT_WARMUP);
    let base_iters = env_usize("VECADD_ITERS", DEFAULT_BASE_ITERS);
    let samples = env_usize("VECADD_SAMPLES", DEFAULT_SAMPLES);
    if max_elements == 0 || max_elements > u32::MAX as usize {
        return Err("VECADD_MAX_ELEMENTS must be in 1..=u32::MAX".into());
    }
    if base_iters == 0 || samples == 0 {
        return Err("VECADD_ITERS and VECADD_SAMPLES must be non-zero".into());
    }

    let ctx = CudaContext::new(device_id)?;
    let stream = ctx.default_stream();
    let device_name = ctx.device_name()?;
    let aps = ctx.multiprocessor_count()?;
    let module = kernels::load(&ctx)?;

    println!("Oxide C500 FP32 vecadd bandwidth benchmark");
    println!("device       : {device_id} ({device_name})");
    println!("APs          : {aps}");
    println!(
        "max elements : {max_elements} ({:.1} MiB per array)",
        (max_elements * core::mem::size_of::<f32>()) as f64 / (1 << 20) as f64
    );
    println!("timing       : GPU events, {warmup} warmup, {samples} median samples");
    println!("byte model   : 12 bytes/element (2 reads + 1 write)\n");

    let configs = tuning_configs(aps);
    let u32_configs = u32_tuning_configs(aps);
    let max_u32_stride = u32_configs
        .iter()
        .map(|config| config.blocks * config.threads)
        .max()
        .expect("non-empty u32 tuning configs");
    let max_u32_step = max_u32_stride
        .checked_mul(4)
        .ok_or("C500 launch geometry exceeds the u32 index range")?;
    let max_safe_elements = u32::MAX - max_u32_step + 1;
    if max_elements > max_safe_elements as usize {
        return Err(format!(
            "VECADD_MAX_ELEMENTS must be <= {max_safe_elements} for the u32 grid-stride kernels"
        )
        .into());
    }
    verify_launch_dimensions(&module, &stream)?;
    println!("launch dims  : PASS (grid 7x3x2, block 32x4x2)");
    verify_all(&module, &stream, u32_configs[0])?;
    println!("correctness  : PASS (1,000,003 logical elements)\n");

    println!("Allocating scalar buffers...");
    let scalar_a_host = vec![1.25_f32; max_elements];
    let scalar_a = DeviceBuffer::from_host(&stream, &scalar_a_host)?;
    drop(scalar_a_host);
    let scalar_b_host = vec![2.5_f32; max_elements];
    let scalar_b = DeviceBuffer::from_host(&stream, &scalar_b_host)?;
    drop(scalar_b_host);
    let mut scalar_c = DeviceBuffer::<f32>::zeroed(&stream, max_elements)?;
    stream.synchronize()?;
    assert_eq!(
        scalar_a.cu_deviceptr() & 15,
        0,
        "packed A pointer alignment"
    );
    assert_eq!(
        scalar_b.cu_deviceptr() & 15,
        0,
        "packed B pointer alignment"
    );
    assert_eq!(
        scalar_c.cu_deviceptr() & 15,
        0,
        "packed C pointer alignment"
    );

    println!("\nTuning scalar grid-stride kernels at N={max_elements}");
    let scalar_grid = tune_scalar_grid(
        &module,
        &stream,
        &scalar_a,
        &scalar_b,
        &mut scalar_c,
        max_elements,
        &u32_configs,
        warmup,
        base_iters,
        samples,
        false,
    )?;
    let scalar_u4 = tune_scalar_grid(
        &module,
        &stream,
        &scalar_a,
        &scalar_b,
        &mut scalar_c,
        max_elements,
        &u32_configs,
        warmup,
        base_iters,
        samples,
        true,
    )?;
    println!("\nTuning explicit packed-128 kernels at N={max_elements}");
    let packed_grid = tune_packed_grid(
        &module,
        &stream,
        &scalar_a,
        &scalar_b,
        &mut scalar_c,
        max_elements,
        &configs,
        warmup,
        base_iters,
        samples,
        false,
    )?;
    let packed_u4 = tune_packed_grid(
        &module,
        &stream,
        &scalar_a,
        &scalar_b,
        &mut scalar_c,
        max_elements,
        &configs,
        warmup,
        base_iters,
        samples,
        true,
    )?;

    let sizes = benchmark_sizes(max_elements);
    println!("\nPerformance curve (median of {samples} samples)");
    println!(
        "{:<20} {:>11} {:>10} {:>11} {:>9} {:>9} {:>13}",
        "kernel", "elements", "ms", "GB/s", "nominal", "torch", "grid"
    );

    let mut large_results = Vec::new();
    for &n in &sizes {
        let direct = benchmark(&stream, n, warmup, base_iters, samples, || unsafe {
            module.vecadd_direct(
                &stream,
                LaunchConfig::for_num_elems(n as u32),
                &scalar_a,
                &scalar_b,
                &mut scalar_c,
            )
        })?;
        print_result("direct", n, None, &direct);
        if n == max_elements {
            large_results.push(("direct", direct.clone(), None));
        }

        let scalar = benchmark(&stream, n, warmup, base_iters, samples, || unsafe {
            let stride = scalar_grid.blocks * scalar_grid.threads;
            module.vecadd_scalar_grid(
                &stream,
                scalar_grid.launch(),
                &scalar_a,
                &scalar_b,
                &mut scalar_c,
                n as u32,
                stride,
            )
        })?;
        print_result("scalar-grid", n, Some(scalar_grid), &scalar);
        if n == max_elements {
            large_results.push(("scalar-grid", scalar.clone(), Some(scalar_grid)));
        }

        let scalar4 = benchmark(&stream, n, warmup, base_iters, samples, || unsafe {
            let stride = scalar_u4.blocks * scalar_u4.threads;
            module.vecadd_scalar_grid_u4(
                &stream,
                scalar_u4.launch(),
                &scalar_a,
                &scalar_b,
                &mut scalar_c,
                n as u32,
                stride,
            )
        })?;
        print_result("scalar-grid-u4", n, Some(scalar_u4), &scalar4);
        if n == max_elements {
            large_results.push(("scalar-grid-u4", scalar4.clone(), Some(scalar_u4)));
        }

        let packed = benchmark(&stream, n, warmup, base_iters, samples, || unsafe {
            module.vecadd_packed128_grid(
                &stream,
                packed_grid.launch(),
                &scalar_a,
                &scalar_b,
                &mut scalar_c,
                n,
                packed_grid.blocks as usize * packed_grid.threads as usize,
            )
        })?;
        print_result("packed128-grid", n, Some(packed_grid), &packed);
        if n == max_elements {
            large_results.push(("packed128-grid", packed.clone(), Some(packed_grid)));
        }

        let packed4 = benchmark(&stream, n, warmup, base_iters, samples, || unsafe {
            module.vecadd_packed128_grid_u4(
                &stream,
                packed_u4.launch(),
                &scalar_a,
                &scalar_b,
                &mut scalar_c,
                n,
                packed_u4.blocks as usize * packed_u4.threads as usize,
            )
        })?;
        print_result("packed128-grid-u4", n, Some(packed_u4), &packed4);
        if n == max_elements {
            large_results.push(("packed128-grid-u4", packed4.clone(), Some(packed_u4)));
        }
    }

    let direct_ms = large_results
        .iter()
        .find(|(name, _, _)| *name == "direct")
        .expect("direct result")
        .1
        .median_ms;
    large_results.sort_by(|a, b| a.1.median_ms.total_cmp(&b.1.median_ms));
    let (best_name, best_timing, best_config) = &large_results[0];
    let best_gbps = bandwidth_gbps(max_elements, best_timing.median_ms);
    println!("\nBest HBM-size result");
    println!("kernel       : {best_name}");
    if let Some(config) = best_config {
        println!(
            "launch       : {} blocks x {} threads",
            config.blocks, config.threads
        );
    }
    println!(
        "latency      : {:.4} ms (min {:.4}, max {:.4})",
        best_timing.median_ms, best_timing.min_ms, best_timing.max_ms
    );
    println!("bandwidth    : {:.1} GB/s", best_gbps);
    println!(
        "vs direct    : {:.2}x ({:.1}% higher bandwidth)",
        direct_ms / best_timing.median_ms,
        100.0 * (direct_ms / best_timing.median_ms - 1.0)
    );
    println!(
        "C500 nominal : {:.1}% of {:.0} GB/s",
        100.0 * best_gbps / C500_NOMINAL_GBPS,
        C500_NOMINAL_GBPS
    );
    println!(
        "mcPyTorch ref: {:.1}% of {:.0} GB/s",
        100.0 * best_gbps / MCPYTORCH_REFERENCE_GBPS,
        MCPYTORCH_REFERENCE_GBPS
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tune_scalar_grid(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
    n: usize,
    configs: &[GridConfig],
    warmup: usize,
    base_iters: usize,
    samples: usize,
    unroll4: bool,
) -> Result<GridConfig, Box<dyn Error>> {
    let label = if unroll4 { "scalar-u4" } else { "scalar" };
    let mut best = None::<(GridConfig, f64)>;
    for &config in configs {
        let timing = benchmark(stream, n, warmup, base_iters, samples, || unsafe {
            if unroll4 {
                module.vecadd_scalar_grid_u4(
                    stream,
                    config.launch(),
                    a,
                    b,
                    c,
                    n as u32,
                    config.blocks * config.threads,
                )
            } else {
                module.vecadd_scalar_grid(
                    stream,
                    config.launch(),
                    a,
                    b,
                    c,
                    n as u32,
                    config.blocks * config.threads,
                )
            }
        })?;
        let gbps = bandwidth_gbps(n, timing.median_ms);
        println!(
            "  {label:<10} {:>4} blocks/AP x {:>3} threads: {:>8.1} GB/s ({:.4} ms)",
            config.blocks / (configs[0].blocks / 4),
            config.threads,
            gbps,
            timing.median_ms
        );
        if best.is_none_or(|(_, ms)| timing.median_ms < ms) {
            best = Some((config, timing.median_ms));
        }
    }
    Ok(best.expect("non-empty tuning config").0)
}

#[allow(clippy::too_many_arguments)]
fn tune_packed_grid(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
    n: usize,
    configs: &[GridConfig],
    warmup: usize,
    base_iters: usize,
    samples: usize,
    unroll4: bool,
) -> Result<GridConfig, Box<dyn Error>> {
    let label = if unroll4 { "packed-u4" } else { "packed128" };
    let mut best = None::<(GridConfig, f64)>;
    for &config in configs {
        let chunk_stride = config.blocks as usize * config.threads as usize;
        let timing = benchmark(stream, n, warmup, base_iters, samples, || unsafe {
            if unroll4 {
                module.vecadd_packed128_grid_u4(stream, config.launch(), a, b, c, n, chunk_stride)
            } else {
                module.vecadd_packed128_grid(stream, config.launch(), a, b, c, n, chunk_stride)
            }
        })?;
        let gbps = bandwidth_gbps(n, timing.median_ms);
        println!(
            "  {label:<10} {:>4} blocks/AP x {:>3} threads: {:>8.1} GB/s ({:.4} ms)",
            config.blocks / (configs[0].blocks / 4),
            config.threads,
            gbps,
            timing.median_ms
        );
        if best.is_none_or(|(_, ms)| timing.median_ms < ms) {
            best = Some((config, timing.median_ms));
        }
    }
    Ok(best.expect("non-empty tuning config").0)
}

fn benchmark<F>(
    stream: &CudaStream,
    elements: usize,
    warmup: usize,
    base_iters: usize,
    samples: usize,
    mut launch: F,
) -> Result<Timing, Box<dyn Error>>
where
    F: FnMut() -> Result<(), cuda_core::DriverError>,
{
    for _ in 0..warmup {
        launch()?;
    }
    stream.synchronize()?;

    let per_launch = bytes(elements) as u64;
    let adaptive = TARGET_BYTES_PER_SAMPLE.div_ceil(per_launch) as usize;
    let iterations = base_iters.max(adaptive).min(10_000);
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start =
            stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        for _ in 0..iterations {
            launch()?;
        }
        let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        times.push(start.elapsed_ms(&end)? as f64 / iterations as f64);
    }
    times.sort_by(f64::total_cmp);
    Ok(Timing {
        median_ms: times[times.len() / 2],
        min_ms: times[0],
        max_ms: times[times.len() - 1],
        iterations,
    })
}

fn verify_all(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: GridConfig,
) -> Result<(), Box<dyn Error>> {
    const N: usize = 1_000_003;
    let a_host: Vec<f32> = (0..N).map(input_a).collect();
    let b_host: Vec<f32> = (0..N).map(input_b).collect();
    let a = DeviceBuffer::from_host(stream, &a_host)?;
    let b = DeviceBuffer::from_host(stream, &b_host)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, N)?;

    unsafe {
        module.vecadd_direct(
            stream,
            LaunchConfig::for_num_elems(N as u32),
            &a,
            &b,
            &mut c,
        )?;
    }
    check_scalar(&c.to_host_vec(stream)?).map_err(|error| format!("vecadd_direct: {error}"))?;

    let stride = config.blocks as usize * config.threads as usize;
    c = DeviceBuffer::<f32>::zeroed(stream, N)?;
    unsafe {
        module.vecadd_scalar_grid(
            stream,
            config.launch(),
            &a,
            &b,
            &mut c,
            N as u32,
            config.blocks * config.threads,
        )?;
    };
    check_scalar(&c.to_host_vec(stream)?)
        .map_err(|error| format!("vecadd_scalar_grid: {error}"))?;

    c = DeviceBuffer::<f32>::zeroed(stream, N)?;
    unsafe {
        module.vecadd_scalar_grid_u4(
            stream,
            config.launch(),
            &a,
            &b,
            &mut c,
            N as u32,
            config.blocks * config.threads,
        )?;
    };
    check_scalar(&c.to_host_vec(stream)?)
        .map_err(|error| format!("vecadd_scalar_grid_u4: {error}"))?;

    assert_eq!(a.cu_deviceptr() & 15, 0, "packed A pointer alignment");
    assert_eq!(b.cu_deviceptr() & 15, 0, "packed B pointer alignment");
    c = DeviceBuffer::<f32>::zeroed(stream, N)?;
    assert_eq!(c.cu_deviceptr() & 15, 0, "packed C pointer alignment");
    unsafe {
        module.vecadd_packed128_grid(stream, config.launch(), &a, &b, &mut c, N, stride)?;
    }
    check_scalar(&c.to_host_vec(stream)?)
        .map_err(|error| format!("vecadd_packed128_grid: {error}"))?;

    c = DeviceBuffer::<f32>::zeroed(stream, N)?;
    assert_eq!(c.cu_deviceptr() & 15, 0, "packed C pointer alignment");
    unsafe {
        module.vecadd_packed128_grid_u4(stream, config.launch(), &a, &b, &mut c, N, stride)?;
    }
    check_scalar(&c.to_host_vec(stream)?)
        .map_err(|error| format!("vecadd_packed128_grid_u4: {error}"))?;
    Ok(())
}

fn verify_launch_dimensions(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    const EXPECTED: [u32; 6] = [7, 3, 2, 32, 4, 2];
    let mut dimensions = DeviceBuffer::<u32>::zeroed(stream, EXPECTED.len())?;
    unsafe {
        module.report_launch_dimensions(
            stream,
            LaunchConfig {
                grid_dim: (EXPECTED[0], EXPECTED[1], EXPECTED[2]),
                block_dim: (EXPECTED[3], EXPECTED[4], EXPECTED[5]),
                shared_mem_bytes: 0,
            },
            &mut dimensions,
        )?;
    }
    let actual = dimensions.to_host_vec(stream)?;
    if actual != EXPECTED {
        return Err(format!("launch dimensions: expected {EXPECTED:?}, got {actual:?}").into());
    }
    Ok(())
}

fn check_scalar(values: &[f32]) -> Result<(), Box<dyn Error>> {
    for (i, &actual) in values.iter().enumerate() {
        let expected = input_a(i) + input_b(i);
        if actual != expected {
            return Err(
                format!("scalar mismatch at {i}: expected {expected}, got {actual}").into(),
            );
        }
    }
    Ok(())
}

fn tuning_configs(aps: u32) -> Vec<GridConfig> {
    let mut configs = Vec::new();
    for blocks_per_ap in [4, 8, 16, 32] {
        for threads in [128, 256] {
            configs.push(GridConfig {
                blocks: aps * blocks_per_ap,
                threads,
            });
        }
    }
    configs
}

fn u32_tuning_configs(aps: u32) -> Vec<GridConfig> {
    [(4, 512), (32, 64), (16, 128), (8, 256)]
        .into_iter()
        .map(|(blocks_per_ap, threads)| GridConfig {
            blocks: aps * blocks_per_ap,
            threads,
        })
        .collect()
}

fn benchmark_sizes(max_elements: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    for n in [1 << 20, 1 << 22, 1 << 24, 1 << 26, max_elements] {
        if n <= max_elements && !sizes.contains(&n) {
            sizes.push(n);
        }
    }
    sizes.sort_unstable();
    sizes
}

fn print_result(name: &str, n: usize, config: Option<GridConfig>, timing: &Timing) {
    let gbps = bandwidth_gbps(n, timing.median_ms);
    let grid = config.map_or_else(
        || format!("{}/256", n.div_ceil(256)),
        |cfg| format!("{}/{}", cfg.blocks, cfg.threads),
    );
    println!(
        "{name:<20} {n:>11} {:>10.4} {:>11.1} {:>8.1}% {:>8.1}% {:>13}  [{} iters, {:.4}-{:.4}]",
        timing.median_ms,
        gbps,
        100.0 * gbps / C500_NOMINAL_GBPS,
        100.0 * gbps / MCPYTORCH_REFERENCE_GBPS,
        grid,
        timing.iterations,
        timing.min_ms,
        timing.max_ms,
    );
}

fn bytes(elements: usize) -> usize {
    elements * 3 * core::mem::size_of::<f32>()
}

fn bandwidth_gbps(elements: usize, ms: f64) -> f64 {
    bytes(elements) as f64 / ms / 1.0e6
}

fn input_a(i: usize) -> f32 {
    (i % 251) as f32 * 0.5
}

fn input_b(i: usize) -> f32 {
    (i % 127) as f32 * 0.25
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
