# Oxide C500 VecAdd Benchmark

This example measures FP32 vector-add kernel bandwidth on MetaX C500. It keeps
the official Oxide one-thread-per-element algorithm as `vecadd_direct` and
compares it with fixed-grid grid-stride and explicit 128-bit variants.

The timed region contains kernel launches only. Allocations and host/device
copies are excluded. Effective bandwidth is:

```text
GB/s = (2 input reads + 1 output write) * N * sizeof(f32) / elapsed time
     = 12 * N / elapsed time
```

## Build and run

Set up the MACA cu-bridge environment:

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUDA_PATH=/opt/maca/tools/cu-bridge
export CUDA_TOOLKIT_PATH=/opt/maca/tools/cu-bridge
export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=/opt/maca/tools/cu-bridge/lib:/opt/maca/lib:${LD_LIBRARY_PATH:-}
export BINDGEN_EXTRA_CLANG_ARGS='-I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/opt/maca/include/mcr -I/opt/maca/include'
```

Build or run through the Oxide command:

```bash
cargo oxide build vecadd_bench --target maca
VECADD_DEVICE=1 cargo oxide run vecadd_bench --target maca
```

The default run allocates three 256 MiB buffers (`N = 2^26`), checks all five
kernels at an odd size of 1,000,003 elements, tunes launch configurations, and
then measures `N = 2^20, 2^22, 2^24, 2^26`. GPU events report the median of five
samples after ten warmup launches. Each sample transfers at least 32 GiB in
aggregate to reduce timer noise.

Useful controls:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `VECADD_DEVICE` | `0` | MACA device ordinal |
| `VECADD_MAX_ELEMENTS` | `67108864` | Largest vector and tuning size |
| `VECADD_WARMUP` | `10` | Warmup launches per candidate |
| `VECADD_ITERS` | `100` | Minimum timed launches per sample |
| `VECADD_SAMPLES` | `5` | Samples used for the median |

## C500 result

Measured on a 104-AP MetaX C500 with `N = 67,108,864` on 2026-07-17:

| Kernel | Launch | Median ms | Effective GB/s |
| --- | --- | ---: | ---: |
| `vecadd_direct` | `262144 x 256` | 0.8447 | 953.3 |
| `vecadd_scalar_grid` | `416 x 512` | 0.5537 | **1454.4** |
| `vecadd_scalar_grid_u4` | `416 x 512` | 0.5548 | 1451.6 |
| `vecadd_packed128_grid` | `416 x 128` | 0.5593 | 1439.8 |
| `vecadd_packed128_grid_u4` | `416 x 128` | 0.5615 | 1434.2 |

The fixed-grid scalar kernel is 1.53x the effective bandwidth of the official
algorithm. It reaches 79.4% of the C500 nominal 1832 GB/s and 97.6% of a local
mcPyTorch `torch.add(out=...)` reference measured at 1489.8 GB/s. The seven
HBM-size samples span 0.5536--0.5541 ms.

The final MACA device bitcode for `vecadd_packed128_grid` contains two aligned
`load i128` operations and one aligned `store i128`. It does not outperform the
scalar fixed-grid path on this chip. The material optimizations are reducing
block scheduling overhead and keeping the grid-stride induction variable in
C500's native 32-bit launch-index width. The kernel retains Oxide slice
arguments and is launched directly through the generated module API; raw
pointers did not improve performance. Results can vary with clocks and other
GPU work.
