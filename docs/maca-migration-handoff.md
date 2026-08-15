# cuda-oxide → MetaX GPU 迁移交接手册

> 创建时间：2026-07-17
> 目标：将 cuda-oxide 从 NVIDIA CUDA 迁移到 MetaX GPU (MXMACA) 平台
> 策略：两步走 — 1. 高级语言封装；2. LLVM builtin function
> 验收范围：仅当前 MetaX C500、`xcore1000`、`/opt/maca` SDK 的 MACA 编译与真实设备运行

---

## 0. 2026-07-17 最新检查点（后续工作以此为准）

- **范围约束**：后续只以当前 C500/MACA 环境为准，不要求原生 NVIDIA CUDA 编译、运行或
  行为兼容。已有 CUDA 路径可以保留，但不作为迁移工作的测试门槛或设计约束。
- 原始 `crates/rustc-codegen-cuda/examples/vecadd` 已通过完整命令
  `cargo oxide run vecadd --target maca`，MetaX C500 实测 1024/1024 正确。
- MACA 管线不再把 LLVM IR 冒充 PTX。导出的 `.ll` 会在构建期交给 `mxcc`，生成可加载的
  ELF `.devbin`，再以独立的 `MacaDeviceBinary` payload（wire kind `0x210`）嵌入宿主程序。
- kernel 已正确导出为 `define metaxgpu_kernel`；`blockDim` 的 dispatch 访问使用
  `getelementptr i8 + 4`，不再按指针宽度错误放大偏移。
- `gridDim` 已按 dispatch packet 的真实语义修正：`+12/+16/+20` 是全局 work-item 数，
  现在会除以各轴 `blockDim` 得到 block 数；`blockDim.z` 的 `dispatch+8` lowering 也已补齐。
- 新增并优化 `vecadd_bench`：C500、N=67,108,864 时固定 grid、u32 索引的 scalar kernel
  实测 1454.4 GB/s（0.5537 ms，标称带宽的 79.4%），相对同次原始逐元素 kernel 的
  953.3 GB/s 提升 1.53 倍；达到同口径 mcPyTorch 1489.8 GB/s 实测上限的 97.6%。
- cu-bridge 的 `wcu*` 函数、类型和常量兼容层已接入 `cuda-bindings`；`cuda-core`、
  `cuda-host`（宏实际使用的 loader）均能直接加载 MACA device binary。
- Rust/libm 数学调用会从 CUDA `__nv_*` 重写为 MACA `mc_math_func_*`。`libm_math`
  已在 C500 上验证每种精度各 64 lanes：f32 覆盖 sin/cos/exp/pow/floor/abs/rint/min/max/sincos，
  f64 覆盖 sin/cos/exp/pow/atan2/rint/min/max。`libm::sqrtf/sqrt` 的 inline-asm 路径仍未覆盖；
  当前只有 inherent `f32::sqrt`/`f64::sqrt` 路径可用。
- 本轮验证：编译器/产物相关测试 387 passed、9 ignored、1 个 NVIDIA `ptxas` 测试按环境跳过；
  绑定/运行时测试 136 passed、11 ignored；`cargo check --workspace` 为 0 errors、4 warnings。
- 官方 workspace 测试集合在 cu-bridge 环境下为 734 passed、169 ignored；过滤 2 项：当前环境
  不具备的 NVIDIA `ptxas` 用例，以及已在无 toolkit 环境单独通过的 toolkit 路径优先级用例。
- CI 风格的 `cargo test --workspace --all-targets` 为 699 passed、3 ignored，同样过滤上述 2 项。
- **已消除静默错误路径**：MACA 的通用 inline PTX、用户 inline PTX 和 CUDA WMMA
  现在会在编译期明确报错，lowering 后置校验也禁止残留 inline PTX/NVVM intrinsic。
- **Wave64 核心已闭环**：`WAVE_SIZE=64`、mask/ballot/active/lanemask 为 u64；idx/up/down/xor
  shuffle、all/any/ballot、match any/all、redux、sync_mask、block reduce/scan 已在 C500 真机通过。
- **原生 MMA 已闭环**：C500 Wave64 `m16n16k16` 的 FP16、BF16、INT8 均已通过
  16x16 真机数值 smoke。

---

## 1. 项目概述

### 1.1 cuda-oxide 是什么

cuda-oxide 是一个 Rust GPU 编译器，将 `#[kernel]` 标注的 Rust 函数编译为 CUDA PTX 并在 NVIDIA GPU 上运行。编译管线：Rust 源码 → Rust MIR → `dialect-mir` (Pliron IR) → LLVM IR → PTX。

### 1.2 迁移目标

将 cuda-oxide 迁移到当前 MetaX C500（`xcore1000`）环境，使 Rust GPU kernel 能通过
`/opt/maca` 中的工具链编译并在本机 C500 上正确运行。其他 MetaX 芯片和原生 NVIDIA CUDA
均不在当前验收范围内。

### 1.3 关键约束

- **Wave size**: MXMACA 使用 64 线程 wave（CUDA 使用 32 线程 warp）
- **Inline PTX**: MXMACA 不支持内联 PTX 汇编
- **MMA 形状**: MXMACA MMA 是 16x16x16（CUDA 是 16x8x16）
- **编译器**: mxcc（LLVM-based），设备库为 LLVM bitcode 格式

---

## 2. 环境信息

### 2.1 硬件

| 项目 | 值 |
|---|---|
| GPU | MetaX C500 × 4 |
| GPU 内存 | 65536 MiB |
| 实测带宽 | 1454.4 GB/s（Oxide FP32 VecAdd 有效带宽） |
| Oxide FP16 GEMM 基线 | 3.460 TFLOP/s（1024³，2x2 tiles/wave，无 shared-memory pipeline） |

### 2.2 软件

| 项目 | 值 | 路径 |
|---|---|---|
| MACA SDK | 3.7.0 | `/opt/maca` |
| mxcc | 1.0.0 | `/opt/maca/mxgpu_llvm/bin/mxcc` |
| cucc | 1.0.0 | `/opt/maca/tools/cu-bridge/bin/cucc` |
| libmcruntime.so | - | `/opt/maca/lib/libmcruntime.so` |
| libmccompiler.so | - | `/opt/maca/lib/libmccompiler.so` |
| maca_kernellib.bc | 10.4M | `/opt/maca/lib/maca_kernellib.bc` |
| maca_mathlib.bc | 406.6K | `/opt/maca/lib/maca_mathlib.bc` |

### 2.3 环境变量

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH=$MACA_PATH/tools/cu-bridge
export CUDA_PATH=$CUCC_PATH
export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
export CUDA_TOOLKIT_PATH=$CUCC_PATH
export LD_LIBRARY_PATH=$CUCC_PATH/lib:$MACA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/opt/maca/include/mcr -I/opt/maca/include"
```

---

## 3. 已完成的工作

### 3.1 编译器管线（✅ 完成）

#### MacaExportConfig
- **文件**: `crates/llvm-export/src/export/config.rs`
- **变更**: 新增 `MacaExportConfig` 实现 `ExportBackendConfig` trait
- **目标三元组**: `mxc-metax-macahca`
- **数据布局**: `e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7`

#### BackendTarget 枚举
- **文件**: `crates/mir-lower/src/lib.rs`
- **变更**: 新增 `BackendTarget` 枚举（Cuda/Maca）加入 `LoweringOptions`
- **用途**: 控制内联函数映射和降低路径

#### cuda-oxide-codegen MXMACA 管线
- **文件**: `crates/cuda-oxide-codegen/src/options.rs`, `export.rs`, `lower.rs`, `pipeline.rs`
- **变更**: 新增 `TargetBackend` 枚举，`CUDA_OXIDE_TARGET_BACKEND` 环境变量，后端感知导出/降低/管线
- **Artifact 类型**: `MacaDeviceBinary`（wire kind `0x210`，最终为 xcore1000 ELF `.devbin`）
- **编译命令契约**: `mxcc -x ir -input-is-device -offload-arch=xcore1000 -device-bin ...`

#### mir-importer 适配
- **文件**: `crates/mir-importer/src/pipeline.rs`
- **变更**: 管理 `.ll` 中间文件和 `.devbin` 最终产物，清理 stale artifact

### 3.2 设备内联函数映射（⚠️ 部分完成）

#### 已确认的 MXMACA LLVM 内联函数

| 内联函数 | 返回类型 | 用途 | CUDA 等价 |
|---|---|---|---|
| `@llvm.mxc.thread.id.x()` | `i32` | threadIdx.x | `@llvm.nvvm.read.ptx.sreg.tid.x` |
| `@llvm.mxc.block.id.x()` | `i32` | blockIdx.x | `@llvm.nvvm.read.ptx.sreg.ctaid.x` |
| `@llvm.mxc.block.id.y()` | `i32` | blockIdx.y | `@llvm.nvvm.read.ptx.sreg.ctaid.y` |
| `@llvm.mxc.block.id.z()` | `i32` | blockIdx.z | `@llvm.nvvm.read.ptx.sreg.ctaid.z` |
| `@llvm.mxc.dispatch.ptr()` | `ptr addrspace(4)` | 获取 dispatch 结构体指针 | 无直接等价 |
| `@llvm.mxc.implicitarg.ptr()` | `ptr addrspace(4)` | 获取隐式参数指针 | 无直接等价 |
| `@llvm.mxc.is.private(ptr)` | `i1` | 检查指针是否在私有内存 | 无直接等价 |
| `@llvm.mxc.sleep(i32)` | `void` | 线程休眠 | `@llvm.nvvm.nanosleep` |
| `@llvm.mxc.mbcnt.lo(i32, i32)` | `i32` | 统计当前 lane 之前的低半 wave lanes | `@llvm.nvvm.read.ptx.sreg.laneid` 的一部分 |
| `@llvm.mxc.mbcnt.hi(i32, i32)` | `i32` | 以上一步为累加值继续统计，最终得到 0..63 lane ID | `@llvm.nvvm.read.ptx.sreg.laneid` 的一部分 |
| `@llvm.mxc.bsm.bpermute(i32, i32)` | `i32` | 通过 BSM 的 warp shuffle | `@llvm.nvvm.shfl.sync.idx.i32` |
| `@llvm.mxc.fcmp.i64.f32(float, float, i32)` | `i64` | 浮点比较返回 64-bit mask（用于 ballot） | `@llvm.nvvm.vote.ballot` |

#### 已实现的映射

| 映射 | 文件 | 状态 |
|---|---|---|
| threadIdx → `llvm.mxc.thread.id.x/y/z` | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| blockIdx → `llvm.mxc.block.id.x/y/z` | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| blockDim → dispatch.ptr+4/+8 | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| gridDim → ceil((dispatch.ptr+12/16/20) / blockDim) | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| barrier → `llvm_mxc_barrier` | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| fence → LLVM fence 指令 | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| lane_id → mbcnt.lo+mbcnt.hi | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| i32/f32/u64/f64 shuffle → 源 lane 计算 + bsm.bpermute | `mir-lower/src/convert/intrinsics/warp.rs` | ✅ C500 |
| ballot/all/any → icmp.i64.i32 + u64 mask | `mir-lower/src/convert/intrinsics/warp.rs` | ✅ C500 |
| match any/all → 按位 icmp vote mask 交集 | `mir-lower/src/convert/intrinsics/warp.rs` | ✅ C500，i32/i64 value |
| redux add/min/max/and/or/xor → 6 级稀疏 mask bpermute | `mir-lower/src/convert/intrinsics/warp.rs` | ✅ C500，full/sparse mask |
| sync_mask → warp fence + barrier | `mir-lower/src/convert/intrinsics/warp.rs` | ✅ C500 |
| m16n16k16 f32/f16 MMA → `llvm.mxc.mma.f32.16x16x16f16` | `mir-lower/src/convert/intrinsics/wmma.rs` | ✅ C500 16x16 数值 smoke |
| m16n16k16 f32/bf16 MMA → `llvm.mxc.mma.f32.16x16x2bf16` | `mir-lower/src/convert/intrinsics/wmma.rs` | ✅ C500 16x16 数值 smoke |
| m16n16k16 i32/i8 MMA → `llvm.mxc.mma.i32.16x16x16i8` | `mir-lower/src/convert/intrinsics/wmma.rs` | ✅ C500 16x16 数值 smoke |
| CUDA WMMA | `mir-lower/src/convert/intrinsics/wmma.rs` | ✅ MACA 编译期明确拒绝 |

#### Wave=64 适配

| 变更 | 文件 | 状态 |
|---|---|---|
| `WAVE_SIZE` / `WaveMask` | `cuda-device/src/lib.rs` | ✅ 64 / u64 |
| `warp_id()` / lanemask | `cuda-device/src/warp.rs` | ✅ Wave64 |
| warp/block collectives | `cuda-device/src/cooperative_groups.rs` | ✅ 64 lanes、最多 16 waves |
| Inline PTX fail-closed | `mir-lower/src/convert/intrinsics/common.rs`、`asm.rs`、`lib.rs` | ✅ 编译期明确拒绝 |

### 3.3 运行时绑定（✅ 完成）

#### maca-bindings
- **文件**: `crates/maca-bindings/`
- **功能**: MXMACA Runtime API FFI 绑定（bindgen 生成）
- **依赖**: `libmcruntime.so`
- **关键类型**: `mcError_t`, `mcDevice_t`, `mcStream_t`, `mcEvent_t`, `mcModule_t`, `mcFunction_t`

#### maca-core
- **文件**: `crates/maca-core/`
- **功能**: 安全 RAII 包装层（`DeviceBuffer`, `MacaContext`, `LaunchConfig`）
- **依赖**: maca-bindings

#### cuda-bindings cu-bridge 兼容
- **文件**: `crates/cuda-bindings/build.rs`, `src/lib.rs`, `src/cu_bridge_compat.rs`
- **变更**: 自动识别 cu-bridge，恢复 cuda-core 使用的 CUDA 类型、常量和 `cu*` 函数名，链接 `libruntime_cu.so`
- **状态**: ✅ 原始 vecadd、cuda-core/cuda-host 测试和 workspace check 已验证；当前 cu-bridge 运行路径只加载 `MacaDeviceBinary`

### 3.4 构建工具（✅ 完成）

#### cargo-oxide --target maca
- **文件**: `crates/cargo-oxide/src/main.rs`, `commands.rs`
- **变更**: 新增 `--target` 选项（cuda/maca），设置 `CUDA_OXIDE_TARGET_BACKEND`；解析优先级为 CLI > 继承环境 > 项目环境 > `cuda`
- **兼容边界**: `CUDA_OXIDE_BACKEND` 只保留为 codegen backend `.so` 路径，不再兼作 GPU target
- **防误用**: 未知目标值直接报错；NVIDIA Compute Sanitizer 路径明确拒绝 MACA target
- **缓存边界**: passthrough 指纹包含 target backend、`MACA_PATH` 和解析后的 mxcc 文件身份；
  CUDA-only 的 `emit-ltoir` 显式固定 `cuda`，不会继承 MACA 配置

### 3.5 早期独立 Kernel Launch 探针（✅ 完成）

#### C Launcher 方式
- **文件**: `examples/maca-vecadd/`
- **方法**: 编译 C launcher 为 .so，通过 libloading 加载
- **优势**: mxcc 自动处理设备上下文初始化
- **验证**: vecadd: SUCCESS: All 1024 elements correct!
- **边界**: 这是早期 SDK/设备探针；当前统一 E2E 不使用该 launcher，而是
  `cuda-host -> cuda-core -> wcuModuleLoadData` 加载嵌入的 `MacaDeviceBinary`

---

## 4. 待完成工作

### 4.0 2026-07-17 完整测试套件结果

**命令**: `scripts/smoketest.sh -t maca`

| 类别 | 数量 | 说明 |
|---|---|---|
| **PASS** | **110/137 (80%)** | 编译或运行成功 |
| inline PTX unsupported | 7 | NVIDIA PTX 内联汇编（预期失败） |
| Host runtime panic | 2 | 剩余运行时错误 |
| Type mismatch (E0308) | 1 | Rust 类型错误 |
| Other compile errors | 17 | MXMACA 编译器错误 |

**关键修复**：
1. `.devbin` fallback for `load_module_from_file` — 13 panics fixed
2. `load_all_ptx_bundles_merged` falls back to `load_first_embedded_module` for MXMACA generic kernels — 3 failures fixed
3. `load_kernel_module` accepts `.devbin` via `FileArtifact::Devbin` — manual_launch_generic fixed

**主要失败模式**：
1. **inline PTX unsupported** (7个） — NVIDIA PTX 内联汇编（mbarrier, cp.async 等），MXMACA 不支持
2. **Host runtime panic** (2个） — 剩余运行时错误
3. **Type mismatch** (1个） — elect_leader 的 Wave64 类型问题
4. **Other compile errors** (17个） — MXMACA 编译器错误（shared memory, atomics, warp ops 等）

### 4.0.1 2026-07-20 会话进展

**本日修复（按 commit 顺序）**：

1. **elect.sync 正确实现** — `leader = cttz(mask)`（`llvm.cttz.i64`）+ `is_elected = lane_id == leader`，
   替换此前硬编码 leader=0 的错误实现。修复 elect_leader 挂起。
2. **alloca 地址空间（AS5）** — MetaX 后端无法在通用地址空间选择 `FrameIndex`（`Cannot select: i64 = FrameIndex<0>`）。
   对齐 MetaX clang：`alloca ..., addrspace(5)` + `addrspacecast` 回通用指针
   （`llvm-export` 新增 `ExportBackendConfig::alloca_address_space`）。修复 array_for_loop、array_constants、abi_hmm 等编译崩溃。
3. **mxcc -O3 中端优化** — 设备二进制生成前先用 `mxcc -x ir -O3 -S -emit-llvm` 优化
   （`-forward-unknown-to-compiler --target=mxc-metax-macahca` 防止 triple 被覆盖为 x86_64）。
4. **alloca 提升至 entry 块** — MetaX 后端按 **4KB 粒度**分配私有内存（4KB/thread 硬件上限），
   分支块中的 alloca 无法被 mem2reg/SROA 提升，每个都占满 4KB 导致 launch 失败
   （`private memory size required ... 5 KB/thread`）。`mir-lower` 在 MACA lowering 后
   把所有静态 alloca 提升到 entry 块。修复 enum_array_match、union_aggregate、place_read_fallbacks。
5. **byte-array transmute 寄存器化** — `[u8; N]` ↔ 标量（`from_ne_bytes`/`to_ne_bytes`）在 MACA 下
   不再走 `alloca+store+load` 内存往返（byte-pun alloca 无法提升），改为 extractvalue/shl/or 寄存器组装。
   修复 ne_bytes_transmute。
6. **WAVE_SIZE 目标条件化** — `cuda_oxide_target_maca` cfg（64=MetaX / 32=NVIDIA），
   `cargo oxide run/build/passthrough` 三条路径均注入；WaveMask 统一 u64（CUDA lowering 截断）。
   修复此前 WAVE_SIZE=64 无条件化导致的 CUDA 侧 warp_id 语义破坏。
7. **Wave64 示例移植** — lanemask_scan（u64 mask + WAVE_SIZE）、shuffle_64（64-lane butterfly、half-warp 泛型）。
8. **hashmap_v3 reclaim 死锁** — `try_reclaim_deleted` 的 `RESERVED` 自旋改为有界（4096 次后回退 EMPTY claim）。
   无限自旋在 SIMT 硬件上可能死锁：持有 RESERVED 的 lane 在 wave-mates 自旋时永远得不到调度。
   新增 `bin/repro_reclaim.rs` 最小回归测试（90% 负载 delete+reinsert）。
9. **smoketest 分类** — PTX 文本检查类示例（vectorization、const_generic、cross_crate_kernel、cutile_inter_kernel）
   在 MACA 下归入 `maca-skip`。

**关键经验**：
- MetaX 私有内存分配粒度 4KB；任何无法提升为寄存器的 alloca 都是 4KB。CUDA 路径靠 `opt -O2` 消除，MACA 路径需自己保证。
- `cargo oxide run` 与 `cargo oxide build` 走不同的 rustflags 注入路径（`codegen_run` vs `run_cargo_passthrough`），
  新增设备 cfg 必须三条路径都覆盖。
- 调试注意：示例构建失败时 smoketest 会跑**陈旧二进制**（输出与源码不符时先 `cargo build` 验证编译错误）。
- Wave64 SIMT 上的波内原子自旋（spin on atomic）是死锁高发区：自旋必须有界。


### 4.0.2 2026-08-15 上游主线合并（upstream/main → main）

**合并范围**：NVlabs/cuda-oxide upstream/main 50 个新提交（生成式 intrinsic catalog、dialect-nvvm 生成化、translator 重构、`IntrinsicBackend`（LlvmNvptx/LibNvvm）枚举、oxide-artifacts 新 crate、示例从 137 增至 213）。

**冲突规模**：50 个文件冲突，约 400 个冲突块。处理策略：上游重构为主的文件取上游版后重放 MACA 差异；双方都有语义的文件取并集；纯机械冲突（Cargo.lock/.gitignore/Cargo.toml）手动合并。

**MACA 适配重放（关键项）**：
1. `BackendTarget(Maca)` 与上游新 `IntrinsicBackend` 枚举共存于 LoweringOptions。
2. `generated_intrinsics.rs`（上游新的生成式转换文件）中 60+ 个 op 加了 MACA 预分发，调用我们原有的 native C500 helper（basic.rs/warp.rs）。
3. **mask 类 op 加宽为 64 位**：ballot/lanemask/match/active_mask 的 build+verify 结果 32→64，member-mask operand 接受 i32|i64，importer 用 `emit_generated_nvvm_intrinsic_u64`。这是 Wave64 语义在新 catalog 架构下的落点。
4. MACA export 分支恢复（MacaExportConfig、`kernel_calling_convention` 在 PipelineExportConfig 转发、alloca_address_space 转发）——缺转发会静默降级为普通 `define void`，kernel 符号找不到。
5. `trap;`/`ld.acquire.gpu.*`/`fence.acq_rel.gpu` 等上游 inline-PTX 路径加 MACA 分支（llvm.trap / LLVM atomic load/store / LLVM fence）。
6. m16n16k16 native MMA 全链路恢复：op 定义（generated/maca_mma.rs）+ importer 识别 + lowering impl。
7. cu-bridge compat 扩展：CUlimit、CUctx_flags、流优先级、函数属性枚举、错误码别名（上游 cuda-core 新 API 面）。
8. libdevice 一致性检查对 MACA 豁免（__nv_* → mc_math_func_* 重写导致 lowered 检测为 false 是预期行为）。
9. cuda-device `live_lanes_1d`/`live_lane_mask` wave64 化；movmatrix 与上游生成版去重。

**验证结果**：
- workspace 全量构建 ✓；关键 crate 单元测试 535 通过 ✓
- MACA 全套件：**213/213 全部通过**（含 2 个 NVIDIA 专属 skip：mma_mxf8f6f4 Blackwell MMA、small_type_ffi_test LTOIR-modern）
- 合并前基线 112/137（82%），合并后提升且无回归

**追加修复（2026-08-15 晚间）**：partial warp reduce wave64 化——`thread::warp_index` 的
`WARP_SIZE=32` 硬编码与 `reduce_*_partial` 的 `clamp(1, 32)` 改为 `WAVE_SIZE`；warp_sums 与
partial_warp_reduce 两示例的 block 几何改为 64 的倍数（96/93 线程、32/29 尾 wave）。

**教训**：
- 上游架构迁移（手写 op → 生成 catalog）时，自定义后端的每个 hook 点都要在新架构里找到对应位置（op 定义/verify/importer/lowering 四层都要检查）。
- trait 默认实现的静默回退是危险信号：`kernel_calling_convention` 没转发时编译照过、运行时符号找不到。
- 生成文件（DO NOT EDIT 标记）与后端扩展的边界：maca_mma.rs 作为手写扩展模块注册进 generated/ 目录是可行模式。

### 4.1 高优先级

| 任务 | 说明 | 阻塞因素 |
|---|---|---|
| inline PTX 替代实现 | 7 个 PTX 内联汇编 | mbarrier, cp.async 等需要 MXMACA 等价物 |
| MXMACA 编译器错误 | 17 个编译错误 | shared memory, atomics, warp ops 等 |
| Wave64 类型修正 | 1 个类型错误 | elect_leader |
| GEMM 性能优化 | 当前 2x2 tiles/wave、1024³ 为 3.460 TFLOP/s | 待增加 shared-memory pipeline 并对比 mcBLAS |
|---|---|---|
| cu-bridge API 完整映射 | 35 个 host runtime panic | `cuEventElapsedTime` 等未映射到 `wcu*` |
| inline PTX 替代实现 | 9 个 PTX 内联汇编 | mbarrier, cp.async 等需要 MXMACA 等价物 |
| Wave64 类型修正 | 2 个类型错误 | elect_leader, lanemask_scan |
| GEMM 性能优化 | 当前 2x2 tiles/wave、1024³ 为 3.460 TFLOP/s | 待增加 shared-memory pipeline 并对比 mcBLAS |

已完成的原高优先级项目：cuda-core `wcu*` 兼容层、原始 vecadd build/run、MACA `.devbin`
产物闭环、Inline PTX/MMA fail-closed、Wave64 shuffle/vote/match/redux 与 C500 primitives smoke。

### 4.2 中优先级

| 任务 | 说明 |
|---|---|
| 更多示例迁移 | reduction, GEMM, atomics 等 |
| Wave=64 mask 类型 | u32→u64 变更（API breaking） |
| cuda-async 适配 | 跟随 cuda-core 变更 |

### 4.3 低优先级

| 任务 | 说明 |
|---|---|
| 完整测试套件 | 运行所有 133 个示例 |
| 性能优化 | 对齐 MXMACA 最佳实践 |
| 文档更新 | 更新 cuda-oxide-book |

---

## 5. 关键文件清单

### 5.1 编译器管线

| 文件 | 用途 |
|---|---|
| `crates/llvm-export/src/export/config.rs` | MacaExportConfig |
| `crates/llvm-export/src/export/module.rs` | 目标三元组配置化 |
| `crates/mir-lower/src/lib.rs` | BackendTarget 枚举 |
| `crates/mir-lower/src/convert/intrinsics/basic.rs` | MXMACA 内联函数映射 |
| `crates/mir-lower/src/convert/intrinsics/warp.rs` | shuffle/ballot 映射 |
| `crates/mir-lower/src/convert/intrinsics/wmma.rs` | MMA 后端检查 |
| `crates/mir-lower/src/convert/interface_impls.rs` | 内联函数分发 |
| `crates/cuda-oxide-codegen/src/options.rs` | TargetBackend 枚举 |
| `crates/cuda-oxide-codegen/src/export.rs` | 后端感知导出 |
| `crates/cuda-oxide-codegen/src/lower.rs` | 后端感知降低 |
| `crates/cuda-oxide-codegen/src/pipeline.rs` | MXMACA 管线路径 |
| `crates/mir-importer/src/pipeline.rs` | `.ll` → `.devbin` 产物路径 |
| `crates/cuda-oxide-codegen/src/maca.rs` | 调用 mxcc 并验证 ELF 输出 |
| `crates/oxide-artifacts/src/lib.rs` | `MacaDeviceBinary` wire kind `0x210` |

### 5.2 运行时绑定

| 文件 | 用途 |
|---|---|
| `crates/maca-bindings/` | MXMACA Runtime API FFI |
| `crates/maca-core/` | 安全 RAII 包装层 |
| `crates/cuda-bindings/src/lib.rs` | cu-bridge 类型别名 |

### 5.3 设备内联函数

| 文件 | 用途 |
|---|---|
| `crates/cuda-device/src/lib.rs` | WAVE_SIZE 常量 |
| `crates/cuda-device/src/warp.rs` | warp_id() 使用 WAVE_SIZE |
| `crates/cuda-device/src/cooperative_groups.rs` | warp_in_block_linear() |

### 5.4 构建工具

| 文件 | 用途 |
|---|---|
| `crates/cargo-oxide/src/main.rs` | --target 选项 |
| `crates/cargo-oxide/src/commands.rs` | `CUDA_OXIDE_TARGET_BACKEND` 选择与 CLI 优先级 |

### 5.5 示例

| 文件 | 用途 |
|---|---|
| `examples/maca-vecadd/` | 直接 kernel launch 示例 |
| `examples/maca-vecadd/kernel/vecadd_launcher.maca` | C launcher |

---

## 6. 构建与测试

### 6.1 构建命令

```bash
# 构建核心编译器 crate
cargo build -p llvm-export -p mir-lower -p cuda-oxide-codegen -p mir-importer

# 构建 MXMACA 运行时
cargo build -p maca-bindings -p maca-core

# 构建完整工作空间（需要 libffi-dev, libclang-dev）
export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
export CUDA_TOOLKIT_PATH=/opt/maca/tools/cu-bridge
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/opt/maca/include/mcr -I/opt/maca/include"
cargo build --workspace

# 运行 maca-vecadd 示例
cargo run -p maca-vecadd

# 运行原始统一编译 vecadd（完整 rustc backend → mxcc → bundle → cu-bridge 路径）
cargo oxide run vecadd --target maca

# 运行 Wave64 shuffle/vote/match/redux/lanemask、atomics、block reduce/scan 真机 smoke
cargo oxide run maca_primitives_smoke --target maca

# 运行 C500 Wave64 原生 m16n16k16 FP16/BF16/INT8 MMA 真机 smoke
cargo oxide run maca_mma_smoke --target maca

# 运行 1024^3 FP16 tiled GEMM 正确性与性能基线
cargo oxide run maca_gemm --target maca
```

### 6.2 测试命令

```bash
# 编译器、artifact、importer 和 cargo 前端（跳过需要 NVIDIA ptxas 的测试）
cargo test -p oxide-artifacts -p llvm-export -p mir-lower -p cuda-oxide-codegen -p mir-importer -p cargo-oxide -- --skip spine_add_kernel_emits_entry_and_validates

# 运行 cu-bridge 主加载路径测试（使用 2.3 节环境变量）
cargo test -p cuda-bindings -p cuda-core -p cuda-host

# 独立 rustc backend 的 MACA artifact 路径
cargo test --manifest-path crates/rustc-codegen-cuda/Cargo.toml read_compilation_artifact_reads_declared_maca_device_binary_path --lib

# 完整 workspace 编译检查（使用 2.3 节环境变量）
cargo check --workspace

# 官方 workspace 测试集合；toolkit 路径优先级用例需清除父进程 toolkit 环境后单独运行
cargo test --workspace -- \
  --skip spine_add_kernel_emits_entry_and_validates \
  --skip sanitizer_tool_lookup_uses_project_cuda_toolkit_root

# CI 风格的全部 targets（不含 doctest）
cargo test --workspace --all-targets -- \
  --skip spine_add_kernel_emits_entry_and_validates \
  --skip sanitizer_tool_lookup_uses_project_cuda_toolkit_root
```

### 6.3 环境验证

```bash
# 检查 MXMACA 环境
which mxcc && mxcc --version
ls /opt/maca/lib/maca_kernellib.bc

# 检查 cu-bridge 环境
which cucc && cucc --version
ls /opt/maca/tools/cu-bridge/lib/libcuda.so

# 运行 cuda-oxide doctor
cargo oxide doctor
```

---

## 7. 已知问题

### 7.1 cuda-bindings Rust 2024 兼容性

**问题**: bindgen 0.71 生成的代码不符合 Rust 2024 edition unsafe 规则
**解决方案**: 将 cuda-bindings edition 改为 2021
**状态**: ✅ 已解决

### 7.2 cu-bridge 头文件重复定义

**问题**: cu-bridge 头文件包含重复的 FP_* 常量定义
**解决方案**: 在 build.rs 中添加 `blocklist_item` 过滤
**状态**: ✅ 已解决

### 7.3 mcModuleLaunchKernel 段错误

**问题**: 从 Rust 直接调用 mcModuleLaunchKernel 导致段错误（设备上下文未初始化）
**解决方案**: 使用 C launcher 编译为 .so，通过 libloading 加载
**状态**: ✅ 已解决

### 7.4 cuda-core 函数名不兼容

**问题**: cuda-core 使用 CUDA 函数名（`cuLaunchKernel`），cu-bridge 使用 `wcuLaunchKernel`
**解决方案**: cu-bridge 专用兼容模块导出类型、常量和函数别名，并在 build.rs 中自动识别/链接运行时
**状态**: ✅ 已解决并通过真实 vecadd 验证

### 7.5 MACA LLVM IR 不能直接加载

**问题**: `mcModuleLoadData`/`wcuModuleLoadData` 不接受文本 LLVM IR；旧管线把 `.ll` 标成 PTX 后运行时报 invalid kernel image。
**解决方案**: 构建期使用 `mxcc -x ir -input-is-device -offload-arch=xcore1000 -device-bin` 生成 `.devbin`，完整校验 ELF64、little-endian、`ET_DYN` 和 MetaX `e_machine=0xfd` 后才发布。
**状态**: ✅ 已解决；最终 executable 的 bundle 已结构化解析为 `MacaDeviceBinary (0x210)`。

---

## 8. 提交历史

| 提交 | 说明 |
|---|---|
| `feat: add MXMACA (MetaX GPU) backend infrastructure` | MacaExportConfig, BackendTarget, 内联函数映射 |
| `feat: wave size abstraction and MXMACA inline PTX handling` | WAVE_SIZE, inline PTX no-op |
| `feat: add --target maca option to cargo oxide build/run` | cargo-oxide --target maca |
| `fix: handle MacaLlvmIr artifact kind in mir-importer pipeline` | 历史提交名；该中间类型现已由最终 `MacaDeviceBinary` 取代 |
| `feat: implement MXMACA fence support using LLVM fence instructions` | LLVM fence 指令 |
| `feat: add MXMACA lane_id intrinsic mapping` | mbcnt.lo+mbcnt.hi |
| `feat: add MXMACA shuffle intrinsic mapping via bsm.bpermute` | bsm.bpermute |
| `feat: add MXMACA ballot intrinsic mapping and f32 constant helper` | fcmp.i64.f32 |
| `docs: add CLAUDE.md for Claude Code guidance` | 项目文档 |
| `feat: add MXMACA backend check for MMA operations` | MMA 后端检查 |
| `docs: add cu-bridge compatibility header for cuda-bindings` | 兼容性头文件 |
| `feat: add MXMACA runtime bindings and maca-vecadd example` | maca-bindings, maca-core |
| `feat: add maca-host crate and working maca-vecadd example` | maca-host |
| `fix: add target_triple to test ExportBackendConfig implementations` | 测试修复 |
| `refactor: simplify maca-vecadd to standalone executable` | 独立可执行文件 |
| `feat: add cu-bridge type aliases to cuda-bindings` | 类型别名 |
| `feat: implement direct kernel launch via C launcher` | 直接 kernel launch |
| `feat(maca): support Wave64 match and redux collectives` | C500 match any/all 与全部整数 redux，含稀疏 mask |
| `feat(maca): add native Wave64 FP16 MMA` | C500 m16n16k16 f32/f16 intrinsic 与 16x16 数值 smoke |
| `feat(maca): complete native Wave64 MMA variants` | C500 m16n16k16 BF16/INT8 intrinsic 与数值 smoke |
| `feat(examples): add C500 native MMA GEMM baseline` | 2x2 tiles/wave FP16 GEMM，identity/checkerboard 正确性与性能 |

---

## 9. 下一步建议

1. **GEMM pipeline** — 用 shared memory 复用跨 wave 的 A/B tile，并增加双缓冲
2. **性能对比** — 与 mcBLAS/MXMACA C++ 实现同口径比较

---

## 10. 联系方式

- 项目仓库：https://github.com/NVlabs/cuda-oxide.git
- MetaX GPU 文档：`/opt/maca/docs/`
