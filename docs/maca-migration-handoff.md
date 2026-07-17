# cuda-oxide → MetaX GPU 迁移交接手册

> 创建时间：2026-07-17
> 目标：将 cuda-oxide 从 NVIDIA CUDA 迁移到 MetaX GPU (MXMACA) 平台
> 策略：两步走 — 1. 高级语言封装；2. LLVM builtin function

---

## 1. 项目概述

### 1.1 cuda-oxide 是什么

cuda-oxide 是一个 Rust GPU 编译器，将 `#[kernel]` 标注的 Rust 函数编译为 CUDA PTX 并在 NVIDIA GPU 上运行。编译管线：Rust 源码 → Rust MIR → `dialect-mir` (Pliron IR) → LLVM IR → PTX。

### 1.2 迁移目标

将 cuda-oxide 迁移到 MetaX GPU (MXMACA) 平台，使 Rust GPU kernel 能在 MetaX C500 等 GPU 上运行。

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
| 实测带宽 | 1385 GB/s |

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
- **变更**: 新增 `TargetBackend` 枚举，`CUDA_OXIDE_BACKEND` 环境变量，后端感知导出/降低/管线
- **Artifact 类型**: `MacaLlvmIr`（MXMACA LLVM IR，由 mxcc 消费）

#### mir-importer 适配
- **文件**: `crates/mir-importer/src/pipeline.rs`
- **变更**: 处理 `MacaLlvmIr` artifact 类型

### 3.2 设备内联函数映射（✅ 完成）

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
| `@llvm.mxc.mbcnt.lo(i32, i32)` | `i32` | lane ID 低 32 位 | `@llvm.nvvm.read.ptx.sreg.laneid` |
| `@llvm.mxc.mbcnt.hi(i32, i32)` | `i32` | lane ID 高 32 位（组合为 64-bit wave） | 无直接等价 |
| `@llvm.mxc.bsm.bpermute(i32, i32)` | `i32` | 通过 BSM 的 warp shuffle | `@llvm.nvvm.shfl.sync.idx.i32` |
| `@llvm.mxc.fcmp.i64.f32(float, float, i32)` | `i64` | 浮点比较返回 64-bit mask（用于 ballot） | `@llvm.nvvm.vote.ballot` |

#### 已实现的映射

| 映射 | 文件 | 状态 |
|---|---|---|
| threadIdx → `llvm.mxc.thread.id.x/y/z` | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| blockIdx → `llvm.mxc.block.id.x/y/z` | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| blockDim → dispatch.ptr+4 | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| gridDim → dispatch.ptr+12/16/20 | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| barrier → `llvm_mxc_barrier` | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| fence → LLVM fence 指令 | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| lane_id → mbcnt.lo+mbcnt.hi | `mir-lower/src/convert/intrinsics/basic.rs` | ✅ |
| shuffle → bsm.bpermute | `mir-lower/src/convert/intrinsics/warp.rs` | ✅ |
| ballot → fcmp.i64.f32 | `mir-lower/src/convert/intrinsics/warp.rs` | ✅ |
| MMA → no-op（待实现） | `mir-lower/src/convert/intrinsics/wmma.rs` | ⏳ |

#### Wave=64 适配

| 变更 | 文件 | 状态 |
|---|---|---|
| `WAVE_SIZE` 常量 (32) | `cuda-device/src/lib.rs` | ✅ |
| `warp_id()` 使用 WAVE_SIZE | `cuda-device/src/warp.rs` | ✅ |
| `warp_in_block_linear()` 使用 WAVE_SIZE | `cuda-device/src/cooperative_groups.rs` | ✅ |
| Inline PTX → no-op | `mir-lower/src/convert/intrinsics/common.rs` | ✅ |

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
- **文件**: `crates/cuda-bindings/src/lib.rs`
- **变更**: 添加 cu-bridge 类型别名（`CUresult` → `mcDrvError_enum` 等）
- **状态**: 类型别名完成，函数名别名（`wcu*` 前缀）待完成

### 3.4 构建工具（✅ 完成）

#### cargo-oxide --target maca
- **文件**: `crates/cargo-oxide/src/main.rs`, `commands.rs`
- **变更**: 新增 `--target` 选项（cuda/maca），设置 `CUDA_OXIDE_BACKEND=maca`

### 3.5 直接 Kernel Launch（✅ 完成）

#### C Launcher 方式
- **文件**: `examples/maca-vecadd/`
- **方法**: 编译 C launcher 为 .so，通过 libloading 加载
- **优势**: mxcc 自动处理设备上下文初始化
- **验证**: vecadd: SUCCESS: All 1024 elements correct!

---

## 4. 待完成工作

### 4.1 高优先级

| 任务 | 说明 | 阻塞因素 |
|---|---|---|
| cuda-core 函数名别名 | cu-bridge 使用 `wcu*` 前缀 | 需要添加 ~50 个函数别名 |
| 原始 vecadd 示例 | 通过完整 cuda-oxide 管线运行 | cuda-core 兼容性 |
| MMA builtin 映射 | `__builtin_mxc_mma_16x16x16f16/bf16/i8` | 需要确认 LLVM 内联函数名 |

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
| `crates/mir-importer/src/pipeline.rs` | MacaLlvmIr 处理 |

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
| `crates/cargo-oxide/src/commands.rs` | CUDA_OXIDE_BACKEND 环境变量 |

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
```

### 6.2 测试命令

```bash
# 运行核心 crate 测试
cargo test -p llvm-export -p dialect-mir -p dialect-nvvm -p mir-lower -p mir-transforms -p nvvm-transforms -p reserved-oxide-symbols -p oxide-artifacts

# 运行 MXMACA 绑定测试
cargo test -p maca-bindings -p maca-core

# 运行 cuda-oxide-codegen 测试（跳过需要 ptxas 的测试）
cargo test -p cuda-oxide-codegen -- --skip spine_add_kernel_emits_entry_and_validates
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
**解决方案**: 需要添加函数名别名
**状态**: ⏳ 待解决

---

## 8. 提交历史

| 提交 | 说明 |
|---|---|
| `feat: add MXMACA (MetaX GPU) backend infrastructure` | MacaExportConfig, BackendTarget, 内联函数映射 |
| `feat: wave size abstraction and MXMACA inline PTX handling` | WAVE_SIZE, inline PTX no-op |
| `feat: add --target maca option to cargo oxide build/run` | cargo-oxide --target maca |
| `fix: handle MacaLlvmIr artifact kind in mir-importer pipeline` | MacaLlvmIr 处理 |
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

---

## 9. 下一步建议

1. **cuda-core 函数名别名** — 添加 `wcu*` 函数别名，使 cuda-core 能与 cu-bridge 兼容
2. **原始 vecadd 示例** — 通过完整 cuda-oxide 管线运行原始 vecadd
3. **MMA builtin 映射** — 实现 `__builtin_mxc_mma_16x16x16f16/bf16/i8`
4. **更多示例迁移** — reduction, GEMM, atomics 等
5. **性能优化** — 对齐 MXMACA 最佳实践

---

## 10. 联系方式

- 项目仓库：https://github.com/NVlabs/cuda-oxide.git
- MetaX GPU 文档：`/opt/maca/docs/`
- MXMACA 编程指南：`references/markdown/chapter-1-cpp-extensions.md`
