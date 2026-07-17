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
  现在会在编译期明确报错，lowering 后置校验也禁止残留 inline PTX。
- **仍未完成**：`WAVE_SIZE` 仍为 32，u32 warp mask 尚未迁移到 Wave64；现有 shuffle
  丢失 mode，ballot 类型/mask 语义不完整，16x16x16 原生 MMA 尚未实现。

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
| shuffle → bsm.bpermute 原型 | `mir-lower/src/convert/intrinsics/warp.rs` | ⚠️ mode/mask 语义待替换 |
| ballot → fcmp.i64.f32 原型 | `mir-lower/src/convert/intrinsics/warp.rs` | ⚠️ i64/mask 契约待闭环 |
| CUDA WMMA | `mir-lower/src/convert/intrinsics/wmma.rs` | ✅ MACA 编译期明确拒绝 |

#### Wave=64 适配

| 变更 | 文件 | 状态 |
|---|---|---|
| `WAVE_SIZE` 常量 | `cuda-device/src/lib.rs` | ⏳ 仍为 32 |
| `warp_id()` 使用 WAVE_SIZE | `cuda-device/src/warp.rs` | ✅ 仅完成抽象 |
| `warp_in_block_linear()` 使用 WAVE_SIZE | `cuda-device/src/cooperative_groups.rs` | ✅ 仅完成抽象 |
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

### 4.1 高优先级

| 任务 | 说明 | 阻塞因素 |
|---|---|---|
| Wave64 API 与 mask | `WAVE_SIZE=64`，warp mask 从 u32 迁移并处理 API 兼容 | API breaking |
| MMA builtin 映射 | `__builtin_mxc_mma_16x16x16f16/bf16/i8` | 需要确认 builtin/LLVM lowering 与布局 |

已完成的原高优先级项目：cuda-core `wcu*` 兼容层、原始 vecadd build/run、MACA `.devbin`
产物闭环、Inline PTX/MMA fail-closed。

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

---

## 9. 下一步建议

1. **Wave64 语义闭环** — 更新 `WAVE_SIZE`、mask 类型和 warp/cooperative-groups 测试
2. **原生 vote/shuffle lowering** — 修复 i64 mask/result、mode 和上下边界语义
3. **MMA builtin 映射** — 实现并验证 `__builtin_mxc_mma_16x16x16f16/bf16/i8`
4. **扩展示例矩阵** — reduction、atomics、shuffle/ballot，再进入 GEMM
5. **性能基线** — 在正确性示例稳定后与 MXMACA C++/库实现同口径比较

---

## 10. 联系方式

- 项目仓库：https://github.com/NVlabs/cuda-oxide.git
- MetaX GPU 文档：`/opt/maca/docs/`
