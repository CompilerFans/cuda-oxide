/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Basic NVVM intrinsic conversion: thread IDs, block IDs, barrier.
//!
//! | Operation    | LLVM Intrinsic                    |
//! |--------------|-----------------------------------|
//! | `ReadTidX`   | `llvm_nvvm_read_ptx_sreg_tid_x`   |
//! | `ReadCtaidX` | `llvm_nvvm_read_ptx_sreg_ctaid_x` |
//! | `ReadNtidX`  | `llvm_nvvm_read_ptx_sreg_ntid_x`  |
//! | `Barrier0`   | `llvm_nvvm_barrier0`              |
//! | `ThreadfenceBlock` | inline PTX `membar.cta`      |
//! | `Threadfence` | inline PTX `membar.gl`           |
//! | `ThreadfenceSystem` | inline PTX `membar.sys`     |

use crate::BackendTarget;
use crate::convert::intrinsics::common::*;
use crate::context::lowering_options;
use llvm_export::attributes::LlvmAtomicOrdering;
use llvm_export::op_interfaces::BinArithOp;
use llvm_export::ops as llvm;
use llvm_export::ops::{AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Map a CUDA NVVM intrinsic name to its MXMACA equivalent.
///
/// Returns `None` if there is no MXMACA equivalent (caller should skip or error).
fn maca_intrinsic_for(cuda_name: &str) -> Option<&'static str> {
    match cuda_name {
        // Thread/Block IDs
        "llvm_nvvm_read_ptx_sreg_tid_x" => Some("llvm_mxc_thread_id_x"),
        "llvm_nvvm_read_ptx_sreg_tid_y" => Some("llvm_mxc_thread_id_y"),
        "llvm_nvvm_read_ptx_sreg_tid_z" => Some("llvm_mxc_thread_id_z"),
        "llvm_nvvm_read_ptx_sreg_ctaid_x" => Some("llvm_mxc_block_id_x"),
        "llvm_nvvm_read_ptx_sreg_ctaid_y" => Some("llvm_mxc_block_id_y"),
        "llvm_nvvm_read_ptx_sreg_ctaid_z" => Some("llvm_mxc_block_id_z"),
        // blockDim/gridDim — MXMACA uses dispatch.ptr, no direct intrinsic
        "llvm_nvvm_read_ptx_sreg_ntid_x" => None,
        "llvm_nvvm_read_ptx_sreg_ntid_y" => None,
        "llvm_nvvm_read_ptx_sreg_ntid_z" => None,
        "llvm_nvvm_read_ptx_sreg_nctaid_x" => None,
        "llvm_nvvm_read_ptx_sreg_nctaid_y" => None,
        "llvm_nvvm_read_ptx_sreg_nctaid_z" => None,
        // Warp
        "llvm_nvvm_read_ptx_sreg_laneid" => Some("llvm_mxc_lane_id"),
        "llvm_nvvm_read_ptx_sreg_warpsize" => Some("llvm_mxc_wave_size"),
        // Barrier
        "llvm_nvvm_barrier0" => Some("llvm_mxc_barrier"),
        _ => None,
    }
}

pub(crate) fn convert_sreg_read_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let opts = lowering_options(ctx);
    let resolved_name = match opts.backend {
        BackendTarget::Cuda => intrinsic_name,
        BackendTarget::Maca => {
            maca_intrinsic_for(intrinsic_name).unwrap_or(intrinsic_name)
        }
    };
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let func_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), vec![], false);
    let call_op = call_intrinsic(ctx, rewriter, op, resolved_name, func_ty, vec![])?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Lower a special-register read through exact inline PTX.
///
/// This is used when no LLVM intrinsic exists on every supported LLVM
/// version, when the modern PTX result is wider than LLVM's legacy intrinsic,
/// or when the register is a location sample that must be read again at every
/// source call. `kind` selects whether LLVM may common or remove the read.
pub(crate) fn convert_sreg_read_inline(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    result_width: u32,
    asm_template: &str,
    constraints: &str,
    kind: AsmKind,
) -> Result<()> {
    let result_ty = IntegerType::get(ctx, result_width, Signedness::Signless);
    let inline_asm = llvm_export::ops::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        vec![],
        asm_template,
        constraints,
        kind,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `mir.barrier0` to `llvm.nvvm.barrier0` (CUDA) or `llvm_mxc_barrier` (MXMACA).
pub(crate) fn convert_barrier0(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let opts = lowering_options(ctx);
    let intrinsic_name = match opts.backend {
        BackendTarget::Cuda => "llvm_nvvm_barrier0",
        BackendTarget::Maca => "llvm_mxc_barrier",
    };
    let void_ty = llvm_types::VoidType::get(ctx);
    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);
    call_intrinsic(ctx, rewriter, op, intrinsic_name, func_ty, vec![])?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

// ============================================================================
// MXMACA blockDim/gridDim via dispatch.ptr
// ============================================================================

/// MXMACA dispatch structure offsets (from LLVM IR analysis).
const MACA_DISPATCH_BLOCKDIM_OFFSET: u32 = 4;   // blockDim.x (i16) | blockDim.y (i16) packed as i32
const MACA_DISPATCH_GRIDDIM_X_OFFSET: u32 = 12;  // gridDim.x (i32)
const MACA_DISPATCH_GRIDDIM_Y_OFFSET: u32 = 16;  // gridDim.y (i32)
const MACA_DISPATCH_GRIDDIM_Z_OFFSET: u32 = 20;  // gridDim.z (i32)

/// Call `@llvm.mxc.dispatch.ptr()` and return the dispatch pointer value.
fn get_maca_dispatch_ptr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
) -> Result<pliron::value::Value> {
    let ptr_ty = llvm_types::PointerType::get(ctx, 4); // addrspace(4)
    let func_ty = llvm_types::FuncType::get(ctx, ptr_ty.into(), vec![], false);
    let call_op = call_intrinsic(ctx, rewriter, current_op, "llvm_mxc_dispatch_ptr", func_ty, vec![])?;
    Ok(call_op.deref(ctx).get_result(0))
}

/// Read an i32 value from the MXMACA dispatch structure at the given byte offset.
fn read_dispatch_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    offset: u32,
) -> Result<pliron::value::Value> {
    let dispatch_ptr = get_maca_dispatch_ptr(ctx, rewriter, current_op)?;

    // GEP to the offset (byte-level GEP via i8 pointer)
    let i8_ptr_ty = llvm_types::PointerType::get(ctx, 4); // addrspace(4) i8*
    let gep_indices = vec![
        llvm::GepIndex::Constant(offset),
    ];
    let gep_op = llvm::GetElementPtrOp::new(ctx, dispatch_ptr, gep_indices, i8_ptr_ty.into());
    rewriter.insert_operation(ctx, gep_op.get_operation());
    let field_ptr = gep_op.get_operation().deref(ctx).get_result(0);

    // Load i32
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let load_op = llvm::LoadOp::new(ctx, field_ptr, i32_ty.into());
    rewriter.insert_operation(ctx, load_op.get_operation());
    Ok(load_op.get_operation().deref(ctx).get_result(0))
}

/// Convert `ntid.x` (blockDim.x) for MXMACA: read i32 from dispatch+4, AND with 0xFFFF.
pub(crate) fn convert_maca_blockdim_x(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let packed = read_dispatch_i32(ctx, rewriter, op, MACA_DISPATCH_BLOCKDIM_OFFSET)?;
    // blockDim.x is the lower 16 bits
    let mask = create_i32_const(ctx, rewriter, 0xFFFF);
    let and_op = llvm::AndOp::new(ctx, packed, mask);
    rewriter.insert_operation(ctx, and_op.get_operation());
    rewriter.replace_operation(ctx, op, and_op.get_operation());
    Ok(())
}

/// Convert `ntid.y` (blockDim.y) for MXMACA: read i32 from dispatch+4, LSHR by 16.
pub(crate) fn convert_maca_blockdim_y(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let packed = read_dispatch_i32(ctx, rewriter, op, MACA_DISPATCH_BLOCKDIM_OFFSET)?;
    // blockDim.y is the upper 16 bits
    let shift = create_i32_const(ctx, rewriter, 16);
    let lshr_op = llvm::LShrOp::new(ctx, packed, shift);
    rewriter.insert_operation(ctx, lshr_op.get_operation());
    let lshr_result = lshr_op.get_operation().deref(ctx).get_result(0);
    let mask = create_i32_const(ctx, rewriter, 0xFFFF);
    let and_op = llvm::AndOp::new(ctx, lshr_result, mask);
    rewriter.insert_operation(ctx, and_op.get_operation());
    rewriter.replace_operation(ctx, op, and_op.get_operation());
    Ok(())
}

/// Convert `nctaid.x` (gridDim.x) for MXMACA: read i32 from dispatch+12.
pub(crate) fn convert_maca_griddim_x(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let val = read_dispatch_i32(ctx, rewriter, op, MACA_DISPATCH_GRIDDIM_X_OFFSET)?;
    rewriter.replace_operation(ctx, op, val.defining_op().unwrap());
    Ok(())
}

/// Convert `nctaid.y` (gridDim.y) for MXMACA: read i32 from dispatch+16.
pub(crate) fn convert_maca_griddim_y(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let val = read_dispatch_i32(ctx, rewriter, op, MACA_DISPATCH_GRIDDIM_Y_OFFSET)?;
    rewriter.replace_operation(ctx, op, val.defining_op().unwrap());
    Ok(())
}

/// Convert `nctaid.z` (gridDim.z) for MXMACA: read i32 from dispatch+20.
pub(crate) fn convert_maca_griddim_z(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let val = read_dispatch_i32(ctx, rewriter, op, MACA_DISPATCH_GRIDDIM_Z_OFFSET)?;
    rewriter.replace_operation(ctx, op, val.defining_op().unwrap());
    Ok(())
}

fn convert_membar(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    asm_template: &str,
) -> Result<()> {
    let opts = lowering_options(ctx);
    if opts.backend == BackendTarget::Maca {
        // MXMACA uses standard LLVM fence instructions with syncscopes
        let scope = match asm_template {
            "membar.cta;" => Some("block".to_string()),
            "membar.gl;" => Some("device".to_string()),
            "membar.sys;" => None, // system scope = no syncscope
            _ => None,
        };
        let ordering = LlvmAtomicOrdering::SeqCst;
        let fence_op = llvm::FenceOp::new(ctx, ordering, scope);
        rewriter.insert_operation(ctx, fence_op.get_operation());
        rewriter.erase_operation(ctx, op);
    } else {
        let void_ty = llvm_types::VoidType::get(ctx);
        inline_asm_convergent(
            ctx,
            rewriter,
            void_ty.into(),
            vec![],
            asm_template,
            "~{memory}",
        );
        rewriter.erase_operation(ctx, op);
    }
    Ok(())
}

/// Convert a block-scoped memory fence to inline PTX `membar.cta`.
pub(crate) fn convert_threadfence_block(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_membar(ctx, rewriter, op, "membar.cta;")
}

/// Convert a device-scoped memory fence to inline PTX `membar.gl`.
pub(crate) fn convert_threadfence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_membar(ctx, rewriter, op, "membar.gl;")
}

/// Convert a system-scoped memory fence to inline PTX `membar.sys`.
pub(crate) fn convert_threadfence_system(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_membar(ctx, rewriter, op, "membar.sys;")
}
