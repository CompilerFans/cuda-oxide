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
use crate::context::lowering_options;
use crate::convert::intrinsics::common::*;
use llvm_export::attributes::{ICmpPredicateAttr, IntegerOverflowFlagsAttr, LlvmAtomicOrdering};
use llvm_export::op_interfaces::{
    BinArithOp, CastOpWithNNegInterface, IntBinArithOpWithOverflowFlag,
};
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
        BackendTarget::Maca => maca_intrinsic_for(intrinsic_name).unwrap_or(intrinsic_name),
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
// MXMACA lane_id via mbcnt.lo + mbcnt.hi
// ============================================================================

/// Convert `lane_id` for MXMACA: `mbcnt.lo(-1, 0)` + `mbcnt.hi(-1, lo)`.
///
/// MXMACA uses two `i32` intrinsics to count preceding lanes across both halves
/// of the 64-thread wave. The second call continues from the first call's
/// accumulator and yields the lane ID in the range 0..63.
pub(crate) fn convert_maca_lane_id(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let neg1 = create_i32_const(ctx, rewriter, -1);
    let zero = create_i32_const(ctx, rewriter, 0);

    // mbcnt.lo(-1, 0)
    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into()],
        false,
    );
    let mbcnt_lo = call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_mxc_mbcnt_lo",
        func_ty,
        vec![neg1, zero],
    )?;
    let lo_val = mbcnt_lo.deref(ctx).get_result(0);

    // mbcnt.hi(-1, lo)
    let func_ty2 = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into()],
        false,
    );
    let mbcnt_hi = call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_mxc_mbcnt_hi",
        func_ty2,
        vec![neg1, lo_val],
    )?;

    rewriter.replace_operation(ctx, op, mbcnt_hi);
    Ok(())
}

// ============================================================================
// MXMACA blockDim/gridDim via dispatch.ptr
// ============================================================================

/// MXMACA dispatch structure offsets (from LLVM IR analysis).
const MACA_DISPATCH_BLOCKDIM_XY_OFFSET: u32 = 4; // blockDim.x (i16) | blockDim.y (i16)
const MACA_DISPATCH_BLOCKDIM_Z_OFFSET: u32 = 8; // blockDim.z (low i16)
const MACA_DISPATCH_GLOBAL_SIZE_X_OFFSET: u32 = 12; // global_size.x (i32)
const MACA_DISPATCH_GLOBAL_SIZE_Y_OFFSET: u32 = 16; // global_size.y (i32)
const MACA_DISPATCH_GLOBAL_SIZE_Z_OFFSET: u32 = 20; // global_size.z (i32)

/// Call `@llvm.mxc.dispatch.ptr()` and return the dispatch pointer value.
fn get_maca_dispatch_ptr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
) -> Result<pliron::value::Value> {
    let ptr_ty = llvm_types::PointerType::get(ctx, 4); // addrspace(4)
    let func_ty = llvm_types::FuncType::get(ctx, ptr_ty.into(), vec![], false);
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        current_op,
        "llvm_mxc_dispatch_ptr",
        func_ty,
        vec![],
    )?;
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

    // GEP to the offset using i8 as the source element type.
    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
    let gep_indices = vec![llvm::GepIndex::Constant(offset)];
    let gep_op = llvm::GetElementPtrOp::new(ctx, dispatch_ptr, gep_indices, i8_ty.into());
    rewriter.insert_operation(ctx, gep_op.get_operation());
    let field_ptr = gep_op.get_operation().deref(ctx).get_result(0);

    // Load i32
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let load_op = llvm::LoadOp::new(ctx, field_ptr, i32_ty.into());
    rewriter.insert_operation(ctx, load_op.get_operation());
    Ok(load_op.get_operation().deref(ctx).get_result(0))
}

fn read_maca_blockdim(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    offset: u32,
    shift: i32,
) -> Result<pliron::value::Value> {
    let packed = read_dispatch_i32(ctx, rewriter, current_op, offset)?;
    let value = if shift == 0 {
        packed
    } else {
        let shift = create_i32_const(ctx, rewriter, shift);
        let lshr_op = llvm::LShrOp::new(ctx, packed, shift);
        rewriter.insert_operation(ctx, lshr_op.get_operation());
        lshr_op.get_operation().deref(ctx).get_result(0)
    };
    let mask = create_i32_const(ctx, rewriter, 0xFFFF);
    let and_op = llvm::AndOp::new(ctx, value, mask);
    rewriter.insert_operation(ctx, and_op.get_operation());
    Ok(and_op.get_operation().deref(ctx).get_result(0))
}

fn convert_maca_blockdim(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    offset: u32,
    shift: i32,
) -> Result<()> {
    let value = read_maca_blockdim(ctx, rewriter, op, offset, shift)?;
    rewriter.replace_operation(ctx, op, value.defining_op().unwrap());
    Ok(())
}

/// Convert `ntid.x` (blockDim.x) for MXMACA.
pub(crate) fn convert_maca_blockdim_x(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_maca_blockdim(ctx, rewriter, op, MACA_DISPATCH_BLOCKDIM_XY_OFFSET, 0)
}

/// Convert `ntid.y` (blockDim.y) for MXMACA.
pub(crate) fn convert_maca_blockdim_y(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_maca_blockdim(ctx, rewriter, op, MACA_DISPATCH_BLOCKDIM_XY_OFFSET, 16)
}

/// Convert `ntid.z` (blockDim.z) for MXMACA.
pub(crate) fn convert_maca_blockdim_z(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_maca_blockdim(ctx, rewriter, op, MACA_DISPATCH_BLOCKDIM_Z_OFFSET, 0)
}

fn convert_maca_griddim(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    global_size_offset: u32,
    blockdim_offset: u32,
    blockdim_shift: i32,
) -> Result<()> {
    let global_size = read_dispatch_i32(ctx, rewriter, op, global_size_offset)?;
    let blockdim = read_maca_blockdim(ctx, rewriter, op, blockdim_offset, blockdim_shift)?;

    // The dispatch packet stores global work-item counts. CUDA gridDim is the
    // number of blocks, so use overflow-safe ceil(global_size / blockdim).
    let quotient_op = llvm::UDivOp::new(ctx, global_size, blockdim);
    rewriter.insert_operation(ctx, quotient_op.get_operation());
    let quotient = quotient_op.get_operation().deref(ctx).get_result(0);
    let overflow_flags = IntegerOverflowFlagsAttr::default();
    let product_op =
        llvm::MulOp::new_with_overflow_flag(ctx, quotient, blockdim, overflow_flags.clone());
    rewriter.insert_operation(ctx, product_op.get_operation());
    let product = product_op.get_operation().deref(ctx).get_result(0);
    let remainder_op = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::UGT, global_size, product);
    rewriter.insert_operation(ctx, remainder_op.get_operation());
    let has_remainder = remainder_op.get_operation().deref(ctx).get_result(0);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let extend_op = llvm::ZExtOp::new_with_nneg(ctx, has_remainder, i32_ty.into(), false);
    rewriter.insert_operation(ctx, extend_op.get_operation());
    let remainder = extend_op.get_operation().deref(ctx).get_result(0);
    let result_op = llvm::AddOp::new_with_overflow_flag(ctx, quotient, remainder, overflow_flags);
    rewriter.insert_operation(ctx, result_op.get_operation());
    rewriter.replace_operation(ctx, op, result_op.get_operation());
    Ok(())
}

/// Convert `nctaid.x` (gridDim.x) for MXMACA.
pub(crate) fn convert_maca_griddim_x(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_maca_griddim(
        ctx,
        rewriter,
        op,
        MACA_DISPATCH_GLOBAL_SIZE_X_OFFSET,
        MACA_DISPATCH_BLOCKDIM_XY_OFFSET,
        0,
    )
}

/// Convert `nctaid.y` (gridDim.y) for MXMACA.
pub(crate) fn convert_maca_griddim_y(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_maca_griddim(
        ctx,
        rewriter,
        op,
        MACA_DISPATCH_GLOBAL_SIZE_Y_OFFSET,
        MACA_DISPATCH_BLOCKDIM_XY_OFFSET,
        16,
    )
}

/// Convert `nctaid.z` (gridDim.z) for MXMACA.
pub(crate) fn convert_maca_griddim_z(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_maca_griddim(
        ctx,
        rewriter,
        op,
        MACA_DISPATCH_GLOBAL_SIZE_Z_OFFSET,
        MACA_DISPATCH_BLOCKDIM_Z_OFFSET,
        0,
    )
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
