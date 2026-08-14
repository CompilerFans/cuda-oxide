/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Atomic operation conversion: NVVM atomic dialect → LLVM atomic instructions.
//!
//! Converts NVVM atomic ops to standard LLVM atomic instructions with
//! proper ordering and syncscope.
//!
//! # Lowering Strategy
//!
//! Unlike most GPU intrinsics that lower to LLVM NVVM intrinsic calls or
//! inline PTX, atomic operations lower to **standard LLVM IR instructions**:
//!
//! | NVVM Op                 | LLVM IR                                  |
//! |-------------------------|------------------------------------------|
//! | `NvvmAtomicLoadOp`      | `load atomic ... syncscope("device")`    |
//! | `NvvmAtomicStoreOp`     | `store atomic ... syncscope("device")`   |
//! | `NvvmAtomicRmwOp`       | `atomicrmw ... syncscope("device")` `[*]`  |
//! | `NvvmAtomicCmpxchgOp`   | `cmpxchg ... syncscope("device")`        |
//!
//! `[*]` atomicrmw uses fence splitting workaround -- see below.
//!
//! # atomicrmw Fence Splitting Workaround
//!
//! LLVM's NVPTX backend silently drops orderings on `atomicrmw`
//! (fix is in LLVM 23 via PR #176015). Until then, we emit:
//!
//! ```text
//! Relaxed:  atomicrmw ... monotonic
//! Acquire:  atomicrmw ... monotonic  +  fence acquire
//! Release:  fence release  +  atomicrmw ... monotonic
//! AcqRel:   fence release  +  atomicrmw ... monotonic  +  fence acquire
//! SeqCst:   fence seq_cst  +  atomicrmw ... monotonic  +  fence seq_cst
//! ```
//!
//! All fences carry the same syncscope as the atomic op.
//!
//! # Scope → Syncscope Mapping
//!
//! | NVVM Scope | LLVM syncscope     | PTX scope |
//! |------------|--------------------|-----------|
//! | Device     | `"device"`         | `.gpu`    |
//! | Block      | `"block"`          | `.cta`    |
//! | System     | (default)          | `.sys`    |

use crate::convert::types::convert_type;

use dialect_nvvm::ops::atomic::{
    AtomicOrdering as NvvmOrdering, AtomicRmwKind as NvvmRmwKind, AtomicScope as NvvmScope,
    NvvmAtomicCmpxchgOp, NvvmAtomicLoadOp, NvvmAtomicOpInterface, NvvmAtomicRmwOp,
    NvvmAtomicStoreOp,
};
use llvm_export::attributes::{LlvmAtomicOrdering, LlvmAtomicRmwKind, LlvmSyncScope};
use llvm_export::op_interfaces::{
    BinArithOp, CastOpInterface, CastOpWithNNegInterface, FloatBinArithOpWithFastMathFlags,
    IntBinArithOpWithOverflowFlag,
};
use llvm_export::ops as llvm;
use llvm_export::ops::{AsmKind, InlineAsmOpExt};

use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::linked_list::ContainsLinkedList;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

// =============================================================================
// Scope / Ordering Mapping
// =============================================================================

fn map_scope(scope: &NvvmScope) -> LlvmSyncScope {
    match scope {
        NvvmScope::Device => LlvmSyncScope::Device,
        NvvmScope::Block => LlvmSyncScope::Block,
        NvvmScope::System => LlvmSyncScope::System,
    }
}

fn map_ordering(ord: &NvvmOrdering) -> LlvmAtomicOrdering {
    match ord {
        NvvmOrdering::Relaxed => LlvmAtomicOrdering::Monotonic,
        NvvmOrdering::Acquire => LlvmAtomicOrdering::Acquire,
        NvvmOrdering::Release => LlvmAtomicOrdering::Release,
        NvvmOrdering::AcqRel => LlvmAtomicOrdering::AcqRel,
        NvvmOrdering::SeqCst => LlvmAtomicOrdering::SeqCst,
    }
}

fn map_rmw_kind(kind: &NvvmRmwKind) -> LlvmAtomicRmwKind {
    match kind {
        NvvmRmwKind::Add => LlvmAtomicRmwKind::Add,
        NvvmRmwKind::Sub => LlvmAtomicRmwKind::Sub,
        NvvmRmwKind::And => LlvmAtomicRmwKind::And,
        NvvmRmwKind::Or => LlvmAtomicRmwKind::Or,
        NvvmRmwKind::Xor => LlvmAtomicRmwKind::Xor,
        NvvmRmwKind::Xchg => LlvmAtomicRmwKind::Xchg,
        NvvmRmwKind::Min => LlvmAtomicRmwKind::Min,
        NvvmRmwKind::Max => LlvmAtomicRmwKind::Max,
        NvvmRmwKind::UMin => LlvmAtomicRmwKind::UMin,
        NvvmRmwKind::UMax => LlvmAtomicRmwKind::UMax,
        NvvmRmwKind::FAdd => LlvmAtomicRmwKind::FAdd,
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn emit_fence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ordering: LlvmAtomicOrdering,
    syncscope: LlvmSyncScope,
) {
    let fence = llvm::FenceOp::new(ctx, ordering, syncscope.to_pliron());
    rewriter.insert_operation(ctx, fence.get_operation());
}

// =============================================================================
// Load
// =============================================================================

pub(crate) fn convert_atomic_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicLoadOp::new(op);
    let ordering = map_ordering(&nvvm_op.ordering(ctx));
    let syncscope = map_scope(&nvvm_op.scope(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let ptr = operands[0];
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty =
        convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let llvm_load = llvm::AtomicLoadOp::new(ctx, ptr, result_ty, ordering, syncscope.to_pliron());
    rewriter.insert_operation(ctx, llvm_load.get_operation());
    rewriter.replace_operation(ctx, op, llvm_load.get_operation());

    Ok(())
}

// =============================================================================
// Store
// =============================================================================

pub(crate) fn convert_atomic_store(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicStoreOp::new(op);
    let ordering = map_ordering(&nvvm_op.ordering(ctx));
    let syncscope = map_scope(&nvvm_op.scope(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let val = operands[0];
    let ptr = operands[1];

    let llvm_store = llvm::AtomicStoreOp::new(ctx, val, ptr, ordering, syncscope.to_pliron());
    rewriter.insert_operation(ctx, llvm_store.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

// =============================================================================
// Read-Modify-Write (with fence splitting workaround)
// =============================================================================

pub(crate) fn convert_atomic_rmw(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicRmwOp::new(op);
    let nvvm_ordering = nvvm_op.ordering(ctx);
    let syncscope = map_scope(&nvvm_op.scope(ctx));
    let rmw_kind = map_rmw_kind(&nvvm_op.rmw_kind(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let ptr = operands[0];
    let val = operands[1];

    // The C500 backend rejects every `atomicrmw` on f16 ("the AtomicInst is
    // not support in backend") but accepts `cmpxchg i16`. Lower f16 RMWs to
    // a compare-exchange loop in a lazily-created device helper.
    if crate::context::lowering_options(ctx).backend == crate::BackendTarget::Maca
        && matches!(
            rmw_kind,
            LlvmAtomicRmwKind::FAdd | LlvmAtomicRmwKind::FSub | LlvmAtomicRmwKind::Xchg
        )
        && val.get_type(ctx).deref(ctx).is::<llvm_export::types::HalfType>()
    {
        let (helper_name, helper_ty) = ensure_maca_f16_rmw_helper(ctx, op, rmw_kind, syncscope)?;
        let callee = pliron::builtin::op_interfaces::CallOpCallable::Direct(helper_name);
        let call = llvm::CallOp::new(ctx, callee, helper_ty, vec![ptr, val]);
        rewriter.insert_operation(ctx, call.get_operation());
        rewriter.replace_operation(ctx, op, call.get_operation());
        return Ok(());
    }

    // Fence splitting workaround for LLVM NVPTX atomicrmw ordering bug.
    // We emit: [optional pre-fence] + atomicrmw monotonic + [optional post-fence]
    // The actual atomicrmw always uses Monotonic because LLVM drops the
    // ordering anyway. The fences provide the correct ordering semantics.

    // Pre-fence (if needed)
    match nvvm_ordering {
        NvvmOrdering::Release | NvvmOrdering::AcqRel => {
            emit_fence(ctx, rewriter, LlvmAtomicOrdering::Release, syncscope);
        }
        NvvmOrdering::SeqCst => {
            emit_fence(ctx, rewriter, LlvmAtomicOrdering::SeqCst, syncscope);
        }
        NvvmOrdering::Relaxed | NvvmOrdering::Acquire => {}
    }

    // The atomicrmw itself -- always Monotonic
    let llvm_rmw = llvm::AtomicRmwOp::new(
        ctx,
        ptr,
        val,
        rmw_kind,
        LlvmAtomicOrdering::Monotonic,
        syncscope.to_pliron(),
    );
    rewriter.insert_operation(ctx, llvm_rmw.get_operation());

    // Post-fence (if needed)
    match nvvm_ordering {
        NvvmOrdering::Acquire | NvvmOrdering::AcqRel => {
            emit_fence(ctx, rewriter, LlvmAtomicOrdering::Acquire, syncscope);
        }
        NvvmOrdering::SeqCst => {
            emit_fence(ctx, rewriter, LlvmAtomicOrdering::SeqCst, syncscope);
        }
        NvvmOrdering::Relaxed | NvvmOrdering::Release => {}
    }

    rewriter.replace_operation(ctx, op, llvm_rmw.get_operation());

    Ok(())
}

/// Lazily create the `__cuda_oxide_maca_f16_{fadd,fsub,xchg}` device helper:
/// an f16 atomic read-modify-write built from a `cmpxchg i16` loop (the only
/// 16-bit read-modify-write the C500 backend accepts). Returns the helper's
/// symbol and function type for the call site.
///
/// Body for `fadd` (opaque pointers, so the f16 pointer needs no bitcast):
/// ```llvm
/// loop:
///   %old  = load atomic i16, ptr %p monotonic
///   %oldf = bitcast i16 %old to half
///   %newf = fadd half %oldf, %v      ; fsub for `fsub`; %v itself for `xchg`
///   %new  = bitcast half %newf to i16
///   %pair = cmpxchg ptr %p, i16 %old, i16 %new monotonic monotonic
///   %ok   = extractvalue { i16, i1 } %pair, 1
///   br i1 %ok, label %exit, label %loop
/// exit:
///   %r = bitcast i16 (extractvalue %pair, 0) to half
///   ret half %r
/// ```
fn ensure_maca_f16_rmw_helper(
    ctx: &mut Context,
    op: Ptr<Operation>,
    rmw_kind: LlvmAtomicRmwKind,
    syncscope: LlvmSyncScope,
) -> Result<(
    pliron::identifier::Identifier,
    pliron::r#type::TypedHandle<llvm_export::types::FuncType>,
)> {
    let kind_suffix = match rmw_kind {
        LlvmAtomicRmwKind::FAdd => "fadd",
        LlvmAtomicRmwKind::FSub => "fsub",
        LlvmAtomicRmwKind::Xchg => "xchg",
        other => {
            return pliron::input_err_noloc!("MACA f16 RMW helper: unsupported kind {other:?}")
        }
    };
    let helper_name = format!("__cuda_oxide_maca_f16_{kind_suffix}");

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| pliron::input_error_noloc!("f16 rmw: op has no parent block"))?;
    let func_op = parent_block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| pliron::input_error_noloc!("f16 rmw: block has no parent function"))?;
    let module_op = func_op
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| pliron::input_error_noloc!("f16 rmw: function has no parent module"))?;
    let module_block = module_op
        .deref(ctx)
        .regions()
        .next()
        .and_then(|region| region.deref(ctx).iter(ctx).next())
        .ok_or_else(|| pliron::input_error_noloc!("f16 rmw: module has no block"))?;

    let half_ty = llvm_export::types::HalfType::get(ctx);
    let ptr_ty = llvm_export::types::PointerType::get(ctx, 0);
    let i16_ty = IntegerType::get(ctx, 16, Signedness::Signless);
    let func_ty = llvm_export::types::FuncType::get(
        ctx,
        half_ty.into(),
        vec![ptr_ty.into(), half_ty.into()],
        false,
    );

    // Reuse the existing helper when the module already has one.
    for existing in module_block.deref(ctx).iter(ctx) {
        if let Some(func) = Operation::get_op::<llvm::FuncOp>(existing, ctx)
            && func.get_symbol_name(ctx).to_string() == helper_name
        {
            return Ok((helper_name.as_str().try_into().unwrap(), func_ty));
        }
    }

    let helper = llvm::FuncOp::new(ctx, helper_name.as_str().try_into().unwrap(), func_ty);
    let helper_op = helper.get_operation();
    helper_op.insert_at_back(module_block, ctx);

    let entry = helper.get_or_create_entry_block(ctx);
    let region = entry
        .deref(ctx)
        .get_parent_region()
        .ok_or_else(|| pliron::input_error_noloc!("f16 rmw: entry block has no region"))?;
    let p = entry.deref(ctx).get_argument(0);
    let v = entry.deref(ctx).get_argument(1);

    // The C500 backend's expansion of a 16-bit `cmpxchg` is not safe when
    // wave-mates concurrently update the sibling half of the same 32-bit
    // word (adjacent f16 bins double-count). Do the word-level CAS by hand
    // with only 32-bit memory ops, which the backend handles correctly.
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    // Word-aligned address and the bit shift of this half within the word.
    // All entry-block ops (including constants used by later blocks) are
    // inserted at once, so every loop/exit operand is dominated.
    let loop_bb = pliron::basic_block::BasicBlock::new(ctx, None, vec![]);
    loop_bb.insert_at_back(region, ctx);
    let exit_bb = pliron::basic_block::BasicBlock::new(ctx, None, vec![]);
    exit_bb.insert_at_back(region, ctx);

    let const3 = emit_i64_const(ctx, entry, i64_ty, 3);
    let const8 = emit_i64_const(ctx, entry, i64_ty, 8);
    let const_m4 = emit_i64_const(ctx, entry, i64_ty, -4);
    let const_ffff = emit_i64_const(ctx, entry, i64_ty, 0xFFFF);
    let const_m1 = emit_i64_const(ctx, entry, i64_ty, -1);

    let pi = llvm::PtrToIntOp::new(ctx, p, i64_ty.into());
    pi.get_operation().insert_at_back(entry, ctx);
    let pi_v = pi.get_operation().deref(ctx).get_result(0);

    let byte_off = llvm::AndOp::new(ctx, pi_v, const3);
    byte_off.get_operation().insert_at_back(entry, ctx);
    let byte_off_v = byte_off.get_operation().deref(ctx).get_result(0);

    let shift = llvm::MulOp::new_with_overflow_flag(
        ctx,
        byte_off_v,
        const8,
        llvm_export::attributes::IntegerOverflowFlagsAttr::default(),
    );
    shift.get_operation().insert_at_back(entry, ctx);
    let shift_v = shift.get_operation().deref(ctx).get_result(0);

    let word_addr = llvm::AndOp::new(ctx, pi_v, const_m4);
    word_addr.get_operation().insert_at_back(entry, ctx);
    let word_addr_v = word_addr.get_operation().deref(ctx).get_result(0);

    let wp = llvm::IntToPtrOp::new(ctx, word_addr_v, ptr_ty.into());
    wp.get_operation().insert_at_back(entry, ctx);
    let wp_v = wp.get_operation().deref(ctx).get_result(0);

    llvm::BrOp::new(ctx, loop_bb, vec![])
        .get_operation()
        .insert_at_back(entry, ctx);

    // Loop body: word load, extract the half, apply the update, rebuild the
    // word, compare-exchange the whole word, branch on success.
    let word = llvm::AtomicLoadOp::new(
        ctx,
        wp_v,
        i32_ty.into(),
        LlvmAtomicOrdering::Monotonic,
        syncscope.to_pliron(),
    );
    word.get_operation().insert_at_back(loop_bb, ctx);
    let word_v = word.get_operation().deref(ctx).get_result(0);

    let word64 = llvm::ZExtOp::new_with_nneg(ctx, word_v, i64_ty.into(), false);
    word64.get_operation().insert_at_back(loop_bb, ctx);
    let word64_v = word64.get_operation().deref(ctx).get_result(0);

    let old_shifted = llvm::LShrOp::new(ctx, word64_v, shift_v);
    old_shifted.get_operation().insert_at_back(loop_bb, ctx);
    let old_shifted_v = old_shifted.get_operation().deref(ctx).get_result(0);

    let old16 = llvm::TruncOp::new(ctx, old_shifted_v, i16_ty.into());
    old16.get_operation().insert_at_back(loop_bb, ctx);
    let old16_v = old16.get_operation().deref(ctx).get_result(0);

    // The updated half as i16. For xchg the update is just the input.
    let new16_v = if rmw_kind == LlvmAtomicRmwKind::Xchg {
        let as_i16 = llvm::BitcastOp::new(ctx, v, i16_ty.into());
        as_i16.get_operation().insert_at_back(loop_bb, ctx);
        as_i16.get_operation().deref(ctx).get_result(0)
    } else {
        let old_f = llvm::BitcastOp::new(ctx, old16_v, half_ty.into());
        old_f.get_operation().insert_at_back(loop_bb, ctx);
        let old_fv = old_f.get_operation().deref(ctx).get_result(0);

        let flags = llvm_export::attributes::FastmathFlagsAttr::default();
        let new_f_op = if rmw_kind == LlvmAtomicRmwKind::FAdd {
            llvm::FAddOp::new_with_fast_math_flags(ctx, old_fv, v, flags).get_operation()
        } else {
            llvm::FSubOp::new_with_fast_math_flags(ctx, old_fv, v, flags).get_operation()
        };
        new_f_op.insert_at_back(loop_bb, ctx);
        let new_fv = new_f_op.deref(ctx).get_result(0);

        let new_i = llvm::BitcastOp::new(ctx, new_fv, i16_ty.into());
        new_i.get_operation().insert_at_back(loop_bb, ctx);
        new_i.get_operation().deref(ctx).get_result(0)
    };

    let new64 = llvm::ZExtOp::new_with_nneg(ctx, new16_v, i64_ty.into(), false);
    new64.get_operation().insert_at_back(loop_bb, ctx);
    let new64_v = new64.get_operation().deref(ctx).get_result(0);

    let new_sh = llvm::ShlOp::new_with_overflow_flag(
        ctx,
        new64_v,
        shift_v,
        llvm_export::attributes::IntegerOverflowFlagsAttr::default(),
    );
    new_sh.get_operation().insert_at_back(loop_bb, ctx);
    let new_sh_v = new_sh.get_operation().deref(ctx).get_result(0);

    let mask_sh = llvm::ShlOp::new_with_overflow_flag(
        ctx,
        const_ffff,
        shift_v,
        llvm_export::attributes::IntegerOverflowFlagsAttr::default(),
    );
    mask_sh.get_operation().insert_at_back(loop_bb, ctx);
    let mask_sh_v = mask_sh.get_operation().deref(ctx).get_result(0);

    let mask_neg = llvm::XorOp::new(ctx, mask_sh_v, const_m1);
    mask_neg.get_operation().insert_at_back(loop_bb, ctx);
    let mask_neg_v = mask_neg.get_operation().deref(ctx).get_result(0);

    let word_masked = llvm::AndOp::new(ctx, word64_v, mask_neg_v);
    word_masked.get_operation().insert_at_back(loop_bb, ctx);
    let word_masked_v = word_masked.get_operation().deref(ctx).get_result(0);

    let new_word64 = llvm::OrOp::new(ctx, word_masked_v, new_sh_v);
    new_word64.get_operation().insert_at_back(loop_bb, ctx);
    let new_word64_v = new_word64.get_operation().deref(ctx).get_result(0);

    let new_word = llvm::TruncOp::new(ctx, new_word64_v, i32_ty.into());
    new_word.get_operation().insert_at_back(loop_bb, ctx);
    let new_word_v = new_word.get_operation().deref(ctx).get_result(0);

    let pair = llvm::AtomicCmpxchgOp::new(
        ctx,
        wp_v,
        word_v,
        new_word_v,
        LlvmAtomicOrdering::Monotonic,
        LlvmAtomicOrdering::Monotonic,
        syncscope.to_pliron(),
    );
    pair.get_operation().insert_at_back(loop_bb, ctx);
    let pair_v = pair.get_operation().deref(ctx).get_result(0);

    let ok = llvm::ExtractValueOp::new(ctx, pair_v, vec![1])
        .map_err(|e| pliron::input_error_noloc!("f16 rmw extractvalue: {e}"))?;
    ok.get_operation().insert_at_back(loop_bb, ctx);
    let ok_v = ok.get_operation().deref(ctx).get_result(0);

    llvm::CondBrOp::new(ctx, ok_v, exit_bb, vec![], loop_bb, vec![])
        .get_operation()
        .insert_at_back(loop_bb, ctx);

    // On success the cmpxchg's returned word still holds the OLD half
    // (the successful CAS swapped in `new_word` and returned the previous
    // word); extract it back to the f16 result.
    let act = llvm::ExtractValueOp::new(ctx, pair_v, vec![0])
        .map_err(|e| pliron::input_error_noloc!("f16 rmw extractvalue: {e}"))?;
    act.get_operation().insert_at_back(exit_bb, ctx);
    let act_v = act.get_operation().deref(ctx).get_result(0);

    let act64 = llvm::ZExtOp::new_with_nneg(ctx, act_v, i64_ty.into(), false);
    act64.get_operation().insert_at_back(exit_bb, ctx);
    let act64_v = act64.get_operation().deref(ctx).get_result(0);

    let act_sh = llvm::LShrOp::new(ctx, act64_v, shift_v);
    act_sh.get_operation().insert_at_back(exit_bb, ctx);
    let act_sh_v = act_sh.get_operation().deref(ctx).get_result(0);

    let act16 = llvm::TruncOp::new(ctx, act_sh_v, i16_ty.into());
    act16.get_operation().insert_at_back(exit_bb, ctx);
    let act16_v = act16.get_operation().deref(ctx).get_result(0);

    let res = llvm::BitcastOp::new(ctx, act16_v, half_ty.into());
    res.get_operation().insert_at_back(exit_bb, ctx);
    let res_v = res.get_operation().deref(ctx).get_result(0);
    llvm::ReturnOp::new(ctx, Some(res_v))
        .get_operation()
        .insert_at_back(exit_bb, ctx);

    Ok((helper_name.as_str().try_into().unwrap(), func_ty))
}

/// Create an `i64` constant op inserted at the back of `block`, returning
/// its result (used while building the f16 RMW helper's entry block; the
/// rewriter is not in scope there).
fn emit_i64_const(
    ctx: &mut Context,
    block: Ptr<pliron::basic_block::BasicBlock>,
    i64_ty: pliron::r#type::TypedHandle<IntegerType>,
    n: i64,
) -> pliron::value::Value {
    let apint = pliron::utils::apint::APInt::from_i64(n, std::num::NonZeroUsize::new(64).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
    let c = llvm::ConstantOp::new(ctx, attr.into());
    c.get_operation().insert_at_back(block, ctx);
    c.get_operation().deref(ctx).get_result(0)
}

// =============================================================================
// Compare-and-Exchange
// =============================================================================

pub(crate) fn convert_atomic_cmpxchg(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicCmpxchgOp::new(op);
    let success_ord = map_ordering(&nvvm_op.success_ordering(ctx));
    let failure_ord = map_ordering(&nvvm_op.failure_ordering(ctx));
    let syncscope = map_scope(&nvvm_op.scope(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let ptr = operands[0];
    let cmp = operands[1];
    let new_val = operands[2];
    let llvm_cmpxchg = llvm::AtomicCmpxchgOp::new(
        ctx,
        ptr,
        cmp,
        new_val,
        success_ord,
        failure_ord,
        syncscope.to_pliron(),
    );
    rewriter.insert_operation(ctx, llvm_cmpxchg.get_operation());

    // Upstream `cmpxchg` returns `{ T, i1 }`, but the NVVM op models only the
    // loaded value `T`. Extract element 0 and replace the NVVM op with it; this
    // emits the same `cmpxchg` + `extractvalue` LLVM as the pre-migration path.
    let cmpxchg_res = llvm_cmpxchg.get_operation().deref(ctx).get_result(0);
    let extract = llvm::ExtractValueOp::new(ctx, cmpxchg_res, vec![0])
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;
    rewriter.insert_operation(ctx, extract.get_operation());
    rewriter.replace_operation(ctx, op, extract.get_operation());

    Ok(())
}

// =============================================================================
// Packed Atomic Add (f16x2, bf16x2) -- inline PTX
// =============================================================================

/// Convert a packed atomic add op to inline PTX.
///
/// Constraints: `=r,l,r,~{memory}` -- output register, address pointer, input
/// register, memory clobber.
///
/// Uses `SideEffect` (not convergent): atomics are per-thread, not
/// warp-synchronous.
fn convert_packed_atom_add(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_type: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!(
            "packed atomic add requires 2 operands (address, addend), got {}",
            operands.len()
        );
    }
    let addr = operands[0];
    let val = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![addr, val],
        &format!("atom.global.add.noftz.{ptx_type} $0, [$1], $2;"),
        "=r,l,r,~{memory}",
        AsmKind::SideEffect,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `nvvm.atom_add_f16x2` to inline PTX.
pub(crate) fn convert_atom_add_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_packed_atom_add(ctx, rewriter, op, "f16x2")
}

/// Convert `nvvm.atom_add_bf16x2` to inline PTX.
pub(crate) fn convert_atom_add_bf16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_packed_atom_add(ctx, rewriter, op, "bf16x2")
}
