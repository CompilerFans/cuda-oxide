/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level intrinsic conversion: shuffle and vote operations.
//!
//! # Shuffle Operations
//!
//! | Operation          | LLVM Intrinsic                 | Description       |
//! |--------------------|--------------------------------|-------------------|
//! | `ShflSyncIdxI32`   | `llvm.nvvm.shfl.sync.idx.i32`  | Indexed shuffle   |
//! | `ShflSyncBflyI32`  | `llvm.nvvm.shfl.sync.bfly.i32` | Butterfly shuffle |
//! | `ShflSyncDownI32`  | `llvm.nvvm.shfl.sync.down.i32` | Down shuffle      |
//! | `ShflSyncUpI32`    | `llvm.nvvm.shfl.sync.up.i32`   | Up shuffle        |
//!
//! # Vote Operations
//!
//! | Operation        | LLVM Intrinsic               | Description           |
//! |------------------|------------------------------|-----------------------|
//! | `VoteSyncAll`    | `llvm.nvvm.vote.all.sync`    | All lanes true        |
//! | `VoteSyncAny`    | `llvm.nvvm.vote.any.sync`    | Any lane true         |
//! | `VoteSyncBallot` | `llvm.nvvm.vote.ballot.sync` | Bitmask of predicates |
//!
//! # Match Operations (sm_70+)
//!
//! | Operation         | LLVM Intrinsic                    | Description                  |
//! |-------------------|-----------------------------------|------------------------------|
//! | `MatchAnySyncI32` | `llvm.nvvm.match.any.sync.i32`    | Mask of equal-value lanes    |
//! | `MatchAnySyncI64` | `llvm.nvvm.match.any.sync.i64`    | 64-bit variant               |
//! | `MatchAllSyncI32` | `llvm.nvvm.match.all.sync.i32p`   | Full mask iff all agree      |
//! | `MatchAllSyncI64` | `llvm.nvvm.match.all.sync.i64p`   | 64-bit variant               |

use crate::BackendTarget;
use crate::context::lowering_options;
use crate::convert::intrinsics::common::*;
use llvm_export::attributes::{ICmpPredicateAttr, IntegerOverflowFlagsAttr, LlvmAtomicOrdering};
use llvm_export::op_interfaces::{
    BinArithOp, CastOpInterface, CastOpWithNNegInterface, IntBinArithOpWithOverflowFlag,
};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert i32 shuffle operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, value, lane_or_delta]`. The mask reaches us
/// already type-converted by the framework (any `u32`/`i32` carrier
/// works); we forward it straight to the intrinsic. For full-warp ops
/// the mask is just `0xFFFFFFFF` baked in by the caller.
pub(crate) fn convert_shuffle_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    clamp: i32,
) -> Result<()> {
    let opts = lowering_options(ctx);
    if opts.backend == BackendTarget::Maca {
        return convert_maca_shuffle_i32(ctx, rewriter, op, intrinsic_name);
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle i32 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    let clamp_val = create_i32_const(ctx, rewriter, clamp);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, val, lane_or_delta, clamp_val],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert i32 shuffle for MXMACA using `bsm.bpermute(offset, value)`.
///
/// MXMACA shuffle is through BSM (shared memory), not register shuffle.
/// The offset is `lane * 4` (byte offset for i32).
fn convert_maca_shuffle_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic_name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle i32 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (_mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);
    let result = emit_maca_shuffle_i32(ctx, rewriter, op, val, lane_or_delta, intrinsic_name)?;
    rewriter.replace_operation_with_values(ctx, op, vec![result]);
    Ok(())
}

fn emit_maca_shuffle_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    val: pliron::value::Value,
    lane_or_delta: pliron::value::Value,
    intrinsic_name: &str,
) -> Result<pliron::value::Value> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let lane = crate::convert::intrinsics::basic::emit_maca_lane_id(ctx, rewriter, current_op)?;
    let wave_mask = create_i32_const(ctx, rewriter, 63);
    let wave_size = create_i32_const(ctx, rewriter, 64);
    let overflow_flags = IntegerOverflowFlagsAttr::default();

    let source_lane = if intrinsic_name.contains("idx") {
        let masked = llvm::AndOp::new(ctx, lane_or_delta, wave_mask);
        rewriter.insert_operation(ctx, masked.get_operation());
        masked.get_operation().deref(ctx).get_result(0)
    } else if intrinsic_name.contains("up") {
        let candidate =
            llvm::SubOp::new_with_overflow_flag(ctx, lane, lane_or_delta, overflow_flags.clone());
        rewriter.insert_operation(ctx, candidate.get_operation());
        let out_of_range = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::ULT, lane, lane_or_delta);
        rewriter.insert_operation(ctx, out_of_range.get_operation());
        let out_of_range_value = out_of_range.get_operation().deref(ctx).get_result(0);
        let candidate_value = candidate.get_operation().deref(ctx).get_result(0);
        let selected = llvm::SelectOp::new(ctx, out_of_range_value, lane, candidate_value);
        rewriter.insert_operation(ctx, selected.get_operation());
        selected.get_operation().deref(ctx).get_result(0)
    } else {
        let candidate = if intrinsic_name.contains("down") {
            let candidate = llvm::AddOp::new_with_overflow_flag(
                ctx,
                lane,
                lane_or_delta,
                overflow_flags.clone(),
            );
            rewriter.insert_operation(ctx, candidate.get_operation());
            candidate.get_operation().deref(ctx).get_result(0)
        } else if intrinsic_name.contains("bfly") {
            let candidate = llvm::XorOp::new(ctx, lane, lane_or_delta);
            rewriter.insert_operation(ctx, candidate.get_operation());
            candidate.get_operation().deref(ctx).get_result(0)
        } else {
            return pliron::input_err_noloc!(
                "unknown MACA shuffle mode for intrinsic `{}`",
                intrinsic_name
            );
        };
        let out_of_range = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::UGE, candidate, wave_size);
        rewriter.insert_operation(ctx, out_of_range.get_operation());
        let out_of_range_value = out_of_range.get_operation().deref(ctx).get_result(0);
        let selected = llvm::SelectOp::new(ctx, out_of_range_value, lane, candidate);
        rewriter.insert_operation(ctx, selected.get_operation());
        selected.get_operation().deref(ctx).get_result(0)
    };

    let two = create_i32_const(ctx, rewriter, 2);
    let byte_offset = llvm::ShlOp::new_with_overflow_flag(ctx, source_lane, two, overflow_flags);
    rewriter.insert_operation(ctx, byte_offset.get_operation());
    let byte_offset_value = byte_offset.get_operation().deref(ctx).get_result(0);
    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into()],
        false,
    );
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        current_op,
        "llvm_mxc_bsm_bpermute",
        func_ty,
        vec![byte_offset_value, val],
    )?;
    Ok(call_op.deref(ctx).get_result(0))
}

/// Convert f32 shuffle operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, value, lane_or_delta]`. See `convert_shuffle_i32`
/// for the mask forwarding rationale.
pub(crate) fn convert_shuffle_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    clamp: i32,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let f32_ty = FP32Type::get(ctx);

    if lowering_options(ctx).backend == BackendTarget::Maca {
        let operands: Vec<_> = op.deref(ctx).operands().collect();
        if operands.len() != 3 {
            return pliron::input_err_noloc!(
                "Warp shuffle f32 requires 3 operands [mask, value, lane_or_delta]"
            );
        }
        let value_bits = llvm::BitcastOp::new(ctx, operands[1], i32_ty.into());
        rewriter.insert_operation(ctx, value_bits.get_operation());
        let value_bits_value = value_bits.get_operation().deref(ctx).get_result(0);
        let shuffled_bits = emit_maca_shuffle_i32(
            ctx,
            rewriter,
            op,
            value_bits_value,
            operands[2],
            intrinsic_name,
        )?;
        let result = llvm::BitcastOp::new(ctx, shuffled_bits, f32_ty.into());
        rewriter.insert_operation(ctx, result.get_operation());
        rewriter.replace_operation(ctx, op, result.get_operation());
        return Ok(());
    }

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle f32 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    let clamp_val = create_i32_const(ctx, rewriter, clamp);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        f32_ty.into(),
        vec![i32_ty.into(), f32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, val, lane_or_delta, clamp_val],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a 64-bit shuffle op to convergent inline PTX.
///
/// PTX `shfl.sync` only moves 32-bit registers (no `.b64` form, no
/// `@llvm.nvvm.shfl.sync.*.i64` intrinsic), so a 64-bit shuffle is two 32-bit
/// shuffles. We emit a single inline-PTX block that unpacks the value into
/// `{lo, hi}` halves with `mov.b64`, runs `shfl.sync.<mode>.b32` on each half
/// with the shared lane and membermask operands, then repacks the result.
/// Keeping both halves inside one convergent asm block keeps the pair a single
/// fused warp collective, the same way the elect.sync lowering uses inline PTX
/// to dodge a missing intrinsic.
///
/// The shfl `c` (clamp/segmentation) operand is baked into the template per
/// mode: `31` for idx/bfly/down and `0` for up — exactly the value the 32-bit
/// intrinsic path passes (see [`convert_shuffle_i32`]).
///
/// Operand layout: `[mask, value, lane_or_delta]`. Inline-asm operand order is
/// `$0`=result, `$1`=value (i64, `l`), `$2`=lane/delta (i32, `r`),
/// `$3`=membermask (i32, `r`).
pub(crate) fn convert_shuffle_i64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    mode: &str,
    clamp: i32,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle i64 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    let asm_template = format!(
        "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; \
         shfl.sync.{mode}.b32 lo, lo, $2, {clamp}, $3; \
         shfl.sync.{mode}.b32 hi, hi, $2, {clamp}, $3; \
         mov.b64 $0, {{lo, hi}}; }}"
    );
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i64_ty.into(),
        vec![val, lane_or_delta, mask],
        &asm_template,
        "=l,l,r,r",
    )?;
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert vote operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, predicate]`. See `convert_shuffle_i32` for
/// the mask forwarding rationale.
pub(crate) fn convert_vote(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let opts = lowering_options(ctx);
    if opts.backend == BackendTarget::Maca {
        return convert_maca_vote(ctx, rewriter, op, intrinsic_name);
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("Warp vote requires 2 operands [mask, predicate]");
    }
    let (mask, predicate) = (operands[0], operands[1]);

    let result_ty: pliron::r#type::TypeHandle = if intrinsic_name.contains("ballot") {
        i32_ty.into()
    } else {
        i1_ty.into()
    };

    let func_ty =
        llvm_types::FuncType::get(ctx, result_ty, vec![i32_ty.into(), i1_ty.into()], false);
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, predicate],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert Wave64 vote operations using the integer predicate-mask intrinsic.
fn convert_maca_vote(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic_name: &str,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("Warp ballot requires 2 operands [mask, predicate]");
    }
    let (mask, predicate) = (operands[0], operands[1]);
    let predicate_i32 = llvm::ZExtOp::new_with_nneg(ctx, predicate, i32_ty.into(), false);
    rewriter.insert_operation(ctx, predicate_i32.get_operation());
    let predicate_i32_value = predicate_i32.get_operation().deref(ctx).get_result(0);
    let zero = create_i32_const(ctx, rewriter, 0);
    let compare_ne = create_i32_const(ctx, rewriter, 33);
    let func_ty = llvm_types::FuncType::get(
        ctx,
        i64_ty.into(),
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_mxc_icmp_i64_i32",
        func_ty,
        vec![predicate_i32_value, zero, compare_ne],
    )?;
    let predicate_mask = call_op.deref(ctx).get_result(0);
    let masked = llvm::AndOp::new(ctx, predicate_mask, mask);
    rewriter.insert_operation(ctx, masked.get_operation());
    let masked_value = masked.get_operation().deref(ctx).get_result(0);

    if intrinsic_name.contains("ballot") {
        rewriter.replace_operation(ctx, op, masked.get_operation());
    } else {
        let zero_mask = create_i64_const(ctx, rewriter, 0);
        let (predicate, rhs) = if intrinsic_name.contains("all") {
            (ICmpPredicateAttr::EQ, mask)
        } else if intrinsic_name.contains("any") {
            (ICmpPredicateAttr::NE, zero_mask)
        } else {
            return pliron::input_err_noloc!("unknown MACA vote intrinsic `{}`", intrinsic_name);
        };
        let result = llvm::ICmpOp::new(ctx, predicate, masked_value, rhs);
        rewriter.insert_operation(ctx, result.get_operation());
        rewriter.replace_operation(ctx, op, result.get_operation());
    }
    Ok(())
}

/// Convert a `match.any.sync` op to its LLVM intrinsic call.
///
/// Operand layout: `[mask, value]`. Result is i32 (bitmask of equal-value lanes).
/// The `value_ty` is i32 or i64 to pick `@llvm.nvvm.match.any.sync.{i32,i64}`.
pub(crate) fn convert_match_any(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    value_ty: pliron::r#type::TypeHandle,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("match.any.sync requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    let func_ty =
        llvm_types::FuncType::get(ctx, i32_ty.into(), vec![i32_ty.into(), value_ty], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, value],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a `redux.sync.add` op to its LLVM intrinsic call.
///
/// Op operand layout is `[mask, value]` (matching the other `*_sync`
/// collectives), but the LLVM intrinsic signature is `(src, membermask)`, so
/// we forward the operands flipped as `[value, mask]`. Result is i32.
pub(crate) fn convert_redux(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("redux requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into()],
        false,
    );

    // LLVM intrinsic wants (src, membermask): flip to [value, mask].
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![value, mask],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert an `activemask` op to its LLVM intrinsic call.
///
/// Lowers to `call i32 @llvm.nvvm.activemask()`. The op has no operands.
pub(crate) fn convert_active_mask(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    if lowering_options(ctx).backend == BackendTarget::Maca {
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let one = create_i32_const(ctx, rewriter, 1);
        let zero = create_i32_const(ctx, rewriter, 0);
        let compare_ne = create_i32_const(ctx, rewriter, 33);
        let func_ty = llvm_types::FuncType::get(
            ctx,
            i64_ty.into(),
            vec![i32_ty.into(), i32_ty.into(), i32_ty.into()],
            false,
        );
        let call = call_intrinsic(
            ctx,
            rewriter,
            op,
            "llvm_mxc_icmp_i64_i32",
            func_ty,
            vec![one, zero, compare_ne],
        )?;
        rewriter.replace_operation(ctx, op, call);
        return Ok(());
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let func_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), vec![], false);

    let call_op = call_intrinsic(ctx, rewriter, op, "llvm_nvvm_activemask", func_ty, vec![])?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a `bar.warp.sync` op to its LLVM intrinsic call.
///
/// Lowers to `call void @llvm.nvvm.bar.warp.sync(i32 mask)`. The op has one
/// operand (the participation mask) and no result.
pub(crate) fn convert_bar_warp_sync(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    if lowering_options(ctx).backend == BackendTarget::Maca {
        let operands: Vec<_> = op.deref(ctx).operands().collect();
        if operands.len() != 1 {
            return pliron::input_err_noloc!("bar.warp.sync requires 1 operand [mask]");
        }
        let release =
            llvm::FenceOp::new(ctx, LlvmAtomicOrdering::Release, Some("warp".to_string()));
        rewriter.insert_operation(ctx, release.get_operation());
        let void_ty = llvm_types::VoidType::get(ctx);
        let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);
        call_intrinsic(ctx, rewriter, op, "llvm_mxc_barrier_warp", func_ty, vec![])?;
        let acquire =
            llvm::FenceOp::new(ctx, LlvmAtomicOrdering::Acquire, Some("warp".to_string()));
        rewriter.insert_operation(ctx, acquire.get_operation());
        rewriter.erase_operation(ctx, op);
        return Ok(());
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("bar.warp.sync requires 1 operand [mask]");
    }
    let mask = operands[0];

    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![i32_ty.into()], false);
    call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_nvvm_bar_warp_sync",
        func_ty,
        vec![mask],
    )?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert a `match.all.sync` op to its LLVM intrinsic call.
///
/// The LLVM intrinsic signature is `{i32, i1} @llvm.nvvm.match.all.sync.*p(i32 mask, T value)`:
/// field 0 is the matching mask, field 1 is the all-match predicate. We expose
/// only the mask (callers can recover the predicate as `result != 0`); the
/// extracted i1 is dead and gets removed by LLVM DCE.
pub(crate) fn convert_match_all(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    value_ty: pliron::r#type::TypeHandle,
) -> Result<()> {
    use llvm_export::ops::ExtractValueOp;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("match.all.sync requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    let struct_ty = llvm_types::StructType::get_unnamed(ctx, vec![i32_ty.into(), i1_ty.into()]);
    let func_ty =
        llvm_types::FuncType::get(ctx, struct_ty.into(), vec![i32_ty.into(), value_ty], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, value],
    )?;
    let struct_result = call_op.deref(ctx).get_result(0);

    let extract_op = ExtractValueOp::new(ctx, struct_result, vec![0])
        .map_err(|e| pliron::input_error_noloc!("match.all.sync extractvalue: {}", e))?;
    rewriter.insert_operation(ctx, extract_op.get_operation());
    let mask_result = extract_op.get_operation().deref(ctx).get_result(0);

    rewriter.replace_operation_with_values(ctx, op, vec![mask_result]);
    Ok(())
}

/// Convert an `elect.sync` op to inline PTX.
///
/// `elect.sync` is Hopper-only (sm_90+). LLVM declares the
/// `@llvm.nvvm.elect.sync` intrinsic but the NVPTX backend ships **no
/// instruction-selection pattern** for it (llc dies with "Cannot select:
/// intrinsic %llvm.nvvm.elect.sync" even with `-mcpu=sm_90`), so we emit the
/// instruction directly as convergent inline PTX instead — the same approach
/// the tcgen05 ops use to dodge missing intrinsic lowerings.
///
/// PTX `elect.sync d|p, membermask;` writes the leader lane id into `d` and the
/// per-lane "I am the leader" predicate into `p`. Inline asm can't yield a
/// `.pred` directly, so we `selp.b32` it into a 0/1 register and truncate to i1.
/// The op has two results — leader (i32) and is_elected (i1) — bound to the two
/// asm outputs. The single operand (the membermask) is the asm input; either
/// result may be unused at the call site and is then removed by LLVM DCE.
pub(crate) fn convert_elect_sync(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use llvm_export::ops::ExtractValueOp;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("elect.sync requires 1 operand [mask]");
    }
    let mask = operands[0];

    // Two register outputs: $0 = leader lane id, $1 = predicate materialized as
    // 0/1; $2 = membermask input. The `.pred p` is scoped to the asm block.
    let asm_template = "{ .reg .pred p; elect.sync $0|p, $2; selp.b32 $1, 1, 0, p; }";
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, vec![i32_ty.into(), i32_ty.into()]);
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        struct_ty.into(),
        vec![mask],
        asm_template,
        "=r,=r,r",
    )?;
    let struct_result = asm_op.deref(ctx).get_result(0);

    // Field 0 → leader lane id (result 0). Field 1 → predicate as 0/1 i32,
    // truncated to the i1 is_elected result (result 1).
    let leader = {
        let extract_op = ExtractValueOp::new(ctx, struct_result, vec![0])
            .map_err(|e| pliron::input_error_noloc!("elect.sync extractvalue: {}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        extract_op.get_operation().deref(ctx).get_result(0)
    };
    let elected_i32 = {
        let extract_op = ExtractValueOp::new(ctx, struct_result, vec![1])
            .map_err(|e| pliron::input_error_noloc!("elect.sync extractvalue: {}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        extract_op.get_operation().deref(ctx).get_result(0)
    };
    let is_elected = trunc_to_i1(ctx, rewriter, elected_i32);

    rewriter.replace_operation_with_values(ctx, op, vec![leader, is_elected]);
    Ok(())
}
