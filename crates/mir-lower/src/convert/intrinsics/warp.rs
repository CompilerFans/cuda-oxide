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
use llvm_export::attributes::{
    FCmpPredicateAttr, ICmpPredicateAttr, IntegerOverflowFlagsAttr, LlvmAtomicOrdering,
    SyncScopeAttr,
};
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
use pliron::r#type::Typed;

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
/// Keeping both halves in one compiler-visible convergent block prevents code
/// motion between them. Hardware still executes two sequential `b32` shuffles.
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
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle i64 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    if lowering_options(ctx).backend == BackendTarget::Maca {
        let low = llvm::TruncOp::new(ctx, val, i32_ty.into());
        rewriter.insert_operation(ctx, low.get_operation());
        let low = low.get_operation().deref(ctx).get_result(0);
        let shift_32 = create_i64_const(ctx, rewriter, 32);
        let high_shifted = llvm::LShrOp::new(ctx, val, shift_32);
        rewriter.insert_operation(ctx, high_shifted.get_operation());
        let high_shifted = high_shifted.get_operation().deref(ctx).get_result(0);
        let high = llvm::TruncOp::new(ctx, high_shifted, i32_ty.into());
        rewriter.insert_operation(ctx, high.get_operation());
        let high = high.get_operation().deref(ctx).get_result(0);

        let intrinsic_name = match mode {
            "idx" => "llvm_nvvm_shfl_sync_idx_i64",
            "bfly" => "llvm_nvvm_shfl_sync_bfly_i64",
            "down" => "llvm_nvvm_shfl_sync_down_i64",
            "up" => "llvm_nvvm_shfl_sync_up_i64",
            _ => return pliron::input_err_noloc!("unknown MACA i64 shuffle mode `{}`", mode),
        };
        let low = emit_maca_shuffle_i32(ctx, rewriter, op, low, lane_or_delta, intrinsic_name)?;
        let high = emit_maca_shuffle_i32(ctx, rewriter, op, high, lane_or_delta, intrinsic_name)?;
        let low = llvm::ZExtOp::new_with_nneg(ctx, low, i64_ty.into(), false);
        rewriter.insert_operation(ctx, low.get_operation());
        let low = low.get_operation().deref(ctx).get_result(0);
        let high = llvm::ZExtOp::new_with_nneg(ctx, high, i64_ty.into(), false);
        rewriter.insert_operation(ctx, high.get_operation());
        let high = high.get_operation().deref(ctx).get_result(0);
        let high = llvm::ShlOp::new_with_overflow_flag(
            ctx,
            high,
            shift_32,
            IntegerOverflowFlagsAttr::default(),
        );
        rewriter.insert_operation(ctx, high.get_operation());
        let high = high.get_operation().deref(ctx).get_result(0);
        let combined = llvm::OrOp::new(ctx, low, high);
        rewriter.insert_operation(ctx, combined.get_operation());
        rewriter.replace_operation(ctx, op, combined.get_operation());
        return Ok(());
    }

    let asm_template = format!(
        "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; \
         shfl.sync.{mode}.b32 lo, lo, $2, {clamp}, $3; \
         shfl.sync.{mode}.b32 hi, hi, $2, {clamp}, $3; \
         mov.b64 $0, {{lo, hi}}; }}"
    );
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        i64_ty.into(),
        vec![val, lane_or_delta, mask],
        &asm_template,
        "=l,l,r,r",
    );
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
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("Warp ballot requires 2 operands [mask, predicate]");
    }
    let (mask, predicate) = (operands[0], operands[1]);
    let predicate_mask = emit_maca_predicate_mask(ctx, rewriter, op, predicate)?;
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

fn emit_maca_predicate_mask(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    predicate: pliron::value::Value,
) -> Result<pliron::value::Value> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
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
        current_op,
        "llvm_mxc_icmp_i64_i32",
        func_ty,
        vec![predicate_i32_value, zero, compare_ne],
    )?;
    Ok(call_op.deref(ctx).get_result(0))
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
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("match.any.sync requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    if lowering_options(ctx).backend == BackendTarget::Maca {
        let result = emit_maca_match_any(ctx, rewriter, op, mask, value, intrinsic_name)?;
        rewriter.replace_operation_with_values(ctx, op, vec![result]);
        return Ok(());
    }

    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let mask_i32 = llvm::TruncOp::new(ctx, mask, i32_ty.into());
    rewriter.insert_operation(ctx, mask_i32.get_operation());
    let mask_i32 = mask_i32.get_operation().deref(ctx).get_result(0);

    let func_ty =
        llvm_types::FuncType::get(ctx, i32_ty.into(), vec![i32_ty.into(), value_ty], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask_i32, value],
    )?;
    let call_result = call_op.deref(ctx).get_result(0);
    let result = llvm::ZExtOp::new_with_nneg(ctx, call_result, i64_ty.into(), false);
    rewriter.insert_operation(ctx, result.get_operation());
    rewriter.replace_operation(ctx, op, result.get_operation());
    Ok(())
}

fn emit_maca_match_any(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    mask: pliron::value::Value,
    value: pliron::value::Value,
    intrinsic_name: &str,
) -> Result<pliron::value::Value> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let bit_count = if intrinsic_name.contains("i64") {
        64
    } else {
        32
    };
    let mut matched = mask;
    for bit in 0..bit_count {
        let bit_value = if bit_count == 64 {
            let shift = create_i64_const(ctx, rewriter, bit);
            let shifted = llvm::LShrOp::new(ctx, value, shift);
            rewriter.insert_operation(ctx, shifted.get_operation());
            let shifted = shifted.get_operation().deref(ctx).get_result(0);
            let low = llvm::TruncOp::new(ctx, shifted, i32_ty.into());
            rewriter.insert_operation(ctx, low.get_operation());
            low.get_operation().deref(ctx).get_result(0)
        } else {
            let shift = create_i32_const(ctx, rewriter, bit as i32);
            let shifted = llvm::LShrOp::new(ctx, value, shift);
            rewriter.insert_operation(ctx, shifted.get_operation());
            shifted.get_operation().deref(ctx).get_result(0)
        };
        let one = create_i32_const(ctx, rewriter, 1);
        let bit_value = llvm::AndOp::new(ctx, bit_value, one);
        rewriter.insert_operation(ctx, bit_value.get_operation());
        let bit_value = bit_value.get_operation().deref(ctx).get_result(0);
        let zero = create_i32_const(ctx, rewriter, 0);
        let own_bit = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, bit_value, zero);
        rewriter.insert_operation(ctx, own_bit.get_operation());
        let own_bit = own_bit.get_operation().deref(ctx).get_result(0);
        let lanes_with_bit = emit_maca_predicate_mask(ctx, rewriter, current_op, own_bit)?;
        let all_ones = create_i64_const(ctx, rewriter, -1);
        let complement = llvm::XorOp::new(ctx, lanes_with_bit, all_ones);
        rewriter.insert_operation(ctx, complement.get_operation());
        let complement = complement.get_operation().deref(ctx).get_result(0);
        let same_bit = llvm::SelectOp::new(ctx, own_bit, lanes_with_bit, complement);
        rewriter.insert_operation(ctx, same_bit.get_operation());
        let same_bit = same_bit.get_operation().deref(ctx).get_result(0);
        let next = llvm::AndOp::new(ctx, matched, same_bit);
        rewriter.insert_operation(ctx, next.get_operation());
        matched = next.get_operation().deref(ctx).get_result(0);
    }
    Ok(matched)
}

/// Convert a `redux.sync.add` op to its LLVM intrinsic call.
///
/// Op operand layout is `[mask, value]` (matching the other `*_sync`
/// collectives), but the LLVM intrinsic signature is `(src, membermask)`, so
/// we forward the operands flipped as `[value, mask]`. The value and result
/// types are carried by the dialect record.
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

    if lowering_options(ctx).backend == BackendTarget::Maca {
        let value_ty_ref = value.get_type(ctx);
        let is_f32 = value_ty_ref
            .deref(ctx)
            .is::<FP32Type>();
        if is_f32 {
            let result =
                emit_maca_redux_f32(ctx, rewriter, op, mask, value, intrinsic_name)?;
            rewriter.replace_operation_with_values(ctx, op, vec![result]);
            return Ok(());
        }
        let result = emit_maca_redux(ctx, rewriter, op, mask, value, intrinsic_name)?;
        rewriter.replace_operation_with_values(ctx, op, vec![result]);
        return Ok(());
    }

    // Mask may already be i32 (the f32 redux device fns historically take
    // `u32`); only widen-from-i64 needs a trunc.
    let mask_is_i64 = mask
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|i| i.width() == 64);
    let mask_i32 = if mask_is_i64 {
        let truncated = llvm::TruncOp::new(ctx, mask, i32_ty.into());
        rewriter.insert_operation(ctx, truncated.get_operation());
        truncated.get_operation().deref(ctx).get_result(0)
    } else {
        mask
    };

    let value_ty = value.get_type(ctx);
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let func_ty = llvm_types::FuncType::get(ctx, result_ty, vec![value_ty, i32_ty.into()], false);

    // LLVM intrinsic wants (src, membermask): flip to [value, mask].
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![value, mask_i32],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

fn emit_maca_redux(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    mask: pliron::value::Value,
    value: pliron::value::Value,
    intrinsic_name: &str,
) -> Result<pliron::value::Value> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let lane = crate::convert::intrinsics::basic::emit_maca_lane_id(ctx, rewriter, current_op)?;
    let original = value;
    let mut result = value;
    let overflow_flags = IntegerOverflowFlagsAttr::default();

    for (log_stride, stride) in [1i32, 2, 4, 8, 16, 32].into_iter().enumerate() {
        let stride_value = create_i32_const(ctx, rewriter, stride);
        let candidate = llvm::XorOp::new(ctx, lane, stride_value);
        rewriter.insert_operation(ctx, candidate.get_operation());
        let candidate = candidate.get_operation().deref(ctx).get_result(0);

        let group = if log_stride == 0 {
            candidate
        } else {
            let shift = create_i32_const(ctx, rewriter, log_stride as i32);
            let group = llvm::LShrOp::new(ctx, candidate, shift);
            rewriter.insert_operation(ctx, group.get_operation());
            group.get_operation().deref(ctx).get_result(0)
        };
        let group_base = if log_stride == 0 {
            group
        } else {
            let shift = create_i32_const(ctx, rewriter, log_stride as i32);
            let base =
                llvm::ShlOp::new_with_overflow_flag(ctx, group, shift, overflow_flags.clone());
            rewriter.insert_operation(ctx, base.get_operation());
            base.get_operation().deref(ctx).get_result(0)
        };
        let group_base = llvm::ZExtOp::new_with_nneg(ctx, group_base, i64_ty.into(), false);
        rewriter.insert_operation(ctx, group_base.get_operation());
        let group_base = group_base.get_operation().deref(ctx).get_result(0);
        let half = create_i64_const(ctx, rewriter, (1i64 << stride) - 1);
        let half =
            llvm::ShlOp::new_with_overflow_flag(ctx, half, group_base, overflow_flags.clone());
        rewriter.insert_operation(ctx, half.get_operation());
        let half = half.get_operation().deref(ctx).get_result(0);
        let half = llvm::AndOp::new(ctx, half, mask);
        rewriter.insert_operation(ctx, half.get_operation());
        let half = half.get_operation().deref(ctx).get_result(0);

        let candidate_i64 = llvm::ZExtOp::new_with_nneg(ctx, candidate, i64_ty.into(), false);
        rewriter.insert_operation(ctx, candidate_i64.get_operation());
        let candidate_i64 = candidate_i64.get_operation().deref(ctx).get_result(0);
        let one = create_i64_const(ctx, rewriter, 1);
        let candidate_bit =
            llvm::ShlOp::new_with_overflow_flag(ctx, one, candidate_i64, overflow_flags.clone());
        rewriter.insert_operation(ctx, candidate_bit.get_operation());
        let candidate_bit = candidate_bit.get_operation().deref(ctx).get_result(0);
        let candidate_member = llvm::AndOp::new(ctx, mask, candidate_bit);
        rewriter.insert_operation(ctx, candidate_member.get_operation());
        let candidate_member = candidate_member.get_operation().deref(ctx).get_result(0);
        let zero_i64 = create_i64_const(ctx, rewriter, 0);
        let candidate_member =
            llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, candidate_member, zero_i64);
        rewriter.insert_operation(ctx, candidate_member.get_operation());
        let candidate_member = candidate_member.get_operation().deref(ctx).get_result(0);

        let zero_undef = create_i1_const(ctx, rewriter, false);
        let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let cttz_ty =
            llvm_types::FuncType::get(ctx, i64_ty.into(), vec![i64_ty.into(), i1_ty.into()], false);
        let fallback = call_intrinsic(
            ctx,
            rewriter,
            current_op,
            "llvm_cttz_i64",
            cttz_ty,
            vec![half, zero_undef],
        )?;
        let fallback = fallback.deref(ctx).get_result(0);
        let fallback = llvm::TruncOp::new(ctx, fallback, i32_ty.into());
        rewriter.insert_operation(ctx, fallback.get_operation());
        let fallback = fallback.get_operation().deref(ctx).get_result(0);
        let source = llvm::SelectOp::new(ctx, candidate_member, candidate, fallback);
        rewriter.insert_operation(ctx, source.get_operation());
        let source = source.get_operation().deref(ctx).get_result(0);
        let shuffled = emit_maca_shuffle_i32(
            ctx,
            rewriter,
            current_op,
            result,
            source,
            "llvm_nvvm_shfl_sync_idx_i32",
        )?;

        let source_i64 = llvm::ZExtOp::new_with_nneg(ctx, source, i64_ty.into(), false);
        rewriter.insert_operation(ctx, source_i64.get_operation());
        let source_i64 = source_i64.get_operation().deref(ctx).get_result(0);
        let one = create_i64_const(ctx, rewriter, 1);
        let source_bit =
            llvm::ShlOp::new_with_overflow_flag(ctx, one, source_i64, overflow_flags.clone());
        rewriter.insert_operation(ctx, source_bit.get_operation());
        let source_bit = source_bit.get_operation().deref(ctx).get_result(0);
        let source_member = llvm::AndOp::new(ctx, mask, source_bit);
        rewriter.insert_operation(ctx, source_member.get_operation());
        let source_member = source_member.get_operation().deref(ctx).get_result(0);
        let zero_i64 = create_i64_const(ctx, rewriter, 0);
        let source_member = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, source_member, zero_i64);
        rewriter.insert_operation(ctx, source_member.get_operation());
        let source_member = source_member.get_operation().deref(ctx).get_result(0);

        let identity = if intrinsic_name.contains("and") {
            create_i32_const(ctx, rewriter, -1)
        } else if intrinsic_name.contains("umin") {
            create_i32_const(ctx, rewriter, -1)
        } else if intrinsic_name.contains("min") {
            create_i32_const(ctx, rewriter, i32::MAX)
        } else if intrinsic_name.contains("max") && !intrinsic_name.contains("umax") {
            create_i32_const(ctx, rewriter, i32::MIN)
        } else {
            create_i32_const(ctx, rewriter, 0)
        };
        let rhs = llvm::SelectOp::new(ctx, source_member, shuffled, identity);
        rewriter.insert_operation(ctx, rhs.get_operation());
        let rhs = rhs.get_operation().deref(ctx).get_result(0);
        result = if intrinsic_name.contains("add") {
            let next =
                llvm::AddOp::new_with_overflow_flag(ctx, result, rhs, overflow_flags.clone());
            rewriter.insert_operation(ctx, next.get_operation());
            next.get_operation().deref(ctx).get_result(0)
        } else if intrinsic_name.contains("and") {
            let next = llvm::AndOp::new(ctx, result, rhs);
            rewriter.insert_operation(ctx, next.get_operation());
            next.get_operation().deref(ctx).get_result(0)
        } else if intrinsic_name.contains("xor") {
            let next = llvm::XorOp::new(ctx, result, rhs);
            rewriter.insert_operation(ctx, next.get_operation());
            next.get_operation().deref(ctx).get_result(0)
        } else if intrinsic_name.contains("or") {
            let next = llvm::OrOp::new(ctx, result, rhs);
            rewriter.insert_operation(ctx, next.get_operation());
            next.get_operation().deref(ctx).get_result(0)
        } else {
            let predicate = if intrinsic_name.contains("umin") {
                ICmpPredicateAttr::ULT
            } else if intrinsic_name.contains("umax") {
                ICmpPredicateAttr::UGT
            } else if intrinsic_name.contains("min") {
                ICmpPredicateAttr::SLT
            } else if intrinsic_name.contains("max") {
                ICmpPredicateAttr::SGT
            } else {
                return pliron::input_err_noloc!(
                    "unknown MACA redux intrinsic `{}`",
                    intrinsic_name
                );
            };
            let choose_rhs = llvm::ICmpOp::new(ctx, predicate, rhs, result);
            rewriter.insert_operation(ctx, choose_rhs.get_operation());
            let choose_rhs = choose_rhs.get_operation().deref(ctx).get_result(0);
            let next = llvm::SelectOp::new(ctx, choose_rhs, rhs, result);
            rewriter.insert_operation(ctx, next.get_operation());
            next.get_operation().deref(ctx).get_result(0)
        };
    }

    let lane_i64 = llvm::ZExtOp::new_with_nneg(ctx, lane, i64_ty.into(), false);
    rewriter.insert_operation(ctx, lane_i64.get_operation());
    let lane_i64 = lane_i64.get_operation().deref(ctx).get_result(0);
    let one = create_i64_const(ctx, rewriter, 1);
    let lane_bit = llvm::ShlOp::new_with_overflow_flag(ctx, one, lane_i64, overflow_flags);
    rewriter.insert_operation(ctx, lane_bit.get_operation());
    let lane_bit = lane_bit.get_operation().deref(ctx).get_result(0);
    let lane_member = llvm::AndOp::new(ctx, mask, lane_bit);
    rewriter.insert_operation(ctx, lane_member.get_operation());
    let lane_member = lane_member.get_operation().deref(ctx).get_result(0);
    let zero_i64 = create_i64_const(ctx, rewriter, 0);
    let lane_member = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, lane_member, zero_i64);
    rewriter.insert_operation(ctx, lane_member.get_operation());
    let lane_member = lane_member.get_operation().deref(ctx).get_result(0);
    let final_value = llvm::SelectOp::new(ctx, lane_member, result, original);
    rewriter.insert_operation(ctx, final_value.get_operation());
    Ok(final_value.get_operation().deref(ctx).get_result(0))
}

/// Wave-wide f32 redux for MXMACA: the i32 bpermute loop operates on the
/// bit pattern, with float comparisons (or fadd) after each round.
fn emit_maca_redux_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    mask: pliron::value::Value,
    value: pliron::value::Value,
    intrinsic_name: &str,
) -> Result<pliron::value::Value> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(ctx);
    let lane = crate::convert::intrinsics::basic::emit_maca_lane_id(ctx, rewriter, current_op)?;
    let overflow_flags = IntegerOverflowFlagsAttr::default();

    // Work on the i32 bit pattern of the f32 value.
    let bits = llvm::BitcastOp::new(ctx, value, i32_ty.into());
    rewriter.insert_operation(ctx, bits.get_operation());
    let mut result = bits.get_operation().deref(ctx).get_result(0);
    let original_bits = result;

    // Normalize the mask to i64: the f32 redux device fns historically take
    // `u32` masks, but the MACA member checks run in i64.
    let mask = {
        let mask_ty = mask.get_type(ctx);
        let is_i32 = mask_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|i| i.width() == 32);
        if is_i32 {
            let widened = llvm::ZExtOp::new_with_nneg(ctx, mask, i64_ty.into(), false);
            rewriter.insert_operation(ctx, widened.get_operation());
            widened.get_operation().deref(ctx).get_result(0)
        } else {
            mask
        }
    };

    let is_add = intrinsic_name.contains("add");
    // fmin/fmax comparisons happen on f32; NaN variants ignore NaN inputs
    // (LLVM fcmp ogt/olt already do), abs variants compare |x|.
    let strip_abs = intrinsic_name.contains("_abs");
    let want_max = intrinsic_name.contains("fmax");

    for stride in [1i32, 2, 4, 8, 16, 32] {
        let stride_value = create_i32_const(ctx, rewriter, stride);
        let candidate = llvm::XorOp::new(ctx, lane, stride_value);
        rewriter.insert_operation(ctx, candidate.get_operation());
        let candidate = candidate.get_operation().deref(ctx).get_result(0);
        let candidate_i64 = llvm::ZExtOp::new_with_nneg(ctx, candidate, i64_ty.into(), false);
        rewriter.insert_operation(ctx, candidate_i64.get_operation());
        let candidate_i64 = candidate_i64.get_operation().deref(ctx).get_result(0);
        let one = create_i64_const(ctx, rewriter, 1);
        let member_bit = llvm::ShlOp::new_with_overflow_flag(
            ctx,
            one,
            candidate_i64,
            overflow_flags.clone(),
        );
        rewriter.insert_operation(ctx, member_bit.get_operation());
        let member_bit = member_bit.get_operation().deref(ctx).get_result(0);
        let member = llvm::AndOp::new(ctx, mask, member_bit);
        rewriter.insert_operation(ctx, member.get_operation());
        let member = member.get_operation().deref(ctx).get_result(0);
        let zero_i64 = create_i64_const(ctx, rewriter, 0);
        let member = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, member, zero_i64);
        rewriter.insert_operation(ctx, member.get_operation());
        let member = member.get_operation().deref(ctx).get_result(0);

        // Broadcast the candidate lane's bits via the i32 bpermute:
        // offset = lane * 4 bytes.
        let two = create_i32_const(ctx, rewriter, 2);
        let byte_offset =
            llvm::ShlOp::new_with_overflow_flag(ctx, candidate, two, overflow_flags.clone());
        rewriter.insert_operation(ctx, byte_offset.get_operation());
        let byte_offset = byte_offset.get_operation().deref(ctx).get_result(0);
        let bpermute_ty =
            llvm_types::FuncType::get(ctx, i32_ty.into(), vec![i32_ty.into(), i32_ty.into()], false);
        let bpermute = crate::convert::intrinsics::common::call_intrinsic(
            ctx,
            rewriter,
            current_op,
            "llvm_mxc_bsm_bpermute",
            bpermute_ty,
            vec![byte_offset, result],
        )?;
        let shuffled = bpermute.deref(ctx).get_result(0);

        if is_add {
            let zero_bits = create_i32_const(ctx, rewriter, 0);
            let rhs = llvm::SelectOp::new(ctx, member, shuffled, zero_bits);
            rewriter.insert_operation(ctx, rhs.get_operation());
            let rhs = rhs.get_operation().deref(ctx).get_result(0);
            let lhs_f = llvm::BitcastOp::new(ctx, result, f32_ty.into());
            rewriter.insert_operation(ctx, lhs_f.get_operation());
            let lhs_f = lhs_f.get_operation().deref(ctx).get_result(0);
            let rhs_f = llvm::BitcastOp::new(ctx, rhs, f32_ty.into());
            rewriter.insert_operation(ctx, rhs_f.get_operation());
            let rhs_f = rhs_f.get_operation().deref(ctx).get_result(0);
            let sum = llvm::FAddOp::new(ctx, lhs_f, rhs_f);
            rewriter.insert_operation(ctx, sum.get_operation());
            let sum = sum.get_operation().deref(ctx).get_result(0);
            let bits = llvm::BitcastOp::new(ctx, sum, i32_ty.into());
            rewriter.insert_operation(ctx, bits.get_operation());
            result = bits.get_operation().deref(ctx).get_result(0);
        } else {
            let lhs_f = llvm::BitcastOp::new(ctx, result, f32_ty.into());
            rewriter.insert_operation(ctx, lhs_f.get_operation());
            let lhs_f = lhs_f.get_operation().deref(ctx).get_result(0);
            let rhs_f0 = llvm::BitcastOp::new(ctx, shuffled, f32_ty.into());
            rewriter.insert_operation(ctx, rhs_f0.get_operation());
            let rhs_f0 = rhs_f0.get_operation().deref(ctx).get_result(0);
            let rhs_f = if strip_abs {
                // fabs via sign-mask clear on the incoming lane value
                let rhs_bits = llvm::BitcastOp::new(ctx, rhs_f0, i32_ty.into());
                rewriter.insert_operation(ctx, rhs_bits.get_operation());
                let rhs_bits = rhs_bits.get_operation().deref(ctx).get_result(0);
                let sign_mask = create_i32_const(ctx, rewriter, 0x7fff_ffff);
                let abs_bits = llvm::AndOp::new(ctx, rhs_bits, sign_mask);
                rewriter.insert_operation(ctx, abs_bits.get_operation());
                let abs_bits = abs_bits.get_operation().deref(ctx).get_result(0);
                let abs_f = llvm::BitcastOp::new(ctx, abs_bits, f32_ty.into());
                rewriter.insert_operation(ctx, abs_f.get_operation());
                abs_f.get_operation().deref(ctx).get_result(0)
            } else {
                rhs_f0
            };
            let pred = if want_max {
                FCmpPredicateAttr::OGT
            } else {
                FCmpPredicateAttr::OLT
            };
            let cmp = llvm::FCmpOp::new(ctx, pred, rhs_f, lhs_f);
            rewriter.insert_operation(ctx, cmp.get_operation());
            let cmp_v = cmp.get_operation().deref(ctx).get_result(0);
            let lhs_bits = llvm::BitcastOp::new(ctx, lhs_f, i32_ty.into());
            rewriter.insert_operation(ctx, lhs_bits.get_operation());
            let lhs_bits = lhs_bits.get_operation().deref(ctx).get_result(0);
            let rhs_bits = llvm::BitcastOp::new(ctx, rhs_f, i32_ty.into());
            rewriter.insert_operation(ctx, rhs_bits.get_operation());
            let rhs_bits = rhs_bits.get_operation().deref(ctx).get_result(0);
            let chosen = llvm::SelectOp::new(ctx, cmp_v, rhs_bits, lhs_bits);
            rewriter.insert_operation(ctx, chosen.get_operation());
            let chosen = chosen.get_operation().deref(ctx).get_result(0);
            let chosen = llvm::SelectOp::new(ctx, member, chosen, result);
            rewriter.insert_operation(ctx, chosen.get_operation());
            result = chosen.get_operation().deref(ctx).get_result(0);
        }
    }

    // Keep the calling lane's own value when it is not a mask member.
    let lane_i64 = llvm::ZExtOp::new_with_nneg(ctx, lane, i64_ty.into(), false);
    rewriter.insert_operation(ctx, lane_i64.get_operation());
    let lane_i64 = lane_i64.get_operation().deref(ctx).get_result(0);
    let one = create_i64_const(ctx, rewriter, 1);
    let lane_bit = llvm::ShlOp::new_with_overflow_flag(ctx, one, lane_i64, overflow_flags);
    rewriter.insert_operation(ctx, lane_bit.get_operation());
    let lane_bit = lane_bit.get_operation().deref(ctx).get_result(0);
    let lane_member = llvm::AndOp::new(ctx, mask, lane_bit);
    rewriter.insert_operation(ctx, lane_member.get_operation());
    let lane_member = lane_member.get_operation().deref(ctx).get_result(0);
    let zero_i64 = create_i64_const(ctx, rewriter, 0);
    let lane_member = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, lane_member, zero_i64);
    rewriter.insert_operation(ctx, lane_member.get_operation());
    let lane_member = lane_member.get_operation().deref(ctx).get_result(0);
    let final_bits = llvm::SelectOp::new(ctx, lane_member, result, original_bits);
    rewriter.insert_operation(ctx, final_bits.get_operation());
    let final_bits = final_bits.get_operation().deref(ctx).get_result(0);
    let final_f = llvm::BitcastOp::new(ctx, final_bits, f32_ty.into());
    rewriter.insert_operation(ctx, final_f.get_operation());
    Ok(final_f.get_operation().deref(ctx).get_result(0))
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
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    // libNVVM has no activemask intrinsic; the generated-catalog backend
    // route uses exact convergent inline PTX there. The op result is a
    // 64-bit wave mask, so both paths zero-extend the i32 hardware result.
    let mask_i32 = if lowering_options(ctx).intrinsic_backend
        == crate::IntrinsicBackend::LibNvvm
    {
        inline_asm_convergent(
            ctx,
            rewriter,
            op,
            i32_ty.into(),
            vec![],
            "activemask.b32 $0;",
            "=r,~{memory}",
        )
        .deref(ctx)
        .get_result(0)
    } else {
        let func_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), vec![], false);
        let call_op = call_intrinsic(ctx, rewriter, op, "llvm_nvvm_activemask", func_ty, vec![])?;
        call_op.deref(ctx).get_result(0)
    };
    let extended = llvm::ZExtOp::new_with_nneg(ctx, mask_i32, i64_ty.into(), true);
    rewriter.insert_operation(ctx, extended.get_operation());
    rewriter.replace_operation(ctx, op, extended.get_operation());
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
            llvm::FenceOp::new(
                ctx,
                LlvmAtomicOrdering::Release,
                SyncScopeAttr::NamedScope(pliron::builtin::attributes::StringAttr::new("warp".to_string())),
            );
        rewriter.insert_operation(ctx, release.get_operation());
        let void_ty = llvm_types::VoidType::get(ctx);
        let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);
        call_intrinsic(ctx, rewriter, op, "llvm_mxc_barrier_warp", func_ty, vec![])?;
        let acquire =
            llvm::FenceOp::new(
                ctx,
                LlvmAtomicOrdering::Acquire,
                SyncScopeAttr::NamedScope(pliron::builtin::attributes::StringAttr::new("warp".to_string())),
            );
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

    if lowering_options(ctx).backend == BackendTarget::Maca {
        let matched = emit_maca_match_any(ctx, rewriter, op, mask, value, intrinsic_name)?;
        let all = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::EQ, matched, mask);
        rewriter.insert_operation(ctx, all.get_operation());
        let zero = create_i64_const(ctx, rewriter, 0);
        let all = all.get_operation().deref(ctx).get_result(0);
        let result = llvm::SelectOp::new(ctx, all, mask, zero);
        rewriter.insert_operation(ctx, result.get_operation());
        rewriter.replace_operation(ctx, op, result.get_operation());
        return Ok(());
    }

    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let mask_i32 = llvm::TruncOp::new(ctx, mask, i32_ty.into());
    rewriter.insert_operation(ctx, mask_i32.get_operation());
    let mask_i32 = mask_i32.get_operation().deref(ctx).get_result(0);

    let struct_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![i32_ty.into(), i1_ty.into()],
            llvm_types::StructLayout::Unpacked,
        ),
    );
    let func_ty =
        llvm_types::FuncType::get(ctx, struct_ty.into(), vec![i32_ty.into(), value_ty], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask_i32, value],
    )?;
    let struct_result = call_op.deref(ctx).get_result(0);

    let extract_op = ExtractValueOp::new(ctx, struct_result, vec![0])
        .map_err(|e| pliron::input_error_noloc!("match.all.sync extractvalue: {}", e))?;
    rewriter.insert_operation(ctx, extract_op.get_operation());
    let mask_result = extract_op.get_operation().deref(ctx).get_result(0);

    let mask_result = llvm::ZExtOp::new_with_nneg(ctx, mask_result, i64_ty.into(), false);
    rewriter.insert_operation(ctx, mask_result.get_operation());
    rewriter.replace_operation(ctx, op, mask_result.get_operation());
    Ok(())
}

/// Convert `elect.sync` through LLVM's typed NVVM intrinsic.
pub(crate) fn convert_elect_sync_typed(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    use llvm_export::ops::ExtractValueOp;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("elect.sync requires 1 operand [mask]");
    }

    let struct_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![i32_ty.into(), i1_ty.into()],
            llvm_types::StructLayout::Unpacked,
        ),
    );
    let func_ty = llvm_types::FuncType::get(ctx, struct_ty.into(), vec![i32_ty.into()], false);
    let call = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![operands[0]],
    )?;
    let result = call.deref(ctx).get_result(0);
    let leader = ExtractValueOp::new(ctx, result, vec![0])
        .map_err(|error| pliron::input_error_noloc!("elect.sync extractvalue: {}", error))?;
    rewriter.insert_operation(ctx, leader.get_operation());
    let elected = ExtractValueOp::new(ctx, result, vec![1])
        .map_err(|error| pliron::input_error_noloc!("elect.sync extractvalue: {}", error))?;
    rewriter.insert_operation(ctx, elected.get_operation());
    let leader = leader.get_operation().deref(ctx).get_result(0);
    let elected = elected.get_operation().deref(ctx).get_result(0);
    rewriter.replace_operation_with_values(ctx, op, vec![leader, elected]);
    Ok(())
}

/// Convert `elect.sync` to convergent inline PTX.
///
/// PTX `elect.sync d|p, membermask;` writes the leader lane id into `d` and the
/// per-lane "I am the leader" predicate into `p`. Inline asm can't yield a
/// `.pred` directly, so we `selp.b32` it into a 0/1 register and truncate to i1.
/// The op has two results — leader (i32) and is_elected (i1) — bound to the two
/// asm outputs. The single operand (the membermask) is the asm input; either
/// result may be unused at the call site and is then removed by LLVM DCE.
pub(crate) fn convert_elect_sync_inline(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let opts = lowering_options(ctx);
    if opts.backend == crate::BackendTarget::Maca {
        return convert_elect_sync_maca(ctx, rewriter, op);
    }

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
    let struct_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![i32_ty.into(), i32_ty.into()],
            llvm_types::StructLayout::Unpacked,
        ),
    );
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        struct_ty.into(),
        vec![mask],
        asm_template,
        "=r,=r,r",
    );
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

/// Convert `elect.sync` for MXMACA using find-first-set on the mask.
///
/// MXMACA has no native `elect.sync` instruction, so we compute the
/// elected leader as the lowest-numbered lane in the participant mask:
/// `leader = cttz(mask)`. A lane is elected iff its lane id equals the
/// leader. This matches `elect.sync` semantics for both full-wave and
/// subset (partial mask) elections.
fn convert_elect_sync_maca(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("elect.sync requires 1 operand [mask]");
    }
    let mask = operands[0];

    // Leader = index of the lowest set bit in the mask: cttz(mask).
    let zero_undef = create_i1_const(ctx, rewriter, false);
    let cttz_ty =
        llvm_types::FuncType::get(ctx, i64_ty.into(), vec![i64_ty.into(), i1_ty.into()], false);
    let leader_i64 = call_intrinsic(ctx, rewriter, op, "llvm_cttz_i64", cttz_ty, vec![mask, zero_undef])?;
    let leader_i64 = leader_i64.deref(ctx).get_result(0);
    let leader = llvm::TruncOp::new(ctx, leader_i64, i32_ty.into());
    rewriter.insert_operation(ctx, leader.get_operation());
    let leader = leader.get_operation().deref(ctx).get_result(0);

    // is_elected = (lane_id == leader)
    let lane_id = crate::convert::intrinsics::basic::emit_maca_lane_id(ctx, rewriter, op)?;
    let is_elected = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::EQ, lane_id, leader);
    rewriter.insert_operation(ctx, is_elected.get_operation());
    let is_elected = is_elected.get_operation().deref(ctx).get_result(0);

    rewriter.replace_operation_with_values(ctx, op, vec![leader, is_elected]);
    Ok(())
}
