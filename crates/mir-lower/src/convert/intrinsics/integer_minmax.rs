// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lowering helper for generated extended integer min/max operations.
//!
//! Every variant is a pure two-operand instruction over 32-bit registers:
//! the scalar `.relu` forms operate on one `s32`, and the packed forms carry
//! an `s16x2`/`u16x2` pair in one `b32` register.

use llvm_export::op_interfaces::{BinArithOp, IntBinArithOpWithOverflowFlag};
use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Lower one generated integer min/max operation to its reviewed PTX
/// instruction.
pub(crate) fn convert_generated_integer_minmax(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!(
            "generated integer min/max operation requires 2 operands, got {}",
            operands.len()
        );
    }
    // MXMACA has no integer min/max inline asm; lower to icmp+select, which
    // mxcc compiles natively. Relu variants clamp the result to zero.
    if crate::context::lowering_options(ctx).backend == crate::BackendTarget::Maca {
        // Packed s16x2/u16x2 forms: compare per 16-bit half, then re-merge.
        if ptx_mnemonic.contains("16x2") {
            return emit_maca_packed_minmax(ctx, rewriter, op, operands, ptx_mnemonic);
        }
        let relu = ptx_mnemonic.contains("relu");
        let is_min = ptx_mnemonic.starts_with("min");
        let (a, b) = (operands[0], operands[1]);
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let pred = if is_min {
            llvm_export::attributes::ICmpPredicateAttr::SLT
        } else {
            llvm_export::attributes::ICmpPredicateAttr::SGT
        };
        let cmp = llvm::ICmpOp::new(ctx, pred, a, b);
        rewriter.insert_operation(ctx, cmp.get_operation());
        let cmp_v = cmp.get_operation().deref(ctx).get_result(0);
        let zero = {
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::utils::apint::APInt;
            let apint = APInt::from_i64(0, std::num::NonZeroUsize::new(32).unwrap());
            let attr = IntegerAttr::new(i32_ty, apint);
            let c = llvm::ConstantOp::new(ctx, attr.into());
            rewriter.insert_operation(ctx, c.get_operation());
            c.get_operation().deref(ctx).get_result(0)
        };
        // min: pick a when a<b; max: pick a when a>b. relu: select max(0, v).
        let selected = llvm::SelectOp::new(ctx, cmp_v, a, b);
        rewriter.insert_operation(ctx, selected.get_operation());
        let selected_v = selected.get_operation().deref(ctx).get_result(0);
        if relu {
            let cmp_zero = llvm::ICmpOp::new(
                ctx,
                llvm_export::attributes::ICmpPredicateAttr::SGT,
                selected_v,
                zero,
            );
            rewriter.insert_operation(ctx, cmp_zero.get_operation());
            let cmp_zero_v = cmp_zero.get_operation().deref(ctx).get_result(0);
            let clamped = llvm::SelectOp::new(ctx, cmp_zero_v, selected_v, zero);
            rewriter.insert_operation(ctx, clamped.get_operation());
            rewriter.replace_operation(ctx, op, clamped.get_operation());
        } else {
            rewriter.replace_operation(ctx, op, selected.get_operation());
        }
        let _ = i1_ty;
        return Ok(());
    }

    let result_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        operands,
        &format!("{ptx_mnemonic} $0, $1, $2;"),
        "=r,r,r",
        AsmKind::Pure,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Per-half 16-bit min/max for the packed `s16x2`/`u16x2` variants on MACA.
///
/// Each half is extracted with shifts, compared, selected, and the halves
/// are re-merged with shifts+or. `relu` clamps each half to zero.
fn emit_maca_packed_minmax(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands: Vec<pliron::value::Value>,
    ptx_mnemonic: &str,
) -> Result<()> {
    use llvm_export::attributes::{ICmpPredicateAttr, IntegerOverflowFlagsAttr};
    let (a, b) = (operands[0], operands[1]);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let overflow = IntegerOverflowFlagsAttr::default();
    let unsigned = ptx_mnemonic.contains("u16x2");
    let is_min = ptx_mnemonic.starts_with("min");
    let relu = ptx_mnemonic.contains("relu");

    fn cst(ctx: &mut Context, rewriter: &mut DialectConversionRewriter, n: i64) -> pliron::value::Value {
        use pliron::builtin::attributes::IntegerAttr;
        use pliron::utils::apint::APInt;
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let attr = IntegerAttr::new(
            i32_ty,
            APInt::from_i64(n, std::num::NonZeroUsize::new(32).unwrap()),
        );
        let c = llvm::ConstantOp::new(ctx, attr.into());
        rewriter.insert_operation(ctx, c.get_operation());
        c.get_operation().deref(ctx).get_result(0)
    }
    let shift16 = cst(ctx, rewriter, 16);
    let mask16_pre = cst(ctx, rewriter, 0xffff);

    fn half(
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        v: pliron::value::Value,
        which: usize,
        shift16: pliron::value::Value,
        overflow: &llvm_export::attributes::IntegerOverflowFlagsAttr,
    ) -> pliron::value::Value {
        if which == 0 {
            // low half: (v << 16) >> 16 logical
            let shifted = llvm::ShlOp::new_with_overflow_flag(ctx, v, shift16, overflow.clone());
            rewriter.insert_operation(ctx, shifted.get_operation());
            let shifted = shifted.get_operation().deref(ctx).get_result(0);
            let lo = llvm::LShrOp::new(ctx, shifted, shift16);
            rewriter.insert_operation(ctx, lo.get_operation());
            lo.get_operation().deref(ctx).get_result(0)
        } else {
            let hi = llvm::LShrOp::new(ctx, v, shift16);
            rewriter.insert_operation(ctx, hi.get_operation());
            hi.get_operation().deref(ctx).get_result(0)
        }
    }

    fn sign_extend(
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        v: pliron::value::Value,
        shift16: pliron::value::Value,
        overflow: &llvm_export::attributes::IntegerOverflowFlagsAttr,
        unsigned: bool,
    ) -> pliron::value::Value {
        // (v << 16) >> 16 arithmetic for signed; for unsigned keep as-is
        // (values are already zero-extended in the low half).
        if unsigned {
            v
        } else {
            let shifted = llvm::ShlOp::new_with_overflow_flag(ctx, v, shift16, overflow.clone());
            rewriter.insert_operation(ctx, shifted.get_operation());
            let shifted = shifted.get_operation().deref(ctx).get_result(0);
            let sext = llvm::AShrOp::new(ctx, shifted, shift16);
            rewriter.insert_operation(ctx, sext.get_operation());
            sext.get_operation().deref(ctx).get_result(0)
        }
    }

    // Signed compares need the 16-bit halves sign-extended to i32;
    // unsigned compares compare the zero-extended halves directly.
    let widen = |ctx: &mut Context, rewriter: &mut DialectConversionRewriter, v: pliron::value::Value| {
        if unsigned {
            v
        } else {
            sign_extend(ctx, rewriter, v, shift16, &overflow, unsigned)
        }
    };
    let a_lo_raw = half(ctx, rewriter, a, 0, shift16, &overflow);
    let a_lo = widen(ctx, rewriter, a_lo_raw);
    let a_hi_raw = half(ctx, rewriter, a, 1, shift16, &overflow);
    let a_hi = widen(ctx, rewriter, a_hi_raw);
    let b_lo_raw = half(ctx, rewriter, b, 0, shift16, &overflow);
    let b_lo = widen(ctx, rewriter, b_lo_raw);
    let b_hi_raw = half(ctx, rewriter, b, 1, shift16, &overflow);
    let b_hi = widen(ctx, rewriter, b_hi_raw);

    fn minmax(
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        x: pliron::value::Value,
        y: pliron::value::Value,
        is_min: bool,
        unsigned: bool,
    ) -> pliron::value::Value {
        let pred = if is_min {
            if unsigned {
                ICmpPredicateAttr::ULT
            } else {
                ICmpPredicateAttr::SLT
            }
        } else if unsigned {
            ICmpPredicateAttr::UGT
        } else {
            ICmpPredicateAttr::SGT
        };
        let cmp = llvm::ICmpOp::new(ctx, pred, x, y);
        rewriter.insert_operation(ctx, cmp.get_operation());
        let cmp_v = cmp.get_operation().deref(ctx).get_result(0);
        let sel = llvm::SelectOp::new(ctx, cmp_v, x, y);
        rewriter.insert_operation(ctx, sel.get_operation());
        sel.get_operation().deref(ctx).get_result(0)
    }

    let mut lo = minmax(ctx, rewriter, a_lo, b_lo, is_min, unsigned);
    let mut hi = minmax(ctx, rewriter, a_hi, b_hi, is_min, unsigned);

    if relu {
        let zero = cst(ctx, rewriter, 0);
        // relu on s16x2 clamps each signed half at 0; on u16x2 values are
        // already non-negative so this is a no-op for that half.
        fn clamp(
            ctx: &mut Context,
            rewriter: &mut DialectConversionRewriter,
            v: pliron::value::Value,
            unsigned: bool,
            zero: pliron::value::Value,
        ) -> pliron::value::Value {
            if unsigned {
                return v;
            }
            let pred = ICmpPredicateAttr::SLT;
            let cmp = llvm::ICmpOp::new(ctx, pred, v, zero);
            rewriter.insert_operation(ctx, cmp.get_operation());
            let cmp_v = cmp.get_operation().deref(ctx).get_result(0);
            let sel = llvm::SelectOp::new(ctx, cmp_v, zero, v);
            rewriter.insert_operation(ctx, sel.get_operation());
            sel.get_operation().deref(ctx).get_result(0)
        }
        // After min/max the halves are in 16-bit range; clamp then re-extend.
        lo = clamp(ctx, rewriter, lo, unsigned, zero);
        hi = clamp(ctx, rewriter, hi, unsigned, zero);
    }

    // Truncate each half to 16 bits (mask) and merge: (hi << 16) | (lo & 0xffff)
    let lo_masked = llvm::AndOp::new(ctx, lo, mask16_pre);
    rewriter.insert_operation(ctx, lo_masked.get_operation());
    let lo_masked = lo_masked.get_operation().deref(ctx).get_result(0);
    let hi_shifted = llvm::ShlOp::new_with_overflow_flag(ctx, hi, shift16, overflow);
    rewriter.insert_operation(ctx, hi_shifted.get_operation());
    let hi_shifted = hi_shifted.get_operation().deref(ctx).get_result(0);
    let merged = llvm::OrOp::new(ctx, hi_shifted, lo_masked);
    rewriter.insert_operation(ctx, merged.get_operation());
    rewriter.replace_operation(ctx, op, merged.get_operation());
    Ok(())
}
