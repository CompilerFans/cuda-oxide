/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix intrinsic lowering (`movmatrix`, `mma.sync`).

use crate::BackendTarget;
use crate::context::lowering_options;
use crate::convert::intrinsics::common::*;
use llvm_export::attributes::IntegerOverflowFlagsAttr;
use llvm_export::op_interfaces::{
    BinArithOp, CastOpInterface, CastOpWithNNegInterface, IntBinArithOpWithOverflowFlag,
};
use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::utils::apint::APInt;
use std::num::NonZeroUsize;

fn reject_cuda_warp_matrix_on_maca(ctx: &Context, operation: &str) -> Result<()> {
    if lowering_options(ctx).backend == BackendTarget::Maca {
        return pliron::input_err_noloc!(
            "CUDA warp-matrix operation `{}` is unsupported for MACA target; C500 requires a native 16x16x16 lowering",
            operation
        );
    }
    Ok(())
}

/// Lower the C500-native Wave64 FP16 MMA to the exact mxcc LLVM intrinsic ABI.
pub(crate) fn convert_mma_m16n16k16_f32_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_m16n16k16_f32_16bit(ctx, rewriter, op, "llvm_mxc_mma_f32_16x16x16f16")
}

pub(crate) fn convert_mma_m16n16k16_f32_bf16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_m16n16k16_f32_16bit(ctx, rewriter, op, "llvm_mxc_mma_f32_16x16x2bf16")
}

fn convert_mma_m16n16k16_f32_16bit(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic_name: &str,
) -> Result<()> {
    if lowering_options(ctx).backend != BackendTarget::Maca {
        return pliron::input_err_noloc!(
            "native m16n16k16 MMA is only supported by the MACA backend"
        );
    }
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 8 {
        return pliron::input_err_noloc!(
            "mma_m16n16k16_f32_f16 requires 8 register operands, got {}",
            operands.len()
        );
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i128_ty = IntegerType::get(ctx, 128, Signedness::Signless);
    let f32_ty = FP32Type::get(ctx);
    let half_ty = llvm_types::HalfType::get(ctx);
    let half4_ty =
        llvm_types::VectorType::get(ctx, half_ty.into(), 4, llvm_types::VectorTypeKind::Fixed);
    let f32x4_ty =
        llvm_types::VectorType::get(ctx, f32_ty.into(), 4, llvm_types::VectorTypeKind::Fixed);
    let overflow = IntegerOverflowFlagsAttr::default();

    let pack_pair = |ctx: &mut Context,
                     rewriter: &mut DialectConversionRewriter,
                     low: pliron::value::Value,
                     high: pliron::value::Value|
     -> pliron::value::Value {
        let low = llvm::ZExtOp::new_with_nneg(ctx, low, i64_ty.into(), false);
        rewriter.insert_operation(ctx, low.get_operation());
        let low = low.get_operation().deref(ctx).get_result(0);
        let high = llvm::ZExtOp::new_with_nneg(ctx, high, i64_ty.into(), false);
        rewriter.insert_operation(ctx, high.get_operation());
        let high = high.get_operation().deref(ctx).get_result(0);
        let shift = create_i64_const(ctx, rewriter, 32);
        let high = llvm::ShlOp::new_with_overflow_flag(ctx, high, shift, overflow.clone());
        rewriter.insert_operation(ctx, high.get_operation());
        let high = high.get_operation().deref(ctx).get_result(0);
        let packed = llvm::OrOp::new(ctx, low, high);
        rewriter.insert_operation(ctx, packed.get_operation());
        packed.get_operation().deref(ctx).get_result(0)
    };
    let a = pack_pair(ctx, rewriter, operands[4], operands[5]);
    let b = pack_pair(ctx, rewriter, operands[6], operands[7]);
    let a = llvm::BitcastOp::new(ctx, a, half4_ty.into());
    rewriter.insert_operation(ctx, a.get_operation());
    let a = a.get_operation().deref(ctx).get_result(0);
    let b = llvm::BitcastOp::new(ctx, b, half4_ty.into());
    rewriter.insert_operation(ctx, b.get_operation());
    let b = b.get_operation().deref(ctx).get_result(0);

    let mut packed_c = create_i128_const(ctx, rewriter, 0);
    for (index, accumulator) in operands.iter().take(4).enumerate() {
        let bits = llvm::BitcastOp::new(ctx, *accumulator, i32_ty.into());
        rewriter.insert_operation(ctx, bits.get_operation());
        let bits = bits.get_operation().deref(ctx).get_result(0);
        let bits = llvm::ZExtOp::new_with_nneg(ctx, bits, i128_ty.into(), false);
        rewriter.insert_operation(ctx, bits.get_operation());
        let mut bits = bits.get_operation().deref(ctx).get_result(0);
        if index != 0 {
            let shift = create_i128_const(ctx, rewriter, (index * 32) as i128);
            let shifted = llvm::ShlOp::new_with_overflow_flag(ctx, bits, shift, overflow.clone());
            rewriter.insert_operation(ctx, shifted.get_operation());
            bits = shifted.get_operation().deref(ctx).get_result(0);
        }
        let next = llvm::OrOp::new(ctx, packed_c, bits);
        rewriter.insert_operation(ctx, next.get_operation());
        packed_c = next.get_operation().deref(ctx).get_result(0);
    }
    let c = llvm::BitcastOp::new(ctx, packed_c, f32x4_ty.into());
    rewriter.insert_operation(ctx, c.get_operation());
    let c = c.get_operation().deref(ctx).get_result(0);
    let func_ty = llvm_types::FuncType::get(
        ctx,
        f32x4_ty.into(),
        vec![half4_ty.into(), half4_ty.into(), f32x4_ty.into()],
        false,
    );
    let call = call_intrinsic(ctx, rewriter, op, intrinsic_name, func_ty, vec![a, b, c])?;
    let result = call.deref(ctx).get_result(0);
    let packed = llvm::BitcastOp::new(ctx, result, i128_ty.into());
    rewriter.insert_operation(ctx, packed.get_operation());
    let packed = packed.get_operation().deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(4);
    for index in 0..4 {
        let mut bits = packed;
        if index != 0 {
            let shift = create_i128_const(ctx, rewriter, (index * 32) as i128);
            let shifted = llvm::LShrOp::new(ctx, bits, shift);
            rewriter.insert_operation(ctx, shifted.get_operation());
            bits = shifted.get_operation().deref(ctx).get_result(0);
        }
        let bits = llvm::TruncOp::new(ctx, bits, i32_ty.into());
        rewriter.insert_operation(ctx, bits.get_operation());
        let bits = bits.get_operation().deref(ctx).get_result(0);
        let value = llvm::BitcastOp::new(ctx, bits, f32_ty.into());
        rewriter.insert_operation(ctx, value.get_operation());
        results.push(value.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}

pub(crate) fn convert_mma_m16n16k16_i32_i8(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    if lowering_options(ctx).backend != BackendTarget::Maca {
        return pliron::input_err_noloc!(
            "mma_m16n16k16_i32_i8 is only supported by the MACA backend"
        );
    }
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 6 {
        return pliron::input_err_noloc!(
            "mma_m16n16k16_i32_i8 requires 6 register operands, got {}",
            operands.len()
        );
    }
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i128_ty = IntegerType::get(ctx, 128, Signedness::Signless);
    let i32x4_ty =
        llvm_types::VectorType::get(ctx, i32_ty.into(), 4, llvm_types::VectorTypeKind::Fixed);
    let overflow = IntegerOverflowFlagsAttr::default();
    let mut packed_c = create_i128_const(ctx, rewriter, 0);
    for (index, accumulator) in operands.iter().take(4).enumerate() {
        let bits = llvm::ZExtOp::new_with_nneg(ctx, *accumulator, i128_ty.into(), false);
        rewriter.insert_operation(ctx, bits.get_operation());
        let mut bits = bits.get_operation().deref(ctx).get_result(0);
        if index != 0 {
            let shift = create_i128_const(ctx, rewriter, (index * 32) as i128);
            let shifted = llvm::ShlOp::new_with_overflow_flag(ctx, bits, shift, overflow.clone());
            rewriter.insert_operation(ctx, shifted.get_operation());
            bits = shifted.get_operation().deref(ctx).get_result(0);
        }
        let next = llvm::OrOp::new(ctx, packed_c, bits);
        rewriter.insert_operation(ctx, next.get_operation());
        packed_c = next.get_operation().deref(ctx).get_result(0);
    }
    let c = llvm::BitcastOp::new(ctx, packed_c, i32x4_ty.into());
    rewriter.insert_operation(ctx, c.get_operation());
    let c = c.get_operation().deref(ctx).get_result(0);
    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32x4_ty.into(),
        vec![i32_ty.into(), i32_ty.into(), i32x4_ty.into()],
        false,
    );
    let call = call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_mxc_mma_i32_16x16x16i8",
        func_ty,
        vec![operands[4], operands[5], c],
    )?;
    let result = call.deref(ctx).get_result(0);
    let packed = llvm::BitcastOp::new(ctx, result, i128_ty.into());
    rewriter.insert_operation(ctx, packed.get_operation());
    let packed = packed.get_operation().deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(4);
    for index in 0..4 {
        let mut bits = packed;
        if index != 0 {
            let shift = create_i128_const(ctx, rewriter, (index * 32) as i128);
            let shifted = llvm::LShrOp::new(ctx, bits, shift);
            rewriter.insert_operation(ctx, shifted.get_operation());
            bits = shifted.get_operation().deref(ctx).get_result(0);
        }
        let bits = llvm::TruncOp::new(ctx, bits, i32_ty.into());
        rewriter.insert_operation(ctx, bits.get_operation());
        results.push(bits.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}

fn create_i128_const(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: i128,
) -> pliron::value::Value {
    let ty = IntegerType::get(ctx, 128, Signedness::Signless);
    let apint = APInt::from_i128(value, NonZeroUsize::new(128).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(ty, apint);
    let constant = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, constant.get_operation());
    constant.get_operation().deref(ctx).get_result(0)
}

/// Convert `nvvm.movmatrix_trans_b16` to inline PTX.
///
/// `movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;`
///
/// Warp-synchronous, uses convergent inline assembly.
pub(crate) fn convert_movmatrix_trans_b16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    reject_cuda_warp_matrix_on_maca(ctx, "movmatrix.sync.aligned.m8n8.trans.b16")?;
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!(
            "movmatrix_trans_b16 requires 1 operand, got {}",
            operands.len()
        );
    }

    let a_val = operands[0];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        AsmKind::Convergent,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `mma_m16n8k16_f32_bf16` to one register-only inline PTX operation.
///
/// Operand order is C[0..4], A[0..4], B[0..2]. The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results. There are no hidden pointer, stack, load, or store operands.
///
/// MXMACA rejects this CUDA-specific shape until native 16x16x16 lowering is available.
pub(crate) fn convert_mma_m16n8k16_f32_bf16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    reject_cuda_warp_matrix_on_maca(ctx, "mma.sync.m16n8k16.f32.bf16")?;

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k16_f32_bf16 requires 10 register operands, got {}",
            operands.len()
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f32_ty.into(); 4]);
    let template = concat!(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
        "{$0, $1, $2, $3}, ",
        "{$8, $9, $10, $11}, ",
        "{$12, $13}, ",
        "{$4, $5, $6, $7};"
    );
    let constraints = "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r";
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        template,
        constraints,
    )?;

    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(4);
    for index in 0..4 {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        results.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}

/// Convert `mma_m16n8k16_f32_f16` to one register-only inline PTX operation.
///
/// Operand order is C[0..4], A[0..4], B[0..2]. The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results. There are no hidden pointer, stack, load, or store operands.
///
/// MXMACA rejects this CUDA-specific shape until native 16x16x16 lowering is available.
pub(crate) fn convert_mma_m16n8k16_f32_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    reject_cuda_warp_matrix_on_maca(ctx, "mma.sync.m16n8k16.f32.f16")?;

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k16_f32_f16 requires 10 register operands, got {}",
            operands.len()
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f32_ty.into(); 4]);
    let template = concat!(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
        "{$0, $1, $2, $3}, ",
        "{$8, $9, $10, $11}, ",
        "{$12, $13}, ",
        "{$4, $5, $6, $7};"
    );
    let constraints = "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r";
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        template,
        constraints,
    )?;

    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(4);
    for index in 0..4 {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        results.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}

/// Convert `mma_m16n8k8_f32_tf32` to one register-only inline PTX operation.
///
/// Operand order is C[0..4], A[0..4], B[0..2]. The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results. There are no hidden pointer, stack, load, or store operands.
pub(crate) fn convert_mma_m16n8k8_f32_tf32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    reject_cuda_warp_matrix_on_maca(ctx, "mma.sync.m16n8k8.f32.tf32")?;
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k8_f32_tf32 requires 10 register operands, got {}",
            operands.len()
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f32_ty.into(); 4]);
    let template = concat!(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 ",
        "{$0, $1, $2, $3}, ",
        "{$8, $9, $10, $11}, ",
        "{$12, $13}, ",
        "{$4, $5, $6, $7};"
    );
    let constraints = "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r";
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        template,
        constraints,
    )?;

    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(4);
    for index in 0..4 {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        results.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}

/// Convert `mma_m16n8k32_s32_s8` to one register-only inline PTX operation.
///
/// Operand order is C[0..4], A[0..4], B[0..2]. The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results. All operands and results use the `r` (integer) constraint.
pub(crate) fn convert_mma_m16n8k32_s32_s8(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    reject_cuda_warp_matrix_on_maca(ctx, "mma.sync.m16n8k32.s32.s8")?;
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k32_s32_s8 requires 10 register operands, got {}",
            operands.len()
        );
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![i32_ty.into(); 4]);
    let template = concat!(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
        "{$0, $1, $2, $3}, ",
        "{$8, $9, $10, $11}, ",
        "{$12, $13}, ",
        "{$4, $5, $6, $7};"
    );
    let constraints = "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r";
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        template,
        constraints,
    )?;

    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(4);
    for index in 0..4 {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        results.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}

/// Convert `mma_m8n8k4_f64` to inline PTX assembly.
///
/// The operation consumes the two C registers plus A and B directly, and
/// returns both D fragment registers. No pointer or memory operand is involved.
pub(crate) fn convert_mma_m8n8k4_f64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    reject_cuda_warp_matrix_on_maca(ctx, "mma.sync.m8n8k4.f64")?;
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 4 {
        return pliron::input_err_noloc!(
            "mma_m8n8k4_f64 requires 4 f64 operands (c0, c1, a, b), got {}",
            operands.len()
        );
    }

    let f64_ty = FP64Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f64_ty.into(), f64_ty.into()]);
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \
         {$0, $1}, {$4}, {$5}, {$2, $3};",
        "=d,=d,d,d,d,d",
    )?;

    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(2);
    for index in 0..2 {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index])?;
        rewriter.insert_operation(ctx, extract.get_operation());
        results.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}
