/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MXMACA (MetaX C500) native Wave64 matrix intrinsics.
//!
//! Hand-written extension for the three `mma_m16n16k16_*` device intrinsics.
//! These are not part of the generated `cuda-intrinsics-gen` catalog: the
//! C500 wave width (64), tile shape (16x16x16), and per-lane fragment ABI
//! differ from CUDA's `m16n8k16` operations. This module restores the
//! recognition that predated the generated-intrinsics refactor.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::{
    attributes::FieldIndexAttr,
    ops::{MirConstructArrayOp, MirExtractFieldOp},
    types::MirArrayType,
};
use dialect_nvvm::ops::{
    MmaM16N16K16F32Bf16Op, MmaM16N16K16F32F16Op, MmaM16N16K16I32I8Op,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

/// Dispatch the C500-native `cuda_device::wmma::mma_m16n16k16_*` intrinsics.
///
/// Returns `Ok(None)` for any other path so the caller can continue with the
/// remaining intrinsic categories.
#[allow(clippy::too_many_arguments)]
pub fn try_dispatch(
    ctx: &mut Context,
    body: &mir::Body,
    name: &str,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Option<Ptr<Operation>>> {
    let emit = match name {
        "cuda_device::wmma::mma_m16n16k16_f32_f16" => emit_mma_m16n16k16_f32_f16,
        "cuda_device::wmma::mma_m16n16k16_f32_bf16" => emit_mma_m16n16k16_f32_bf16,
        "cuda_device::wmma::mma_m16n16k16_i32_i8" => emit_mma_m16n16k16_i32_i8,
        _ => return Ok(None),
    };
    Ok(Some(emit(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )?))
}

/// Emit the C500-native Wave64 `mma_m16n16k16_f32_f16` operation.
#[allow(clippy::too_many_arguments)]
pub fn emit_mma_m16n16k16_f32_f16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_mma_m16n16k16_f32_16bit(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        false,
    )
}

/// Emit the C500-native Wave64 `mma_m16n16k16_f32_bf16` operation.
#[allow(clippy::too_many_arguments)]
pub fn emit_mma_m16n16k16_f32_bf16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_mma_m16n16k16_f32_16bit(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        true,
    )
}

/// Shared lowering for the f16/bf16 variants: 8 register operands
/// (C[0..4] f32, A[0..2] packed i32, B[0..2] packed i32) and 4 f32 results.
#[allow(clippy::too_many_arguments)]
fn emit_mma_m16n16k16_f32_16bit(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    bf16: bool,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "m16n16k16 f32 MMA expects 3 arguments (c, a, b), got {}",
                args.len()
            ))
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let (c_array, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (c_registers, last_op) = extract_array_registers(
        ctx,
        c_array,
        f32_ty.into(),
        4,
        block_ptr,
        last_op,
        loc.clone(),
        "C",
    )?;
    let (a_array, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        Some(last_op),
        loc.clone(),
    )?;
    let (a_registers, last_op) = extract_array_registers(
        ctx,
        a_array,
        u32_ty.into(),
        2,
        block_ptr,
        last_op_after,
        loc.clone(),
        "A",
    )?;
    let (b_array, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        Some(last_op),
        loc.clone(),
    )?;
    let (b_registers, last_op) = extract_array_registers(
        ctx,
        b_array,
        u32_ty.into(),
        2,
        block_ptr,
        last_op_after,
        loc.clone(),
        "B",
    )?;

    let mut operands = c_registers;
    operands.extend(a_registers);
    operands.extend(b_registers);
    let op_info = if bf16 {
        MmaM16N16K16F32Bf16Op::get_concrete_op_info()
    } else {
        MmaM16N16K16F32F16Op::get_concrete_op_info()
    };
    let mma_op = Operation::new(ctx, op_info, vec![f32_ty.into(); 4], operands, vec![], 0);
    mma_op.deref_mut(ctx).set_loc(loc.clone());
    mma_op.insert_after(ctx, last_op);

    let d_registers = (0..4)
        .map(|index| mma_op.deref(ctx).get_result(index))
        .collect();
    let array_ty = MirArrayType::get(ctx, f32_ty.into(), 4);
    let d_array = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        d_registers,
        vec![],
        0,
    );
    d_array.deref_mut(ctx).set_loc(loc.clone());
    d_array.insert_after(ctx, mma_op);
    let result = d_array.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        d_array,
        value_map,
        block_map,
        loc,
        "m16n16k16 f32 MMA call without target block",
    )
}

/// Emit the C500-native Wave64 `mma_m16n16k16_i32_i8` operation:
/// 4 i32 accumulator registers, then one packed A and one packed B register.
#[allow(clippy::too_many_arguments)]
pub fn emit_mma_m16n16k16_i32_i8(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_m16n16k16_i32_i8 expects 3 arguments (c, a, b), got {}",
                args.len()
            ))
        );
    }
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let (c_array, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (mut operands, last_op) = extract_array_registers(
        ctx,
        c_array,
        i32_ty.into(),
        4,
        block_ptr,
        last_op,
        loc.clone(),
        "C",
    )?;
    let (a, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        Some(last_op),
        loc.clone(),
    )?;
    operands.push(a);
    let last_op = last_op.expect("translated A operand must produce an operation");
    let (b, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        Some(last_op),
        loc.clone(),
    )?;
    operands.push(b);
    let last_op = last_op.expect("translated B operand must produce an operation");
    let mma_op = Operation::new(
        ctx,
        MmaM16N16K16I32I8Op::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());
    mma_op.insert_after(ctx, last_op);
    let results = (0..4)
        .map(|index| mma_op.deref(ctx).get_result(index))
        .collect();
    let array_ty = MirArrayType::get(ctx, i32_ty.into(), 4);
    let array = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array.deref_mut(ctx).set_loc(loc.clone());
    array.insert_after(ctx, mma_op);
    let result = array.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        array,
        value_map,
        block_map,
        loc,
        "mma_m16n16k16_i32_i8 call without target block",
    )
}

/// Extract a fixed-size Rust array into scalar SSA register values.
///
/// Constant-field extraction lowers to LLVM `extractvalue`, so no temporary
/// stack slot is introduced for the MMA fragments.
fn extract_array_registers(
    ctx: &mut Context,
    array: Value,
    expected_element_ty: TypeHandle,
    expected_len: usize,
    block_ptr: Ptr<BasicBlock>,
    mut last_op: Option<Ptr<Operation>>,
    loc: Location,
    fragment_name: &str,
) -> TranslationResult<(Vec<Value>, Ptr<Operation>)> {
    let array_ty = array.get_type(ctx);
    let valid_array = {
        let array_ty = array_ty.deref(ctx);
        array_ty
            .downcast_ref::<MirArrayType>()
            .is_some_and(|array_ty| {
                array_ty.size() == expected_len as u64
                    && array_ty.element_type() == expected_element_ty
            })
    };
    if !valid_array {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "MMA {fragment_name} fragment must be an array of {expected_len} scalar registers"
            ))
        );
    }

    let mut registers = Vec::with_capacity(expected_len);
    for index in 0..expected_len {
        let extract = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![expected_element_ty],
            vec![array],
            vec![],
            0,
        );
        extract.deref_mut(ctx).set_loc(loc.clone());
        let extract = MirExtractFieldOp::new(extract);
        extract.set_attr_index(ctx, FieldIndexAttr(index as u32));
        if let Some(previous) = last_op {
            extract.get_operation().insert_after(ctx, previous);
        } else {
            extract.get_operation().insert_at_front(block_ptr, ctx);
        }
        last_op = Some(extract.get_operation());
        registers.push(extract.get_operation().deref(ctx).get_result(0));
    }

    Ok((registers, last_op.expect("non-empty MMA fragments")))
}
