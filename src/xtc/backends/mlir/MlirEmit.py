#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Emit MLIR from an explicit OpDesc (axes, tensor shapes, access maps, scalar body)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Type, TypeAlias, cast

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import affine, arith, builtin, linalg, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    IndexType,
    MemRefType,
    StringAttr,
    TensorType,
    UnitAttr,
    f32,
    f64,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap

__all__ = [
    "Axis",
    "LEAKY_RELU_ALPHA",
    "OpDesc",
    "ScalarBody",
    "TensorDesc",
    "emit_op",
    "resolve_epilogue",
]

OpAttrs: TypeAlias = dict[str, Any]
AccessMap: TypeAlias = Callable[..., tuple[Any, ...]]
ScalarBody: TypeAlias = Callable[[Sequence[SSAValue], SSAValue], SSAValue]


@dataclass(frozen=True)
class Axis:
    name: str
    size: int
    kind: Literal["P", "R"]


@dataclass(frozen=True)
class TensorDesc:
    name: str
    shape: tuple[int, ...]
    dtype: str
    access: AccessMap


@dataclass(frozen=True)
class OpDesc:
    name: str
    axes: tuple[Axis, ...]
    inputs: tuple[TensorDesc, ...]
    output: TensorDesc
    body: ScalarBody
    init: float | None = 0.0
    epilogue: ScalarBody | None = None


def dtype_to_xdsl(dtype: str) -> tuple[Any, int]:
    return {"float32": (f32, 32), "float64": (f64, 64)}[dtype]


LEAKY_RELU_ALPHA = 0.01


def resolve_epilogue(name: str | None, dtype: str) -> ScalarBody | None:
    """Build the elementwise ScalarBody applied after the reduction."""
    if name is None:
        return None
    _, element_bitwidth = dtype_to_xdsl(dtype)

    if name == "relu":

        def relu(_inputs: Sequence[SSAValue], value: SSAValue) -> SSAValue:
            zero = arith.ConstantOp(builtin.FloatAttr(0.0, element_bitwidth))
            return arith.MaximumfOp(value, zero.results[0]).results[0]

        return relu

    if name == "leaky_relu":

        def leaky_relu(_inputs: Sequence[SSAValue], value: SSAValue) -> SSAValue:
            alpha = arith.ConstantOp(
                builtin.FloatAttr(LEAKY_RELU_ALPHA, element_bitwidth)
            )
            return arith.MaximumfOp(
                value, arith.MulfOp(value, alpha.results[0])
            ).results[0]

        return leaky_relu

    raise NotImplementedError(f"unsupported epilogue {name!r}")


def _emit_linalg(
    desc: OpDesc,
    *,
    block: Block | None,
    args: Sequence[SSAValue],
    op_type: Type[TensorType] | Type[MemRefType],
) -> tuple[Block, OpAttrs]:
    if not desc.axes:
        raise ValueError("OpDesc.axes must be non-empty")

    axis_size_by_name = {axis.name: axis.size for axis in desc.axes}
    axis_names = [axis.name for axis in desc.axes]
    fill_projected_axes = desc.output.access(*axis_names)
    fill_dims = {name: axis_size_by_name[name] for name in fill_projected_axes}
    element_type, element_bitwidth = dtype_to_xdsl(desc.output.dtype)
    init_value = 0.0 if desc.init is None else desc.init

    operand_shapes = [input_desc.shape for input_desc in desc.inputs] + [
        desc.output.shape
    ]

    if block is None:
        assert args == (), "standalone emission requires args=()"
        block = Block(
            arg_types=[op_type(element_type, shape) for shape in operand_shapes]
        )
        args = block.args
    assert len(args) == len(operand_shapes)
    assert all(isinstance(arg.type, op_type) for arg in args)

    # Parent block: fill and linalg.generic ops wired to graph operands.
    with ImplicitBuilder(block):
        num_inputs = len(desc.inputs)
        input_operands = list(args[:num_inputs])
        output_operand = args[num_inputs]

        init_constant = arith.ConstantOp(
            builtin.FloatAttr(init_value, element_bitwidth)
        )
        result_types = (output_operand.type,) if op_type == TensorType else ()
        fill_op = linalg.FillOp(
            res=result_types,
            inputs=(init_constant.results[0],),
            outputs=(output_operand,),
        )
        output_after_fill: SSAValue = (
            fill_op.results[0] if op_type == TensorType else output_operand
        )

        # Region body: one scalar iteration of the reduction (linalg.generic).
        generic_body_block = Block(arg_types=[element_type] * (num_inputs + 1))
        with ImplicitBuilder(generic_body_block):
            input_elements = list(generic_body_block.args[:num_inputs])
            accumulator = generic_body_block.args[num_inputs]
            updated_accumulator = desc.body(input_elements, accumulator)
            linalg.YieldOp(updated_accumulator)
        compute_op = linalg.GenericOp(
            inputs=tuple(input_operands),
            outputs=(output_after_fill,),
            body=Region([generic_body_block]),  # type: ignore[arg-type]
            indexing_maps=[
                AffineMapAttr(AffineMap.from_callable(tensor_desc.access))
                for tensor_desc in (*desc.inputs, desc.output)
            ],
            iterator_types=[
                StringAttr({"P": "parallel", "R": "reduction"}[axis.kind])
                for axis in desc.axes
            ],
            result_types=result_types,
        )
        last_op = compute_op
        output_after_compute: SSAValue = (
            compute_op.results[0] if op_type == TensorType else output_operand
        )
        epilogue_op: Any | None = None

        if desc.epilogue is not None:
            output_rank = len(desc.output.shape)
            identity_index_map = AffineMap(
                output_rank,
                0,
                tuple(AffineExpr.dimension(dim) for dim in range(output_rank)),
            )
            if op_type == TensorType:
                epilogue_output = tensor.EmptyOp(
                    dynamic_sizes=[],
                    tensor_type=cast(TensorType, output_operand.type),
                ).results[0]
                epilogue_result_types = (epilogue_output.type,)
            else:
                epilogue_output = output_operand
                epilogue_result_types = ()
            # Region body: elementwise epilogue on the output tensor.
            epilogue_body_block = Block(arg_types=[element_type, element_type])
            with ImplicitBuilder(epilogue_body_block):
                output_element = epilogue_body_block.args[0]
                epilogue_result = desc.epilogue([], output_element)
                linalg.YieldOp(epilogue_result)
            epilogue_op = linalg.GenericOp(
                inputs=(output_after_compute,),
                outputs=(epilogue_output,),
                body=Region([epilogue_body_block]),  # type: ignore[arg-type]
                indexing_maps=[
                    AffineMapAttr(identity_index_map),
                    AffineMapAttr(identity_index_map),
                ],
                iterator_types=[StringAttr("parallel") for _ in range(output_rank)],
                result_types=epilogue_result_types,
            )
            last_op = epilogue_op

        compute_dims = {axis.name: axis.size for axis in desc.axes}
        fill_node_id = f"{desc.name}_0"
        compute_node_id = f"{desc.name}"
        fill_op.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        compute_op.attributes[f"__xtc_id_{compute_node_id}_"] = UnitAttr()
        nodes_map: dict[str, Any] = {
            fill_node_id: fill_op,
            compute_node_id: compute_op,
        }
        dims_sizes: list[dict[str, int]] = [fill_dims, compute_dims]
        if epilogue_op is not None:
            epilogue_node_id = f"{desc.name}_1"
            epilogue_op.attributes[f"__xtc_id_{epilogue_node_id}_"] = UnitAttr()
            nodes_map[epilogue_node_id] = epilogue_op
            dims_sizes.append(fill_dims)
        attrs: OpAttrs = {
            "nodes_map": nodes_map,
            "dims_sizes": dims_sizes,
            "output_nodes": [last_op],
            "root_node": compute_node_id,
        }
    return block, attrs


def _emit_affine(
    desc: OpDesc,
    *,
    block: Block | None,
    args: Sequence[SSAValue],
    op_type: Type[TensorType] | Type[MemRefType],
) -> tuple[Block, OpAttrs]:
    if op_type is not TensorType:
        raise NotImplementedError(
            "emit_op affine currently requires tensor dialect "
            "(use_tensor_dialect=True); memref affine emission is not implemented yet"
        )
    if not desc.axes:
        raise ValueError("OpDesc.axes must be non-empty")

    # Compile each access lambda into one AffineMap per tensor index (affine.apply
    # requires a unidimensional map).
    def apply_access(
        index_maps: list[AffineMap], loop_indices: list[SSAValue]
    ) -> list[SSAValue]:
        return [
            affine.ApplyOp(loop_indices, AffineMapAttr(index_map)).results[0]
            for index_map in index_maps
        ]

    input_index_maps = []
    for input_desc in desc.inputs:
        multi = AffineMap.from_callable(input_desc.access)
        input_index_maps.append(
            [
                AffineMap(multi.num_dims, multi.num_symbols, (result,))
                for result in multi.results
            ]
        )
    output_multi = AffineMap.from_callable(desc.output.access)
    output_index_maps = [
        AffineMap(output_multi.num_dims, output_multi.num_symbols, (result,))
        for result in output_multi.results
    ]

    def nest_affine(
        loop_upper_bounds: Sequence[int],
        initial_iter_arg: SSAValue,
        innermost_body: Callable[[list[SSAValue], SSAValue], SSAValue],
        iter_arg_type: Any,
    ) -> affine.ForOp:
        """Nest affine.for over ``loop_upper_bounds``; call ``innermost_body`` at the leaf."""
        if not loop_upper_bounds:
            raise ValueError("nest_affine requires at least one dimension")

        def build_loop_body(
            depth: int, outer_loop_indices: list[SSAValue]
        ) -> Callable[[SSAValue, SSAValue], SSAValue]:
            def loop_body(loop_index: SSAValue, iter_arg: SSAValue) -> SSAValue:
                loop_indices = outer_loop_indices + [loop_index]
                if depth + 1 == len(loop_upper_bounds):
                    return innermost_body(loop_indices, iter_arg)

                inner_block = Block(arg_types=[IndexType(), iter_arg_type])
                with ImplicitBuilder(inner_block):
                    affine.YieldOp.get(
                        build_loop_body(depth + 1, loop_indices)(
                            inner_block.args[0], inner_block.args[1]
                        )
                    )
                return affine.ForOp.from_region(
                    [],
                    [],
                    [iter_arg],
                    [iter_arg_type],
                    0,
                    loop_upper_bounds[depth + 1],
                    Region([inner_block]),
                    1,
                ).results[0]

            return loop_body

        outer_block = Block(arg_types=[IndexType(), iter_arg_type])
        with ImplicitBuilder(outer_block):
            affine.YieldOp.get(
                build_loop_body(0, [])(outer_block.args[0], outer_block.args[1])
            )
        return affine.ForOp.from_region(
            [],
            [],
            [initial_iter_arg],
            [iter_arg_type],
            0,
            loop_upper_bounds[0],
            Region([outer_block]),
            1,
        )

    def compute_leaf(
        loop_indices: list[SSAValue], output_carry: SSAValue
    ) -> SSAValue:
        # Build tensor indices via affine.apply (supports non-projection maps).
        input_elements = [
            tensor.ExtractOp(
                input_operand,
                apply_access(index_maps, loop_indices),
                element_type,
            ).results[0]
            for input_operand, index_maps in zip(input_operands, input_index_maps)
        ]
        output_indices = apply_access(output_index_maps, loop_indices)
        output_element = tensor.ExtractOp(
            output_carry,
            output_indices,
            element_type,
        ).results[0]
        updated_element = desc.body(input_elements, output_element)
        return tensor.InsertOp(
            updated_element,
            output_carry,
            output_indices,
        ).results[0]

    def epilogue_leaf(
        output_indices: list[SSAValue], output_carry: SSAValue
    ) -> SSAValue:
        assert desc.epilogue is not None
        output_element = tensor.ExtractOp(
            output_carry, output_indices, element_type
        ).results[0]
        updated_element = desc.epilogue([], output_element)
        return tensor.InsertOp(
            updated_element, output_carry, output_indices
        ).results[0]

    axis_size_by_name = {axis.name: axis.size for axis in desc.axes}
    axis_names = [axis.name for axis in desc.axes]
    fill_projected_axes = desc.output.access(*axis_names)
    fill_sizes = [axis_size_by_name[name] for name in fill_projected_axes]
    fill_dims = {name: axis_size_by_name[name] for name in fill_projected_axes}
    element_type, element_bitwidth = dtype_to_xdsl(desc.output.dtype)
    init_value = 0.0 if desc.init is None else desc.init

    operand_shapes = [input_desc.shape for input_desc in desc.inputs] + [
        desc.output.shape
    ]

    if block is None:
        assert args == (), "standalone emission requires args=()"
        block = Block(
            arg_types=[TensorType(element_type, shape) for shape in operand_shapes]
        )
        args = block.args
    assert len(args) == len(operand_shapes)
    assert all(isinstance(arg.type, TensorType) for arg in args)

    num_inputs = len(desc.inputs)
    input_operands = list(args[:num_inputs])
    output_operand = args[num_inputs]
    output_tensor_type = cast(TensorType, output_operand.type)
    iteration_axis_sizes = [axis.size for axis in desc.axes]

    # Parent block: fill, affine nest compute, optional epilogue nest.
    with ImplicitBuilder(block):
        init_constant = arith.ConstantOp(
            builtin.FloatAttr(init_value, element_bitwidth)
        )
        # Fill nest iterates output dims directly; loop indices are output indices.
        fill_op = nest_affine(
            fill_sizes,
            output_operand,
            lambda output_fill_indices, output_carry: tensor.InsertOp(
                init_constant.results[0],
                output_carry,
                output_fill_indices,
            ).results[0],
            output_tensor_type,
        )

        compute_op = nest_affine(
            iteration_axis_sizes,
            fill_op.results[0],
            compute_leaf,
            output_tensor_type,
        )
        last_op = compute_op
        output_tensor = compute_op.results[0]
        epilogue_op: Any | None = None

        if desc.epilogue is not None:
            epilogue_op = nest_affine(
                list(desc.output.shape),
                output_tensor,
                epilogue_leaf,
                output_tensor_type,
            )
            last_op = epilogue_op

        # Nodes are exposed like linalg, but not schedulable yet: affine.ForOp
        # clones only remap the iter-arg operand, not parent-block input tensors
        # captured inside the region. Use get_scheduler(nodes=[]).
        compute_dims = {axis.name: axis.size for axis in desc.axes}
        fill_node_id = f"{desc.name}_0"
        compute_node_id = f"{desc.name}"
        fill_op.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        compute_op.attributes[f"__xtc_id_{compute_node_id}_"] = UnitAttr()
        nodes_map: dict[str, Any] = {
            fill_node_id: fill_op,
            compute_node_id: compute_op,
        }
        dims_sizes: list[dict[str, int]] = [fill_dims, compute_dims]
        if epilogue_op is not None:
            epilogue_node_id = f"{desc.name}_1"
            epilogue_op.attributes[f"__xtc_id_{epilogue_node_id}_"] = UnitAttr()
            nodes_map[epilogue_node_id] = epilogue_op
            dims_sizes.append(fill_dims)
        attrs: OpAttrs = {
            "nodes_map": nodes_map,
            "dims_sizes": dims_sizes,
            "output_nodes": [last_op],
            "root_node": compute_node_id,
        }
    return block, attrs


def emit_op(
    desc: OpDesc,
    *,
    dialect: Literal["linalg", "affine"],
    block: Block | None = None,
    args: Sequence[SSAValue] = (),
    op_type: Type[TensorType] | Type[MemRefType] = MemRefType,
) -> tuple[Block, OpAttrs]:
    if dialect == "linalg":
        return _emit_linalg(desc, block=block, args=args, op_type=op_type)
    if dialect == "affine":
        return _emit_affine(desc, block=block, args=args, op_type=op_type)
    raise ValueError(f"unknown dialect {dialect!r}")
