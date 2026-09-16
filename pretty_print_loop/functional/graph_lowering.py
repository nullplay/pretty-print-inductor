from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import sympy
import torch
from torch._inductor import ir
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import OpsValue, V

from .ast import (
    BinaryOp,
    Dummy,
    Expr,
    Index,
    Input,
    Load,
    Loop,
    Num,
    Output,
    Stmt,
    Store,
    Tensor,
    TensorRef,
    UnaryOp,
    Var,
    seq,
)

if TYPE_CHECKING:
    from torch._inductor.graph import GraphLowering


class UnsupportedNodeError(RuntimeError):
    pass


def _unwrap(value: Any) -> Any:
    if isinstance(value, OpsValue):
        return value.value
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def _fold_binary(name: str, values: Sequence[Expr]) -> Expr:
    if not values:
        raise UnsupportedNodeError(f"cannot build empty {name} expression")
    result = values[0]
    for value in values[1:]:
        result = BinaryOp(name, result, value)
    return result


def sympy_to_expr(value: object) -> Expr:
    value = sympy.sympify(value)
    if isinstance(value, sympy.Integer):
        return Num(int(value))
    if isinstance(value, sympy.Symbol):
        return Var(str(value))
    if isinstance(value, sympy.Add):
        return _fold_binary("+", tuple(sympy_to_expr(arg) for arg in value.args))
    if isinstance(value, sympy.Mul):
        return _fold_binary("*", tuple(sympy_to_expr(arg) for arg in value.args))
    raise UnsupportedNodeError(
        f"unsupported flat-index expression: {type(value).__name__}({value})"
    )


def _make_indices(
    prefix: str, ranges: Sequence[object]
) -> tuple[list[Var], list[sympy.Symbol]]:
    variables = [Var(f"{prefix}{position}") for position in range(len(ranges))]
    symbols = [sympy_index_symbol(variable.name) for variable in variables]
    return variables, symbols


def _nest_loops(variables: Sequence[Var], ranges: Sequence[object], body: Stmt) -> Stmt:
    result = body
    for variable, extent in reversed(tuple(zip(variables, ranges))):
        result = Loop(variable, sympy_to_expr(extent), result)
    return result


class _ScalarOpsHandler(DefaultHandler):
    name = "ScalarLoopASTOpsHandler"

    def __init__(self, resolve_tensor: Any) -> None:
        self.resolve_tensor = resolve_tensor

    @staticmethod
    def _expr(value: object) -> Expr:
        value = _unwrap(value)
        if not isinstance(value, Expr):
            raise UnsupportedNodeError(
                f"expected scalar Expr, got {type(value).__name__}"
            )
        return value

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> Expr:
        if kwargs:
            raise UnsupportedNodeError(f"operator {name} has unsupported kwargs")
        operands = tuple(self._expr(arg) for arg in args)
        if len(operands) == 1:
            return UnaryOp(name, operands[0])
        if len(operands) == 2:
            operator = {"add": "+", "mul": "*"}.get(name, name)
            return BinaryOp(operator, operands[0], operands[1])
        raise UnsupportedNodeError(f"operator {name} has arity {len(operands)}")

    def load(self, name: str, index: sympy.Expr) -> Expr:
        return Load(self.resolve_tensor(name), Index(sympy_to_expr(index)))

    def constant(self, value: object, dtype: torch.dtype) -> Expr:
        if isinstance(value, bool):
            raise UnsupportedNodeError("boolean constants are not in the scalar AST")
        if isinstance(value, int):
            return Num(value)
        if isinstance(value, float) and value.is_integer():
            return Num(int(value))
        raise UnsupportedNodeError(f"non-integer constant {value!r} is unsupported")

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Expr:
        return sympy_to_expr(expr)

    def value_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Expr:
        return sympy_to_expr(expr)

    def identity(self, value: object) -> Expr:
        return self._expr(value)

    def to_dtype(
        self,
        value: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> Expr:
        return UnaryOp(str(dtype).removeprefix("torch."), self._expr(value))

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: object,
    ) -> Expr:
        if reduction_type != "sum":
            raise UnsupportedNodeError(
                f"only local sum is supported, got {reduction_type}"
            )
        return UnaryOp("rsum", self._expr(value))


class GraphLoweringToScalarAST:
    def __init__(self, graph: GraphLowering) -> None:
        self.graph = graph
        self._operation_names = {
            output.get_name()
            for operation in graph.operations
            for output in operation.get_outputs()
        }
        self._output_names = self._collect_output_names()
        self._accumulator_index = 0

    def _collect_output_names(self) -> set[str]:
        names: set[str] = set()
        for output in self.graph.graph_outputs:
            try:
                names.add(output.get_name())
            except (AttributeError, NotImplementedError):
                continue
        return names

    def _tensor(self, name: str) -> TensorRef:
        symbol = Var(name)
        if name in self._output_names:
            return Output(symbol)
        if name in self._operation_names:
            return Tensor(symbol)
        return Input(symbol)

    def _output_tensor(self, buffer: ir.ComputedBuffer) -> TensorRef:
        name = buffer.get_name()
        return Output(Var(name)) if name in self._output_names else Tensor(Var(name))

    @staticmethod
    def _output_index(
        buffer: ir.ComputedBuffer, indices: Sequence[sympy.Symbol]
    ) -> Index:
        try:
            indexer = buffer.get_layout().make_indexer()
            flat_index = indexer(indices)
        except (AttributeError, NotImplementedError) as error:
            raise UnsupportedNodeError(
                f"cannot recover flat output index for {buffer.get_name()}"
            ) from error
        return Index(sympy_to_expr(flat_index))

    def _evaluate(self, function: Any, *indices: Sequence[sympy.Symbol]) -> Expr:
        handler = _ScalarOpsHandler(self._tensor)
        with (
            V.set_graph_handler(self.graph),
            V.set_ops_handler(handler),
            patch.object(ir.FlexibleLayout, "allow_indexing", True),
        ):
            value = _unwrap(function(*indices))
        if not isinstance(value, Expr):
            raise UnsupportedNodeError(
                f"inner function returned {type(value).__name__}, expected Expr"
            )
        return value

    def _pointwise(self, buffer: ir.ComputedBuffer) -> Stmt:
        data = buffer.data
        if not isinstance(data, ir.Pointwise):
            raise TypeError(f"expected Pointwise, got {type(data).__name__}")
        variables, symbols = _make_indices("i", data.ranges)
        value = self._evaluate(data.inner_fn, symbols)
        store = Store(
            self._output_tensor(buffer),
            self._output_index(buffer, symbols),
            value,
        )
        return _nest_loops(variables, data.ranges, store)

    def _reduction(self, buffer: ir.ComputedBuffer) -> Stmt:
        data = buffer.data
        if not isinstance(data, ir.Reduction):
            raise TypeError(f"expected Reduction, got {type(data).__name__}")
        if data.reduction_type != "sum":
            raise UnsupportedNodeError(
                f"only sum Reduction is supported, got {data.reduction_type}"
            )

        variables, symbols = _make_indices("i", data.ranges)
        reduction_variables, reduction_symbols = _make_indices(
            "r", data.reduction_ranges
        )
        contribution = self._evaluate(data.inner_fn, symbols, reduction_symbols)
        output_index = self._output_index(buffer, symbols)
        accumulator = Tensor(Var(f"acc_{self._accumulator_index}"))
        self._accumulator_index += 1
        accumulator_load = Load(accumulator, output_index)
        update = Store(
            accumulator,
            output_index,
            BinaryOp("+", accumulator_load, contribution),
        )
        reduction_loop = _nest_loops(reduction_variables, data.reduction_ranges, update)
        body = seq(
            Store(accumulator, output_index, Num(0)),
            reduction_loop,
            Store(
                self._output_tensor(buffer),
                output_index,
                Load(accumulator, output_index),
            ),
        )
        return _nest_loops(variables, data.ranges, body)

    def operation(self, operation: ir.Operation) -> Stmt:
        if isinstance(operation, ir.ComputedBuffer):
            if type(operation.data) is ir.Pointwise:
                return self._pointwise(operation)
            if type(operation.data) is ir.Reduction:
                return self._reduction(operation)
        raise UnsupportedNodeError(f"unsupported operation {type(operation).__name__}")

    def build(self) -> Stmt:
        operations = tuple(
            self.operation(operation) for operation in self.graph.operations
        )
        return Dummy() if not operations else seq(*operations)


def build_graph_lowering_ast(graph: GraphLowering) -> Stmt:
    return GraphLoweringToScalarAST(graph).build()


__all__ = [
    "GraphLoweringToScalarAST",
    "UnsupportedNodeError",
    "build_graph_lowering_ast",
    "sympy_to_expr",
]
