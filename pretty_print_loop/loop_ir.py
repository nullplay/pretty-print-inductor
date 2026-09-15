from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.common import deduce_output_dtype_by_name
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import OpsValue, V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.printers import PythonPrinter

if TYPE_CHECKING:
    from torch._inductor.graph import GraphLowering


class Expr:
    pass


class Type:
    pass


@dataclasses.dataclass(frozen=True)
class ScalarType(Type):
    dtype: torch.dtype


@dataclasses.dataclass(frozen=True)
class TensorType(Type):
    dtype: torch.dtype | None
    shape: tuple[sympy.Expr, ...] | None


class StorageClass(enum.Enum):
    GLOBAL = "global"
    LOCAL = "local"


@dataclasses.dataclass(frozen=True)
class Constant(Expr):
    value: object


@dataclasses.dataclass(frozen=True)
class Variable(Expr):
    name: str
    type: Type
    storage: StorageClass

    @property
    def dtype(self) -> torch.dtype | None:
        if isinstance(self.type, (ScalarType, TensorType)):
            return self.type.dtype
        return None


@dataclasses.dataclass(frozen=True)
class TensorAccess(Expr):
    tensor: Variable
    indices: tuple[object, ...]
    mode: Literal["logical", "buffer"] = "logical"

    def __post_init__(self) -> None:
        if not isinstance(self.tensor.type, TensorType):
            raise TypeError("TensorAccess requires a tensor variable")
        if (
            self.mode == "logical"
            and self.tensor.type.shape is not None
            and len(self.indices) != len(self.tensor.type.shape)
        ):
            raise ValueError("logical access rank must match tensor rank")


@dataclasses.dataclass(frozen=True)
class Call(Expr):
    op: str
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...] = ()


@dataclasses.dataclass(frozen=True)
class Select(Expr):
    condition: object
    true_value: object
    false_value: object


class Stmt:
    pass


@dataclasses.dataclass(frozen=True)
class Declare(Stmt):
    variable: Variable
    initializer: object | None = None

    def __post_init__(self) -> None:
        if self.variable.storage is not StorageClass.LOCAL:
            raise ValueError("only local variables are declared inside a fused body")


@dataclasses.dataclass(frozen=True)
class Assign(Stmt):
    target: Expr
    value: object

    def __post_init__(self) -> None:
        if isinstance(self.target, Variable) and not isinstance(
            self.target.type, ScalarType
        ):
            raise TypeError("direct assignment requires a scalar variable")
        if not isinstance(self.target, (Variable, TensorAccess)):
            raise TypeError("assignment target must be a scalar variable or access")


@dataclasses.dataclass(frozen=True)
class Update(Stmt):
    target: Expr
    operator: str
    value: object

    def __post_init__(self) -> None:
        if isinstance(self.target, Variable) and not isinstance(
            self.target.type, ScalarType
        ):
            raise TypeError("direct update requires a scalar variable")
        if not isinstance(self.target, (Variable, TensorAccess)):
            raise TypeError("update target must be a scalar variable or access")


@dataclasses.dataclass(frozen=True)
class For(Stmt):
    variable: sympy.Symbol
    extent: sympy.Expr
    body: tuple[Stmt, ...]


@dataclasses.dataclass(frozen=True)
class Unimplemented(Stmt):
    operation_type: str


@dataclasses.dataclass(frozen=True)
class Fused:
    name: str | None
    inputs: tuple[Variable, ...]
    outputs: tuple[Variable, ...]
    body: tuple[Stmt, ...]
    source_ir: str | None = None

    def __post_init__(self) -> None:
        for variable in (*self.inputs, *self.outputs):
            if variable.storage is not StorageClass.GLOBAL:
                raise ValueError("fused inputs and outputs must be global")
            if not isinstance(variable.type, TensorType):
                raise TypeError("fused inputs and outputs must be tensors")


@dataclasses.dataclass(frozen=True)
class Program:
    fused: tuple[Fused, ...]
    returns: tuple[object, ...]

    def __post_init__(self) -> None:
        for value in self.returns:
            if (
                isinstance(value, Variable)
                and value.storage is StorageClass.LOCAL
            ):
                raise ValueError("local variables cannot escape a program")


_BINARY_OPERATORS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "lshift": "<<",
    "rshift": ">>",
    "bitwise_and": "&",
    "bitwise_or": "|",
    "bitwise_xor": "^",
    "logical_and": "and",
    "logical_or": "or",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "gt": ">",
    "le": "<=",
    "ge": ">=",
}

_UNARY_OPERATORS = {
    "neg": "-",
    "logical_not": "not ",
    "bitwise_not": "~",
}

_INDEX_PRINTER = PythonPrinter()


def _render_index(index: sympy.Expr) -> str:
    return _INDEX_PRINTER.doprint(index)


def _unwrap(value: Any) -> Any:
    if isinstance(value, OpsValue):
        return value.value
    if isinstance(value, tuple):
        return tuple(_unwrap(x) for x in value)
    if isinstance(value, list):
        return [_unwrap(x) for x in value]
    return value


def _local_scalar(name: str, dtype: torch.dtype) -> Variable:
    return Variable(name, ScalarType(dtype), StorageClass.LOCAL)


def _global_tensor(
    graph: GraphLowering,
    name: str,
    value: ir.IRNode | None = None,
    *,
    display_name: str | None = None,
) -> Variable:
    if value is None:
        candidate = graph.try_get_buffer(name)
        value = candidate if isinstance(candidate, ir.IRNode) else None
    if value is None or not value.has_tensor_output():
        tensor_type = TensorType(None, None)
    else:
        tensor_type = TensorType(
            value.get_dtype(),
            tuple(sympy.sympify(size) for size in value.get_size()),
        )
    return Variable(display_name or name, tensor_type, StorageClass.GLOBAL)


class _LogicalOpsHandler(DefaultHandler):
    name = "PostLoweringLogicalOpsHandler"

    def __init__(
        self,
        graph: GraphLowering,
        var_ranges: dict[sympy.Symbol, sympy.Expr],
        use_temporaries: bool = True,
        prefer_linearized_accesses: bool = True,
    ) -> None:
        self.graph = graph
        self.var_ranges = var_ranges
        self.reads = OrderedSet[str]()
        self.bindings: list[Stmt] = []
        self.input_renames: dict[str, str] = {}
        self.use_temporaries = use_temporaries
        self.prefer_linearized_accesses = prefer_linearized_accesses
        self._temporary_count = 0
        self._indirect_var_count = 0
        self._expression_dtypes: dict[int, torch.dtype | None] = {}

    def _record_dtype(self, value: object, dtype: torch.dtype | None) -> object:
        self._expression_dtypes[id(value)] = dtype
        return value

    def _dtype_of(self, value: object) -> torch.dtype | None:
        if isinstance(value, Variable):
            return value.dtype
        if isinstance(value, Expr):
            return self._expression_dtypes.get(id(value))
        if isinstance(value, bool):
            return torch.bool
        if isinstance(value, int):
            return torch.int64
        if isinstance(value, float):
            return torch.float64
        return None

    def _promote_args(self, args: Sequence[object]) -> torch.dtype | None:
        dtypes = [dtype for arg in args if (dtype := self._dtype_of(arg)) is not None]
        if not dtypes:
            return None
        result = dtypes[0]
        for dtype in dtypes[1:]:
            result = torch.promote_types(result, dtype)
        return result

    def _bind(self, value: object, dtype: torch.dtype | None) -> object:
        self._record_dtype(value, dtype)
        if not self.use_temporaries:
            return value
        if dtype is None:
            raise TypeError("temporary expression has no scalar dtype")
        temporary = _local_scalar(f"tmp{self._temporary_count}", dtype)
        self._temporary_count += 1
        self.bindings.append(Declare(temporary, value))
        self._record_dtype(temporary, dtype)
        return temporary

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> Expr:
        dtype = deduce_output_dtype_by_name(name, *args, **kwargs)
        if dtype is None:
            dtype = self._promote_args(args)
        if name in _BINARY_OPERATORS and len(args) == 2 and not kwargs:
            return self._bind(Call(name, args), dtype)
        if name in _UNARY_OPERATORS and len(args) == 1 and not kwargs:
            return self._bind(Call(name, args), dtype)
        if name == "maximum":
            name = "max"
        elif name == "minimum":
            name = "min"
        return self._bind(Call(name, args, tuple(kwargs.items())), dtype)

    def constant(self, value: bool | float, dtype: torch.dtype) -> Expr:
        return self._bind(Constant(value), dtype)

    def load(self, name: str, index: sympy.Expr) -> Expr:
        self.reads.add(name)
        return self._bind(self._make_access(name, index), self.graph.get_dtype(name))

    def load_seed(self, name: str, offset: object) -> Expr:
        self.reads.add(name)
        display_name = name if name.endswith("_seed") else f"{name}_seed"
        self.input_renames[name] = display_name
        return self._bind(
            TensorAccess(
                _global_tensor(self.graph, name, display_name=display_name),
                (offset,),
                mode="buffer",
            ),
            torch.int64,
        )

    def _make_access(self, name: str, index: sympy.Expr) -> Expr:
        buffer = self.graph.try_get_buffer(name)
        if buffer is None:
            return TensorAccess(
                _global_tensor(self.graph, name), (index,), mode="buffer"
            )

        try:
            layout = buffer.get_layout()
        except NotImplementedError:
            return TensorAccess(
                _global_tensor(self.graph, name, buffer),
                (index,),
                mode="buffer",
            )

        if not isinstance(layout, ir.Layout) or not self._is_dense(layout):
            return TensorAccess(
                _global_tensor(self.graph, name, buffer),
                (index,),
                mode="buffer",
            )

        delta = sympy.sympify(index) - sympy.sympify(layout.offset)
        indices: list[sympy.Expr] = []
        for size, stride in zip(layout.size, layout.stride):
            if self.graph.sizevars.statically_known_equals(size, 1):
                coordinate = sympy.S.Zero
            else:
                coordinate = ModularIndexing(delta, stride, size)
                coordinate = self.graph.sizevars.simplify_with_ranges(
                    coordinate, self.var_ranges
                )
                if isinstance(coordinate, ModularIndexing):
                    base, divisor, modulus = coordinate.args
                    quotient = base if divisor == 1 else FloorDiv(base, divisor)
                    coordinate = sympy.Mod(quotient, modulus)
            indices.append(coordinate)
        if (
            self.prefer_linearized_accesses
            and isinstance(index, sympy.Symbol)
            and any(
                isinstance(term, (FloorDiv, ModularIndexing, sympy.Mod))
                for coordinate in indices
                for term in sympy.preorder_traversal(coordinate)
            )
        ):
            return TensorAccess(
                _global_tensor(self.graph, name, buffer),
                (index,),
                mode="buffer",
            )
        return TensorAccess(_global_tensor(self.graph, name, buffer), tuple(indices))

    def _is_dense(self, layout: ir.Layout) -> bool:
        if layout.is_contiguous():
            return True
        if not all(
            isinstance(x, (int, sympy.Integer))
            for x in (*layout.size, *layout.stride)
        ):
            return False

        active_dims = [i for i, size in enumerate(layout.size) if int(size) != 1]
        active_dims.sort(key=lambda i: int(layout.stride[i]))
        expected_stride = 1
        for dim in active_dims:
            if int(layout.stride[dim]) != expected_stride:
                return False
            expected_stride *= int(layout.size[dim])
        return True

    def where(self, condition: object, input: object, other: object) -> Expr:
        dtype = self._promote_args((input, other))
        return self._bind(Select(condition, input, other), dtype)

    def masked(self, mask: object, body: Any, other: object) -> Expr:
        use_temporaries = self.use_temporaries
        self.use_temporaries = False
        try:
            value = _unwrap(body())
        finally:
            self.use_temporaries = use_temporaries
        dtype = self._promote_args((value, other))
        return self._bind(Select(mask, value, other), dtype)

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> object:
        return self._bind(expr, dtype)

    def value_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Expr:
        return self._bind(Call(_dtype_name(dtype), (expr,)), dtype)

    def identity(self, value: object) -> object:
        return value

    def to_dtype(
        self,
        value: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> Expr:
        return self._bind(Call(_dtype_name(dtype), (value,)), dtype)

    def indirect_indexing(
        self,
        index: object,
        size: sympy.Expr,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        name = f"indirect{self._indirect_var_count}"
        self._indirect_var_count += 1
        dtype = self._dtype_of(index) or torch.int64
        target = _local_scalar(name, dtype)
        size = sympy.sympify(size)
        value = (
            Select(
                Call("lt", (index, Constant(0))),
                Call("add", (index, size)),
                index,
            )
            if wrap_neg
            else index
        )
        self.bindings.append(Declare(target, value))
        self._record_dtype(target, dtype)

        variable = sympy_index_symbol(name)
        self.var_ranges[variable] = size
        return variable


def _make_index_vars(
    prefix: str, ranges: Sequence[sympy.Expr]
) -> tuple[list[sympy.Symbol], dict[sympy.Symbol, sympy.Expr]]:
    variables = [sympy_index_symbol(f"{prefix}{i}") for i in range(len(ranges))]
    return variables, dict(zip(variables, ranges))


def _make_loop_nest(
    variables: Sequence[sympy.Symbol],
    ranges: Sequence[sympy.Expr],
    body: tuple[Stmt, ...],
) -> tuple[Stmt, ...]:
    result = body
    for variable, extent in reversed(tuple(zip(variables, ranges))):
        result = (For(variable, sympy.sympify(extent), result),)
    return result


def _reduction_identity(reduction_type: str, dtype: torch.dtype) -> object:
    if reduction_type in ("sum", "dot", "xor_sum"):
        return Constant(0)
    if reduction_type == "any":
        return Constant(False)
    if reduction_type == "prod":
        return Constant(1)
    if reduction_type in ("max", "fmax"):
        if dtype.is_floating_point:
            return Constant(-math.inf)
        return Call("dtype_min", (dtype,))
    if reduction_type == "min":
        if dtype.is_floating_point:
            return Constant(math.inf)
        return Call("dtype_max", (dtype,))
    return Call("reduction_identity", (reduction_type, dtype))


def _reduction_operator(reduction_type: str) -> str:
    return {
        "sum": "+=",
        "dot": "+=",
        "prod": "*=",
        "max": "max=",
        "fmax": "max=",
        "min": "min=",
        "any": "or=",
        "xor_sum": "^=",
    }.get(reduction_type, f"{reduction_type}=")


def _output_view(output: ir.IRNode) -> ir.BaseView | None:
    """Find the logical view beneath lowering-only Tensor/Storage boxes."""
    while isinstance(output, ir.MutableBox):
        output = output.data
    return output if isinstance(output, ir.BaseView) else None


class PostLoweringFormatter:
    def __init__(
        self,
        graph: GraphLowering,
        use_temporaries: bool = True,
        include_raw_ir: bool = False,
    ) -> None:
        self.graph = graph
        self.use_temporaries = use_temporaries
        self.include_raw_ir = include_raw_ir

    def build(self) -> Program:
        with V.set_graph_handler(self.graph):
            fused = []
            for operation in self.graph.operations:
                item = self._format_operation(operation)
                if self.include_raw_ir:
                    item = dataclasses.replace(item, source_ir=repr(operation))
                fused.append(item)
            fused.extend(self._format_output_views())
            return Program(tuple(fused), self._format_returns())

    def format(self) -> str:
        return render_loop_ir(self.build())

    def _format_returns(self) -> tuple[object, ...]:
        values: list[object] = []
        for index, output in enumerate(self.graph.graph_outputs):
            if (output_view := _output_view(output)) is not None:
                values.append(
                    _global_tensor(self.graph, f"output{index}", output_view)
                )
            elif isinstance(output, (ir.MutableBox, ir.Buffer)):
                values.append(
                    _global_tensor(self.graph, output.get_name(), output)
                )
            elif isinstance(output, sympy.Expr):
                values.append(output)
            elif output is None or isinstance(output, (bool, float, int)):
                values.append(Constant(output))
            else:
                values.append(Call("unimplemented_output", (type(output).__name__,)))
        return tuple(values)

    def _input_variables(
        self, names: Sequence[str], renames: dict[str, str] | None = None
    ) -> tuple[Variable, ...]:
        renames = renames or {}
        variables = []
        for name in names:
            variables.append(
                _global_tensor(
                    self.graph,
                    name,
                    display_name=renames.get(name, name),
                )
            )
        return tuple(variables)

    def _format_operation(self, operation: ir.Operation) -> Fused:
        if isinstance(operation, ir.ComputedBuffer):
            if type(operation.data) is ir.Pointwise:
                return self._format_pointwise(operation)
            if type(operation.data) is ir.Reduction:
                return self._format_reduction(operation)
            if type(operation.data) is ir.WelfordReduction:
                return self._format_welford_reduction(operation)

        return self._format_unimplemented(operation)

    def _format_unimplemented(self, operation: ir.Operation) -> Fused:
        inputs = self._input_variables(tuple(operation.get_read_names()))
        outputs = tuple(
            _global_tensor(self.graph, output.get_name(), output)
            for output in operation.get_outputs()
        )
        operation_type = type(operation).__name__
        if isinstance(operation, ir.ComputedBuffer):
            operation_type += f"({type(operation.data).__name__})"
        return Fused(
            operation.get_operation_name(),
            inputs,
            outputs,
            (Unimplemented(operation_type),),
        )

    @staticmethod
    def _first_reduction_iteration(
        reduction_variables: Sequence[sympy.Symbol],
    ) -> object:
        if not reduction_variables:
            return Constant(True)
        result: object = Call("eq", (reduction_variables[0], Constant(0)))
        for variable in reduction_variables[1:]:
            result = Call(
                "logical_and",
                (result, Call("eq", (variable, Constant(0)))),
            )
        return result

    @staticmethod
    def _welford_reduce_update(
        value: object,
        reduction_variables: Sequence[sympy.Symbol],
        mean: Variable,
        m2: Variable,
        weight: Variable,
        dtype: torch.dtype,
        name_suffix: str = "0",
    ) -> tuple[Stmt, ...]:
        first = _local_scalar(f"first_{name_suffix}", torch.bool)
        delta = _local_scalar(f"delta_{name_suffix}", dtype)
        next_weight = _local_scalar(f"weight_next_{name_suffix}", dtype)
        next_mean = _local_scalar(f"mean_next_{name_suffix}", dtype)
        next_m2 = _local_scalar(f"m2_next_{name_suffix}", dtype)

        return (
            Declare(
                first,
                PostLoweringFormatter._first_reduction_iteration(
                    reduction_variables
                ),
            ),
            Declare(delta, Call("sub", (value, mean))),
            Declare(next_weight, Call("add", (weight, Constant(1)))),
            Declare(
                next_mean,
                Select(
                    first,
                    value,
                    Call(
                        "add",
                        (mean, Call("truediv", (delta, next_weight))),
                    ),
                ),
            ),
            Declare(
                next_m2,
                Select(
                    first,
                    Constant(0),
                    Call(
                        "add",
                        (
                            m2,
                            Call(
                                "mul",
                                (delta, Call("sub", (value, next_mean))),
                            ),
                        ),
                    ),
                ),
            ),
            Assign(mean, next_mean),
            Assign(m2, next_m2),
            Assign(weight, next_weight),
        )

    @staticmethod
    def _welford_combine_update(
        contribution: tuple[object, object, object],
        mean: Variable,
        m2: Variable,
        weight: Variable,
        dtype: torch.dtype,
        name_suffix: str = "0",
    ) -> tuple[Stmt, ...]:
        peer_mean, peer_m2, peer_weight = contribution
        delta = _local_scalar(f"delta_{name_suffix}", dtype)
        next_weight = _local_scalar(f"weight_next_{name_suffix}", dtype)
        ratio = _local_scalar(f"weight_ratio_{name_suffix}", dtype)
        next_mean = _local_scalar(f"mean_next_{name_suffix}", dtype)
        next_m2 = _local_scalar(f"m2_next_{name_suffix}", dtype)

        weighted_delta = Call(
            "mul",
            (
                Call("mul", (Call("mul", (delta, delta)), weight)),
                ratio,
            ),
        )
        return (
            Declare(
                delta,
                Select(
                    Call("eq", (mean, peer_mean)),
                    Constant(0),
                    Call("sub", (peer_mean, mean)),
                ),
            ),
            Declare(next_weight, Call("add", (weight, peer_weight))),
            Declare(
                ratio,
                Select(
                    Call("eq", (next_weight, Constant(0))),
                    Constant(0),
                    Call("truediv", (peer_weight, next_weight)),
                ),
            ),
            Declare(
                next_mean,
                Call("add", (mean, Call("mul", (delta, ratio)))),
            ),
            Declare(
                next_m2,
                Call(
                    "add",
                    (Call("add", (m2, peer_m2)), weighted_delta),
                ),
            ),
            Assign(mean, next_mean),
            Assign(m2, next_m2),
            Assign(weight, next_weight),
        )

    def _format_welford_reduction(self, buffer: ir.ComputedBuffer) -> Fused:
        data = buffer.data
        if not isinstance(data, ir.WelfordReduction):
            raise TypeError(f"expected WelfordReduction, got {type(data)}")
        if data.output_index not in (0, 1, 2):
            return self._format_unimplemented(buffer)

        variables, var_ranges = _make_index_vars("i", data.ranges)
        reduction_variables, reduction_var_ranges = _make_index_vars(
            "r", data.reduction_ranges
        )
        var_ranges.update(reduction_var_ranges)
        handler = _LogicalOpsHandler(
            self.graph,
            var_ranges,
            use_temporaries=self.use_temporaries,
        )
        with (
            V.set_graph_handler(self.graph),
            V.set_ops_handler(handler),
            patch.object(ir.FlexibleLayout, "allow_indexing", True),
        ):
            contribution = _unwrap(
                data.inner_fn(variables, reduction_variables)
            )

        mean = _local_scalar("mean_0", data.dtype)
        m2 = _local_scalar("m2_0", data.dtype)
        weight = _local_scalar("weight_0", data.dtype)
        if data.reduction_type == "welford_reduce" and not isinstance(
            contribution, tuple
        ):
            update = self._welford_reduce_update(
                contribution,
                reduction_variables,
                mean,
                m2,
                weight,
                data.dtype,
            )
        elif (
            data.reduction_type == "welford_combine"
            and isinstance(contribution, tuple)
            and len(contribution) == 3
        ):
            update = self._welford_combine_update(
                contribution,
                mean,
                m2,
                weight,
                data.dtype,
            )
        else:
            return self._format_unimplemented(buffer)

        reduction_body = _make_loop_nest(
            reduction_variables,
            data.reduction_ranges,
            (*handler.bindings, *update),
        )
        state = (mean, m2, weight)
        output_variable = _global_tensor(
            self.graph, buffer.get_name(), buffer
        )
        output = TensorAccess(output_variable, tuple(variables))
        body = _make_loop_nest(
            variables,
            data.ranges,
            (
                Declare(mean, Constant(0)),
                Declare(m2, Constant(0)),
                Declare(weight, Constant(0)),
                *reduction_body,
                Assign(output, state[data.output_index]),
            ),
        )
        return Fused(
            buffer.get_operation_name(),
            self._input_variables(
                tuple(buffer.get_read_names()), handler.input_renames
            ),
            (output_variable,),
            body,
        )

    def _format_pointwise(self, buffer: ir.ComputedBuffer) -> Fused:
        data = buffer.data
        if not isinstance(data, ir.Pointwise):
            raise TypeError(f"expected Pointwise, got {type(data)}")
        variables, var_ranges = _make_index_vars("i", data.ranges)
        handler = _LogicalOpsHandler(
            self.graph,
            var_ranges,
            use_temporaries=self.use_temporaries,
        )
        with (
            V.set_graph_handler(self.graph),
            V.set_ops_handler(handler),
            patch.object(ir.FlexibleLayout, "allow_indexing", True),
        ):
            value = _unwrap(data.inner_fn(variables))

        output_variable = _global_tensor(
            self.graph, buffer.get_name(), buffer
        )
        target = TensorAccess(output_variable, tuple(variables))
        body = _make_loop_nest(
            variables,
            data.ranges,
            (*handler.bindings, Assign(target, value)),
        )
        return Fused(
            buffer.get_operation_name(),
            self._input_variables(
                tuple(buffer.get_read_names()), handler.input_renames
            ),
            (output_variable,),
            body,
        )

    def _format_reduction(self, buffer: ir.ComputedBuffer) -> Fused:
        data = buffer.data
        if not isinstance(data, ir.Reduction):
            raise TypeError(f"expected Reduction, got {type(data)}")
        variables, var_ranges = _make_index_vars("i", data.ranges)
        reduction_variables, reduction_var_ranges = _make_index_vars(
            "r", data.reduction_ranges
        )
        var_ranges.update(reduction_var_ranges)
        handler = _LogicalOpsHandler(
            self.graph,
            var_ranges,
            use_temporaries=self.use_temporaries,
        )
        with (
            V.set_graph_handler(self.graph),
            V.set_ops_handler(handler),
            patch.object(ir.FlexibleLayout, "allow_indexing", True),
        ):
            value = _unwrap(data.inner_fn(variables, reduction_variables))

        accumulator = _local_scalar("acc_0", data.dtype)
        reduction_body = _make_loop_nest(
            reduction_variables,
            data.reduction_ranges,
            (
                *handler.bindings,
                Update(
                    accumulator,
                    _reduction_operator(data.reduction_type),
                    value,
                ),
            ),
        )
        output_variable = _global_tensor(
            self.graph, buffer.get_name(), buffer
        )
        output = TensorAccess(output_variable, tuple(variables))
        body = (
            Declare(
                accumulator,
                _reduction_identity(data.reduction_type, data.dtype),
            ),
            *reduction_body,
            Assign(output, accumulator),
        )
        body = _make_loop_nest(variables, data.ranges, body)
        return Fused(
            buffer.get_operation_name(),
            self._input_variables(
                tuple(buffer.get_read_names()), handler.input_renames
            ),
            (output_variable,),
            body,
        )

    def _format_output_views(self) -> list[Fused]:
        fused: list[Fused] = []
        for index, output in enumerate(self.graph.graph_outputs):
            output = _output_view(output)
            if output is None:
                continue
            variables, var_ranges = _make_index_vars("i", output.get_size())
            handler = _LogicalOpsHandler(
                self.graph,
                var_ranges,
                use_temporaries=False,
                prefer_linearized_accesses=False,
            )
            with (
                V.set_graph_handler(self.graph),
                V.set_ops_handler(handler),
                patch.object(ir.FlexibleLayout, "allow_indexing", True),
            ):
                value = _unwrap(output.make_loader()(variables))
            output_name = f"output{index}"
            output_variable = _global_tensor(
                self.graph, output_name, output
            )
            body = _make_loop_nest(
                variables,
                output.get_size(),
                (Assign(TensorAccess(output_variable, tuple(variables)), value),),
            )
            fused.append(
                Fused(
                    None,
                    self._input_variables(tuple(handler.reads)),
                    (output_variable,),
                    body,
                )
            )
        return fused


def format_post_lowering(
    graph: GraphLowering,
    *,
    use_temporaries: bool = True,
    include_raw_ir: bool = False,
) -> str:
    return PostLoweringFormatter(graph, use_temporaries, include_raw_ir).format()


def build_post_lowering_loop_ir(
    graph: GraphLowering,
    *,
    use_temporaries: bool = True,
    include_raw_ir: bool = False,
) -> Program:
    return PostLoweringFormatter(graph, use_temporaries, include_raw_ir).build()


_PRECEDENCE = {
    "or": 10,
    "and": 20,
    "==": 30,
    "!=": 30,
    "<": 30,
    ">": 30,
    "<=": 30,
    ">=": 30,
    "|": 40,
    "^": 41,
    "&": 42,
    "<<": 45,
    ">>": 45,
    "+": 50,
    "-": 50,
    "*": 60,
    "/": 60,
    "//": 60,
    "%": 60,
    "**": 70,
}

_DTYPE_NAMES = {
    torch.bool: "bool",
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.uint16: "u16",
    torch.uint32: "u32",
    torch.uint64: "u64",
}


def _dtype_name(dtype: torch.dtype) -> str:
    return _DTYPE_NAMES.get(dtype, str(dtype).removeprefix("torch."))


def _render_type(value_type: Type) -> str:
    if isinstance(value_type, ScalarType):
        return _dtype_name(value_type.dtype)
    if isinstance(value_type, TensorType):
        dtype = "?" if value_type.dtype is None else _dtype_name(value_type.dtype)
        shape = (
            "?"
            if value_type.shape is None
            else ", ".join(_render_index(size) for size in value_type.shape)
        )
        return f"{dtype}[{shape}]"
    raise AssertionError(f"unsupported type {type(value_type)}")


def _render_variable_declaration(variable: Variable) -> str:
    return f"{variable.name}: {_render_type(variable.type)}"


def _render_atom(value: object, parent_precedence: int = 0) -> str:
    if isinstance(value, Constant):
        if isinstance(value.value, float) and math.isinf(value.value):
            return "-inf" if value.value < 0 else "inf"
        return repr(value.value)
    if isinstance(value, Variable):
        return value.name
    if isinstance(value, TensorAccess):
        indices = ", ".join(_render_atom(index) for index in value.indices)
        if value.mode == "logical":
            return f"{value.tensor.name}[{indices}]"
        return f"{value.tensor.name}.buf[{indices}]"
    if isinstance(value, Call) and value.op in _UNARY_OPERATORS:
        result = f"{_UNARY_OPERATORS[value.op]}{_render_atom(value.args[0], 80)}"
        return f"({result})" if 80 < parent_precedence else result
    if isinstance(value, Call) and value.op in _BINARY_OPERATORS:
        operator = _BINARY_OPERATORS[value.op]
        precedence = _PRECEDENCE[operator]
        left = _render_atom(value.args[0], precedence)
        right_precedence = precedence + (
            operator not in ("+", "*", "and", "or")
        )
        right = _render_atom(value.args[1], right_precedence)
        result = f"{left} {operator} {right}"
        return f"({result})" if precedence < parent_precedence else result
    if isinstance(value, Call):
        arguments = [_render_atom(x) for x in value.args]
        arguments.extend(
            f"{key}={_render_atom(item)}" for key, item in value.kwargs
        )
        return f"{value.op}({', '.join(arguments)})"
    if isinstance(value, Select):
        return (
            f"select({_render_atom(value.condition)}, "
            f"{_render_atom(value.true_value)}, {_render_atom(value.false_value)})"
        )
    if isinstance(value, sympy.Expr):
        return _render_index(value)
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, tuple):
        suffix = "," if len(value) == 1 else ""
        return f"({', '.join(_render_atom(x) for x in value)}{suffix})"
    if isinstance(value, list):
        return f"[{', '.join(_render_atom(x) for x in value)}]"
    return repr(value)


class _Renderer:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def render_fused(self, fused: Fused) -> str:
        if fused.source_ir is not None:
            self.lines.append("# original Inductor IR:")
            self.lines.extend(f"# {line}" for line in fused.source_ir.splitlines())
        inputs = ", ".join(
            _render_variable_declaration(value) for value in fused.inputs
        )
        outputs = ", ".join(
            _render_variable_declaration(value) for value in fused.outputs
        )
        if len(fused.outputs) > 1:
            outputs = f"({outputs})"
        prefix = f"region {fused.name}" if fused.name is not None else "no_kernel"
        self.lines.append(f"{prefix}({inputs}) -> {outputs}:")
        self.render_block(fused.body, 1)
        return "\n".join(self.lines)

    @staticmethod
    def render_program(program: Program) -> str:
        rendered_fused = [_Renderer().render_fused(fused) for fused in program.fused]
        values = ", ".join(_render_atom(value) for value in program.returns)
        if len(program.returns) == 1:
            values += ","
        rendered_fused.append(f"return ({values})")
        return "\n\n".join(rendered_fused)

    def render_block(self, statements: Sequence[Stmt], indent: int) -> None:
        if not statements:
            self.lines.append(f"{'    ' * indent}pass")
            return
        for statement in statements:
            self.render_statement(statement, indent)

    def render_statement(self, statement: Stmt, indent: int) -> None:
        prefix = "    " * indent
        if isinstance(statement, Declare):
            declaration = _render_variable_declaration(statement.variable)
            initializer = (
                ""
                if statement.initializer is None
                else f" = {_render_atom(statement.initializer)}"
            )
            self.lines.append(f"{prefix}{declaration}{initializer}")
        elif isinstance(statement, Assign):
            self.lines.append(
                f"{prefix}{_render_atom(statement.target)} = "
                f"{_render_atom(statement.value)}"
            )
        elif isinstance(statement, Update):
            self.lines.append(
                f"{prefix}{_render_atom(statement.target)} {statement.operator} "
                f"{_render_atom(statement.value)}"
            )
        elif isinstance(statement, For):
            self.lines.append(
                f"{prefix}for {_render_index(statement.variable)} in "
                f"[0, {_render_index(statement.extent)}):"
            )
            self.render_block(statement.body, indent + 1)
        elif isinstance(statement, Unimplemented):
            self.lines.append(f"{prefix}unimplemented {statement.operation_type}")
        else:
            raise TypeError(f"unsupported statement {type(statement)}")


def render_loop_ir(node: Program | Fused | Stmt | Expr) -> str:
    if isinstance(node, Program):
        return _Renderer.render_program(node)
    if isinstance(node, Fused):
        return _Renderer().render_fused(node)
    if isinstance(node, Expr):
        return _render_atom(node)
    renderer = _Renderer()
    renderer.render_statement(node, 0)
    return "\n".join(renderer.lines)
