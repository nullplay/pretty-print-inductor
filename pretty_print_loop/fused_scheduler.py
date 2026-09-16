from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import sympy
from torch._inductor.codegen.simd import SIMDKernel, SIMDScheduling
from torch._inductor.codegen.simd_kernel_features import (
    DisableReduction,
    EnableReduction,
)
from torch._inductor.scheduler import (
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import ModularIndexing

from .loop_ir import (
    Assign,
    Constant,
    Declare,
    Fused,
    PostLoweringFormatter,
    Program,
    Stmt,
    Unimplemented,
    Update,
    _build_scan_update,
    _global_tensor,
    _local_scalar,
    _LogicalOpsHandler,
    _make_index_vars,
    _make_loop_nest,
    _reduction_identity,
    _reduction_operator,
    _unwrap,
    render_loop_ir,
)


def _product(values: Iterable[sympy.Expr]) -> sympy.Expr:
    return sympy.prod(values)


def _flatten(
    variables: Sequence[sympy.Expr], sizes: Sequence[sympy.Expr]
) -> sympy.Expr:
    result = sympy.S.Zero
    for variable, size in zip(variables, sizes):
        result = result * size + variable
    return result


def _unflatten(
    index: sympy.Expr, sizes: Sequence[sympy.Expr]
) -> list[sympy.Expr]:
    return [
        sympy.S.Zero
        if V.graph.sizevars.statically_known_equals(size, 1)
        else V.graph.sizevars.simplify(
            ModularIndexing(index, _product(sizes[position + 1 :]), size)
        )
        for position, size in enumerate(sizes)
    ]


class _LogicalRangeMapper:
    def __init__(
        self,
        pointwise_variables: Sequence[sympy.Expr],
        pointwise_sizes: Sequence[sympy.Expr],
        reduction_variables: Sequence[sympy.Expr],
        reduction_sizes: Sequence[sympy.Expr],
    ) -> None:
        self.pointwise_flat = _flatten(pointwise_variables, pointwise_sizes)
        self.reduction_flat = _flatten(reduction_variables, reduction_sizes)

    def _set_ranges(
        self, *factor_groups: Sequence[sympy.Expr]
    ) -> list[list[sympy.Expr]]:
        common_indices = (self.pointwise_flat, self.reduction_flat)
        return [
            _unflatten(index, factors)
            for index, factors in zip(common_indices, factor_groups)
        ]

    def map_child(
        self,
        child: SchedulerNode,
        pointwise_numel: sympy.Expr,
        reduction_numel: sympy.Expr,
        *,
        reduction_active: bool,
    ) -> list[list[sympy.Expr]]:
        groups = (
            pointwise_numel,
            reduction_numel if reduction_active else sympy.S.One,
        )
        return SIMDKernel.map_kernel_groups_to_node_sizes(
            groups,
            child.get_ranges(),
            self._set_ranges,
        )


class _Phase:
    def __init__(self, reduction_active: bool) -> None:
        self.reduction_active = reduction_active
        self.nodes: list[SchedulerNode] = []


def _split_phases(node_schedule, has_reduction: bool) -> list[_Phase]:
    phases = [_Phase(has_reduction)]

    for entry in node_schedule:
        if entry is DisableReduction:
            if phases[-1].nodes:
                phases.append(_Phase(False))
            else:
                phases[-1].reduction_active = False
        elif entry is EnableReduction:
            if phases[-1].nodes:
                phases.append(_Phase(True))
            else:
                phases[-1].reduction_active = True
        else:
            if not isinstance(entry, SchedulerNode):
                raise TypeError(f"unsupported schedule entry: {type(entry)}")
            phases[-1].nodes.append(entry)

    return [phase for phase in phases if phase.nodes]


class _FusedLogicalOpsHandler(_LogicalOpsHandler):
    def __init__(
        self, graph, var_ranges, reduction_variables, materialized_buffers
    ) -> None:
        super().__init__(graph, var_ranges, use_temporaries=True)
        self.reduction_variables = tuple(reduction_variables)
        self.materialized_buffers = materialized_buffers
        self.current: list[Stmt] = []
        self.initializers: list[Stmt] = []
        self.finalizers: list[Stmt] = []
        # Ordinary stores may only be forwarded inside one schedule phase.
        # Finalized reductions can additionally feed a later epilogue phase.
        self.forwarded: dict[tuple[str, str], object] = {}
        self.forwarded_reductions: dict[tuple[str, str], object] = {}
        self.accumulator_count = itertools.count()
        self.scan_state_count = itertools.count()

    def start_phase(self, statements: list[Stmt]) -> None:
        self.current = statements
        self.initializers = []
        self.finalizers = []
        self.forwarded.clear()

    def flush_bindings(self) -> None:
        self.current.extend(self.bindings)
        self.bindings.clear()

    def _key(self, name: str, index: sympy.Expr) -> tuple[str, str]:
        index = self.graph.sizevars.simplify_with_ranges(index, self.var_ranges)
        index = sympy.expand(index)
        return name, sympy.srepr(index)

    def load(self, name: str, index: sympy.Expr):
        forwarded = self.forwarded.get(self._key(name, index))
        if forwarded is None:
            forwarded = self.forwarded_reductions.get(self._key(name, index))
        if forwarded is not None:
            return forwarded
        return super().load(name, index)

    def store(self, name, index, value, mode=None):
        value = _unwrap(value)
        self.flush_bindings()
        self.forwarded[self._key(name, index)] = value
        target = self._make_access(name, index)

        if mode == "atomic_add":
            self.current.append(Update(target, "+=", value))
        elif name in self.materialized_buffers:
            self.current.append(Assign(target, value))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        value = _unwrap(value)
        self.flush_bindings()

        accumulator_index = next(self.accumulator_count)
        if reduction_type in ("welford_reduce", "welford_combine"):
            mean = _local_scalar(f"mean_{accumulator_index}", dtype)
            m2 = _local_scalar(f"m2_{accumulator_index}", dtype)
            weight = _local_scalar(f"weight_{accumulator_index}", dtype)
            self.initializers.extend(
                (
                    Declare(mean, Constant(0)),
                    Declare(m2, Constant(0)),
                    Declare(weight, Constant(0)),
                )
            )
            if reduction_type == "welford_reduce" and not isinstance(
                value, tuple
            ):
                self.current.extend(
                    PostLoweringFormatter._welford_reduce_update(
                        value,
                        self.reduction_variables,
                        mean,
                        m2,
                        weight,
                        dtype,
                        str(accumulator_index),
                    )
                )
            elif (
                reduction_type == "welford_combine"
                and isinstance(value, tuple)
                and len(value) == 3
            ):
                self.current.extend(
                    PostLoweringFormatter._welford_combine_update(
                        value,
                        mean,
                        m2,
                        weight,
                        dtype,
                        str(accumulator_index),
                    )
                )
            else:
                raise TypeError(f"invalid {reduction_type} contribution")
            return mean, m2, weight

        accumulator = _local_scalar(f"acc_{accumulator_index}", dtype)
        self.initializers.append(
            Declare(accumulator, _reduction_identity(reduction_type, dtype))
        )
        self.current.append(
            Update(accumulator, _reduction_operator(reduction_type), value)
        )
        return accumulator

    def scan(self, dtypes, combine_fn, values):
        state_indices = tuple(next(self.scan_state_count) for _ in dtypes)
        states, initializers, updates = _build_scan_update(
            self,
            dtypes,
            combine_fn,
            values,
            self.reduction_variables,
            state_indices,
        )
        self.initializers.extend(initializers)
        self.current.extend(updates)
        return states

    def store_reduction(self, name, index, value):
        value = _unwrap(value)
        self.flush_bindings()
        self.forwarded_reductions[self._key(name, index)] = value
        if name in self.materialized_buffers:
            self.finalizers.append(Assign(self._make_access(name, index), value))


def _get_simd_backend(node: FusedSchedulerNode) -> SIMDScheduling:
    backend = node.scheduler.get_backend(node.get_device())
    choose = getattr(backend, "choose_node_backend", None)
    if choose is not None:
        backend = choose(node)
    if not isinstance(backend, SIMDScheduling):
        raise TypeError(f"unsupported fused backend: {type(backend).__name__}")
    return backend


def _materialized_buffer_names(
    children: Sequence[SchedulerNode],
) -> OrderedSet[str]:
    # Conservative first version. Local-only store removal is independent of
    # recovering the fused loop structure and can be added later.
    return OrderedSet(
        output.get_name()
        for child in children
        for output in child.get_outputs()
    )


def build_fused_node_loop_ir(
    node: FusedSchedulerNode, *, include_raw_ir: bool = False
) -> Fused:
    if type(node) is not FusedSchedulerNode:
        return Fused(
            node.get_name(),
            (),
            (),
            (Unimplemented(type(node).__name__),),
            source_ir=node.debug_str() if include_raw_ir else None,
        )

    children = list(node.get_nodes())
    if not children or not all(isinstance(child, SchedulerNode) for child in children):
        raise TypeError("ordinary fused nodes must contain SchedulerNode children")

    backend = _get_simd_backend(node)
    parent = max(children, key=lambda child: int(child.is_reduction()))
    pointwise_numel, reduction_numel = parent.group[1]
    node_schedule = backend.generate_node_schedule(
        children, pointwise_numel, reduction_numel
    )

    pointwise_sizes, reduction_sizes = parent.get_ranges()
    pointwise_variables, pointwise_ranges = _make_index_vars(
        "p", pointwise_sizes
    )
    reduction_variables, reduction_ranges = _make_index_vars(
        "r", reduction_sizes
    )
    var_ranges = {**pointwise_ranges, **reduction_ranges}
    mapper = _LogicalRangeMapper(
        pointwise_variables,
        pointwise_sizes,
        reduction_variables,
        reduction_sizes,
    )

    materialized = _materialized_buffer_names(children)
    handler = _FusedLogicalOpsHandler(
        V.graph, var_ranges, reduction_variables, materialized
    )
    outer_statements: list[Stmt] = []

    for phase in _split_phases(node_schedule, reduction_numel != 1):
        phase_statements: list[Stmt] = []
        handler.start_phase(phase_statements)

        for child in phase.nodes:
            indices = mapper.map_child(
                child,
                pointwise_numel,
                reduction_numel,
                reduction_active=phase.reduction_active,
            )
            with V.set_ops_handler(handler):
                child._body(*indices)

        handler.flush_bindings()

        if phase.reduction_active and reduction_numel != 1:
            outer_statements.extend(handler.initializers)
            outer_statements.extend(
                _make_loop_nest(
                    reduction_variables,
                    reduction_sizes,
                    tuple(phase_statements),
                )
            )
            outer_statements.extend(handler.finalizers)
        else:
            outer_statements.extend(phase_statements)

    written_names = OrderedSet(
        output.get_name()
        for child in children
        for output in child.get_outputs()
    )
    inputs = tuple(
        _global_tensor(V.graph, name)
        for name in handler.reads
        if name not in written_names
    )
    outputs = tuple(
        _global_tensor(V.graph, name)
        for name in written_names
        if name in materialized
    )

    return Fused(
        node.get_name(),
        inputs,
        outputs,
        _make_loop_nest(
            pointwise_variables,
            pointwise_sizes,
            tuple(outer_statements),
        ),
        source_ir=node.debug_str() if include_raw_ir else None,
    )


def build_fused_scheduler_loop_ir(
    scheduler: Scheduler, *, include_raw_ir: bool = False
) -> Program:
    graph_formatter = PostLoweringFormatter(V.graph)
    regions = []
    for node in scheduler.nodes:
        if isinstance(node, FusedSchedulerNode):
            regions.append(
                build_fused_node_loop_ir(node, include_raw_ir=include_raw_ir)
            )
        elif isinstance(node, SchedulerNode) and node.node is not None:
            regions.append(graph_formatter._format_operation(node.node))
        else:
            regions.append(
                Fused(
                    node.get_name(),
                    (),
                    (),
                    (Unimplemented(type(node).__name__),),
                    source_ir=node.debug_str() if include_raw_ir else None,
                )
            )
    regions.extend(graph_formatter._format_output_views())
    return Program(tuple(regions), graph_formatter._format_returns())


def format_fused_scheduler(
    scheduler: Scheduler, *, include_raw_ir: bool = False
) -> str:
    return render_loop_ir(
        build_fused_scheduler_loop_ir(
            scheduler, include_raw_ir=include_raw_ir
        )
    )
