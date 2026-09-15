from __future__ import annotations

import contextlib
import dataclasses
import itertools
from collections.abc import Iterator, Sequence
from unittest.mock import patch

from torch._inductor.scheduler import BaseSchedulerNode, Scheduler
from torch._inductor.virtualized import V

from .fused_scheduler import build_fused_node_loop_ir
from .loop_ir import format_post_lowering, render_loop_ir


@dataclasses.dataclass
class FusionStep:
    lhs: str
    rhs: str
    result: str
    live_nodes: tuple[str, ...]
    scheduler_ir: str
    loop_ir: str


@dataclasses.dataclass
class FusionPass:
    index: int
    graph_id: int | None
    is_reorder: bool
    before_nodes: tuple[str, ...]
    after_nodes: tuple[str, ...] = ()
    steps: list[FusionStep] = dataclasses.field(default_factory=list)
    before_scheduler_ir: str | None = None
    after_scheduler_ir: str | None = None


@dataclasses.dataclass
class FusionTrace:
    original_loop_ir: str | None = None
    passes: list[FusionPass] = dataclasses.field(default_factory=list)


def _ordered_nodes(nodes: Sequence[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    return sorted(
        nodes,
        key=lambda node: (
            node.min_order,
            node.max_order,
            node.get_name(),
        ),
    )


def _node_names(nodes: Sequence[BaseSchedulerNode]) -> tuple[str, ...]:
    return tuple(node.get_name() for node in _ordered_nodes(nodes))


def _dump_nodes(nodes: Sequence[BaseSchedulerNode]) -> str:
    return "\n\n\n".join(node.debug_str() for node in _ordered_nodes(nodes))


@contextlib.contextmanager
def capture_fusion_steps(
    *,
    include_scheduler_ir: bool = False,
    original_loop_ir: str | None = None,
) -> Iterator[FusionTrace]:
    """Capture successful fusion rewrites while Scheduler is being constructed.

    This is a process-wide debugging patch and is not intended for concurrent
    compilation. Intermediate nodes are serialized immediately because later
    loop-reordering passes mutate their child SchedulerNodes.
    """

    if original_loop_ir is None:
        try:
            original_loop_ir = format_post_lowering(V.graph)
        # Tracing is diagnostic: retain an explicit section even when a
        # nonstandard caller has not installed a GraphLowering handler.
        except Exception as exc:  # noqa: BLE001
            original_loop_ir = (
                "unimplemented post_graph_lowering"
                f"({type(exc).__name__}: {exc})"
            )

    trace = FusionTrace(original_loop_ir=original_loop_ir)
    pass_counter = itertools.count(1)
    pass_stack: list[FusionPass] = []
    original_once = Scheduler.fuse_nodes_once
    original_two = Scheduler.fuse_two_nodes

    def traced_fuse_nodes_once(
        scheduler: Scheduler,
        nodes: list[BaseSchedulerNode],
        is_reorder_round: bool,
    ) -> list[BaseSchedulerNode]:
        fusion_pass = FusionPass(
            index=next(pass_counter),
            graph_id=getattr(scheduler, "post_grad_graph_id", None),
            is_reorder=is_reorder_round,
            before_nodes=_node_names(nodes),
            before_scheduler_ir=(
                _dump_nodes(nodes) if include_scheduler_ir else None
            ),
        )
        trace.passes.append(fusion_pass)
        pass_stack.append(fusion_pass)
        try:
            result = original_once(scheduler, nodes, is_reorder_round)
            fusion_pass.after_nodes = _node_names(result)
            fusion_pass.after_scheduler_ir = (
                _dump_nodes(result) if include_scheduler_ir else None
            )
            return result
        finally:
            pass_stack.pop()

    def traced_fuse_two_nodes(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        fused_nodes,
    ) -> BaseSchedulerNode:
        result = original_two(scheduler, node1, node2, fused_nodes)

        try:
            loop_ir = render_loop_ir(build_fused_node_loop_ir(result))
        # A trace is diagnostic output: one unsupported intermediate must not
        # abort the compilation whose fusion history we are observing.
        except Exception as exc:  # noqa: BLE001
            loop_ir = f"unimplemented fusion_trace({type(exc).__name__}: {exc})"

        if pass_stack:
            fusion_pass = pass_stack[-1]
        else:
            fusion_pass = FusionPass(
                index=next(pass_counter),
                graph_id=getattr(scheduler, "post_grad_graph_id", None),
                is_reorder=False,
                before_nodes=(node1.get_name(), node2.get_name()),
            )
            trace.passes.append(fusion_pass)

        fusion_pass.steps.append(
            FusionStep(
                lhs=node1.get_name(),
                rhs=node2.get_name(),
                result=result.get_name(),
                live_nodes=_node_names(fused_nodes),
                scheduler_ir=result.debug_str(),
                loop_ir=loop_ir,
            )
        )
        return result

    with (
        patch.object(Scheduler, "fuse_nodes_once", traced_fuse_nodes_once),
        patch.object(Scheduler, "fuse_two_nodes", traced_fuse_two_nodes),
    ):
        yield trace


def format_fusion_trace(
    trace: FusionTrace, *, include_scheduler_ir: bool = False
) -> str:
    original_loop_ir = trace.original_loop_ir or "unavailable"
    sections = [f"post_graph_lowering:\n\n{original_loop_ir}"]
    pass_sections: list[str] = []

    for fusion_pass in trace.passes:
        pass_kind = "loop-reordering" if fusion_pass.is_reorder else "ordinary"
        lines = [
            f"fusion_pass {fusion_pass.index} ({pass_kind}):",
            f"    before: {', '.join(fusion_pass.before_nodes)}",
        ]

        if not fusion_pass.steps:
            lines.append("    no successful fusion")

        for index, step in enumerate(fusion_pass.steps, 1):
            lines.extend(
                (
                    "",
                    f"    step {index}: {step.lhs} + {step.rhs} -> {step.result}",
                    f"    live: {', '.join(step.live_nodes)}",
                    "",
                    step.loop_ir,
                )
            )
            if include_scheduler_ir:
                lines.extend(("", step.scheduler_ir))

        lines.extend(("", f"    after: {', '.join(fusion_pass.after_nodes)}"))
        if include_scheduler_ir:
            if fusion_pass.before_scheduler_ir is not None:
                lines.extend(("", "    scheduler before:", fusion_pass.before_scheduler_ir))
            if fusion_pass.after_scheduler_ir is not None:
                lines.extend(("", "    scheduler after:", fusion_pass.after_scheduler_ir))

        pass_sections.append("\n".join(lines))

    fusion_history = (
        "\n\n".join(pass_sections) if pass_sections else "no fusion passes"
    )
    sections.append(f"fusion_history:\n\n{fusion_history}")
    return "\n\n".join(sections)
