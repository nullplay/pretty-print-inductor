from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import Scheduler

from pretty_print_loop.fused_scheduler import format_fused_scheduler
from pretty_print_loop.fusion_trace import capture_fusion_steps, format_fusion_trace
from pretty_print_loop.loop_ir import format_post_lowering
from pretty_print_loop.lowering import graph_context, lower_callable

BETTER_BENCHMARK_ROOT = Path(
    os.environ.get(
        "BETTER_BENCHMARK_ROOT",
        Path(__file__).resolve().parents[2] / "better-benchmark",
    )
)
CANONICAL_ROOT = BETTER_BENCHMARK_ROOT / "repros" / "canonical"


@dataclass(frozen=True)
class ReproRenderings:
    post_lowering: str
    fused_scheduler: str
    fusion_trace: str


def _load_repro(repro_id: str) -> ModuleType:
    path = CANONICAL_ROOT / repro_id / "repro.py"
    spec = importlib.util.spec_from_file_location(f"repro_{repro_id}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lower_repro(repro_id: str) -> GraphLowering:
    module = _load_repro(repro_id)

    # Match benchmark_repro()'s default input path. The generated make_inputs()
    # reconstructs recorded shape, stride, storage offset, dtype, and aliases.
    from repro_harness import make_inputs_safely

    all_inputs = make_inputs_safely(module.make_inputs)
    tensor_inputs = [value for value in all_inputs if isinstance(value, torch.Tensor)]
    model = module.Repro()

    def bind_static_shape_parameters(*tensors):
        tensor_iter = iter(tensors)
        args = [
            next(tensor_iter) if isinstance(value, torch.Tensor) else value
            for value in all_inputs
        ]
        return model(*args)

    return lower_callable(bind_static_shape_parameters, *tensor_inputs)


def render_repro(
    repro_id: str,
    *,
    include_raw_ir: bool = False,
    include_scheduler_ir: bool = False,
) -> ReproRenderings:
    graph = lower_repro(repro_id)
    post_lowering = format_post_lowering(graph, include_raw_ir=include_raw_ir)
    with graph_context(graph):
        with capture_fusion_steps(
            include_scheduler_ir=include_scheduler_ir,
            original_loop_ir=post_lowering,
        ) as trace:
            scheduler = Scheduler(graph.operations)
        fused_scheduler = format_fused_scheduler(
            scheduler, include_raw_ir=include_raw_ir
        )
    fusion_trace = format_fusion_trace(
        trace, include_scheduler_ir=include_scheduler_ir
    )
    return ReproRenderings(post_lowering, fused_scheduler, fusion_trace)


def trace_repro(
    repro_id: str, *, include_scheduler_ir: bool = False
) -> str:
    return render_repro(
        repro_id, include_scheduler_ir=include_scheduler_ir
    ).fusion_trace
