from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch._inductor.compile_fx import record_original_output_strides
from torch._inductor.debug import DebugContext
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx


def mark_user_visible_outputs(gm: torch.fx.GraphModule) -> None:
    output_node = gm.graph.find_nodes(op="output")[0]
    record_original_output_strides(gm)
    output_values = (
        output_node.args
        if isinstance(output_node.args[0], torch.fx.Node)
        else output_node.args[0]
    )
    output_node.meta["user_visible_output_idxs"] = [
        index
        for index, value in enumerate(output_values)
        if isinstance(value, torch.fx.Node)
    ]


@contextlib.contextmanager
def graph_context(graph: GraphLowering) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        stack.enter_context(V.set_graph_handler(graph))
        stack.enter_context(V.set_debug_handler(DebugContext()))
        stack.enter_context(V.set_extern_kernel_nodes([]))
        yield


def lower_callable(
    fn: Callable[..., Any], *inputs: torch.Tensor
) -> GraphLowering:
    gm = make_fx(fn, decomposition_table=select_decomp_table())(*inputs)
    mark_user_visible_outputs(gm)
    graph = GraphLowering(gm, example_inputs=inputs)
    with graph_context(graph):
        graph.run(*inputs)
    return graph
