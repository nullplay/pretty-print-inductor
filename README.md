# Inductor loop IR pretty-printer

This directory contains an out-of-tree prototype for rendering Inductor's
post-GraphLowering and post-fusion scheduler IR as logical nested loops. It
does not modify or register anything in PyTorch.

The post-lowering entry point is:

```python
from pretty_print_loop import format_post_lowering

text = format_post_lowering(graph_lowering)
```

Pass `include_raw_ir=True` to place the exact multiline `repr()` of every
original Inductor operation in comments immediately above its corresponding
pretty-printed region. Output-only `no_kernel` views have no operation comment.

By default, each scalar operation is retained as an indexed temporary:

```text
tmp0: f32 = a[i, j]
tmp1: f32 = sin(tmp0)
buf0[i, j] = tmp1
```

Pass `use_temporaries=False` to render fully inlined expressions instead.
Fused arguments and outputs carry compact dtype/shape declarations, and scalar
temporaries carry compact dtype annotations such as `f32`, `f64`, and `bf16`.

Call it after `GraphLowering.run(...)` and before constructing `Scheduler`.
The formatter currently handles:

- `ComputedBuffer` containing `Pointwise` or `Reduction`;
- logical multidimensional accesses for dense layouts;
- indirect indexing with Python-style negative-index normalization;
- no-kernel view outputs such as permute, transpose, reshape, squeeze,
  unsqueeze, and expand; and
- `tensor.buf[address]` fallback for layouts that cannot be decoded safely.

Other top-level Inductor operations emit `unimplemented <type>` rather than
attempting a potentially misleading rendering. `WelfordReduction` supports
both scalar `welford_reduce` and tuple-valued `welford_combine`; other
multi-output reduction types remain unimplemented.

The final `return (...)` records graph-output slots. A direct
`StorageBox(ComputedBuffer(bufN))` returns `bufN`; a `ReinterpretView` keeps its
`no_kernel` mapping and the return references the corresponding `outputN`.

The formatter first builds a small AST. Every variable has a scalar or tensor
type and a `global` or `local` storage class. Region inputs and outputs are
global tensors; scalar temporaries and reduction accumulators are local.
Storage classes remain available to transformations and validation in the AST,
but are intentionally omitted from pretty-printed declarations.
`Declare` introduces local scalars or tensors, and a tensor `Access` is a scalar
lvalue, so both `cache[i] = value` and `cache[i] += value` are representable.
Other statements include `Assign`, `Update`, and serial `For`; ordered statement
blocks are plain tuples rather than a separate AST node. Rendering is separate.

## Fused scheduler nodes

Constructing `Scheduler(graph.operations)` runs Inductor's fusion passes. The
final schedule can be rendered with:

```python
from torch._inductor.scheduler import Scheduler

from pretty_print_loop import format_fused_scheduler
from pretty_print_loop.lowering import graph_context

with graph_context(graph):
    scheduler = Scheduler(graph.operations)
    text = format_fused_scheduler(scheduler)
```

The fused formatter asks Inductor's SIMD backend for
`generate_node_schedule(...)`. It therefore uses the same child ordering and
reduction enable/disable boundaries that code generation sees, then replays
each child through the existing logical ops handler. Exact producer stores are
forwarded to consumer loads within one phase; finalized reduction values may
also be forwarded into a later epilogue. This is deliberately store forwarding,
not general common-subexpression elimination.

Ordinary reductions and Welford reductions are printed as explicit scalar
accumulators. Specialized scheduler subclasses currently produce an explicit
`unimplemented` body instead of guessing their loop structure.

To observe every successful rewrite, the capture context must surround
`Scheduler(...)` construction:

```python
from pretty_print_loop import capture_fusion_steps, format_fusion_trace

with graph_context(graph), capture_fusion_steps() as trace:
    scheduler = Scheduler(graph.operations)

print(format_fusion_trace(trace))
```

Every trace starts with the original post-GraphLowering `op0`, `op1`, ... loop
regions. Its fusion history then records each pass's input nodes, every
`lhs + rhs -> result` rewrite, the immediately reconstructed Loop IR, and the
final live-node list. Both sections are snapshotted before later
loop-reordering passes can mutate child nodes.

The source is split by responsibility:

- `loop_ir.py`: AST, post-lowering replay, and text rendering;
- `fused_scheduler.py`: final `SchedulerNode`/`FusedSchedulerNode` replay;
- `fusion_trace.py`: scoped fusion-pass instrumentation;
- `lowering.py`: shared FX-to-GraphLowering setup.

All unit tests, corpus support code, and expected files live under `tests/`:

- `tests/test_unit.py`: post-lowering Loop IR tests;
- `tests/test_fused_scheduler.py`: final fusion and complete fusion-trace tests;
- `tests/benchmark_harness.py`: BetterBenchmark loading and rendering; and
- `tests/expected/`: complete post-lowering, fusion-trace, and final-scheduler
  goldens.

Run its tests with:

```shell
cd pretty-print-inductor
python -m tests.test_unit -v
python -m tests.test_fused_scheduler -v
python -m tests.test_better_benchmark -v
```

The corpus-backed tests load ten canonical repros from a sibling
`better-benchmark` checkout without modifying it. Set
`BETTER_BENCHMARK_ROOT=/path/to/better-benchmark` when it is elsewhere. Every
case compares its post-lowering IR, complete fusion history, and final
scheduler IR against `tests/expected/post_lowering/`,
`tests/expected/fusion_trace/`, and `tests/expected/fused_scheduler/`. Tests are
skipped when the corpus or CUDA is unavailable. Inputs are produced through
BetterBenchmark's
`make_inputs_safely(repro.make_inputs)` path, which reconstructs recorded
shapes, strides, storage offsets, dtypes, and alias groups. Before invoking
`GraphLowering` directly, the tests also install the original-output stride and
user-visible-output metadata normally supplied by `compile_fx`.
