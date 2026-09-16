# Owner(s): ["module: inductor"]

import unittest

import sympy
import torch
import torch._inductor.inductor_prims
from torch._inductor import ir
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import ops
from torch.testing._internal.common_utils import TestCase, run_tests

from pretty_print_loop.loop_ir import (
    Assign,
    Call,
    Constant,
    Declare,
    For,
    Fused,
    Program,
    ScalarType,
    StorageClass,
    TensorAccess,
    TensorType,
    Update,
    Variable,
    build_post_lowering_loop_ir,
    format_post_lowering,
    render_loop_ir,
)
from pretty_print_loop.lowering import lower_callable


class TestLoopIR(TestCase):
    @staticmethod
    def format_graph(fn, *inputs):
        graph = lower_callable(fn, *inputs)
        return graph, format_post_lowering(graph)

    def test_pointwise(self):
        def fn(a, b, c):
            return torch.sin(a + b) * c

        graph, result = self.format_graph(
            fn, *(torch.randn(2, 3) for _ in range(3))
        )
        self.assertExpectedInline(
            result,
            """\
region op0(a_1: f32[2, 3], b_1: f32[2, 3], c_1: f32[2, 3]) -> buf0: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            tmp0: f32 = a_1[i0, i1]
            tmp1: f32 = b_1[i0, i1]
            tmp2: f32 = tmp0 + tmp1
            tmp3: f32 = sin(tmp2)
            tmp4: f32 = c_1[i0, i1]
            tmp5: f32 = tmp3 * tmp4
            buf0[i0, i1] = tmp5

return (buf0,)""",  # noqa: B950
        )
        program = build_post_lowering_loop_ir(graph)
        self.assertIsInstance(program.fused[0].body[0], For)
        self.assertExpectedInline(
            format_post_lowering(graph, use_temporaries=False),
            """\
region op0(a_1: f32[2, 3], b_1: f32[2, 3], c_1: f32[2, 3]) -> buf0: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            buf0[i0, i1] = sin(a_1[i0, i1] + b_1[i0, i1]) * c_1[i0, i1]

return (buf0,)""",  # noqa: B950
        )

    def test_independent_pointwise_outputs_remain_separate_regions(self):
        def fn(a):
            return torch.sin(a), torch.cos(a)

        _, result = self.format_graph(fn, torch.randn(2, 3))
        self.assertExpectedInline(
            result,
            """\
region op0(a_1: f32[2, 3]) -> buf0: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            tmp0: f32 = a_1[i0, i1]
            tmp1: f32 = sin(tmp0)
            buf0[i0, i1] = tmp1

region op1(a_1: f32[2, 3]) -> buf1: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            tmp0: f32 = a_1[i0, i1]
            tmp1: f32 = cos(tmp0)
            buf1[i0, i1] = tmp1

return (buf0, buf1)""",
        )

    def test_include_raw_inductor_ir_comments(self):
        def fn(a):
            return torch.sin(a)

        graph, _ = self.format_graph(fn, torch.randn(2, 3))
        result = format_post_lowering(graph, include_raw_ir=True)
        self.assertTrue(result.startswith("# original Inductor IR:\n# ComputedBuffer("))
        self.assertIn("#   def inner_fn(index):", result)
        self.assertIn("\nregion op0(", result)

    def test_reduction(self):
        def fn(a):
            return torch.sin(a).sum(dim=1)

        graph, result = self.format_graph(fn, torch.randn(2, 128))
        self.assertExpectedInline(
            result,
            """\
region op0(a_1: f32[2, 128]) -> buf0: f32[2]:
    for i0 in [0, 2):
        acc_0: f32 = 0
        for r0 in [0, 128):
            tmp0: f32 = a_1[i0, r0]
            tmp1: f32 = sin(tmp0)
            acc_0 += tmp1
        buf0[i0] = acc_0

return (buf0,)""",
        )
        outer_loop = build_post_lowering_loop_ir(graph).fused[0].body[0]
        self.assertIsInstance(outer_loop, For)
        self.assertIsInstance(outer_loop.body[0], Declare)
        self.assertIsInstance(outer_loop.body[1], For)
        reduction_loop = outer_loop.body[1]
        self.assertIsInstance(reduction_loop.body[2], Update)
        self.assertIsInstance(outer_loop.body[2], Assign)

    def test_scan(self):
        def first_inner_fn(index):
            return ops.constant(2.0, torch.float32)

        def second_inner_fn(index):
            return ops.constant(3.0, torch.float32)

        def combine_fn(left, right):
            return ops.add(left[0], right[0]), ops.mul(left[1], right[1])

        inner_fns = (first_inner_fn, second_inner_fn)
        data = ir.Scan(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=inner_fns[1],
            ranges=[2],
            scan_ranges=[4],
            size=[2, 4],
            combine_fn=combine_fn,
            reindex=lambda index, scan_index: [*index, *scan_index],
            reduction_hint=ir.ReductionHint.DEFAULT,
            output_index=1,
            dtypes=(torch.float32, torch.float32),
            inner_fns=inner_fns,
        )
        buffer = ir.ComputedBuffer(
            name="buf0",
            layout=ir.FixedLayout(torch.device("cpu"), torch.float32, [2, 4]),
            data=data,
        )
        buffer.operation_name = "op0"

        fx_graph = torch.fx.Graph()
        fx_graph.output(())
        graph = GraphLowering(torch.fx.GraphModule(torch.nn.Module(), fx_graph))
        graph.operations = [buffer]
        graph.graph_outputs = []
        self.assertExpectedInline(
            format_post_lowering(graph),
            """\
region op0() -> buf0: f32[2, 4]:
    for i0 in [0, 2):
        scan_0: f32
        scan_1: f32
        for r0 in [0, 4):
            tmp0: f32 = 2.0
            tmp1: f32 = 3.0
            tmp2: f32 = scan_0 + tmp0
            tmp3: f32 = scan_1 * tmp1
            next_0: f32 = select(r0 == 0, tmp0, tmp2)
            next_1: f32 = select(r0 == 0, tmp1, tmp3)
            scan_0 = next_0
            scan_1 = next_1
            buf0[i0, r0] = scan_1

return ()""",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cummax_scan_tracks_all_tuple_inputs(self):
        _, result = self.format_graph(
            lambda x: torch.cummax(x, dim=1),
            torch.randn(2, 4, device="cuda"),
        )

        self.assertIn("region op0(x_1: f32[2, 4])", result)
        self.assertIn("region op1(x_1: f32[2, 4])", result)
        self.assertNotIn("unimplemented ComputedBuffer(Scan)", result)

    def test_welford_reduce(self):
        def inner_fn(index, reduction_index):
            return ops.constant(1.0, torch.float32)

        data = ir.WelfordReduction(
            torch.device("cpu"),
            torch.float32,
            (inner_fn,),
            [2],
            [4],
            "welford_reduce",
            torch.float32,
            ir.ReductionHint.DEFAULT,
            0,
        )
        buffer = ir.ComputedBuffer(
            name="buf0",
            layout=ir.FixedLayout(torch.device("cpu"), torch.float32, [2]),
            data=data,
        )
        buffer.operation_name = "op0"

        fx_graph = torch.fx.Graph()
        fx_graph.output(())
        graph = GraphLowering(torch.fx.GraphModule(torch.nn.Module(), fx_graph))
        graph.operations = [buffer]
        graph.graph_outputs = []
        self.assertExpectedInline(
            format_post_lowering(graph),
            """\
region op0() -> buf0: f32[2]:
    for i0 in [0, 2):
        mean_0: f32 = 0
        m2_0: f32 = 0
        weight_0: f32 = 0
        for r0 in [0, 4):
            tmp0: f32 = 1.0
            first_0: bool = r0 == 0
            delta_0: f32 = tmp0 - mean_0
            weight_next_0: f32 = weight_0 + 1
            mean_next_0: f32 = select(first_0, tmp0, mean_0 + delta_0 / weight_next_0)
            m2_next_0: f32 = select(first_0, 0, m2_0 + delta_0 * (tmp0 - mean_next_0))
            mean_0 = mean_next_0
            m2_0 = m2_next_0
            weight_0 = weight_next_0
        buf0[i0] = mean_0

return ()""",
        )

    def test_welford_combine(self):
        def make_inner_fn(value):
            def inner_fn(index, reduction_index):
                return ops.constant(value, torch.float32)

            return inner_fn

        data = ir.WelfordReduction(
            torch.device("cpu"),
            torch.float32,
            (make_inner_fn(2), make_inner_fn(3), make_inner_fn(4)),
            [2],
            [4],
            "welford_combine",
            torch.float32,
            ir.ReductionHint.DEFAULT,
            1,
        )
        buffer = ir.ComputedBuffer(
            name="buf0",
            layout=ir.FixedLayout(torch.device("cpu"), torch.float32, [2]),
            data=data,
        )
        buffer.operation_name = "op0"

        fx_graph = torch.fx.Graph()
        fx_graph.output(())
        graph = GraphLowering(torch.fx.GraphModule(torch.nn.Module(), fx_graph))
        graph.operations = [buffer]
        graph.graph_outputs = []
        self.assertExpectedInline(
            format_post_lowering(graph),
            """\
region op0() -> buf0: f32[2]:
    for i0 in [0, 2):
        mean_0: f32 = 0
        m2_0: f32 = 0
        weight_0: f32 = 0
        for r0 in [0, 4):
            tmp0: f32 = 2
            tmp1: f32 = 3
            tmp2: f32 = 4
            delta_0: f32 = select(mean_0 == tmp0, 0, tmp0 - mean_0)
            weight_next_0: f32 = weight_0 + tmp2
            weight_ratio_0: f32 = select(weight_next_0 == 0, 0, tmp2 / weight_next_0)
            mean_next_0: f32 = mean_0 + delta_0 * weight_ratio_0
            m2_next_0: f32 = m2_0 + tmp1 + delta_0 * delta_0 * weight_0 * weight_ratio_0
            mean_0 = mean_next_0
            m2_0 = m2_next_0
            weight_0 = weight_next_0
        buf0[i0] = m2_0

return ()""",
        )

    def test_unrolled_reduction_is_pointwise(self):
        def fn(a):
            return torch.sin(a).sum(dim=1)

        _, result = self.format_graph(fn, torch.randn(2, 3))
        self.assertExpectedInline(
            result,
            """\
region op0(a_1: f32[2, 3]) -> buf0: f32[2]:
    for i0 in [0, 2):
        tmp0: f32 = a_1[i0, 0]
        tmp1: f32 = sin(tmp0)
        tmp2: f32 = a_1[i0, 1]
        tmp3: f32 = sin(tmp2)
        tmp4: f32 = tmp1 + tmp3
        tmp5: f32 = a_1[i0, 2]
        tmp6: f32 = sin(tmp5)
        tmp7: f32 = tmp4 + tmp6
        buf0[i0] = tmp7

return (buf0,)""",
        )

    def test_internal_flatten_uses_linear_buffer_access(self):
        def fn(a):
            mask = a > 0
            return mask, mask.reshape(-1).any()

        _, result = self.format_graph(fn, torch.randn(8, 1024))
        self.assertIn("tmp0: bool = buf0.buf[r0]", result)
        self.assertNotIn("buf0[r0 // 1024, r0 % 1024]", result)

    def test_indirect_index_wraps_negative_values(self):
        def fn(table, indices):
            return table[indices]

        _, result = self.format_graph(
            fn,
            torch.randn(8, 4),
            torch.tensor([1, 3], dtype=torch.int64),
        )
        self.assertExpectedInline(
            result,
            """\
region op0(indices_1: i64[2], table_1: f32[8, 4]) -> buf0: f32[2, 4]:
    for i0 in [0, 2):
        for i1 in [0, 4):
            tmp0: i64 = indices_1[i0]
            indirect0: i64 = select(tmp0 < 0, tmp0 + 8, tmp0)
            tmp1: f32 = table_1[indirect0, i1]
            buf0[i0, i1] = tmp1

return (buf0,)""",
        )

    def test_gather_does_not_wrap_negative_values(self):
        def fn(table, indices):
            return torch.gather(table, 1, indices)

        _, result = self.format_graph(
            fn,
            torch.randn(2, 5),
            torch.tensor([[0, 2], [1, 4]], dtype=torch.int64),
        )
        self.assertExpectedInline(
            result,
            """\
region op0(indices_1: i64[2, 2], table_1: f32[2, 5]) -> buf0: f32[2, 2]:
    for i0 in [0, 2):
        for i1 in [0, 2):
            tmp0: i64 = indices_1[i0, i1]
            indirect0: i64 = tmp0
            tmp1: f32 = table_1[i0, indirect0]
            buf0[i0, i1] = tmp1

return (buf0,)""",
        )

    def test_random_seed_dependency_and_index_arithmetic(self):
        def fn(seeds):
            seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 3)
            return torch.ops.prims.inductor_random.default([16], seed, "rand")

        _, result = self.format_graph(
            fn,
            torch.randint(2**63 - 1, (5,), dtype=torch.int64),
        )
        self.assertExpectedInline(
            result,
            """\
region op0(seeds_1_seed: i64[5]) -> buf0: f32[16]:
    for i0 in [0, 16):
        tmp0: i64 = seeds_1_seed.buf[3]
        tmp1: i32 = i0
        tmp2: f32 = rand(tmp0, tmp1)
        buf0[i0] = tmp2

return (buf0,)""",
        )

    def test_permute_output(self):
        def fn(a):
            return a.permute(1, 0)

        _, result = self.format_graph(fn, torch.randn(2, 3))
        self.assertExpectedInline(
            result,
            """\
no_kernel(a_1: f32[2, 3]) -> output0: f32[3, 2]:
    for i0 in [0, 3):
        for i1 in [0, 2):
            output0[i0, i1] = a_1[i1, i0]

return (output0,)""",
        )

    def test_reshape_output(self):
        def fn(a):
            return a.reshape(3, 2)

        _, result = self.format_graph(fn, torch.randn(2, 3))
        self.assertExpectedInline(
            result,
            """\
no_kernel(a_1: f32[2, 3]) -> output0: f32[3, 2]:
    for i0 in [0, 3):
        for i1 in [0, 2):
            output0[i0, i1] = a_1[(i1 + 2*i0) // 3, (i1 + 2*i0) % 3]

return (output0,)""",  # noqa: B950
        )

    def test_transpose_output(self):
        def fn(a):
            return a.transpose(0, 1)

        _, result = self.format_graph(fn, torch.randn(2, 3))
        self.assertExpectedInline(
            result,
            """\
no_kernel(a_1: f32[2, 3]) -> output0: f32[3, 2]:
    for i0 in [0, 3):
        for i1 in [0, 2):
            output0[i0, i1] = a_1[i1, i0]

return (output0,)""",
        )

    def test_unsqueeze_output(self):
        def fn(a):
            return a.unsqueeze(1)

        _, result = self.format_graph(fn, torch.randn(2, 3))
        self.assertExpectedInline(
            result,
            """\
no_kernel(a_1: f32[2, 3]) -> output0: f32[2, 1, 3]:
    for i0 in [0, 2):
        for i1 in [0, 1):
            for i2 in [0, 3):
                output0[i0, i1, i2] = a_1[i0, i2]

return (output0,)""",
        )

    def test_squeeze_output(self):
        def fn(a):
            return a.squeeze(1)

        _, result = self.format_graph(fn, torch.randn(2, 1, 3))
        self.assertExpectedInline(
            result,
            """\
no_kernel(a_1: f32[2, 1, 3]) -> output0: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            output0[i0, i1] = a_1[i0, 0, i1]

return (output0,)""",
        )

    def test_expand_output(self):
        def fn(a):
            return a.unsqueeze(0).expand(2, 3)

        _, result = self.format_graph(fn, torch.randn(3))
        self.assertExpectedInline(
            result,
            """\
no_kernel(a_1: f32[3]) -> output0: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            output0[i0, i1] = a_1[i1]

return (output0,)""",
        )

    def test_view_of_computed_buffer(self):
        def fn(a):
            return torch.sin(a).permute(1, 0)

        _, result = self.format_graph(fn, torch.randn(2, 3))
        self.assertExpectedInline(
            result,
            """\
region op0(a_1: f32[2, 3]) -> buf0: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            tmp0: f32 = a_1[i0, i1]
            tmp1: f32 = sin(tmp0)
            buf0[i0, i1] = tmp1

no_kernel(buf0: f32[2, 3]) -> output0: f32[3, 2]:
    for i0 in [0, 3):
        for i1 in [0, 2):
            output0[i0, i1] = buf0[i1, i0]

return (output0,)""",
        )

    def test_tensorbox_wrapped_output_view(self):
        def fn(a):
            return torch.sin(a).clone().unsqueeze(0)

        graph, result = self.format_graph(fn, torch.randn(2, 3))
        self.assertIsInstance(graph.graph_outputs[0], ir.TensorBox)
        self.assertIsInstance(graph.graph_outputs[0].data, ir.ReinterpretView)
        self.assertExpectedInline(
            result,
            """\
region op0(a_1: f32[2, 3]) -> buf0: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            tmp0: f32 = a_1[i0, i1]
            tmp1: f32 = sin(tmp0)
            buf0[i0, i1] = tmp1

no_kernel(buf0: f32[2, 3]) -> output0: f32[1, 2, 3]:
    for i0 in [0, 1):
        for i1 in [0, 2):
            for i2 in [0, 3):
                output0[i0, i1, i2] = buf0[i1, i2]

return (output0,)""",
        )

    def test_local_tensor_storage(self):
        tokens, hidden_size = sympy.symbols("T H", integer=True, positive=True)
        token, hidden = sympy.symbols(
            "token hidden", integer=True, nonnegative=True
        )

        def tensor(name, dtype, shape, storage):
            return Variable(name, TensorType(dtype, shape), storage)

        x = tensor("x", torch.bfloat16, (tokens, hidden_size), StorageClass.GLOBAL)
        residual = tensor(
            "residual", torch.bfloat16, (tokens, hidden_size), StorageClass.GLOBAL
        )
        weight = tensor("weight", torch.bfloat16, (hidden_size,), StorageClass.GLOBAL)
        residual_out = tensor(
            "residual_out",
            torch.bfloat16,
            (tokens, hidden_size),
            StorageClass.GLOBAL,
        )
        quantized = tensor(
            "q", torch.int8, (tokens, hidden_size), StorageClass.GLOBAL
        )
        output_scale = tensor(
            "output_scale", torch.float32, (tokens,), StorageClass.GLOBAL
        )
        value_cache = tensor(
            "value_cache", torch.float32, (hidden_size,), StorageClass.LOCAL
        )
        norm_cache = tensor(
            "norm_cache", torch.float32, (hidden_size,), StorageClass.LOCAL
        )

        sum_sq = Variable("sum_sq", ScalarType(torch.float32), StorageClass.LOCAL)
        value = Variable("value", ScalarType(torch.float32), StorageClass.LOCAL)
        inv_rms = Variable("inv_rms", ScalarType(torch.float32), StorageClass.LOCAL)
        abs_max = Variable("abs_max", ScalarType(torch.float32), StorageClass.LOCAL)
        normalized = Variable(
            "normalized", ScalarType(torch.float32), StorageClass.LOCAL
        )
        quant_scale = Variable(
            "quant_scale", ScalarType(torch.float32), StorageClass.LOCAL
        )

        x_access = TensorAccess(x, (token, hidden))
        residual_access = TensorAccess(residual, (token, hidden))
        value_access = TensorAccess(value_cache, (hidden,))
        norm_access = TensorAccess(norm_cache, (hidden,))
        normalized_value = Call(
            "mul",
            (
                Call("mul", (value_access, inv_rms)),
                Call("f32", (TensorAccess(weight, (hidden,)),)),
            ),
        )

        program = Program(
            (
                Fused(
                    "add_rmsnorm_quant",
                    (x, residual, weight),
                    (residual_out, quantized, output_scale),
                    (
                        For(
                            token,
                            tokens,
                            (
                                Declare(value_cache),
                                Declare(norm_cache),
                                Declare(sum_sq, Constant(0)),
                                For(
                                    hidden,
                                    hidden_size,
                                    (
                                        Declare(
                                            value,
                                            Call(
                                                "add",
                                                (
                                                    Call("f32", (x_access,)),
                                                    Call("f32", (residual_access,)),
                                                ),
                                            ),
                                        ),
                                        Assign(value_access, value),
                                        Assign(
                                            TensorAccess(
                                                residual_out, (token, hidden)
                                            ),
                                            Call("bf16", (value,)),
                                        ),
                                        Update(
                                            sum_sq,
                                            "+=",
                                            Call("mul", (value, value)),
                                        ),
                                    ),
                                ),
                                Declare(
                                    inv_rms,
                                    Call(
                                        "rsqrt",
                                        (
                                            Call(
                                                "add",
                                                (
                                                    Call(
                                                        "truediv",
                                                        (sum_sq, hidden_size),
                                                    ),
                                                    Constant(1e-5),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                Declare(abs_max, Constant(0)),
                                For(
                                    hidden,
                                    hidden_size,
                                    (
                                        Declare(normalized, normalized_value),
                                        Assign(norm_access, normalized),
                                        Update(
                                            abs_max,
                                            "max=",
                                            Call("abs", (normalized,)),
                                        ),
                                    ),
                                ),
                                Declare(
                                    quant_scale,
                                    Call(
                                        "truediv",
                                        (
                                            Call(
                                                "max",
                                                (abs_max, Constant(1e-12)),
                                            ),
                                            Constant(127),
                                        ),
                                    ),
                                ),
                                Assign(
                                    TensorAccess(output_scale, (token,)),
                                    quant_scale,
                                ),
                                For(
                                    hidden,
                                    hidden_size,
                                    (
                                        Assign(
                                            TensorAccess(
                                                quantized, (token, hidden)
                                            ),
                                            Call(
                                                "quantize",
                                                (
                                                    Call(
                                                        "truediv",
                                                        (norm_access, quant_scale),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            (residual_out, quantized, output_scale),
        )

        self.assertIsInstance(program.fused[0].body[0], For)
        token_loop = program.fused[0].body[0]
        self.assertIsInstance(token_loop.body[0], Declare)
        self.assertIsInstance(token_loop.body[0].variable.type, TensorType)
        self.assertIs(
            token_loop.body[0].variable.storage, StorageClass.LOCAL
        )
        self.assertEqual(
            render_loop_ir(Update(value_access, "+=", value)),
            "value_cache[hidden] += value",
        )
        self.assertExpectedInline(
            render_loop_ir(program),
            """\
region add_rmsnorm_quant(x: bf16[T, H], residual: bf16[T, H], weight: bf16[H]) -> (residual_out: bf16[T, H], q: i8[T, H], output_scale: f32[T]):
    for token in [0, T):
        value_cache: f32[H]
        norm_cache: f32[H]
        sum_sq: f32 = 0
        for hidden in [0, H):
            value: f32 = f32(x[token, hidden]) + f32(residual[token, hidden])
            value_cache[hidden] = value
            residual_out[token, hidden] = bf16(value)
            sum_sq += value * value
        inv_rms: f32 = rsqrt(sum_sq / H + 1e-05)
        abs_max: f32 = 0
        for hidden in [0, H):
            normalized: f32 = value_cache[hidden] * inv_rms * f32(weight[hidden])
            norm_cache[hidden] = normalized
            abs_max max= abs(normalized)
        quant_scale: f32 = max(abs_max, 1e-12) / 127
        output_scale[token] = quant_scale
        for hidden in [0, H):
            q[token, hidden] = quantize(norm_cache[hidden] / quant_scale)

return (residual_out, q, output_scale)""",
        )

    def test_buffer_access_fallback(self):
        def fn(a):
            return torch.sin(a)

        _, result = self.format_graph(fn, torch.empty_strided((2, 3), (4, 1)))
        self.assertExpectedInline(
            result,
            """\
region op0(a_1: f32[2, 3]) -> buf0: f32[2, 3]:
    for i0 in [0, 2):
        for i1 in [0, 3):
            tmp0: f32 = a_1.buf[i1 + 4*i0]
            tmp1: f32 = sin(tmp0)
            buf0[i0, i1] = tmp1

return (buf0,)""",
        )


if __name__ == "__main__":
    run_tests()
