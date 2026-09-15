# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor.scheduler import FusedSchedulerNode, Scheduler
from torch.testing._internal.common_utils import TestCase, run_tests

from pretty_print_loop import (
    capture_fusion_steps,
    format_fused_scheduler,
    format_fusion_trace,
)
from pretty_print_loop.lowering import graph_context, lower_callable


def _fusion_example(x, bias):
    shifted = x + bias
    left = torch.sin(shifted)
    right = torch.cos(x)
    product = left * right
    transposed = product.permute(0, 2, 1)
    total = transposed.sum(-1)
    maximum = transposed.amax(-1)
    out = torch.sigmoid(total) * torch.tanh(maximum)
    return left, right, total, maximum, out


def _inputs():
    return (
        torch.randn(4, 128, 64, device="cuda"),
        torch.randn(64, device="cuda"),
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedSchedulerLoopIR(TestCase):
    def test_pointwise_horizontal_reductions_and_epilogue(self):
        graph = lower_callable(_fusion_example, *_inputs())

        with graph_context(graph):
            scheduler = Scheduler(graph.operations)
            result = format_fused_scheduler(scheduler)

        self.assertEqual(len(scheduler.nodes), 1)
        self.assertIsInstance(scheduler.nodes[0], FusedSchedulerNode)
        self.assertExpectedInline(
            result,
            """\
region op0_op1_op2_op3_op4(x_1: f32[4, 128, 64], bias_1: f32[64]) -> (buf0: f32[4, 128, 64], buf1: f32[4, 128, 64], buf2: f32[4, 64], buf3: f32[4, 64], buf4: f32[4, 64]):
    for p0 in [0, 4):
        for p1 in [0, 64):
            acc_0: f32 = 0
            acc_1: f32 = -inf
            for r0 in [0, 128):
                tmp0: f32 = x_1[p0, r0, p1]
                tmp1: f32 = bias_1[p1]
                tmp2: f32 = tmp0 + tmp1
                tmp3: f32 = sin(tmp2)
                buf0[p0, r0, p1] = tmp3
                tmp4: f32 = x_1[p0, r0, p1]
                tmp5: f32 = cos(tmp4)
                buf1[p0, r0, p1] = tmp5
                tmp6: f32 = tmp3 * tmp5
                acc_0 += tmp6
                tmp7: f32 = tmp3 * tmp5
                acc_1 max= tmp7
            buf2[p0, p1] = acc_0
            buf3[p0, p1] = acc_1
            tmp8: f32 = sigmoid(acc_0)
            tmp9: f32 = tanh(acc_1)
            tmp10: f32 = tmp8 * tmp9
            buf4[p0, p1] = tmp10

return (buf0, buf1, buf2, buf3, buf4)""",
        )

    def test_all_fusion_steps(self):
        graph = lower_callable(_fusion_example, *_inputs())

        with graph_context(graph), capture_fusion_steps() as trace:
            Scheduler(graph.operations)

        self.assertEqual(len(trace.passes), 3)
        self.assertEqual(len(trace.passes[0].steps), 3)
        self.assertEqual(len(trace.passes[1].steps), 0)
        self.assertEqual(len(trace.passes[2].steps), 1)
        self.assertTrue(trace.passes[2].is_reorder)
        self.assertExpectedInline(
            format_fusion_trace(trace),
            """\
post_graph_lowering:

region op0(x_1: f32[4, 128, 64], bias_1: f32[64]) -> buf0: f32[4, 128, 64]:
    for i0 in [0, 4):
        for i1 in [0, 128):
            for i2 in [0, 64):
                tmp0: f32 = x_1[i0, i1, i2]
                tmp1: f32 = bias_1[i2]
                tmp2: f32 = tmp0 + tmp1
                tmp3: f32 = sin(tmp2)
                buf0[i0, i1, i2] = tmp3

region op1(x_1: f32[4, 128, 64]) -> buf1: f32[4, 128, 64]:
    for i0 in [0, 4):
        for i1 in [0, 128):
            for i2 in [0, 64):
                tmp0: f32 = x_1[i0, i1, i2]
                tmp1: f32 = cos(tmp0)
                buf1[i0, i1, i2] = tmp1

region op2(buf0: f32[4, 128, 64], buf1: f32[4, 128, 64]) -> buf2: f32[4, 64]:
    for i0 in [0, 4):
        for i1 in [0, 64):
            acc_0: f32 = 0
            for r0 in [0, 128):
                tmp0: f32 = buf0[i0, r0, i1]
                tmp1: f32 = buf1[i0, r0, i1]
                tmp2: f32 = tmp0 * tmp1
                acc_0 += tmp2
            buf2[i0, i1] = acc_0

region op3(buf0: f32[4, 128, 64], buf1: f32[4, 128, 64]) -> buf3: f32[4, 64]:
    for i0 in [0, 4):
        for i1 in [0, 64):
            acc_0: f32 = -inf
            for r0 in [0, 128):
                tmp0: f32 = buf0[i0, r0, i1]
                tmp1: f32 = buf1[i0, r0, i1]
                tmp2: f32 = tmp0 * tmp1
                acc_0 max= tmp2
            buf3[i0, i1] = acc_0

region op4(buf2: f32[4, 64], buf3: f32[4, 64]) -> buf4: f32[4, 64]:
    for i0 in [0, 4):
        for i1 in [0, 64):
            tmp0: f32 = buf2[i0, i1]
            tmp1: f32 = sigmoid(tmp0)
            tmp2: f32 = buf3[i0, i1]
            tmp3: f32 = tanh(tmp2)
            tmp4: f32 = tmp1 * tmp3
            buf4[i0, i1] = tmp4

return (buf0, buf1, buf2, buf3, buf4)

fusion_history:

fusion_pass 1 (ordinary):
    before: op0, op1, op2, op3, op4

    step 1: op2 + op3 -> op2_op3
    live: op0, op1, op2_op3, op4

region op2_op3(buf0: f32[4, 128, 64], buf1: f32[4, 128, 64]) -> (buf2: f32[4, 64], buf3: f32[4, 64]):
    for p0 in [0, 4):
        for p1 in [0, 64):
            acc_0: f32 = 0
            acc_1: f32 = -inf
            for r0 in [0, 128):
                tmp0: f32 = buf0[p0, r0, p1]
                tmp1: f32 = buf1[p0, r0, p1]
                tmp2: f32 = tmp0 * tmp1
                acc_0 += tmp2
                tmp3: f32 = buf0[p0, r0, p1]
                tmp4: f32 = buf1[p0, r0, p1]
                tmp5: f32 = tmp3 * tmp4
                acc_1 max= tmp5
            buf2[p0, p1] = acc_0
            buf3[p0, p1] = acc_1

    step 2: op0 + op1 -> op0_op1
    live: op0_op1, op2_op3, op4

region op0_op1(x_1: f32[4, 128, 64], bias_1: f32[64]) -> (buf0: f32[4, 128, 64], buf1: f32[4, 128, 64]):
    for p0 in [0, 4):
        for p1 in [0, 128):
            for p2 in [0, 64):
                tmp0: f32 = x_1[p0, p1, p2]
                tmp1: f32 = bias_1[p2]
                tmp2: f32 = tmp0 + tmp1
                tmp3: f32 = sin(tmp2)
                buf0[p0, p1, p2] = tmp3
                tmp4: f32 = x_1[p0, p1, p2]
                tmp5: f32 = cos(tmp4)
                buf1[p0, p1, p2] = tmp5

    step 3: op2_op3 + op4 -> op2_op3_op4
    live: op0_op1, op2_op3_op4

region op2_op3_op4(buf0: f32[4, 128, 64], buf1: f32[4, 128, 64]) -> (buf2: f32[4, 64], buf3: f32[4, 64], buf4: f32[4, 64]):
    for p0 in [0, 4):
        for p1 in [0, 64):
            acc_0: f32 = 0
            acc_1: f32 = -inf
            for r0 in [0, 128):
                tmp0: f32 = buf0[p0, r0, p1]
                tmp1: f32 = buf1[p0, r0, p1]
                tmp2: f32 = tmp0 * tmp1
                acc_0 += tmp2
                tmp3: f32 = buf0[p0, r0, p1]
                tmp4: f32 = buf1[p0, r0, p1]
                tmp5: f32 = tmp3 * tmp4
                acc_1 max= tmp5
            buf2[p0, p1] = acc_0
            buf3[p0, p1] = acc_1
            tmp6: f32 = sigmoid(acc_0)
            tmp7: f32 = tanh(acc_1)
            tmp8: f32 = tmp6 * tmp7
            buf4[p0, p1] = tmp8

    after: op0_op1, op2_op3_op4

fusion_pass 2 (ordinary):
    before: op0_op1, op2_op3_op4
    no successful fusion

    after: op0_op1, op2_op3_op4

fusion_pass 3 (loop-reordering):
    before: op0_op1, op2_op3_op4

    step 1: op0_op1 + op2_op3_op4 -> op0_op1_op2_op3_op4
    live: op0_op1_op2_op3_op4

region op0_op1_op2_op3_op4(x_1: f32[4, 128, 64], bias_1: f32[64]) -> (buf0: f32[4, 128, 64], buf1: f32[4, 128, 64], buf2: f32[4, 64], buf3: f32[4, 64], buf4: f32[4, 64]):
    for p0 in [0, 4):
        for p1 in [0, 64):
            acc_0: f32 = 0
            acc_1: f32 = -inf
            for r0 in [0, 128):
                tmp0: f32 = x_1[p0, r0, p1]
                tmp1: f32 = bias_1[p1]
                tmp2: f32 = tmp0 + tmp1
                tmp3: f32 = sin(tmp2)
                buf0[p0, r0, p1] = tmp3
                tmp4: f32 = x_1[p0, r0, p1]
                tmp5: f32 = cos(tmp4)
                buf1[p0, r0, p1] = tmp5
                tmp6: f32 = tmp3 * tmp5
                acc_0 += tmp6
                tmp7: f32 = tmp3 * tmp5
                acc_1 max= tmp7
            buf2[p0, p1] = acc_0
            buf3[p0, p1] = acc_1
            tmp8: f32 = sigmoid(acc_0)
            tmp9: f32 = tanh(acc_1)
            tmp10: f32 = tmp8 * tmp9
            buf4[p0, p1] = tmp10

    after: op0_op1_op2_op3_op4""",
        )


if __name__ == "__main__":
    run_tests()
