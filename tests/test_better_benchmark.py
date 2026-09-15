import unittest
from pathlib import Path

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from tests.benchmark_harness import (
    CANONICAL_ROOT,
    render_repro,
    trace_repro,
)

EXPECTED_ROOT = Path(__file__).parent / "expected"


@unittest.skipUnless(CANONICAL_ROOT.is_dir(), "better-benchmark corpus not found")
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBetterBenchmark(TestCase):
    @staticmethod
    def format_repro(repro_id: str, *, include_raw_ir: bool = False) -> str:
        return render_repro(
            repro_id, include_raw_ir=include_raw_ir
        ).post_lowering

    @staticmethod
    def trace_repro(repro_id: str, *, include_scheduler_ir: bool = False) -> str:
        return trace_repro(
            repro_id, include_scheduler_ir=include_scheduler_ir
        )

    def assert_repro(self, repro_id: str) -> None:
        result = render_repro(repro_id)
        post_lowering_path = (
            EXPECTED_ROOT / "post_lowering" / f"{repro_id}.txt"
        )
        fused_scheduler_path = (
            EXPECTED_ROOT / "fused_scheduler" / f"{repro_id}.txt"
        )
        fusion_trace_path = EXPECTED_ROOT / "fusion_trace" / f"{repro_id}.txt"
        self.assertEqual(
            result.post_lowering, post_lowering_path.read_text().rstrip()
        )
        self.assertEqual(
            result.fused_scheduler, fused_scheduler_path.read_text().rstrip()
        )
        self.assertEqual(
            result.fusion_trace, fusion_trace_path.read_text().rstrip()
        )
        self.assertNotIn("unimplemented", result.post_lowering)
        self.assertNotIn("unimplemented", result.fused_scheduler)
        self.assertNotIn("unimplemented", result.fusion_trace)

    # Source: repros/canonical/sum_9552a61d796d
    # Model: torchbench/train/lennard_jones
    def test_sum_with_view_and_conversions(self):
        self.assert_repro("sum_9552a61d796d")

    # Source: repros/canonical/sum_sum_b6dba5d75a7b
    # Models: torchbench/{infer,train}/BERT_pytorch
    def test_unrolled_sum_reduction_and_permute(self):
        self.assert_repro("sum_sum_b6dba5d75a7b")

    # Source: repros/canonical/any_1918882eece2
    # Model: hf/{infer,train}/AllenaiLongformerBase
    def test_boolean_any_over_reshape(self):
        self.assert_repro("any_1918882eece2")

    # Source: repros/canonical/amax_sum_9675d6f1ba21
    # Models: hf/train/{GPTJForCausalLM,GPTJForQuestionAnswering}
    def test_softmax_reductions_with_output_views(self):
        self.assert_repro("amax_sum_9675d6f1ba21")

    # Source: repros/canonical/sum_c902430e8a5b
    # Model: torchbench/train/BERT_pytorch
    def test_dense_alias_chain_around_sum(self):
        self.assert_repro("sum_c902430e8a5b")

    # Source: repros/canonical/mean_var_83ed19c04171
    # Models: torchbench/{infer,train}/BERT_pytorch
    def test_welford_variance_epilogue(self):
        self.assert_repro("mean_var_83ed19c04171")

    # Source: repros/canonical/var_mean_ff1e9ea5f167
    # Model: torchbench/train/functorch_dp_cifar10
    def test_welford_multi_axis_reduction(self):
        self.assert_repro("var_mean_ff1e9ea5f167")

    # Source: repros/canonical/pointwise_299ce2ae2b07
    # Model: hf/train/M2M100ForConditionalGeneration
    def test_advanced_index_with_negative_wrapping(self):
        self.assert_repro("pointwise_299ce2ae2b07")

    # Source: repros/canonical/pointwise_bae56b847d32
    # Models: timm/{infer,train}/beit_base_patch16_224
    def test_view_heavy_advanced_index(self):
        self.assert_repro("pointwise_bae56b847d32")

    # Source: repros/canonical/amax_sum_sum_9ff2f9544913
    # Models: hf/{infer,train}/GPTJForQuestionAnswering
    def test_gather_without_negative_wrapping(self):
        self.assert_repro("amax_sum_sum_9ff2f9544913")


    # TODO : Scatter / Masked / Concat,Stack / Mutation
    # Dynamic Symbol shapes

if __name__ == "__main__":
    run_tests()
