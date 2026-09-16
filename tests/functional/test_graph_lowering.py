import unittest

import sympy
import torch

import pretty_print_loop.functional as ast
from pretty_print_loop.lowering import lower_callable


class TestGraphLoweringToScalarAST(unittest.TestCase):
    def test_pointwise_computed_buffer(self):
        def fn(a, b, c):
            return torch.sin(a + b) * c

        graph = lower_callable(
            fn,
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3),
        )
        tree = ast.build_graph_lowering_ast(graph)

        self.assertIsInstance(tree, ast.Loop)
        self.assertIsInstance(tree.body, ast.Loop)
        self.assertIsInstance(tree.body.body, ast.Store)
        self.assertIsInstance(tree.body.body.value, ast.BinaryOp)
        self.assertEqual(
            ast.to_sexpr(tree),
            "(loop i0 2 (loop i1 3 "
            "(store (output buf0) (index (+ i1 (* 3 i0))) "
            "(* (sin (+ "
            "(load (input a_1) (index (+ i1 (* 3 i0)))) "
            "(load (input b_1) (index (+ i1 (* 3 i0)))))) "
            "(load (input c_1) (index (+ i1 (* 3 i0))))))))",
        )

    def test_sum_reduction_computed_buffer(self):
        def fn(a):
            return torch.sin(a).sum(dim=1)

        graph = lower_callable(fn, torch.randn(2, 128))
        tree = ast.build_graph_lowering_ast(graph)

        self.assertIsInstance(tree, ast.Loop)
        self.assertIsInstance(tree.body, ast.Seq)
        self.assertEqual(
            ast.to_sexpr(tree),
            "(loop i0 2 (seq (store (tensor acc_0) (index i0) 0) (seq (loop r0 128 (store (tensor acc_0) (index i0) (+ (load (tensor acc_0) (index i0)) (sin (load (input a_1) (index (+ r0 (* 128 i0)))))))) (store (output buf0) (index i0) (load (tensor acc_0) (index i0))))))",
        )

    def test_multiple_computed_buffers_form_right_associated_sequence(self):
        def fn(a):
            return torch.sin(a), torch.cos(a)

        graph = lower_callable(fn, torch.randn(2, 3))
        tree = ast.build_graph_lowering_ast(graph)

        self.assertIsInstance(tree, ast.Seq)
        self.assertIsInstance(tree.first, ast.Loop)
        self.assertIsInstance(tree.second, ast.Loop)
        rendered = ast.to_sexpr(tree)
        self.assertIn("(sin (load (input a_1)", rendered)
        self.assertIn("(cos (load (input a_1)", rendered)

    def test_sympy_indices_are_flat_add_mul_expressions(self):
        i, j, width = sympy.symbols("i j width", integer=True)
        expression = ast.sympy_to_expr(i * width + j)

        self.assertEqual(
            ast.to_sexpr(ast.Index(expression)), "(index (+ j (* i width)))"
        )

    def test_unsupported_reduction_is_rejected(self):
        graph = lower_callable(lambda a: a.amax(dim=1), torch.randn(2, 128))

        with self.assertRaisesRegex(ast.UnsupportedNodeError, "only sum Reduction"):
            ast.build_graph_lowering_ast(graph)


if __name__ == "__main__":
    unittest.main()
