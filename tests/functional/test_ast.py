import dataclasses
import unittest

import pretty_print_loop.functional as ast


class TestScalarLoopAST(unittest.TestCase):
    def test_ast_is_immutable_and_typed_by_role(self):
        loop = ast.Loop(ast.Var("i"), ast.Var("N"), ast.Dummy())

        self.assertIsInstance(loop, ast.Node)
        self.assertIsInstance(loop, ast.Stmt)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            loop.body = ast.Dummy()

    def test_loop_has_implicit_zero_start_and_unit_step(self):
        loop = ast.Loop(ast.Var("i"), ast.Var("N"), ast.Dummy())

        self.assertEqual(ast.to_sexpr(loop), "(loop i N dummy)")
        with self.assertRaises(TypeError):
            ast.Loop(ast.Num(0), ast.Var("N"), ast.Num(1), ast.Var("i"), ast.Dummy())

    def test_index_is_one_flat_scalar_expression(self):
        i = ast.Var("i")
        j = ast.Var("j")
        width = ast.Var("N")
        flat = ast.BinaryOp("+", ast.BinaryOp("*", i, width), j)

        index = ast.Index(flat)

        self.assertEqual(ast.to_sexpr(index), "(index (+ (* i N) j))")
        with self.assertRaises(TypeError):
            ast.Index((i, j))

    def test_nested_running_sum_uses_flat_addresses(self):
        i = ast.Var("i")
        j = ast.Var("j")
        n = ast.Var("N")
        acc = ast.Tensor(ast.Var("acc"))
        x = ast.Input(ast.Var("x"))
        y = ast.Output(ast.Var("y"))
        acc_i = ast.Index(i)
        x_ij = ast.Index(ast.BinaryOp("+", ast.BinaryOp("*", i, n), j))

        program = ast.Loop(
            i,
            ast.Var("M"),
            ast.seq(
                ast.Store(acc, acc_i, ast.Num(0)),
                ast.Loop(
                    j,
                    n,
                    ast.Store(
                        acc,
                        acc_i,
                        ast.BinaryOp("+", ast.Load(acc, acc_i), ast.Load(x, x_ij)),
                    ),
                ),
                ast.Store(y, acc_i, ast.Load(acc, acc_i)),
            ),
        )

        self.assertEqual(
            ast.to_sexpr(program),
            "(loop i M (seq (store (tensor acc) (index i) 0) (seq (loop j N (store (tensor acc) (index i) (+ (load (tensor acc) (index i)) (load (input x) (index (+ (* i N) j)))))) (store (output y) (index i) (load (tensor acc) (index i))))))",
        )

    def test_seq_builder_is_right_associated(self):
        index = ast.Index(ast.Num(0))
        first = ast.Store(ast.Tensor(ast.Var("a")), index, ast.Num(1))
        second = ast.Store(ast.Tensor(ast.Var("b")), index, ast.Num(2))
        third = ast.Store(ast.Tensor(ast.Var("c")), index, ast.Num(3))

        result = ast.seq(first, second, third)

        self.assertIsInstance(result, ast.Seq)
        self.assertIs(result.first, first)
        self.assertIsInstance(result.second, ast.Seq)
        self.assertIs(result.second.first, second)
        self.assertIs(result.second.second, third)

    def test_memory_and_compute_roles_are_checked(self):
        index = ast.Index(ast.Num(0))
        with self.assertRaisesRegex(TypeError, "Load.base"):
            ast.Load(ast.Var("a"), index)
        with self.assertRaisesRegex(TypeError, "Store.value"):
            ast.Store(ast.Tensor(ast.Var("a")), index, ast.Dummy())
        with self.assertRaisesRegex(TypeError, "Loop.body"):
            ast.Loop(ast.Var("i"), ast.Var("N"), ast.Num(0))

    def test_generic_unary_and_binary_operators_render(self):
        a = ast.Var("a")
        b = ast.Var("b")

        self.assertEqual(ast.to_sexpr(ast.BinaryOp("+", a, b)), "(+ a b)")
        self.assertEqual(ast.to_sexpr(ast.BinaryOp("*", a, b)), "(* a b)")
        self.assertEqual(ast.to_sexpr(ast.UnaryOp("sin", a)), "(sin a)")
        self.assertEqual(ast.to_sexpr(ast.UnaryOp("rsum", a)), "(rsum a)")


if __name__ == "__main__":
    unittest.main()
