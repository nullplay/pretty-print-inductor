# Scalar Loop AST

`ast.py` defines syntax without PyTorch, FX, scheduling, dependency analysis,
or rewrite integration. `graph_lowering.py` currently converts
`ComputedBuffer(Pointwise)` and sum `ComputedBuffer(Reduction)` nodes.

The language is deliberately small:

```text
node  := expr | stmt | tensor-ref | index

expr  := Num(value)
       | Var(name)
       | Load(tensor-ref, Index)
       | UnaryOp(name, expr)
       | BinaryOp(name, expr, expr)

index := Index(expr)

stmt  := Store(tensor-ref, Index, expr)
       | Seq(stmt, stmt)
       | Loop(var, end, stmt)
       | Dummy

tensor-ref := Input(Var) | Output(Var) | Tensor(Var)
```

Every tensor is a flat array, so `Index` always contains exactly one scalar
address expression.  For a logical `(i, j)` access to a row-major tensor with
row width `N`, write:

```text
(index (+ (* i N) j))
```

`Loop(i, N, body)` means `for i in range(0, N)`: start zero and step one are
implicit.  `Seq` remains binary and the `seq(...)` helper constructs its
canonical right-associated form.

Running reductions use explicit memory state.  Initialize an accumulator with
`Store`, update it inside a loop using `Load` plus `Add`, and store the final
value normally.  A local sum primitive is `UnaryOp("rsum", value)`; it is not a
replacement for the loop-carried pattern.

## Python construction

```python
from pretty_print_loop.functional import (
    BinaryOp,
    Index,
    Input,
    Load,
    Loop,
    Num,
    Output,
    Store,
    Var,
)

i = Var("i")
x = Input(Var("x"))
y = Output(Var("y"))

tree = Loop(
    i,
    Var("N"),
    Store(
        y,
        Index(i),
        BinaryOp("*", Load(x, Index(i)), Num(2)),
    ),
)
```

`to_sexpr(tree)` produces:

```text
(loop i N
  (store (output y) (index i)
    (* (load (input x) (index i)) 2)))
```

## GraphLowering conversion

```python
from pretty_print_loop.functional import build_graph_lowering_ast, to_sexpr

tree = build_graph_lowering_ast(graph_lowering)
print(to_sexpr(tree))
```

Currently supported:

- `ComputedBuffer(Pointwise)`;
- sum `ComputedBuffer(Reduction)`;
- generic unary and binary scalar callbacks;
- flat SymPy symbols, integers, additions, and multiplications; and
- multiple operations joined by a right-associated `Seq`.

The sum converter emits an explicit accumulator initialization, inner running
update, and final output store. Non-sum reductions and unsupported flat-index
expressions fail with `UnsupportedNodeError` rather than extending the AST.

## Tests

```shell
python -m unittest discover -s tests/functional -t . -v
```
