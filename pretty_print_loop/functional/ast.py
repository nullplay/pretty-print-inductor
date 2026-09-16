from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import ClassVar

# node       ::= tensor_ref | index | expr | stmt
# tensor_ref ::= Input(Var) | Output(Var) | Tensor(Var)
# index      ::= Index(expr)
# expr       ::= Num | Var | Load | UnaryOp | BinaryOp
# stmt       ::= Dummy | Store | Seq | Loop


# Core


@dataclass(frozen=True, slots=True)
class Node:
    op: ClassVar[str]


@dataclass(frozen=True, slots=True)
class Expr(Node):
    pass


@dataclass(frozen=True, slots=True)
class Stmt(Node):
    pass


# Scalar leaves


@dataclass(frozen=True, slots=True)
class Num(Expr):
    # 42
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("Num expects an integer")


@dataclass(frozen=True, slots=True)
class Var(Expr):
    # i
    name: str

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() or char in "()" for char in self.name):
            raise ValueError("Var name must be a non-empty S-expression atom")


# Tensor references


@dataclass(frozen=True, slots=True)
class _TensorRef(Node):
    name: Var

    def __post_init__(self) -> None:
        if not isinstance(self.name, Var):
            raise TypeError("tensor name must be a Var")


@dataclass(frozen=True, slots=True)
class Input(_TensorRef):
    # (input x)
    op: ClassVar[str] = "input"


@dataclass(frozen=True, slots=True)
class Output(_TensorRef):
    # (output y)
    op: ClassVar[str] = "output"


@dataclass(frozen=True, slots=True)
class Tensor(_TensorRef):
    # (tensor tmp)
    op: ClassVar[str] = "tensor"


TensorRef = Input | Output | Tensor


# Scalar computation


def _validate_op_name(name: str) -> None:
    if not name or any(char.isspace() or char in "()" for char in name):
        raise ValueError("operator name must be a non-empty S-expression atom")


@dataclass(frozen=True, slots=True)
class UnaryOp(Expr):
    # (sin x), (rsum x)
    name: str
    value: Expr

    def __post_init__(self) -> None:
        _validate_op_name(self.name)
        if not isinstance(self.value, Expr):
            raise TypeError("UnaryOp.value must be an Expr")


@dataclass(frozen=True, slots=True)
class BinaryOp(Expr):
    # (+ a b), (* a b)
    name: str
    lhs: Expr
    rhs: Expr

    def __post_init__(self) -> None:
        _validate_op_name(self.name)
        if not isinstance(self.lhs, Expr) or not isinstance(self.rhs, Expr):
            raise TypeError("BinaryOp operands must be Exprs")


# Flat addressing and reads


@dataclass(frozen=True, slots=True)
class Index(Node):
    # (index (+ (* i N) j))
    value: Expr
    op: ClassVar[str] = "index"

    def __post_init__(self) -> None:
        if not isinstance(self.value, Expr):
            raise TypeError("Index.value must be an Expr")


@dataclass(frozen=True, slots=True)
class Load(Expr):
    # (load (input x) (index i))
    base: TensorRef
    index: Index
    op: ClassVar[str] = "load"

    def __post_init__(self) -> None:
        if not isinstance(self.base, (Input, Output, Tensor)):
            raise TypeError("Load.base must be Input, Output, or Tensor")
        if not isinstance(self.index, Index):
            raise TypeError("Load.index must be an Index")


# Statements


@dataclass(frozen=True, slots=True)
class Dummy(Stmt):
    # dummy
    op: ClassVar[str] = "dummy"


@dataclass(frozen=True, slots=True)
class Store(Stmt):
    # (store (output y) (index i) value)
    base: TensorRef
    index: Index
    value: Expr
    op: ClassVar[str] = "store"

    def __post_init__(self) -> None:
        if not isinstance(self.base, (Input, Output, Tensor)):
            raise TypeError("Store.base must be Input, Output, or Tensor")
        if not isinstance(self.index, Index):
            raise TypeError("Store.index must be an Index")
        if not isinstance(self.value, Expr):
            raise TypeError("Store.value must be an Expr")


@dataclass(frozen=True, slots=True)
class Seq(Stmt):
    # (seq first second)
    first: Stmt
    second: Stmt
    op: ClassVar[str] = "seq"

    def __post_init__(self) -> None:
        if not isinstance(self.first, Stmt) or not isinstance(self.second, Stmt):
            raise TypeError("Seq children must be Stmts")


@dataclass(frozen=True, slots=True)
class Loop(Stmt):
    # (loop i N body) == for i in range(0, N)
    var: Var
    end: Expr
    body: Stmt
    op: ClassVar[str] = "loop"

    def __post_init__(self) -> None:
        if not isinstance(self.var, Var):
            raise TypeError("Loop.var must be a Var")
        if not isinstance(self.end, Expr):
            raise TypeError("Loop.end must be an Expr")
        if not isinstance(self.body, Stmt):
            raise TypeError("Loop.body must be a Stmt")


# Builders


def seq(*statements: Stmt) -> Stmt:
    # seq(a, b, c) -> Seq(a, Seq(b, c))
    if not statements:
        raise ValueError("seq() requires at least one statement")
    if not all(isinstance(statement, Stmt) for statement in statements):
        raise TypeError("seq() accepts only Stmts")
    return reduce(
        lambda rest, statement: Seq(statement, rest),
        reversed(statements[:-1]),
        statements[-1],
    )


__all__ = [
    "BinaryOp",
    "Dummy",
    "Expr",
    "Index",
    "Input",
    "Load",
    "Loop",
    "Node",
    "Num",
    "Output",
    "Seq",
    "Stmt",
    "Store",
    "Tensor",
    "TensorRef",
    "UnaryOp",
    "Var",
    "seq",
]
