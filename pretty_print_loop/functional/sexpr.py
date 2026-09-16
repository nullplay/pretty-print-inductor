"""Canonical S-expression rendering for the scalar loop AST."""

from dataclasses import fields

from .ast import BinaryOp, Dummy, Node, Num, UnaryOp, Var


def to_sexpr(node: Node) -> str:
    if not isinstance(node, Node):
        raise TypeError(f"node must be a Node, got {type(node).__name__}")
    if isinstance(node, Num):
        return str(node.value)
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Dummy):
        return node.op
    if isinstance(node, UnaryOp):
        return f"({node.name} {to_sexpr(node.value)})"
    if isinstance(node, BinaryOp):
        return f"({node.name} {to_sexpr(node.lhs)} {to_sexpr(node.rhs)})"

    children = " ".join(
        to_sexpr(getattr(node, field.name)) for field in fields(node) if field.init
    )
    return f"({node.op}{' ' if children else ''}{children})"


__all__ = ["to_sexpr"]
