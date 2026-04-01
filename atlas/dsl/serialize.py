"""Expression serialization: S-expression strings and dicts."""
from __future__ import annotations

from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.operators import Op


def to_str(expr: Expr) -> str:
    if isinstance(expr, Const):
        return f"(const {expr.value})"
    if isinstance(expr, Var):
        return f"(var {expr.name})"
    if isinstance(expr, UnaryOp):
        return f"({expr.op.value} {to_str(expr.operand)})"
    if isinstance(expr, BinOp):
        return f"({expr.op.value} {to_str(expr.left)} {to_str(expr.right)})"
    if isinstance(expr, NAryOp):
        children_str = " ".join(to_str(c) for c in expr.children)
        return f"(nary_{expr.op.value} {children_str})"
    raise TypeError(f"Unknown expr type: {type(expr)}")


def from_str(s: str) -> Expr:
    tokens = _tokenize(s)
    expr, pos = _parse(tokens, 0)
    return expr


def _tokenize(s: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for ch in s:
        if ch in ("(", ")"):
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
        elif ch == " ":
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens


def _parse(tokens: list[str], pos: int) -> tuple[Expr, int]:
    if tokens[pos] != "(":
        raise ValueError(f"Expected '(' at pos {pos}, got '{tokens[pos]}'")
    pos += 1
    tag = tokens[pos]
    pos += 1

    if tag == "const":
        value = float(tokens[pos])
        pos += 1
        pos += 1  # skip ')'
        return Const(value), pos

    if tag == "var":
        name = tokens[pos]
        pos += 1
        pos += 1  # skip ')'
        return Var(name), pos

    if tag.startswith("nary_"):
        op_name = tag[5:]
        op = Op(op_name)
        children: list[Expr] = []
        while tokens[pos] != ")":
            child, pos = _parse(tokens, pos)
            children.append(child)
        pos += 1
        return NAryOp(op, children), pos

    op = Op(tag)
    first, pos = _parse(tokens, pos)
    if tokens[pos] == ")":
        pos += 1
        return UnaryOp(op, first), pos
    second, pos = _parse(tokens, pos)
    pos += 1
    return BinOp(op, first, second), pos


def to_dict(expr: Expr) -> dict:
    if isinstance(expr, Const):
        return {"type": "const", "value": expr.value}
    if isinstance(expr, Var):
        return {"type": "var", "name": expr.name}
    if isinstance(expr, UnaryOp):
        return {"type": "unary", "op": expr.op.value, "operand": to_dict(expr.operand)}
    if isinstance(expr, BinOp):
        return {"type": "binary", "op": expr.op.value,
                "left": to_dict(expr.left), "right": to_dict(expr.right)}
    if isinstance(expr, NAryOp):
        return {"type": "nary", "op": expr.op.value,
                "children": [to_dict(c) for c in expr.children]}
    raise TypeError(f"Unknown expr type: {type(expr)}")


def from_dict(d: dict) -> Expr:
    t = d["type"]
    if t == "const":
        return Const(d["value"])
    if t == "var":
        return Var(d["name"])
    if t == "unary":
        return UnaryOp(Op(d["op"]), from_dict(d["operand"]))
    if t == "binary":
        return BinOp(Op(d["op"]), from_dict(d["left"]), from_dict(d["right"]))
    if t == "nary":
        return NAryOp(Op(d["op"]), [from_dict(c) for c in d["children"]])
    raise ValueError(f"Unknown type: {t}")
