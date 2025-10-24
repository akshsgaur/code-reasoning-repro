"""Translate DSL-List ASTs to imperative Python programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .ast import (
    And,
    Append,
    BoolLit,
    Empty,
    Eq,
    Expr,
    Extend,
    Gt,
    If,
    Index,
    Init,
    IntLit,
    Length,
    Map,
    Not,
    Or,
    Tail,
    Lt,
    Var,
)
from .sampler import MAP_ELEMENT_NAME


@dataclass
class TranspileCtx:
    arity: int
    lines: List[str] = field(default_factory=list)
    init_lines: List[str] = field(default_factory=list)
    indent_level: int = 1
    temp_counter: int = 0
    map_stack: List[str] = field(default_factory=list)

    def new_temp(self, prefix: str = "v") -> str:
        self.temp_counter += 1
        return f"{prefix}{self.temp_counter}"

    def add_line(self, line: str) -> None:
        self.lines.append("    " * self.indent_level + line)

    def add_init(self, line: str) -> None:
        self.init_lines.append("    " + line)

    def push_indent(self) -> None:
        self.indent_level += 1

    def pop_indent(self) -> None:
        if self.indent_level <= 0:
            raise ValueError("Negative indentation")
        self.indent_level -= 1


def _emit(expr: Expr, ctx: TranspileCtx) -> str:
    if isinstance(expr, Var):
        if expr.name == MAP_ELEMENT_NAME:
            if not ctx.map_stack:
                raise ValueError("Map element referenced outside of map.")
            return ctx.map_stack[-1]
        return expr.name
    if isinstance(expr, IntLit):
        return str(expr.value)
    if isinstance(expr, BoolLit):
        return "True" if expr.value else "False"
    if isinstance(expr, Empty):
        name = ctx.new_temp()
        ctx.add_init(f"{name} = []")
        return name
    if isinstance(expr, Append):
        lst = _emit(expr.lst, ctx)
        value = _emit(expr.x, ctx)
        ctx.add_line(f"{lst}.append({value})")
        return lst
    if isinstance(expr, Extend):
        first = _emit(expr.a, ctx)
        second = _emit(expr.b, ctx)
        ctx.add_line(f"{second}.extend({first})")
        return second
    if isinstance(expr, Init):
        lst = _emit(expr.lst, ctx)
        ctx.add_line(f"{lst}.pop()")
        return lst
    if isinstance(expr, Tail):
        lst = _emit(expr.lst, ctx)
        ctx.add_line(f"{lst}.pop(0)")
        return lst
    if isinstance(expr, Length):
        lst = _emit(expr.lst, ctx)
        return f"len({lst})"
    if isinstance(expr, Index):
        lst = _emit(expr.lst, ctx)
        idx = _emit(expr.i, ctx)
        return f"{lst}[{idx}]"
    if isinstance(expr, Eq):
        return f"({ _emit(expr.a, ctx) }) == ({ _emit(expr.b, ctx) })"
    if isinstance(expr, Gt):
        return f"({ _emit(expr.a, ctx) }) > ({ _emit(expr.b, ctx) })"
    if isinstance(expr, And):
        return f"({ _emit(expr.a, ctx) }) and ({ _emit(expr.b, ctx) })"
    if isinstance(expr, Or):
        return f"({ _emit(expr.a, ctx) }) or ({ _emit(expr.b, ctx) })"
    if isinstance(expr, Not):
        return f"not ({ _emit(expr.a, ctx) })"
    if isinstance(expr, If):
        cond = _emit(expr.cond, ctx)
        temp = ctx.new_temp()
        ctx.add_line(f"if {cond}:")
        ctx.push_indent()
        then_expr = _emit(expr.then_, ctx)
        ctx.add_line(f"{temp} = {then_expr}")
        ctx.pop_indent()
        ctx.add_line("else:")
        ctx.push_indent()
        else_expr = _emit(expr.else_, ctx)
        ctx.add_line(f"{temp} = {else_expr}")
        ctx.pop_indent()
        return temp
    if isinstance(expr, Map):
        lst = _emit(expr.lst, ctx)
        idx_name = ctx.new_temp(prefix="i")
        ctx.add_line(f"for {idx_name} in range(len({lst})):")
        ctx.push_indent()
        ctx.map_stack.append(f"{lst}[{idx_name}]")
        body = _emit(expr.fn, ctx)
        ctx.add_line(f"{lst}[{idx_name}] = {body}")
        ctx.map_stack.pop()
        ctx.pop_indent()
        return lst
    if isinstance(expr, Lt):
        return f"({ _emit(expr.a, ctx) }) < ({ _emit(expr.b, ctx) })"
    raise NotImplementedError(f"Unsupported expression: {expr}")


def transpile_to_python(expr: Expr, arity: int) -> str:
    """Generate executable Python code implementing `expr`."""

    ctx = TranspileCtx(arity=arity)
    result_expr = _emit(expr, ctx)

    params = ", ".join(f"a{i+1}" for i in range(arity))
    header = f"def f({params}):"

    body_lines: List[str] = []
    body_lines.extend(ctx.init_lines)
    body_lines.extend(ctx.lines)
    body_lines.append("    return " + result_expr)

    body = "\n".join(body_lines)
    return "\n".join([header, body])
