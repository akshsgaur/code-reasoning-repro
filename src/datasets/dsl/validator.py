"""Syntactic and runtime validation for sampled DSL programs."""

from __future__ import annotations

import copy
import traceback
from typing import Iterable, Set, Tuple

from .ast import And, Empty, Eq, Expr, Extend, Gt, If, Index, Init, IntLit, Length, Lt, Map, Not, Or, Tail, Var


def _iter_children(expr: Expr) -> Iterable[Tuple[str, Expr]]:
    for field, value in expr.__dict__.items():
        if isinstance(value, Expr):
            yield field, value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Expr):
                    yield field, item


def violates_syntactic_constraints(expr: Expr) -> Tuple[bool, Set[str]]:
    """Check the additional constraints described in Appendix A.1.1."""

    violations = False
    used_params: Set[str] = set()

    def visit(node: Expr, parent: Expr | None = None, attr: str | None = None) -> None:
        nonlocal violations
        if violations:
            return

        if isinstance(node, Var):
            if node.name.startswith("a"):
                used_params.add(node.name)
            return

        if isinstance(node, (Eq, Lt, Gt)):
            if isinstance(node.a, IntLit):
                violations = True
                return
            if repr(node.a) == repr(node.b):
                violations = True
                return

        if isinstance(node, And) and repr(node.a) == repr(node.b):
            violations = True
            return

        if isinstance(node, Or) and repr(node.a) == repr(node.b):
            violations = True
            return

        if isinstance(node, Length) and isinstance(node.lst, Empty):
            violations = True
            return

        if isinstance(node, Map) and isinstance(node.lst, Empty):
            violations = True
            return

        if isinstance(node, Extend):
            if isinstance(node.b, Empty):
                violations = True
                return
            if repr(node.a) == repr(node.b):
                violations = True
                return

        if isinstance(node, (Init, Tail)) and isinstance(node.lst, Empty):
            violations = True
            return

        if isinstance(node, Index):
            if isinstance(node.lst, Empty):
                violations = True
                return
            if not isinstance(node.i, IntLit) or node.i.value != -1:
                violations = True
                return

        if isinstance(node, If) and repr(node.then_) == repr(node.else_):
            violations = True
            return

        if isinstance(node, IntLit) and node.value == -1:
            if not (isinstance(parent, Index) and attr == "i"):
                violations = True
                return

        for child_attr, child in _iter_children(node):
            visit(child, node, child_attr)

    visit(expr)
    return violations, used_params


def _run_program(py_src: str, inputs):
    namespace = {}
    try:
        exec(py_src, namespace, namespace)
        func = namespace["f"]
        outputs = []
        for args in inputs:
            safe_args = [copy.deepcopy(arg) for arg in args]
            outputs.append(func(*safe_args))
        return True, outputs
    except Exception:
        return False, traceback.format_exc()


def is_valid(py_src: str, inputs, expr: Expr, arity: int) -> Tuple[bool, list]:
    bad, used_params = violates_syntactic_constraints(expr)
    if bad:
        return False, []
    expected_params = {f"a{i+1}" for i in range(arity)}
    if not expected_params.issubset(used_params):
        return False, []
    ok, result = _run_program(py_src, inputs)
    if not ok:
        return False, []
    outputs = result
    if len(outputs) < 2:
        return False, []
    if all(outputs[0] == out for out in outputs[1:]):
        return False, []
    return True, outputs
