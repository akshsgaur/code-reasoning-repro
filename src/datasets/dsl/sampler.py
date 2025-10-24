"""Probabilistic grammar for sampling DSL-List programs."""

from __future__ import annotations

import random
from typing import Dict

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
    Lt,
    Not,
    Or,
    Scope,
    Tail,
    Ty,
    Var,
)

MAP_ELEMENT_NAME = "elem"
INT_RANGE = range(0, 6)

WEIGHTS: Dict[str, float] = {
    "append": 1.0,
    "extend": 0.05,
    "init": 1.0,
    "tail": 1.0,
    "if": 1.0,
    "map": 5.0,
    "var": 1.0,
    "empty": 1.0,
    "length": 1.0,
    "index": 1.0,
    "intlit": 1.0,
    "eq": 1.0,
    "lt": 1.0,
    "gt": 1.0,
    "and": 1.0,
    "or": 1.0,
    "not": 1.0,
    "boollit": 1.0,
}


def _weighted_choice(pairs, rng: random.Random):
    total = sum(weight for _, weight in pairs)
    r = rng.random() * total
    upto = 0.0
    for item, weight in pairs:
        upto += weight
        if upto >= r:
            return item
    return pairs[-1][0]


def _ensure_non_empty_list_expr(expr_fn, rng: random.Random, vars_in_scope: Scope, depth: int) -> Expr:
    """Resample until the returned list expression is not the Empty literal."""

    for _ in range(10):
        candidate = expr_fn()
        if not isinstance(candidate, Empty):
            return candidate
    # Fall back to a variable if available; otherwise accept the last draw.
    for var in vars_in_scope:
        if var.name.startswith("a"):
            return var
    return candidate


def sample_expr(ty: Ty, depth: int, vars_in_scope: Scope, rng: random.Random) -> Expr:
    """Sample an expression of type `ty` with maximum recursion `depth`."""

    if depth <= 0:
        if ty == Ty.INT:
            return IntLit(rng.choice(list(INT_RANGE)))
        if ty == Ty.BOOL:
            return BoolLit(rng.choice([True, False]))
        if ty == Ty.LIST_INT:
            candidates = []
            for var in vars_in_scope:
                if var.name.startswith("a"):
                    candidates.append((var, WEIGHTS["var"]))
            candidates.append((Empty(), WEIGHTS["empty"]))
            return _weighted_choice(candidates, rng)
        raise ValueError(f"Unsupported base type: {ty}")

    if ty == Ty.LIST_INT:
        choices = [
            ("append", WEIGHTS["append"]),
            ("extend", WEIGHTS["extend"]),
            ("init", WEIGHTS["init"]),
            ("tail", WEIGHTS["tail"]),
            ("if", WEIGHTS["if"]),
            ("map", WEIGHTS["map"]),
            ("var", WEIGHTS["var"]),
            ("empty", WEIGHTS["empty"]),
        ]
        kind = _weighted_choice(choices, rng)
        if kind == "append":
            value = sample_expr(Ty.INT, depth - 1, vars_in_scope, rng)
            lst = sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng)
            return Append(value, lst)
        if kind == "extend":
            a_expr = sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng)
            b_expr = sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng)
            if repr(a_expr) == repr(b_expr):
                return sample_expr(ty, depth - 1, vars_in_scope, rng)
            return Extend(a_expr, b_expr)
        if kind == "init":
            lst = _ensure_non_empty_list_expr(
                lambda: sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng),
                rng,
                vars_in_scope,
                depth - 1,
            )
            return Init(lst)
        if kind == "tail":
            lst = _ensure_non_empty_list_expr(
                lambda: sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng),
                rng,
                vars_in_scope,
                depth - 1,
            )
            return Tail(lst)
        if kind == "if":
            cond = sample_expr(Ty.BOOL, depth - 1, vars_in_scope, rng)
            then_ = sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng)
            else_ = sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng)
            if repr(then_) == repr(else_):
                return sample_expr(ty, depth - 1, vars_in_scope, rng)
            return If(cond, then_, else_)
        if kind == "map":
            body = sample_expr(
                Ty.INT,
                depth - 1,
                vars_in_scope + (Var(MAP_ELEMENT_NAME),),
                rng,
            )
            lst = _ensure_non_empty_list_expr(
                lambda: sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng),
                rng,
                vars_in_scope,
                depth - 1,
            )
            return Map(body, lst)
        if kind == "var":
            candidates = [var for var in vars_in_scope if var.name.startswith("a")]
            if not candidates:
                return Empty()
            return rng.choice(candidates)
        return Empty()

    if ty == Ty.INT:
        choices = [
            ("length", WEIGHTS["length"]),
            ("index", WEIGHTS["index"]),
            ("intlit", WEIGHTS["intlit"]),
            ("if", WEIGHTS["if"]),
        ]
        kind = _weighted_choice(choices, rng)
        if kind == "length":
            lst = _ensure_non_empty_list_expr(
                lambda: sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng),
                rng,
                vars_in_scope,
                depth - 1,
            )
            return Length(lst)
        if kind == "index":
            lst = _ensure_non_empty_list_expr(
                lambda: sample_expr(Ty.LIST_INT, depth - 1, vars_in_scope, rng),
                rng,
                vars_in_scope,
                depth - 1,
            )
            return Index(IntLit(-1), lst)
        if kind == "if":
            cond = sample_expr(Ty.BOOL, depth - 1, vars_in_scope, rng)
            then_ = sample_expr(Ty.INT, depth - 1, vars_in_scope, rng)
            else_ = sample_expr(Ty.INT, depth - 1, vars_in_scope, rng)
            if repr(then_) == repr(else_):
                return sample_expr(ty, depth - 1, vars_in_scope, rng)
            return If(cond, then_, else_)
        return IntLit(rng.choice(list(INT_RANGE)))

    if ty == Ty.BOOL:
        choices = [
            ("eq", WEIGHTS["eq"]),
            ("lt", WEIGHTS["lt"]),
            ("gt", WEIGHTS["gt"]),
            ("and", WEIGHTS["and"]),
            ("or", WEIGHTS["or"]),
            ("not", WEIGHTS["not"]),
            ("boollit", WEIGHTS["boollit"]),
        ]
        kind = _weighted_choice(choices, rng)
        if kind in {"eq", "lt", "gt"}:
            left = sample_expr(Ty.INT, depth - 1, vars_in_scope, rng)
            right = sample_expr(Ty.INT, depth - 1, vars_in_scope, rng)
            if repr(left) == repr(right):
                return sample_expr(ty, depth - 1, vars_in_scope, rng)
            if kind == "eq":
                return Eq(left, right)
            if kind == "lt":
                return Lt(left, right)
            return Gt(left, right)
        if kind == "and":
            left = sample_expr(Ty.BOOL, depth - 1, vars_in_scope, rng)
            right = sample_expr(Ty.BOOL, depth - 1, vars_in_scope, rng)
            if repr(left) == repr(right):
                return sample_expr(ty, depth - 1, vars_in_scope, rng)
            return And(left, right)
        if kind == "or":
            left = sample_expr(Ty.BOOL, depth - 1, vars_in_scope, rng)
            right = sample_expr(Ty.BOOL, depth - 1, vars_in_scope, rng)
            if repr(left) == repr(right):
                return sample_expr(ty, depth - 1, vars_in_scope, rng)
            return Or(left, right)
        if kind == "not":
            return Not(sample_expr(Ty.BOOL, depth - 1, vars_in_scope, rng))
        return BoolLit(rng.choice([True, False]))

    raise ValueError(f"Unsupported type {ty}")
