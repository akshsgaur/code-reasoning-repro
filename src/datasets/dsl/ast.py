"""Abstract syntax tree definitions for the DSL-List language.

These mirror the primitives used by DeepSynth and the description in the paper.
The DSL is first-order and purely functional, but several primitives mutate
lists in place when translated to Python (e.g. ``append``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union


class Ty(Enum):
    """Finite set of types supported by the DSL."""

    INT = "int"
    BOOL = "bool"
    LIST_INT = "List(int)"


@dataclass(frozen=True)
class Var:
    name: str  # parameters (a1, a2, …), locals (v1, v2, …) or the implicit map var


@dataclass(frozen=True)
class IntLit:
    value: int


@dataclass(frozen=True)
class BoolLit:
    value: bool


@dataclass(frozen=True)
class Empty:
    """Construct a fresh empty list."""


@dataclass(frozen=True)
class Append:
    x: "Expr"
    lst: "Expr"


@dataclass(frozen=True)
class Extend:
    a: "Expr"
    b: "Expr"


@dataclass(frozen=True)
class Init:
    lst: "Expr"


@dataclass(frozen=True)
class Tail:
    lst: "Expr"


@dataclass(frozen=True)
class Length:
    lst: "Expr"


@dataclass(frozen=True)
class Index:
    i: "Expr"
    lst: "Expr"


@dataclass(frozen=True)
class If:
    cond: "Expr"
    then_: "Expr"
    else_: "Expr"


@dataclass(frozen=True)
class Eq:
    a: "Expr"
    b: "Expr"


@dataclass(frozen=True)
class Lt:
    a: "Expr"
    b: "Expr"


@dataclass(frozen=True)
class Gt:
    a: "Expr"
    b: "Expr"


@dataclass(frozen=True)
class And:
    a: "Expr"
    b: "Expr"


@dataclass(frozen=True)
class Or:
    a: "Expr"
    b: "Expr"


@dataclass(frozen=True)
class Not:
    a: "Expr"


@dataclass(frozen=True)
class Map:
    fn: "Expr"
    lst: "Expr"


Expr = Union[
    Var,
    IntLit,
    BoolLit,
    Empty,
    Append,
    Extend,
    Init,
    Tail,
    Length,
    Index,
    If,
    Eq,
    Lt,
    Gt,
    And,
    Or,
    Not,
    Map,
]

Scope = Tuple[Var, ...]

