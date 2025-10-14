# dsl.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Optional, Tuple

# Types
class Ty(Enum):
    INT = "int"
    BOOL = "bool"
    LIST_INT = "List(int)"

# AST nodes
@dataclass
class Var: name: str                 # a1, a2, v1, ...
@dataclass
class IntLit: value: int
@dataclass
class BoolLit: value: bool
@dataclass
class Empty: pass                    # empty : L(t0)
@dataclass
class Append:                        # append: t0 → L(t0) → L(t0)
    x: 'Expr'
    lst: 'Expr'
@dataclass
class Extend:                        # extend: L(t0) → L(t0) → L(t0)
    a: 'Expr'
    b: 'Expr'
@dataclass
class Init:  lst: 'Expr'            # init:  L(t0) → L(t0)  (pop last)
@dataclass
class Tail:  lst: 'Expr'            # tail:  L(t0) → L(t0)  (pop first)
@dataclass
class Length: lst: 'Expr'           # length: L(t0) → int
@dataclass
class Index:                         # index: int → L(t0) → t0
    i: 'Expr'
    lst: 'Expr'
@dataclass
class If:                            # if: bool → t → t → t
    cond: 'Expr'
    then_: 'Expr'
    else_: 'Expr'
# comparisons + boolean ops
@dataclass
class Eq:   a:'Expr'; b:'Expr'
@dataclass
class Lt:   a:'Expr'; b:'Expr'
@dataclass
class Gt:   a:'Expr'; b:'Expr'
@dataclass
class And:  a:'Expr'; b:'Expr'
@dataclass
class Or:   a:'Expr'; b:'Expr'
@dataclass
class Not:  a:'Expr'
@dataclass
class Map:                          # map: (t0→t1) → L(t0) → L(t1)
    fn: 'Expr'     # lambda-body as an Expr expecting implicit var
    lst:'Expr'

Expr = Union[Var, IntLit, BoolLit, Empty, Append, Extend, Init, Tail,
             Length, Index, If, Eq, Lt, Gt, And, Or, Not, Map]
