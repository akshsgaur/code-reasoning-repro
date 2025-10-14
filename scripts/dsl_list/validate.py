import types, traceback
from dsl import *  # import your Expr classes

# -------------------------------
# Phase 1 — Syntactic Constraints
# -------------------------------
def violates_syntactic_constraints(expr, seen_params=None):
    if seen_params is None:
        seen_params = set()

    # track usage of parameters
    if isinstance(expr, Var):
        seen_params.add(expr.name)
        return False, seen_params

    # (1) first arg of comparison must not be int literal
    if isinstance(expr, (Eq, Lt, Gt)):
        if isinstance(expr.a, IntLit):
            return True, seen_params
        # (1a) same expr on both sides (v1 == v1)
        if repr(expr.a) == repr(expr.b):
            return True, seen_params

    # (2) last arg to extend, length, map must not be Empty
    if isinstance(expr, (Extend, Length, Map)):
        lst = expr.b if isinstance(expr, Extend) else expr.lst
        if isinstance(lst, Empty):
            return True, seen_params

    # (3) -1 literal only allowed inside Index
    if isinstance(expr, IntLit) and expr.value == -1:
        return True, seen_params  # flagged unless caught below
    if isinstance(expr, Index):
        if isinstance(expr.i, IntLit) and expr.i.value == -1:
            # override: allowed
            pass
        elif isinstance(expr.i, IntLit) and expr.i.value == -1:
            return True, seen_params

    # (4) list cannot extend itself
    if isinstance(expr, Extend):
        if repr(expr.a) == repr(expr.b):
            return True, seen_params

    # (5) same expr on both sides of if branches
    if isinstance(expr, If):
        if repr(expr.then_) == repr(expr.else_):
            return True, seen_params

    # recursively check sub-expressions
    for child in expr.__dict__.values():
        if isinstance(child, Expr):
            bad, seen_params = violates_syntactic_constraints(child, seen_params)
            if bad:
                return True, seen_params
        elif isinstance(child, (list, tuple)):
            for c in child:
                if isinstance(c, Expr):
                    bad, seen_params = violates_syntactic_constraints(c, seen_params)
                    if bad:
                        return True, seen_params

    return False, seen_params


# -------------------------------
# Phase 2 — Runtime Validation
# -------------------------------
def run_func(py_src: str, inputs):
    ns = {}
    try:
        exec(py_src, ns, ns)
        f = ns['f']
        outs = []
        for args in inputs:
            # copy lists so we can safely mutate
            outs.append(f(*[list(a) if isinstance(a, list) else a for a in args]))
        return True, outs, ""
    except Exception:
        return False, None, traceback.format_exc()


# -------------------------------
# Main API
# -------------------------------
def is_valid(py_src: str, inputs, expr=None, arity=None):
    # 1. syntactic
    if expr is not None:
        bad, used_params = violates_syntactic_constraints(expr)
        if bad:
            return False, None
        # (6) must contain all function parameters
        if arity and len(used_params) < arity:
            return False, None

    # 2. runtime
    ok, outs, _ = run_func(py_src, inputs)
    if not ok or not outs:
        return False, None

    # (7) must not produce same output for all inputs
    all_same = all(outs[0] == o for o in outs)
    return (not all_same), outs
