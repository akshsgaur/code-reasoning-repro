# transpile.py
from make_dsl import *

class TranspileCtx:
    def __init__(self, params:int):
        self.lines = []
        self.locals = []
        self.counter = 0
        self.params = params
        self.result_expr = None
        self.map_elem = None  # current implicit element name within a map

    def new_local(self):
        self.counter += 1
        name = f"v{self.counter}"
        self.locals.append(name)
        return name

def to_py(expr: Expr, ctx: TranspileCtx) -> str:
    # returns a Python expression string that represents the value of expr
    if isinstance(expr, Var):
        # params a1,a2 exist; local vars are v*
        return expr.name
    if isinstance(expr, IntLit): return str(expr.value)
    if isinstance(expr, BoolLit): return "True" if expr.value else "False"
    if isinstance(expr, Empty):
        name = ctx.new_local()
        # initialize later at function start as []
        return name
    if isinstance(expr, Append):
        lst = to_py(expr.lst, ctx)
        x   = to_py(expr.x, ctx)
        ctx.lines.append(f"{lst}.append({x})")
        return lst
    if isinstance(expr, Extend):
        a = to_py(expr.a, ctx)
        b = to_py(expr.b, ctx)
        ctx.lines.append(f"{b}.extend({a})")
        return b
    if isinstance(expr, Init):
        lst = to_py(expr.lst, ctx)
        ctx.lines.append(f"{lst}.pop()")
        return lst
    if isinstance(expr, Tail):
        lst = to_py(expr.lst, ctx)
        ctx.lines.append(f"{lst}.pop(0)")
        return lst
    if isinstance(expr, Length):
        lst = to_py(expr.lst, ctx)
        return f"len({lst})"
    if isinstance(expr, Index):
        lst = to_py(expr.lst, ctx)
        i   = to_py(expr.i, ctx)
        return f"{lst}[{i}]"
    if isinstance(expr, Eq): return f"({to_py(expr.a,ctx)}) == ({to_py(expr.b,ctx)})"
    if isinstance(expr, Lt): return f"({to_py(expr.a,ctx)}) <  ({to_py(expr.b,ctx)})"
    if isinstance(expr, Gt): return f"({to_py(expr.a,ctx)}) >  ({to_py(expr.b,ctx)})"
    if isinstance(expr, And): return f"({to_py(expr.a,ctx)}) and ({to_py(expr.b,ctx)})"
    if isinstance(expr, Or):  return f"({to_py(expr.a,ctx)}) or  ({to_py(expr.b,ctx)})"
    if isinstance(expr, Not): return f"not ({to_py(expr.a,ctx)})"
    if isinstance(expr, If):
        cond = to_py(expr.cond, ctx)
        then_expr = to_py(expr.then_, ctx)
        else_expr = to_py(expr.else_, ctx)
        name = ctx.new_local()
        ctx.lines.append(f"if {cond}:")
        ctx.lines.append(f"    {name} = {then_expr}")
        ctx.lines.append(f"else:")
        ctx.lines.append(f"    {name} = {else_expr}")
        return name
    if isinstance(expr, Map):
        lst = to_py(expr.lst, ctx)
        name = ctx.new_local()     # list being mapped (same underlying list)
        # we treat map as element-wise assignment into that list
        ctx.lines.append(f"for i in range(len({lst})):")
        prev = ctx.map_elem
        ctx.map_elem = f"{lst}[i]"
        body = to_py(expr.fn, ctx)               # compute mapped value for element
        ctx.lines.append(f"    {lst}[i] = {body}")
        ctx.map_elem = prev
        return lst
    raise NotImplementedError

def transpile_to_python(expr: Expr, arity:int) -> str:
    ctx = TranspileCtx(params=arity)
    params = ", ".join([f"a{i+1}" for i in range(arity)])
    header = f"def f({params}):"
    # initialize locals for empties used anywhere
    # we’ll scan during code emission: just predeclare as tuple:
    # collect at the end via simple parse of ctx.locals
    result = to_py(expr, ctx)
    # hoist locals init (only for names in ctx.locals)
    locals_decl = ""
    if ctx.locals:
        uniq = sorted(set(ctx.locals), key=lambda s:int(s[1:]))
        locals_decl = f"    {', '.join(uniq)} = " + ", ".join("[]"*len(uniq) or []) + "\n"
        if uniq:
            locals_decl = f"    {', '.join(uniq)} = " + ", ".join(["[]"]*len(uniq)) + "\n"
    body = "\n".join("    "+line for line in ctx.lines)
    ret  = f"    return {result}"
    return "\n".join([header, locals_decl + body, ret])
