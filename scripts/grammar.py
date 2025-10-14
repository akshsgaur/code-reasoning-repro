# grammar.py
import random
from make_dsl import *

WEIGHTS = {
    'if'    : 1.0,
    'map'   : 5.0,     # heavier
    'extend': 0.05,    # much rarer
    'append': 1.0, 'init':1.0, 'tail':1.0,
    'length':1.0, 'index':1.0,
    'eq':1.0,'lt':1.0,'gt':1.0,'and':1.0,'or':1.0,'not':1.0,
    'empty':1.0, 'intlit':1.0, 'var':1.0
}

INT_RANGE   = range(-1, 6)  # [-1,5]
LIST_RANGE  = range(3, 6)   # list lengths 3..5 (for input gen)

def weighted_choice(pairs):
    total = sum(w for _, w in pairs)
    r = random.random() * total
    upto = 0
    for item, w in pairs:
        upto += w
        if upto >= r: return item
    return pairs[-1][0]

def sample_expr(ty: Ty, depth:int, vars_in_scope:Tuple[Var,...]) -> Expr:
    """
    Sample an Expr of 'ty' with max recursion 'depth' and using 'vars_in_scope':
    a1, a2 (lists) and any v1..vn introduced by empties/ifs.
    """
    if depth == 0:
        # base: literals or variables of correct type
        if ty == Ty.INT:
            return IntLit(random.choice(list(INT_RANGE)))
        if ty == Ty.BOOL:
            # pick a relational op over ints 50% or boolean literal
            if random.random() < 0.5:
                return BoolLit(random.choice([True, False]))
            # fallthrough to small relational
            a = IntLit(random.choice(list(INT_RANGE)))
            b = IntLit(random.choice(list(INT_RANGE)))
            # avoid (0==0)-like trivialities
            while isinstance(a, IntLit) and isinstance(b, IntLit) and a.value == b.value:
                b = IntLit(random.choice(list(INT_RANGE)))
            return Eq(a,b)
        if ty == Ty.LIST_INT:
            # allow empty or one of the list vars
            choices = []
            if any(isinstance(v, Var) and v.name.startswith('a') for v in vars_in_scope):
                for v in vars_in_scope:
                    if v.name.startswith('a'):
                        choices.append((v, WEIGHTS['var']))
            choices.append((Empty(), WEIGHTS['empty']))
            return weighted_choice(choices)
        raise ValueError

    # recursive cases by desired type
    if ty == Ty.LIST_INT:
        choices = [
            ('append', WEIGHTS['append']),
            ('extend', WEIGHTS['extend']),
            ('init',   WEIGHTS['init']),
            ('tail',   WEIGHTS['tail']),
            ('if',     WEIGHTS['if']),
            ('map',    WEIGHTS['map']),
            ('var',    WEIGHTS['var']),
            ('empty',  WEIGHTS['empty'])
        ]
        kind = weighted_choice(choices)
        if kind == 'append':
            x   = sample_expr(Ty.INT, depth-1, vars_in_scope)
            lst = sample_expr(Ty.LIST_INT, depth-1, vars_in_scope)
            return Append(x,lst)
        if kind == 'extend':
            a = sample_expr(Ty.LIST_INT, depth-1, vars_in_scope)
            b = sample_expr(Ty.LIST_INT, depth-1, vars_in_scope)
            # avoid self-extend
            if isinstance(a, Var) and isinstance(b, Var) and a.name == b.name:
                return sample_expr(ty, depth-1, vars_in_scope)
            return Extend(a,b)
        if kind == 'init':  return Init(sample_expr(Ty.LIST_INT, depth-1, vars_in_scope))
        if kind == 'tail':  return Tail(sample_expr(Ty.LIST_INT, depth-1, vars_in_scope))
        if kind == 'if':
            c = sample_expr(Ty.BOOL, depth-1, vars_in_scope)
            t = sample_expr(Ty.LIST_INT, depth-1, vars_in_scope)
            e = sample_expr(Ty.LIST_INT, depth-1, vars_in_scope)
            # avoid identical branches
            if repr(t)==repr(e): return sample_expr(ty, depth-1, vars_in_scope)
            return If(c,t,e)
        if kind == 'map':
            # fn: t0->t1; we’ll do int->int to stay in List(int)
            element = Var('elem')           # implicit map var
            body = sample_expr(Ty.INT, depth-1, vars_in_scope+(element,))
            lst  = sample_expr(Ty.LIST_INT, depth-1, vars_in_scope)
            return Map(body,lst)
        if kind == 'var':
            # choose from in-scope list vars (a1,a2,vk’s)
            pool = [v for v in vars_in_scope if v.name.startswith(('a','v'))]
            if not pool: return Empty()
            return random.choice(pool)
        if kind == 'empty':
            return Empty()

    if ty == Ty.INT:
        choices = [('length',WEIGHTS['length'] ),('index',WEIGHTS['index']),
                   ('intlit',WEIGHTS['intlit']),('if',WEIGHTS['if'])]
        kind = weighted_choice(choices)
        if kind == 'length':
            return Length(sample_expr(Ty.LIST_INT, depth-1, vars_in_scope))
        if kind == 'index':
            lst = sample_expr(Ty.LIST_INT, depth-1, vars_in_scope)
            i   = sample_expr(Ty.INT, depth-1, vars_in_scope)
            return Index(i,lst)
        if kind == 'if':
            c = sample_expr(Ty.BOOL, depth-1, vars_in_scope)
            t = sample_expr(Ty.INT, depth-1, vars_in_scope)
            e = sample_expr(Ty.INT, depth-1, vars_in_scope)
            if repr(t)==repr(e): return sample_expr(ty, depth-1, vars_in_scope)
            return If(c,t,e)
        return IntLit(random.choice(list(INT_RANGE)))

    if ty == Ty.BOOL:
        choices = [('eq',WEIGHTS['eq']),('lt',WEIGHTS['lt']),('gt',WEIGHTS['gt']),
                   ('and',WEIGHTS['and']),('or',WEIGHTS['or']),('not',WEIGHTS['not'])]
        kind = weighted_choice(choices)
        if kind in ('eq','lt','gt'):
            a = sample_expr(Ty.INT, depth-1, vars_in_scope)
            b = sample_expr(Ty.INT, depth-1, vars_in_scope)
            if kind=='eq': return Eq(a,b)
            if kind=='lt': return Lt(a,b)
            return Gt(a,b)
        if kind == 'and': return And(sample_expr(Ty.BOOL, depth-1, vars_in_scope),
                                     sample_expr(Ty.BOOL, depth-1, vars_in_scope))
        if kind == 'or' : return Or( sample_expr(Ty.BOOL, depth-1, vars_in_scope),
                                     sample_expr(Ty.BOOL, depth-1, vars_in_scope))
        return Not(sample_expr(Ty.BOOL, depth-1, vars_in_scope))

    raise NotImplementedError
