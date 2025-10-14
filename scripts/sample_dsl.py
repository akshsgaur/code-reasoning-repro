# sample_dataset.py
import json, random
from make_dsl import Var, Ty
from grammar import sample_expr
from transpile import transpile_to_python
from inputs import sample_inputs
from validate import is_valid

def sample_programs(n:int, depth:int, arity:int, seed=0, bin_by_loc=False):
    random.seed(seed)
    out = []
    while len(out) < n:
        vars_scope = tuple([Var(f"a{i+1}") for i in range(arity)])  # parameters
        expr = sample_expr(Ty.LIST_INT, depth, vars_scope)
        py = transpile_to_python(expr, arity=arity)
        ins = sample_inputs(arity, trials=3)
        ok, outs = is_valid(py, ins)
        if not ok: continue
        out.append({
            "arity": arity,
            "depth": depth,
            "dsl": repr(expr),
            "python": py,
            "inputs": ins,
            "outputs": outs
        })
    return out

if __name__ == "__main__":
    dataset = []
    # replicate paper’s idea: for each (type,depth) combination, sample 1000 valid, then downselect
    combos = [(1,4),(1,5),(2,4),(2,5)]
    for arity, depth in combos:
        dataset += sample_programs(n=100, depth=depth, arity=arity, seed=arity*100+depth)
    with open("data/dsl_list.jsonl","w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")
