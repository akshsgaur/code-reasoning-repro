# sample_dataset.py
import json, random
from dsl import Var, Ty
from grammar import sample_expr
from transpile import transpile_to_python
from inputs import sample_inputs
from validate import is_valid
import argparse

def sample_programs(n:int, depth:int, arity:int, seed=0):
    random.seed()
    out = []
    printed_debug = False  # only print for the first valid program

    while len(out) < n:
        vars_scope = tuple([Var(f"a{i+1}") for i in range(arity)])  # parameters
        expr = sample_expr(Ty.LIST_INT, depth, vars_scope)
        py = transpile_to_python(expr, arity=arity)
        ins = sample_inputs(arity, trials=3)
        ok, outs = is_valid(py, ins)

        # only debug the first valid case
        if not printed_debug and ok:
            print("\n================= DEBUG: FIRST TEST CASE =================")
            print(f"Arity: {arity}, Depth: {depth}, Seed: {seed}")
            print(f"Vars Scope: {[v for v in vars_scope]}")
            print(f"Sampled Expression (DSL): {repr(expr)}")
            print(f"Transpiled Python Code:\n{py}")
            print(f"Sampled Inputs: {ins}")
            print(f"Outputs: {outs}")
            print("===========================================================\n")
            printed_debug = True

        if not ok:
            continue

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--arity", type=int, default=1)
    args = parser.parse_args()

    dataset = sample_programs(n=args.n, depth=args.depth, arity=args.arity, seed=None)
    with open(f"../data/dsl_list.jsonl","w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")
