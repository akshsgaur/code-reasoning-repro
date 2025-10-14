# validate.py
import types, traceback

def run_func(py_src:str, inputs):
    ns = {}
    try:
        exec(py_src, ns, ns)
        f = ns['f']
        outs = []
        for args in inputs:
            outs.append(f(*[list(a) if isinstance(a,list) else a for a in args]))
        return True, outs, ""
    except Exception as e:
        return False, None, traceback.format_exc()

def is_valid(py_src:str, inputs):
    ok, outs, _ = run_func(py_src, inputs)
    if not ok: return False, None
    # non-degenerate outputs
    all_same = all(outs[0] == o for o in outs)
    return (not all_same), outs
