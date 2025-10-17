# inputs.py
import random

def sample_inputs(arity:int, trials:int=3):
    def one_list():
        n = random.randint(3,5)
        return [random.randint(0,5) for _ in range(n)]
    data = []
    for _ in range(trials):
        if arity == 1:
            data.append((one_list(),))
        elif arity == 2:
            data.append((one_list(), one_list()))
    return data
