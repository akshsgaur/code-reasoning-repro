"""Input sampling utilities."""

from __future__ import annotations

import random
from typing import List


def _sample_list(rng: random.Random) -> List[int]:
    length = rng.randint(3, 5)
    return [rng.randint(0, 5) for _ in range(length)]


def sample_inputs(arity: int, trials: int, rng: random.Random) -> List[List[List[int]]]:
    """Generate `trials` input tuples for a program of arity `arity`."""

    data: List[List[List[int]]] = []
    for _ in range(trials):
        row: List[List[int]] = []
        for _ in range(arity):
            row.append(_sample_list(rng))
        data.append(row)
    return data

