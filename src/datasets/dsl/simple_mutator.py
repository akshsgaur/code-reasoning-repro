#!/usr/bin/env python3
"""
Simple mutation operators adapted for DSL-generated Python functions.

We reuse the same mutation categories as the LeetCode pipeline:
1. Arithmetic Operator: + ↔ -, * ↔ /, // ↔ *, % ↔ //
2. Relational Operator: < ↔ <=, > ↔ >=, == ↔ !=
3. Logical Operator: and ↔ or
4. Keyword: continue ↔ break, True ↔ False (rare in DSL but supported)
5. Numerical Literal: n → n±1
"""

import ast
import copy
from typing import List, Tuple


class SimpleMutator(ast.NodeTransformer):
    """Apply a single mutation to a Python AST."""

    def __init__(self, mutation_type: str, mutation_index: int):
        self.mutation_type = mutation_type
        self.mutation_index = mutation_index
        self.current_index = 0
        self.mutation_applied = False

    def visit_BinOp(self, node):
        if self.mutation_applied:
            return node

        if self.mutation_type == "arithmetic":
            mutations = {
                ast.Add: ast.Sub,
                ast.Sub: ast.Add,
                ast.Mult: ast.Div,
                ast.Div: ast.Mult,
                ast.FloorDiv: ast.Mult,
                ast.Mod: ast.FloorDiv,
            }
            if type(node.op) in mutations:
                if self.current_index == self.mutation_index:
                    node = copy.deepcopy(node)
                    node.op = mutations[type(node.op)]()
                    self.mutation_applied = True
                    return node
                self.current_index += 1

        return self.generic_visit(node)

    def visit_Compare(self, node):
        if self.mutation_applied:
            return node

        if self.mutation_type == "relational":
            mutations = {
                ast.Lt: ast.LtE,
                ast.LtE: ast.Lt,
                ast.Gt: ast.GtE,
                ast.GtE: ast.Gt,
                ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq,
            }
            for i, op in enumerate(node.ops):
                if type(op) in mutations:
                    if self.current_index == self.mutation_index:
                        node = copy.deepcopy(node)
                        node.ops[i] = mutations[type(op)]()
                        self.mutation_applied = True
                        return node
                    self.current_index += 1

        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        if self.mutation_applied:
            return node

        if self.mutation_type == "logical":
            mutations = {ast.And: ast.Or, ast.Or: ast.And}
            if type(node.op) in mutations:
                if self.current_index == self.mutation_index:
                    node = copy.deepcopy(node)
                    node.op = mutations[type(node.op)]()
                    self.mutation_applied = True
                    return node
                self.current_index += 1

        return self.generic_visit(node)

    def visit_Continue(self, node):
        if self.mutation_applied:
            return node

        if self.mutation_type == "keyword":
            if self.current_index == self.mutation_index:
                self.mutation_applied = True
                return ast.Break()
            self.current_index += 1
        return node

    def visit_Break(self, node):
        if self.mutation_applied:
            return node

        if self.mutation_type == "keyword":
            if self.current_index == self.mutation_index:
                self.mutation_applied = True
                return ast.Continue()
            self.current_index += 1
        return node

    def visit_Constant(self, node):
        if self.mutation_applied:
            return node

        if self.mutation_type == "keyword" and isinstance(node.value, bool):
            if self.current_index == self.mutation_index:
                node = copy.deepcopy(node)
                node.value = not node.value
                self.mutation_applied = True
                return node
            self.current_index += 1

        if self.mutation_type == "number" and isinstance(node.value, (int, float)):
            if self.current_index == self.mutation_index:
                node = copy.deepcopy(node)
                node.value = node.value + 1 if self.mutation_index % 2 == 0 else node.value - 1
                self.mutation_applied = True
                return node
            self.current_index += 1

        return node


def generate_all_mutants(code: str) -> List[Tuple[str, str, int]]:
    """Generate all possible mutants of code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    mutants: List[Tuple[str, str, int]] = []
    mutation_types = ["arithmetic", "relational", "logical", "keyword", "number"]

    for mutation_type in mutation_types:
        mutation_index = 0
        while mutation_index < 100:
            mutator = SimpleMutator(mutation_type, mutation_index)
            mutated_tree = mutator.visit(copy.deepcopy(tree))
            if not mutator.mutation_applied:
                break
            try:
                mutated_code = ast.unparse(mutated_tree)
                if mutated_code != code:
                    mutants.append((mutated_code, mutation_type, mutation_index))
            except Exception:
                pass
            mutation_index += 1
    return mutants
