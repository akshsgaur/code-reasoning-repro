#!/usr/bin/env python3
"""
Simple mutation operators based on paper Table 1 (page 6)

Mutation Operators:
1. Arithmetic Operator: + ↔ -, * ↔ /, // ↔ *, % ↔ //
2. Relational Operator: < ↔ <=, > ↔ >=, == ↔ !=
3. Logical Operator: and ↔ or
4. Keyword: continue ↔ break, True ↔ False
5. Numerical Literal: n → n+1, n → n-1
"""

import ast
import copy
from typing import List, Tuple

class SimpleMutator(ast.NodeTransformer):
    """Apply single mutation to AST"""

    def __init__(self, mutation_type: str, mutation_index: int):
        self.mutation_type = mutation_type
        self.mutation_index = mutation_index
        self.current_index = 0
        self.mutation_applied = False

    def visit_BinOp(self, node):
        """Mutate binary operators (arithmetic)"""
        if self.mutation_applied:
            return node

        if self.mutation_type == 'arithmetic':
            mutations = {
                ast.Add: ast.Sub,
                ast.Sub: ast.Add,
                ast.Mult: ast.FloorDiv,
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

        self.generic_visit(node)
        return node

    def visit_Compare(self, node):
        """Mutate comparison operators (relational)"""
        if self.mutation_applied:
            return node

        if self.mutation_type == 'relational':
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

        self.generic_visit(node)
        return node

    def visit_BoolOp(self, node):
        """Mutate boolean operators (logical)"""
        if self.mutation_applied:
            return node

        if self.mutation_type == 'logical':
            mutations = {
                ast.And: ast.Or,
                ast.Or: ast.And,
            }

            if type(node.op) in mutations:
                if self.current_index == self.mutation_index:
                    node = copy.deepcopy(node)
                    node.op = mutations[type(node.op)]()
                    self.mutation_applied = True
                    return node
                self.current_index += 1

        self.generic_visit(node)
        return node

    def visit_Continue(self, node):
        """Mutate continue to break"""
        if self.mutation_applied:
            return node

        if self.mutation_type == 'keyword':
            if self.current_index == self.mutation_index:
                self.mutation_applied = True
                return ast.Break()
            self.current_index += 1

        return node

    def visit_Break(self, node):
        """Mutate break to continue"""
        if self.mutation_applied:
            return node

        if self.mutation_type == 'keyword':
            if self.current_index == self.mutation_index:
                self.mutation_applied = True
                return ast.Continue()
            self.current_index += 1

        return node

    def visit_Constant(self, node):
        """Mutate constants (numbers and booleans)"""
        if self.mutation_applied:
            return node

        # Boolean constants
        if self.mutation_type == 'keyword' and isinstance(node.value, bool):
            if self.current_index == self.mutation_index:
                node = copy.deepcopy(node)
                node.value = not node.value
                self.mutation_applied = True
                return node
            self.current_index += 1

        # Numeric constants
        if self.mutation_type == 'number' and isinstance(node.value, (int, float)):
            if self.current_index == self.mutation_index:
                node = copy.deepcopy(node)
                # Try n+1 or n-1 depending on index
                if self.mutation_index % 2 == 0:
                    node.value = node.value + 1
                else:
                    node.value = node.value - 1
                self.mutation_applied = True
                return node
            self.current_index += 1

        return node


def generate_all_mutants(code: str) -> List[Tuple[str, str, int]]:
    """
    Generate all possible mutants of code

    Returns:
        List of (mutated_code, mutation_type, mutation_index) tuples
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    mutants = []
    mutation_types = ['arithmetic', 'relational', 'logical', 'keyword', 'number']

    for mutation_type in mutation_types:
        mutation_index = 0

        while mutation_index < 100:  # Safety limit
            mutator = SimpleMutator(mutation_type, mutation_index)
            mutated_tree = mutator.visit(copy.deepcopy(tree))

            if not mutator.mutation_applied:
                # No more mutations of this type
                break

            try:
                mutated_code = ast.unparse(mutated_tree)

                # Only keep if code actually changed
                if mutated_code != code:
                    mutants.append((mutated_code, mutation_type, mutation_index))

            except Exception:
                pass  # Skip invalid mutations

            mutation_index += 1

    return mutants


# Test
if __name__ == '__main__':
    test_code = """def add(a, b):
    return a + b"""

    print("Original code:")
    print(test_code)
    print("\nMutants:")

    mutants = generate_all_mutants(test_code)
    for i, (mutated, mut_type, mut_idx) in enumerate(mutants):
        print(f"\n{i+1}. {mut_type}[{mut_idx}]:")
        print(mutated)

    print(f"\nTotal: {len(mutants)} mutants")
