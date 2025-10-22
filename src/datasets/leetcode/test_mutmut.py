#!/usr/bin/env python3
"""Test mutmut mutation generation"""

import sys
sys.path.insert(0, '/Users/akshitgaur/Desktop/CMU/IDL/mutmut_src')

import __init__ as mutmut_module

# Simple test code
test_code = """
def add(a, b):
    return a + b
"""

print("Testing mutation generation...")
print("Original code:")
print(test_code)
print("\nGenerating mutants...")

mutation_index = 0
mutant_count = 0

while mutation_index < 100:
    try:
        # Use ALL to get all mutations, or construct proper RelativeMutationID
        context = mutmut_module.Context(
            source=test_code,
            filename="test.py",
            mutation_id=mutmut_module.ALL,  # Generate all mutations
            dict_synonyms=['dict'],
        )

        mutated_code, num_mutations = mutmut_module.mutate(context)

        if num_mutations == 0:
            print(f"No mutations at index {mutation_index}")
            break

        if mutated_code != test_code:
            mutant_count += 1
            print(f"\nMutant #{mutant_count} (index {mutation_index}):")
            print(mutated_code)

        mutation_index += 1

    except StopIteration:
        break
    except Exception as e:
        print(f"Error at index {mutation_index}: {e}")
        mutation_index += 1

print(f"\nTotal mutants generated: {mutant_count}")
