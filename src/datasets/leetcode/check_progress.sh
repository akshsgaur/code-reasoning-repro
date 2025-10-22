#!/bin/bash
# Check mutation progress

echo "=========================================="
echo "Mutation Dataset Generation Progress"
echo "=========================================="

# Get current problem number
current=$(grep -E "\[[0-9]+/347\]" mutation_full.log | tail -1 | grep -o "\[[0-9]*/347\]")
echo "Current: $current"

# Count successes and failures
successes=$(grep "✓ Created mutant" mutation_full.log | wc -l | tr -d ' ')
failures=$(grep "✗ No valid" mutation_full.log | wc -l | tr -d ' ')

echo "Successful mutations: $successes"
echo "Failed mutations: $failures"
echo "Total processed: $((successes + failures))"

# Check if process is still running
if ps aux | grep -q "[p]ython3 -u mutate_dataset.py"; then
    echo "Status: RUNNING ✓"
else
    echo "Status: NOT RUNNING (check mutation_full.log for completion)"
fi

echo ""
echo "To monitor live: tail -f mutation_full.log"
echo "To check last 30 lines: tail -30 mutation_full.log"
