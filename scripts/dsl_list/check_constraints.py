import json, re

file_path = "../data/dsl_list.jsonl"  # your dataset file

def violates_constraints(entry):
    code = entry.get("python", "")
    violations = []

    # (1) Comparison first arg must not be literal
    if re.search(r"\(\s*\d+\s*[=!<>]=\s*", code):
        violations.append("comparison with literal first arg")

    # (2) last argument to extend, length, map must not be empty
    if re.search(r"extend\(\s*\[\s*\]\s*\)", code) or re.search(r"len\(\[\s*\]\)", code) or re.search(r"map\(\s*.*,\s*\[\s*\]\)", code):
        violations.append("extend/length/map with empty")

    # (3) -1 only used in index
    if "-1" in code and not re.search(r"\[-1\]", code):
        violations.append("-1 not only in index")

    # (4) index/init/tail on empty list
    if re.search(r"\[\]\.pop", code):
        violations.append("index/init/tail on empty")

    # (5) identical sides of comparison/logical ops
    if re.search(r"([av]\d+)\s*==\s*\1", code):
        violations.append("same expr both sides comparison")

    # (6) list extending itself
    if re.search(r"([av]\d+)\.extend\(\s*\1\s*\)", code):
        violations.append("list extends itself")

    # (7) missing parameters
    arity = entry.get("arity", 1)
    for i in range(arity):
        if f"a{i+1}" not in code:
            violations.append(f"missing param a{i+1}")

    # (8) same outputs for all inputs
    outputs = entry.get("outputs", [])
    if len(set(json.dumps(o) for o in outputs)) == 1:
        violations.append("same outputs for all inputs")

    return violations


# --- main ---
with open(file_path, "r") as f:
    data = [json.loads(line) for line in f]

results = []
for i, entry in enumerate(data):
    v = violates_constraints(entry)
    if v:
        results.append({"index": i, "violations": v})

print("Total programs:", len(data))
print("Violations found:", len(results))
if data:
    print("Invalid %:", round(len(results) / len(data) * 100, 2))

print("\nSample Violations (up to 5):")
for r in results[:5]:
    print(f"\n#{r['index']} -> {r['violations']}")
