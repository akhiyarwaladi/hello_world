"""
Create mapping for renumbering references in sequential order
"""
import re
import json

paper_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_UPDATED_2025.md"

with open(paper_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find first appearance of each citation
first_appearance = {}  # {cite_num: line_num}
for line_num, line in enumerate(lines, 1):
    # Skip the REFERENCES section itself (don't count reference definitions)
    if line_num >= 257:  # References section starts around line 257
        break

    # Find all [number] patterns
    matches = re.findall(r'\[(\d+)\]', line)
    for match in matches:
        cite_num = int(match)
        if cite_num not in first_appearance:
            first_appearance[cite_num] = line_num

# Sort by line number to get sequential order
sorted_citations = sorted(first_appearance.items(), key=lambda x: x[1])

# Create mapping: old_number -> new_number
mapping = {}
for new_num, (old_num, line_num) in enumerate(sorted_citations, 1):
    mapping[old_num] = new_num

# Create reverse mapping for reference section: new_number -> old_number
reverse_mapping = {v: k for k, v in mapping.items()}

print("=" * 100)
print("REFERENCE RENUMBERING MAPPING")
print("=" * 100)
print()
print(f"Total references: {len(mapping)}")
print()
print("OLD → NEW Mapping:")
print("-" * 100)
for old_num in sorted(mapping.keys()):
    new_num = mapping[old_num]
    change = "" if old_num == new_num else f"  ⚠️ CHANGED from [{old_num}]"
    line = first_appearance[old_num]
    print(f"[{old_num:2d}] → [{new_num:2d}]  (first appeared at line {line:3d}){change}")

# Save mapping to JSON for use in replacement script
output = {
    "old_to_new": mapping,
    "new_to_old": reverse_mapping,
    "first_appearance": first_appearance
}

with open('reference_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
unchanged = sum(1 for old, new in mapping.items() if old == new)
changed = len(mapping) - unchanged
print(f"Total references: {len(mapping)}")
print(f"Unchanged: {unchanged}")
print(f"Need renumbering: {changed}")
print()
print("✅ Mapping saved to: reference_mapping.json")
