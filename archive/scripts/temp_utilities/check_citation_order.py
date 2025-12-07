"""
Check citation order in paper to ensure sequential appearance
"""
import re

paper_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_UPDATED_2025.md"

with open(paper_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract all citations with their line numbers
citations = []
for line_num, line in enumerate(lines, 1):
    # Find all [number] patterns
    matches = re.findall(r'\[(\d+)\]', line)
    for match in matches:
        citations.append((line_num, int(match), line.strip()[:80]))

print("=" * 100)
print("CITATION ORDER CHECK")
print("=" * 100)
print()

# Check if citations appear in order
max_seen = 0
out_of_order = []

for line_num, cite_num, line_text in citations:
    if cite_num < max_seen:
        # Citation appears out of order
        out_of_order.append((line_num, cite_num, max_seen, line_text))
        print(f"❌ Line {line_num}: [{cite_num}] appears AFTER [{max_seen}] (out of order)")
        print(f"   Text: {line_text}")
        print()
    else:
        if cite_num > max_seen:
            max_seen = cite_num
            print(f"✅ Line {line_num}: [{cite_num}] (first appearance)")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Total citations found: {len(citations)}")
print(f"Unique citations: {len(set([c[1] for c in citations]))}")
print(f"Highest reference number: {max([c[1] for c in citations])}")
print(f"Out of order citations: {len(out_of_order)}")
print()

if out_of_order:
    print("⚠️ CITATIONS ARE NOT IN SEQUENTIAL ORDER!")
    print()
    print("Out of order citations:")
    for line_num, cite_num, prev_max, line_text in out_of_order:
        print(f"  Line {line_num}: [{cite_num}] came after [{prev_max}]")
else:
    print("✅ ALL CITATIONS ARE IN SEQUENTIAL ORDER!")

# Show first appearance order
print()
print("=" * 100)
print("FIRST APPEARANCE ORDER (should be [1], [2], [3], ...)")
print("=" * 100)
seen = set()
first_appearances = []
for line_num, cite_num, line_text in citations:
    if cite_num not in seen:
        seen.add(cite_num)
        first_appearances.append((cite_num, line_num))

first_appearances.sort(key=lambda x: x[1])  # Sort by line number
for i, (cite_num, line_num) in enumerate(first_appearances[:50], 1):
    status = "✅" if cite_num == i else f"❌ Expected [{i}]"
    print(f"{status} Position {i}: [{cite_num}] first appears at line {line_num}")
