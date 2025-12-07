"""
Renumber all citations in paper to sequential order
"""
import re
import json

# Load mapping
with open('reference_mapping.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    old_to_new = {int(k): v for k, v in data['old_to_new'].items()}

paper_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_UPDATED_2025.md"

with open(paper_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Replace all citations with temporary placeholders
# This avoids conflicts where [10] → [9] might conflict with existing [9]
print("Step 1: Replacing citations with placeholders...")
temp_content = content
for old_num in sorted(old_to_new.keys(), reverse=True):  # Reverse to avoid conflicts
    # Replace [old_num] with [TEMPREF_old_num]
    pattern = f'\\[{old_num}\\]'
    replacement = f'[TEMPREF_{old_num}]'
    temp_content = re.sub(pattern, replacement, temp_content)
    count = len(re.findall(pattern, content))
    if count > 0:
        print(f"  [{old_num}] → {replacement} ({count} occurrences)")

# Step 2: Replace placeholders with new numbers
print()
print("Step 2: Replacing placeholders with new numbers...")
final_content = temp_content
for old_num, new_num in old_to_new.items():
    pattern = f'\\[TEMPREF_{old_num}\\]'
    replacement = f'[{new_num}]'
    final_content = re.sub(pattern, replacement, final_content)
    count = len(re.findall(pattern, temp_content))
    if count > 0:
        change_marker = "" if old_num == new_num else " ⚠️"
        print(f"  [TEMPREF_{old_num}] → [{new_num}]{change_marker} ({count} occurrences)")

# Write back
output_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_RENUMBERED.md"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(final_content)

print()
print("=" * 100)
print("CITATIONS RENUMBERED")
print("=" * 100)
print(f"✅ Updated paper saved to: KINETIK_PAPER_DRAFT_RENUMBERED.md")
print()
print("Summary:")
print(f"  - Total references renumbered: {len(old_to_new)}")
print(f"  - Changed: {sum(1 for old, new in old_to_new.items() if old != new)}")
print(f"  - Unchanged: {sum(1 for old, new in old_to_new.items() if old == new)}")
print()
print("⚠️ Next step: Reorder REFERENCES section to match new numbering")
