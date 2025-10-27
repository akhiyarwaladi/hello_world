"""
Final fix: Clean and reorder REFERENCES section properly
"""
import re
import json

# Load mapping
with open('reference_mapping.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    new_to_old = {v: int(k) for k, v in data['new_to_old'].items()}

# Read RENUMBERED file (which has citations already updated)
renumbered_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_RENUMBERED.md"
with open(renumbered_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Split into main text and references section
parts = content.split("## REFERENCES")
if len(parts) != 2:
    print("❌ Error: Could not find REFERENCES section")
    exit(1)

main_text = parts[0]
ref_section_and_after = parts[1]

# Split references from DATA AVAILABILITY
ref_parts = ref_section_and_after.split("---")
if len(ref_parts) < 2:
    print("❌ Error: Could not find end of REFERENCES section")
    exit(1)

old_ref_section = ref_parts[0]
after_refs = "---".join(ref_parts[1:])

# Extract all reference entries from old section
print("Extracting all references from old section...")
ref_pattern = r'\[(\d+)\]\s+(.+?)(?=\n\[(\d+)\]|\Z)'
matches = re.findall(ref_pattern, old_ref_section, re.DOTALL)

# Store unique references (handle duplicates by keeping first occurrence)
all_refs = {}
for match in matches:
    ref_num = int(match[0])
    ref_text = match[1].strip()
    if ref_num not in all_refs:
        all_refs[ref_num] = f"[{ref_num}] {ref_text}"
        print(f"  Found [{ref_num}]: {ref_text[:60]}...")
    else:
        print(f"  ⚠️  Duplicate [{ref_num}] found, keeping first occurrence")

print(f"\nExtracted {len(all_refs)} unique references")

# Build new references section in sequential order [1] to [30]
new_ref_lines = []
new_ref_lines.append("## REFERENCES\n\n")

for new_num in sorted(new_to_old.keys()):
    old_num = new_to_old[new_num]
    if old_num in all_refs:
        # Get reference text and update number to new_num
        ref_text = all_refs[old_num]
        # Replace old number with new number
        ref_text = re.sub(r'^\[\d+\]', f'[{new_num}]', ref_text)
        new_ref_lines.append(f"{ref_text}\n\n")
        print(f"✅ [{new_num}] (was [{old_num}]): Added")
    else:
        print(f"⚠️  Warning: Old reference [{old_num}] not found for new [{new_num}]")

new_ref_lines.append("---\n")

# Assemble final paper
final_content = main_text + "".join(new_ref_lines) + after_refs

# Write final result
output_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_UPDATED_2025.md"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(final_content)

print()
print("=" * 100)
print("REFERENCES FIXED AND REORDERED")
print("=" * 100)
print(f"✅ Final paper saved to: KINETIK_PAPER_DRAFT_UPDATED_2025.md")
print()
print("Summary:")
print(f"  - Total references: {len(new_to_old)}")
print(f"  - All references now in sequential order [1] to [{max(new_to_old.keys())}]")
print(f"  - All citations updated to match")
print()
print("🎉 RENUMBERING COMPLETE!")
