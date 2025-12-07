"""
Reorder REFERENCES section to match new citation numbers
"""
import re
import json

# Load mapping
with open('reference_mapping.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    old_to_new = {int(k): v for k, v in data['old_to_new'].items()}
    new_to_old = {v: int(k) for k, v in data['new_to_old'].items()}

# Read original paper to extract references
original_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_UPDATED_2025.md"
with open(original_path, 'r', encoding='utf-8') as f:
    original_lines = f.readlines()

# Extract all reference entries from original
references = {}  # {old_num: reference_text}
current_ref = None
current_text = []

in_references = False
for line in original_lines:
    # Check if we're in REFERENCES section
    if line.strip() == "## REFERENCES":
        in_references = True
        continue

    # Check if we've left REFERENCES section
    if in_references and line.startswith("---"):
        # Save last reference if any
        if current_ref is not None and current_text:
            references[current_ref] = ''.join(current_text)
        break

    if in_references:
        # Check for new reference entry [X]
        match = re.match(r'^\[(\d+)\]\s+(.+)', line)
        if match:
            # Save previous reference if any
            if current_ref is not None and current_text:
                references[current_ref] = ''.join(current_text)

            # Start new reference
            current_ref = int(match.group(1))
            current_text = [line]
        elif current_ref is not None:
            # Continue current reference
            current_text.append(line)

print("=" * 100)
print("EXTRACTING REFERENCES FROM ORIGINAL")
print("=" * 100)
print(f"Extracted {len(references)} references")
print()

# Create new ordered references
new_references = {}
for new_num in range(1, len(old_to_new) + 1):
    if new_num in new_to_old:
        old_num = new_to_old[new_num]
        if old_num in references:
            # Replace old number with new number in text
            ref_text = references[old_num]
            ref_text = re.sub(r'^\[(\d+)\]', f'[{new_num}]', ref_text)
            new_references[new_num] = ref_text
        else:
            print(f"⚠️ Warning: Old reference [{old_num}] not found in original")

print("=" * 100)
print("REORDERED REFERENCES")
print("=" * 100)
for new_num in sorted(new_references.keys()):
    old_num = new_to_old[new_num]
    marker = "" if old_num == new_num else f"  (was [{old_num}])"
    # Show first 80 chars of reference
    ref_preview = new_references[new_num].strip()[:80]
    print(f"[{new_num:2d}]{marker}: {ref_preview}...")

# Now read the renumbered paper and replace REFERENCES section
renumbered_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_RENUMBERED.md"
with open(renumbered_path, 'r', encoding='utf-8') as f:
    paper_lines = f.readlines()

# Find REFERENCES section
ref_start = None
ref_end = None
for i, line in enumerate(paper_lines):
    if line.strip() == "## REFERENCES":
        ref_start = i
    elif ref_start is not None and line.startswith("---"):
        ref_end = i
        break

if ref_start is None or ref_end is None:
    print("❌ Error: Could not find REFERENCES section")
    exit(1)

# Build new REFERENCES section
new_ref_section = ["## REFERENCES\n", "\n"]
for new_num in sorted(new_references.keys()):
    new_ref_section.append(new_references[new_num])

new_ref_section.append("\n---\n")

# Replace REFERENCES section
final_lines = paper_lines[:ref_start] + new_ref_section + paper_lines[ref_end+1:]

# Write final paper
final_path = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\KINETIK_PAPER_DRAFT_UPDATED_2025.md"
with open(final_path, 'w', encoding='utf-8') as f:
    f.writelines(final_lines)

print()
print("=" * 100)
print("REFERENCES REORDERED")
print("=" * 100)
print(f"✅ Final paper saved to: KINETIK_PAPER_DRAFT_UPDATED_2025.md")
print(f"✅ Temporary file: KINETIK_PAPER_DRAFT_RENUMBERED.md (can be deleted)")
print()
print("Summary:")
print(f"  - Total references: {len(new_references)}")
print(f"  - References reordered and renumbered")
print(f"  - Citations updated to match")
print()
print("⚠️ Next step: Verify all citations are sequential")
