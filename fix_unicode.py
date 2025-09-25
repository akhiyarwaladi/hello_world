#!/usr/bin/env python3
import re

# Read the file and remove all non-ASCII characters
with open('scripts/analysis/unified_journal_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace any problematic Unicode characters
content_fixed = content.replace('\U0001f9ec', '[DNA]')  # DNA emoji
content_fixed = content_fixed.replace('🧬', '[DNA]')  # DNA emoji direct
content_fixed = content_fixed.replace('→', '->')  # Arrow symbol
content_fixed = content_fixed.replace('—', '--')  # Em dash

# Also check for any f-strings or prints that might contain Unicode
print('Content checked, looking for changes...')
if content != content_fixed:
    print('Changes made - writing back to file')
    with open('scripts/analysis/unified_journal_analysis.py', 'w', encoding='utf-8') as f:
        f.write(content_fixed)
    print('File updated successfully')
else:
    print('No Unicode issues found')
    # Let's also check for any non-ASCII characters
    non_ascii_lines = []
    for i, line in enumerate(content.split('\n'), 1):
        try:
            line.encode('cp1252')
        except UnicodeEncodeError as e:
            non_ascii_lines.append((i, line, str(e)))

    if non_ascii_lines:
        print(f'Found {len(non_ascii_lines)} lines with cp1252 encoding issues:')
        for line_num, line, error in non_ascii_lines[:5]:  # Show first 5
            print(f'Line {line_num}: {repr(line[:100])}')
            print(f'Error: {error}')
            print('---')
    else:
        print('No cp1252 encoding issues found')