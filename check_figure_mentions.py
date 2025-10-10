"""
Check if all figures have proper narrative mentions in Laporan_Kemajuan.md
"""

import re

def check_figure_mentions(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    # Find all figure placeholders
    figures = []
    for i, line in enumerate(lines):
        if 'INSERT GAMBAR' in line or 'INSERT FIGURE' in line or 'INSERT TABEL' in line:
            # Extract figure name - handle "GAMBAR 1 DI SINI" -> extract just "1"
            match = re.search(r'\[.*?(GAMBAR|FIGURE|TABEL)\s+([A-Z0-9]+)', line)
            if match:
                fig_name = match.group(2).strip()
                figures.append((i+1, fig_name, line))

    print(f"Found {len(figures)} figures/tables\n")
    print("=" * 80)

    # Check each figure for narrative mention
    missing_mentions = []
    for line_no, fig_name, placeholder in figures:
        # Look for narrative mention in the next 5 lines after placeholder
        start = line_no
        end = min(line_no + 5, len(lines))

        has_mention = False
        mention_line = None
        for i in range(start, end):
            # Look for narrative verbs: menampilkan, mengilustrasikan, memvisualisasikan, menyajikan
            if re.search(rf'(Gambar|Figure|Tabel)\s+{re.escape(fig_name)}.*(menampilkan|mengilustrasikan|memvisualisasikan|menyajikan)', lines[i], re.IGNORECASE):
                has_mention = True
                mention_line = i + 1
                break
            # Also check for English equivalents
            if re.search(rf'(Gambar|Figure|Tabel)\s+{re.escape(fig_name)}.*(visualizes|presents|shows|illustrates|displays)', lines[i], re.IGNORECASE):
                has_mention = True
                mention_line = i + 1
                break

        status = "[OK]" if has_mention else "[MISSING]"

        print(f"\n{status} - {fig_name}")
        print(f"   Placeholder line: {line_no}")
        if has_mention:
            print(f"   Mention found at line: {mention_line}")
            print(f"   Text: {lines[mention_line-1][:100]}...")
        else:
            print(f"   WARNING: NO NARRATIVE MENTION FOUND in next 5 lines!")
            missing_mentions.append(fig_name)

    print("\n" + "=" * 80)
    if missing_mentions:
        print(f"\nWARNING: {len(missing_mentions)} figures MISSING narrative mentions:")
        for fig in missing_mentions:
            print(f"   - {fig}")
        return False
    else:
        print("\nSUCCESS: All figures have proper narrative mentions!")
        return True

if __name__ == '__main__':
    filename = 'luaran/Laporan_Kemajuan.md'
    all_ok = check_figure_mentions(filename)
    exit(0 if all_ok else 1)
