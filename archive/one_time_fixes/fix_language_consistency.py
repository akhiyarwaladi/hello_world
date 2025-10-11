"""
Script to fix excessive code-switching in Laporan_Kemajuan.md
Replace Bahasa Inggris yang berlebihan dengan Bahasa Indonesia yang natural
"""

# Dictionary of common excessive English phrases and their Indonesian replacements
replacements = {
    # Common verbs - expanded
    r'\bfocus pada\b': 'berfokus pada',
    r'\bfinalizing\b': 'menyelesaikan',
    r'\bconducting\b': 'melakukan',
    r'\baddressing\b': 'mengatasi',
    r'\bimproving\b': 'meningkatkan',
    r'\benhancing\b': 'meningkatkan',
    r'\bvalidating\b': 'memvalidasi',
    r'\bdemonstrating\b': 'mendemonstrasikan',
    r'\bconfirming\b': 'mengonfirmasi',
    r'\bexploring\b': 'mengeksplorasi',
    r'\binvestigating\b': 'menginvestigasi',
    r'\bgenerating\b': 'menghasilkan',
    r'\bperforming\b': 'melakukan',
    r'\bproviding\b': 'memberikan',
    r'\bquantifying\b': 'mengukur',
    r'\binclude\b': 'mencakup',

    # Common nouns (only non-technical ones)
    r'\bPhase immediate\b': 'Fase berikutnya',
    r'\bFase berikutnya berikutnya\b': 'Fase berikutnya',  # Fix duplicate
    r'\bjournal manuscript\b': 'naskah jurnal',
    r'\badditional experiments\b': 'eksperimen tambahan',
    r'\badditional visualizations\b': 'visualisasi tambahan',
    r'\bfuture work\b': 'pekerjaan mendatang',
    r'\bkey findings\b': 'temuan kunci',
    r'\bmain contributions\b': 'kontribusi utama',
    r'\bcomprehensive performance assessment\b': 'penilaian performa komprehensif',
    r'\barchitectural components\b': 'komponen arsitektur',
    r'\btraining strategies\b': 'strategi pelatihan',
    r'\bstatistical significance testing\b': 'uji signifikansi statistik',
    r'\bperformance differences\b': 'perbedaan performa',
    r'\bablation studies\b': 'studi ablasi',
    r'\bPlanned activities\b': 'Aktivitas yang direncanakan',

    # Phrases - expanded
    r'\buntuk addressing\b': 'untuk mengatasi',
    r'\buntuk improving\b': 'untuk meningkatkan',
    r'\buntuk enhancing\b': 'untuk meningkatkan',
    r'\buntuk finalizing\b': 'untuk menyelesaikan',
    r'\buntuk providing\b': 'untuk memberikan',
    r'\buntuk quantifying\b': 'untuk mengukur',
    r'\buntuk rigorously\b': 'untuk secara ketat',
    r'\bakan focus\b': 'akan berfokus',
    r'\bakan conduct\b': 'akan melakukan',
    r'\bakan explore\b': 'akan mengeksplorasi',
    r'\bakan investigate\b': 'akan menginvestigasi',

    # Mixed phrases
    r'\bmemerlukan additional\b': 'memerlukan tambahan',
    r'\bdengan potential\b': 'dengan potensi',
    r'\byang critical\b': 'yang kritis',
    r'\byang essential\b': 'yang esensial',
    r'\bsangat important\b': 'sangat penting',
    r'\blebih efficient\b': 'lebih efisien',
    r'\blebih effective\b': 'lebih efektif',
    r'\breviewer potential concerns\b': 'potensi kekhawatiran reviewer',
    r'\bcontribution dari\b': 'kontribusi dari',
    r'\bbetween models\b': 'antara model',
    r'\bbetween\b': 'antara',
}

import re

def fix_language_consistency(text):
    """Apply all replacements to text"""
    fixed_text = text
    changes_made = []

    for pattern, replacement in replacements.items():
        matches = re.findall(pattern, fixed_text, re.IGNORECASE)
        if matches:
            fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
            changes_made.append(f"  - '{matches[0]}' -> '{replacement}' ({len(matches)} occurrences)")

    return fixed_text, changes_made

def main():
    input_file = 'luaran/Laporan_Kemajuan.md'
    output_file = 'luaran/Laporan_Kemajuan_FIXED.md'

    print("Reading file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print("\nApplying language consistency fixes...")
    fixed_content, changes = fix_language_consistency(content)

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"\nSUCCESS: Fixed file saved to: {output_file}")
    if changes:
        print(f"\nChanges made ({len(changes)} types):")
        for change in changes[:20]:  # Show first 20 types
            print(change)
        if len(changes) > 20:
            print(f"  ... and {len(changes) - 20} more types")
    else:
        print("\nNo changes needed - file already uses consistent Indonesian!")

if __name__ == '__main__':
    main()
