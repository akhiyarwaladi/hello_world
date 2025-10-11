"""
Comprehensive code-switching fixer for Laporan_Kemajuan.md
Fix ALL remaining English in Indonesian narrative text
"""

import re

# Expanded dictionary with ALL patterns found
replacements = {
    # ===== VERBS =====
    r'\bpresenting\b': 'menyajikan',
    r'\bquantifies\b': 'mengkuantifikasi',
    r'\bhighlights\b': 'menyoroti',
    r'\bexceeding\b': 'melebihi',
    r'\bdemonstrates\b': 'mendemonstrasikan',
    r'\brequires\b': 'memerlukan',
    r'\brepresenting\b': 'merepresentasikan',
    r'\bassuming\b': 'dengan asumsi',
    r'\bcompletes\b': 'menyelesaikan',
    r'\benabling\b': 'memungkinkan',
    r'\boffering\b': 'menawarkan',
    r'\butilizing\b': 'menggunakan',
    r'\btotaling\b': 'berjumlah',
    r'\bremains\b': 'tetap',
    r'\bevidenced\b': 'dibuktikan',
    r'\battributable\b': 'disebabkan',
    r'\boriginated\b': 'berasal',
    r'\baffecting\b': 'mempengaruhi',
    r'\btesting\b': 'menguji',
    r'\bencountered\b': 'ditemui',

    # ===== MODAL/AUXILIARY =====
    r'\bharus mencakup\b': 'harus menyertakan',
    r'\bwarrant\b': 'memerlukan',
    r'\bprovide\b': 'memberikan',

    # ===== ADJECTIVES =====
    r'\bsuperior\b': 'unggul',
    r'\bcritical\b': 'kritis',
    r'\bsignificant\b': 'signifikan',
    r'\binsufficient\b': 'tidak mencukupi',
    r'\boptimally\b': 'secara optimal',
    r'\btheoretical\b': 'teoretis',
    r'\bstandardized\b': 'terstandarisasi',
    r'\bconsistent\b': 'konsisten',
    r'\buniform\b': 'seragam',
    r'\bessential\b': 'esensial',
    r'\bdiverse\b': 'beragam',
    r'\bheterogeneous\b': 'heterogen',
    r'\bvarying\b': 'bervariasi',
    r'\bactual\b': 'aktual',
    r'\brealistic\b': 'realistis',

    # ===== NOUNS (non-technical) =====
    r'\bdetection results\b': 'hasil deteksi',
    r'\bcompetitive performance\b': 'performa kompetitif',
    r'\bcomparison\b': 'perbandingan',
    r'\bthorough analysis\b': 'analisis menyeluruh',
    r'\bspeedup\b': 'percepatan',
    r'\bbatch processing\b': 'pemrosesan batch',
    r'\bmanual examination\b': 'pemeriksaan manual',
    r'\boperator expertise\b': 'keahlian operator',
    r'\bdata collection effort\b': 'upaya pengumpulan data',
    r'\bminority class performance\b': 'performa kelas minoritas',
    r'\bcontinuous learning pipelines\b': 'pipeline pembelajaran berkelanjutan',
    r'\bclinical usage\b': 'penggunaan klinis',
    r'\bongoing model improvement\b': 'peningkatan model berkelanjutan',
    r'\bcontrolled laboratory settings\b': 'pengaturan laboratorium terkontrol',
    r'\bstaining protocols\b': 'protokol pewarnaan',
    r'\bimaging conditions\b': 'kondisi pencitraan',
    r'\bslide preparation techniques\b': 'teknik preparasi slide',
    r'\bfield-collected samples\b': 'sampel dari lapangan',
    r'\bstaining quality\b': 'kualitas pewarnaan',
    r'\bmicroscope types\b': 'tipe mikroskop',
    r'\bimage acquisition settings\b': 'pengaturan akuisisi gambar',
    r'\btechnician expertise levels\b': 'tingkat keahlian teknisi',
    r'\bdomain shift robustness\b': 'ketahanan domain shift',
    r'\bdeployment conditions\b': 'kondisi deployment',
    r'\bclinical practice\b': 'praktik klinis',

    # ===== PHRASES =====
    r'\bacross kedua datasets untuk four\b': 'pada kedua dataset untuk empat',
    r'\bacross kedua\b': 'pada kedua',
    r'\bfor all three\b': 'untuk ketiga',
    r'\bcolumns untuk\b': 'kolom untuk',
    r'\bunder 25\b': 'di bawah 25',
    r'\bper image\b': 'per gambar',
    r'\bper slide\b': 'per slide',
    r'\bfor single-image\b': 'untuk pemrosesan gambar tunggal',
    r'\bfor complete\b': 'untuk',
    r'\bper slide assuming\b': 'per slide dengan asumsi',
    r'\bfields per slide\b': 'field per slide',
    r'\bwithin 180-250\b': 'dalam 180-250',
    r'\bstill dramatically\b': 'masih secara dramatis',
    r'\bindependent dari\b': 'independen dari',
    r'\bseveral limitations yang\b': 'beberapa keterbatasan yang',
    r'\bdirection untuk\b': 'arah untuk',
    r'\bfuture investigations\b': 'investigasi masa depan',
    r'\btwo MP-IDB datasets\b': 'dua dataset MP-IDB',
    r'\btask critical untuk\b': 'tugas yang kritis untuk',
    r'\bStrategies untuk expansion\b': 'Strategi untuk ekspansi',
    r'\bExternal validation pada\b': 'Validasi eksternal pada',
    r'\bPlanned Phase 2\b': 'Fase 2 yang direncanakan',
    r'\bsystem robustness terhadap\b': 'ketahanan sistem terhadap',

    # ===== SPECIFIC LONG PHRASES =====
    r'\bdemonstrates practical feasibility untuk\b': 'mendemonstrasikan kelayakan praktis untuk',
    r'\bmenyertakan crowdsourced annotation platforms leveraging distributed expertise dari\b': 'menyertakan platform anotasi crowdsourced yang memanfaatkan keahlian terdistribusi dari',
    r'\bsystematic collaborations dengan\b': 'kolaborasi sistematis dengan',
    r'\bcontributing anonymized patient samples\b': 'berkontribusi sampel pasien anonim',
    r'\bimplementation dari\b': 'implementasi',
    r'\bdeployed systems collect\b': 'sistem yang di-deploy mengumpulkan',
}

def fix_code_switching(text):
    """Apply all replacements"""
    fixed_text = text
    changes_count = 0

    for pattern, replacement in replacements.items():
        matches = re.findall(pattern, fixed_text, re.IGNORECASE)
        if matches:
            fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
            changes_count += len(matches)

    return fixed_text, changes_count

def main():
    input_file = 'luaran/Laporan_Kemajuan.md'
    output_file = 'luaran/Laporan_Kemajuan.md'  # Overwrite directly

    print("Reading file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Applying comprehensive code-switching fixes...")
    fixed_content, changes = fix_code_switching(content)

    if changes > 0:
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        print(f"\nSUCCESS: Fixed {changes} code-switching instances")
        print(f"   File: {output_file}")
        print(f"   - Replaced English verbs, nouns, adjectives, phrases")
        print(f"   - Narrative now uses consistent Indonesian")
    else:
        print("\nNo changes needed!")

if __name__ == '__main__':
    main()
