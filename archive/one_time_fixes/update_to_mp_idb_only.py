"""
Auto-update Laporan Kemajuan & JICEST Paper to MP-IDB only (remove IML Lifecycle)
"""

import re

def update_document(file_path):
    """Update document to remove IML Lifecycle and update metrics"""

    print(f"\n[Processing] {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_length = len(content)

    # ========== REMOVE IML LIFECYCLE SECTIONS ==========

    # Remove IML dataset description (section a)
    content = re.sub(
        r'#### a\) IML Malaria Lifecycle Dataset.*?(?=#### [bc]\))',
        '',
        content,
        flags=re.DOTALL
    )

    # Update section labels (b→a, c→b)
    content = content.replace('#### b) MP-IDB Species', '#### a) MP-IDB Species')
    content = content.replace('#### c) MP-IDB Stages', '#### b) MP-IDB Stages')

    # Remove IML rows from tables
    content = re.sub(r'\|.*?IML.*?\n', '', content)
    content = re.sub(r'\| IML.*?\n', '', content)

    # Remove IML sections in results
    content = re.sub(
        r'\*\*a\) IML Lifecycle.*?(?=\*\*[bc]\))',
        '',
        content,
        flags=re.DOTALL
    )

    # Update result section labels
    content = content.replace('**b) MP-IDB Species', '**a) MP-IDB Species')
    content = content.replace('**c) MP-IDB Stages', '**b) MP-IDB Stages')

    # ========== UPDATE METRICS ==========

    # Dataset count
    content = content.replace('tiga dataset publik', 'dua dataset publik MP-IDB')
    content = content.replace('three public datasets', 'two public MP-IDB datasets')
    content = content.replace('three diverse datasets', 'two MP-IDB datasets')

    # Image counts
    content = content.replace('731 images', '418 images')
    content = content.replace('731 citra', '418 citra')
    content = content.replace('(IML Lifecycle: 313 images, MP-IDB Species: 209 images, MP-IDB Stages: 209 images)',
                              '(MP-IDB Species: 209 images, MP-IDB Stages: 209 images)')
    content = content.replace('(IML: 313, MP-IDB Species: 209, MP-IDB Stages: 209)',
                              '(MP-IDB Species: 209, MP-IDB Stages: 209)')

    # Split counts
    content = content.replace('510 training, 146 validation, 75 testing', '292 training, 84 validation, 42 testing')
    content = content.replace('510 train, 146 val, 75 test', '292 train, 84 val, 42 test')

    # Class counts
    content = content.replace('12 kelas berbeda', '8 kelas berbeda')
    content = content.replace('12 distinct classes', '8 distinct classes')
    content = content.replace('12 classes', '8 classes')
    content = content.replace('(4 tahapan hidup lifecycle, 4 spesies, 4 tahapan hidup stages)', '(4 spesies + 4 tahapan hidup)')

    # Model counts
    content = content.replace('9 model deteksi', '6 model deteksi')
    content = content.replace('9 models', '6 models')
    content = content.replace('9 YOLO models', '6 YOLO models')
    content = content.replace('(3 YOLO × 3 datasets)', '(3 YOLO × 2 datasets)')

    content = content.replace('18 model klasifikasi', '12 model klasifikasi')
    content = content.replace('(6 architectures × 3 datasets)', '(6 architectures × 2 datasets)')
    content = content.replace('(6 CNN × 3 datasets)', '(6 CNN × 2 datasets)')

    # Total experiments
    content = content.replace('27 experiments', '18 experiments')
    content = content.replace('(9 detection + 18 classification)', '(6 detection + 12 classification)')

    # Augmented counts
    content = content.replace('2,236', '1,280')
    content = content.replace('1,789', '1,024')

    # ========== UPDATE TABLE REFERENCES ==========

    # Update table paths to MP-IDB only versions
    content = content.replace('Table1_Detection_Performance_UPDATED.csv', 'Table1_Detection_Performance_MP-IDB.csv')
    content = content.replace('Table2_Classification_Performance_UPDATED.csv', 'Table2_Classification_Performance_MP-IDB.csv')
    content = content.replace('Table3_Dataset_Statistics_UPDATED.csv', 'Table3_Dataset_Statistics_MP-IDB.csv')

    # Add Table9 references
    content = re.sub(
        r'(MP-IDB Species.*?classification.*?)',
        r'\1\n\n**INSERT FULL TABLE 9 FOR SPECIES:**\n- **Path**: `luaran/tables/Table9_MP-IDB_Species_Full.csv`\n- **Format**: 4 classes × 6 models × 4 metrics per class\n- **Shows**: Complete per-class performance breakdown\n',
        content,
        flags=re.I
    )

    # ========== REMOVE IML-SPECIFIC DISCUSSIONS ==========

    # Remove IML mentions in performance discussions
    content = re.sub(r'IML Lifecycle[^.]*?\.', '', content)
    content = re.sub(r'\(313 images[^\)]*?\)', '', content)
    content = re.sub(r'IML schizont.*?\.', '', content)
    content = re.sub(r'IML trophozoite.*?\.', '', content)

    # Clean up multiple blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)

    # ========== FINAL CLEANUP ==========

    # Fix any broken table formatting
    content = re.sub(r'\|\s*\|\s*', '| ', content)

    # Fix bullet points
    content = re.sub(r'- \s+', '- ', content)

    new_length = len(content)
    reduction = original_length - new_length

    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Updated: {file_path}")
    print(f"   Original: {original_length:,} chars")
    print(f"   New: {new_length:,} chars")
    print(f"   Reduction: {reduction:,} chars (-{reduction/original_length*100:.1f}%)")

def main():
    """Main function"""
    print("="*60)
    print("AUTO-UPDATE TO MP-IDB ONLY (Remove IML Lifecycle)")
    print("="*60)

    files = [
        'luaran/Laporan_Kemajuan_FINAL_WITH_TABLES.md',
        'luaran/JICEST_Paper_FINAL_WITH_TABLES.md'
    ]

    for file_path in files:
        try:
            update_document(file_path)
        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("UPDATE COMPLETE!")
    print("="*60)
    print("\nNEXT STEPS:")
    print("1. Review updated files")
    print("2. Check table references")
    print("3. Verify metrics (418 images, 8 classes, 2 datasets)")
    print("4. Commit changes")

if __name__ == "__main__":
    main()
