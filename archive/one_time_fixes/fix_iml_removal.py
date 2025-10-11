"""
Fix incomplete IML Lifecycle removal from documents
"""

import re

def fix_document(file_path):
    """Fix document to completely remove IML Lifecycle references"""

    print(f"\n[Processing] {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_length = len(content)
    original_iml_count = content.count('IML')

    # ========== REMOVE IML DATASET SECTIONS ==========

    # Remove IML dataset description (bold format: **a) IML...)
    content = re.sub(
        r'\*\*a\) IML Malaria Lifecycle Dataset\*\*.*?(?=\*\*[bc]\))',
        '',
        content,
        flags=re.DOTALL
    )

    # Remove three datasets description and replace with two
    content = re.sub(
        r'Three publicly available malaria microscopy datasets were used for comprehensive validation:',
        'Two publicly available malaria microscopy datasets (MP-IDB) were used for comprehensive validation:',
        content
    )

    # Update section labels (b→a, c→b) in bold format
    content = content.replace('**b) MP-IDB Species', '**a) MP-IDB Species')
    content = content.replace('**c) MP-IDB Stages', '**b) MP-IDB Stages')

    # Remove IML rows from tables
    content = re.sub(r'\|.*?IML.*?\|.*?\n', '', content)

    # Remove IML-specific performance mentions
    content = re.sub(r'IML [A-Za-z]+ \(\d+ samples?\): \d+\.?\d*% F1-score[,;.]?\s*', '', content)

    # ========== UPDATE METRICS ==========

    # Dataset count variations (English)
    content = content.replace('three datasets', 'two MP-IDB datasets')
    content = content.replace('three YOLO variants (v10, v11, v12) and three datasets',
                              'three YOLO variants (v10, v11, v12) and two MP-IDB datasets')
    content = content.replace('six CNN architectures across three datasets',
                              'six CNN architectures across two MP-IDB datasets')
    content = content.replace('across all three datasets', 'across both MP-IDB datasets')
    content = content.replace('on three datasets', 'on two MP-IDB datasets')

    # Dataset count variations (Indonesian)
    content = content.replace('tiga dataset', 'dua dataset MP-IDB')
    content = content.replace('ketiga dataset', 'kedua dataset MP-IDB')
    content = content.replace('dari ketiga dataset', 'dari kedua dataset MP-IDB')
    content = content.replace('di antara ketiga dataset', 'di antara kedua dataset MP-IDB')
    content = content.replace('pada tiga dataset', 'pada dua dataset MP-IDB')
    content = content.replace('untuk ketiga dataset', 'untuk kedua dataset MP-IDB')

    # Image counts - be very specific
    content = content.replace('731 total images', '418 total images')
    content = content.replace('total 418 images', 'total 418 images')  # Keep this
    content = re.sub(r'(\d+-)313 images', r'\g<1>209 images', content)  # Range like 209-313
    content = content.replace('313 images', '209 images')
    content = content.replace('313 thin blood smear images', '209 thin blood smear images')

    # Split counts
    content = re.sub(
        r'731 total images, \d+ train, \d+ val, \d+ test',
        '418 total images, 292 train, 84 val, 42 test',
        content
    )

    # Fix embedded table totals
    content = content.replace('| **731** | **510** | **146** | **75** |',
                            '| **418** | **292** | **84** | **42** |')

    # Fix future work mentions
    content = content.replace('1000 images vs 313', '1000 images vs 209')
    content = content.replace('(1000 images vs 313)', '(1000 images vs 209)')

    # Class counts
    content = content.replace('12 classes', '8 classes')
    content = content.replace('12 distinct classes', '8 distinct classes')

    # Model counts
    content = re.sub(r'27 experiments? \(9 detection \+ 18 classification\)',
                     '18 experiments (6 detection + 12 classification)', content)

    # ========== CLEAN UP IML REFERENCES ==========

    # Keep only the acknowledgment IML reference, remove others
    # Remove IML from challenge descriptions
    content = re.sub(
        r'limited annotated datasets \(209-313 images per task\)',
        'limited annotated datasets (209 images per task)',
        content
    )

    # Remove IML class imbalance mentions
    content = re.sub(r'4-272 samples per class', '4-69 samples per class', content)

    # Remove schizont mentions that reference IML data
    content = re.sub(r'schizont: 4-7 samples', 'schizont: 7 samples', content)

    # Remove IML class mentions from challenge descriptions
    content = re.sub(r'IML schizont=4, ', '', content)
    content = re.sub(r', IML trophozoite=16', '', content)
    content = re.sub(r'IML trophozoite=16, ', '', content)
    content = re.sub(r'\(IML trophozoite=16, ', '(', content)
    content = re.sub(r'IML [Ss]chizont \(\d+ samples?\)', 'MP-IDB schizont (7 samples)', content)

    # Remove IML-specific bullet points and sections
    content = re.sub(r'- \*\*IML [A-Za-z]+\*\* \(\d+ samples?\):.*?(?=\n- |\n\n)', '', content, flags=re.DOTALL)

    # Remove IML from feature lists
    content = re.sub(r'\(MP-IDB, IML\)', '(MP-IDB)', content)
    content = re.sub(r'\(IML, MP-IDB\)', '(MP-IDB)', content)

    # ========== UPDATE TABLE REFERENCES ==========

    # Update table paths
    content = content.replace('Table1_Detection_Performance_UPDATED.csv',
                            'Table1_Detection_Performance_MP-IDB.csv')
    content = content.replace('Table2_Classification_Performance_UPDATED.csv',
                            'Table2_Classification_Performance_MP-IDB.csv')
    content = content.replace('Table3_Dataset_Statistics_UPDATED.csv',
                            'Table3_Dataset_Statistics_MP-IDB.csv')

    # ========== CLEAN UP ==========

    # Remove multiple blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)

    # Fix broken formatting
    content = re.sub(r'\*\*\s+\*\*', '', content)
    content = re.sub(r'- \s+', '- ', content)

    new_length = len(content)
    new_iml_count = content.count('IML')
    reduction = original_length - new_length
    iml_reduction = original_iml_count - new_iml_count

    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Fixed: {file_path}")
    print(f"   Original: {original_length:,} chars, {original_iml_count} IML references")
    print(f"   New: {new_length:,} chars, {new_iml_count} IML references")
    print(f"   Reduction: {reduction:,} chars (-{reduction/original_length*100:.1f}%), {iml_reduction} IML refs removed")

    # Show remaining IML references
    if new_iml_count > 0:
        print(f"\n   [INFO] Remaining IML references (should be acknowledgments only):")
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'IML' in line:
                # Use .encode to handle Unicode characters on Windows
                safe_line = line.strip()[:100].encode('ascii', errors='ignore').decode('ascii')
                print(f"      Line {i}: {safe_line}")

def main():
    """Main function"""
    print("="*60)
    print("FIX INCOMPLETE IML REMOVAL")
    print("="*60)

    files = [
        'luaran/JICEST_Paper_FINAL_WITH_TABLES.md',
        'luaran/Laporan_Kemajuan_FINAL_WITH_TABLES.md'
    ]

    for file_path in files:
        try:
            fix_document(file_path)
        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("FIX COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
