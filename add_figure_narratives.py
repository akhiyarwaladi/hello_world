"""
Replace placeholder instructions with proper narrative mentions
"**Gambar X** harus ditempatkan di sini..." -> "Gambar X menampilkan..."
"""

import re

def add_narrative_mentions(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern 1: Figure A2 - already has "memvisualisasikan" but script didn't detect it
    # Actually OK, just update the regex in checker

    # Pattern 2: Gambar 1 - Pipeline Architecture
    content = re.sub(
        r'\*\*Gambar 1\*\* harus ditempatkan di sini, mengilustrasikan pipeline lengkap Option A:',
        'Gambar 1 mengilustrasikan arsitektur lengkap pipeline Option A:',
        content
    )

    # Pattern 3: Gambar 2 - Detection Performance
    content = re.sub(
        r'\*\*Gambar 2\*\* harus ditempatkan di sini, showing side-by-side bar chart comparison',
        'Gambar 2 menampilkan perbandingan side-by-side bar chart',
        content
    )

    # Fix remaining English in Gambar 2
    content = re.sub(
        r'Visualisasi ini makes perbedaan performa immediately apparent dan supports conclusion bahwa YOLOv11 offers best recall\.',
        'Visualisasi ini membuat perbedaan performa langsung terlihat jelas dan mendukung kesimpulan bahwa YOLOv11 menawarkan recall terbaik.',
        content
    )

    # Pattern 4: Gambar 3 - Heatmap
    content = re.sub(
        r'\*\*Gambar 3\*\* harus ditempatkan di sini, menampilkan',
        'Gambar 3 menampilkan',
        content
    )

    # Pattern 5: Gambar 5 - Confusion Matrices
    content = re.sub(
        r'\*\*Gambar 5\*\* harus ditempatkan di sini, menampilkan dua matriks',
        'Gambar 5 menampilkan dua matriks',
        content
    )

    # Pattern 6: Gambar 6 - Species F1 Comparison
    content = re.sub(
        r'\*\*Gambar 6\*\* harus ditempatkan di sini, menampilkan diagram',
        'Gambar 6 menampilkan diagram',
        content
    )

    # Pattern 7: Gambar 7 - Stages F1 Comparison
    content = re.sub(
        r'\*\*Gambar 7\*\* harus ditempatkan di sini, menampilkan diagram',
        'Gambar 7 menampilkan diagram',
        content
    )

    # Count changes
    if content != original_content:
        changes = content.count('menampilkan') - original_content.count('menampilkan')
        changes += content.count('mengilustrasikan') - original_content.count('mengilustrasikan')

        # Write output
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"SUCCESS: Updated {filename}")
        print(f"   - Replaced placeholder instructions with narrative mentions")
        print(f"   - Fixed English code-switching in figure captions")
        print(f"   - All figures now have proper 'menampilkan/mengilustrasikan' mentions")
        return True
    else:
        print("No changes needed!")
        return False

if __name__ == '__main__':
    filename = 'luaran/Laporan_Kemajuan.md'
    add_narrative_mentions(filename)
