import pandas as pd
import os

print('=' * 80)
print('ESTIMASI JUMLAH HALAMAN PAPER DI MICROSOFT WORD')
print('=' * 80)

# ============================================================================
# PART 1: ANALISIS KOMPONEN PAPER
# ============================================================================

print('\n📊 PART 1: KOMPONEN PAPER')
print('-' * 80)

# Word counts
main_text_words = 6179
references_words = 1093
total_words = 7272

print(f'Main Text (tanpa References):  {main_text_words:>6} kata')
print(f'References Section:            {references_words:>6} kata')
print(f'TOTAL TEXT:                    {total_words:>6} kata')

# Tables
tables = [
    'Table1_Dataset_Augmentation.xlsx',
    'Table2_Detection_Performance.xlsx',
    'Table3_IML_Classification.xlsx',
    'Table4_Species_Classification.xlsx',
    'Table5_Stages_Classification.xlsx',
    'Table6_MD2019_Classification.xlsx',
    'Table7_Comparison_SOTA.xlsx'
]

print(f'\nTabel yang digunakan:          {len(tables)} tabel')

# Figures
figures = 14  # 1, 2, 3a-f (6), 4a-f (6)
print(f'Gambar yang digunakan:         {figures} gambar')

# ============================================================================
# PART 2: ANALISIS UKURAN TABEL
# ============================================================================

print('\n\n📏 PART 2: ANALISIS UKURAN TABEL')
print('-' * 80)

table_dimensions = {}
table_path = 'luaran/templates/tables/'

for table_file in tables:
    try:
        df = pd.read_excel(table_path + table_file)
        rows, cols = df.shape
        table_dimensions[table_file] = (rows, cols)
        table_name = table_file.replace('.xlsx', '').replace('_', ' ')
        print(f'{table_name:45s} {rows:3d} rows × {cols:2d} cols')
    except Exception as e:
        print(f'{table_file:45s} ERROR: {str(e)[:30]}')

# ============================================================================
# PART 3: ESTIMASI HALAMAN FORMAT WORD
# ============================================================================

print('\n\n📄 PART 3: ESTIMASI HALAMAN - FORMAT WORD STANDAR')
print('-' * 80)
print('(Asumsi: Format single-column, font 12pt, margin normal)')
print()

# Text pages
words_per_page_single = 500  # Single column, 12pt, normal spacing
text_pages = total_words / words_per_page_single

print(f'Text-only (main + references):')
print(f'  {total_words} kata ÷ {words_per_page_single} kata/halaman = {text_pages:.1f} halaman')

# Table pages estimation
# Small tables (4-6 rows): 0.3-0.4 page
# Medium tables (7-12 rows): 0.5-0.7 page
# Large tables (13+ rows): 0.8-1.2 pages

table_pages = 0
print(f'\nEstimasi halaman untuk tabel:')
for table_file, (rows, cols) in table_dimensions.items():
    if rows <= 6:
        pages = 0.35
    elif rows <= 12:
        pages = 0.6
    else:
        pages = 1.0
    table_pages += pages
    table_name = table_file.replace('.xlsx', '').replace('_', ' ')
    print(f'  {table_name:45s} ≈ {pages:.2f} halaman')

print(f'  {"TOTAL TABEL":45s} ≈ {table_pages:.1f} halaman')

# Figure pages estimation
# Each figure: ~0.4 page (including caption and spacing)
figure_pages = figures * 0.4
print(f'\nEstimasi halaman untuk gambar:')
print(f'  {figures} gambar × 0.4 halaman/gambar = {figure_pages:.1f} halaman')

# Total
total_pages_single = text_pages + table_pages + figure_pages
print(f'\n{"="*80}')
print(f'TOTAL ESTIMASI (Single-Column Format):')
print(f'  Text:    {text_pages:5.1f} halaman')
print(f'  Tabel:   {table_pages:5.1f} halaman')
print(f'  Gambar:  {figure_pages:5.1f} halaman')
print(f'  {"─"*30}')
print(f'  TOTAL:   {total_pages_single:5.1f} halaman')
print(f'{"="*80}')

# ============================================================================
# PART 4: ESTIMASI HALAMAN FORMAT JURNAL DOUBLE-COLUMN
# ============================================================================

print('\n\n📄 PART 4: ESTIMASI HALAMAN - FORMAT JURNAL KINETIK')
print('-' * 80)
print('(Asumsi: Format double-column, font 10pt, IEEE/ACM style)')
print()

# In double-column format, more words fit per page
words_per_page_double = 750  # Double column, 10pt, tight spacing
text_pages_double = total_words / words_per_page_double

print(f'Text-only (main + references):')
print(f'  {total_words} kata ÷ {words_per_page_double} kata/halaman = {text_pages_double:.1f} halaman')

# In double-column, tables often span full width (more space)
table_pages_double = 0
print(f'\nEstimasi halaman untuk tabel (full-width):')
for table_file, (rows, cols) in table_dimensions.items():
    if rows <= 6:
        pages = 0.4
    elif rows <= 12:
        pages = 0.7
    else:
        pages = 1.2
    table_pages_double += pages
    table_name = table_file.replace('.xlsx', '').replace('_', ' ')
    print(f'  {table_name:45s} ≈ {pages:.2f} halaman')

print(f'  {"TOTAL TABEL":45s} ≈ {table_pages_double:.1f} halaman')

# Figures in double-column: some can be single-column width (saves space)
# Assume: 8 figures single-column (0.25 page), 6 figures full-width (0.5 page)
figure_pages_double = (8 * 0.25) + (6 * 0.5)
print(f'\nEstimasi halaman untuk gambar:')
print(f'  8 gambar × 0.25 halaman (single-col) = {8*0.25:.1f} halaman')
print(f'  6 gambar × 0.50 halaman (full-width)  = {6*0.5:.1f} halaman')
print(f'  {"TOTAL GAMBAR":45s} = {figure_pages_double:.1f} halaman')

# Total
total_pages_double = text_pages_double + table_pages_double + figure_pages_double
print(f'\n{"="*80}')
print(f'TOTAL ESTIMASI (Double-Column Format - KINETIK):')
print(f'  Text:    {text_pages_double:5.1f} halaman')
print(f'  Tabel:   {table_pages_double:5.1f} halaman')
print(f'  Gambar:  {figure_pages_double:5.1f} halaman')
print(f'  {"─"*30}')
print(f'  TOTAL:   {total_pages_double:5.1f} halaman')
print(f'{"="*80}')

# ============================================================================
# PART 5: KESIMPULAN DAN REKOMENDASI
# ============================================================================

print('\n\n💡 PART 5: KESIMPULAN DAN REKOMENDASI')
print('-' * 80)

print(f'\n1. FORMAT WORD STANDAR (Single-Column):')
print(f'   Estimasi: {total_pages_single:.0f}-{total_pages_single+1:.0f} halaman')
print(f'   → Cocok untuk: Draft review, submission ke arXiv')

print(f'\n2. FORMAT JURNAL KINETIK (Double-Column):')
print(f'   Estimasi: {total_pages_double:.0f}-{total_pages_double+1:.0f} halaman')
print(f'   → Cocok untuk: Final journal submission')

print(f'\n3. PERBANDINGAN DENGAN TARGET JURNAL:')
print(f'   - KINETIK biasanya: 8-12 halaman per paper')
print(f'   - Paper ini: {total_pages_double:.0f}-{total_pages_double+1:.0f} halaman')
if total_pages_double <= 12:
    print(f'   ✅ STATUS: SESUAI dengan range jurnal KINETIK')
else:
    print(f'   ⚠️  STATUS: Sedikit melebihi range optimal')

print(f'\n4. BREAKDOWN KONTEN:')
print(f'   - Teks (main + refs): {(text_pages_double/total_pages_double)*100:.1f}%')
print(f'   - Tabel:              {(table_pages_double/total_pages_double)*100:.1f}%')
print(f'   - Gambar:             {(figure_pages_double/total_pages_double)*100:.1f}%')

print(f'\n5. CATATAN PENTING:')
print(f'   • Paper ini termasuk DENSE dengan 7 tabel + 14 gambar')
print(f'   • Rasio visual/text sangat tinggi → bagus untuk technical paper')
print(f'   • Total 33 references → comprehensive literature review')
print(f'   • Word count {total_words} kata → dalam range journal article')

print('\n' + '='*80)
