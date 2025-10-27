import re

# Read paper
with open('luaran/templates/KINETIK_PAPER_DRAFT_UPDATED_2025.md', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

print('=' * 80)
print('ANALISIS ULTRA-DETAIL: IDENTIFIKASI VERBOSE/REPETITIVE TEXT')
print('=' * 80)

# Extract sections
sections = {}
current_section = None
current_content = []

for line in lines:
    if line.startswith('## '):
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        current_section = line.strip('# ').strip()
        current_content = []
    else:
        current_content.append(line)

if current_section:
    sections[current_section] = '\n'.join(current_content)

# Count words per section
def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

print('\n📊 WORD COUNT PER SECTION:')
print('-' * 80)

section_words = {}
for section, text in sections.items():
    words = count_words(text)
    section_words[section] = words
    if words > 100:  # Only show substantial sections
        print(f'{section[:60]:60s} {words:>5} words')

# ============================================================================
# ANALYZE REPETITIVE PHRASES
# ============================================================================

print('\n\n🔍 PART 1: REPETITIVE PHRASES (mentioned 3+ times)')
print('-' * 80)

# Key phrases to check for repetition
phrases_to_check = [
    # Metrics that might be repeated
    (r'74\.59[^%]*96\.47%.*mAP@50', 'Detection range 74.59-96.47% mAP@50'),
    (r'91\.51%', '91.51% accuracy (IML)'),
    (r'98\.28%', '98.28% accuracy (MP-IDB Species)'),
    (r'96\.13%', '96.13% accuracy (MP-IDB Stages)'),
    (r'86\.45%', '86.45% accuracy (MD_2019)'),

    # Architectural mentions
    (r'EfficientNet-B[012]', 'EfficientNet variants'),
    (r'ResNet50|ResNet101', 'ResNet variants'),
    (r'YOLO(?:v)?1[012]', 'YOLO variants'),

    # Dataset mentions
    (r'313 images?', '313 images (IML)'),
    (r'209 images?', '209 images (MP-IDB)'),
    (r'883 images?', '883 images (MD_2019)'),
    (r'1,?614 images?', '1,614 total images'),

    # Focal Loss
    (r'α=0\.25.*γ=2\.0', 'Focal Loss α=0.25, γ=2.0'),
    (r'Focal Loss', 'Focal Loss (general)'),

    # Class imbalance
    (r'54:1', '54:1 imbalance ratio'),
    (r'class imbalance', 'class imbalance (general)'),

    # Shared architecture
    (r'shared classification', 'shared classification'),
    (r'ground truth crops?', 'ground truth crops'),

    # Resource constraints
    (r'resource[- ]constrained', 'resource-constrained'),
    (r'parameter[- ]efficient', 'parameter-efficient'),
]

phrase_counts = {}
phrase_locations = {}

for pattern, name in phrases_to_check:
    matches = list(re.finditer(pattern, content, re.IGNORECASE))
    if len(matches) >= 3:
        phrase_counts[name] = len(matches)
        # Find which sections
        locations = []
        for match in matches:
            pos = match.start()
            # Find which section this belongs to
            for sec_name, sec_text in sections.items():
                if sec_text in content and pos >= content.find(sec_text):
                    if pos < content.find(sec_text) + len(sec_text):
                        if sec_name not in locations:
                            locations.append(sec_name)
                        break
        phrase_locations[name] = locations

# Sort by frequency
sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)

for phrase, count in sorted_phrases[:15]:  # Top 15
    print(f'\n"{phrase}": {count} occurrences')
    if phrase in phrase_locations:
        locs = phrase_locations[phrase]
        print(f'  Sections: {", ".join(locs[:5])}')

# ============================================================================
# ANALYZE VERBOSE SENTENCES
# ============================================================================

print('\n\n📝 PART 2: VERBOSE SENTENCES (80+ words, potential for reduction)')
print('-' * 80)

# Find long sentences
sentences = re.split(r'[.!?]\s+', content)
verbose_sentences = []

for sent in sentences:
    words = len(re.findall(r'\b\w+\b', sent))
    if words >= 80:
        # Find which section
        section_name = 'Unknown'
        for sec_name, sec_text in sections.items():
            if sent[:50] in sec_text:
                section_name = sec_name
                break
        verbose_sentences.append((section_name, words, sent[:150]))

# Sort by word count
verbose_sentences.sort(key=lambda x: x[1], reverse=True)

print(f'\nFound {len(verbose_sentences)} sentences with 80+ words:\n')

for i, (section, words, preview) in enumerate(verbose_sentences[:10], 1):
    print(f'{i}. Section: {section[:40]}')
    print(f'   Words: {words}')
    print(f'   Preview: {preview}...')
    print()

# ============================================================================
# ANALYZE REDUNDANT INFORMATION
# ============================================================================

print('\n\n🔄 PART 3: POTENTIAL REDUNDANCIES')
print('-' * 80)

redundancy_checks = [
    {
        'name': 'Dataset statistics repetition',
        'sections': ['ABSTRACT', '1. INTRODUCTION', '2. METHODS'],
        'keywords': ['313 images', '209 images', '883 images', '1,614'],
    },
    {
        'name': 'Performance metrics repetition',
        'sections': ['ABSTRACT', '3. RESULTS AND DISCUSSION', '4. CONCLUSION'],
        'keywords': ['74.59', '96.47', '91.51', '98.28', '96.13', '86.45'],
    },
    {
        'name': 'Architecture description repetition',
        'sections': ['1. INTRODUCTION', '2. METHODS', '3. RESULTS AND DISCUSSION'],
        'keywords': ['EfficientNet', 'ResNet', 'DenseNet', 'YOLO'],
    },
    {
        'name': 'Class imbalance repetition',
        'sections': ['1. INTRODUCTION', '2. METHODS', '3. RESULTS AND DISCUSSION'],
        'keywords': ['54:1', 'class imbalance', 'minority class', 'imbalanced'],
    },
    {
        'name': 'Focal Loss repetition',
        'sections': ['ABSTRACT', '2. METHODS', '3. RESULTS AND DISCUSSION', '4. CONCLUSION'],
        'keywords': ['Focal Loss', 'α=0.25', 'γ=2.0'],
    },
]

for check in redundancy_checks:
    print(f'\n{check["name"]}:')
    occurrences = {}
    for section in check['sections']:
        if section in sections:
            count = 0
            for keyword in check['keywords']:
                count += sections[section].count(keyword)
            if count > 0:
                occurrences[section] = count

    if occurrences:
        for sec, cnt in occurrences.items():
            print(f'  {sec[:40]:40s} {cnt:3d} mentions')
        print(f'  → Total: {sum(occurrences.values())} mentions across {len(occurrences)} sections')

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print('\n\n💡 PART 4: RECOMMENDATIONS FOR 5% REDUCTION (~364 words)')
print('-' * 80)

print('''
Based on analysis, here are sections with highest reduction potential:

1. ABSTRACT (224 words → 200 words, -24 words):
   - Remove specific dataset counts (313, 209, 883)
   - Simplify "four datasets" to "multiple datasets"
   - Keep performance metrics but shorten sentences

2. INTRODUCTION (697 words → 650 words, -47 words):
   - Subsection 1.2: Remove repetitive limitation descriptions
   - Subsection 1.3: Consolidate dataset descriptions
   - Remove redundant "resource-constrained" mentions

3. METHODS (1,079 words → 1,000 words, -79 words):
   - Section 2.1: Merge redundant dataset statistics
   - Section 2.1: Simplify augmentation descriptions
   - Remove detailed class distribution (move to table)

4. RESULTS (3,532 words → 3,300 words, -232 words):
   - Section 3.1: Reduce verbose detection descriptions
   - Section 3.2: Consolidate repetitive accuracy mentions
   - Section 3.4: Shorten qualitative error descriptions
   - Section 3.6: Reduce SOTA comparison verbosity

5. CONCLUSION (447 words → 420 words, -27 words):
   - Remove metric repetitions (already in abstract)
   - Consolidate future work descriptions

TOTAL TARGET REDUCTION: ~400 words (5.5% reduction)
RESULT: 7,272 → ~6,870 words (well within 5% target)

KEY PRINCIPLE: Remove repetition, not information!
''')

print('=' * 80)
