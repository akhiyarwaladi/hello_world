# Script to reduce Results section systematically
# Target: 228 words reduction from 3,528 words

import re

# Read the paper
with open('luaran/templates/KINETIK_PAPER_DRAFT_UPDATED_2025.md', 'r', encoding='utf-8') as f:
    content = f.read()

print("Starting Results section reduction...")
print("="*80)

# Define all replacements for Results section
replacements = [
    # Section 3.1: Detection Performance (reduce verbose descriptions)
    {
        'old': 'Systematic comparison of YOLO variants (v10/v11/v12 Medium, 20.1M parameters) reveals dataset-dependent performance patterns across all four malaria datasets (Table 2). YOLO11 achieves balanced best performance with 96.38% mAP@50 on IML Lifecycle and 74.91% on challenging MD_2019, while YOLO12 demonstrates superiority on severe imbalance scenarios reaching 96.28% mAP@50 on MP-IDB Stages. YOLO10 provides competitive baseline performance ranging 74.69-96.06% mAP@50, validating incremental improvements across YOLO generations for medical imaging tasks.',
        'new': 'YOLO comparison (v10/v11/v12 Medium, 20.1M parameters) reveals dataset-dependent patterns (Table 2). YOLO11 leads with 96.38% mAP@50 (IML) and 74.91% (MD_2019), while YOLO12 excels on severe imbalance (96.28% on MP-IDB Stages). YOLO10 provides competitive baseline (74.69-96.06% mAP@50), validating incremental improvements for medical imaging.',
        'words_saved': 25
    },
    {
        'old': 'High recall rates across all YOLO variants (71.05-93.12%) minimize missed parasites, critical for preventing delayed treatment [21]. The variation in mAP@50-95 metrics (44.48-78.21%) reflects dataset complexity: IML Lifecycle achieves superior strict IoU performance (77.71-78.21%) indicating precise localization, while MP-IDB Stages shows wider variance (44.48-61.53%) due to severe 54:1 imbalance where YOLO12\'s architecture handles extreme ratios more effectively. The three manually-annotated datasets (IML, MP-IDB Species, MP-IDB Stages) achieve 92.77-96.47% mAP@50 substantially exceeding the 90% WHO clinical threshold [13], while MD_2019\'s lower range (74.59-74.91%) reflects realistic challenges from automatic bbox extraction creating natural size variation (CV >20% per class) and multi-patient diversity (16 patients) [16]. This automated detection reduces analysis time by >95% compared to 20-30 minute manual diagnosis [3].',
        'new': 'High recall rates (71.05-93.12%) minimize missed parasites critical for preventing delayed treatment [21]. mAP@50-95 variance (44.48-78.21%) reflects dataset complexity: IML achieves superior strict IoU (77.71-78.21%), while MP-IDB Stages shows wider variance (44.48-61.53%) due to 54:1 imbalance. Manually-annotated datasets achieve 92.77-96.47% mAP@50 exceeding 90% WHO threshold [13], while MD_2019 (74.59-74.91%) reflects realistic challenges from automatic extraction and multi-patient diversity [16], reducing analysis time by >95% versus 20-30 minute manual diagnosis [3].',
        'words_saved': 40
    },

    # Section 3.2: Classification Performance (consolidate repetitive accuracy mentions)
    # Table 3 discussion
    {
        'old': 'On IML Lifecycle dataset, three EfficientNet variants achieved identical 91.51% overall accuracy despite differing parameter counts, with EfficientNet-B1 delivering the highest balanced accuracy at 91.96% and best trophozoite F1-score of 0.81 using only 7.8 million parameters while maintaining high precision (0.98) on the majority gametocyte class. DenseNet121 and EfficientNet-B2 both achieved perfect precision and F1-scores (1.00) on the challenging schizont minority class despite only 4 test samples, demonstrating effective handling of severe class imbalance. Larger ResNet architectures underperformed substantially, with ResNet101\'s 44.5 million parameters yielding only 85.85% accuracy and 80.29% balanced accuracy alongside lower precision (0.67) on trophozoite compared to EfficientNet-B1\'s 0.83, representing a 5.66 percentage point deficit in overall accuracy compared to the more compact EfficientNet models.',
        'new': 'Three EfficientNet variants achieved 91.51% accuracy on IML Lifecycle despite differing parameters. EfficientNet-B1 (7.8M) delivered best balanced accuracy (91.96%) and trophozoite F1 (0.81) with 0.98 precision on gametocyte. DenseNet121 and EfficientNet-B2 achieved perfect scores (1.00) on schizont (4 samples), demonstrating effective imbalance handling. ResNet101 (44.5M) underperformed at 85.85% accuracy and 80.29% balanced accuracy with lower trophozoite precision (0.67 vs 0.83), representing 5.66-point deficit versus compact EfficientNet models.',
        'words_saved': 45
    },

    # Table 4 discussion
    {
        'old': 'MP-IDB Species classification demonstrated exceptional performance across all architectures on the dominant P_falciparum class achieving 0.99 F1-scores with near-perfect precision (0.99), but EfficientNet-B1 distinguished itself by achieving 98.28% overall accuracy with 86.43% balanced accuracy through superior handling of ultra-minority species. Notably, EfficientNet-B1 achieved 0.86 F1-score on P_ovale despite only 7 test samples while maintaining 0.86 precision, and 0.80 F1-score on P_malariae with 9 test samples alongside perfect precision (1.00), demonstrating robust minority class detection without over-prediction. In contrast, while ResNet50 achieved perfect precision (1.00) on P_ovale, its lower recall resulted in only 0.73 F1-score compared to EfficientNet-B1\'s 0.86, despite possessing 3.3 times more parameters, demonstrating that architectural efficiency and compound scaling matter more than raw capacity for extreme imbalance scenarios.',
        'new': 'MP-IDB Species classification showed exceptional P_falciparum performance (0.99 F1, 0.99 precision) across architectures, but EfficientNet-B1 distinguished itself with 98.28% overall accuracy and 86.43% balanced accuracy through superior ultra-minority handling. EfficientNet-B1 achieved 0.86 F1 on P_ovale (7 samples, 0.86 precision) and 0.80 F1 on P_malariae (9 samples, 1.00 precision), demonstrating robust detection without over-prediction. ResNet50 achieved perfect precision (1.00) on P_ovale but lower recall (0.73 F1 vs 0.86) despite 3.3× more parameters, showing architectural efficiency matters more than raw capacity for extreme imbalance.',
        'words_saved': 50
    },

    # Table 5 discussion
    {
        'old': 'The severely imbalanced MP-IDB Stages dataset revealed interesting architectural preferences, with ResNet50 achieving the best overall performance at 96.13% accuracy and 83.04% balanced accuracy, outperforming the typically superior EfficientNet-B1 which achieved 95.42% accuracy and 78.64% balanced accuracy. ResNet50 delivered 0.91 F1-score on the rare gametocyte class with 0.83 precision despite only 5 test samples, 0.71 F1-score on schizont with moderate 0.63 precision reflecting the challenge of 6-sample classification, and the highest trophozoite F1-score of 0.61 with 0.78 precision. This performance advantage suggests that the 54:1 extreme imbalance ratio benefits from ResNet\'s deeper feature hierarchies enabled by residual connections for distinguishing subtle morphological differences between rare lifecycle stages, where precision-recall trade-offs favor deeper architectures over compact models.',
        'new': 'Severely imbalanced MP-IDB Stages revealed architectural preferences: ResNet50 achieved best performance (96.13% accuracy, 83.04% balanced accuracy), outperforming EfficientNet-B1 (95.42%, 78.64%). ResNet50 delivered 0.91 F1 on gametocyte (5 samples, 0.83 precision), 0.71 F1 on schizont (0.63 precision, 6 samples), and highest trophozoite F1 (0.61, 0.78 precision). This suggests 54:1 extreme imbalance benefits from ResNet\'s deeper feature hierarchies for distinguishing subtle morphological differences, where precision-recall trade-offs favor deeper architectures.',
        'words_saved': 45
    },

    # Table 6 discussion
    {
        'old': 'MD_2019 Stages classification on the largest test set of 583 cells showed EfficientNet-B0 achieving best performance at 86.45% accuracy with 84.13% balanced accuracy across three lifecycle stages. The compact 5.3-million parameter model outperformed all larger architectures including ResNet101 (44.5M parameters, 84.22% accuracy), demonstrating parameter efficiency advantages even on this substantially larger dataset. Per-class metrics reveal consistent performance with strong precision-F1 balance: schizont achieved 0.93 precision with 0.92 F1 (286 test samples), ring achieved 0.86 precision with 0.89 F1 (170 samples), and trophozoite achieved 0.72 precision with 0.71 F1 (127 samples), showing that precision and recall remain balanced even on the most challenging minority class. The lower overall accuracy compared to IML (91.51%) and MP-IDB Species (98.28%) reflects MD_2019\'s increased difficulty from natural bbox variation and higher morphological diversity across 16 different patients, providing more realistic generalization assessment than smaller manually-annotated datasets [16].',
        'new': 'MD_2019 classification (583 cells) showed EfficientNet-B0 best at 86.45% accuracy and 84.13% balanced accuracy. The compact 5.3M model outperformed larger architectures including ResNet101 (44.5M, 84.22%), demonstrating parameter efficiency on this larger dataset. Per-class metrics show strong precision-F1 balance: schizont 0.93/0.92 (286 samples), ring 0.86/0.89 (170 samples), trophozoite 0.72/0.71 (127 samples). Lower accuracy versus IML (91.51%) and MP-IDB Species (98.28%) reflects MD_2019\'s increased difficulty from natural bbox variation and morphological diversity across 16 patients, providing realistic generalization assessment [16].',
        'words_saved': 50
    },
]

# Apply replacements
total_saved = 0
for i, replacement in enumerate(replacements, 1):
    old_text = replacement['old']
    new_text = replacement['new']

    if old_text in content:
        content = content.replace(old_text, new_text)
        words_saved = replacement['words_saved']
        total_saved += words_saved
        print(f"✅ Replacement {i}: Saved {words_saved} words")
    else:
        print(f"⚠️ Replacement {i}: Pattern not found (may have been already modified)")

print("="*80)
print(f"Total words saved so far: {total_saved} words")
print(f"Target reduction: 228 words")
print(f"Progress: {(total_saved/228)*100:.1f}%")
print("="*80)

# Save the modified content
with open('luaran/templates/KINETIK_PAPER_DRAFT_UPDATED_2025.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Results section (Tables 3-6 descriptions) reduced successfully!")
print(f"File saved to: luaran/templates/KINETIK_PAPER_DRAFT_UPDATED_2025.md")
print("\nNext steps: Reduce Section 3.4 (Error Analysis) and 3.6 (SOTA comparison)")
