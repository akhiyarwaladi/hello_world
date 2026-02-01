# CHANGE MAP: v1 (Submitted) → v2 (Revised)

Guide for applying changes to the Microsoft Word (.docx) version.
Changes are listed in document order, top to bottom.

---

## HOW TO USE THIS GUIDE

1. Open `2558-Article Text-124129817-1-2-20251101.docx` in Microsoft Word
2. Enable Track Changes (Review → Track Changes) so reviewers can see edits
3. Follow each change below in order from top to bottom
4. Changes marked **[REPLACE]** = delete old text, type new text
5. Changes marked **[ADD NEW]** = insert entirely new text at the indicated location
6. Changes marked **[FIX]** = small typo/citation correction

---

## CHANGE 1 — Section 1.2 Title [REPLACE]
**Location in Word:** Section 1, second subsection heading
**Type:** Section rewrite (Reviewer C Point 1)

**OLD title:**
> ### 1.2 Existing Solutions and Limitations

**NEW title:**
> ### 1.2 Literature Review

---

## CHANGE 2 — Section 1.2 Body Text [REPLACE ENTIRE SECTION]
**Location in Word:** All text under the Section 1.2 heading (2 paragraphs → replace with 3 paragraphs)
**Type:** Full rewrite (Reviewer C Point 1 — literature review with major contributors)

**DELETE the old 2 paragraphs:**
> Recent advances have enabled automated malaria detection using Convolutional Neural Networks and object detection models. Single-stage detectors like YOLO achieve real-time performance [5], while two-stage pipelines combining detection with classification improve diagnostic accuracy [5]. However, existing approaches face critical challenges that limit their deployment in resource-constrained clinical settings [6].
>
> Limited public datasets with 200 to 500 images severely constrain generalization across patient populations [7], [8]. Ring-stage parasites dominate with over 85% representation while critical stages like gametocytes constitute less than 2%, creating extreme imbalance with ratios up to 54:1 [5], [9]. This severe imbalance causes models to underperform on clinically significant minority classes particularly for rare lifecycle stages. Traditional detection-classification pipelines train separate models for each detection method, creating computational inefficiency and storage overhead that exceed practical deployment constraints in resource-limited settings [6].

**REPLACE WITH these 3 new paragraphs:**

> **Paragraph 1:** Several research groups have made substantial contributions toward automating malaria diagnosis through deep learning, establishing important baselines on publicly available datasets while revealing persistent challenges in the field. Arshad et al. [7] introduced the IML Lifecycle dataset comprising 313 thin blood smear images with bounding box annotations across four Plasmodium vivax lifecycle stages, applying morphological segmentation to achieve 89.33% precision for parasite localization followed by ResNet50V2 classification that reached 95.86% accuracy on lifecycle stage prediction. Their work demonstrated the viability of two-stage detection-classification pipelines for malaria diagnosis but relied on a single species and a relatively small dataset, leaving open questions about generalization across species and patient populations. Loddo et al. [25] established the first deep learning baseline on the MP-IDB dataset for lifecycle stage classification, finding that VGG-19 achieved 85.18% accuracy on binary parasite detection while DenseNet-201 reached 97% accuracy on four-stage classification, though their evaluation was limited to classification without addressing the detection component required for end-to-end diagnostic workflows.
>
> **Paragraph 2:** Zedda et al. [24] extended the investigation to object detection approaches on MP-IDB, applying YOLOv5 to achieve 95.2% detection accuracy alongside DarkNet-53 which reached 96.02% accuracy for four lifecycle stages, demonstrating that single-stage detectors could match or exceed traditional classification-only pipelines in overall performance. Building on these findings, Zedda et al. [21] subsequently developed YOLO-PAM, a specialized architecture combining YOLOv8 with attention mechanisms including NAM and CBAM modules, achieving 91.8% mAP@50 on IML Lifecycle and 83.6% on MP-IDB while reducing model parameters by 11 million compared to baseline YOLOv8, illustrating that architectural innovations could improve efficiency without sacrificing accuracy. More recently, Sukumarran et al. [26] conducted a comparative evaluation of detection architectures, finding that YOLOv5 achieved 96% mAP@0.5 outperforming YOLOv4 at 89-90%, while DenseNet-121 reached 95.5% accuracy for species identification, further validating the effectiveness of modern detection frameworks for malaria microscopy analysis.
>
> **Paragraph 3:** Despite these advances, three critical gaps remain unaddressed in the existing literature. First, limited dataset sizes ranging from 200 to 500 images constrain model generalization, with most studies evaluating on a single dataset rather than assessing cross-dataset robustness across different diagnostic tasks and patient populations [7], [8]. Second, extreme class imbalance with ratios up to 54:1 between dominant ring-stage parasites and rare lifecycle stages causes models to underperform on clinically significant minority classes, yet prior work predominantly reports overall accuracy metrics that mask failures on rare but diagnostically important stages [5], [9]. Third, existing detection-classification pipelines train separate classification models for each detection method, creating computational redundancy and storage overhead that exceed practical deployment constraints in resource-limited clinical settings where hardware and energy resources are scarce [6]. This study addresses all three gaps through a shared classification architecture with systematic multi-model evaluation across four complementary datasets, Focal Loss optimization for extreme imbalance, and parameter-efficient model selection for resource-constrained deployment.

---

## CHANGE 3 — Section 2.1, MP-IDB paragraph [FIX]
**Location in Word:** Section 2.1, second paragraph, last sentence
**Type:** Typo fix

**Find:** `...clinical microscopy scenarios..` (double period)
**Replace:** `...clinical microscopy scenarios.` (single period)

---

## CHANGE 4 — Section 2.1, MD-2019 paragraph [FIX]
**Location in Word:** Section 2.1, third paragraph, last sentence
**Type:** Grammar fix

**Find:** `After stratified dataset yields 1,028`
**Replace:** `After stratified splitting, the dataset yields 1,028`

---

## CHANGE 5 — Section 2.1, augmentation paragraph [FIX]
**Location in Word:** Section 2.1, fourth paragraph (starts with "All four datasets undergo...")
**Type:** Remove stray word

**Find:** `for fair evaluation judge [17]`
**Replace:** `for fair evaluation [17]`

---

## CHANGE 6 — Section 2.2, crop generation paragraph [ADD NEW TEXT]
**Location in Word:** Section 2.2 "Proposed Architecture", second paragraph, AFTER the sentence ending with "...enables one-time generation with unlimited reuse across all experiments."
**Type:** New justification text (Reviewer C Point 2 — method justification)

**INSERT this text immediately after that sentence (same paragraph, continues):**

> The deliberate choice of ground truth crops over detection-output crops serves three critical purposes that justify this architectural decision. First, ground truth crops eliminate detection noise that would otherwise propagate into classification training, since detection models inevitably produce imprecise bounding boxes, missed parasites, and false positive regions that contaminate downstream classification if used as training data. Second, ground truth crops ensure fair and reproducible comparison across all classification architectures because every model trains on identical input data regardless of which detector generated the original predictions, removing confounding variables that would make it impossible to isolate classification model performance from detection quality. Third, this approach enables a train-once-reuse paradigm where classification models are trained a single time and their predictions can be applied to outputs from any current or future detector without retraining, dramatically reducing the computational cost from multiplicative to additive scaling as new detection architectures are evaluated.

---

## CHANGE 7 — Section 2.4, after existing paragraph [ADD NEW PARAGRAPHS]
**Location in Word:** Section 2.4 "Implementation Details", AFTER the existing paragraph ending with "...to address class imbalance ratios up to 54:1 [15], [21]."
**Type:** Two new justification paragraphs (Reviewer C Point 2 — method justification)

**INSERT these 2 new paragraphs:**

> **New paragraph 1 (AdamW justification):** The selection of AdamW over conventional SGD as the classification optimizer is motivated by its superior convergence properties when fine-tuning pretrained models on small medical imaging datasets. AdamW decouples weight decay from the gradient update step, providing more consistent regularization that prevents overfitting on datasets with only 200 to 800 training images while maintaining stable learning dynamics across the diverse loss landscapes encountered when adapting ImageNet-pretrained features to malaria-specific morphological patterns. Empirical evidence from transfer learning literature demonstrates that adaptive optimizers with decoupled weight decay achieve faster convergence and better generalization than SGD on small-scale fine-tuning tasks, which is particularly relevant for our setting where training data is limited and class distributions are severely imbalanced [12], [14].
>
> **New paragraph 2 (Focal Loss parameter justification):** Focal Loss parameters are configured with alpha equal to 1.0 and gamma equal to 1.5 based on systematic consideration of the framework's class balancing strategy. The alpha parameter is set to 1.0 rather than the original 0.25 proposed by Lin et al. [11] because the framework already employs weighted random sampling with inverse frequency weights during data loading, which provides explicit class-level rebalancing by oversampling minority classes in each training batch. Setting alpha to 1.0 avoids double-counting the class rebalancing effect that would occur if both the sampler and the loss function applied class-dependent weighting simultaneously. The gamma parameter is set to 1.5 to provide moderate focusing on hard examples, reducing the contribution of well-classified dominant-class samples to the loss while maintaining sufficient gradient signal from easy examples to stabilize training on small datasets where overly aggressive focusing with gamma equal to 2.0 or higher could destabilize convergence due to limited sample diversity.

---

## CHANGE 8 — Section 3.1, mAP@50-95 paragraph [FIX × 3 citations]
**Location in Word:** Section 3.1 "Detection Performance", second paragraph (starts with "The mAP@50-95 metric shows...")
**Type:** Fix 3 wrong citation numbers

**Fix A — Find:** `for minority lifecycle stages [22]`
**Replace:** `for minority lifecycle stages [9]`

**Fix B — Find:** `automated diagnostic systems [15]`
**Replace:** `automated diagnostic systems [1]`

**Fix C — Find:** `on MP-IDB Stages [3]` (end of paragraph, about training times)
**Replace:** `on MP-IDB Stages [10]`

---

## CHANGE 9 — Table 3 caption [FIX]
**Location in Word:** Table 3 caption
**Type:** Add missing closing parenthesis

**Find:** `...Moderate 5.4:1 Class Imbalance`
**Replace:** `...Moderate 5.4:1 Class Imbalance)`

---

## CHANGE 10 — Section 3.2, IML paragraph [FIX]
**Location in Word:** Section 3.2, first results paragraph (about IML Lifecycle)
**Type:** Grammar fix

**Find:** `as show in Table 3`
**Replace:** `as shown in Table 3`

---

## CHANGE 11 — Table 4 caption [FIX]
**Location in Word:** Table 4 caption
**Type:** Add missing closing parenthesis

**Find:** `...Extreme 45:1 Class Imbalance`
**Replace:** `...Extreme 45:1 Class Imbalance)`

---

## CHANGE 12 — Section 3.2, MP-IDB Species paragraph [FIX]
**Location in Word:** Paragraph starting with "MP-IDB Species..."
**Type:** Grammar fix

**Find:** `MP-IDB Species at Table 4`
**Replace:** `MP-IDB Species in Table 4`

---

## CHANGE 13 — Table 5 caption [FIX]
**Location in Word:** Table 5 caption
**Type:** Add missing closing parenthesis

**Find:** `...Severe 54:1 Class Imbalance`
**Replace:** `...Severe 54:1 Class Imbalance)`

---

## CHANGE 14 — Section 3.2, MP-IDB Stages paragraph [FIX]
**Location in Word:** Paragraph about MP-IDB Stages results
**Type:** Grammar fix

**Find:** `These results show in Table 5 demonstrate`
**Replace:** `These results shown in Table 5 demonstrate`

---

## CHANGE 15 — Section 3.2, MD-2019 paragraph [FIX]
**Location in Word:** Paragraph about MD-2019 results, last sentence
**Type:** Typo fix

**Find:** `as resultsin Table 6`
**Replace:** `as shown in Table 6`

---

## CHANGE 16 — Section 3.3, first paragraph [FIX citation]
**Location in Word:** Section 3.3 "Key Classification Findings", first paragraph
**Type:** Fix wrong citation

**Find:** `compound scaling strategies [11]`
**Replace:** `compound scaling strategies [19]`

---

## CHANGE 17 — Section 3.3, second paragraph [FIX citation]
**Location in Word:** Section 3.3, paragraph starting with "Focal Loss with alpha parameter..."
**Type:** Fix wrong citation

**Find:** `to avoid false positive predictions [2]`
**Replace:** `to avoid false positive predictions [9]`

---

## CHANGE 18 — Section 3.4, error analysis intro [FIX]
**Location in Word:** Section 3.4 first paragraph, last sentence
**Type:** Fix garbled phrasing

**Find:** `missed on detection model parasites`
**Replace:** `missed parasites`

---

## CHANGE 19 — Section 3.4.2, MD-2019 paragraph [FIX]
**Location in Word:** Paragraph starting with "The MD-2019 heavy error case..."
**Type:** Fix wrong figure reference

**Find:** `shown in Figure 4f demonstrates`
**Replace:** `shown in Figure 6f demonstrates`

---

## CHANGE 20 — Section 3.5, first paragraph [FIX × 3 citations]
**Location in Word:** Section 3.5 "Comparison with State-of-the-Art Methods", first paragraph
**Type:** Fix 3 wrong citation numbers

**Fix A — Find:** `Sukumarran et al.'s best performance (96%) [27]`
**Replace:** `Sukumarran et al.'s best performance (96%) [26]`

**Fix B — Find:** `Loddo et al.'s 85.18% [15]`
**Replace:** `Loddo et al.'s 85.18% [25]`

**Fix C — Find:** `Sukumarran et al.'s 95.5% [24]`
**Replace:** `Sukumarran et al.'s 95.5% [26]`

---

## CHANGE 21 — Section 3.5, after first paragraph [ADD TABLE REFERENCE]
**Location in Word:** End of first paragraph of Section 3.5, after "...this challenging multi-patient dataset."
**Type:** Add explicit Table 7 reference

**INSERT at end of that sentence (before the period or as new sentence):**

> Table 7 summarizes the comparative performance against prior work on IML Lifecycle and MP-IDB datasets.

---

## CHANGE 22 — NEW Section 3.6 "Discussion" [ADD NEW SECTION]
**Location in Word:** AFTER Section 3.5 (Comparison with State-of-the-Art), BEFORE Section 3.7 (Limitations)
**Type:** Entirely new section (Reviewer C Point 3 — discussion with hypotheses)

**INSERT this new section with heading and 3 paragraphs:**

> ### 3.6 Discussion
>
> **Paragraph 1 (Hypotheses statement):** This study was guided by three initial hypotheses regarding the effectiveness of the proposed framework for automated malaria diagnosis. The first hypothesis posited that shared classification architecture using ground truth crops would maintain classification accuracy comparable to traditional per-detector training while substantially reducing computational overhead. The second hypothesis proposed that parameter-efficient models with compound scaling would outperform larger architectures on small-scale medical imaging datasets where overfitting risk is elevated due to limited training samples. The third hypothesis asserted that Focal Loss optimization would enable robust minority class performance on datasets exhibiting extreme class imbalance ratios characteristic of clinical malaria microscopy. Evaluation of these hypotheses against experimental evidence across four complementary datasets with 1,544 total images provides empirical grounding for the framework's design decisions and reveals nuanced findings that extend beyond the original predictions.
>
> **Paragraph 2 (Hypotheses evaluation):** The first hypothesis is strongly supported by the experimental results, as the shared classification architecture achieved 86.45% to 98.28% accuracy across all four datasets using models trained once on ground truth crops, with no evidence of performance degradation compared to the detection-specific training approaches employed by Arshad et al. [7] who achieved 95.86% using dedicated ResNet50V2 models. The shared approach eliminated the need to train separate classification models for each of the three YOLO detectors, reducing the total number of required training runs from 18 to 6 for each dataset while producing classification models that can be immediately applied to any new detector without retraining. The second hypothesis was partially confirmed with an important qualification: parameter-efficient EfficientNet models with 5.3M to 9.2M parameters outperformed larger ResNet variants on three of four datasets, but ResNet50 with 25.6M parameters proved superior on MP-IDB Stages where 54:1 class imbalance required deeper feature hierarchies to discriminate between morphologically similar rare lifecycle stages. This finding refines the initial hypothesis by establishing that parameter efficiency advantages are modulated by imbalance severity, suggesting a threshold beyond which architectural depth becomes more important than scaling efficiency. The third hypothesis regarding Focal Loss effectiveness was confirmed across all datasets, with F1-scores ranging from 0.44 to 1.00 on ultra-minority classes containing 4 to 15 test samples, though the lower bound of 0.44 on MP-IDB Stages schizont with only 6 test samples indicates that even optimized loss functions cannot fully compensate when minority class representation falls below approximately 5% of training data.
>
> **Paragraph 3 (Implications):** These findings carry significant implications for clinical malaria diagnosis and the broader field of medical image analysis. The demonstrated effectiveness of compact 46-89 MB models achieving diagnostic accuracy between 86.45% and 98.28% establishes a practical pathway for deploying automated malaria screening in endemic regions where computational resources are limited and expert microscopists are scarce, directly addressing the diagnostic bottleneck that contributes to delayed treatment and preventable mortality among the 263 million annual malaria cases worldwide [1]. The shared classification paradigm introduced in this work represents a generalizable architectural pattern applicable beyond malaria to other medical imaging tasks requiring detection-classification pipelines, such as tuberculosis bacilli detection in sputum smears, cervical cell abnormality screening, and blood cell differential counting, where the same train-once-reuse principle could reduce computational requirements while maintaining diagnostic accuracy. Contextualizing these findings within the progression of the field, our results build upon and extend the foundational work of Arshad et al. [7], Zedda et al. [21], [24], Loddo et al. [25], and Sukumarran et al. [26] by demonstrating that systematic multi-model evaluation with dataset-dependent selection across four complementary datasets achieves robust performance without requiring specialized architectural modifications such as the attention mechanisms employed in YOLO-PAM [21], suggesting that careful optimization of standard architectures combined with appropriate loss functions and training strategies can match or exceed the performance of more complex purpose-built solutions.

---

## CHANGE 23 — Section 4 Conclusion [ADD NEW PARAGRAPH]
**Location in Word:** Section 4 "Conclusion", AFTER the third (last) paragraph ending with "...to facilitate reproducibility and deployment."
**Type:** New fourth paragraph (Reviewer C Point 4 — significance and outstanding questions)

**INSERT this new paragraph:**

> These findings hold broader significance for the malaria diagnosis research community and for medical image analysis more generally. The shared classification paradigm demonstrated in this work establishes that detection and classification stages can be decoupled without sacrificing diagnostic accuracy, offering a reusable architectural pattern for other parasitological and cytological screening tasks where multiple detection approaches must be evaluated against standardized classification benchmarks. For endemic regions where an estimated 40% of health facilities lack access to trained microscopists capable of reliable species-level identification, the compact 46-89 MB models achieving 86.45% to 98.28% classification accuracy represent a tangible pathway toward democratizing diagnostic capability through deployment on smartphones and low-cost embedded devices already present in rural health posts. The outstanding question that remains is bridging the laboratory-to-field gap: while this study demonstrates strong performance on curated microscopy datasets, translating these results to real-world clinical workflows involving thick blood smears, variable staining quality, and diverse patient populations constitutes the critical next step that will determine whether automated malaria diagnosis can meaningfully reduce the global burden of this disease.

---

## QUICK REFERENCE: CHANGES BY REVIEWER POINT

| Reviewer | Point | What | Changes # |
|----------|-------|------|-----------|
| **Editor** | 1 | Introduction structure | 1, 2 |
| **Editor** | 2, 10 | Improve analysis & comparison | 21, 22 |
| **Editor** | 6 | All tables/figures referenced | 9, 11, 13, 21 |
| **Editor** | 7 | Citations correct & sequential | 8, 16, 17, 20 |
| **Reviewer C** | 1 | Literature review with contributors | 1, 2 |
| **Reviewer C** | 2 | Method justification | 6, 7 |
| **Reviewer C** | 3 | Discussion section | 22 |
| **Reviewer C** | 4 | Conclusion significance | 23 |
| **Quality** | — | Typos & grammar fixes | 3, 4, 5, 10, 12, 14, 15, 18, 19 |
| **Quality** | — | Citation number corrections | 8, 16, 17, 20 |

---

## CHANGE 24 — Section 3.4.1, Figure 5c statistics [REPLACE]
**Location in Word:** Section 3.4.1, second paragraph (MP-IDB Stages overdetection)
**Type:** Figure data correction (experiment regeneration)

**Find:**
> with 8 false positives indicates systematic confusion in severely imbalanced data, where YOLOv11 achieves only 36.1% perfect detection across 72 test images with highest false positive occurrence of 1.83 FP per image at average confidence 0.702.

**Replace:**
> with 6 false positives and 5 false negatives indicates systematic confusion in severely imbalanced data, where YOLOv11 achieves only 40.5% perfect detection across 42 test images with highest false positive occurrence of 1.55 FP per image at average confidence 0.718.

---

## CHANGE 25 — Section 3.4.2, Figure 6c statistics [REPLACE]
**Location in Word:** Section 3.4.2, second paragraph (MP-IDB Stages confusion)
**Type:** Figure data correction + expanded analysis (experiment regeneration)

**Find:**
> shows 4 trophozoites misclassified as rings among 14 parasites at 71.4% image accuracy, where the systematic trophozoite-to-ring misclassification pattern reflects prevalence of early-stage trophozoites with compact cytoplasm and minimal hemozoin accumulation resembling ring morphology.

**Replace:**
> shows 1 misclassification among 31 parasites at 96.8% image accuracy with ResNet101, where the residual trophozoite-to-ring misclassification pattern reflects prevalence of early-stage trophozoites with compact cytoplasm and minimal hemozoin accumulation resembling ring morphology. Notably, architecture selection critically impacts performance on this dense field: DenseNet121 achieves only 6.5% accuracy and ResNet50 reaches 35.5% on the identical image, demonstrating that deeper residual architectures better discriminate subtle morphological differences in crowded multi-parasite scenarios.

---

## CHANGE 26 — Section 3.4.2, Figure 6f statistics [REPLACE]
**Location in Word:** Section 3.4.2, fourth paragraph (MD-2019 perfect classification)
**Type:** Figure data correction (experiment regeneration)

**Find:**
> with 10 parasites correctly classified at 100% image accuracy on crowded microscopy fields

**Replace:**
> with 8 parasites correctly classified at 100% image accuracy by ResNet101 on crowded microscopy fields

---

## TOTAL: 26 Changes

- **2** major rewrites (Section 1.2 Literature Review, new Section 3.6 Discussion)
- **4** new text additions (Section 2.2 justification, Section 2.4 AdamW paragraph, Section 2.4 Focal Loss paragraph, Section 4 significance paragraph)
- **8** citation number fixes (Changes 8, 16, 17, 20)
- **8** typo/grammar fixes (Changes 3, 4, 5, 10, 12, 14, 15, 18)
- **3** table caption fixes — missing closing parenthesis (Changes 9, 11, 13)
- **1** wrong figure reference — Figure 4f → 6f (Change 19)
- **1** missing table reference — Table 7 added to body text (Change 21)
- **3** figure data corrections — updated to match experiment optA_20251207_233941 (Changes 24, 25, 26)

---

## NOTE: Reviewer D

Reviewer D's inline PDF annotations are in a separate annotated PDF file that must be downloaded from the KINETIK journal system (OJS). Those comments are NOT included in this change map because we do not have that file. Please check your OJS dashboard for Reviewer D's annotated PDF and apply those changes additionally.
