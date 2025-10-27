# CITATION CORRECTION PLAN - KINETIK PAPER
## Step-by-Step Fixes with Exact Changes

**Total Issues**: 6 Critical + 3 High Priority = **9 Must-Fix**
**Approach**: Fix references FIRST, then update all citations in text

---

## PHASE 1: ADD MISSING REFERENCES (Insert New Entries)

### NEW REFERENCE A: MP-IDB Dataset (currently wrongly cited as [15])
**Insert after current [7] or [8]**

```
[NEW-A] A. Loddo and C. Di Ruberto, "MP-IDB: The Malaria Parasite Image Database for Image Processing and Analysis,"
in Processing and Analysis of Biomedical Information, F. Gargiulo, V. Miele, V. Moscato, D. Picariello, and A. Sansone, Eds.
Cham: Springer International Publishing, 2019, pp. 57–65. doi: 10.1007/978-3-030-13835-6_7
```

**Where to cite**: Line 354 "MP-IDB: Available through Loddo et al. [NEW-A]"

---

### NEW REFERENCE B: Zedda YOLO-PAM (currently wrongly cited as [24])
**Replace current [24] (GANs) with**:

```
[24] L. Zedda, A. Loddo, and C. Di Ruberto, "YOLO-PAM: Parasite-Attention-Based Model for Efficient Malaria Detection,"
J. Imaging, vol. 9, no. 12, p. 266, Nov. 2023. doi: 10.3390/jimaging9120266
```

**Where to cite**: Line 243 "Zedda et al. [24] introduced YOLO-PAM"

**What to do with OLD [24] (GANs)?**:
- Move to new number or DELETE if not cited elsewhere
- GANs paper CAN be cited at Line 253 for "synthetic data generation using GANs"

---

### NEW REFERENCE C: Loddo 2022 Classification Study (currently wrongly cited as [23])
**Replace current [23] (Buda Focal Loss) with**:

```
[23] A. Loddo, C. Fadda, and C. Di Ruberto, "An Empirical Evaluation of Convolutional Networks for Malaria Diagnosis,"
J. Imaging, vol. 8, no. 3, p. 66, Mar. 2022. doi: 10.3390/jimaging8030066
```

**Where to cite**: Line 243 "Loddo et al. [23] evaluated multiple CNN architectures"

**What to do with OLD [23] (Buda Focal Loss)?**:
- Can be kept as different number if cited for Focal Loss theory
- Check if actually cited in text - if not, DELETE

---

### NEW REFERENCE D: Verify Zedda MP-IDB Study (currently wrongly cited as [25])

**CRITICAL ISSUE**: Line 243 states "Zedda et al. [25] earlier evaluated deep learning techniques on MP-IDB achieving 95.2% with YOLOv5 and 96.02% with DarkNet-53"

**Current [25]**: Prototypical Networks (Snell et al.) - WRONG!

**Problem**: Web search **DID NOT FIND** any "Zedda MP-IDB evaluation with YOLOv5/DarkNet-53" paper!

**Possible scenarios**:
1. Paper exists but different author (maybe Loddo 2022?)
2. Paper doesn't exist - claim is hallucination
3. Different Zedda paper not yet indexed

**ACTION REQUIRED**:
- **OPTION A**: If you can find the actual paper → add as new reference
- **OPTION B**: If paper doesn't exist → **DELETE Line 243 claim** about "Zedda et al. [25]"
- **OPTION C**: Verify if this is actually from Loddo 2022 J Imaging paper

**Recommendation**: **DELETE the claim** unless you can provide the actual paper

---

## PHASE 2: DELETE DUPLICATE REFERENCE

### DELETE REFERENCE [29] - Duplicate of [25]

**Current [25]**:
```
[25] J. Snell, K. Swersky, and R. Zemel, "Prototypical networks for few-shot learning,"
in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 4077-4087.
```

**Current [29]** (DUPLICATE):
```
[29] J. Snell, K. Swersky, and R. S. Zemel, "Prototypical networks for few-shot learning,"
in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017, pp. 4077-4087.
```

**ACTION**:
1. **DELETE [29] entirely**
2. **KEEP [25]** - update with more complete info from [29]:
```
[25] J. Snell, K. Swersky, and R. S. Zemel, "Prototypical networks for few-shot learning,"
in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017, pp. 4077-4087.
```

3. **UPDATE citations**: Change any [29] in text to [25]
   - Line 218: [29] → [25]
   - Line 257: [29] → [25]
   - Line 269: [29] → [25]

---

## PHASE 3: FIX TEXT CITATIONS

### FIX 1: Line 243 - Sukumarran YOLOv5 → YOLOv4

**Current text**:
```
Sukumarran et al. [26] proposed a two-stage approach combining YOLOv5 detection (96% mAP@0.5)
with DenseNet-121 classification
```

**Problem**: Reference [26] is about **YOLOv4**, not YOLOv5!

**Correct to**:
```
Sukumarran et al. [26] proposed a two-stage approach combining YOLOv4 detection
with DenseNet-121 classification
```

**Note**: Same line also mentions "YOLOv4 achieving 89-90%" - this is CORRECT, keep it

---

### FIX 2: Line 354 - MP-IDB Dataset Citation

**Current**: `MP-IDB: Available through Loddo et al. [15]`
**Current [15]**: Focal Loss paper (WRONG!)

**Correct to**: `MP-IDB: Available through Loddo et al. [NEW-A]`

Where [NEW-A] is the MP-IDB dataset reference added in Phase 1

---

### FIX 3: Line 82 - MP-IDB Introduction Citation

**Check if Line 82 also cites MP-IDB** - if yes, update to [NEW-A]

---

## PHASE 4: VERIFY NO BROKEN CITATIONS

After all changes, verify:
1. Every [X] in text has matching reference
2. No gaps in numbering (1, 2, 3... 30)
3. All new references have DOIs if available

---

## RECOMMENDED SEQUENCE OF FIXES

### Step 1: Handle the Zedda [25] Issue FIRST
**Decision needed**: Delete claim or find paper?

### Step 2: Add 3 New References
- MP-IDB dataset
- Zedda YOLO-PAM
- Loddo 2022 classification

### Step 3: Delete [29] Duplicate

### Step 4: Update ALL Text Citations
- Line 243: YOLOv5 → YOLOv4
- Line 243: Update Loddo, Zedda citations
- Line 354: Update MP-IDB citation
- Lines 218, 257, 269: [29] → [25]

### Step 5: Renumber if Needed
If you ADD references in middle, you may need to renumber all subsequent references

---

## ALTERNATIVE: MINIMAL FIX APPROACH

If full fix is too complex, here's MINIMAL changes to make paper submittable:

### MUST FIX (Non-negotiable):
1. ✅ Add MP-IDB dataset reference → Fix Line 354 citation
2. ✅ Add Zedda YOLO-PAM reference → Fix Line 243
3. ✅ Add Loddo 2022 classification → Fix Line 243
4. ✅ Delete [29] duplicate
5. ✅ Fix Sukumarran YOLOv5 → YOLOv4 (Line 243)

### CAN DEFER:
- Other minor citation mismatches
- Reference formatting consistency

---

## FINAL VERIFICATION CHECKLIST

After making ALL fixes:
- [ ] Reference [1-30] all exist
- [ ] No duplicate references
- [ ] All Line 243 citations match papers
- [ ] Line 354 MP-IDB citation correct
- [ ] No [29] citations in text
- [ ] Sukumarran text says "YOLOv4"
- [ ] All DOIs included where available

---

## QUESTION FOR YOU

**Before I proceed with automated fixes, please confirm**:

1. **Zedda [25] MP-IDB paper** - Should I DELETE the claim or do you have the actual paper?
2. **GANs paper [24]** - Keep for Line 253 synthetic data, or delete entirely?
3. **Buda Focal Loss [23]** - Keep as different number, or delete?

**Your decision will determine the exact sequence of fixes.**

Waiting for your guidance before making mass changes to avoid breaking the paper.
