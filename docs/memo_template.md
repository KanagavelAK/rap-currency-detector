# Memo template (hard limit: 2 pages)

Target ~900–1,000 words plus one small table and five thumbnails. Write in your own words.
Every number must come from `reports/` files. Delete the guidance in *italics* before exporting to PDF.

---

## Indian Banknote Detection & Reasoning: Design Memo
**Name** · **GitHub:** link · **Weights:** release link

### 1. Domain, data and labelling (~180 words)
*Why banknotes: a real task (assistive cash counting, counter automation), non-COCO classes, and overlapping notes suit RT-DETR's NMS-free design.*
*Public source: Mendeley dataset, 2,000 Indian images from one iPhone 8 in Pune. State what your inspection found:*
- *box convention (whole note or not), number of classes vs denominations, share of multi-note images*
- *label problems, images without labels*
- *how many validation images had near-duplicates in training*

*Own data: N photos, N sessions, phones used, what you deliberately captured, labelling rules you followed, and negatives.*
*Pivots: anything you tried first and why you changed (this counts as positive signal).*

### 2. Split strategy (~120 words)
*Perceptual-hash grouping (Hamming distance ≤ 8), so near-duplicates never cross splits. Stratified by dominant class, 75/10/15, seed 42.*
*Why not a random split: burst shots of the same note would inflate test scores.*
*Own photos split by session, never by image, and the test sessions stayed locked.*
*Give numbers: duplicates dropped, groups formed.*

### 3. Metrics and what they mean (~200 words + table)
*Table: Run A vs Run B × public test vs own photos (mAP@50, mAP@50:95, P, R, ₹ error, exact-total %).*

*What they tell you:*
- *the public→own gap = the domain shift the hidden set will expose*
- *the A→B change = whether your own data helped*

*What they don't tell you:*
- *mAP averages over thresholds; the API uses one threshold (chosen on validation)*
- *mAP punishes an old/new ₹100 mix-up as much as ₹10 vs ₹100, but only the second changes the money: give your same-value vs cross-value counts*
- *recall by object size*
- *small test sets mean wide uncertainty: say how many notes each number is based on*

### 4. Five failure cases (~300 words + thumbnails)
*Use images from `reports/*/worst/`. Five different failure types. For each one (~60 words):*
- *what happened (missed / wrong value / false positive / bad box)*
- *evidence: confidence, box size, IoU*
- *root cause*
- *proof it's systematic: a number across the test set, e.g. "recall on small notes 0.41 vs 0.93 on large"*
- *the fix you would try*

### 5. Part B reasoning layer (~180 words)
*Three routes and how the decision is made: rules first, then LLM JSON classification, then a validated default.*
*Why code computes facts and the LLM only phrases them, plus the number check and uncertainty check on LLM output.*
*Guardrails: confidence bands (thresholds from validation), overlap warning, blur/darkness check (thresholds from quality_stats).*
*Routing accuracy: on the question set, and SEPARATELY on questions written by someone else who never saw the rules.*

*One concrete "insufficient information" example from YOUR model: the image, the detections with confidences, and the exact JSON answer.*
*One ambiguous question and how the layer behaves (e.g. "Are there 10 notes?").*

### 6. Limitations and next steps (~60 words)
*Two or three honest sentences about scope and what you would do with another week.*
