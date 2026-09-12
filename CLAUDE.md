# Project context (read this first)

## What this is
Take-home assignment for an ML intern role at Rapid Acceleration Partners (RAP).
**Deadline: Sunday 13 September.** Aim to submit Sunday afternoon.

Three required parts:
- **A:** RT-DETR fine-tuned on a non-COCO dataset → Indian banknote detection by denomination
- **API:** FastAPI `/detect` endpoint
- **B:** a second endpoint (`/ask`) with a **hand-written** reasoning layer.
  **LangChain, LangGraph, CrewAI and AutoGen are strictly banned.**

Deliverables: public GitHub repo, downloadable weights, a memo of **at most 2 pages**, API docs with sample payloads.

Grading: hidden test set 25%, dataset sourcing 15%, failure-case analysis 15%, Part B 15%,
evaluation honesty 10%, code/reproducibility 10%, bonus (Docker/logging/deploy) 10%.
**A reproducibility failure caps Part A at 50%.** There is a verbal round where I must explain specific lines of code,
so prefer simple, explainable code over clever code, and explain anything non-obvious in a comment.

## Repo
https://github.com/KanagavelAK/rap-currency-detector — local path `D:\rap-currency-detector`.
Full step-by-step plan is in `docs/GUIDE.md` (12 phases). Running decisions log: `docs/decisions_log.md`.

## State as of the handoff
- **Phase 0 done.** Code written, 37 tests pass locally (`pytest -q`), pushed to GitHub.
  A Windows encoding bug was fixed: JSON files must be read with `encoding="utf-8"`.
- **Phase 1 in progress.** Dataset selection.

### Dataset decision (important)
**Rejected:** the Mendeley "Indian and Thai banknotes" dataset (doi:10.1016/j.dib.2022.108007).
Inspection showed its boxes annotate **features inside a single note** (numeral, Gandhi portrait, RBI seal,
Ashoka pillar), class ids are invalid (`-1` and `2`), and the denomination is encoded only in the folder name.
It is built for single-note classification, not multi-note detection, so it cannot support counting or summing.
*This pivot belongs in the memo as a dataset-sourcing finding; the drawn-boxes image is the evidence.*

**Chosen:** Roboflow Universe — "indian currency/notes" by omkar patkar, CC BY 4.0, 2,163 images
(1740 train / 280 valid / 143 test), whole-note boxes, class names are denominations. Exported as YOLOv8.

Two things to verify on this dataset:
1. Annotations are rotated polygons in Roboflow, so the YOLO export gives axis-aligned boxes that include
   background on tilted notes. Acceptable, but note it as a limitation.
2. Roboflow exports often contain augmented near-duplicate copies **across splits**. Ignore Roboflow's split
   entirely: pool all three folders and re-split with `scripts/prepare_data.py`, which drops near-identical
   images and keeps near-duplicate groups inside one split.

## Codebase map
- `src/labels.py` — YOLO label IO, class-name discovery, `value_from_class_name()` → rupee value
- `src/phash.py` — perceptual hash (DCT + median), near-duplicate pairs, union-find grouping
- `src/metrics.py` — greedy IoU matching, rupee total error, **same-value vs cross-value** confusion, recall by size
- `scripts/` — `fix_orientation`, `inspect_data`, `prepare_data`, `train`, `evaluate`, `eval_routing`, `quality_stats`
- `app/main.py` — FastAPI: `/health`, `/detect`, `/ask`; request-id logging; weights auto-download from `WEIGHTS_URL`
- `app/reasoning.py` — **Part B**: rules-first routing (3 routes: detector / general / out_of_scope), then an
  LLM JSON classifier fallback; facts and arithmetic computed in code; the LLM only rephrases a deterministic
  draft and its output is rejected if it invents or drops numbers or removes uncertainty
- `configs/train.yaml`, `configs/inference.json` — hyperparameters and thresholds, each with a reason in a comment
- `tests/` — 37 tests, including a 38-question routing suite

## Next steps (see docs/GUIDE.md for detail)
1. Upload the Roboflow dataset to Kaggle, run `scripts/inspect_data.py`, confirm class names and box areas.
2. Run `scripts/prepare_data.py` (pool all Roboflow splits, re-split, seed 42).
3. Smoke test (`train.py --smoke`), then **Run A** on public data only.
4. Shoot ~150 own phone photos in 5 sessions, run `fix_orientation.py` **before** labelling in CVAT,
   then **Run B** on public + own training sessions. Sessions s4/s5 are a locked real-world test set.
5. Sweep the confidence threshold on **validation only**, then evaluate both runs on both test sets.
6. Pick 5 genuinely different failure cases from `reports/*/worst/` and prove each is systematic with a number.
7. Write the memo (`docs/memo_template.md`), fill every `<FILL>` in the README, publish weights as a
   GitHub Release asset, then verify by cloning fresh and following the README exactly.

## Working preferences
- Keep answers short and direct; no long preambles.
- Never invent numbers: every metric in the README or memo must come from a file in `reports/`.
- `docs/GUIDE.md` is my personal checklist — remove it from the repo before submitting.
