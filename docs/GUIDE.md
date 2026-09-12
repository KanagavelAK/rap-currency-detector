# RAP Take-Home: Step-by-Step Guide (Indian Banknote Detector)

**Deadline: Sunday 13 September.** Aim to submit by **Sunday 2 PM** and keep the rest as buffer.
The code is already written and tested. Your job now is **data → training → evaluation → honest write-up → delivery**.

Golden rules:
1. **Never invent a number.** Every metric in the README and memo comes from a file in `reports/`.
2. **Commit to GitHub after every phase.** A steady commit history looks good and protects your work.
3. **Keep `docs/decisions_log.md` open all the time.** Write one line whenever you decide something. That becomes your memo.
4. **Understand every file you submit.** The verbal round will ask about specific lines (see Phase 12).

---

## Timeline at a glance

| When | Phase | Output |
|---|---|---|
| Sat 08:00–09:00 | 0. Accounts + repo | Code on GitHub, Kaggle verified |
| Sat 09:00–10:30 | 1. Data download + inspection | `reports/inspect`, class decisions |
| Sat 10:30–11:00 | 2. Smoke test, then start **Run A** in background | Run A training |
| Sat 11:00–14:30 | 3. Shoot + label own photos (while Run A trains) | `own_photos/` in 5 sessions |
| Sat 14:30–15:00 | 4. Start **Run B** + evaluation in background | Run B training |
| Sat 15:00–19:00 | 5. API running locally with Run A weights; Part B checks | Working `/detect` + `/ask` |
| Sat 19:00–23:00 | 6–7. Results + five failure cases | Filled results table, 5 cases |
| Sun 08:00–10:30 | 8. Memo | `docs/memo.pdf` |
| Sun 10:30–12:00 | 9. Weights release, README, samples | Release v1.0, no `<FILL>` left |
| Sun 12:00–13:00 | 10. Fresh-clone test (+ Docker if time) | Proof it reproduces |
| Sun 13:00–14:00 | 11. Final checklist → **submit** | Done |

**Cut rules if you fall behind (in this order):** skip Hugging Face deployment → skip Docker testing → cut epochs to 30
→ merge Run A/B into a single run on public+own data (still evaluate on your own test photos). **Never cut:**
own test photos, failure analysis, the memo, or the fresh-clone test.

---

## Phase 0: Setup (1 hour)

### 0.1 Accounts
- **GitHub**: make sure you can log in. Install **GitHub Desktop** if you've never pushed code before; it's the easiest way.
- **Kaggle** (free GPU): kaggle.com → Settings → **Phone verification**. Without it you get no GPU and no internet in notebooks.
- **CVAT** (labelling): create a free account at app.cvat.ai.
- **Groq** (free LLM API, optional but recommended): console.groq.com → API Keys → create a key. Check available model IDs:
  `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer YOUR_KEY"` (e.g. `openai/gpt-oss-20b`).

### 0.2 Local Python
Install Python 3.11 (python.org). Then, in the unzipped project folder:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # downloads PyTorch too; takes 10-20 min, so start it early
pytest -q                            # all tests should pass
```

### 0.3 Put the code on GitHub
1. github.com → **New repository** → name `rap-currency-detector` → **Public** → do NOT add a README (you already have one).
2. Push:
```bash
git init
git add .
git commit -m "Project structure: data pipeline, training, evaluation, API, reasoning layer, tests"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rap-currency-detector.git
git push -u origin main
```
(If authentication fails, open the folder in GitHub Desktop → Publish.)

✅ **Checkpoint:** the repo is visible on GitHub and `pytest -q` passes locally.

---

## Phase 1: Get and inspect the data (1.5 hours)

### 1.1 Download inside Kaggle (fastest, no big upload from home)
1. Open https://data.mendeley.com/datasets/2kfz5yc7pt/1, then right-click **Download All** → **Copy link address**.
2. Kaggle → **Create → New Notebook**. Right panel → **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
3. Run these cells:

```python
# Cell 1: setup
!nvidia-smi
!git clone https://github.com/YOUR_USERNAME/rap-currency-detector.git
%cd rap-currency-detector
!pip install -q -r requirements-train.txt
!pip freeze | grep -iE "^(ultralytics|torch|torchvision|numpy|pillow|pyyaml)==" > requirements-train.lock
!cat requirements-train.lock
```
```python
# Cell 2: download and unzip the dataset
!wget -q -O /kaggle/working/banknotes.zip "PASTE_THE_COPIED_LINK_HERE"
!unzip -q -o /kaggle/working/banknotes.zip -d /kaggle/working/raw
!find /kaggle/working/raw -name "*.zip"          # if this prints zips, unzip them too
!find /kaggle/working/raw -maxdepth 5 -type d -iname "*indian*"
```
If `wget` fails (the link needs a browser), download the zip in your browser instead. Then upload it via
Kaggle → Datasets → **New Dataset** (Private) and add it to the notebook with **Add Input**. It appears under `/kaggle/input/...`.

```python
# Cell 3: inspect (set PUBLIC to the IndianBankNotes folder printed above)
PUBLIC = "/kaggle/working/raw/PATH/TO/IndianBankNotes"
!python scripts/inspect_data.py --data-dir "{PUBLIC}" --out reports/inspect
```

### 1.2 Decide from what you see (write each answer in `docs/decisions_log.md`)
Open `reports/inspect/samples/` (right panel file browser → download a few):
- [ ] **Class names**: do they map to rupee values? If a class prints `NO RUPEE VALUE`, add it to `value_overrides` in `configs/inference.json`.
- [ ] **Separate old/new designs?** (e.g. two ₹100 classes.) Fine: the metrics treat them as same-value confusions.
- [ ] **Box convention**: whole note or only part? Your own labels must follow the same rule.
- [ ] **Front and back sides** both present?
- [ ] **Multi-note share**: how many images have 2+ notes? (Your own photos should add many more.)
- [ ] **Leakage**: "X of Y validation images have a near-duplicate in training". Write this number down; it justifies your split in the memo.
- [ ] Label problems / images without labels: note the counts.

⚠️ **Stop sign:** if boxes cover only numerals, or classes can't be mapped to values, rethink before training.
Fallback: a Roboflow Universe Indian currency dataset (check for duplicates with the same scripts).

✅ **Checkpoint:** you know the exact class names. Write them down; you need them for CVAT.

---

## Phase 2: Smoke test, then Run A in the background (30 min)

```python
# Cell 4: prepare public data (own photos come later; public splits stay identical)
!python scripts/prepare_data.py --public-dir "{PUBLIC}" --out datasets/currency --seed 42
```
```python
# Cell 5: smoke test (1 epoch, 10% data). Catches path and label bugs in minutes
!python scripts/train.py --data datasets/currency/data_public.yaml --name smoke --smoke --device 0
```
Check that there are no errors, the loss is not `nan`, and it prints `best_weights`. Note the time per epoch.
- `CUDA out of memory` → add `--batch 4`.
- `nan` losses → set `amp: false` in `configs/train.yaml`.
- If the smoke epoch × 60 > 3 hours → use `--epochs 30`.

**Start Run A as a background job** (it keeps running if you close the browser):
1. Add the final cells below. Delete the smoke-test cell, or leave it since it's quick.
```python
# Cell 6: Run A
!python scripts/train.py --data datasets/currency/data_public.yaml --name run_a --device 0
# Cell 7: choose threshold on validation, then evaluate on the public test split
!python scripts/evaluate.py --weights weights/run_a_best.pt --data datasets/currency/data_public.yaml --split val --sweep --out reports/run_a_val --device 0
!python scripts/evaluate.py --weights weights/run_a_best.pt --data datasets/currency/data_public.yaml --split test --conf-file reports/run_a_val/threshold.json --out reports/run_a_public_test --device 0
# Cell 8: keep outputs small (the dataset copies are not needed in the output)
!rm -rf datasets /kaggle/working/raw /kaggle/working/banknotes.zip
```
2. Click **Save Version → Save & Run All (Commit)**. When it finishes, the files are in the version's **Output** tab.

Commit locally: `git add configs docs && git commit -m "Data inspection decisions" && git push`.

---

## Phase 3: Your own photos (3.5 hours, while Run A trains)

This is your biggest differentiator. It's the closest thing you have to the reviewers' hidden test set.

### 3.1 Shoot about 150 photos in 5 sessions
A session = one place + one lighting + one phone + one time. **Never mix sessions**; this prevents near-duplicates leaking into your test set.

| Session | Setting | ~Photos |
|---|---|---|
| s1 | Desk, daylight, phone A | 30 |
| s2 | Bed/cloth background, tube light, phone A | 30 |
| s3 | Hand-held notes, outdoors or shop counter, phone B | 30 |
| s4 (**test**) | Different room, evening/dim light, phone B | 30 |
| s5 (**test**) | Mixed hard cases, phone A | 30 |

In every session include:
- single notes
- 2–6 notes spread apart
- fanned/overlapping stacks
- a folded note
- the back side
- far-away shots (notes small in frame)
- a mix of all denominations you have

In s4/s5 also add:
- 3 deliberately **blurry** photos
- 3 very **dark** photos
- 4 **negatives** with no notes: a bill/receipt, ID card, coins, a wallet

These last ones test your guardrails and give you the "insufficient information" example.
Borrow notes from family to cover denominations you don't have. Keep faces and personal documents out of frame.

### 3.2 Fix orientation BEFORE labelling (prevents misaligned boxes)
Copy the photos to your laptop into `raw_photos/s1 ... raw_photos/s5`, then:
```bash
for s in s1 s2 s3 s4 s5; do python scripts/fix_orientation.py --src raw_photos/$s --dst own_photos/$s; done
```
(Windows PowerShell: `foreach ($s in "s1","s2","s3","s4","s5") { python scripts/fix_orientation.py --src raw_photos/$s --dst own_photos/$s }`)

### 3.3 Label in CVAT
1. **Projects → +** → name `banknotes` → add labels with the **exact same names as the public class file**, typed carefully.
2. **Tasks → +** → one task per session (`s1` … `s5`) in that project → upload `own_photos/sN/*.jpg`.
3. Annotate with the **rectangle** tool. Use the same box convention as the public data.
   - Label a partly hidden note only if you can still tell its denomination.
   - Negatives get no boxes.
   - Write your rules in `decisions_log.md`.
4. For each task: **Actions → Export task dataset** → format **YOLO 1.1**, tick **Save images** → download.
5. Unzip each export. Put the contents of `obj_train_data/` (images + .txt files) into `own_photos/sN/`.
   Put `obj.names` into `own_photos/obj.names`. The same file comes from every task.

Labelling takes about 20 seconds per image. **Don't label s4/s5 more loosely**; test labels must be the most careful ones.

### 3.4 Upload photos to Kaggle
Zip the folder (`own_photos_v1.zip` containing `own_photos/`) → Kaggle → Datasets → **New Dataset** (Private).
Keep this zip: you'll also attach it to the GitHub release so reviewers can reproduce Run B.

---

## Phase 4: Run B + full evaluation in the background (30 min)

New Kaggle notebook (GPU T4, internet on) → **Add Input**:
- your `own_photos` dataset
- the Run A notebook's output (so Run A can also be evaluated on your photos)

```python
!git clone https://github.com/YOUR_USERNAME/rap-currency-detector.git
%cd rap-currency-detector
!pip install -q -r requirements-train.txt
!find /kaggle/input -maxdepth 4 -name "obj.names"; find /kaggle/input -name "run_a_best.pt"
```
```python
PUBLIC = "..."      # same dataset download cells as Phase 1, or add it as an input
OWN = "/kaggle/input/YOUR-OWN-PHOTOS/own_photos"
RUN_A = "/kaggle/input/PATH/TO/run_a_best.pt"
!python scripts/prepare_data.py --public-dir "{PUBLIC}" --out datasets/currency --own-dir "{OWN}" --own-classes "{OWN}/obj.names" --own-test-sessions s4,s5 --seed 42
!python scripts/train.py --data datasets/currency/data_plus_own.yaml --name run_b --device 0
```
```python
# Run B: threshold on val, then both test sets
!python scripts/evaluate.py --weights weights/run_b_best.pt --data datasets/currency/data_plus_own.yaml --split val --sweep --out reports/run_b_val --device 0
!python scripts/evaluate.py --weights weights/run_b_best.pt --data datasets/currency/data_public.yaml --split test --conf-file reports/run_b_val/threshold.json --out reports/run_b_public_test --device 0
!python scripts/evaluate.py --weights weights/run_b_best.pt --data datasets/currency/data_own_test.yaml --split test --conf-file reports/run_b_val/threshold.json --out reports/run_b_own_test --device 0
# Run A on your real-world photos (its threshold was chosen in the Run A notebook; if needed, re-run its val sweep here)
!python scripts/evaluate.py --weights "{RUN_A}" --data datasets/currency/data_public.yaml --split val --sweep --out reports/run_a_val --device 0
!python scripts/evaluate.py --weights "{RUN_A}" --data datasets/currency/data_own_test.yaml --split test --conf-file reports/run_a_val/threshold.json --out reports/run_a_own_test --device 0
!rm -rf datasets     # reports/data (split report + split lists) is kept
```
**Save Version → Save & Run All (Commit).**

When both runs finish, download from the Output tabs:
- `weights/`, `reports/`
- `runs/*/run_info.json`, `runs/*/results.csv`, `runs/*/*.png`
- `requirements-train.lock`

Copy `reports/` and the lock file into your local repo (**not** the `.pt` files) → commit.

---

## Phase 5: API running locally (4 hours)

1. Copy `run_b_best.pt` (or Run A's while B trains) to `weights/best.pt`.
2. Set thresholds: copy `conf_high` from `reports/run_b_val/threshold.json` into `configs/inference.json`.
   Set `conf_low` to about half of it (e.g. 0.5 → 0.25). Write the reason in the log.
3. Calibrate the image-quality guardrail on your photos:
   ```bash
   python scripts/quality_stats.py --images own_photos/s1       # percentiles for normal photos
   ```
   In step 5 you'll send your deliberately blurry and dark shots to `/ask`. If they don't get `blurry` / `too_dark` in
   `evidence.image_issues`, read their `evidence.image_quality` values and set `min_sharpness` / `min_brightness` in
   `configs/inference.json` between those values and the 5th percentile of normal photos. Restart the API after editing.
4. Run it:
   ```bash
   cp .env.example .env        # fill LLM_API_KEY; set WEIGHTS_URL later
   uvicorn app.main:app --port 8000 --env-file .env
   ```
5. Open http://localhost:8000/docs and try:
   - `/detect` with 3 photos
   - `/ask` with: a total question, a count, "is there a ₹500?", "is this fake?", "what can you do?"
   - `/ask` with a **blurry** photo, a **dark** photo, and a **negative** photo
6. Save real payloads for the README:
   ```bash
   cp own_photos/s5/<a good multi-note photo>.jpg samples/notes_example.jpg
   curl -s -X POST localhost:8000/detect -F "file=@samples/notes_example.jpg" > samples/detect_response.json
   curl -s -X POST localhost:8000/ask -F "question=How much money is in this photo?" -F "file=@samples/notes_example.jpg" > samples/ask_total.json
   curl -s -X POST localhost:8000/ask -F "question=How much money is here?" -F "file=@own_photos/s4/<dark photo>.jpg" > samples/ask_insufficient.json
   ```
   Delete the illustrative `ask_total_partial.json` once you have real ones.
7. Routing honesty check: ask a friend to write **15 questions** about a photo of money **without showing them the rules**.
   Add the expected route/intent to `tests/routing_unseen.json` (same format), then run
   `python scripts/eval_routing.py --questions tests/routing_unseen.json --out reports/routing_unseen.json`.
   Report this number in the memo as it is. You may fix rules afterwards, but report the "before" score.

Commit: `git add configs samples reports tests && git commit -m "Calibrated thresholds, real sample payloads, routing eval" && git push`.

---

## Phase 6: Results table (1 hour)

From each `reports/*/summary.json`, take:
- `official.mAP50`, `mAP50_95`, `precision_ultralytics`, `recall_ultralytics`
- `application.total_value_mae_rupees`, `exact_total_accuracy`

Fill the README table. Then answer in the log:
- [ ] **Domain shift:** public test vs own photos for Run A. How big is the gap?
- [ ] **Did own data help?** Run A vs Run B on own photos. If it didn't help, say so; that's still a result.
- [ ] Same-value vs cross-value confusions (`application.same_value_confusions` / `cross_value_confusions`, `top_confusions`).
- [ ] `recall_by_size`: are small notes the weak spot?
- [ ] How many notes each number is based on (`matched_notes + missed_notes`). Small test sets = uncertain numbers, and you should say so.

---

## Phase 7: Five failure cases (3 hours)

Open `reports/run_b_own_test/worst/` and `reports/run_b_public_test/worst/` (green = truth, red = prediction).
Pick **five different failure types**. Don't pick five of the same kind. For each, write:

1. **What happened**: missed note / wrong value / false positive / duplicate box / bad box.
2. **Evidence**: confidence, box size (`area` from label), which classes.
3. **Root cause**, backed by a check. Examples of checks:
   - small objects → `recall_by_size` small vs large
   - a confusion pair → count in `top_confusions`
   - a lighting issue → do the other dark photos fail too?
   - background false positives → how many negatives got a detection?
   - training data gap → how many training boxes of that situation exist (`split_report.json`)?
4. **Fix you'd try**: more data of that kind, a higher input size, a threshold change, an augmentation, a labelling rule.

"Systematic" means proven across images, not a guess from one image. Copy the 5 images into `docs/failures/`.

---

## Phase 8: Memo (2.5 hours, Sunday morning)

Use `docs/memo_template.md` (section order and word budgets are there). Write in Google Docs or Word.
- Maximum 2 pages. Use small thumbnails for the five failures.
- Must include:
  - dataset sourcing and labelling process, including pivots
  - split justification
  - metrics plus what they don't show
  - five failure cases with root cause
  - Part B routing logic
  - one concrete **insufficient information** example (use `samples/ask_insufficient.json`)
- Export to PDF → `docs/memo.pdf` → commit.

---

## Phase 9: Weights, README, final repo (1.5 hours)

### 9.1 Publish weights (GitHub Release, no extra account needed)
1. Rename your deployed weights to `best.pt`.
2. GitHub repo → **Releases → Draft a new release** → tag `v1.0`.
3. Attach `best.pt` and `own_photos_v1.zip` → **Publish**.
4. Test the link in a private browser window, or run:
   `curl -L -o test.pt https://github.com/YOUR_USERNAME/rap-currency-detector/releases/download/v1.0/best.pt`
5. Put that URL in `.env.example` and the README.

### 9.2 README
Replace **every** `<FILL>`. Check with `grep -rn "<FILL" README.md` (no output = done).
Fill in:
- hardware and training minutes (from `run_info.json`)
- chosen threshold
- test sessions
- number of own photos
- measured limitations

### 9.3 Clean-up
- Remove `docs/GUIDE.md` from the repo: it's your personal checklist, not reviewer documentation. AI tools are allowed in
  this assignment, so if you're asked how you worked, say so openly and show you understand every line.
- Make sure `.env` is **not** committed (`git status` shouldn't list it).
- Commit the lock files, `reports/`, `samples/` and `docs/memo.pdf`.

---

## Phase 10: Prove it reproduces (1 hour)

A reproducibility failure caps Part A at 50%, so this hour is worth a lot.
```bash
cd /tmp   # any empty folder
git clone https://github.com/YOUR_USERNAME/rap-currency-detector.git && cd rap-currency-detector
python -m venv .venv && source .venv/bin/activate && pip install -r requirements-api.txt
cp .env.example .env
uvicorn app.main:app --port 8000 --env-file .env      # must download weights by itself
curl -X POST localhost:8000/detect -F "file=@samples/notes_example.jpg"
```
Follow **your README only**. Every place you had to think or guess is a README bug, so fix it.

**Docker (bonus, only if time):**
`docker build -t currency-api . && docker run -p 8000:8000 --env-file .env currency-api` → test `/health`.

**Optional live deployment (only if everything else is done):** Hugging Face → New Space → SDK **Docker**. Push:
- `app/`, `src/`, `configs/`, `Dockerfile`, `requirements-api.txt`
- a Space `README.md` whose YAML header includes `sdk: docker` and `app_port: 8000`

Add `WEIGHTS_URL`, `LLM_MODEL` and `LLM_EXTRA_BODY` as variables, and `LLM_API_KEY` as a **secret**. Put the live URL in the README.

---

## Phase 11: Final checklist → submit

| Requirement | Where it's satisfied | ✓ |
|---|---|---|
| RT-DETR fine-tuned, ≥1 non-COCO class | `scripts/train.py`, `configs/train.yaml` | ☐ |
| `/detect` FastAPI endpoint | `app/main.py` | ☐ |
| Second endpoint with hand-written reasoning (no LangChain etc.) | `/ask`, `app/reasoning.py` | ☐ |
| Routes: detector / no detector / out of scope | `route_question()` | ☐ |
| Confidence guardrail + insufficient info | `build_facts()`, `compose()` | ☐ |
| GitHub repo, public, runs from README | fresh-clone test done | ☐ |
| Weights with working download path | Release v1.0 link tested | ☐ |
| Memo ≤ 2 pages with all required sections | `docs/memo.pdf` | ☐ |
| API docs with sample payloads | README + `samples/` real outputs | ☐ |
| Reproducible: steps, hardware, time, hyperparameters, seed | README §3, `run_info.json`, lock files, split lists | ☐ |
| Honest evaluation (what metrics don't show) | memo §3 | ☐ |
| 5 failure cases with systematic root cause | memo §4, `docs/failures/` | ☐ |
| Bonus: Docker, logging, (deploy) | `Dockerfile`, request-ID logs | ☐ |
| No `<FILL>` left, no `.env` committed | grep + git status | ☐ |

Submit the repo link, release link and memo exactly as the email asks. Then stop editing, because late commits can look bad.

---

## Phase 12: Prepare for the verbal round

Be able to explain, **in your own words**, without notes:
- **`src/phash.py`**: why the DCT/median fingerprint finds near-duplicates, and why the Hamming thresholds are 2 (drop) and 8 (group).
- **`split_groups()`**: why whole groups go into one split; why stratify by dominant class.
- **`configs/train.yaml`**: why `fliplr: 0`, small `hsv_h`, `degrees: 0`, `deterministic: false`.
- **`greedy_match()`**: why match by confidence first; what IoU 0.5 means.
- **Same-value vs cross-value**: why mAP can't tell them apart.
- **Threshold choice**: why tune on val not test; what F1-best means; why two thresholds (conf_high/conf_low).
- **`route_question()`**: why rules before the LLM; what happens with invalid LLM JSON; one question the router gets wrong.
- **`llm_text_is_safe()`**: the three checks and why the LLM never computes totals.
- **`Detector` lock**: why predictions are serialised across request threads.
- **`read_image()`**: why EXIF transpose matters for phone photos.
- **RT-DETR vs YOLO**: no NMS (a set of predictions via bipartite matching), why that suits overlapping notes, and the costs (memory, slower convergence).
- **Your numbers**: the domain-shift gap, and whether own data helped. Which failure worries you most for the hidden set, and why?

Practise: open any file at a random line and explain it out loud. If you can't, read it until you can.
