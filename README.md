# Indian Banknote Detection & Reasoning API

RT-DETR fine-tuned to detect Indian banknotes by denomination, served through FastAPI, with a hand-written
reasoning layer that answers questions like *"How much money is in this photo?"* and says so explicitly when
the evidence is not good enough.

> Replace every `<FILL>` with your real numbers before submitting. Never copy numbers from examples.

## Results

Confidence threshold `<FILL>`, chosen as the F1-best value on the validation split (never on test).

| Model | Test set | mAP@50 | mAP@50:95 | Precision | Recall | Total-value error (₹/image) | Exact total |
|---|---|---|---|---|---|---|---|
| Run A (public only) | Public test split | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| Run A (public only) | Own photos (real world) | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| Run B (public + own) | Public test split | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| Run B (public + own) | Own photos (real world) | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |

Deployed weights: **Run `<FILL>`**. Full reports are in `reports/`. Five analysed failure cases are in `docs/memo.pdf`.

## Repository layout

```
app/        FastAPI service: main.py (endpoints), detector.py, reasoning.py (Part B), llm.py, quality.py
src/        shared code: labels.py, phash.py (duplicate detection), metrics.py (application metrics)
scripts/    fix_orientation, inspect_data, prepare_data, train, evaluate, eval_routing, quality_stats
configs/    train.yaml (hyperparameters), inference.json (thresholds and guardrail limits)
tests/      unit + API tests, routing question set
reports/    inspection, split report and split lists, evaluation summaries, failure images
docs/       memo
samples/    example request/response payloads
```

## 1. Run the API

Requires Python 3.10+.

```bash
git clone https://github.com/<FILL_USERNAME>/rap-currency-detector.git
cd rap-currency-detector
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-api.txt
cp .env.example .env                                    # then set LLM_API_KEY (optional) and WEIGHTS_URL
uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

The weights download automatically on first start from `WEIGHTS_URL`:
`https://github.com/<FILL_USERNAME>/rap-currency-detector/releases/download/v1.0/best.pt`.
You can also download them manually into `weights/best.pt`. Open http://localhost:8000/docs for interactive docs.

Without `LLM_API_KEY`, the API still works and answers with deterministic templates (`"answer_source": "template"`).

### Docker

```bash
docker build -t currency-api .
docker run -p 8000:8000 --env-file .env currency-api
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

### `POST /detect`

Accepts a multipart image (JPEG/PNG/WebP/BMP, max 10 MB) and returns every note found with at least `conf_low` confidence.
`confident` marks detections at or above `conf_high`.

```bash
curl -X POST http://localhost:8000/detect -F "file=@samples/notes_example.jpg"
```

```json
{
  "image": {"filename": "notes_example.jpg", "width": 1280, "height": 960},
  "count": 2,
  "detections": [
    {"class_id": 5, "class_name": "500", "value": 500, "confidence": 0.9412,
     "box": {"x1": 112.4, "y1": 80.2, "x2": 598.7, "y2": 330.9}, "confident": true},
    {"class_id": 3, "class_name": "100", "value": 100, "confidence": 0.3127,
     "box": {"x1": 300.2, "y1": 410.0, "x2": 720.9, "y2": 650.4}, "confident": false}
  ],
  "thresholds": {"conf_high": 0.5, "conf_low": 0.25},
  "model": "best.pt",
  "inference_ms": 184.3
}
```

Errors: `400` for an invalid or empty image, `413` if too large, `422` if the file is missing, `503` if the model isn't loaded.

### `POST /ask`

Form fields: `question` (required) and `file` (the image; optional for questions that don't need it).

```bash
curl -X POST http://localhost:8000/ask -F "question=How much money is in this photo?" -F "file=@samples/notes_example.jpg"
```

See `samples/ask_total_partial.json`, `samples/ask_out_of_scope.json` and `samples/ask_general.json` for full responses. Key fields:

| Field | Meaning |
|---|---|
| `route` | `detector`, `general` (answered without the model) or `out_of_scope` (about the photo but not answerable by a note detector) |
| `intent` | e.g. `total_value`, `count_by_value`, `presence_value`, `authenticity` |
| `status` | `answered`, `partial` (some notes uncertain or other warnings) or `insufficient_information` |
| `answer_source` | `template`, `llm` or `template_fallback` (LLM text rejected by the safety check) |
| `evidence` | confident and uncertain notes, confirmed total, overlap and image-quality flags |

## 2. How Part B decides

1. **Routing**: regex rules first (tested on `tests/routing_questions.json`), then an LLM classifier that must return strict JSON, validated against allowed values. If both fail, a safe default applies.
2. **Facts in code**: detections are split into confident (`>= conf_high`) and uncertain (`conf_low` to `conf_high`). Counts and totals are computed in Python. The LLM never does arithmetic.
3. **Guardrails**:
   - "No notes" on a blurry or dark photo returns `insufficient_information`, not ₹0.
   - Uncertain notes make totals "at least ₹X".
   - Heavily overlapping boxes add a count warning.
   - Authenticity, coins, serial numbers, condition and foreign currency are refused without running the detector.
4. **Phrasing**: the LLM may rephrase the code-written draft. The reply is rejected if it adds or drops numbers or removes stated uncertainty.

## 3. Reproduce training

**Hardware used:** `<FILL e.g. Kaggle, 1x NVIDIA Tesla T4 16 GB>` · **Training time:** Run A `<FILL>` min, Run B `<FILL>` min
(from `runs/*/run_info.json`) · **Versions:** see `requirements-train.lock`.

### Data

1. **Public dataset.** Download *Dataset of Indian and Thai banknotes with annotations* (Meshram, Patil & Chumchu, Data in Brief 41, 2022, doi:10.1016/j.dib.2022.108007) from Mendeley Data: https://data.mendeley.com/datasets/2kfz5yc7pt/1. Only the `IndianBankNotes` folder is used.
2. **Own photos.** `<FILL>` images in `<FILL>` sessions, captured on `<FILL phones>`, labelled in CVAT. Download `own_photos_v1.zip` from the v1.0 release and unzip it to `own_photos/`. Sessions `<FILL e.g. s4,s5>` form the locked real-world test set.

### Steps

```bash
pip install -r requirements-train.txt
python scripts/inspect_data.py --data-dir <path>/IndianBankNotes --out reports/inspect
python scripts/prepare_data.py --public-dir <path>/IndianBankNotes --out datasets/currency \
    --own-dir own_photos --own-classes own_photos/obj.names --own-test-sessions <FILL> --seed 42
python scripts/train.py --data datasets/currency/data_public.yaml   --name run_a
python scripts/train.py --data datasets/currency/data_plus_own.yaml --name run_b
```

Hyperparameters are in `configs/train.yaml`, and every non-default value is commented with its reason.
The committed `reports/data/split_lists/*.txt` let you verify you get identical splits.

**Determinism note:** RT-DETR's deformable attention uses `grid_sample`, which has no deterministic CUDA backward pass.
The fixed seed makes data order, augmentation and initialisation repeatable, but metrics can still vary slightly between runs.
Expect differences of roughly `<FILL after comparing two runs, or state "not measured">`.

### Evaluate

```bash
python scripts/evaluate.py --weights weights/run_b_best.pt --data datasets/currency/data_plus_own.yaml --split val --sweep --out reports/run_b_val
python scripts/evaluate.py --weights weights/run_b_best.pt --data datasets/currency/data_public.yaml --split test --conf-file reports/run_b_val/threshold.json --out reports/run_b_public_test
python scripts/evaluate.py --weights weights/run_b_best.pt --data datasets/currency/data_own_test.yaml --split test --conf-file reports/run_b_val/threshold.json --out reports/run_b_own_test
python scripts/eval_routing.py
```

To evaluate on a new labelled folder (e.g. a hidden set), put images in `images/test` and YOLO labels in `labels/test`,
write a data YAML with the same class names, and run `evaluate.py --split test`.

## 4. Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Scope and limitations

- **Designed for:** phone photos of Indian rupee **banknotes** (₹10 to ₹2000), roughly 20 cm to 1 m away.
- **Not supported:** coins, other currencies, authenticity checks, or reading serial numbers.
- **Router ambiguity:** "Are there 10 notes?" is read as "is there a ₹10 note?".
- `<FILL: your measured weaknesses, e.g. heavily fanned stacks, back-side-only notes>`

## Data credit

Public images: V. Meshram, K. Patil, P. Chumchu, *Dataset of Indian and Thai banknotes with annotations*,
Data in Brief 41 (2022) 108007. The public images are not redistributed here; please download them from the source.
