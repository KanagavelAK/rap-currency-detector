"""Measure how accurately the Part B router classifies questions.

Usage:
    python scripts/eval_routing.py                     # rules only (deterministic)
    python scripts/eval_routing.py --use-llm           # rules + LLM fallback (needs LLM_* env vars)

Add your own tricky questions to tests/routing_questions.json. Report the accuracy in the memo,
and be honest about the questions it gets wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.llm import LLMClient  # noqa: E402
from app.reasoning import route_question  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=Path("tests/routing_questions.json"))
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/routing_eval.json"))
    args = parser.parse_args()

    llm = None
    if args.use_llm:
        llm = LLMClient(os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                        os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL"),
                        extra_body=json.loads(os.getenv("LLM_EXTRA_BODY") or "{}"))
        if not llm.enabled:
            sys.exit("Set LLM_API_KEY and LLM_MODEL to use --use-llm")

    cases = json.loads(args.questions.read_text(encoding="utf-8"))
    route_ok = full_ok = 0
    mistakes = []
    for c in cases:
        r = route_question(c["question"], llm)
        route_ok += r.route == c["route"]
        full_ok += (r.route, r.intent) == (c["route"], c["intent"])
        if (r.route, r.intent) != (c["route"], c["intent"]):
            mistakes.append({"question": c["question"], "expected": [c["route"], c["intent"]],
                             "got": [r.route, r.intent], "method": r.method})

    result = {"questions": len(cases), "route_accuracy": round(route_ok / len(cases), 3),
              "route_and_intent_accuracy": round(full_ok / len(cases), 3), "mistakes": mistakes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
