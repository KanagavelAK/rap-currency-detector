"""Tests for the Part B reasoning layer. Run: pytest -q"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.reasoning import (  # noqa: E402
    ANSWERED, INSUFFICIENT, PARTIAL, ReasoningConfig, ReasoningEngine, Route, build_facts, compose,
    llm_text_is_safe, route_question,
)

CFG = ReasoningConfig(conf_high=0.5, conf_low=0.25)
SHARP_METRICS = {"sharpness": 500.0, "brightness": 120.0}
BLURRY_METRICS = {"sharpness": 5.0, "brightness": 120.0}


def det(value, conf, x1=0, y1=0, x2=100, y2=50):
    return {"class_id": 0, "class_name": str(value), "value": value, "confidence": conf,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def sharp_image():
    img = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(img)
    for x in range(0, 640, 20):  # strong edges -> high sharpness
        draw.line([(x, 0), (x, 480)], fill="black", width=3)
    return img


class FakeDetector:
    values = {0: 10, 1: 100, 2: 500}

    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def predict(self, img, conf_floor):
        self.calls += 1
        return [d for d in self.detections if d["confidence"] >= conf_floor], 12.0


class FakeLLM:
    enabled = True

    def __init__(self, reply=None, json_reply=None):
        self.reply = reply
        self.json_reply = json_reply

    def chat(self, system, user, max_tokens=400):
        return self.reply

    def chat_json(self, system, user):
        return self.json_reply


def test_routing_suite():
    cases = json.loads((Path(__file__).parent / "routing_questions.json").read_text(encoding="utf-8"))
    failures = []
    for c in cases:
        r = route_question(c["question"])
        if (r.route, r.intent) != (c["route"], c["intent"]):
            failures.append(f"{c['question']!r}: got {(r.route, r.intent)}, expected {(c['route'], c['intent'])}")
    assert not failures, "\n".join(failures)


def test_denomination_and_claim_extraction():
    assert route_question("How many ₹500 notes?").denomination == 500
    assert route_question("is the total rs.1,200").claimed_total == 1200
    assert route_question("Are there 2 ₹500 notes?").denomination == 500


def test_comparison_and_multi_denomination():
    r = route_question("Do I have more than ₹1000?")
    assert (r.intent, r.claimed_total, r.comparison) == ("total_value", 1000, "more")
    facts = build_facts([det(500, 0.9), det(500, 0.9, x1=300, x2=400), det(100, 0.9, x1=500, x2=600)],
                        SHARP_METRICS, CFG)
    _, text = compose(r, facts)
    assert "yes, that's more than ₹1,000" in text
    r2 = route_question("What's the sum of the 2000 and 500 notes?")
    assert r2.denominations == [2000, 500]
    _, text2 = compose(r2, facts)
    assert "Together that's ₹1,000" in text2


def test_confident_total():
    facts = build_facts([det(500, 0.9), det(100, 0.8, x1=300, x2=400)], SHARP_METRICS, CFG)
    status, text = compose(Route("detector", "total_value"), facts)
    assert status == ANSWERED
    assert "₹600" in text


def test_uncertain_note_gives_partial_total():
    facts = build_facts([det(500, 0.9), det(100, 0.3, x1=300, x2=400)], SHARP_METRICS, CFG)
    status, text = compose(Route("detector", "total_value"), facts)
    assert status == PARTIAL
    assert "at least ₹500" in text


def test_only_uncertain_notes_is_insufficient():
    facts = build_facts([det(500, 0.3)], SHARP_METRICS, CFG)
    status, _ = compose(Route("detector", "count_notes"), facts)
    assert status == INSUFFICIENT


def test_blurry_empty_photo_is_insufficient_not_zero():
    facts = build_facts([], BLURRY_METRICS, CFG)
    status, text = compose(Route("detector", "total_value"), facts)
    assert status == INSUFFICIENT
    assert "blurry" in text


def test_sharp_empty_photo_is_zero():
    facts = build_facts([], SHARP_METRICS, CFG)
    status, text = compose(Route("detector", "total_value"), facts)
    assert status == ANSWERED and "₹0" in text


def test_overlap_warning_makes_answer_partial():
    facts = build_facts([det(500, 0.9), det(500, 0.9, x1=10, x2=110)], SHARP_METRICS, CFG)
    status, text = compose(Route("detector", "count_notes"), facts)
    assert status == PARTIAL and "overlap" in text


def test_presence_of_uncertain_denomination_is_insufficient():
    facts = build_facts([det(100, 0.9), det(500, 0.3, x1=300, x2=400)], SHARP_METRICS, CFG)
    status, _ = compose(Route("detector", "presence_value", denominations=[500]), facts)
    assert status == INSUFFICIENT


def test_most_common_tie():
    facts = build_facts([det(500, 0.9), det(100, 0.9, x1=300, x2=400)], SHARP_METRICS, CFG)
    _, text = compose(Route("detector", "most_common"), facts)
    assert "tie" in text


def test_llm_output_with_invented_number_is_rejected():
    facts = build_facts([det(500, 0.9)], SHARP_METRICS, CFG)
    draft = "I can see 1 ₹500 note, which adds up to ₹500."
    assert llm_text_is_safe("You have one ₹500 note, ₹500 in total.", draft, facts, ANSWERED) is False  # dropped "1"
    assert llm_text_is_safe("You have 1 ₹500 note, so ₹700 in total.", draft, facts, ANSWERED) is False
    assert llm_text_is_safe("You've got 1 ₹500 note, so ₹500 total.", draft, facts, ANSWERED) is True


def test_llm_output_that_drops_uncertainty_is_rejected():
    facts = build_facts([det(500, 0.9), det(100, 0.3, x1=300, x2=400)], SHARP_METRICS, CFG)
    status, draft = compose(Route("detector", "total_value"), facts)
    assert llm_text_is_safe("There is 1 ₹500 note, total ₹500.", draft, facts, status) is False


def test_engine_out_of_scope_never_calls_detector():
    detector = FakeDetector([det(500, 0.9)])
    engine = ReasoningEngine(detector, None, CFG)
    out = engine.ask("Is this note fake?", sharp_image())
    assert out["status"] == INSUFFICIENT and detector.calls == 0


def test_engine_detector_route_uses_template_without_llm():
    detector = FakeDetector([det(500, 0.9), det(100, 0.8, x1=300, x2=400)])
    engine = ReasoningEngine(detector, None, CFG)
    out = engine.ask("How much money is here?", sharp_image())
    assert out["detector_called"] and out["answer_source"] == "template"
    assert out["evidence"]["confirmed_total"] == 600


def test_engine_falls_back_when_llm_hallucinates():
    detector = FakeDetector([det(500, 0.9)])
    engine = ReasoningEngine(detector, FakeLLM(reply="You have ₹5,000!"), CFG)
    out = engine.ask("What's the total?", sharp_image())
    assert out["answer_source"] == "template_fallback"
    assert "₹500" in out["answer"]


def test_engine_accepts_safe_llm_phrasing():
    detector = FakeDetector([det(500, 0.9)])
    engine = ReasoningEngine(detector, FakeLLM(reply="Looks like 1 ₹500 note, so ₹500 altogether."), CFG)
    out = engine.ask("What's the total?", sharp_image())
    assert out["answer_source"] == "llm"


def test_engine_no_image_for_detector_question():
    engine = ReasoningEngine(FakeDetector([]), None, CFG)
    out = engine.ask("How many notes?", None)
    assert out["status"] == INSUFFICIENT and not out["detector_called"]


def test_llm_router_invalid_json_falls_back_to_rules():
    llm = FakeLLM(json_reply={"route": "banana", "intent": "???"})
    r = route_question("Tell me about the notes", llm)
    assert (r.route, r.intent, r.method) == ("detector", "unclear", "rules")


def test_llm_router_valid_json_is_used():
    llm = FakeLLM(json_reply={"route": "detector", "intent": "breakdown", "denomination": None})
    r = route_question("Walk me through what's lying on the table", llm)
    assert (r.route, r.intent, r.method) == ("detector", "breakdown", "llm")
