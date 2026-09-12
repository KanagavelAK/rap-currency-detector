"""Part B: a single hand-written decision layer (no agent frameworks).

Pipeline for every question:
  1. ROUTE    rules first (fast, testable, explainable); LLM classifier only if no rule matches
               -> "general"       answer without the detector (question isn't about the photo)
               -> "out_of_scope"  about the photo, but a note detector can't know it -> say so, don't guess
               -> "detector"      run the model
  2. FACTS    code turns detections into verified facts: confident vs uncertain notes, counts, totals,
               overlap and image-quality warnings. The LLM never counts or adds.
  3. ANSWER   code writes a deterministic draft + status (answered / partial / insufficient_information)
  4. PHRASE   the LLM may rewrite the draft; we reject its text if it adds numbers or drops uncertainty,
               and fall back to the draft. If no LLM is configured, the draft is the answer.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from app import quality
from src.labels import VALID_DENOMINATIONS

GENERAL, OUT_OF_SCOPE, DETECTOR = "general", "out_of_scope", "detector"
ANSWERED, PARTIAL, INSUFFICIENT = "answered", "partial", "insufficient_information"

DETECTOR_INTENTS = {"total_value", "count_notes", "count_by_value", "presence_value", "presence_any",
                    "largest", "smallest", "most_common", "breakdown", "subjective", "unclear"}
GENERAL_INTENTS = {"capabilities", "model_info", "greeting", "general_knowledge", "unrelated", "empty"}

OUT_OF_SCOPE_RULES = [
    ("authenticity", r"\b(fake|counterfeit|genuine|forged|forgery|authentic|duplicate note)\b",
     "I can tell which denomination a note is, but not whether it is genuine. That needs security features "
     "like the watermark and security thread, which my model was never trained to check."),
    ("coins", r"\bcoins?\b",
     "My model only recognises banknotes, not coins, so I can't count or value coins."),
    ("serial_number", r"\bserial\b",
     "I don't read the text printed on notes, so I can't give serial numbers."),
    ("condition", r"\b(torn|damaged|dirty|soiled|mutilated|condition|crumpled)\b",
     "I wasn't trained to judge a note's physical condition, only its denomination."),
    ("foreign_currency", r"\b(dollars?|usd|euros?|pounds?|dirhams?|yen|yuan|baht|taka|ringgit|riyals?)\b",
     "I only recognise Indian rupee notes, so I can't answer about other currencies."),
    ("legal_status", r"\b(legal tender|demoneti[sz]ed|still valid|still accepted)\b",
     "Whether a note is still legal tender can't be verified from a photo."),
    ("ownership", r"\b(whose|who owns|who does .* belong)\b",
     "A photo of notes can't tell me who they belong to."),
]

GENERAL_RULES = [
    ("capabilities", r"\bwhat can you (do|detect|recogni[sz]e)\b|\bwhich (notes|denominations) (can|do) you\b"
                     r"|\bwhat (notes|denominations) (can|do) you (detect|recogni[sz]e|support)\b"
                     r"|\bhow (do|does) (you|this|it) work\b|^help\W*$"),
    ("model_info", r"\brt-?detr\b|\bwhat model\b|\bwhich model\b|\bhow (were|was) (you|it|the model) trained\b"),
    ("greeting", r"^(hi|hello|hey|thanks|thank you)\b"),
]

DEICTIC = r"\b(this|these|here|image|photo|picture|pic|see|i have|in my|shown)\b"
GENERAL_KNOWLEDGE = r"\b(india|indian|rbi|reserve bank|history|who (designed|prints|issues))\b"
IMAGE_WORDS = (r"\b(notes?|money|cash|rupees?|rs|inr|bills?|currency|banknotes?|denominations?|total|"
               r"how much|how many|image|photo|picture|here|this|these|see|count|worth|amount|value)\b")
UNCERTAIN_WORDS = (r"not (sure|confident|certain)|uncertain|unclear|can't|cannot|couldn't|unable|at least|"
                   r"possibl|might|may |maybe|insufficient|not enough|between|rule it out|depends")

_WORD_DENOMS = [("two thousand", 2000), ("five hundred", 500), ("two hundred", 200),
                ("hundred", 100), ("fifty", 50), ("twenty", 20), ("ten", 10)]


@dataclass
class Route:
    route: str
    intent: str
    denominations: list[int] = field(default_factory=list)
    claimed_total: int | None = None
    comparison: str | None = None  # "more" / "less" when the user asks "more than ₹X?"
    message: str = ""
    method: str = "rules"

    @property
    def denomination(self) -> int | None:
        return self.denominations[0] if self.denominations else None


@dataclass
class ReasoningConfig:
    conf_high: float = 0.5
    conf_low: float = 0.25
    overlap_iou_warning: float = 0.45
    min_sharpness: float = 60.0
    min_brightness: float = 40.0
    max_brightness: float = 225.0


class DetectorUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------- 1. routing

def normalise(question: str) -> str:
    t = re.sub(r"\brs\.", "rs ", question.lower().replace("₹", " rs "))
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,200 -> 1200
    t = re.sub(r"(\d)(rs|rupees?)\b", r"\1 \2", t)  # 500rs -> 500 rs
    t = re.sub(r"\b(rs|inr)(\d)", r"\1 \2", t)  # rs500 -> rs 500
    return re.sub(r"\s+", " ", t).strip()


def extract_numbers(t: str, known_values=VALID_DENOMINATIONS) -> tuple[list[int], list[int]]:
    """Return (denominations mentioned, other money amounts mentioned)."""
    denoms: list[int] = []
    others: list[int] = []
    for m in re.finditer(r"\d+", t):
        n = int(m.group())
        before, after = t[:m.start()], t[m.end():]
        money_marked = bool(re.search(r"\b(rs|inr|rupees?)\s*$", before)
                            or re.match(r"\s*(rs|rupees?|inr)\b(?!\s*\d)", after))
        if n in known_values:
            denoms.append(n)
        elif money_marked:
            others.append(n)
    remaining = t
    for phrase, value in _WORD_DENOMS:
        if re.search(rf"\b{phrase}\b", remaining):
            if value in known_values:
                denoms.append(value)
            remaining = re.sub(rf"\b{phrase}\b", " ", remaining)
    return denoms, others


def match_detector_intent(t: str, denoms: list[int], others: list[int]) -> tuple[str | None, int | None]:
    """Return (intent, claimed_total). Order matters: more specific patterns are checked first."""
    if re.search(r"\b(enough|rich|expensive|afford|a lot of money|good amount)\b", t):
        return "subjective", None
    if re.search(r"\b(largest|biggest|highest|maximum)\b", t):
        return "largest", None
    if re.search(r"\b(smallest|lowest|minimum)\b", t):
        return "smallest", None
    if re.search(r"\b(most common|most frequent|mostly|appears? (the )?most|most often)\b", t):
        return "most_common", None
    if re.search(r"\bhow many (rs|rupees?)\b(?!\s*\d)", t):  # "how many rupees" = total, "how many rs 500" = count
        return "total_value", (others or [None])[0]
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", t):
        return ("count_by_value" if denoms else "count_notes"), None
    if re.search(r"\bhow much\b|\btotal\b|\bsum\b|\bworth\b|\bamount\b|\badd up\b|\bvalue\b", t):
        if denoms and re.search(r"\b(notes?|bills?)\b", t):
            return "count_by_value", None
        claims = others or denoms
        return "total_value", (claims[0] if claims else None)
    if re.search(r"\b(which notes|what notes|what denominations|what (kind|kinds|type|types) of notes|list|"
                 r"breakdown|describe|what('s| is) (in|on)|what do you see)\b", t):
        return "breakdown", None
    if re.search(r"\b(is there|are there|any|do you see|can you see|contains?|does (it|this|the (image|photo|picture))"
                 r" (have|contain|show)|is (this|it) an?)\b", t):
        if len(set(denoms)) > 1:
            return "count_by_value", None  # "is there a 500 and a 100?" -> answer each denomination
        return ("presence_value" if denoms else "presence_any"), None
    return None, None


def route_question(question: str, llm=None, known_values=VALID_DENOMINATIONS) -> Route:
    t = normalise(question)
    if not t:
        return Route(GENERAL, "empty", message="Please ask a question about the notes in your photo.")

    for intent, pattern, message in OUT_OF_SCOPE_RULES:
        if re.search(pattern, t):
            return Route(OUT_OF_SCOPE, intent, message=message)

    for intent, pattern in GENERAL_RULES:
        if re.search(pattern, t):
            return Route(GENERAL, intent)

    if re.search(GENERAL_KNOWLEDGE, t) and not re.search(DEICTIC, t):
        return Route(GENERAL, "general_knowledge")

    if re.search(r"\d\s*[-+*/×]\s*\d", t):
        return Route(GENERAL, "unrelated")  # arithmetic like "500 + 200" isn't about the photo

    denoms, others = extract_numbers(t, known_values)
    intent, claimed = match_detector_intent(t, denoms, others)
    comparison = None
    cmp_match = re.search(r"\b(more than|over|above|at least|less than|under|below|at most)\b", t)
    if cmp_match and (others or denoms) and re.search(r"\b(rs|rupees?|inr|money|cash|total)\b", t) \
            and not re.search(r"\b(notes?|bills?)\b", t):
        intent, claimed = "total_value", (others or denoms)[0]
        comparison = "more" if cmp_match.group(1) in ("more than", "over", "above", "at least") else "less"

    if others and not denoms and intent != "total_value":
        allowed = ", ".join(f"₹{v}" for v in sorted(known_values))
        return Route(OUT_OF_SCOPE, "unknown_denomination",
                     message=f"₹{others[0]} isn't a note denomination I recognise. I know {allowed}.")

    if intent:
        unique = list(dict.fromkeys(denoms))
        return Route(DETECTOR, intent, denominations=[] if intent == "total_value" else unique,
                     claimed_total=claimed, comparison=comparison)

    if llm is not None and getattr(llm, "enabled", False):
        llm_route = route_with_llm(question, llm, known_values)
        if llm_route:
            return llm_route

    if re.search(IMAGE_WORDS, t) or denoms:
        return Route(DETECTOR, "unclear", denominations=list(dict.fromkeys(denoms)))
    return Route(GENERAL, "unrelated")


def route_with_llm(question: str, llm, known_values) -> Route | None:
    system = (
        "You classify questions sent to an API that detects Indian banknotes "
        f"({', '.join('Rs ' + str(v) for v in sorted(known_values))}) in a photo. "
        'Reply with ONLY a JSON object: {"route": "...", "intent": "...", "denomination": number or null}. '
        'route "detector": answerable by counting or locating notes and their values in the photo. '
        'route "general": not about the contents of the photo. '
        'route "out_of_scope": about the photo but needs something a note detector cannot know '
        "(authenticity, coins, condition, serial numbers, foreign currency, ownership). "
        f"Detector intents: {sorted(DETECTOR_INTENTS)}. General intents: ['general_knowledge', 'unrelated']. "
        "Out-of-scope intent: 'other'."
    )
    data = llm.chat_json(system, question)
    if not data:
        return None
    route, intent = data.get("route"), data.get("intent")
    denom = data.get("denomination")
    denom = int(denom) if isinstance(denom, (int, float)) and int(denom) in known_values else None
    if route == DETECTOR and intent in DETECTOR_INTENTS:
        if intent in ("count_by_value", "presence_value") and denom is None:
            intent = "count_notes" if intent == "count_by_value" else "presence_any"
        return Route(DETECTOR, intent, denominations=[denom] if denom else [], method="llm")
    if route == GENERAL and intent in {"general_knowledge", "unrelated"}:
        return Route(GENERAL, intent, method="llm")
    if route == OUT_OF_SCOPE:
        return Route(OUT_OF_SCOPE, "other", method="llm",
                     message="Answering that would need information my banknote detector can't provide.")
    return None  # invalid JSON content -> caller uses the rule-based fallback


# --------------------------------------------------------------------------- 2. facts

def _iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union > 0 else 0.0


def build_facts(detections: list[dict], image_metrics: dict, cfg: ReasoningConfig) -> dict:
    confident = [d for d in detections if d["confidence"] >= cfg.conf_high]
    uncertain = [d for d in detections if cfg.conf_low <= d["confidence"] < cfg.conf_high]
    counts = Counter(d["value"] for d in confident)
    max_overlap = max((_iou(a["box"], b["box"]) for i, a in enumerate(confident) for b in confident[i + 1:]),
                      default=0.0)
    return {
        "confident_notes": len(confident),
        "counts_by_value": {str(v): counts[v] for v in sorted(counts)},
        "confirmed_total": sum(d["value"] for d in confident),
        "uncertain_notes": len(uncertain),
        "uncertain_best_guesses": [{"value": d["value"], "confidence": d["confidence"]} for d in uncertain],
        "possible_extra_value": sum(d["value"] for d in uncertain),
        "overlapping_notes": max_overlap >= cfg.overlap_iou_warning,
        "image_quality": image_metrics,
        "image_issues": quality.issues(image_metrics, cfg.min_sharpness, cfg.min_brightness, cfg.max_brightness),
        "thresholds": {"conf_high": cfg.conf_high, "conf_low": cfg.conf_low},
    }


# --------------------------------------------------------------------------- 3. deterministic answers

def rupees(v: int) -> str:
    return f"₹{v:,}"


def notes(n: int) -> str:
    return f"{n} note" if n == 1 else f"{n} notes"


def breakdown(counts: dict[str, int]) -> str:
    parts = [f"{n} ₹{int(v):,} note{'s' if n != 1 else ''}"
             for v, n in sorted(counts.items(), key=lambda kv: int(kv[0]), reverse=True)]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def claim_sentence(route: Route, total: int, exact: bool) -> str:
    """Compare the confirmed total with the amount the user mentioned.

    exact=False means some notes were uncertain, so the true total is only known to be >= total.
    """
    c = rupees(route.claimed_total)
    if route.comparison == "more":
        if total > route.claimed_total:
            return f"So yes, that's more than {c}."
        return f"So no, that's not more than {c}." if exact else f"So I can't confirm whether it's more than {c}."
    if route.comparison == "less":
        if total >= route.claimed_total:
            return f"So no, it's not less than {c}."
        return f"So yes, that's less than {c}." if exact else f"So I can't confirm whether it's less than {c}."
    if exact:
        return f"So yes, it matches {c}." if total == route.claimed_total else f"So no, it isn't {c}."
    if total > route.claimed_total:
        return f"So it's more than {c}."
    return f"So I can't confirm whether it's {c}."


ISSUE_WORDS = {"blurry": "blurry", "too_dark": "too dark", "overexposed": "overexposed"}


def compose(route: Route, facts: dict) -> tuple[str, str]:
    """Return (status, draft answer). Every number comes from `facts`."""
    n, u = facts["confident_notes"], facts["uncertain_notes"]
    counts, total = facts["counts_by_value"], facts["confirmed_total"]
    issues = " and ".join(ISSUE_WORDS[i] for i in facts["image_issues"])
    guesses = [g["value"] for g in facts["uncertain_best_guesses"]]
    intent, v = route.intent, route.denomination

    # Guardrail A: nothing usable was found.
    if n == 0 and u == 0 and issues:
        return INSUFFICIENT, (f"I can't see any notes, but the photo is {issues}, so I can't tell whether there is "
                              "money in it. Please retake it with better light and a steady hand.")
    if n == 0 and u > 0:
        if intent == "presence_value" and v in guesses:
            return INSUFFICIENT, f"There might be a ₹{v:,} note, but I'm not confident enough to say for sure."
        return INSUFFICIENT, (f"I might be seeing {notes(u)}, but I'm not confident enough to identify "
                              f"{'it' if u == 1 else 'them'}. A closer, clearer photo would help.")

    status, text = ANSWERED, ""
    if intent == "total_value":
        if n == 0:
            text = "I don't see any Indian banknotes in this photo, so the total is ₹0."
        elif u:
            status = PARTIAL
            text = (f"I'm confident about {breakdown(counts)}, which adds up to {rupees(total)}. "
                    f"I also found {notes(u)} I couldn't identify reliably, so the real total is at least "
                    f"{rupees(total)}.")
        else:
            text = f"I can see {breakdown(counts)}, which adds up to {rupees(total)}."
        if route.claimed_total is not None:
            text += " " + claim_sentence(route, total, status == ANSWERED)

    elif intent == "count_notes":
        if n == 0:
            text = "I don't see any Indian banknotes in this photo."
        elif u:
            status = PARTIAL
            text = (f"I can confidently count {notes(n)} ({breakdown(counts)}), plus {u} unclear, "
                    f"so there are between {n} and {n + u}.")
        else:
            text = f"I can see {notes(n)}: {breakdown(counts)}."

    elif intent == "count_by_value" and route.denominations:
        sentences, subtotal = [], 0
        for d in route.denominations:
            c = counts.get(str(d), 0)
            subtotal += c * d
            maybe_same = guesses.count(d)
            if c == 0 and maybe_same == 0:
                sentences.append(f"I don't see any ₹{d:,} notes.")
            else:
                sentences.append(f"I can see {c} ₹{d:,} note{'s' if c != 1 else ''}, worth {rupees(c * d)}.")
                if maybe_same:
                    status = PARTIAL
                    sentences.append(f"There may be {maybe_same} more ₹{d:,} that I'm not confident about.")
        if len(route.denominations) > 1:
            sentences.append(f"Together that's {rupees(subtotal)}.")
        maybe_other = sum(1 for g in guesses if g not in route.denominations)
        if maybe_other:
            status = PARTIAL
            sentences.append(f"{notes(maybe_other).capitalize()} couldn't be identified, so I can't fully rule "
                             "out more.")
        text = " ".join(sentences)

    elif intent == "presence_value" and v:
        c = counts.get(str(v), 0)
        if c:
            text = f"Yes, I can see {c} ₹{v:,} note{'s' if c != 1 else ''}."
        elif v in guesses:
            status, text = INSUFFICIENT, f"There might be a ₹{v:,} note, but I'm not confident enough to say for sure."
        elif u:
            status = PARTIAL
            text = f"I don't see a ₹{v:,} note I'm confident about, but {u} unclear note{'s' if u != 1 else ''} could be one."
        elif issues:
            status = INSUFFICIENT
            text = f"I don't see a ₹{v:,} note, but the photo is {issues}, so I can't rule it out."
        else:
            text = f"No, I don't see any ₹{v:,} note."

    elif intent == "presence_any":
        text = (f"Yes, I can see {notes(n)}: {breakdown(counts)}." if n
                else "No, I don't see any Indian banknotes in this photo.")

    elif intent in ("largest", "smallest"):
        if n == 0:
            text = "I don't see any notes to compare."
        else:
            values = [int(k) for k in counts]
            pick = max(values) if intent == "largest" else min(values)
            text = f"The {intent} note I can see is ₹{pick:,} ({notes(counts[str(pick)])})."
            beyond = [g for g in guesses if (g > pick if intent == "largest" else g < pick)]
            if beyond:
                status = PARTIAL
                text += f" An unclear note might be ₹{beyond[0]:,}, which would change this."

    elif intent == "most_common":
        if n == 0:
            text = "I don't see any notes."
        else:
            top = max(counts.values())
            winners = [int(k) for k, c in counts.items() if c == top]
            if len(winners) == 1:
                text = f"The most common note is ₹{winners[0]:,} ({notes(top)})."
            else:
                text = (f"There's a tie: {' and '.join(f'₹{w:,}' for w in sorted(winners))} each appear "
                        f"{top} time{'s' if top != 1 else ''}.")
            if u:
                status = PARTIAL
                text += f" {u} unclear note{'s' if u != 1 else ''} could change this."

    else:  # breakdown, subjective, unclear
        seen = (f"I can see {breakdown(counts)}, totalling {rupees(total)}." if n
                else "I don't see any Indian banknotes in this photo.")
        if intent == "subjective":
            status = PARTIAL
            text = seen + " Whether that's enough depends on what you need it for, which I can't judge from a photo."
        elif intent == "unclear":
            status = PARTIAL
            text = "I'm not sure exactly what you're asking, so here is what I can see: " + seen
        else:
            text = seen
        if u:
            status = PARTIAL
            text += f" {notes(u).capitalize()} couldn't be identified reliably."

    # Guardrail B: warnings that weaken an otherwise confident answer.
    if n and facts["overlapping_notes"]:
        status = PARTIAL if status == ANSWERED else status
        text += " Some notes overlap heavily, so the count may be off."
    if n and issues:
        status = PARTIAL if status == ANSWERED else status
        text += f" The photo is {issues}, which may hide some notes."
    return status, text


GENERAL_TEMPLATES = {
    "capabilities": ("I detect Indian banknotes (₹10 to ₹2,000) in a photo and answer questions about them, "
                     "like how many notes there are or their total value. I can't check whether notes are "
                     "genuine, count coins or read serial numbers."),
    "model_info": ("I use an RT-DETR object detector fine-tuned on photos of Indian banknotes. Counts and totals "
                   "are calculated in code from its detections, not guessed by a language model."),
    "greeting": "Hi! Send a photo of Indian banknotes and ask something like 'How much money is here?'",
    "general_knowledge": ("That's a general question rather than one about your photo, so I didn't run the "
                          "detector. I can only answer questions about the banknotes in an image."),
    "unrelated": ("That question isn't about the notes in your photo, so I didn't run the detector. "
                  "I can answer questions like 'How many notes are there?' or 'What's the total?'"),
    "empty": "Please ask a question about the notes in your photo.",
}


# --------------------------------------------------------------------------- 4. phrasing + orchestration

def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def llm_text_is_safe(llm_text: str, draft: str, facts: dict, status: str) -> bool:
    if not llm_text or len(llm_text) > 600:
        return False
    allowed = numbers_in(draft) | numbers_in(json.dumps(facts))
    if not numbers_in(llm_text) <= allowed:
        return False  # the LLM invented a number
    if not numbers_in(draft) <= numbers_in(llm_text):
        return False  # the LLM dropped a number the answer depends on
    if status != ANSWERED and not re.search(UNCERTAIN_WORDS, llm_text.lower()):
        return False  # the LLM removed the uncertainty
    return True


class ReasoningEngine:
    def __init__(self, detector, llm, cfg: ReasoningConfig):
        self.detector = detector
        self.llm = llm
        self.cfg = cfg

    @property
    def known_values(self) -> tuple[int, ...]:
        if self.detector is not None:
            return tuple(sorted({v for v in self.detector.values.values() if v}))
        return VALID_DENOMINATIONS

    def ask(self, question: str, image=None) -> dict:
        start = time.perf_counter()
        route = route_question(question, self.llm, self.known_values)
        out = {"question": question, "route": route.route, "intent": route.intent,
               "routing_method": route.method, "detector_called": False}

        if route.route == OUT_OF_SCOPE:
            out.update(status=INSUFFICIENT, answer=route.message, answer_source="template")
        elif route.route == GENERAL:
            out.update(status=ANSWERED, **self._general(route, question))
        elif image is None:
            out.update(status=INSUFFICIENT, answer_source="template",
                       answer="This question is about banknotes in a photo, but no image was attached.")
        else:
            if self.detector is None:
                raise DetectorUnavailable("Model is not loaded")
            detections, inference_ms = self.detector.predict(image, self.cfg.conf_low)
            facts = build_facts(detections, quality.measure(image), self.cfg)
            status, draft = compose(route, facts)
            answer, source = self._phrase(question, facts, draft, status)
            out.update(status=status, answer=answer, answer_source=source, detector_called=True,
                       evidence=facts, detections=detections, inference_ms=inference_ms)
            if route.denominations:
                out["denominations_asked"] = route.denominations

        out["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return out

    def _general(self, route: Route, question: str) -> dict:
        if route.intent in ("general_knowledge", "unrelated") and self.llm is not None and self.llm.enabled:
            reply = self.llm.chat(
                "You are the assistant inside an Indian banknote detection API. Answer the user's general "
                "question briefly (at most 2 sentences). Do not describe any photo; you cannot see it.",
                question, max_tokens=800)
            if reply and len(reply) <= 800:
                return {"answer": reply, "answer_source": "llm"}
        return {"answer": GENERAL_TEMPLATES[route.intent], "answer_source": "template"}

    def _phrase(self, question: str, facts: dict, draft: str, status: str) -> tuple[str, str]:
        if self.llm is None or not self.llm.enabled:
            return draft, "template"
        system = ("You rewrite answers for a banknote-counting assistant. Rewrite DRAFT as a short, friendly "
                  "answer to QUESTION. Rules: use only information in DRAFT; keep every number exactly as in "
                  "DRAFT and write numbers as digits; never add notes, amounts or guesses; if DRAFT expresses "
                  "uncertainty, keep it clearly. Output only the answer.")
        user = f"QUESTION: {question}\nDRAFT: {draft}\nFACTS (for context only): {json.dumps(facts)}"
        reply = self.llm.chat(system, user)
        if reply and llm_text_is_safe(reply, draft, facts, status):
            return reply, "llm"
        return draft, "template_fallback" if reply else "template"
