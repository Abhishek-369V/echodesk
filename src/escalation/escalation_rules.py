"""
Hybrid escalation logic: rule-based first pass (fast, free, deterministic),
LLM fallback only when rules find nothing (catches nuance keyword-matching misses — sarcasm, 
indirect repeat-signals, implied frustration).

Why hybrid, not pure rule-based or pure LLM:
  - Rule-based alone measured at 68% recall / 20% false-positive rate on our 176 real labeled examples 
    (see decision log) — misses 1/3 of true escalations. Not good enough to "trust" on its own.
  - Pure LLM-only would cost money on every single message and be harder to defend as deterministic/explainable.
  - Hybrid: rules handle the cheap, obvious cases instantly and for free.
    LLM is only called when rules stay silent, and is FORCED to return a structured reason — so every decision, 
    rule- or LLM-triggered, is still explainable.

Usage:
    python -m src.escalation.escalation_rules --text "..." [--use-llm-fallback]
"""
import argparse
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

REPEAT_CONTACT_PATTERNS = [
    r"\balready\b", r"\bstill\b", r"\bagain\b", r"\b\d+(st|nd|rd|th)\s+time\b",
    r"\bsame\s+(issue|problem)\b", r"\btried\b",
    r"\bkeeps?\b.{0,15}\b(happening|freezing|dying|resetting|crashing|coming back|doing this)\b",
    r"\bkeep(ing)?\s+having\b", r"\b\d+\s*(x|attempts?|times?)\b",
    r"\b(hours?|days?)\s+(now|ago|and)\b", r"\bover\s+\d+\s*(hours?|hrs|days?)\b",
    r"\bever since\b", r"\bstill\s+not\b", r"\bhasn.?t\s+(fixed|resolved)\b",
]

FRUSTRATION_PATTERNS_CASE_INSENSITIVE = [
    r"\b(worst|ridiculous|unacceptable|ruined|awful|garbage|joke|hate|sucks|terrible|disgusted|pathetic)\b",
    r"\b(so|beyond|extremely)\s+(upset|frustrated|angry|disappointed|annoyed)\b",
]
FRUSTRATION_PATTERNS_CASE_SENSITIVE = [
    r"[!]{2,}",
    r"\b[A-Z]{4,}\b",
]

CONFIDENCE_THRESHOLD = 0.6
SIMILARITY_THRESHOLD = 0.15

LLM_FALLBACK_SYSTEM_PROMPT = """You are deciding whether an AppleSupport customer
message needs human escalation, given that keyword rules found no obvious signal.
Look for subtler cues: sarcasm, implied repeat contact, quiet exasperation, or a
request too sensitive/complex for a bot (legal threats, safety issues, refund disputes).

Respond with ONLY JSON: {"escalate": true/false, "reason": "<one sentence, specific to this message>"}
No markdown fences, no other text.
"""


def check_repeat_contact(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in REPEAT_CONTACT_PATTERNS)


def check_frustration(text: str) -> bool:
    text_lower = text.lower()
    if any(re.search(p, text_lower) for p in FRUSTRATION_PATTERNS_CASE_INSENSITIVE):
        return True
    if any(re.search(p, text) for p in FRUSTRATION_PATTERNS_CASE_SENSITIVE):
        return True
    return False


def _llm_fallback_check(text: str, model: str = "gpt-4o-mini") -> dict:
    """Only called when rules find nothing. Costs one API call."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"escalate": False, "reason": "LLM fallback skipped: OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        result = json.loads(raw)
        return {"escalate": bool(result.get("escalate", False)), "reason": result.get("reason", "")}
    except (json.JSONDecodeError, KeyError):
        return {"escalate": False, "reason": f"LLM fallback returned unparseable output: {raw!r}"}


def decide_escalation(text: str, classifier_confidence: float, top_retrieval_similarity: float,
                       confidence_threshold: float = CONFIDENCE_THRESHOLD,
                       similarity_threshold: float = SIMILARITY_THRESHOLD,
                       use_llm_fallback: bool = False) -> dict:
    reasons = []

    if check_repeat_contact(text):
        reasons.append("repeat_contact_signal: customer language suggests they already tried a fix")

    if check_frustration(text):
        reasons.append("frustration_signal: strong negative language or shouting detected")

    if classifier_confidence < confidence_threshold:
        reasons.append(
            f"low_classifier_confidence: {classifier_confidence:.2f} < threshold {confidence_threshold}"
        )

    if top_retrieval_similarity < similarity_threshold:
        reasons.append(
            f"low_retrieval_grounding: best historical match similarity {top_retrieval_similarity:.2f} "
            f"< threshold {similarity_threshold} (nothing similar enough in history to draft safely)"
        )

    if reasons:
        return {"escalate": True, "reasons": reasons, "method": "rule-based"}

    if use_llm_fallback:
        llm_result = _llm_fallback_check(text)
        if llm_result["escalate"]:
            return {
                "escalate": True,
                "reasons": [f"llm_fallback: {llm_result['reason']}"],
                "method": "llm-fallback",
            }
        return {
            "escalate": False,
            "reasons": [f"llm_fallback checked, no escalation needed: {llm_result['reason']}"],
            "method": "llm-fallback",
        }

    return {"escalate": False, "reasons": ["no escalation triggers matched (rules only)"], "method": "rule-based"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--similarity", type=float, default=0.3)
    parser.add_argument("--use-llm-fallback", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    result = decide_escalation(args.text, args.confidence, args.similarity,
                                use_llm_fallback=args.use_llm_fallback)
    print(result)