"""
The EchoDesk agent -- wires classifier + retriever + reply generator + escalation logic into ONE callable pipeline. 
It can classify, draft a reply, and decide escalation[True/False] - as per assessment deliverable!"

Usage examples:
    python -m src.agent.agent --text "my bluetooth keeps disconnecting"
    python -m src.agent.agent --text "guess I'll just switch to Android then" --use-llm-escalation-fallback
"""

import argparse
import json
from pathlib import Path

from src.classifier.fewshot_classifier import FewShotClassifier
from src.retrieval.reply_generator import ReplyGenerator
from src.escalation.escalation_rules import decide_escalation
from dotenv import load_dotenv

load_dotenv()


class EchoDeskAgent:
    def __init__(self, model: str = "gpt-4o-mini", use_llm_escalation_fallback: bool = False):
        self.classifier = FewShotClassifier(model=model)
        self.generator = ReplyGenerator(model=model)
        self.use_llm_escalation_fallback = use_llm_escalation_fallback

    def handle(self, customer_text: str) -> dict:
        # Step 1: classify intent
        intent_result = self.classifier.classify(customer_text)

        # Step 2: retrieve + generate grounded reply
        gen_result = self.generator.generate(customer_text)
        top_similarity = gen_result["grounded_on"][0]["similarity"] if gen_result["grounded_on"] else 0.0

        # Step 3: decide escalation, informed by classifier confidence + retrieval grounding
        escalation_result = decide_escalation(
            customer_text,
            classifier_confidence=intent_result.get("confidence", 0.0),
            top_retrieval_similarity=top_similarity,
            use_llm_fallback=self.use_llm_escalation_fallback,
        )

        return {
            "customer_text": customer_text,
            "intent": intent_result["intent"],
            "intent_confidence": intent_result.get("confidence"),
            "draft_reply": gen_result["generated_reply"],
            "escalate": escalation_result["escalate"],
            "escalation_reasons": escalation_result["reasons"],
            "escalation_method": escalation_result["method"],
            "grounded_on_similarity": top_similarity,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--use-llm-escalation-fallback", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    agent = EchoDeskAgent(model=args.model, use_llm_escalation_fallback=args.use_llm_escalation_fallback)
    result = agent.handle(args.text)
    print(json.dumps(result, indent=2))