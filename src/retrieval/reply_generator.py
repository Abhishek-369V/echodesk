"""
Grounded reply generator (RAG): retrieves similar historical resolved
AppleSupport replies, then prompts an LLM to draft a NEW reply grounded in
those examples — not a copy-paste, but style/content-consistent with how
Apple actually resolved similar issues.

Requires OPENAI_API_KEY (see fewshot_classifier.py for setup).

Usage:
    python -m src.retrieval.reply_generator --text "my bluetooth keeps disconnecting"
"""
import argparse
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from src.retrieval.reply_retriever import ReplyRetriever

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "data" / "processed" / "retrieval_corpus.csv"

SYSTEM_PROMPT = """You are drafting a customer support reply as AppleSupport on Twitter. 
You are given the customer's message and 2-3 similar historical cases with how Apple actually resolved them. 
Use these as grounding for tone, structure, and likely solution — but write a NEW reply tailored to THIS 
customer's message, never copy a historical reply verbatim.

Keep it under 280 characters (Twitter constraint), professional, empathetic, and actionable 
(point to a specific next step, not just "we're here to help").
"""

def build_prompt(customer_text: str, retrieved: list) -> str:
    context = "\n\n".join(
        f"Similar past case (similarity={r['similarity']:.2f}):\n"
        f"  Customer: {r['historical_customer_text']}\n"
        f"  Apple's reply: {r['historical_reply']}"
        for r in retrieved
    )
    return f"""Customer's current message: "{customer_text}"

    {context}

    Draft a new reply for the current customer's message."""


class ReplyGenerator:
    def __init__(self, corpus_path: Path = DEFAULT_CORPUS, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set!..")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retriever = ReplyRetriever(corpus_path)

    def generate(self, customer_text: str, k: int = 3) -> dict:
        retrieved = self.retriever.retrieve(customer_text, k=k)
        prompt = build_prompt(customer_text, retrieved)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return {
            "generated_reply": response.choices[0].message.content.strip(),
            "grounded_on": retrieved,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    gen = ReplyGenerator(model=args.model)
    result = gen.generate(args.text, k=args.k)
    print("GENERATED REPLY:\n", result["generated_reply"])
    print("\nGROUNDED ON:")
    for r in result["grounded_on"]:
        print(f"  [{r['similarity']:.2f}] {r['historical_reply'][:120]}")