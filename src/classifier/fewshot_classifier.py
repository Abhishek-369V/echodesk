"""
Few-shot intent classifier using an LLM prompt instead of a trained model.
This is the PRIMARY classifier — the TF-IDF baseline exists only for comparison.

Usage:
    python -m src.classifier.fewshot_classifier --text "my bluetooth keeps disconnecting"
"""

import argparse
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INTENTS = [
    "software_bug", "battery_hardware", "connectivity",
    "app_subscription", "account_security",
]

FEW_SHOT_EXAMPLES = """
Examples:
Message: "autocorrect keeps changing I to a random letter, so annoying"
Intent: software_bug

Message: "my battery drains so fast since the update, from 100 to 20 in an hour"
Intent: battery_hardware

Message: "bluetooth won't connect to my car since ios 11"
Intent: connectivity

Message: "apple music says I'm not subscribed but I'm being charged every month"
Intent: app_subscription

Message: "my phone was stolen, how do I wipe it remotely with find my iphone"
Intent: account_security
"""

SYSTEM_PROMPT = f"""You are an intent classifier for AppleSupport customer messages.
Classify the message into EXACTLY ONE of these categories: {", ".join(INTENTS)}.

{FEW_SHOT_EXAMPLES}

Respond with ONLY a JSON object: {{"intent": "<category>", "confidence": <0-1 float>}}
No other text, no markdown fences.
"""


class FewShotClassifier:
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Put it in a .env file and load it "
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def classify(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        # GPT models sometimes wrap JSON in markdown fences despite instructions
        # not to — strip them before parsing rather than trusting the prompt alone.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Model returned non-JSON output: {raw!r}")
        if result.get("intent") not in INTENTS:
            raise ValueError(f"Model returned invalid intent: {result.get('intent')!r}")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    clf = FewShotClassifier(model=args.model)
    result = clf.classify(args.text)
    print(json.dumps(result, indent=2))