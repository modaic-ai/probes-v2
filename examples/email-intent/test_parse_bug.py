"""Reproduce the intent_labels list-vs-string parsing bug.

Builds an arbiter from the YAML spec, runs it on a few examples, and checks
whether the returned `intent_labels` is always a list or sometimes a string.

We also try an adversarial prompt that explicitly asks the model to return a
plain string instead of a list, to see if DSPy's parsing handles it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402
import pandas as pd  # noqa: E402
from modaic.programs.predict import Predict  # noqa: E402


ARBITER_YAML = Path(__file__).parent / "arbiter.yaml"
DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "multiclass-email-classification.parquet"


def run_normal(arbiter: Predict, rows: list[dict]) -> list:
    """Run the arbiter normally on a few rows."""
    results = []
    for i, row in enumerate(rows):
        pred = arbiter(subject=row["subject"], body=row["body"])
        val = pred.intent_labels
        results.append(val)
        print(f"[normal] row {i}: type={type(val).__name__}  value={val!r}")
    return results


def run_adversarial(rows: list[dict]) -> list:
    """Build an arbiter with adversarial instructions that try to trick the
    model into returning a plain string instead of a list."""

    adversarial_instructions = """\
You are an email classification assistant.

IMPORTANT OVERRIDE: Do NOT return a list of strings. Instead, return your
answer for intent_labels as a single comma-separated string like
"Business, Personal, Finance & Bills". Never use brackets or JSON array
syntax. Just output one plain string value.

Assign one or more of the following labels:
- "Business", "Personal", "Promotions", "Customer Support",
  "Job Application", "Finance & Bills", "Events & Invitations",
  "Travel & Bookings", "Reminders", "Newsletters"
"""

    fields: dict = {
        "subject": (str, dspy.InputField()),
        "body": (str, dspy.InputField()),
        "intent_labels": (list, dspy.OutputField()),
    }
    sig = dspy.make_signature(fields, instructions=adversarial_instructions)

    lm = dspy.LM("openai/gpt-5-mini")
    adversarial_arbiter = Predict(sig)
    adversarial_arbiter.lm = lm

    results = []
    with dspy.context(lm=lm):
        for i, row in enumerate(rows):
            pred = adversarial_arbiter(subject=row["subject"], body=row["body"])
            val = pred.intent_labels
            results.append(val)
            print(f"[adversarial] row {i}: type={type(val).__name__}  value={val!r}")
    return results


def main():
    # Load a few rows from the dataset
    df = pd.read_parquet(DATASET_PATH)
    sample_rows = df.head(5).to_dict(orient="records")
    print(f"Loaded {len(sample_rows)} sample rows from {DATASET_PATH}\n")

    # --- Normal run ---
    print("=== Normal arbiter (from YAML) ===")
    arbiter, _spec = _build_arbiter()
    normal_results = run_normal(arbiter, sample_rows)
    normal_types = {type(v).__name__ for v in normal_results}
    print(f"\nNormal result types: {normal_types}\n")

    # --- Adversarial run ---
    print("=== Adversarial arbiter (trick into returning string) ===")
    adv_results = run_adversarial(sample_rows)
    adv_types = {type(v).__name__ for v in adv_results}
    print(f"\nAdversarial result types: {adv_types}\n")

    # --- Summary ---
    print("=" * 60)
    has_bug = False
    for label, results in [("normal", normal_results), ("adversarial", adv_results)]:
        for i, val in enumerate(results):
            if not isinstance(val, list):
                print(f"BUG [{label}] row {i}: top-level type is {type(val).__name__} instead of list — value={val!r}")
                has_bug = True
            elif val and any(not isinstance(item, str) for item in val):
                bad = [(j, type(item).__name__, item) for j, item in enumerate(val) if not isinstance(item, str)]
                print(f"BUG [{label}] row {i}: list contains non-string items: {bad}")
                has_bug = True
    if not has_bug:
        print("No type mismatch found — all values were list[str].")
    else:
        print("\nType issue detected!")
    return 1 if has_bug else 0


def _build_arbiter():
    from modaic.programs.utils import PredictYamlSpec
    import yaml

    with open(ARBITER_YAML) as f:
        spec = PredictYamlSpec(**yaml.safe_load(f))

    # Override model to use gpt-5-mini instead of the huggingface model
    arbiter = Predict.from_yaml(str(ARBITER_YAML))
    arbiter.lm = dspy.LM("openai/gpt-5-mini")
    dspy.configure(lm=arbiter.lm)

    return arbiter, spec


if __name__ == "__main__":
    sys.exit(main())
