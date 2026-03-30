"""Add correctness labels by comparing ground-truth `labels` to predicted `intent_labels`."""

import ast
import argparse
from pathlib import Path

import pandas as pd


def parse_set(value) -> set[str]:
    """Parse a string like "{'Business', 'Reminders'}" into a Python set."""
    return set(ast.literal_eval(value))


def main():
    parser = argparse.ArgumentParser(description="Add correctness labels to predictions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/multiclass-email-classification.predictions.parquet"),
        help="Path to predictions parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to overwriting input)",
    )
    args = parser.parse_args()

    output = args.output or args.input

    df = pd.read_parquet(args.input)

    df["correct"] = [
        set(row["labels"]) == parse_set(row["intent_labels"])
        if pd.notna(row["intent_labels"])
        else False
        for _, row in df.iterrows()
    ]

    df.to_parquet(output, index=False)

    n = len(df)
    n_correct = df["correct"].sum()
    print(f"Accuracy: {n_correct}/{n} ({n_correct / n:.2%})")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
