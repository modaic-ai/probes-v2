import argparse
import json
import logging
from pathlib import Path

import yaml
import dspy
from datasets import Dataset, load_dataset

from modaic import Predict
from modaic.safe_lm import SafeLM
from modaic.programs.utils import PredictYamlSpec

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM inference on a dataset using a YAML-defined arbiter")
    parser.add_argument("--arbiter", required=True, help="Path to arbiter YAML spec")
    parser.add_argument("--input", required=True, help="Path to input dataset (.jsonl, .parquet, .csv, .arrow)")
    parser.add_argument("--output", required=True, help="Path to output dataset")
    parser.add_argument(
        "--threads",
        type=int,
        default=64,
        help="Maximum number of concurrent DSPy requests. Defaults to DSPy's configured thread count.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=WARNING, -vv=INFO, -vvv=DEBUG)",
    )
    return parser.parse_args()


def configure_logging(verbosity: int) -> None:
    if verbosity >= 3:
        level = logging.DEBUG
    elif verbosity >= 2:
        level = logging.INFO
    elif verbosity >= 1:
        level = logging.WARNING
    else:
        level = logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_input_dataset(path: str) -> Dataset:
    logger.info("Loading dataset from %s", path)
    ext = Path(path).suffix.lower()
    logger.debug("Detected file extension: %s", ext)
    if ext == ".jsonl":
        ds = Dataset.from_json(path)
    elif ext == ".parquet":
        ds = Dataset.from_parquet(path)
    elif ext == ".csv":
        ds = Dataset.from_csv(path)
    elif ext == ".arrow":
        ds = load_dataset("arrow", data_files=path, split="train")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .jsonl, .parquet, .csv, or .arrow")
    logger.info("Dataset loaded: %d rows, columns=%s", len(ds), ds.column_names)
    return ds


def save_output_dataset(dataset: Dataset, path: str) -> None:
    logger.info("Saving dataset to %s", path)
    ext = Path(path).suffix.lower()
    if ext == ".jsonl":
        dataset.to_json(path)
    elif ext == ".parquet":
        dataset.to_parquet(path)
    elif ext == ".csv":
        dataset.to_csv(path)
    elif ext == ".arrow":
        dataset.save_to_disk(path)
    else:
        raise ValueError(f"Unsupported output format: {ext}. Use .jsonl, .parquet, .csv, or .arrow")
    logger.info("Dataset saved to %s", path)


def build_arbiter(yaml_path: str) -> tuple[Predict, PredictYamlSpec]:
    with open(yaml_path) as f:
        spec = PredictYamlSpec(**yaml.safe_load(f))

    arbiter = Predict.from_yaml(yaml_path).as_arbiter()

    safe_lm = SafeLM.from_lm(arbiter.lm)
    arbiter.lm = safe_lm
    dspy.configure(lm=safe_lm)

    return arbiter, spec


def _append_prediction(
    new_columns: dict[str, list],
    prediction: dspy.Prediction,
    output_fields: list[str],
) -> None:
    for field in output_fields:
        new_columns[field].append(getattr(prediction, field, None))

    messages_obj = {
        "messages": getattr(prediction, "_messages", []),
        "outputs": getattr(prediction, "_outputs", {}),
    }
    new_columns["messages"].append(json.dumps(messages_obj))


def process_dataset(
    arbiter: Predict,
    dataset: Dataset,
    input_fields: list[str],
    output_fields: list[str],
    threads: int | None = None,
) -> Dataset:
    total = len(dataset)
    if threads is None:
        threads = getattr(dspy.settings, "num_threads", 8)
    if threads < 1:
        raise ValueError(f"--threads must be >= 1, got {threads}")

    resolved_threads = min(threads, total) if total else 0
    logger.info(
        "Processing %d rows with dspy.Parallel (%d workers)",
        total,
        resolved_threads,
    )
    new_columns: dict[str, list] = {field: [] for field in output_fields}
    new_columns["messages"] = []
    chunk_exec_pairs: list[tuple[Predict, dict]] = []

    for i, row in enumerate(dataset):
        kwargs = {field: row[field] for field in input_fields}
        kwargs["return_messages"] = True
        logger.debug("Row %d/%d input: %s", i + 1, total, kwargs)
        chunk_exec_pairs.append((arbiter, kwargs))

        is_chunk_boundary = len(chunk_exec_pairs) == resolved_threads or i + 1 == total
        if not is_chunk_boundary:
            continue

        parallel = dspy.Parallel(
            num_threads=len(chunk_exec_pairs),
            return_failed_examples=True,
            provide_traceback=True,
            disable_progress_bar=True,
        )
        predictions, _, exceptions = parallel(chunk_exec_pairs)
        if exceptions:
            failed_offset = next(
                (offset for offset, prediction in enumerate(predictions) if prediction is None),
                0,
            )
            failed_row = i + 1 - len(chunk_exec_pairs) + failed_offset + 1
            raise RuntimeError(f"Prediction failed for row {failed_row}") from exceptions[0]

        chunk_start = i + 1 - len(chunk_exec_pairs)
        for offset, prediction in enumerate(predictions):
            row_num = chunk_start + offset + 1
            _append_prediction(new_columns, prediction, output_fields)
            logger.debug(
                "Row %d/%d output fields: %s",
                row_num,
                total,
                {field: getattr(prediction, field, None) for field in output_fields},
            )

        logger.info("Progress: %d/%d rows (%.0f%%)", i + 1, total, (i + 1) / total * 100)
        chunk_exec_pairs.clear()

    for col_name, values in new_columns.items():
        dataset = dataset.add_column(col_name, values)

    return dataset


def main():
    args = parse_args()
    configure_logging(args.verbose)

    dataset = load_input_dataset(args.input)

    arbiter, spec = build_arbiter(args.arbiter)

    input_fields = [f.name for f in spec.inputs]
    output_fields = [f.name for f in spec.outputs]
    logger.info("Input fields: %s", input_fields)
    logger.info("Output fields: %s", output_fields)

    for field in input_fields:
        if field not in dataset.column_names:
            raise ValueError(f"Input field '{field}' not found in dataset columns: {dataset.column_names}")

    result = process_dataset(
        arbiter,
        dataset,
        input_fields,
        output_fields,
        threads=args.threads,
    )

    save_output_dataset(result, args.output)
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
