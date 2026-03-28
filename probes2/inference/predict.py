import asyncio
import argparse
import copy
import json
import logging
import os
import random
from pathlib import Path

import yaml
import dspy
from datasets import Dataset, load_dataset

import probes2.inference.registry  # noqa: F401 — registers model capabilities

from modaic import Predict
from modaic.safe_lm import SafeLM
from modaic.programs.utils import PredictYamlSpec

logger = logging.getLogger(__name__)


def _get_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return int(value)


def _get_env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return float(value)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM inference on a dataset using a YAML-defined arbiter")
    parser.add_argument("-a", "--arbiter", required=True, help="Path to arbiter YAML spec")
    parser.add_argument("-i", "--input", required=True, help="Path to input dataset (.jsonl, .parquet, .arrow)")
    parser.add_argument("-o", "--output", required=True, help="Path to output dataset")
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=64,
        help="Maximum number of concurrent DSPy requests. Defaults to DSPy's configured thread count.",
    )
    parser.add_argument(
        "-sc", "--self-consistency",
        type=float,
        default=None,
        help="Fraction of rows (0.0-1.0) to run self-consistency scoring on (10 additional predictions each)",
    )
    parser.add_argument(
        "-cc", "--cross-consistency",
        type=float,
        default=None,
        help="Fraction of remaining rows (after SC) to run cross-consistency scoring on",
    )
    parser.add_argument(
        "--council",
        type=str,
        default=None,
        help="Path to council YAML file specifying frontier models for cross-consistency",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=WARNING, -vv=INFO, -vvv=DEBUG)",
    )
    parser.add_argument(
        "--batch-client",
        choices=("vllm", "modal"),
        default=None,
        help=(
            "Optional batch execution backend. Defaults to regular DSPy parallel execution. "
            "When set, the main arbiter pass and self-consistency use modaic.Predict.abatch."
        ),
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        help="vLLM reasoning parser name (e.g. 'qwen3'). Automatically enables thinking mode.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graphs in vLLM (useful for debugging).",
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
    for noisy_logger in (
        "httpcore",
        "httpcore.http11",
        "httpx",
        "httpx._client",
        "httpx._transports.default",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


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
        raise ValueError(f"Unsupported file format: {ext}. Use .jsonl, .parquet, or .arrow")
    logger.info("Dataset loaded: %d rows, columns=%s", len(ds), ds.column_names)
    return ds


def save_output_dataset(dataset: Dataset, path: str) -> None:
    logger.info("Saving dataset to %s", path)
    ext = Path(path).suffix.lower()
    if ext == ".jsonl":
        dataset.to_json(path)
    elif ext == ".parquet":
        dataset.to_parquet(path)
    elif ext == ".arrow":
        dataset.save_to_disk(path)
    else:
        raise ValueError(f"Unsupported output format: {ext}. Use .jsonl, .parquet, or .arrow")
    logger.info("Dataset saved to %s", path)


def build_arbiter(yaml_path: str) -> tuple[Predict, PredictYamlSpec]:
    with open(yaml_path) as f:
        spec = PredictYamlSpec(**yaml.safe_load(f))

    arbiter = Predict.from_yaml(yaml_path).as_arbiter()

    safe_lm = SafeLM.from_lm(arbiter.lm)
    arbiter.lm = safe_lm
    dspy.configure(lm=safe_lm)

    return arbiter, spec


def build_council_arbiters(arbiter: Predict, council_path: str) -> list[Predict]:
    with open(council_path) as f:
        council_specs = yaml.safe_load(f)

    council_arbiters = []
    for spec in council_specs:
        spec = dict(spec)
        model = spec.pop("model")
        council_arbiter = copy.deepcopy(arbiter)
        lm = dspy.LM(model, **spec)
        council_arbiter.lm = SafeLM.from_lm(lm)
        council_arbiters.append(council_arbiter)
        logger.info("Council arbiter created: %s", model)
    return council_arbiters


def _build_batch_client(
    batch_client_name: str,
    lm: dspy.LM,
    reasoning_parser: str | None = None,
    enforce_eager: bool = False,
):
    model = getattr(lm, "model", None)
    if not isinstance(model, str) or not model.startswith("huggingface/"):
        raise ValueError(
            "Batch execution requires an arbiter LM with model='huggingface/...'. "
            "Use model='huggingface/Qwen/Qwen3.5-4B' and omit api_base."
        )

    shared_kwargs: dict[str, object] = {}
    if reasoning_parser:
        shared_kwargs["reasoning_parser"] = reasoning_parser
    if enforce_eager:
        shared_kwargs["enforce_eager"] = enforce_eager
    if (max_model_len := _get_env_int("MODAIC_BATCH_MAX_MODEL_LEN")) is not None:
        shared_kwargs["max_model_len"] = max_model_len
    if (gpu_mem := _get_env_float("MODAIC_BATCH_GPU_MEMORY_UTILIZATION")) is not None:
        shared_kwargs["gpu_memory_utilization"] = gpu_mem
    if (tp := _get_env_int("MODAIC_BATCH_TENSOR_PARALLEL_SIZE")) is not None:
        shared_kwargs["tensor_parallel_size"] = tp

    try:
        if batch_client_name == "modal":
            from modaic.batch import ModalBatchClient

            modal_kwargs = {
                "gpu": os.environ.get("MODAIC_MODAL_GPU") or "A100:2",
                **shared_kwargs,
            }
            logger.info("Creating ModalBatchClient with %s", modal_kwargs)
            return ModalBatchClient(lm=lm, **modal_kwargs)
        if batch_client_name == "vllm":
            from modaic.batch import VLLMBatchClient

            logger.info("Creating VLLMBatchClient with %s", shared_kwargs)
            return VLLMBatchClient(lm=lm, **shared_kwargs)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Batch client '{batch_client_name}' is unavailable in the current environment. "
            "Install the required batch dependencies (for example duckdb, and vllm for the local vLLM client) and retry."
        ) from exc

    raise ValueError(f"Unsupported batch client: {batch_client_name}")


def _append_prediction(
    new_columns: dict[str, list],
    prediction: object,
    output_fields: list[str],
) -> None:
    prediction_error = getattr(prediction, "error", None) if _is_failed_prediction(prediction) else None
    for field in output_fields:
        new_columns[field].append(getattr(prediction, field, None))

    prediction_obj = {field: getattr(prediction, field, None) for field in output_fields}
    if prediction_error is not None:
        prediction_obj["error"] = prediction_error
    new_columns["prediction"].append(prediction_obj)

    messages_obj = {
        "messages": getattr(prediction, "_messages", []),
        "outputs": getattr(prediction, "_outputs", {}),
    }
    new_columns["messages"].append(messages_obj)
    new_columns["prediction_error"].append(prediction_error)


def _compute_consistency_confidence(
    original_predictions: list[dspy.Prediction],
    sc_predictions: list[list[dspy.Prediction]],
    output_fields: list[str],
) -> list[float]:
    confidences = []
    for orig, samples in zip(original_predictions, sc_predictions):
        orig_values = tuple(str(getattr(orig, f, "")).strip() for f in output_fields)
        matches = sum(
            1 for s in samples
            if tuple(str(getattr(s, f, "")).strip() for f in output_fields) == orig_values
        )
        confidences.append(matches / len(samples))
    return confidences


def _run_parallel_chunks(
    exec_pairs: list[tuple[Predict, dict]],
    resolved_threads: int,
    label: str,
) -> list[dspy.Prediction]:
    all_predictions: list[dspy.Prediction] = []
    total_pairs = len(exec_pairs)
    for start in range(0, total_pairs, resolved_threads):
        chunk = exec_pairs[start : start + resolved_threads]
        parallel = dspy.Parallel(
            num_threads=len(chunk),
            return_failed_examples=True,
            provide_traceback=True,
            disable_progress_bar=True,
        )
        predictions, _, exceptions = parallel(chunk)
        if exceptions:
            failed_offset = next(
                (offset for offset, p in enumerate(predictions) if p is None),
                0,
            )
            raise RuntimeError(
                f"{label} failed at pair {start + failed_offset + 1}"
            ) from exceptions[0]
        all_predictions.extend(predictions)
        logger.info(
            "%s progress: %d/%d pairs (%.0f%%)",
            label,
            start + len(chunk),
            total_pairs,
            (start + len(chunk)) / total_pairs * 100,
        )
    return all_predictions


def _is_failed_prediction(prediction: object) -> bool:
    return (
        prediction.__class__.__name__ == "FailedPrediction"
        and hasattr(prediction, "error")
        and hasattr(prediction, "index")
    )


def _run_batch_predictions(
    arbiter: Predict,
    input_rows: list[dict],
    batch_client_name: str,
    label: str,
    row_numbers: list[int] | None = None,
    reasoning_parser: str | None = None,
    enforce_eager: bool = False,
) -> list[object]:
    if not input_rows:
        return []

    client = _build_batch_client(batch_client_name, arbiter.lm, reasoning_parser=reasoning_parser, enforce_eager=enforce_eager)
    result = asyncio.run(
        arbiter.abatch(
            input_rows,
            show_progress=False,
            return_messages=True,
            client=client,
        )
    )

    predictions: list[object] = []
    failed_rows: list[str] = []
    for offset, row in enumerate(result):
        prediction = row.prediction
        if _is_failed_prediction(prediction):
            failed_row = row_numbers[offset] if row_numbers is not None else offset + 1
            error = getattr(prediction, "error", "unknown batch error")
            failed_rows.append(f"{failed_row}: {error}")
        predictions.append(prediction)
    if failed_rows:
        logger.warning(
            "%s completed with %d failed rows after adapter retry: %s",
            label,
            len(failed_rows),
            "; ".join(failed_rows[:10]),
        )
    return predictions


def process_dataset(
    arbiter: Predict,
    dataset: Dataset,
    input_fields: list[str],
    output_fields: list[str],
    threads: int | None = None,
    batch_client_name: str | None = None,
    self_consistency: float | None = None,
    cross_consistency: float | None = None,
    council_arbiters: list[Predict] | None = None,
    reasoning_parser: str | None = None,
    enforce_eager: bool = False,
) -> Dataset:
    SC_NUM_SAMPLES = 10

    total = len(dataset)
    if threads is None:
        threads = getattr(dspy.settings, "num_threads", 8)
    if threads < 1:
        raise ValueError(f"--threads must be >= 1, got {threads}")

    resolved_threads = min(threads, total) if total else 0
    if batch_client_name is None:
        logger.info(
            "Processing %d rows with dspy.Parallel (%d workers)",
            total,
            resolved_threads,
        )
    else:
        logger.info("Processing %d rows with modaic.Predict.abatch via %s", total, batch_client_name)
        logger.info("--threads is ignored for the main arbiter pass and self-consistency in batch mode")
    new_columns: dict[str, list] = {field: [] for field in output_fields}
    new_columns["prediction"] = []
    new_columns["messages"] = []
    new_columns["prediction_error"] = []
    all_predictions: list[object] = []
    input_rows = []
    for i, row in enumerate(dataset):
        kwargs = {field: row[field] for field in input_fields}
        logger.debug("Row %d/%d input: %s", i + 1, total, kwargs)
        input_rows.append(kwargs)

    if batch_client_name is None:
        chunk_exec_pairs: list[tuple[Predict, dict]] = []

        for i, kwargs in enumerate(input_rows):
            kwargs_with_messages = dict(kwargs)
            kwargs_with_messages["return_messages"] = True
            chunk_exec_pairs.append((arbiter, kwargs_with_messages))

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
                all_predictions.append(prediction)
                _append_prediction(new_columns, prediction, output_fields)
                logger.debug(
                    "Row %d/%d output fields: %s",
                    row_num,
                    total,
                    {field: getattr(prediction, field, None) for field in output_fields},
                )

            logger.info("Progress: %d/%d rows (%.0f%%)", i + 1, total, (i + 1) / total * 100)
            chunk_exec_pairs.clear()
    else:
        predictions = _run_batch_predictions(
            arbiter=arbiter,
            input_rows=input_rows,
            batch_client_name=batch_client_name,
            label="Prediction",
            row_numbers=list(range(1, total + 1)),
            reasoning_parser=reasoning_parser,
            enforce_eager=enforce_eager,
        )
        for row_num, prediction in enumerate(predictions, start=1):
            all_predictions.append(prediction)
            _append_prediction(new_columns, prediction, output_fields)
            logger.debug(
                "Row %d/%d output fields: %s",
                row_num,
                total,
                {field: getattr(prediction, field, None) for field in output_fields},
            )
        if total > 0:
            logger.info("Progress: %d/%d rows (100%%)", total, total)

    # Self-consistency pass
    if self_consistency is not None and total > 0:
        eligible_sc_indices = [idx for idx, prediction in enumerate(all_predictions) if not _is_failed_prediction(prediction)]
        sc_indices: set[int] = set()
        if eligible_sc_indices:
            sc_count = max(1, int(len(eligible_sc_indices) * self_consistency))
            sc_indices = set(random.sample(eligible_sc_indices, min(sc_count, len(eligible_sc_indices))))
            logger.info(
                "Running self-consistency pass: %d/%d eligible rows selected (%.0f%%), %d samples each",
                len(sc_indices), len(eligible_sc_indices), self_consistency * 100, SC_NUM_SAMPLES,
            )
        else:
            logger.warning("Skipping self-consistency because the main pass produced no successful predictions")

        sc_exec_pairs: list[tuple[Predict, dict]] = []
        sc_input_rows: list[dict] = []
        sc_row_numbers: list[int] = []
        for row_idx in sorted(sc_indices):
            row = dataset[row_idx]
            kwargs = {field: row[field] for field in input_fields}
            for _ in range(SC_NUM_SAMPLES):
                if batch_client_name is None:
                    sc_exec_pairs.append((arbiter, kwargs))
                else:
                    sc_input_rows.append(dict(kwargs))
                    sc_row_numbers.append(row_idx + 1)

        sc_confidence_col: list[float | None] = [None] * total
        if sc_indices:
            if batch_client_name is None:
                sc_all = _run_parallel_chunks(sc_exec_pairs, resolved_threads, "Self-consistency")
            else:
                sc_all = _run_batch_predictions(
                    arbiter=arbiter,
                    input_rows=sc_input_rows,
                    batch_client_name=batch_client_name,
                    label="Self-consistency",
                    row_numbers=sc_row_numbers,
                    reasoning_parser=reasoning_parser,
                    enforce_eager=enforce_eager,
                )

            # Group SC predictions by selected row
            sc_by_selected: list[list[dspy.Prediction]] = []
            for i in range(len(sc_indices)):
                start = i * SC_NUM_SAMPLES
                sc_by_selected.append(sc_all[start : start + SC_NUM_SAMPLES])

            # Compute confidence for selected rows
            selected_preds = [all_predictions[idx] for idx in sorted(sc_indices)]
            sc_confidences = _compute_consistency_confidence(selected_preds, sc_by_selected, output_fields)

            for conf, row_idx in zip(sc_confidences, sorted(sc_indices)):
                sc_confidence_col[row_idx] = conf

        new_columns["sc_confidence"] = sc_confidence_col

    # Cross-consistency pass
    if cross_consistency is not None and council_arbiters and total > 0:
        sc_indices_used = sc_indices if self_consistency is not None else set()
        remaining_indices = sorted(
            idx
            for idx, prediction in enumerate(all_predictions)
            if idx not in sc_indices_used and not _is_failed_prediction(prediction)
        )
        cc_indices: set[int] = set()
        num_models = len(council_arbiters)
        if remaining_indices:
            cc_count = max(1, int(len(remaining_indices) * cross_consistency))
            cc_indices = set(random.sample(remaining_indices, min(cc_count, len(remaining_indices))))
            logger.info(
                "Running cross-consistency pass: %d/%d eligible rows selected, %d council models",
                len(cc_indices), len(remaining_indices), num_models,
            )
        else:
            logger.warning("Skipping cross-consistency because no successful unscored predictions remain")

        cc_exec_pairs: list[tuple[Predict, dict]] = []
        for row_idx in sorted(cc_indices):
            row = dataset[row_idx]
            kwargs = {field: row[field] for field in input_fields}
            for council_arbiter in council_arbiters:
                cc_exec_pairs.append((council_arbiter, kwargs))

        cc_confidence_col: list[float | None] = [None] * total
        if cc_indices:
            cc_all = _run_parallel_chunks(cc_exec_pairs, resolved_threads, "Cross-consistency")

            # Group CC predictions: [row][model]
            cc_by_row: list[list[dspy.Prediction]] = []
            for i in range(len(cc_indices)):
                start = i * num_models
                cc_by_row.append(cc_all[start : start + num_models])

            selected_preds = [all_predictions[idx] for idx in sorted(cc_indices)]
            cc_confidences = _compute_consistency_confidence(selected_preds, cc_by_row, output_fields)

            for conf, row_idx in zip(cc_confidences, sorted(cc_indices)):
                cc_confidence_col[row_idx] = conf
        new_columns["cc_confidence"] = cc_confidence_col

    for col_name, values in new_columns.items():
        dataset = dataset.add_column(col_name, values)

    return dataset


def main():
    args = parse_args()
    configure_logging(args.verbose)

    dataset = load_input_dataset(args.input)

    if args.cross_consistency is not None and args.council is None:
        parser_err = "--cross-consistency requires --council"
        raise SystemExit(f"error: {parser_err}")
    if args.council is not None and args.cross_consistency is None:
        raise SystemExit("error: --council requires --cross-consistency")

    arbiter, spec = build_arbiter(args.arbiter)

    council_arbiters = None
    if args.council:
        council_arbiters = build_council_arbiters(arbiter, args.council)

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
        batch_client_name=args.batch_client,
        self_consistency=args.self_consistency,
        cross_consistency=args.cross_consistency,
        council_arbiters=council_arbiters,
        reasoning_parser=args.reasoning_parser,
        enforce_eager=args.enforce_eager,
    )

    save_output_dataset(result, args.output)
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
