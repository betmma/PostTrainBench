#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm_common import (
    VllmMultimodalRunner,
    add_common_args,
    evaluate_records,
    load_records,
    normalize_text,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser.parse_args()


def prompt(record: dict) -> str:
    return (
        "Answer the visual question. Return only the short answer, with no explanation.\n"
        f"Question: {record.get('question', record.get('query', ''))}"
    )


def score(record: dict, response: str) -> bool:
    answers = record.get("answer", record.get("answers", record.get("label")))
    if not isinstance(answers, list):
        answers = [answers]
    prediction = normalize_text(response)
    return any(prediction == normalize_text(answer) for answer in answers if answer is not None)


def main() -> None:
    args = parse_args()
    records = load_records(args.data)
    if args.limit != -1:
        records = records[:args.limit]
    runner = VllmMultimodalRunner(
        args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_connections=args.max_connections,
    )
    metrics = evaluate_records(
        records, runner, image_root=args.image_root, prompt_fn=prompt, score_fn=score
    )
    metrics["benchmark"] = "babyvision"
    write_metrics(args.json_output_file, metrics)


if __name__ == "__main__":
    main()
