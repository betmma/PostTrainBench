#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm_common import (
    VllmMultimodalRunner,
    add_common_args,
    evaluate_records,
    load_records,
    normalize_text,
    write_metrics,
    write_image_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser.parse_args()


def load_mathvision(path: str) -> tuple[list[dict], tempfile.TemporaryDirectory | None]:
    source = Path(path)
    if source.suffix.lower() != ".parquet":
        return load_records(source), None
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("MathVision Parquet input requires pyarrow (`pip install pyarrow`).") from exc
    tmp = tempfile.TemporaryDirectory(prefix="mathvision-")
    image_dir = Path(tmp.name)
    records = pq.read_table(source).to_pylist()
    for index, record in enumerate(records):
        blob = record.get("decoded_image") or record.get("image") or {}
        data = blob.get("bytes") if isinstance(blob, dict) else None
        if not data:
            raise ValueError(f"MathVision row {index} has no embedded image bytes")
        record["image"] = write_image_bytes(data, image_dir, f"{index}.png")
    return records, tmp


def prompt(record: dict) -> str:
    options = record.get("options") or record.get("choices") or []
    choices = "\n".join(f"{chr(65+i)}. {value}" for i, value in enumerate(options))
    return (
        "Solve the visual math problem. Return only the option letter or exact answer.\n"
        f"Problem: {record.get('question', record.get('problem', ''))}\n{choices}"
    )


def score(record: dict, response: str) -> bool:
    answer = record.get("answer", record.get("label", record.get("答案")))
    prediction = normalize_text(response)
    if isinstance(answer, int):
        return bool(re.search(rf"\b{chr(65 + answer)}\b", response.upper()))
    return prediction == normalize_text(answer)


def main() -> None:
    args = parse_args()
    records, temporary = load_mathvision(args.data)
    if args.limit != -1:
        records = records[:args.limit]
    runner = VllmMultimodalRunner(args.model_path, max_tokens=args.max_tokens, temperature=args.temperature,
                                   gpu_memory_utilization=args.gpu_memory_utilization,
                                   max_connections=args.max_connections)
    metrics = evaluate_records(records, runner, image_root=args.image_root, prompt_fn=prompt, score_fn=score)
    metrics["benchmark"] = "mathvision"
    write_metrics(args.json_output_file, metrics)
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
