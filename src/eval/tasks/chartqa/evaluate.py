#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm_common import (
    VllmMultimodalRunner,
    add_common_args,
    evaluate_records,
    load_hf_split,
    load_records,
    normalize_text,
    write_pil_image,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser.parse_args()


def load_chartqa(path: str, image_root: str) -> tuple[list[dict], tempfile.TemporaryDirectory | None]:
    source = Path(path)
    if source.suffix.lower() != ".zip":
        return load_records(source), None
    tmp = tempfile.TemporaryDirectory(prefix="chartqa-")
    root = Path(tmp.name)
    records = []
    with zipfile.ZipFile(source) as archive:
        for subset in ("human", "augmented"):
            rows = json.loads(archive.read(f"ChartQA Dataset/test/test_{subset}.json"))
            for index, row in enumerate(rows):
                member = f"ChartQA Dataset/test/png/{row['imgname']}"
                target = root / row["imgname"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
                records.append({
                    "id": f"{subset}:{index}",
                    "subset": subset,
                    "image": str(target),
                    "question": row["query"],
                    "answer": row["label"],
                })
    return records, tmp


def load_cached_chartqa() -> tuple[list[dict], tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="chartqa-")
    image_dir = Path(tmp.name)
    records = []
    for index, row in enumerate(load_hf_split("HuggingFaceM4/ChartQA", "test")):
        records.append({
            "id": index,
            "image": write_pil_image(row["image"], image_dir, f"{index}.png"),
            "question": row["query"],
            "answer": row["label"],
            "subset": row.get("human_or_machine"),
        })
    return records, tmp


def prompt(record: dict) -> str:
    return (
        "Answer the chart question. Return only the concise answer, with no explanation.\n"
        f"Question: {record.get('question', record.get('query', ''))}"
    )


def score(record: dict, response: str) -> bool:
    answers = record.get("answer", record.get("answers", record.get("label")))
    if not isinstance(answers, list):
        answers = [answers]
    prediction = normalize_text(response)
    for answer in answers:
        if answer is None:
            continue
        target = normalize_text(answer)
        try:
            predicted_number, target_number = float(prediction), float(target)
            if math.isfinite(predicted_number) and math.isfinite(target_number):
                if abs(predicted_number - target_number) <= 0.05 * abs(target_number):
                    return True
                continue
        except ValueError:
            pass
        if prediction == target:
            return True
    return False


def main() -> None:
    args = parse_args()
    if args.data:
        records, temporary = load_chartqa(args.data, args.image_root)
    else:
        records, temporary = load_cached_chartqa()
    if args.limit != -1:
        records = records[:args.limit]
    runner = VllmMultimodalRunner(
        args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_connections=args.max_connections,
    )
    metrics = evaluate_records(records, runner, image_root=args.image_root, prompt_fn=prompt, score_fn=score)
    metrics["benchmark"] = "chartqa"
    write_metrics(args.json_output_file, metrics)
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
