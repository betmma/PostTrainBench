#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import zipfile

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


def load_babyvision(path: str) -> tuple[list[dict], tempfile.TemporaryDirectory | None]:
    source = Path(path)
    if source.suffix.lower() != ".zip":
        return load_records(source), None
    tmp = tempfile.TemporaryDirectory(prefix="babyvision-")
    root = Path(tmp.name)
    records = []
    with zipfile.ZipFile(source) as archive:
        metadata = archive.read("babyvision_data/meta_data.jsonl").decode()
        for line in metadata.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            member = f"babyvision_data/{row['image']}"
            target = root / row["image"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            if row.get("ansType") == "blank":
                question, answer = row["question"], row["blankAns"]
            else:
                options = row.get("options") or []
                choices = "\n".join(f"{chr(65+i)}. {value}" for i, value in enumerate(options))
                question = f"{row['question']}\nChoices:\n{choices}"
                answer = chr(65 + int(row["choiceAns"]))
            records.append({
                **row,
                "image": str(target),
                "question": question,
                "answer": answer,
            })
    return records, tmp


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
    records, temporary = load_babyvision(args.data)
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
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
