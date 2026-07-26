#!/usr/bin/env python3
"""Evaluate visual ARC-AGI-1 using rendered puzzle images."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
EVALUATION_CODE = Path(__file__).resolve().parent / "evaluation_code"
if str(EVALUATION_CODE) not in sys.path:
    sys.path.insert(0, str(EVALUATION_CODE))

from vlm_common import (
    VllmMultimodalRunner,
    add_common_args,
    extract_json,
    load_hf_split,
    load_records,
    resolve_image,
    write_metrics,
)
from puzzle.generator import ArcPuzzleGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--task-root")
    return parser.parse_args()


def load_cached_arcagi() -> tuple[list[dict], tempfile.TemporaryDirectory]:
    """Render the official 400 ARC-AGI-1 evaluation tasks from the HF cache."""
    temporary = tempfile.TemporaryDirectory(prefix="arcagi-visual-")
    root = Path(temporary.name)
    task_dir = root / "tasks"
    rendered_dir = root / "rendered"
    task_dir.mkdir(parents=True)

    rows = load_hf_split("dataartist/arc-agi", "evaluation")
    task_paths: list[tuple[Path, str]] = []
    for index, row in enumerate(rows):
        task_id = str(row.get("id") or f"task-{index:04d}")
        task_path = task_dir / f"{task_id}.json"
        task_path.write_text(
            json.dumps({"train": row["train"], "test": row["test"]}),
            encoding="utf-8",
        )
        task_paths.append((task_path, task_id))

    generator = ArcPuzzleGenerator(
        dataset_dir=task_dir,
        output_dir=rendered_dir,
        cell_size=32,
        prompt=(
            "Each row contains input and output grids. Learn the pattern and "
            "generate the output grid for the last input while keeping existing "
            "patterns without modification."
        ),
        seed=0,
    )
    records = [
        generator.create_puzzle(task_path=task_path, puzzle_id=task_id).to_dict()
        for task_path, task_id in task_paths
    ]
    return records, temporary


def parse_grid(text: str) -> list[list[int]]:
    # arcagiVisualKit extracts the last bracketed grid from free-form output;
    # it accepts Python-style lists (single quotes, trailing commas), not only
    # strict JSON.
    compact = text.replace("```json", "").replace("```python", "").replace("```", "")
    # Find balanced list expressions rather than relying on a regex. This
    # handles nested rows, whitespace, Python literals, and multiple candidate
    # grids in a reasoning trace.
    candidates: list[str] = []
    starts = [m.start() for m in re.finditer(r"\[\[", compact)]
    for start in starts:
        depth = 0
        in_string: str | None = None
        escaped = False
        for index in range(start, len(compact)):
            char = compact[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
                continue
            if char in "'\"":
                in_string = char
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(compact[start : index + 1])
                    break
    value = None
    for candidate in reversed(candidates):
        try:
            value = ast.literal_eval(candidate)
            break
        except (SyntaxError, ValueError):
            continue
    if value is None:
        value = extract_json(text)
    if isinstance(value, dict):
        for key in ("grid", "output", "answer", "prediction"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        raise ValueError("prediction is not a 2D array")
    grid = [[int(cell) for cell in row] for row in value]
    if any(cell < 0 or cell > 9 for row in grid for cell in row):
        raise ValueError("grid contains a value outside 0..9")
    return grid


ARC_PROMPT_SUFFIX = (
    "\nColor palette: 0: black, 1: blue, 2: red, 3: green, 4: yellow, "
    "5: gray, 6: magenta, 7: orange, 8: cyan, 9: brown. "
    "Output a row-major 2d array representing the output grid, with each "
    "element an integer from 0 to 9."
)


def main() -> None:
    args = parse_args()
    temporary = None
    if args.data:
        if not args.task_root:
            raise SystemExit("--task-root is required with --data")
        records = load_records(args.data)
        root = Path(args.task_root)
    else:
        records, temporary = load_cached_arcagi()
        root = Path(temporary.name) / "rendered"
    if args.limit != -1:
        records = records[:args.limit]
    runner = VllmMultimodalRunner(
        args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_connections=args.max_connections,
    )
    correct = cell_correct = cell_total = invalid = errors = 0
    perfect = 0
    details = []
    for index, record in enumerate(records):
        try:
            image = resolve_image(record["image"], root)
            prompt = f"{record.get('prompt', '')}{ARC_PROMPT_SUFFIX}"
            prediction = runner.predict(prompt, image)
            if prediction.error:
                errors += 1
                details.append({"index": index, "error": prediction.error})
                continue
            try:
                predicted = parse_grid(prediction.response)
            except Exception as exc:
                invalid += 1
                details.append(
                    {"index": index, "error": f"{type(exc).__name__}: {exc}",
                     "response": prediction.response}
                )
                continue
            expected = record["test_output"]
            rows = len(expected)
            cols = len(expected[0]) if rows else 0
            # Match the kit's renderer: only cells that fit in the expected
            # output region are drawn; absent cells remain blank/incorrect.
            exact = len(predicted) == rows and all(len(row) == cols for row in predicted) and predicted == expected
            correct += int(exact)
            for r, erow in enumerate(expected):
                for c, target in enumerate(erow):
                    cell_total += 1
                    actual = predicted[r][c] if r < len(predicted) and c < len(predicted[r]) else 10
                    cell_correct += int(actual == target)
            perfect += int(exact)
            details.append(
                {
                    "index": index,
                    "task_id": record.get("task_id", record.get("id")),
                    "correct": exact,
                    "predicted_grid": predicted,
                }
            )
        except Exception as exc:
            invalid += 1
            details.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
    total = len(records)
    metrics = {
        "benchmark": "arcagi_visual",
        "num_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "perfect_puzzle_rate": perfect / total if total else 0.0,
        "puzzle_average_accuracy": cell_correct / cell_total if cell_total else 0.0,
        "attempts_total": total,
        "attempts_perfect": perfect,
        "attempts_average_correctness": perfect / total if total else 0.0,
        "puzzles_total": total,
        "puzzles_with_any_perfect_attempt": perfect,
        "puzzle_success_rate": perfect / total if total else 0.0,
        "cell_accuracy": cell_correct / cell_total if cell_total else 0.0,
        "invalid": invalid,
        "inference_errors": errors,
        "details": details,
    }
    write_metrics(args.json_output_file, metrics)
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
