"""Small, dependency-light helpers for local vLLM multimodal evaluation.

The evaluators in ``src/eval/tasks`` intentionally use a common JSON metrics
contract instead of Inspect AI. This keeps image benchmarks usable with
run-local model paths and makes them compatible with ptbctl's existing
``--json-output-file`` handling.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="final_model")
    parser.add_argument("--data", required=True, help="Dataset JSON/JSONL file")
    parser.add_argument("--image-root", default="", help="Root for relative image paths")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-connections", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json-output-file")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        values = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        values = json.loads(source.read_text())
        if isinstance(values, dict):
            for key in ("data", "items", "records", "qa", "questions"):
                if isinstance(values.get(key), list):
                    values = values[key]
                    break
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise ValueError(f"Expected a list of JSON objects in {source}")
    return values


def resolve_image(value: Any, root: str | Path = "") -> Path:
    if isinstance(value, dict):
        value = value.get("path") or value.get("file_name") or value.get("filename")
    if not value:
        raise ValueError("record has no image path")
    path = Path(str(value))
    if not path.is_absolute() and root:
        path = Path(root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/webp" if suffix == ".webp" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return "\n".join(extract_text(item) for item in response)
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            return extract_text((choices[0].get("message") or {}).get("content", ""))
        return str(response.get("content", ""))
    return str(response)


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"```(?:\w+)?", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?`\"'")


def extract_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"(\[[\s\S]*\]|\{[\s\S]*\})", text):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    raise ValueError("response does not contain valid JSON")


@dataclass
class Prediction:
    response: str
    error: str | None = None


class VllmMultimodalRunner:
    """Run image+text prompts through vLLM's offline chat interface."""

    def __init__(self, model_path: str, *, max_tokens: int, temperature: float,
                 gpu_memory_utilization: float, max_connections: int = 1) -> None:
        from vllm import LLM, SamplingParams

        self._sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
        )
        self.max_connections = max_connections

    def predict(self, prompt: str, image: Path) -> Prediction:
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                ],
            }]
            outputs = self._llm.chat(
                messages,
                sampling_params=self._sampling,
                use_tqdm=False,
            )
            text = outputs[0].outputs[0].text if outputs else ""
            return Prediction(response=text)
        except Exception as exc:  # preserve per-sample failures in metrics
            return Prediction(response="", error=f"{type(exc).__name__}: {exc}")


def write_metrics(path: str | None, metrics: dict[str, Any]) -> None:
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def evaluate_records(
    records: Sequence[dict[str, Any]],
    runner: VllmMultimodalRunner,
    *,
    image_root: str | Path,
    prompt_fn,
    score_fn,
) -> dict[str, Any]:
    correct = invalid = errors = 0
    details: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            image = resolve_image(
                record.get("image") or record.get("image_path") or record.get("img") or record.get("imgname"),
                image_root,
            )
            prediction = runner.predict(prompt_fn(record), image)
            if prediction.error:
                errors += 1
                details.append({"index": index, "error": prediction.error})
                continue
            ok = bool(score_fn(record, prediction.response))
            correct += int(ok)
            details.append({"index": index, "correct": ok, "response": prediction.response})
        except Exception as exc:
            invalid += 1
            details.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
    total = len(records)
    metrics = {
        "num_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "invalid": invalid,
        "inference_errors": errors,
        "details": details,
    }
    return metrics
