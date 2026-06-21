from __future__ import annotations

import subprocess
from typing import Literal

ModelMode = Literal["qwen_only", "emma_only"]


MODEL_BY_MODE = {
    "qwen_only": "qwen2.5:7b",
    "emma_only": "gemma3:4b",
}


def get_loaded_models() -> list[str]:
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
        check=True,
    )

    lines = result.stdout.strip().splitlines()
    if len(lines) <= 1:
        return []

    models = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])

    return models


def stop_model(model_name: str) -> None:
    subprocess.run(
        ["ollama", "stop", model_name],
        capture_output=True,
        text=True,
        check=False,
    )


def stop_all_other_models(target_model: str) -> None:
    loaded = get_loaded_models()

    for current in loaded:
        if current != target_model:
            stop_model(current)


def ensure_model_loaded(model_name: str) -> None:
    stop_all_other_models(model_name)

    subprocess.run(
        ["ollama", "run", model_name, "ping"],
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_mode(mode: ModelMode) -> str:
    model_name = MODEL_BY_MODE[mode]
    ensure_model_loaded(model_name)
    return model_name
