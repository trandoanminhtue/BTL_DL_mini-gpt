from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from minigpt.models.gpt import GPTConfig, MiniGPT
from minigpt.models.lstm import LSTMConfig, LSTMLanguageModel
from minigpt.models.ngram import NGramLM


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, NGramLM):
        model.save(path.with_suffix(".json"))
        return
    payload: dict[str, Any] = {
        "model_type": type(model).__name__,
        "model_config": model.cfg.__dict__,
        "model_state": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    path = Path(path)
    if path.suffix == ".json":
        return NGramLM.load(path), {}
    payload = torch.load(path, map_location=device, weights_only=False)
    mtype = payload["model_type"]
    cfg_dict = payload["model_config"]
    if mtype == "MiniGPT":
        model = MiniGPT(GPTConfig(**cfg_dict))
    elif mtype == "LSTMLanguageModel":
        model = LSTMLanguageModel(LSTMConfig(**cfg_dict))
    else:
        raise ValueError(f"Unknown model_type {mtype}")
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload.get("extra", {})
