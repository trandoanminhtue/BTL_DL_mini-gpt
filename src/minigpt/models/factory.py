from __future__ import annotations

from typing import Any

from minigpt.models.gpt import GPTConfig, MiniGPT
from minigpt.models.lstm import LSTMConfig, LSTMLanguageModel
from minigpt.models.ngram import NGramLM


def build_model(cfg: dict[str, Any], vocab_size: int, pad_id: int = 0, unk_id: int = 1):
    mcfg = dict(cfg.get("model", {}))
    name = mcfg.pop("name")
    mcfg["vocab_size"] = vocab_size
    if name == "gpt":
        allowed = {k: mcfg[k] for k in GPTConfig.__dataclass_fields__ if k in mcfg}
        return MiniGPT(GPTConfig(**allowed))
    if name == "lstm":
        allowed = {k: mcfg[k] for k in LSTMConfig.__dataclass_fields__ if k in mcfg}
        return LSTMLanguageModel(LSTMConfig(**allowed))
    if name == "ngram":
        return NGramLM(
            order=int(mcfg.get("order", 3)),
            vocab_size=vocab_size,
            smoothing=mcfg.get("smoothing", "interpolate"),
            laplace_k=float(mcfg.get("laplace_k", 1.0)),
            lambdas=mcfg.get("lambdas"),
            pad_id=pad_id,
            unk_id=unk_id,
        )
    raise ValueError(f"Unknown model name: {name}")
