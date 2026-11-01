from __future__ import annotations

from pathlib import Path

from minigpt.config import load_config
from minigpt.models.factory import build_model
from minigpt.models.gpt import MiniGPT


def test_yaml_inheritance():
    cfg = load_config(Path("configs/gpt_small.yaml"))
    assert cfg["model"]["name"] == "gpt"
    assert cfg["model"]["n_embd"] == 384
    assert "tokenizer" in cfg
    model = build_model(cfg, vocab_size=128)
    assert isinstance(model, MiniGPT)
    assert model.cfg.vocab_size == 128
