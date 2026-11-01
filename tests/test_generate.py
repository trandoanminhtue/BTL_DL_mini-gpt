from __future__ import annotations

import torch

from minigpt.generate.sampler import GenerationConfig, generate
from minigpt.models.gpt import GPTConfig, MiniGPT


def test_generate_grows_sequence():
    cfg = GPTConfig(vocab_size=50, n_layer=1, n_head=2, n_embd=16, n_positions=32, dropout=0.0)
    model = MiniGPT(cfg)
    model.eval()
    ids = torch.tensor([[1, 2, 3]])
    out = generate(model, ids, GenerationConfig(max_new_tokens=5, greedy=True))
    assert out.shape == (1, 8)
