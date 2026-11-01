from __future__ import annotations

import torch

from minigpt.models.gpt import GPTConfig, MiniGPT, estimate_gpt_params
from minigpt.models.lstm import LSTMConfig, LSTMLanguageModel
from minigpt.models.ngram import NGramLM
from minigpt.utils import count_parameters


def test_gpt_forward_and_param_count():
    cfg = GPTConfig(vocab_size=200, n_layer=2, n_head=4, n_embd=32, n_positions=16)
    model = MiniGPT(cfg)
    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    out = model(x, labels=y)
    assert out["logits"].shape == (2, 8, 200)
    assert out["loss"].ndim == 0
    actual = count_parameters(model)
    est = estimate_gpt_params(cfg)
    assert abs(actual - est) / actual < 0.05


def test_lstm_forward():
    cfg = LSTMConfig(vocab_size=100, embed_dim=32, hidden_dim=32, num_layers=1, dropout=0.0)
    model = LSTMLanguageModel(cfg)
    x = torch.randint(0, 100, (2, 10))
    out = model(x, labels=x)
    assert out["logits"].shape == (2, 10, 100)
    assert torch.isfinite(out["loss"])


def test_ngram_fit_and_ppl():
    ids = [1, 2, 3, 2, 3, 4, 1, 2, 3]
    lm = NGramLM(order=3, vocab_size=10, smoothing="interpolate")
    lm.fit(ids)
    nll, n = lm.sequence_nll(ids)
    assert n > 0
    assert nll > 0
    logits = lm.next_logits(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 10)
