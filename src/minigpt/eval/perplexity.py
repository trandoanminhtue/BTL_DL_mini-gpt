from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from minigpt.models.ngram import NGramLM


@torch.no_grad()
def perplexity_neural(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    nll_sum = 0.0
    token_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, labels=labels)
        # cross-entropy is already mean over non-ignored tokens in a batch;
        # approximate corpus NLL by weighting with label count.
        n_tokens = int((labels != -100).sum().item())
        nll_sum += float(out["loss"].item()) * max(n_tokens, 1)
        token_count += max(n_tokens, 1)
    mean_nll = nll_sum / max(token_count, 1)
    return {
        "nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20)),
        "tokens": token_count,
    }


def perplexity_ngram(model: NGramLM, token_ids: list[int]) -> dict[str, float]:
    nll, n = model.sequence_nll(token_ids)
    mean = nll / max(n, 1)
    return {
        "nll": mean,
        "perplexity": math.exp(min(mean, 20)),
        "tokens": n,
    }
