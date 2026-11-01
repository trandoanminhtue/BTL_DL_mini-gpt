from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.9
    repetition_penalty: float = 1.0
    greedy: bool = False


def _apply_repetition_penalty(logits: torch.Tensor, ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    unique = torch.unique(ids)
    scores = logits[unique]
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    logits = logits.clone()
    logits[unique] = scores
    return logits


def _top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k).values[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    cfg: GenerationConfig,
    eos_id: int | None = None,
    pad_id: int = 0,
) -> torch.Tensor:
    """Autoregressive decoding. `model` must implement `next_logits(input_ids)`."""
    model_was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    ids = input_ids
    for _ in range(cfg.max_new_tokens):
        logits = model.next_logits(ids)
        if logits.dim() == 2:
            step_logits = logits[0]
        else:
            step_logits = logits
        step_logits = _apply_repetition_penalty(step_logits, ids[0], cfg.repetition_penalty)
        if cfg.greedy or cfg.temperature <= 0:
            next_id = torch.argmax(step_logits, dim=-1, keepdim=True)
        else:
            step_logits = step_logits / max(cfg.temperature, 1e-6)
            step_logits = _top_k_top_p(step_logits, cfg.top_k, cfg.top_p)
            probs = F.softmax(step_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id.view(1, 1).to(ids.device)], dim=1)
        if eos_id is not None and int(next_id.item()) == eos_id:
            break
    if model_was_training and hasattr(model, "train"):
        model.train()
    return ids
