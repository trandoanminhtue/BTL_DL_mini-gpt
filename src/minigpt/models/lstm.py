from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LSTMConfig:
    vocab_size: int = 16000
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    tie_weights: bool = True


class LSTMLanguageModel(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.proj = (
            nn.Linear(cfg.hidden_dim, cfg.embed_dim, bias=False)
            if cfg.hidden_dim != cfg.embed_dim
            else None
        )
        self.lm_head = nn.Linear(cfg.embed_dim if self.proj is not None else cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            if cfg.hidden_dim != cfg.embed_dim and self.proj is None:
                raise ValueError("tie_weights requires hidden_dim == embed_dim or a projection layer")
            if self.proj is None:
                self.lm_head.weight = self.embed.weight
            else:
                # tie to embedding after projecting hidden -> embed
                self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | tuple]:
        x = self.drop(self.embed(input_ids))
        y, hidden = self.lstm(x, hidden)
        y = self.drop(y)
        if self.proj is not None:
            y = self.proj(y)
        logits = self.lm_head(y)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss, "hidden": hidden}

    @torch.no_grad()
    def next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids)["logits"][:, -1, :]
