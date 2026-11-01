from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16000
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    n_positions: int = 256
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


def estimate_gpt_params(cfg: GPTConfig) -> int:
    """Closed-form parameter count (tied embeddings, GPT-2 style)."""
    d, L, V, T = cfg.n_embd, cfg.n_layer, cfg.vocab_size, cfg.n_positions
    embed = V * d + T * d
    # per block: 4 projection matrices of d×d (q,k,v,o) + MLP  d×4d and 4d×d + 2 LayerNorm
    attn = 4 * d * d
    mlp = 2 * d * (4 * d)
    ln = 2 * 2 * d  # weight+bias for 2 LNs
    if not cfg.bias:
        attn = 4 * d * d  # still 4 matrices; bias terms omitted
        ln = 2 * d
        block = attn + mlp + ln
    else:
        attn_bias = 4 * d
        mlp_bias = 4 * d + d
        block = attn + attn_bias + mlp + mlp_bias + ln
    final_ln = 2 * d if cfg.bias else d
    lm_head = 0 if cfg.tie_weights else V * d
    return embed + L * block + final_ln + lm_head


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.n_positions, cfg.n_positions)).view(
                1, 1, cfg.n_positions, cfg.n_positions
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        dropout_p = self.attn_drop.p if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only Transformer (GPT-2 style) implemented from scratch."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t = input_ids.size()
        if t > self.cfg.n_positions:
            raise ValueError(f"sequence length {t} exceeds n_positions {self.cfg.n_positions}")
        pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.wte(input_ids) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        t = input_ids.size(1)
        if t > self.cfg.n_positions:
            input_ids = input_ids[:, -self.cfg.n_positions :]
        return self.forward(input_ids)["logits"][:, -1, :]
