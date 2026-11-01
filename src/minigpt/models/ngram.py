from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from minigpt.utils import load_json, save_json


class NGramLM:
    """Count-based n-gram LM with Laplace or linear interpolation smoothing.

    Operates on the same BPE token ids as neural models so perplexity is comparable.
    """

    def __init__(
        self,
        order: int = 3,
        vocab_size: int = 16000,
        smoothing: str = "interpolate",
        laplace_k: float = 1.0,
        lambdas: list[float] | None = None,
        unk_id: int = 1,
        pad_id: int = 0,
    ):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.laplace_k = laplace_k
        self.lambdas = lambdas or self._default_lambdas(order)
        if abs(sum(self.lambdas) - 1.0) > 1e-6:
            raise ValueError("lambdas must sum to 1")
        if len(self.lambdas) != order:
            raise ValueError("len(lambdas) must equal n-gram order")
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.counts: list[Counter] = [Counter() for _ in range(order)]
        self.context_totals: list[Counter] = [Counter() for _ in range(order)]
        self.unigram_total = 0
        self._fitted = False

    @staticmethod
    def _default_lambdas(order: int) -> list[float]:
        raw = [1.0 / (i + 1) for i in range(order)]
        s = sum(raw)
        return [x / s for x in raw]

    def fit(self, token_ids: list[int]) -> "NGramLM":
        tokens = [t for t in token_ids if t != self.pad_id]
        n = len(tokens)
        for i, tok in enumerate(tokens):
            self.counts[0][tok] += 1
            self.unigram_total += 1
            for order in range(2, self.order + 1):
                if i - order + 1 < 0:
                    continue
                ctx = tuple(tokens[i - order + 1 : i])
                gram = ctx + (tok,)
                idx = order - 1
                self.counts[idx][gram] += 1
                self.context_totals[idx][ctx] += 1
        self._fitted = True
        return self

    def _prob_ngram(self, context: tuple[int, ...], token: int, order: int) -> float:
        if order == 1:
            num = self.counts[0][token] + self.laplace_k
            den = self.unigram_total + self.laplace_k * self.vocab_size
            return num / max(den, 1e-12)
        ctx = context[-(order - 1) :]
        gram = ctx + (token,)
        idx = order - 1
        if self.smoothing == "laplace":
            num = self.counts[idx][gram] + self.laplace_k
            den = self.context_totals[idx][ctx] + self.laplace_k * self.vocab_size
            return num / max(den, 1e-12)
        # MLE for this order; interpolation mixes orders outside
        den = self.context_totals[idx][ctx]
        if den == 0:
            return 0.0
        return self.counts[idx][gram] / den

    def log_prob(self, context: list[int], token: int) -> float:
        if self.smoothing == "laplace":
            p = self._prob_ngram(tuple(context), token, self.order)
            return float(np.log(max(p, 1e-12)))
        p = 0.0
        for k, lam in enumerate(self.lambdas, start=1):
            p += lam * self._prob_ngram(tuple(context), token, k)
        return float(np.log(max(p, 1e-12)))

    def sequence_nll(self, token_ids: list[int]) -> tuple[float, int]:
        tokens = [t for t in token_ids if t != self.pad_id]
        nll = 0.0
        count = 0
        for i in range(1, len(tokens)):
            ctx = tokens[max(0, i - (self.order - 1)) : i]
            nll -= self.log_prob(ctx, tokens[i])
            count += 1
        return nll, count

    def next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return log-probs over vocab for the last position of each batch row."""
        logs = []
        for row in input_ids.detach().cpu().tolist():
            row = [t for t in row if t != self.pad_id]
            ctx = row[-(self.order - 1) :] if self.order > 1 else []
            lp = np.array([self.log_prob(ctx, tok) for tok in range(self.vocab_size)], dtype=np.float32)
            logs.append(lp)
        return torch.tensor(np.stack(logs), device=input_ids.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Slow teacher-forced NLL for a batch — used only for eval of short batches
        b, t = input_ids.shape
        nlls = []
        for i in range(b):
            ids = input_ids[i].detach().cpu().tolist()
            if labels is None:
                nll, n = self.sequence_nll(ids)
            else:
                lab = labels[i].detach().cpu().tolist()
                tokens = ids + [lab[-1]]
                nll, n = self.sequence_nll(tokens)
            nlls.append(nll / max(n, 1))
        loss = torch.tensor(nlls, device=input_ids.device).mean()
        logits = self.next_logits(input_ids)
        logits = logits.unsqueeze(1).expand(b, t, self.vocab_size)
        return {"logits": logits, "loss": loss}

    def save(self, path: str | Path) -> None:
        payload = {
            "order": self.order,
            "vocab_size": self.vocab_size,
            "smoothing": self.smoothing,
            "laplace_k": self.laplace_k,
            "lambdas": self.lambdas,
            "unk_id": self.unk_id,
            "pad_id": self.pad_id,
            "unigram_total": self.unigram_total,
            "counts": [
                { _key_to_str(k): v for k, v in c.items()} for c in self.counts
            ],
            "context_totals": [
                { _key_to_str(k): v for k, v in c.items()} for c in self.context_totals
            ],
        }
        save_json(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "NGramLM":
        payload = load_json(path)
        model = cls(
            order=payload["order"],
            vocab_size=payload["vocab_size"],
            smoothing=payload["smoothing"],
            laplace_k=payload["laplace_k"],
            lambdas=payload["lambdas"],
            unk_id=payload["unk_id"],
            pad_id=payload["pad_id"],
        )
        model.unigram_total = payload["unigram_total"]
        model.counts = [Counter({_str_to_key(k): v for k, v in c.items()}) for c in payload["counts"]]
        model.context_totals = [
            Counter({_str_to_key(k): v for k, v in c.items()}) for c in payload["context_totals"]
        ]
        model._fitted = True
        return model


def _key_to_str(key) -> str:
    if isinstance(key, tuple):
        return "t:" + ",".join(str(x) for x in key)
    return f"i:{int(key)}"


def _str_to_key(s: str):
    if s.startswith("t:"):
        body = s[2:]
        if body == "":
            return ()
        return tuple(int(x) for x in body.split(",") if x != "")
    if s.startswith("i:"):
        return int(s[2:])
    # backward-compatible parse
    if "," in s:
        return tuple(int(x) for x in s.split(",") if x != "")
    if s == "":
        return ()
    return int(s)
