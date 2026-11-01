from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from minigpt.data.tokenizer import ViTokenizer


def load_corpus_ids(
    path: str | Path,
    tokenizer: ViTokenizer,
    add_special: bool = True,
) -> list[int]:
    ids: list[int] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.extend(tokenizer.encode(line, add_special=add_special))
    return ids


class LMSequenceDataset(Dataset):
    """Packed token stream cut into fixed-length language-modeling windows."""

    def __init__(self, token_ids: list[int], seq_len: int):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.seq_len = seq_len
        tensor = torch.tensor(token_ids, dtype=torch.long)
        n = (len(tensor) - 1) // seq_len
        usable = n * seq_len
        self.inputs = tensor[:usable].view(n, seq_len)
        self.targets = tensor[1 : usable + 1].view(n, seq_len)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.inputs[idx], "labels": self.targets[idx]}


class InstructDataset(Dataset):
    """Prompt–response pairs for light supervised instruction tuning."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: ViTokenizer,
        max_seq_len: int,
        response_only_loss: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.response_only_loss = response_only_loss
        self.samples: list[dict] = []
        with Path(jsonl_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def _format(self, sample: dict) -> tuple[str, str]:
        instruction = sample.get("instruction", "").strip()
        inp = sample.get("input", "").strip()
        response = sample.get("output", sample.get("response", "")).strip()
        if inp:
            prompt = f"Lệnh: {instruction}\nĐầu vào: {inp}\nTrả lời:"
        else:
            prompt = f"Lệnh: {instruction}\nTrả lời:"
        return prompt, response

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt, response = self._format(self.samples[idx])
        prompt_ids = self.tokenizer.encode(prompt, add_special=False)
        resp_ids = self.tokenizer.encode(response, add_special=False)
        ids = [self.tokenizer.bos_id] + prompt_ids + [self.tokenizer.sep_id] + resp_ids + [self.tokenizer.eos_id]
        ids = ids[: self.max_seq_len]
        labels = ids[1:] + [self.tokenizer.pad_id]
        if self.response_only_loss:
            prompt_len = 1 + len(prompt_ids) + 1  # bos + prompt + sep
            ignore = min(prompt_len - 1, len(labels))
            labels = [-100] * ignore + labels[ignore:]
        pad = self.max_seq_len - len(ids)
        if pad > 0:
            ids = ids + [self.tokenizer.pad_id] * pad
            labels = labels + [-100] * pad
        input_ids = torch.tensor(ids[: self.max_seq_len], dtype=torch.long)
        label_ids = torch.tensor(labels[: self.max_seq_len], dtype=torch.long)
        return {"input_ids": input_ids, "labels": label_ids}
