from __future__ import annotations

import random
import re
from pathlib import Path

from minigpt.utils import ensure_dir, write_text_lines


_WS = re.compile(r"\s+")
_ARTICLE_SPLIT = "\n\n"


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    text = _WS.sub(" ", text).strip()
    return text


def split_paragraphs(text: str, min_chars: int = 80) -> list[str]:
    parts = re.split(r"\n{2,}", text)
    out = []
    for p in parts:
        p = normalize_text(p)
        if len(p) >= min_chars:
            out.append(p)
    return out


def split_corpus(
    documents: list[str],
    train_ratio: float = 0.90,
    valid_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    docs = documents[:]
    rng.shuffle(docs)
    n = len(docs)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = docs[:n_train]
    valid = docs[n_train : n_train + n_valid]
    test = docs[n_train + n_valid :]
    if not test and docs:
        test = valid[-max(1, len(valid) // 5) :] or train[-1:]
    return train, valid, test


def write_splits(
    train: list[str],
    valid: list[str],
    test: list[str],
    out_dir: str | Path,
) -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)
    paths = {
        "train": out_dir / "train.txt",
        "valid": out_dir / "valid.txt",
        "test": out_dir / "test.txt",
    }
    write_text_lines(train, paths["train"])
    write_text_lines(valid, paths["valid"])
    write_text_lines(test, paths["test"])
    return paths


def load_local_corpus(path: str | Path, min_chars: int = 80) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    docs = split_paragraphs(text, min_chars=min_chars)
    if not docs:
        # fall back to sentence-ish lines
        docs = [normalize_text(line) for line in text.splitlines() if len(line.strip()) >= 20]
    return docs


def download_wikipedia_vi(
    dataset_name: str,
    config_name: str,
    max_articles: int,
    min_chars: int,
    cache_dir: str | Path | None = None,
) -> list[str]:
    """Download a Vietnamese Wikipedia subset via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name,
        config_name,
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    docs: list[str] = []
    for i, row in enumerate(ds):
        if i >= max_articles:
            break
        text = row.get("text") or row.get("content") or ""
        docs.extend(split_paragraphs(str(text), min_chars=min_chars))
    return docs
