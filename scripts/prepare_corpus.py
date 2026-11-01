"""Build train/valid/test splits from Wikipedia-VI or a local sample file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.config import load_config
from minigpt.data.corpus import (
    download_wikipedia_vi,
    load_local_corpus,
    split_corpus,
    write_splits,
)
from minigpt.utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--source", choices=["sample", "local", "wikipedia"], default="sample")
    p.add_argument("--local-path", default=None, help="Override path for --source local")
    p.add_argument("--max-articles", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    data = cfg["data"]
    min_chars = int(data.get("min_chars", 80))

    if args.source == "sample":
        docs = load_local_corpus(data["sample_path"], min_chars=min(min_chars, 40))
    elif args.source == "local":
        path = args.local_path or data["sample_path"]
        docs = load_local_corpus(path, min_chars=min_chars)
    else:
        docs = download_wikipedia_vi(
            dataset_name=data["hf_dataset"],
            config_name=data["hf_config"],
            max_articles=args.max_articles or int(data["max_articles"]),
            min_chars=min_chars,
            cache_dir=data.get("raw_dir"),
        )

    train, valid, test = split_corpus(
        docs,
        train_ratio=float(data["train_ratio"]),
        valid_ratio=float(data["valid_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    paths = write_splits(train, valid, test, data["processed_dir"])
    stats = {
        "source": args.source,
        "n_docs": len(docs),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "chars_train": sum(len(x) for x in train),
        "files": {k: str(v) for k, v in paths.items()},
    }
    save_json(stats, Path(data["processed_dir"]) / "split_stats.json")
    print(stats)


if __name__ == "__main__":
    main()
