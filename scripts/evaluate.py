"""Compute test perplexity for n-gram / LSTM / Mini-GPT checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.config import load_config
from minigpt.data.dataset import LMSequenceDataset, load_corpus_ids
from minigpt.data.tokenizer import ViTokenizer
from minigpt.eval.perplexity import perplexity_neural, perplexity_ngram
from minigpt.models.ngram import NGramLM
from minigpt.train.checkpoint import load_checkpoint
from minigpt.utils import get_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--out", default=None, help="JSON path for metrics")
    return p.parse_args()


def _default_ckpt(cfg: dict) -> Path | None:
    name = cfg["model"]["name"]
    size = cfg.get("experiment", {}).get("size_name")
    run = f"{name}_{size}" if size else name
    root = Path(cfg.get("checkpoint_dir", "checkpoints")) / run
    if name == "ngram":
        cand = root / "best.json"
        return cand if cand.is_file() else None
    cand = root / "best.pt"
    return cand if cand.is_file() else None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    tok = ViTokenizer.load(Path(cfg["data"]["tokenizer_path"]) / "tokenizer.json")
    split_file = cfg["data"]["test_file"] if args.split == "test" else cfg["data"]["valid_file"]
    ids = load_corpus_ids(split_file, tok)
    ckpt = args.ckpt or _default_ckpt(cfg)
    if ckpt is None:
        raise SystemExit(
            "No checkpoint. Train the model first, then re-run evaluate.\n"
            "Neural: checkpoints/<run>/best.pt   n-gram: checkpoints/ngram/best.json"
        )

    model, extra = load_checkpoint(ckpt, device=device)
    if isinstance(model, NGramLM):
        metrics = perplexity_ngram(model, ids)
    else:
        seq_len = int(cfg["data"].get("max_seq_len", 256))
        if hasattr(model, "cfg") and hasattr(model.cfg, "n_positions"):
            seq_len = min(seq_len, int(model.cfg.n_positions))
        ds = LMSequenceDataset(ids, seq_len)
        loader = DataLoader(ds, batch_size=int(cfg.get("eval", {}).get("batch_size", 16)))
        metrics = perplexity_neural(model, loader, device)

    metrics.update(
        {
            "split": args.split,
            "ckpt": str(ckpt),
            "model": cfg["model"]["name"],
            "n_docs_tokens": len(ids),
        }
    )
    print(metrics)
    out = Path(args.out) if args.out else Path("outputs") / f"ppl_{cfg['model']['name']}_{args.split}.json"
    save_json(metrics, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
