"""Train n-gram, LSTM, or Mini-GPT from a YAML config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.config import load_config, save_config
from minigpt.data.dataset import InstructDataset, LMSequenceDataset, load_corpus_ids
from minigpt.data.tokenizer import ViTokenizer
from minigpt.models.factory import build_model
from minigpt.models.ngram import NGramLM
from minigpt.train.checkpoint import load_checkpoint, save_checkpoint
from minigpt.train.trainer import Trainer
from minigpt.utils import count_parameters, ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--instruct", action="store_true", help="SFT on instruction JSONL")
    p.add_argument("--init-ckpt", default=None, help="Warm-start neural weights")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    data = cfg["data"]
    tok = ViTokenizer.load(Path(data["tokenizer_path"]) / "tokenizer.json")
    cfg["model"]["vocab_size"] = tok.vocab_size
    name = cfg["model"]["name"]
    run_name = args.run_name or name
    if cfg.get("experiment", {}).get("size_name"):
        run_name = f"{name}_{cfg['experiment']['size_name']}"
    if args.instruct:
        run_name = f"{run_name}_instruct"
    run_dir = ensure_dir(Path(cfg.get("checkpoint_dir", "checkpoints")) / run_name)
    save_config(cfg, run_dir / "config.yaml")

    seq_len = int(data.get("max_seq_len", 256))
    if name == "gpt":
        seq_len = min(seq_len, int(cfg["model"].get("n_positions", seq_len)))

    if name == "ngram":
        ids = load_corpus_ids(data["train_file"], tok)
        model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id, unk_id=tok.unk_id)
        assert isinstance(model, NGramLM)
        print(f"Fitting n-gram order={model.order} on {len(ids):,} tokens...")
        model.fit(ids)
        save_checkpoint(run_dir / "best.json", model)
        valid_ids = load_corpus_ids(data["valid_file"], tok)
        nll, n = model.sequence_nll(valid_ids)
        mean = nll / max(n, 1)
        import math

        stats = {"val_nll": mean, "val_ppl": math.exp(min(mean, 20)), "tokens": n}
        save_json(stats, run_dir / "history.json")
        print(stats)
        return

    if args.instruct:
        dataset = InstructDataset(
            data["instruct_file"],
            tok,
            max_seq_len=seq_len,
            response_only_loss=bool(cfg.get("instruct", {}).get("response_only_loss", True)),
        )
        valid = dataset
    else:
        train_ids = load_corpus_ids(data["train_file"], tok)
        valid_ids = load_corpus_ids(data["valid_file"], tok)
        dataset = LMSequenceDataset(train_ids, seq_len)
        valid = LMSequenceDataset(valid_ids, seq_len)

    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid,
        batch_size=int(cfg.get("eval", {}).get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=0,
    )

    if args.init_ckpt:
        model, _ = load_checkpoint(args.init_ckpt, device=device)
    elif args.instruct and cfg.get("instruct", {}).get("checkpoint"):
        model, _ = load_checkpoint(cfg["instruct"]["checkpoint"], device=device)
    else:
        model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id, unk_id=tok.unk_id)

    if args.instruct and bool(cfg.get("instruct", {}).get("freeze_embeddings", False)):
        if hasattr(model, "wte"):
            model.wte.weight.requires_grad = False
        elif hasattr(model, "embed"):
            model.embed.weight.requires_grad = False
        print("Froze token embeddings.")

    print(f"Model={type(model).__name__}  params={count_parameters(model):,}  device={device}")
    trainer = Trainer(model, train_loader, valid_loader, cfg, device, run_dir)
    result = trainer.train()
    print("Done.", {k: result[k] for k in ("best_val_loss", "n_params")})


if __name__ == "__main__":
    main()
