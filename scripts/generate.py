"""Generate Vietnamese text from a prompt (checkpoint can be added later)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.config import load_config
from minigpt.data.tokenizer import ViTokenizer
from minigpt.generate.sampler import GenerationConfig, generate
from minigpt.models.factory import build_model
from minigpt.train.checkpoint import load_checkpoint
from minigpt.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/gpt_small.yaml")
    p.add_argument("--ckpt", default=None, help="Path to best.pt / best.json. Omit with --untrained.")
    p.add_argument("--prompt", default="Ngày xửa ngày xưa,")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--untrained", action="store_true", help="Random-init model (pipeline demo only).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(cfg.get("device", "auto"))
    tok_path = Path(cfg["data"]["tokenizer_path"]) / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(
            f"Tokenizer not found: {tok_path}\n"
            "Run: python scripts/prepare_corpus.py --source sample && python scripts/train_tokenizer.py"
        )
    tok = ViTokenizer.load(tok_path)
    gcfg_raw = cfg.get("generation", {})
    gcfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens or int(gcfg_raw.get("max_new_tokens", 80)),
        temperature=args.temperature if args.temperature is not None else float(gcfg_raw.get("temperature", 0.8)),
        top_k=args.top_k if args.top_k is not None else gcfg_raw.get("top_k", 50),
        top_p=args.top_p if args.top_p is not None else gcfg_raw.get("top_p", 0.9),
        repetition_penalty=float(gcfg_raw.get("repetition_penalty", 1.0)),
        greedy=args.greedy,
    )

    ckpt = args.ckpt
    if ckpt is None and not args.untrained:
        name = cfg["model"]["name"]
        size = cfg.get("experiment", {}).get("size_name")
        run = f"{name}_{size}" if size else name
        default_pt = Path(cfg.get("checkpoint_dir", "checkpoints")) / run / "best.pt"
        default_json = default_pt.with_suffix(".json")
        if default_pt.is_file():
            ckpt = str(default_pt)
        elif default_json.is_file():
            ckpt = str(default_json)
        else:
            raise SystemExit(
                "No checkpoint found. Train first, pass --ckpt, or use --untrained for a dry run.\n"
                f"Looked for: {default_pt}"
            )

    if args.untrained:
        cfg["model"]["vocab_size"] = tok.vocab_size
        model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id, unk_id=tok.unk_id)
        if hasattr(model, "to"):
            model = model.to(device)
            model.eval()
        print("[warn] untrained weights — output is random; replace with a trained checkpoint later.")
    else:
        model, extra = load_checkpoint(ckpt, device=device)
        print(f"Loaded {ckpt}  extra={ {k: extra[k] for k in extra if k != 'history'} }")

    ids = tok.encode(args.prompt, add_special=False)
    input_ids = torch.tensor([[tok.bos_id] + ids], dtype=torch.long, device=device)
    out = generate(model, input_ids, gcfg, eos_id=tok.eos_id, pad_id=tok.pad_id)
    text = tok.decode(out[0].tolist(), skip_special=True)
    print("---")
    print(text)
    print("---")


if __name__ == "__main__":
    main()
