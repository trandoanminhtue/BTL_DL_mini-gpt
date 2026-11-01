"""Sweep temperature / top-k / top-p on a fixed prompt (week 5 extension)."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.config import load_config
from minigpt.data.tokenizer import ViTokenizer
from minigpt.generate.sampler import GenerationConfig, generate
from minigpt.train.checkpoint import load_checkpoint
from minigpt.utils import get_device, save_json, set_seed, write_text_lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/gpt_small.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Ngày xửa ngày xưa,")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--out-dir", default="outputs/sampling")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    tok = ViTokenizer.load(Path(cfg["data"]["tokenizer_path"]) / "tokenizer.json")
    model, _ = load_checkpoint(args.ckpt, device=device)
    ids = torch.tensor(
        [[tok.bos_id] + tok.encode(args.prompt, add_special=False)],
        dtype=torch.long,
        device=device,
    )
    temps = [0.0, 0.7, 1.0]
    top_ks = [None, 20, 50]
    top_ps = [None, 0.9]

    rows = []
    texts = []
    for temp, top_k, top_p in itertools.product(temps, top_ks, top_ps):
        greedy = temp <= 0
        gcfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=max(temp, 1e-6),
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            repetition_penalty=1.1,
        )
        out = generate(model, ids, gcfg, eos_id=tok.eos_id, pad_id=tok.pad_id)
        text = tok.decode(out[0].tolist(), skip_special=True)
        rec = {"temperature": temp, "top_k": top_k, "top_p" : top_p, "greedy": greedy, "text": text}
        rows.append(rec)
        texts.append(f"T={temp} k={top_k} p={top_p}\n{text}\n")
        print(rec)

    out_dir = Path(args.out_dir)
    save_json(rows, out_dir / "sampling_sweep.json")
    write_text_lines(texts, out_dir / "sampling_sweep.txt")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
