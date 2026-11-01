"""Train a Vietnamese BPE tokenizer on the prepared corpus."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.config import load_config
from minigpt.data.tokenizer import train_bpe_tokenizer
from minigpt.utils import ensure_dir, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    tcfg = cfg["tokenizer"]
    files = [data["train_file"]]
    extra = Path(data["processed_dir"]) / "valid.txt"
    if extra.is_file():
        files.append(extra)
    tok = train_bpe_tokenizer(
        files=files,
        vocab_size=int(tcfg["vocab_size"]),
        min_frequency=int(tcfg["min_frequency"]),
        special_tokens=list(tcfg["special_tokens"]),
        strip_accents=False,
    )
    out_dir = ensure_dir(data["tokenizer_path"])
    out_file = out_dir / "tokenizer.json"
    tok.save(out_file)
    demo = "Ngày xửa ngày xưa, ở một làng quê Việt Nam có một chàng trai chăm chỉ."
    ids = tok.encode(demo)
    save_json(
        {
            "vocab_size": tok.vocab_size,
            "path": str(out_file),
            "demo_text": demo,
            "demo_ids": ids,
            "demo_decode": tok.decode(ids),
        },
        out_dir / "tokenizer_info.json",
    )
    print(f"Saved tokenizer to {out_file}  vocab={tok.vocab_size}")
    print("encode:", ids[:40], "...")
    print("decode:", tok.decode(ids))


if __name__ == "__main__":
    main()
