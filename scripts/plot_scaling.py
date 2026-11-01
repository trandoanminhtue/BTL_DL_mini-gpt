"""Plot validation loss vs parameter count and training curves (week 5)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.utils import ensure_dir


def _load_history(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        nargs="+",
        default=[
            "checkpoints/gpt_tiny/history.json",
            "checkpoints/gpt_small/history.json",
            "checkpoints/gpt_medium/history.json",
        ],
    )
    p.add_argument("--out-dir", default="outputs/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    series = []
    missing = []
    for p in args.runs:
        path = Path(p)
        if not path.is_file():
            missing.append(str(path))
            continue
        hist = _load_history(path)
        series.append((path.parent.name, hist))

    if missing:
        print("Missing histories (train those sizes first):")
        for m in missing:
            print(" ", m)
    if not series:
        raise SystemExit("No history.json files found. Train GPT tiny/small/medium first.")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, hist in series:
        xs = [r.get("epoch", i + 1) for i, r in enumerate(hist) if "val_loss" in r]
        ys = [r["val_loss"] for r in hist if "val_loss" in r]
        if xs:
            ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Mini-GPT training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "val_loss_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xs, ys, labels = [], [], []
    for name, hist in series:
        last = [r for r in hist if "val_loss" in r]
        if not last:
            continue
        best = min(last, key=lambda r: r["val_loss"])
        n_params = best.get("n_params")
        if n_params is None:
            continue
        xs.append(n_params / 1e6)
        ys.append(best["val_loss"])
        labels.append(name)
    if xs:
        ax.plot(xs, ys, marker="o")
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6))
        ax.set_xlabel("Parameters (millions)")
        ax.set_ylabel("Best validation loss")
        ax.set_title("Scaling: model size vs. loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "scaling_params_vs_loss.png", dpi=160)
    plt.close(fig)
    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
