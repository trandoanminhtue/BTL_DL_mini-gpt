"""Week-1 environment smoke test."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    try:
        import torch

        print(f"torch {torch.__version__}")
        print(f"cuda  {torch.cuda.is_available()}  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    except ImportError:
        print("torch: NOT INSTALLED")
    for pkg in ("tokenizers", "yaml", "numpy", "tqdm", "datasets"):
        try:
            mod = __import__(pkg if pkg != "yaml" else "yaml")
            print(f"{pkg:12s} OK  {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"{pkg:12s} MISSING")
    from minigpt.models.gpt import GPTConfig, MiniGPT, estimate_gpt_params
    from minigpt.utils import count_parameters

    for name, kwargs in {
        "tiny": dict(n_layer=4, n_head=4, n_embd=256, n_positions=256, vocab_size=16000),
        "small": dict(n_layer=8, n_head=6, n_embd=384, n_positions=256, vocab_size=16000),
        "medium": dict(n_layer=6, n_head=8, n_embd=512, n_positions=256, vocab_size=16000),
    }.items():
        cfg = GPTConfig(**kwargs)
        est = estimate_gpt_params(cfg)
        model = MiniGPT(cfg)
        real = count_parameters(model)
        print(f"GPT-{name:6s}  estimate={est/1e6:.2f}M  actual={real/1e6:.2f}M")
    print("Environment check finished.")


if __name__ == "__main__":
    main()
