from __future__ import annotations

from pathlib import Path

from minigpt.utils import load_json, save_json


def merge_metrics(paths: list[str | Path], out: str | Path) -> list[dict]:
    rows = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            rows.append(load_json(path))
    save_json(rows, out)
    return rows
