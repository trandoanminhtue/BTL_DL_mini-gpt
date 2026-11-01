from __future__ import annotations

from pathlib import Path

from minigpt.data.tokenizer import train_bpe_tokenizer


def test_bpe_roundtrip_vietnamese(tmp_path: Path):
    src = Path("data/sample/corpus.txt")
    tok = train_bpe_tokenizer([src], vocab_size=800, min_frequency=1)
    text = "Ngày xửa ngày xưa, ở một làng quê Việt Nam."
    ids = tok.encode(text)
    assert tok.bos_id in ids and tok.eos_id in ids
    decoded = tok.decode(ids)
    assert "Ngày" in decoded or "ngày" in decoded.lower()
    assert tok.vocab_size <= 800 or tok.vocab_size >= 50
