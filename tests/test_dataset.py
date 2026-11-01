from __future__ import annotations

from minigpt.data.dataset import LMSequenceDataset


def test_packed_windows():
    ids = list(range(20))
    ds = LMSequenceDataset(ids, seq_len=4)
    assert len(ds) == 4
    item = ds[0]
    assert item["input_ids"].tolist() == [0, 1, 2, 3]
    assert item["labels"].tolist() == [1, 2, 3, 4]
