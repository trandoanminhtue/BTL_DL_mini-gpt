from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFC, Sequence, StripAccents

from minigpt.utils import SPECIAL


class ViTokenizer:
    """Thin wrapper around a HuggingFace tokenizers BPE model."""

    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.pad_id = tokenizer.token_to_id(SPECIAL["pad"])
        self.unk_id = tokenizer.token_to_id(SPECIAL["unk"])
        self.bos_id = tokenizer.token_to_id(SPECIAL["bos"])
        self.eos_id = tokenizer.token_to_id(SPECIAL["eos"])
        self.ins_id = tokenizer.token_to_id(SPECIAL["ins"])
        self.sep_id = tokenizer.token_to_id(SPECIAL["sep"])

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = self.tk.encode(text).ids
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id, self.ins_id, self.sep_id}
        if skip_special:
            ids = [i for i in ids if i not in special]
        return self.tk.decode(ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tk.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "ViTokenizer":
        return cls(Tokenizer.from_file(str(path)))


def train_bpe_tokenizer(
    files: list[str | Path],
    vocab_size: int = 16000,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
    strip_accents: bool = False,
) -> ViTokenizer:
    """Train a byte-level BPE tokenizer. Accents are kept by default (Vietnamese)."""
    special_tokens = special_tokens or list(SPECIAL.values())
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL["unk"]))
    # NFC keeps Vietnamese diacritics on the base character (do not NFD-split tones).
    normalizers = [NFC()]
    if strip_accents:
        normalizers.append(StripAccents())
    tokenizer.normalizer = Sequence(normalizers)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train([str(p) for p in files], trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(SPECIAL["pad"]), pad_token=SPECIAL["pad"])
    return ViTokenizer(tokenizer)
