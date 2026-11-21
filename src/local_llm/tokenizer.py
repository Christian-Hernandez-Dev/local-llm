"""Turn text into numbers (tokens) and back.

A model can only do math on numbers, so every LLM starts with a tokenizer.
This one is the simplest kind: every distinct character is one token.
"""


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self._to_id = {ch: i for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build the vocabulary from every distinct character in the text."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self._to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)
