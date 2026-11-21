from local_llm.tokenizer import CharTokenizer


def test_round_trip() -> None:
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"


def test_vocab_is_distinct_sorted_chars() -> None:
    tok = CharTokenizer.from_text("abca")
    assert tok.chars == ["a", "b", "c"]
    assert tok.vocab_size == 3
