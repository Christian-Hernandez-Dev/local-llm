from pathlib import Path

import pytest
import torch

from local_llm.model import BigramModel, TinyGPT
from local_llm.train import TrainConfig, get_batch, load, train

# Trivially predictable text: after a comes b, after b comes c...
TEXT = "abcabcabcabc" * 200


def test_targets_are_inputs_shifted_by_one() -> None:
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, context_len=8, device="cpu")
    assert torch.equal(x[:, 1:], y[:, :-1])


@pytest.mark.parametrize(("kind", "cls"), [("bigram", BigramModel), ("gpt", TinyGPT)])
def test_training_learns_and_checkpoint_loads(kind, cls, tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    cfg = TrainConfig.for_model(kind, steps=150, eval_every=150, batch_size=16, context_len=16)
    loss = train(cfg, text=TEXT, checkpoint=ckpt)
    assert loss < 0.5  # random guessing over 3 chars would be ~1.1
    model, tok = load(ckpt)
    assert tok.chars == ["a", "b", "c"]
    assert isinstance(model, cls)
