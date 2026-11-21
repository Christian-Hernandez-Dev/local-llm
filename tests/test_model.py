from pathlib import Path

import torch

from local_llm.model import BigramModel
from local_llm.train import TrainConfig, get_batch, load, train


def test_forward_shapes_and_loss() -> None:
    model = BigramModel(vocab_size=10)
    x = torch.randint(10, (4, 8))
    logits, loss = model(x, x)
    assert logits.shape == (4, 8, 10)
    assert loss is not None and loss.item() > 0


def test_generate_appends_tokens() -> None:
    model = BigramModel(vocab_size=10)
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=5)
    assert out.shape == (1, 6)


def test_targets_are_inputs_shifted_by_one() -> None:
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, context_len=8, device="cpu")
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_training_lowers_loss_and_checkpoint_loads(tmp_path: Path) -> None:
    text = "abcabcabcabc" * 200  # trivially predictable: after a comes b
    ckpt = tmp_path / "m.pt"
    loss = train(TrainConfig(steps=300, eval_every=300), text=text, checkpoint=ckpt)
    assert loss < 0.5  # random guessing over 3 chars would be ~1.1
    model, tok = load(ckpt)
    assert tok.chars == ["a", "b", "c"]
    assert isinstance(model, BigramModel)
