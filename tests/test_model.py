import pytest
import torch

from local_llm.model import BigramModel, GPTConfig, LanguageModel, TinyGPT

VOCAB = 10


def tiny_gpt(context_len: int = 8) -> TinyGPT:
    cfg = GPTConfig(
        vocab_size=VOCAB,
        context_len=context_len,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    return TinyGPT(cfg).eval()


@pytest.mark.parametrize("model", [BigramModel(VOCAB), tiny_gpt()], ids=["bigram", "gpt"])
def test_forward_shapes_and_loss(model: LanguageModel) -> None:
    x = torch.randint(VOCAB, (4, 8))
    logits, loss = model(x, x)
    assert logits.shape == (4, 8, VOCAB)
    assert loss is not None and loss.item() > 0


@pytest.mark.parametrize("model", [BigramModel(VOCAB), tiny_gpt()], ids=["bigram", "gpt"])
def test_generate_can_run_past_the_context_window(model: LanguageModel) -> None:
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)


def test_gpt_cannot_see_the_future() -> None:
    """Changing a later token must not change predictions at earlier positions.
    If it did, the model could 'cheat' in training by reading the answer."""
    model = tiny_gpt()
    a = torch.randint(VOCAB, (1, 8))
    b = a.clone()
    b[0, 5:] = (b[0, 5:] + 1) % VOCAB  # change only positions 5, 6, 7
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    assert torch.allclose(logits_a[0, :5], logits_b[0, :5], atol=1e-5)
    assert not torch.allclose(logits_a[0, 5:], logits_b[0, 5:])
