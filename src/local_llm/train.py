"""Training: show the model real text and nudge it toward better guesses."""

from dataclasses import dataclass
from pathlib import Path

import torch

from local_llm.data import load_tiny_shakespeare
from local_llm.model import BigramModel
from local_llm.tokenizer import CharTokenizer

CHECKPOINT = Path("checkpoints.nosync/bigram.pt")


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32  # how many snippets per step
    context_len: int = 8  # how many characters per snippet
    learning_rate: float = 1e-2  # how big each nudge is
    eval_every: int = 500
    seed: int = 1337


def pick_device() -> str:
    """Use the Mac GPU (MPS) or an NVIDIA GPU (CUDA) if present, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(
    data: torch.Tensor, batch_size: int, context_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grab random snippets. Targets are the same snippets shifted one to the left,
    so at every position the 'answer' is simply the next character."""
    starts = torch.randint(len(data) - context_len - 1, (batch_size,))
    x = torch.stack([data[s : s + context_len] for s in starts])
    y = torch.stack([data[s + 1 : s + context_len + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: BigramModel, data: torch.Tensor, cfg: TrainConfig, device: str, batches: int = 50
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, cfg.batch_size, cfg.context_len, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: TrainConfig, text: str | None = None, checkpoint: Path = CHECKPOINT) -> float:
    """Train a model, save it to `checkpoint`, and return the final validation loss."""
    torch.manual_seed(cfg.seed)
    device = pick_device()
    text = text if text is not None else load_tiny_shakespeare()
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Hold back 10% of the text the model never trains on. If loss on this
    # "validation" text drops too, the model is learning patterns, not memorizing.
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = BigramModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    print(f"device={device}  vocab={tokenizer.vocab_size} chars  text={len(text):,} chars")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            val = estimate_loss(model, val_data, cfg, device)
            print(f"step {step:5d}  val loss {val:.3f}")
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_len, device)
        _, loss = model(x, y)  # 1. guess and measure how wrong
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()  # 2. work out which way to nudge every number
        optimizer.step()  # 3. nudge them

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "chars": tokenizer.chars}, checkpoint)
    print(f"saved {checkpoint}")
    return estimate_loss(model, val_data, cfg, device)


def load(checkpoint: Path = CHECKPOINT) -> tuple[BigramModel, CharTokenizer]:
    saved = torch.load(checkpoint, weights_only=True)
    tokenizer = CharTokenizer(saved["chars"])
    model = BigramModel(tokenizer.vocab_size)
    model.load_state_dict(saved["state_dict"])
    return model, tokenizer
