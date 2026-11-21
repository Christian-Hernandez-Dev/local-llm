"""Training: show the model real text and nudge it toward better guesses."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from local_llm.data import load_tiny_shakespeare
from local_llm.model import BigramModel, GPTConfig, LanguageModel, TinyGPT, count_parameters
from local_llm.tokenizer import CharTokenizer

ModelKind = Literal["bigram", "gpt"]
CHECKPOINT_DIR = Path("checkpoints.nosync")


@dataclass
class TrainConfig:
    model: ModelKind = "gpt"
    steps: int = 5000
    batch_size: int = 64  # how many snippets per step
    context_len: int = 128  # how many tokens per snippet (and how far back the model sees)
    learning_rate: float = 1e-3  # how big each nudge is
    eval_every: int = 250
    seed: int = 1337

    @classmethod
    def for_model(cls, model: ModelKind, **overrides: int | float) -> "TrainConfig":
        """Sensible defaults per model. The bigram only sees 1 token, so a tiny
        context and a bigger learning rate suit it."""
        base = cls(model="bigram", batch_size=32, context_len=8, learning_rate=1e-2)
        cfg = base if model == "bigram" else cls(model="gpt")
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg


def checkpoint_path(model: ModelKind) -> Path:
    return CHECKPOINT_DIR / f"{model}.pt"


def pick_device() -> str:
    """Use the Mac GPU (MPS) or an NVIDIA GPU (CUDA) if present, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_model(kind: ModelKind, vocab_size: int, context_len: int) -> LanguageModel:
    if kind == "bigram":
        return BigramModel(vocab_size)
    return TinyGPT(GPTConfig(vocab_size=vocab_size, context_len=context_len))


def get_batch(
    data: torch.Tensor, batch_size: int, context_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grab random snippets. Targets are the same snippets shifted one to the left,
    so at every position the 'answer' is simply the next token."""
    starts = torch.randint(len(data) - context_len - 1, (batch_size,))
    x = torch.stack([data[s : s + context_len] for s in starts])
    y = torch.stack([data[s + 1 : s + context_len + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: LanguageModel, data: torch.Tensor, cfg: TrainConfig, device: str, batches: int = 50
) -> float:
    model.eval()  # turns dropout off so we measure the real model
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, cfg.batch_size, cfg.context_len, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: TrainConfig, text: str | None = None, checkpoint: Path | None = None) -> float:
    """Train a model, save it, and return the final validation loss."""
    torch.manual_seed(cfg.seed)
    device = pick_device()
    checkpoint = checkpoint or checkpoint_path(cfg.model)
    text = text if text is not None else load_tiny_shakespeare()
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Hold back 10% of the text the model never trains on. If loss on this
    # "validation" text drops too, the model is learning patterns, not memorizing.
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = build_model(cfg.model, tokenizer.vocab_size, cfg.context_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    print(
        f"model={cfg.model}  params={count_parameters(model):,}  device={device}  "
        f"vocab={tokenizer.vocab_size}  text={len(text):,} chars"
    )

    val = float("nan")
    started = time.perf_counter()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            val = estimate_loss(model, val_data, cfg, device)
            print(f"step {step:5d}  val loss {val:.3f}  ({time.perf_counter() - started:.0f}s)")
        if step == cfg.steps:
            break
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_len, device)
        _, loss = model(x, y)  # 1. guess and measure how wrong
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()  # 2. work out which way to nudge every number
        optimizer.step()  # 3. nudge them

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": cfg.model,
            "context_len": cfg.context_len,
            "chars": tokenizer.chars,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    print(f"saved {checkpoint}")
    return val


def load(checkpoint: Path) -> tuple[LanguageModel, CharTokenizer]:
    saved = torch.load(checkpoint, weights_only=True)
    tokenizer = CharTokenizer(saved["chars"])
    model = build_model(saved["kind"], tokenizer.vocab_size, saved["context_len"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, tokenizer
