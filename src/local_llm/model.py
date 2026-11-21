"""The models: functions that predict the next token.

Both models share one interface:
    forward(tokens, targets) -> (logits, loss)
    generate(tokens, max_new_tokens) -> tokens

BigramModel (slice 1) looks only at the *current* token.
TinyGPT (slice 2) is a transformer: every token can look back at *all*
earlier tokens through self-attention. Same shape as GPT/Claude, just tiny.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    """Shared base: subclasses define forward(); generation works the same for all."""

    context_len: int  # the most tokens the model can look back at

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """Predict a token, append it, repeat. This loop is how every LLM 'writes'.

        temperature < 1 makes it pick safer, likelier tokens; > 1 makes it wilder.
        """
        for _ in range(max_new_tokens):
            window = tokens[:, -self.context_len :]  # the model can only see so far back
            logits, _ = self(window)
            last = logits[:, -1, :] / temperature  # scores for the token after the last one
            probs = F.softmax(last, dim=-1)  # scores -> probabilities summing to 1
            next_token = torch.multinomial(probs, num_samples=1)  # roll weighted dice
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy: how surprised the model was by the real next tokens. Training lowers it."""
    batch, time, vocab = logits.shape
    return F.cross_entropy(logits.view(batch * time, vocab), targets.view(batch * time))


# ---------------------------------------------------------------- slice 1: bigram


class BigramModel(LanguageModel):
    """A vocab x vocab lookup table: row = current token, columns = next-token scores."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.context_len = 1
        self.next_token_scores = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.next_token_scores(tokens)
        return logits, None if targets is None else next_token_loss(logits, targets)


# ---------------------------------------------------------------- slice 2: tiny GPT


@dataclass
class GPTConfig:
    vocab_size: int
    context_len: int = 128  # how many tokens back the model can see
    embed_dim: int = 192  # size of each token's vector ("meaning" + "position")
    num_heads: int = 6  # parallel attention heads, each can learn a different pattern
    num_layers: int = 4  # stacked transformer blocks
    dropout: float = 0.1  # randomly zero 10% of signals in training, to reduce memorizing


class SelfAttention(nn.Module):
    """Each token asks a question (query), every earlier token offers a label (key)
    and some information (value). Good query/key matches pass more value along.

    That's how "the" can pick up on which noun came before, or ":" can learn
    that a character name sits just before it.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0
        self.num_heads = cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=False)
        self.out = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        # Causal mask: position t may only look at positions <= t (no peeking at the answer).
        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, dim = x.shape
        head_dim = dim // self.num_heads
        q, k, v = self.qkv(x).split(dim, dim=2)
        # (batch, time, dim) -> (batch, heads, time, head_dim): each head works separately
        q, k, v = (t.view(batch, time, self.num_heads, head_dim).transpose(1, 2) for t in (q, k, v))

        scores = (q @ k.transpose(-2, -1)) / head_dim**0.5  # how well each query matches each key
        mask: torch.Tensor = self.causal_mask  # type: ignore[assignment]
        scores = scores.masked_fill(~mask[:time, :time], float("-inf"))  # hide the future
        weights = self.dropout(F.softmax(scores, dim=-1))  # each row sums to 1
        mixed = weights @ v  # weighted blend of earlier tokens' information

        mixed = mixed.transpose(1, 2).reshape(batch, time, dim)  # glue heads back together
        return self.dropout(self.out(mixed))


class FeedForward(nn.Module):
    """After tokens share information (attention), each one 'thinks' about it on its own."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (feed-forward).

    The `x + ...` "residual" connections let each block add a small refinement
    instead of rewriting everything, which keeps deep stacks trainable.
    LayerNorm keeps the numbers in a stable range.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = SelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class TinyGPT(LanguageModel):
    """tokens -> embeddings (+ position) -> N blocks -> scores for the next token."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.context_len = cfg.context_len
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)  # what the token is
        self.position_embed = nn.Embedding(cfg.context_len, cfg.embed_dim)  # where it sits
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.to_logits = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = tokens.shape
        positions = torch.arange(time, device=tokens.device)
        x = self.dropout(self.token_embed(tokens) + self.position_embed(positions))
        x = self.final_norm(self.blocks(x))
        logits = self.to_logits(x)
        return logits, None if targets is None else next_token_loss(logits, targets)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
