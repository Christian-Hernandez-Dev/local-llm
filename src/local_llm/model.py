"""The model: a function that predicts the next token.

BigramModel is the smallest possible language model. It looks only at the
*current* token and predicts the next one using a big lookup table:
row = current token, column = a score for each possible next token.
Training fills in that table. Later slices swap this for a transformer,
which looks at *all* previous tokens; the interface stays the same.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # vocab_size x vocab_size table of learned scores ("logits").
        self.next_token_scores = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """tokens: (batch, time) ids -> logits: (batch, time, vocab) scores.

        If targets (the true next tokens) are given, also return the loss:
        how surprised the model was by the real answer. Training lowers it.
        """
        logits = self.next_token_scores(tokens)
        if targets is None:
            return logits, None
        batch, time, vocab = logits.shape
        loss = F.cross_entropy(logits.view(batch * time, vocab), targets.view(batch * time))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Predict a token, append it, repeat. This loop is how every LLM 'writes'."""
        for _ in range(max_new_tokens):
            logits, _ = self(tokens)
            last = logits[:, -1, :]  # scores for the token after the last one
            probs = F.softmax(last, dim=-1)  # scores -> probabilities summing to 1
            next_token = torch.multinomial(probs, num_samples=1)  # roll weighted dice
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
