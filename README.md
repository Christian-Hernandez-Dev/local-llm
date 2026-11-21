# Local LLM

A tiny language model built from scratch in PyTorch, to learn what an LLM is made of.
It runs on a Mac (Apple GPU via MPS). It won't rival Claude; the point is to see every piece.

```
text → tokenizer → model → training → generation
```

## Run it

```bash
make setup      # install deps (into .venv.nosync) + enable the gitleaks hook
make train      # downloads Tiny Shakespeare, trains, saves checkpoints.nosync/bigram.pt
make generate   # writes 500 characters of fake Shakespeare
uv run local-llm generate --prompt "ROMEO:" --chars 300
make check      # lint + types + tests
```

## Roadmap

1. ✅ **Bigram model**: predicts the next character from the current one. Trains in seconds.
2. ⏭ **Tiny GPT**: add self-attention so each token can look back at earlier ones.
3. **Own tokenizer (BPE)**: learn word pieces instead of single characters.
4. **Train on TinyStories**: longer run on the Mac GPU, plot the loss curve.
5. **Web page + free deploy**: Gradio on Hugging Face Spaces.
6. *(Optional)* Fine-tune a small open model and compare it with the from-scratch one.
