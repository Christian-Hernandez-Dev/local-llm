# Decisions

`choice · why · trade-off`

- Python + uv + PyTorch · the standard ML stack; uv is fast · PyTorch is a big (~1 GB) install
- Write the model by hand · see every piece · slower than using a ready-made library
- Tiny Shakespeare first · 1 MB, trains in seconds · output is archaic gibberish
- Character-level tokenizer to start · simplest thing that works · long sequences, weak model; BPE comes in slice 3
- Keep repo in iCloud ~/Documents with `*.nosync` (incl. `.git.nosync`) · user preference · must remember the naming rule
- Hand-written attention (not `F.scaled_dot_product_attention`) · every step of attention is visible · a bit slower on GPU
- Tiny GPT size: 4 layers × 6 heads × 192 dims, 128-char context (~1.8M params) · trains in ~4 min on M4 Max · bigger = better text but slower
- Keep BigramModel alongside TinyGPT (`--model bigram`) · compare simple vs transformer side by side · a little extra code
