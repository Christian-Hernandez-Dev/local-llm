# Decisions

`choice · why · trade-off`

- Python + uv + PyTorch · the standard ML stack; uv is fast · PyTorch is a big (~1 GB) install
- Write the model by hand · see every piece · slower than using a ready-made library
- Tiny Shakespeare first · 1 MB, trains in seconds · output is archaic gibberish
- Character-level tokenizer to start · simplest thing that works · long sequences, weak model; BPE comes in slice 3
- Keep repo in iCloud ~/Documents with `*.nosync` (incl. `.git.nosync`) · user preference · must remember the naming rule
