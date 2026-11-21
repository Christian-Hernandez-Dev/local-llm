"""Command line: `local-llm train` and `local-llm generate`."""

import argparse

import torch

from local_llm.train import CHECKPOINT, TrainConfig, load, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="local-llm")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train the model on Tiny Shakespeare")
    t.add_argument("--steps", type=int, default=TrainConfig.steps)
    g = sub.add_parser("generate", help="write text with the trained model")
    g.add_argument("--chars", type=int, default=500)
    g.add_argument("--prompt", default="\n")
    args = parser.parse_args()

    if args.command == "train":
        train(TrainConfig(steps=args.steps))
    else:
        if not CHECKPOINT.exists():
            parser.error("no trained model yet: run `make train` first")
        model, tokenizer = load()
        start = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
        out = model.generate(start, max_new_tokens=args.chars)
        print(tokenizer.decode(out[0].tolist()))
