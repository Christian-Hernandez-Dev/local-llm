"""Command line: `local-llm train` and `local-llm generate`."""

import argparse

import torch

from local_llm.train import TrainConfig, checkpoint_path, load, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="local-llm")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a model on Tiny Shakespeare")
    t.add_argument("--model", choices=["gpt", "bigram"], default="gpt")
    t.add_argument("--steps", type=int)

    g = sub.add_parser("generate", help="write text with a trained model")
    g.add_argument("--model", choices=["gpt", "bigram"], default="gpt")
    g.add_argument("--chars", type=int, default=500)
    g.add_argument("--prompt", default="\n")
    g.add_argument("--temperature", type=float, default=1.0, help="<1 safer, >1 wilder")
    args = parser.parse_args()

    if args.command == "train":
        overrides = {"steps": args.steps} if args.steps else {}
        train(TrainConfig.for_model(args.model, **overrides))
        return

    path = checkpoint_path(args.model)
    if not path.exists():
        parser.error(f"no trained {args.model} model yet: run `make train` first")
    model, tokenizer = load(path)
    start = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    out = model.generate(start, max_new_tokens=args.chars, temperature=args.temperature)
    print(tokenizer.decode(out[0].tolist()))
