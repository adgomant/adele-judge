#!/usr/bin/env python
from __future__ import annotations

import argparse

from adele_judge.config import load_config
from adele_judge.modeling import load_tokenizer
from adele_judge.pipeline import load_or_prepare_splits
from adele_judge.tokenization import supervised_token_debug_rows
from adele_judge.utils import ensure_dir, project_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    tokenizer = load_tokenizer(config)
    splits = load_or_prepare_splits(config, tokenizer)
    out_dir = ensure_dir(project_output_dir(config))
    out_path = out_dir / "token_supervision_debug.txt"
    chunks = []
    for i, (_, row) in enumerate(splits[args.split].head(args.num_examples).iterrows()):
        chunks.append(f"EXAMPLE {i}\n")
        rows = supervised_token_debug_rows(
            row.to_dict(),
            tokenizer,
            config["prompt"]["system_prompt"],
            int(config["training"]["max_seq_length"]),
            config["data"]["filters"].get("on_sequence_overflow", "skip"),
        )
        for item in rows:
            marker = "SUP" if item["supervised"] else "..."
            token_text = item["token"].replace("\n", "\\n")
            chunks.append(f"{item['position']:05d} {marker} {item['token_id']:>8} {token_text}\n")
        chunks.append("\n")
    out_path.write_text("".join(chunks), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
