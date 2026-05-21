from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small FFHQ subset into inputs_normal.")
    parser.add_argument("--dataset", default="merkol/ffhq-256")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default="data/inputs_normal/ffhq_subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)

    for index, item in enumerate(dataset):
        if index >= args.limit:
            break
        image = item["image"]
        image.save(output_dir / f"{index:05d}.png")
        if index % 100 == 0:
            print(f"Saved {index} images")

    print(f"Done! Saved {min(args.limit, len(dataset))} images to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
