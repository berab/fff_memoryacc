#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value} is not an integer") from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer")
    return ivalue


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random float arrays for leaves and save to lw1.bin, lb1.bin, "
            "lw2.bin, and lb2.bin."
        )
    )
    parser.add_argument(
        "n_leaves",
        type=positive_int,
        nargs="?",
        default=16,
        help="Number of leaves",
    )
    parser.add_argument(
        "leaf_width",
        type=positive_int,
        nargs="?",
        default=16,
        help="Leaf width",
    )
    parser.add_argument(
        "in_features",
        type=positive_int,
        nargs="?",
        default=784,
        help="Input feature count",
    )
    parser.add_argument(
        "out_features",
        type=positive_int,
        nargs="?",
        default=10,
        help="Output feature count",
    )
    args = parser.parse_args()

    lw1_size = args.n_leaves * args.leaf_width * args.in_features
    lb1_size = args.n_leaves * args.leaf_width
    lw2_size = args.n_leaves * args.out_features * args.leaf_width
    lb2_size = args.n_leaves * args.out_features

    rng = np.random.default_rng()
    lw1 = rng.random(lw1_size).astype(np.float32)
    lb1 = rng.random(lb1_size).astype(np.float32)
    lw2 = rng.random(lw2_size).astype(np.float32)
    lb2 = rng.random(lb2_size).astype(np.float32)

    output_dir = Path("leaves")
    output_dir.mkdir(parents=True, exist_ok=True)

    lw1.tofile(output_dir / "lw1.bin")
    lb1.tofile(output_dir / "lb1.bin")
    lw2.tofile(output_dir / "lw2.bin")
    lb2.tofile(output_dir / "lb2.bin")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
