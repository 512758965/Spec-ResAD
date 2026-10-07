"""Write train, test and K-shot reference lists for a dataset folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec_resad.dataset import write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Spec-ResAD reference and test splits")
    parser.add_argument("--data", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = write_splits(args.data, args.k, args.seed)
    print(f"wrote {path.parent}")


if __name__ == "__main__":
    main()
