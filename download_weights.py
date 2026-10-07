"""Download ImageNet-1K checkpoints used to initialize the feature extractor."""

from __future__ import annotations

import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent / "weights"
FILES = {
    "wide_resnet50_2-95faca4d.pth": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "vit_b_16-c867db91.pth": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = DEST / name
        if path.exists() and path.stat().st_size > 10_000_000:
            print(f"skip {path}")
            continue
        print(f"download {url}")
        urllib.request.urlretrieve(url, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
