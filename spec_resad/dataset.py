"""MVTec-style folders for Spec-ResAD.

    root/
      <class>/train/good/*.png
      <class>/test/good/*.png
      <class>/test/<defect>/*.png
      <class>/ground_truth/<defect>/*.png
      reference/<class>/*.png          optional, used before sampling train/good
      splits/train.json
      splits/test.json
      splits/reference_k{K}_seed{S}.json

Images are resized on the short side and center-cropped.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in EXTS and path.is_file()


def discover(root: Path) -> dict:
    root = Path(root)
    classes = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name in {"splits", "reference"}:
            continue
        if (path / "train" / "good").is_dir() or (path / "test").is_dir() or (root / "reference" / path.name).is_dir():
            classes.append(path.name)
    if not classes:
        raise FileNotFoundError(f"No class directories under {root}.")
    train, test, good = [], [], defaultdict(list)
    for name in classes:
        class_dir = root / name
        good_dir = class_dir / "train" / "good"
        if good_dir.is_dir():
            for path in sorted(good_dir.rglob("*")):
                if _is_image(path):
                    rel = path.relative_to(root).as_posix()
                    train.append({"path": rel, "class_name": name, "label": 0, "mask": None})
                    good[name].append(rel)
        test_dir = class_dir / "test"
        if not test_dir.is_dir():
            continue
        for path in sorted(test_dir.rglob("*")):
            if not _is_image(path):
                continue
            defect = path.parent.name
            label = 0 if defect == "good" else 1
            mask = None
            if label == 1:
                mask = _find_mask(class_dir / "ground_truth" / defect, path.stem)
                if mask is not None:
                    mask = mask.relative_to(root).as_posix()
            test.append({
                "path": path.relative_to(root).as_posix(),
                "class_name": name,
                "label": label,
                "mask": mask,
            })
    return {"train": train, "test": test, "good": good, "classes": classes}


def _find_mask(directory: Path, stem: str) -> Path | None:
    if not directory.is_dir():
        return None
    for candidate in (stem, f"{stem}_mask"):
        for ext in EXTS:
            path = directory / f"{candidate}{ext}"
            if path.is_file():
                return path
    return None


def reference_paths(root: Path, class_name: str, good: list[str], k: int, rng: random.Random) -> list[str]:
    folder = root / "reference" / class_name
    if folder.is_dir():
        found = sorted(p.relative_to(root).as_posix() for p in folder.rglob("*") if _is_image(p))
        if found:
            if len(found) >= k:
                return found[:k]
            return found
    if not good:
        raise FileNotFoundError(f"No normal reference images for {class_name}.")
    if len(good) >= k:
        return rng.sample(list(good), k)
    picked = list(good)
    while len(picked) < k:
        picked.append(rng.choice(good))
    return picked


def write_splits(root: str | Path, k: int, seed: int) -> Path:
    root = Path(root)
    found = discover(root)
    rng = random.Random(seed)
    reference = {
        name: reference_paths(root, name, found["good"][name], k, rng)
        for name in found["classes"]
    }
    out = root / "splits"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "k": k,
        "note": "Reference images sampled from reference/<class> or train/good.",
        "classes": reference,
    }
    ref_path = out / f"reference_k{k}_seed{seed}.json"
    ref_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "train.json").write_text(json.dumps(found["train"], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "test.json").write_text(json.dumps(found["test"], indent=2, ensure_ascii=False), encoding="utf-8")
    return ref_path


def load_image(path: Path, resize_size: int, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, resize_size, interpolation=TF.InterpolationMode.BILINEAR)
    image = TF.center_crop(image, image_size)
    return TF.to_tensor(image)


def load_mask(path: Path | None, resize_size: int, image_size: int) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, image_size, image_size)
    mask = Image.open(path).convert("L")
    mask = TF.resize(mask, resize_size, interpolation=TF.InterpolationMode.NEAREST)
    mask = TF.center_crop(mask, image_size)
    return (TF.to_tensor(mask) > 0).float()


class SpecimenDataset(Dataset):
    def __init__(self, root: str | Path, split: str, cfg: dict):
        self.root = Path(root)
        self.cfg = cfg
        manifest = self.root / "splits" / f"{split}.json"
        if not manifest.is_file():
            write_splits(self.root, int(cfg.get("k_shot", 4)), int(cfg.get("seed", 42)))
        self.records = json.loads(manifest.read_text(encoding="utf-8"))
        if not self.records:
            raise FileNotFoundError(f"No samples in {manifest}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_image(self.root / record["path"], int(self.cfg["resize_size"]), int(self.cfg["image_size"]))
        mask_path = None if record["mask"] is None else self.root / record["mask"]
        mask = load_mask(mask_path, int(self.cfg["resize_size"]), int(self.cfg["image_size"]))
        if record["label"] == 0:
            mask = torch.zeros_like(mask)
        return image, mask, int(record["label"]), record["class_name"], record["path"]


def collate_spec(batch):
    return (
        torch.stack([item[0] for item in batch]),
        torch.stack([item[1] for item in batch]),
        torch.tensor([item[2] for item in batch], dtype=torch.long),
        [item[3] for item in batch],
        [item[4] for item in batch],
    )


class NormalBank:
    def __init__(self, root: str | Path, cfg: dict):
        self.root = Path(root)
        self.cfg = cfg
        found = discover(self.root)
        self.good = found["good"]

    def sample(self, class_name: str, k: int, exclude: str | None, rng: random.Random) -> torch.Tensor:
        paths = [p for p in self.good[class_name] if p != exclude] or list(self.good[class_name])
        if not paths:
            raise FileNotFoundError(f"No normal training images for {class_name}.")
        if len(paths) >= k:
            chosen = rng.sample(paths, k)
        else:
            chosen = [rng.choice(paths) for _ in range(k)]
        images = [
            load_image(self.root / rel, int(self.cfg["resize_size"]), int(self.cfg["image_size"]))
            for rel in chosen
        ]
        return torch.stack(images, dim=0)
