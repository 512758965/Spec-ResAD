"""Score a dataset with a trained Spec-ResAD checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spec_resad.dataset import load_image, write_splits
from spec_resad.metrics import auroc
from spec_resad.model import SpecResAD
from train import resolve_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Spec-ResAD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Dataset root with per-class test images and references")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k-shot", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    state = torch.load(args.checkpoint, map_location="cpu")
    if "config" in state:
        saved = state["config"]
        for key in ("backbone", "image_size", "resize_size", "flow_layers", "constraintor_arch",
                    "use_residual", "use_constraintor", "vit_blocks", "image_score"):
            if key in saved:
                cfg[key] = saved[key]
    if args.k_shot is not None:
        cfg["k_shot"] = args.k_shot
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model = SpecResAD(cfg, resolve_weights(cfg, config_path)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    root = Path(args.data)
    k = int(cfg["k_shot"])
    seed = int(cfg["seed"])
    ref_path = root / "splits" / f"reference_k{k}_seed{seed}.json"
    if not ref_path.is_file():
        ref_path = write_splits(root, k, seed)
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    test_path = root / "splits" / "test.json"
    records = json.loads(test_path.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit(f"No test images in {test_path}.")

    by_class = defaultdict(list)
    for record in records:
        by_class[record["class_name"]].append(record)

    out = Path(args.output)
    map_dir = out / "anomaly_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    class_metrics = {}
    reduce = cfg.get("image_score", "max")
    for class_name, items in by_class.items():
        if class_name not in reference["classes"]:
            raise SystemExit(f"Class {class_name} is missing from the reference split.")
        ref_files = reference["classes"][class_name][:k]
        refs = torch.stack([
            load_image(root / rel, int(cfg["resize_size"]), int(cfg["image_size"]))
            for rel in ref_files
        ]).to(device)
        image_scores, image_labels = [], []
        pixel_scores, pixel_labels = [], []
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            images, masks = [], []
            for record in batch:
                images.append(load_image(root / record["path"], int(cfg["resize_size"]), int(cfg["image_size"])))
                if record["mask"] is None or record["label"] == 0:
                    masks.append(torch.zeros(1, int(cfg["image_size"]), int(cfg["image_size"])))
                else:
                    from spec_resad.dataset import load_mask
                    masks.append(load_mask(root / record["mask"], int(cfg["resize_size"]), int(cfg["image_size"])))
            amap = model.score(torch.stack(images).to(device), refs).cpu()
            for record, one_map, mask in zip(batch, amap, masks):
                score_map = one_map[0].numpy().astype(np.float32)
                image_score = float(score_map.max() if reduce == "max" else score_map.mean())
                image_scores.append(image_score)
                image_labels.append(int(record["label"]))
                pixel_scores.append(score_map.reshape(-1))
                pixel_labels.append((mask[0].numpy() > 0.5).astype(np.uint8).reshape(-1))
                stem = Path(record["path"]).stem
                np.save(map_dir / f"{class_name}__{stem}.npy", score_map)
                rows.append({
                    "name": record["path"],
                    "class_name": class_name,
                    "label": int(record["label"]),
                    "image_score": image_score,
                })
        class_metrics[class_name] = {
            "n": len(items),
            "image_auroc": auroc(np.asarray(image_labels), np.asarray(image_scores)),
            "pixel_auroc": auroc(np.concatenate(pixel_labels), np.concatenate(pixel_scores)),
            "reference": ref_files,
        }
        print(class_name, class_metrics[class_name]["image_auroc"], class_metrics[class_name]["pixel_auroc"])

    image_vals = [m["image_auroc"] for m in class_metrics.values() if np.isfinite(m["image_auroc"])]
    pixel_vals = [m["pixel_auroc"] for m in class_metrics.values() if np.isfinite(m["pixel_auroc"])]
    summary = {
        "k_shot": k,
        "seed": seed,
        "image_score": reduce,
        "reference_file": str(ref_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mean_image_auroc": float(np.mean(image_vals)) if image_vals else None,
        "mean_pixel_auroc": float(np.mean(pixel_vals)) if pixel_vals else None,
        "per_class": class_metrics,
        "per_image": rows,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "image_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "class_name", "label", "image_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "mean_image_auroc": summary["mean_image_auroc"],
        "mean_pixel_auroc": summary["mean_pixel_auroc"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
