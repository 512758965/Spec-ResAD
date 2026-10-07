"""Train Spec-ResAD. The log records this run only."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spec_resad.dataset import NormalBank, SpecimenDataset, collate_spec
from spec_resad.model import SpecResAD


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_weights(cfg: dict, config_path: Path) -> str:
    key = "vit_weights" if cfg.get("backbone") == "vit_b_16" else "weights"
    value = cfg[key]
    raw = Path(value)
    if raw.is_file():
        return str(raw.resolve())
    project = config_path.parent.parent
    candidate = (project / value).resolve()
    if candidate.is_file():
        return str(candidate)
    packaged = project.parent / "weights" / raw.name
    if packaged.is_file():
        return str(packaged)
    raise FileNotFoundError(f"Weight file not found: {value}. Run download_weights.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spec-ResAD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone", default=None, choices=["wide_resnet50", "vit_b_16"])
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.backbone is not None:
        cfg["backbone"] = args.backbone
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    seed = int(cfg["seed"])
    set_seed(seed)
    rng = random.Random(seed)

    out = Path(args.output)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    dataset = SpecimenDataset(args.data, "train", cfg)
    if len(dataset) == 0:
        raise SystemExit("Training set is empty. Each class needs <class>/train/good.")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_spec,
        pin_memory=device.startswith("cuda"),
    )
    bank = NormalBank(args.data, cfg)
    model = SpecResAD(cfg, resolve_weights(cfg, config_path)).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg["milestones"]), gamma=float(cfg["lr_gamma"])
    )
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_loss", float("inf")))

    k = int(cfg["k_shot"])
    log_path = out / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({
            "event": "start",
            "train_images": len(dataset),
            "device": device,
            "backbone": cfg["backbone"],
        }) + "\n")
        for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
            model.train()
            totals, occs, mls, bgs = [], [], [], []
            for step, (images, masks, _, class_names, paths) in enumerate(loader, start=1):
                refs = torch.stack(
                    [bank.sample(name, k, exclude=path, rng=rng) for name, path in zip(class_names, paths)],
                    dim=0,
                )
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                refs = refs.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out_step = model.forward_train(images, refs, masks)
                out_step["loss"].backward()
                optimizer.step()
                totals.append(float(out_step["loss"].detach()))
                occs.append(float(out_step["occ"]))
                mls.append(float(out_step["ml"]))
                bgs.append(float(out_step["bg"]))
                if step == 1 or step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{len(loader)} loss {totals[-1]:.4f}")
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": float(np.mean(totals)),
                "occ": float(np.mean(occs)),
                "ml": float(np.mean(mls)),
                "bg": float(np.mean(bgs)),
            }
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "config": cfg,
            }
            torch.save(ckpt, out / "checkpoints" / "last.pth")
            if row["train_loss"] < best_loss:
                best_loss = row["train_loss"]
                ckpt["best_loss"] = best_loss
                torch.save(ckpt, out / "checkpoints" / "best.pth")
            row["best_loss"] = best_loss
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
