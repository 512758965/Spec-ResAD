# Spec-ResAD

Implementation materials for **Residual-Aware Feature Mapping for Anomaly Detection in Pre-analytical Medical Specimens**.

## Release scope and current status

This repository contains the supplied model implementation, configuration, training/evaluation scripts and backbone-weight download script. Its correspondence to the original submitted experiment runs has not yet been established. No completed reproduction of the manuscript results is claimed.

- MedSpec-AD hospital images, masks and specimen-level split records are not included.
- A trained full Spec-ResAD checkpoint and original experiment logs/predictions are not included. The downloadable ImageNet backbone weights are initialization weights, not a trained anomaly detector.
- Public MVTec AD data are obtained separately; see [data/README.md](data/README.md).
- Known implementation/manuscript differences and evaluation limitations are listed in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## Installation and usage

## Setup

```bash
pip install -r requirements.txt
python download_weights.py
```

`weights/wide_resnet50_2-95faca4d.pth` initializes the WideResNet-50 encoder. `weights/vit_b_16-c867db91.pth` is used when `--backbone vit_b_16` is set. Both are ImageNet-1K checkpoints.

## Data

```text
dataset/
  <class>/train/good/
  <class>/test/good/
  <class>/test/<defect>/
  <class>/ground_truth/<defect>/
  reference/<class>/
```

`reference/<class>/` is optional. If it is absent, reference images are sampled from `train/good`.

```bash
python scripts/make_reference_split.py --data /path/to/dataset --k 4 --seed 42
python train.py --config configs/spec_resad.yaml --data /path/to/dataset --output outputs/run
python test.py --config configs/spec_resad.yaml --data /path/to/dataset --checkpoint outputs/run/checkpoints/last.pth --output outputs/eval
```

Defaults: 224 center crop, batch size 32, 100 epochs, Adam, learning rate 1e-5 with drops at epochs 70 and 90, 4 reference images, 8 coupling layers. The feature extractor stays frozen.
