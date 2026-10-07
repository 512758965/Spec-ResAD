# Datasets

## Public MVTec AD subset

The locally supplied data consist of three public MVTec AD categories, not the hospital MedSpec-AD dataset:

| Category | Normal training images | Normal test images | Abnormal test images | Masks |
|---|---:|---:|---:|---:|
| bottle | 209 | 20 | 63 | 63 |
| capsule | 219 | 23 | 109 | 109 |
| pill | 267 | 26 | 141 | 141 |

Obtain the data from the original provider: https://www.mvtec.com/company/research/datasets/mvtec-ad

Credit: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection”, CVPR 2019. The supplied dataset readmes state copyright MVTec Software GmbH and CC BY-NC-SA 4.0. The original data license applies separately from the code; consult the provider for commercial use.

Keep downloaded archives outside the GitHub source upload. Extract the category folders beneath a local dataset root:

```text
dataset/
  bottle/train/good/
  bottle/test/good/
  bottle/test/<defect>/
  bottle/ground_truth/<defect>/
  capsule/...
  pill/...
```

Abnormal test images must not be used for training and then reused as evaluation queries. This subset alone does not reproduce the manuscript's claimed ten-source-category protocol.

## MedSpec-AD

MedSpec-AD images/masks, physical-specimen identifiers, acquisition records and verified train/reference/query splits have not been supplied with this code release. Public bottle/capsule/pill categories are not replacements for the five medical-container categories. No MedSpec-AD dataset download is claimed here.
