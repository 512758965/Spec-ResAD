# Implementation status

The supplied scripts have been packaged without changing their model mathematics. This release has received a static source inspection, not an end-to-end training/evaluation validation.

1. `SpecResAD._occ` uses squared L2 norms; submitted Equation (3) uses unsquared L2 norms. The original experiment version needs verification.
2. Automatic training manifests contain `train/good` images only. This does not document the abnormal-image/mask supervision described in the submitted manuscript. Additional supervised training records must be supplied and separated from evaluation records.
3. `forward_train` averages NLL over all positions. If abnormal training records are supplied, those positions are also included; this differs from a normal-only likelihood objective.
4. `test.py` names saved maps by category and image stem only. Repeated stems across defect subdirectories can overwrite output maps. This needs correction before relying on saved maps for subsequent analysis.
5. Reference/query separation is not verified at physical-specimen level. The training-reference fallback can allow self-reference when no distinct normal candidate is available; explicit reference directories also require overlap checks.
6. The archive does not include the other baseline implementations, original submitted predictions, or trained full-detector checkpoints. Source/category splits must be specified for each experiment; running on every discovered category is not a demonstration of unseen-category transfer.

These notes distinguish the supplied implementation from verified historical experimental results. They are not evidence that any new model experiment has been completed.
