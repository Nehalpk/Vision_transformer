# K-fold Fix Notes

cross-validation support.

## New files

- `hdfnet/training/kfold.py` — full stratified K-fold training/evaluation loop.
- `scripts/train_kfold.py` — CLI entry point.

## Updated files

- `configs/default.yaml` — added `kfold:` and `augmentation:` settings.
- `README.md` — added exact K-fold run commands and output description.

## Paper mapping

- Use `--model vit --n-folds 4` for the DP-ViT cross-validation setting described in the configuration tables.
- Use `--model hdfnet --n-folds 10` for the final generalization experiment described in the Results section.

## Outputs per fold

Each fold writes the split CSVs, training history, classification report, confusion matrix, predictions with probabilities, metrics JSON, and model checkpoint. The root K-fold folder writes mean/std summary metrics.
