# HDFNet Skin Lesion Classification —

This project implements the pipeline: preprocessing, class-aware selective augmentation, ALC-VGG16, DFE-ResNet50, dynamic-patching ViT, feature fusion, HDFNet classification, ROC/confusion-matrix evaluation, and Grad-CAM.

The uploaded CODE USES four-class dermoscopic classifier for **BKL, MEL, NV, BCC** using 150×150×3 images, class-aware selective augmentation, a customized VGG-16 branch, a ResNet-50 branch with dermatological attention, a dynamic-patching ViT branch, and a final HDFNet classifier over a 1536-dimensional fused feature vector.

## Important note about exact reproduction

The Code gives architecture implementation and hyperparameters.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format option 1: simple folder

Put images like this:

```text
data/images/BKL/*.jpg
data/images/MEL/*.jpg
data/images/NV/*.jpg
data/images/BCC/*.jpg
```

Then train:

```bash
python scripts/train.py --config configs/default.yaml --model hdfnet
```

## Data format option 2: manifest CSV

Create `data/manifest.csv`:

```csv
image_path,label,source
/path/img1.jpg,MEL,ISIC2019
/path/img2.jpg,BKL,ISIC2020
/path/img3.jpg,NV,PAD_UFES
/path/img4.jpg,BCC,ISIC2019
```

Then train:

```bash
python scripts/train.py --config configs/default.yaml --model hdfnet
```

## Train individual branches

```bash
python scripts/train.py --model vgg
python scripts/train.py --model resnet
python scripts/train.py --model vit
python scripts/train.py --model hdfnet
```

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --model-path runs/hdfnet_skin_lesion/hdfnet/hdfnet_best.keras \
  --split-csv runs/hdfnet_skin_lesion/hdfnet/test.csv \
  --out-dir runs/hdfnet_skin_lesion/hdfnet/eval_test
```

Outputs:

```text
classification_report.csv
confusion_matrix.csv
confusion_matrix.png
roc_curve.png
auc.txt
predictions.csv
```

## Grad-CAM

```bash
python scripts/gradcam.py \
  --model-path runs/hdfnet_skin_lesion/hdfnet/hdfnet_best.keras \
  --image /path/to/image.jpg \
  --out gradcam.png
```

## Lesion coverage / dynamic patch demo

```bash
python scripts/lesion_coverage_demo.py --image /path/to/image.jpg --out lesion_coverage.png
```

## File map

```text
configs/default.yaml                 Main configuration
hdfnet/data/manifest.py              Dataset harmonization helpers
hdfnet/data/preprocess.py            Resize, normalize, Otsu lesion coverage, patch-size selection
hdfnet/data/augment.py               Class-aware selective augmentation
hdfnet/data/dataset.py               tf.data generator
hdfnet/models/vgg.py                 ALC-VGG16
hdfnet/models/resnet.py              DFE-ResNet50 + dermatological attention
hdfnet/models/vit.py                 Dynamic-patching ViT
hdfnet/models/hdfnet.py              Fusion model and 1536-feature HDFNet classifier
hdfnet/training/train.py             Training loop
hdfnet/eval/metrics.py               ROC, AUC, confusion matrix, classification report
hdfnet/eval/gradcam.py               Grad-CAM visual explanation
scripts/*.py                         CLI entry points
```

## Relationship to the linked repository

The linked GitHub repository contains a ViT implementation, training script, Grad-CAM script, lesion coverage script, and feature-combination script.

## K-fold cross-validation

This fixed version includes K-fold training.

```bash
# DP-ViT configuration tables: 4-fold cross-validation
python scripts/train_kfold.py --config configs/default.yaml --model vit --n-folds 4

# Final HDFNet generalization experiment mentioned in the Results section: 10-fold cross-validation
python scripts/train_kfold.py --config configs/default.yaml --model hdfnet --n-folds 10
```

Each fold saves its own files under:

---

runs/hdfnet*skin_lesion/<model>\_kfold*<K>/fold*01/
runs/hdfnet_skin_lesion/<model>\_kfold*<K>/fold_02/

---

For every fold the code writes `train_original.csv`, `train_balanced.csv`, `val.csv`, `history.csv`, `classification_report.csv`, `confusion_matrix.csv`, `predictions.csv`, `metrics.json`, and model checkpoints. After all folds finish, it writes:

---

kfold_metrics_by_fold.csv
kfold_metrics_summary.csv
README_KFOLD_RESULTS.txt

---

The K-fold splitter uses stratification by class and source dataset when possible. If a source/class pair is too rare for the requested K, it safely falls back to label-only stratification.
