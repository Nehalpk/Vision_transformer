import argparse
from hdfnet.utils.config import load_config
from hdfnet.training.kfold import train_kfold


def main():
    p = argparse.ArgumentParser(description="Train DP-ViT/HDFNet with stratified K-fold cross-validation.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="hdfnet", choices=["vgg", "resnet", "vit", "hdfnet"])
    p.add_argument("--n-folds", type=int, default=None, help="Use 4 for paper DP-ViT CV, 10 for final HDFNet generalization.")
    args = p.parse_args()
    config = load_config(args.config)
    train_kfold(config, model_name=args.model, n_splits=args.n_folds)


if __name__ == "__main__":
    main()
