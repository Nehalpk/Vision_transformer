import argparse
from hdfnet.utils.config import load_config
from hdfnet.training.train import train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="hdfnet", choices=["vgg", "resnet", "vit", "hdfnet"])
    args = p.parse_args()
    config = load_config(args.config)
    train_model(config, model_name=args.model)


if __name__ == "__main__":
    main()
