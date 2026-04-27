import argparse
from hdfnet.utils.config import load_config
from hdfnet.eval.gradcam import save_gradcam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model-path", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="gradcam.png")
    p.add_argument("--layer-name", default=None)
    p.add_argument("--class-index", type=int, default=None)
    args = p.parse_args()
    config = load_config(args.config)
    probs = save_gradcam(args.model_path, args.image, config, args.out, args.class_index, args.layer_name)
    print({cls: float(probs[i]) for i, cls in enumerate(config["classes"])})


if __name__ == "__main__":
    main()
