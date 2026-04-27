import argparse
from hdfnet.utils.config import load_config
from hdfnet.eval.metrics import save_classification_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model-path", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    config = load_config(args.config)
    save_classification_outputs(args.model_path, args.split_csv, config, args.out_dir, args.batch_size)


if __name__ == "__main__":
    main()
