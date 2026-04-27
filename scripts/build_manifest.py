import argparse
from hdfnet.utils.config import load_config
from hdfnet.data.manifest import combine_manifests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="data/manifest.csv")
    args = p.parse_args()
    config = load_config(args.config)
    df = combine_manifests(config)
    df.to_csv(args.out, index=False)
    print(df["label"].value_counts())
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
