import argparse
import cv2
import matplotlib.pyplot as plt
from hdfnet.data.preprocess import read_rgb, estimate_lesion_mask, lesion_coverage, patch_size_from_coverage
from hdfnet.utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="lesion_coverage.png")
    args = p.parse_args()
    cfg = load_config(args.config)
    img = read_rgb(args.image, cfg.get("image_size", 150))
    mask = estimate_lesion_mask(img)
    cov = lesion_coverage(img)
    ps = patch_size_from_coverage(cov, tuple(cfg["vit"].get("coverage_thresholds", [0.25, 0.50])))
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(img); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title(f"Mask cov={cov:.3f}"); ax[1].axis("off")
    ax[2].imshow(overlay); ax[2].set_title(f"Patch={ps}x{ps}"); ax[2].axis("off")
    plt.tight_layout(); plt.savefig(args.out, dpi=200)
    print(f"coverage={cov:.4f}, patch_size={ps}, saved={args.out}")


if __name__ == "__main__":
    main()
