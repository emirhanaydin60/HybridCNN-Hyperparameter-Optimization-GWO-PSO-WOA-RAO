"""Compose existing per-run training_curves.png images into a single grid image with titles.

Usage:
  python tools/compose_training_curves.py --out results/CIFAR10/training_curves_grid.png
"""

from __future__ import annotations

import argparse
import json
import os
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/CIFAR10/training_curves_grid.png")
    parser.add_argument(
        "--images",
        nargs="*",
        default=[
            "results/CIFAR10/GWO/run_01/training_curves.png",
            "results/CIFAR10/WOA/run_03/training_curves.png",
            "results/CIFAR10/GWO/run_03/training_curves.png",
        ],
    )
    args = parser.parse_args()

    imgs = []
    meta = []
    for path in args.images:
        if os.path.exists(path):
            imgs.append(Image.open(path).convert("RGBA"))
            run_dir = os.path.dirname(path)
            summary_path = os.path.join(run_dir, "summary.json")
            summary = {}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as fh:
                        summary = json.load(fh)
                except Exception:
                    summary = {}
            meta.append({"path": path, "run_dir": run_dir, "summary": summary})
        else:
            print(f"Missing image: {path}")

    if not imgs:
        print("No images to compose.")
        return 2

    # Resize to same height
    heights = [im.height for im in imgs]
    target_h = max(heights)
    resized = []
    for im in imgs:
        if im.height != target_h:
            w = int(im.width * (target_h / im.height))
            resized.append(im.resize((w, target_h), Image.LANCZOS))
        else:
            resized.append(im)

    total_w = sum(im.width for im in resized)

    # Title font
    draw_font = None
    font_size = 18
    for fname in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            draw_font = ImageFont.truetype(fname, font_size)
            break
        except Exception:
            draw_font = None
    if draw_font is None:
        try:
            draw_font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            draw_font = ImageFont.load_default()

    captions = []
    for m in meta:
        s = m.get("summary", {}) or {}
        algorithm = (s.get("algorithm") or os.path.basename(os.path.dirname(m["run_dir"]))).upper()
        seed = s.get("run_seed")
        acc = s.get("best_test_accuracy")
        if acc is not None:
            captions.append(f"{algorithm} | Seed: {seed} | Test Accuracy: {acc*100:.2f}%")
        else:
            captions.append(f"{algorithm} | Seed: {seed}")

    # measure caption height
    sample_img = Image.new("RGB", (10, 10))
    sample_draw = ImageDraw.Draw(sample_img)
    bbox = sample_draw.textbbox((0, 0), "Ay", font=draw_font)
    line_height = (bbox[3] - bbox[1]) + 4
    caption_h = line_height + 8

    out_im = Image.new("RGBA", (total_w, target_h + caption_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(out_im)
    x = 0
    for idx, im in enumerate(resized):
        out_im.paste(im, (x, caption_h))
        text = captions[idx] if idx < len(captions) else ""
        text_w = sample_draw.textbbox((0, 0), text, font=draw_font)[2]
        start_x = x + max(0, (im.width - text_w) // 2)
        draw.text((start_x, 4), text, fill="black", font=draw_font)
        x += im.width

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_im.convert("RGB").save(out_path, dpi=(200, 200))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
