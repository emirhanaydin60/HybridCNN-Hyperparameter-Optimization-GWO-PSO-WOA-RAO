"""Compose existing per-run confusion_matrix.png images into a single grid image.

Usage:
  python tools/compose_confusion_grid.py --out results/CIFAR10/confusion_grid.png
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/CIFAR10/confusion_grid.png")
    parser.add_argument(
        "--images",
        nargs="*",
        default=[
            "results/CIFAR10/GWO/run_01/confusion_matrix.png",
            "results/CIFAR10/WOA/run_03/confusion_matrix.png",
            "results/CIFAR10/GWO/run_03/confusion_matrix.png",
        ],
    )
    args = parser.parse_args()

    imgs = []
    meta = []
    for path in args.images:
        if os.path.exists(path):
            imgs.append(Image.open(path).convert("RGBA"))
            # try to load summary.json sitting in the same run directory
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

    # Resize images to same height
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

    # Prepare captions using summaries (single-line title: ALGO | Seed: X | Test Accuracy: YY.YY%)
    draw_font = None
    font_size = 18
    # try common bold fonts on the system for a stronger title appearance
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
    # only show algorithm, seed and accuracy (single-line caption)
    for m in meta:
        s = m.get("summary", {}) or {}
        algorithm = (s.get("algorithm") or os.path.basename(os.path.dirname(m["run_dir"]))).upper()
        seed = s.get("run_seed")
        acc = s.get("best_test_accuracy")
        if acc is not None:
            first_line = f"{algorithm} | Seed: {seed} | Test Accuracy: {acc*100:.2f}%"
        else:
            first_line = f"{algorithm} | Seed: {seed}"
        captions.append([first_line])
    max_caption_lines = 1

    # compute caption height
    sample_img = Image.new("RGB", (10, 10))
    sample_draw = ImageDraw.Draw(sample_img)
    bbox = sample_draw.textbbox((0, 0), "Ay", font=draw_font)
    line_height = (bbox[3] - bbox[1]) + 2
    caption_h = line_height * max_caption_lines + 8

    out_im = Image.new("RGBA", (total_w, target_h + caption_h), (255, 255, 255, 255))
    x = 0
    # paste each image and draw its caption above
    draw = ImageDraw.Draw(out_im)
    for idx, im in enumerate(resized):
        out_im.paste(im, (x, caption_h))
        # draw caption centered above the image
        lines = captions[idx] if idx < len(captions) else [""]
        # compute text block width
        text_block_width = max(sample_draw.textbbox((0, 0), line, font=draw_font)[2] for line in lines) if lines else 0
        start_x = x + max(0, (im.width - text_block_width) // 2)
        y = 4
        for line in lines:
            draw.text((start_x, y), line, fill="black", font=draw_font)
            y += line_height
        x += im.width

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_im.convert("RGB").save(out_path, dpi=(200, 200))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
