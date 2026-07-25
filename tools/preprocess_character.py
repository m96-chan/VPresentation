#!/usr/bin/env python3
"""Preprocess an arbitrary character image into THA4 input format.

THA4 wants a 512x512 RGBA image, transparent background (alpha=0), the
character upright and facing forward, with the head roughly inside the
128x128 box in the middle of the top half.

  python tools/preprocess_character.py <in.png> <out.png> \
      [--head-frac 0.34] [--head-cy 150] [--head-h 150]

--head-frac : fraction of the character bbox height that is the head
--head-cy   : target y of the head centre on the 512 canvas
--head-h    : target head height in px on the 512 canvas
"""
import argparse
import numpy as np
from PIL import Image
from rembg import remove
from scipy import ndimage


def keep_largest_component(rgba):
    """Drop detached blobs (e.g. background props) — keep the biggest one."""
    arr = np.array(rgba)
    mask = arr[:, :, 3] > 16
    labels, n = ndimage.label(mask)
    if n <= 1:
        return rgba
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    arr[labels != biggest, 3] = 0
    return Image.fromarray(arr)


def alpha_bbox(rgba):
    a = np.array(rgba)[:, :, 3]
    ys, xs = np.where(a > 16)
    if len(xs) == 0:
        raise SystemExit("no foreground after background removal")
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--head-frac", type=float, default=0.34)
    ap.add_argument("--head-cy", type=float, default=150.0)
    ap.add_argument("--head-h", type=float, default=150.0)
    args = ap.parse_args()

    src = Image.open(args.inp).convert("RGBA")
    cut = remove(src)  # rembg -> transparent background
    cut = keep_largest_component(cut)
    x0, y0, x1, y1 = alpha_bbox(cut)
    char = cut.crop((x0, y0, x1, y1))
    bw, bh = char.size

    # Estimate head height as a fraction of the character bbox, scale so it
    # matches the target head height, then place the head centre at (256, head_cy).
    est_head_h = bh * args.head_frac
    scale = args.head_h / est_head_h
    new_w, new_h = max(1, round(bw * scale)), max(1, round(bh * scale))
    char = char.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    # head centre sits est_head_frac/2 down from the top of the (scaled) bbox.
    head_cy_in_char = (est_head_h * scale) / 2.0
    paste_x = round(256 - new_w / 2)
    paste_y = round(args.head_cy - head_cy_in_char)
    canvas.paste(char, (paste_x, paste_y), char)
    canvas.save(args.out)
    print(f"[preprocess] {args.inp} -> {args.out}  (scaled {scale:.3f}, paste {paste_x},{paste_y})")


if __name__ == "__main__":
    main()
