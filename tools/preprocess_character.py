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

The head-fraction heuristic assumes a chibi-ish full-body illustration; it
mis-scales anything else (a bust-up crop, a tall ahoge, a wide hairstyle).
For those, measure the face in the source image and place it explicitly:

  python tools/preprocess_character.py <in.png> <out.png> --keep-alpha \
      --src-cx 1030 --src-eye-y 575 --src-mouth-y 695 \
      --dst-eye-y 150 --dst-mouth-y 183.5

The scale then comes from the eye-to-mouth distance, and (src-cx, src-eye-y)
lands on (256, dst-eye-y) — which is what actually has to be right, since
THA4 crops the face out of fixed boxes (eyebrows 192..320 x 64..192, face
192..320 x 80..208).

--keep-alpha skips background removal, for a source that is already cut out.
"""
import argparse
import numpy as np
from PIL import Image
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


def place_by_anchors(src, src_cx, src_eye_y, src_mouth_y, dst_eye_y, dst_mouth_y):
    """Scale/translate the source so the measured face lands where THA4 crops it.

    Nothing is cropped to the alpha bbox here: the framing is defined by the
    face alone, and the body is wherever it falls (usually running off the
    bottom edge, which is fine — the poser only warps what is on the canvas).
    """
    scale = (dst_mouth_y - dst_eye_y) / (src_mouth_y - src_eye_y)
    w, h = src.size
    scaled = src.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    paste_x = round(256 - src_cx * scale)
    paste_y = round(dst_eye_y - src_eye_y * scale)
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    canvas.paste(scaled, (paste_x, paste_y), scaled)
    return canvas, scale, paste_x, paste_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--head-frac", type=float, default=0.34)
    ap.add_argument("--head-cy", type=float, default=150.0)
    ap.add_argument("--head-h", type=float, default=150.0)
    ap.add_argument("--keep-alpha", action="store_true",
                    help="source is already cut out; skip rembg")
    ap.add_argument("--src-cx", type=float, help="face centre x in the source")
    ap.add_argument("--src-eye-y", type=float, help="eye line y in the source")
    ap.add_argument("--src-mouth-y", type=float, help="mouth line y in the source")
    ap.add_argument("--dst-eye-y", type=float, default=150.0)
    ap.add_argument("--dst-mouth-y", type=float, default=190.0)
    args = ap.parse_args()

    src = Image.open(args.inp).convert("RGBA")
    if args.keep_alpha:
        cut = src
    else:
        from rembg import remove  # heavy, and unnecessary with --keep-alpha
        cut = remove(src)  # rembg -> transparent background
        cut = keep_largest_component(cut)

    anchors = (args.src_cx, args.src_eye_y, args.src_mouth_y)
    if any(a is not None for a in anchors):
        if any(a is None for a in anchors):
            raise SystemExit("--src-cx, --src-eye-y and --src-mouth-y go together")
        canvas, scale, paste_x, paste_y = place_by_anchors(
            cut, *anchors, args.dst_eye_y, args.dst_mouth_y)
    else:
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
