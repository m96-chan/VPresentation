#!/usr/bin/env python3
"""Pose a preprocessed 512x512 image with the THA4 *teacher* and save a strip.

Distillation costs ~20 GPU-hours, and the two things that most often make it
come out wrong — the head landing outside THA4's fixed face crop, and a face
mask that misses an organ — are both visible in a handful of teacher-posed
frames beforehand. So render them beforehand.

  .venv-distill/bin/python tools/preview_teacher.py \
      data/images/ronome_512.png out/ronome_teacher.png \
      [--mask data/images/ronome_face_mask.png] [--tha4 third_party/tha4_src]

The teacher poses any image without training, so this says nothing about how
well the student will fit — only whether it is being asked the right question.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

# Slot indices into THA4's 45-parameter pose vector (see
# tha4/poser/modes/pose_parameters.py).
EYEBROW_RAISED = (6, 7)
EYE_WINK = (12, 13)
MOUTH_AAA = 26
MOUTH_III = 27
IRIS_ROTATION_X = 37
HEAD_X, HEAD_Y, NECK_Z = 39, 40, 41
BODY_Y, BODY_Z = 42, 43
BREATHING = 44

POSES = [
    ("rest", {}),
    ("blink", {EYE_WINK[0]: 1.0, EYE_WINK[1]: 1.0}),
    ("aaa", {MOUTH_AAA: 1.0}),
    ("iii", {MOUTH_III: 1.0, EYEBROW_RAISED[0]: 1.0, EYEBROW_RAISED[1]: 1.0}),
    ("turn", {HEAD_X: -0.8, NECK_Z: -0.5, BODY_Y: -0.5, IRIS_ROTATION_X: -0.6}),
    ("up", {HEAD_Y: 0.8, BODY_Z: 0.4, BREATHING: 1.0, MOUTH_AAA: 0.5}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--mask", help="face mask to overlay onto the rest frame")
    ap.add_argument("--tha4", default="third_party/tha4_src")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.tha4, "src"))
    from tha4.poser.modes.mode_07 import create_poser
    from tha4.shion.base.image_util import (
        extract_pytorch_image_from_PIL_image,
        convert_pytorch_image_to_zero_to_one_numpy_image,
        convert_zero_to_one_numpy_image_to_PIL_image)

    weights = {
        name: os.path.abspath(f"data/tha4/{name}.pt")
        for name in ["eyebrow_decomposer", "eyebrow_morphing_combiner",
                     "face_morpher", "body_morpher", "upscaler"]
    }
    device = torch.device(args.device)
    poser = create_poser(device, module_file_names=dict(weights))

    src = Image.open(args.image).convert("RGBA")
    assert src.size == (512, 512), f"expected 512x512, got {src.size}"
    image = extract_pytorch_image_from_PIL_image(src).to(device)

    frames = []
    for name, slots in POSES:
        pose = torch.zeros(poser.get_num_parameters(), device=device)
        for index, value in slots.items():
            pose[index] = value
        with torch.no_grad():
            posed = poser.pose(image, pose)[0]
        frame = convert_zero_to_one_numpy_image_to_PIL_image(
            np.clip(convert_pytorch_image_to_zero_to_one_numpy_image(posed), 0.0, 1.0))
        frames.append((name, frame.convert("RGBA").resize((512, 512), Image.LANCZOS)))
        print(f"[preview] posed {name}")

    # The face crop THA4 actually uses, drawn on the rest frame, plus the mask.
    if args.mask:
        mask = Image.open(args.mask).convert("RGB")
        rest = frames[0][1].copy()
        rest.alpha_composite(Image.merge("RGBA", (*mask.split(), mask.convert("L").point(lambda v: v // 2))))
        frames.insert(1, ("mask", rest))

    strip = Image.new("RGBA", (512 * len(frames), 512 + 20), (255, 255, 255, 255))
    for i, (name, frame) in enumerate(frames):
        strip.alpha_composite(frame, (512 * i, 20))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    strip.convert("RGB").save(args.out)
    print(f"[preview] {args.image} -> {args.out}  ({', '.join(n for n, _ in frames)})")


if __name__ == "__main__":
    main()
