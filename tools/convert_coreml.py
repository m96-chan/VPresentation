#!/usr/bin/env python3
"""Convert THA4 student nets (mode_14) to CoreML f32 for real-time inference.

Runs in the CoreML env (torch 2.7 + coremltools):
    .venv-coreml/bin/python tools/convert_coreml.py [char_dir]

Patches the models to build coordinate grids directly (CoreML's converter
chokes on aten::affine_grid_generator) and to use fixed batch/param shapes.
Verified numerically equivalent (max_abs ~2.7e-4 vs original) at float32.
CoreML f32 runs the 512 body morpher at ~32 fps on an M3.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "tha4_src" / "src"))


def coords(size, device):
    v = (torch.arange(size, dtype=torch.float32, device=device) * 2 + 1) / size - 1
    return v


def apply_patches():
    import tha4.nn.siren.morpher.siren_morpher_03 as MB
    import tha4.nn.siren.face_morpher.siren_face_morpher_00 as MF
    import tha4.nn.image_processing_util as IPU
    from torch.nn.functional import grid_sample

    def body_pos(self, n, s, device):
        xs, ys = coords(s, device), coords(s, device)
        return torch.cat([xs.view(1, 1, 1, s).expand(1, 1, s, s),
                          ys.view(1, 1, s, 1).expand(1, 1, s, s)], dim=1)
    MB.SirenMorpher03.get_position_grid = body_pos
    MB.SirenMorpher03.get_pose_image = lambda self, pose, s: pose.view(1, 45, 1, 1).expand(1, 45, s, s)

    def face_forward(self, pose, position=None):
        s = self.args.image_size
        xs, ys = coords(s, pose.device), coords(s, pose.device)
        pos = torch.cat([xs.view(1, 1, 1, s).expand(1, 1, s, s),
                         ys.view(1, 1, s, 1).expand(1, 1, s, s)], dim=1)
        pose_image = pose.view(1, self.args.pose_size, 1, 1).expand(1, self.args.pose_size, s, s)
        return self.siren.forward(torch.cat([pos, pose_image], dim=1))
    MF.SirenFaceMorpher00.forward = face_forward

    def apply(self, gc, image, align_corners=False):
        n, c, h, w = image.shape
        gc = torch.transpose(gc.reshape(1, 2, h * w), 1, 2).reshape(1, h, w, 2)
        xs, ys = coords(w, image.device), coords(h, image.device)
        base = torch.cat([xs.view(1, 1, w, 1).expand(1, h, w, 1),
                          ys.view(1, h, 1, 1).expand(1, h, w, 1)], dim=3)
        return grid_sample(image, base + gc, mode="bilinear", padding_mode="border",
                           align_corners=align_corners)
    IPU.GridChangeApplier.apply = apply


class First(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, *a):
        return self.m(*a)[0]


def convert(module, example_inputs, input_specs, out_path, ref_wrap=None):
    module = module.eval()
    wrapped = First(module).eval() if ref_wrap else module
    with torch.no_grad():
        ref = wrapped(*example_inputs)
        ref = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()
    ep = torch.export.export(wrapped, tuple(example_inputs)).run_decompositions({})
    ml = ct.convert(ep, minimum_deployment_target=ct.target.macOS15,
                    compute_units=ct.ComputeUnit.CPU_AND_GPU,
                    compute_precision=ct.precision.FLOAT32)
    ml.save(str(out_path))
    names = [i.name for i in ml.get_spec().description.input]
    feed = {n: t.numpy() for n, t in zip(names, example_inputs)}
    out = list(ml.predict(feed).values())[0]
    d = np.abs(out - ref)
    print(f"  {out_path.name}: inputs={names} out={out.shape} max_abs={d.max():.3e}")
    return names


def main():
    char_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/character_models/lambda_00")
    apply_patches()
    from tha4.poser.modes import mode_14
    out_dir = char_dir / "coreml"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    fm = mode_14.load_face_morpher(str(char_dir / "face_morpher.pt"))
    convert(fm, [torch.rand(1, 39)], None, out_dir / "face_morpher.mlpackage")

    bm = mode_14.load_body_morpher(str(char_dir / "body_morpher.pt"))
    convert(bm, [torch.rand(1, 4, 512, 512), torch.rand(1, 45)], None,
            out_dir / "body_morpher.mlpackage", ref_wrap=True)
    print("CoreML models saved to", out_dir)


if __name__ == "__main__":
    main()
