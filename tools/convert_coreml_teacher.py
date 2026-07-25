#!/usr/bin/env python3
"""Convert THA4 *teacher* (general poser, mode_07) nets to CoreML f32.

The teacher poses ANY preprocessed 512 image (e.g. char.png) with no
per-character distillation. Slow in candle (~8s), but CoreML runs it at
~2-3 fps on an M3 — usable for arbitrary characters.

Run in the CoreML env:  .venv-coreml/bin/python tools/convert_coreml_teacher.py
Outputs: data/tha4/coreml/{eyebrow_decomposer,eyebrow_morphing_combiner,
         face_morpher,body_morpher,upscaler}.mlpackage
"""
import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "tha4_src" / "src"))


def coords(s, d):
    return (torch.arange(s, dtype=torch.float32, device=d) * 2 + 1) / s - 1


def patch():
    from torch.nn.functional import grid_sample
    import tha4.nn.image_processing_util as IPU
    import tha4.nn.face_morpher.face_morpher_08 as FM

    def warp(gc, image, align_corners=False):
        n, c, h, w = image.shape
        gc = torch.transpose(gc.reshape(1, 2, h * w), 1, 2).reshape(1, h, w, 2)
        xs, ys = coords(w, image.device), coords(h, image.device)
        base = torch.cat([xs.view(1, 1, w, 1).expand(1, h, w, 1),
                          ys.view(1, h, 1, 1).expand(1, h, w, 1)], dim=3)
        return grid_sample(image, base + gc, mode="bilinear", padding_mode="border",
                           align_corners=align_corners)

    IPU.GridChangeApplier.apply = lambda self, gc, image, align_corners=False: warp(gc, image, align_corners)
    IPU.apply_grid_change = lambda gc, image: warp(gc, image, False)  # module-level fn (combiner)
    FM.FaceMorpher08.apply_grid_change = lambda self, gc, image: warp(gc, image, False)
    # nets import apply_grid_change by name — rebind in their modules too.
    import tha4.nn.eyebrow_morphing_combiner.eyebrow_morphing_combiner_00 as EMC
    EMC.apply_grid_change = IPU.apply_grid_change


class Pick(torch.nn.Module):
    """Wrap a net returning a list; expose only `idxs` outputs."""
    def __init__(self, m, idxs):
        super().__init__()
        self.m = m
        self.idxs = idxs

    def forward(self, *a):
        o = self.m(*a)
        return tuple(o[i] for i in self.idxs)


def convert(module, inputs, out_path):
    module = module.eval()
    with torch.no_grad():
        ref = module(*inputs)
    ep = torch.export.export(module, tuple(inputs)).run_decompositions({})
    ml = ct.convert(ep, minimum_deployment_target=ct.target.macOS15,
                    compute_units=ct.ComputeUnit.CPU_AND_GPU,
                    compute_precision=ct.precision.FLOAT32)
    ml.save(str(out_path))
    names = [i.name for i in ml.get_spec().description.input]
    onames = [o.name for o in ml.get_spec().description.output]
    feed = {n: t.numpy() for n, t in zip(names, inputs)}
    out = ml.predict(feed)
    r0 = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()
    o0 = out[onames[0]]
    print(f"  {out_path.name}: in={names} out={onames} max_abs={np.abs(o0 - r0).max():.2e}")


def main():
    patch()
    from tha4.poser.modes import mode_07
    out = REPO / "data" / "tha4" / "coreml"
    out.mkdir(parents=True, exist_ok=True)
    W = "data/tha4"
    torch.manual_seed(0)

    convert(Pick(mode_07.load_eyebrow_decomposer(f"{W}/eyebrow_decomposer.pt"), [0, 3]),
            [torch.rand(1, 4, 128, 128)], out / "eyebrow_decomposer.mlpackage")
    convert(Pick(mode_07.load_eyebrow_morphing_combiner(f"{W}/eyebrow_morphing_combiner.pt"), [2]),
            [torch.rand(1, 4, 128, 128), torch.rand(1, 4, 128, 128), torch.rand(1, 12)],
            out / "eyebrow_morphing_combiner.mlpackage")
    convert(Pick(mode_07.load_face_morpher(f"{W}/face_morpher.pt"), [0]),
            [torch.rand(1, 4, 192, 192), torch.rand(1, 27)], out / "face_morpher.mlpackage")
    convert(Pick(mode_07.load_morpher_00(f"{W}/body_morpher.pt"), [0, 3]),
            [torch.rand(1, 4, 256, 256), torch.rand(1, 6)], out / "body_morpher.mlpackage")
    convert(Pick(mode_07.load_upscaler_02(f"{W}/upscaler.pt"), [0]),
            [torch.rand(1, 4, 512, 512), torch.rand(1, 4, 512, 512), torch.rand(1, 2, 512, 512), torch.rand(1, 6)],
            out / "upscaler.mlpackage")
    print("teacher CoreML models saved to", out)


if __name__ == "__main__":
    main()
