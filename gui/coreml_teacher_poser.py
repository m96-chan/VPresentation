"""THA4 teacher (general poser) via CoreML — poses ANY 512 image (e.g. char.png)
without per-character distillation. ~2-3 fps on an M3.

Ports FiveStepPoserComputationProtocol (mode_07): eyebrow decomposer ->
combiner -> face morpher (eyebrow pasted) -> full composite -> body morpher
(256) -> upscaler (512). Pose = 45: eyebrow[0:12], face[12:39], rot[39:45].
"""
from pathlib import Path

import numpy as np
import cv2
import coremltools as ct

from coreml_poser import load_thaa_image, to_rgba_uint8  # reused I/O


def _resize(t, size):
    """(1,C,H,W) float -> bilinear resize to (1,C,size,size)."""
    hwc = np.transpose(t[0], (1, 2, 0))
    r = cv2.resize(hwc, (size, size), interpolation=cv2.INTER_LINEAR)
    if r.ndim == 2:
        r = r[:, :, None]
    return np.transpose(r, (2, 0, 1))[None].astype(np.float32)


class _Net:
    def __init__(self, path):
        # ALL lets CoreML also use the Neural Engine where ops allow.
        self.m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        self.ins = [i.name for i in self.m.get_spec().description.input]
        self.outs = [o.name for o in self.m.get_spec().description.output]

    def __call__(self, *arrays):
        out = self.m.predict({n: a for n, a in zip(self.ins, arrays)})
        return [out[o] for o in self.outs]


class CoreMLTeacherPoser:
    def __init__(self, image_path, coreml_dir="data/tha4/coreml"):
        d = Path(coreml_dir)
        self.ebd = _Net(d / "eyebrow_decomposer.mlpackage")
        self.comb = _Net(d / "eyebrow_morphing_combiner.mlpackage")
        self.fm = _Net(d / "face_morpher.mlpackage")
        self.body = _Net(d / "body_morpher.mlpackage")
        self.up = _Net(d / "upscaler.mlpackage")
        self.image = load_thaa_image(image_path)  # (1,4,512,512)
        # eyebrow decomposer only depends on the (fixed) character image -> cache it.
        self._eyebrow_layer, self._bg_layer = self.ebd(self.image[:, :, 64:192, 192:320].copy())

    def pose(self, pose, fast=False):
        """fast=True skips the 512 upscaler (bilinear-upscales the body morpher's
        256 output instead) — ~2x faster, slightly softer detail."""
        pose = np.asarray(pose, np.float32)
        eb = pose[0:12][None]
        face = pose[12:39][None]
        rot = pose[39:45][None]
        img = self.image
        eyebrow_layer, bg_layer = self._eyebrow_layer, self._bg_layer
        # 2. combiner -> eyebrow_morphed (EYEBROW_IMAGE_NO_COMBINE_ALPHA)
        eyebrow_morphed = self.comb(bg_layer, eyebrow_layer, eb)[0]
        # 3. face morpher: 192 crop with eyebrow pasted at (32,32)
        face_crop = img[:, :, 32:224, 160:352].copy()
        face_crop[:, :, 32:160, 32:160] = eyebrow_morphed
        face_morphed = self.fm(face_crop, face)[0]
        # 4. full-res composite
        face_full = img.copy()
        face_full[:, :, 32:224, 160:352] = face_morphed
        # 5-6. body morpher at 256 -> merged, grid_change
        merged, grid_change = self.body(_resize(face_full, 256), rot)
        if fast:
            # skip the heavy 512 upscaler; bilinear-upscale the posed 256 image.
            return _resize(merged, 512)
        coarse_posed = _resize(merged, 512)
        coarse_grid = _resize(grid_change, 512)
        # 7. upscaler -> final 512 image
        return self.up(face_full, coarse_posed, coarse_grid, rot)[0]

    def render_rgba(self, pose, fast=False):
        return to_rgba_uint8(self.pose(pose, fast=fast))
