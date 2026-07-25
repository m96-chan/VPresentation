"""Real-time THA4 student poser via CoreML (mode_14).

face_morpher(pose[:39]) -> 128 face, composite into the 512 character image,
body_morpher(image, pose) -> posed 512 image. ~30fps on an M3.
"""
from pathlib import Path

import numpy as np
import coremltools as ct
from PIL import Image

NUM_POSE = 45


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


def load_thaa_image(path, size=512):
    """PNG -> THA4 tensor (1,4,size,size) in [-1,1] (srgb->linear, premultiplied)."""
    img = Image.open(path).convert("RGBA")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0  # HWC RGBA
    alpha = np.clip(a[:, :, 3:4], 0.0, 1.0)
    rgb = _srgb_to_linear(a[:, :, :3]) * alpha
    hwc = np.concatenate([rgb, alpha], axis=2) * 2.0 - 1.0
    return np.transpose(hwc, (2, 0, 1))[None].astype(np.float32)


def to_rgba_uint8(arr):
    """THA4 tensor (1,4,H,W) in [-1,1] -> HWC RGBA uint8 (straight alpha, srgb)."""
    x = (arr[0] + 1.0) * 0.5
    x = np.transpose(x, (1, 2, 0))  # HWC
    a = np.clip(x[:, :, 3:4], 0.0, 1.0)
    rgb = np.where(a < 1e-5, 0.0, np.clip(x[:, :, :3] / np.maximum(a, 1e-5), 0.0, 1.0))
    rgb = _linear_to_srgb(rgb)
    out = np.concatenate([rgb, a], axis=2)
    return (out * 255.0).round().astype(np.uint8)


class CoreMLPoser:
    def __init__(self, char_dir):
        char_dir = Path(char_dir)
        cm = char_dir / "coreml"
        self.face = ct.models.MLModel(str(cm / "face_morpher.mlpackage"))
        self.body = ct.models.MLModel(str(cm / "body_morpher.mlpackage"))
        self.face_in = self.face.get_spec().description.input[0].name
        b_in = [i.name for i in self.body.get_spec().description.input]
        self.body_img_in, self.body_pose_in = b_in[0], b_in[1]
        self.image = load_thaa_image(str(char_dir / "character.png"))

    def pose(self, pose):
        pose = np.asarray(pose, dtype=np.float32).reshape(1, NUM_POSE)
        face = list(self.face.predict({self.face_in: pose[:, :39]}).values())[0]
        comp = self.image.copy()
        comp[:, :, 80:208, 192:320] = face  # composite at centre (256,144)
        out = self.body.predict({self.body_img_in: comp, self.body_pose_in: pose})
        return list(out.values())[0]  # (1,4,512,512)
