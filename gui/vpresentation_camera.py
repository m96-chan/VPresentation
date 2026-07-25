#!/usr/bin/env python3
"""VPresentation — real-time webcam VTuber.

Webcam -> MediaPipe FaceLandmarker -> THA4 pose converter -> poser -> avatar.

Backends:
  * CoreML (default, ~20-25 fps): in-process CoreML student model. Needs
    <char_dir>/coreml/ (build with tools/convert_coreml.py).
  * Rust serve (--serve): candle engine (student ~2.8fps, or --teacher <img>
    for arbitrary preprocessed characters ~8s/frame).

Run:  .venv/bin/python gui/vpresentation_camera.py [char_dir]
Keys: ESC / q to quit.
"""
import os
import sys
import time
import types
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

REPO = Path(__file__).resolve().parent.parent
SERVE_BIN = REPO / "target" / "release" / "serve"
MODEL = REPO / "data" / "thirdparty" / "mediapipe" / "face_landmarker.task"

# THA4 pose converter (stub wx: only its UI panel needs it; convert() is pure).
class _WxStub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return object

for _m in ("wx", "wx.lib", "wx.lib.newevent"):
    sys.modules[_m] = _WxStub(_m)
sys.path.insert(0, str(REPO / "third_party" / "tha4_src" / "src"))
sys.path.insert(0, str(REPO / "gui"))
from tha4.mocap.mediapipe_face_pose_converter_00 import MediaPoseFacePoseConverter00  # noqa: E402
from tha4.mocap.mediapipe_face_pose import MediaPipeFacePose  # noqa: E402


class CoreMLBackend:
    """In-process CoreML student poser -> HWC RGBA uint8. Fast (~real-time)."""

    def __init__(self, char_dir):
        from coreml_poser import CoreMLPoser, to_rgba_uint8
        self.poser = CoreMLPoser(char_dir)
        self._to_rgba = to_rgba_uint8
        self.device = "coreml"

    def render(self, pose):
        return self._to_rgba(self.poser.pose(pose))  # HWC RGBA uint8

    def close(self):
        pass


class TeacherCoreMLBackend:
    """In-process CoreML *teacher* poser -> HWC RGBA uint8. Poses any 512
    image (e.g. char.png) without distillation, ~2-3 fps."""

    def __init__(self, image_path, fast=True):
        from coreml_teacher_poser import CoreMLTeacherPoser
        self.poser = CoreMLTeacherPoser(image_path)
        self.fast = fast  # skip the 512 upscaler for ~1.6x speed
        self.device = "coreml-teacher" + ("-fast" if fast else "")

    def render(self, pose):
        return self.poser.render_rgba(pose, fast=self.fast)

    def close(self):
        pass


class ServeBackend:
    """Rust candle serve engine -> HWC RGBA uint8 (via PNG)."""

    def __init__(self, char_dir=None, teacher_image=None):
        if teacher_image:
            cmd = [str(SERVE_BIN), teacher_image]
        else:
            cmd = [str(SERVE_BIN), str(Path(char_dir) / "character.png"), "--student", str(char_dir)]
        self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"engine failed: {ready}")
        self.device = "rust:" + ready.split(" ", 1)[-1]
        self._tmp = tempfile.mkdtemp(prefix="vpres_cam_")
        self._n = 0

    def render(self, pose):
        self._n += 1
        out = os.path.join(self._tmp, f"f{self._n % 4}.png")
        self.proc.stdin.write(f"{out};" + ",".join(f"{v:.5f}" for v in pose) + "\n")
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().strip()
        if not resp.startswith("OK"):
            raise RuntimeError(resp)
        bgra = cv2.imread(out, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)

    def close(self):
        try:
            self.proc.stdin.write("quit\n"); self.proc.stdin.flush()
        except Exception:
            pass


def composite_on_bg(rgba, bg=(40, 30, 30)):
    """HWC RGBA uint8 -> BGR uint8 over a solid background (for cv2 display)."""
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = rgb * a + np.array(bg[::-1], np.float32).reshape(1, 1, 3) * (1 - a)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    args = sys.argv[1:]
    teacher_image = None
    teacher_fast = "--full" not in args
    if "--full" in args:
        args.remove("--full")
    use_serve = "--serve" in args
    if "--serve" in args:
        args.remove("--serve")
    if "--teacher" in args:
        i = args.index("--teacher"); teacher_image = args[i + 1]; del args[i:i + 2]; use_serve = True
    char_dir = args[0] if args else str(REPO / "data/character_models/lambda_00")

    if not MODEL.exists():
        sys.exit(f"missing MediaPipe model: {MODEL}")

    # Pick backend: CoreML if available and not forced to serve.
    if teacher_image:
        if not use_serve and (REPO / "data/tha4/coreml").exists():
            backend = TeacherCoreMLBackend(teacher_image, fast=teacher_fast)  # any char
        else:
            backend = ServeBackend(teacher_image=teacher_image)  # Rust, ~8s/frame
    elif not use_serve and (Path(char_dir) / "coreml").exists():
        backend = CoreMLBackend(char_dir)
    else:
        if not SERVE_BIN.exists():
            sys.exit("build engine: cargo build --release -p tha4 --bin serve")
        backend = ServeBackend(char_dir=char_dir)
    print(f"[camera] backend={backend.device} — look at the camera. ESC/q to quit.")

    converter = MediaPoseFacePoseConverter00()
    base = mp.tasks.BaseOptions(model_asset_path=str(MODEL))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base, running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("cannot open webcam (grant camera permission to the terminal)")

    last_pose = [0.0] * 45
    t_start = time.time()
    fps_t, fps_n = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            small = cv2.resize(rgb, (256, 192))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
            result = landmarker.detect_for_video(mp_image, int((time.time() - t_start) * 1000))
            if result.face_blendshapes:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                xform = (np.array(result.facial_transformation_matrixes[0], np.float32)
                         if result.facial_transformation_matrixes else np.eye(4, np.float32))
                try:
                    last_pose = converter.convert(MediaPipeFacePose(bs, xform))
                except Exception as e:
                    print("convert error:", e, file=sys.stderr)

            disp = composite_on_bg(backend.render(last_pose))
            cv2.imshow("VPresentation (ESC/q)", disp)

            fps_n += 1
            if time.time() - fps_t > 2.0:
                print(f"[camera] {fps_n / (time.time() - fps_t):.1f} fps")
                fps_t, fps_n = time.time(), 0
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        backend.close()


if __name__ == "__main__":
    main()
