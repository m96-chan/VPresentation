#!/usr/bin/env python3
"""VPresentation — real-time webcam VTuber.

Webcam -> MediaPipe FaceLandmarker -> THA4 pose converter -> poser -> avatar.

Backends, in the order they are picked:
  * CoreML (~20-25 fps): in-process CoreML student model. Needs
    <char_dir>/coreml/ (build with tools/convert_coreml.py). Apple only.
  * PyTorch/CUDA (~70 fps): the same student in torch. This is the one that
    works on the distillation box, where CoreML does not exist.
  * Rust serve (--serve): candle engine (student ~2.8fps, or --teacher <img>
    for arbitrary preprocessed characters ~8s/frame).

Run:  .venv-distill/bin/python gui/vpresentation_camera.py [char_dir]
Keys: ESC / q to quit.

--record <dir> writes the composited frames as they are displayed, so a demo
can be looked at afterwards instead of only while it is on screen.
"""
import os
import sys
import time
import types
import threading
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


class TorchBackend:
    """In-process PyTorch student poser -> HWC RGBA uint8.

    For the distillation box: CoreML is Apple-only and the Rust engine runs the
    student at ~2.8fps, but the same student in PyTorch on CUDA renders at
    ~72fps here — fast enough to check a freshly distilled character on the
    machine that just distilled it.
    """

    def __init__(self, char_dir, device="cuda"):
        import torch
        from tha4.charmodel.character_model import CharacterModel
        from tha4.shion.base.image_util import convert_pytorch_image_to_zero_to_one_numpy_image

        self._torch = torch
        self._to_numpy = convert_pytorch_image_to_zero_to_one_numpy_image
        self._device = torch.device(device)
        model = CharacterModel.load(str(Path(char_dir) / "character_model.yaml"))
        self.poser = model.get_poser(self._device)
        self.image = model.get_character_image(self._device)
        self.device = f"torch:{device}"

    def render(self, pose):
        torch = self._torch
        pose_t = torch.tensor(pose, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            posed = self.poser.pose(self.image, pose_t)[0]
        # [-1,1] premultiplied -> straight RGBA uint8, which is what the
        # compositor below expects (same layout the other backends return).
        rgba = np.clip(self._to_numpy(posed), 0.0, 1.0)
        rgb, alpha = rgba[:, :, :3], rgba[:, :, 3:4]
        rgb = np.clip(np.divide(rgb, np.maximum(alpha, 1e-6)), 0.0, 1.0) ** (1 / 2.2)
        return (np.concatenate([rgb, alpha], axis=2) * 255).astype(np.uint8)

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


class Tracker(threading.Thread):
    """Runs webcam capture + MediaPipe face tracking on its own thread, so
    rendering never waits on tracking. Exposes the latest 45-dim pose."""

    def __init__(self, landmarker, converter):
        super().__init__(daemon=True)
        self.landmarker = landmarker
        self.converter = converter
        self.cap = cv2.VideoCapture(0)
        self._pose = [0.0] * 45
        self._lock = threading.Lock()
        self._running = self.cap.isOpened()
        self.opened = self.cap.isOpened()
        self._t0 = time.time()

    def run(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            rgb = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            small = cv2.resize(rgb, (256, 192))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
            res = self.landmarker.detect_for_video(mp_image, int((time.time() - self._t0) * 1000))
            if res.face_blendshapes:
                bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
                xform = (np.array(res.facial_transformation_matrixes[0], np.float32)
                         if res.facial_transformation_matrixes else np.eye(4, np.float32))
                try:
                    p = self.converter.convert(MediaPipeFacePose(bs, xform))
                    with self._lock:
                        self._pose = p
                except Exception as e:
                    print("convert error:", e, file=sys.stderr)

    def pose(self):
        with self._lock:
            return list(self._pose)

    def stop(self):
        self._running = False
        self.cap.release()


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


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
    record_dir = None
    if "--record" in args:
        i = args.index("--record"); record_dir = args[i + 1]; del args[i:i + 2]
        os.makedirs(record_dir, exist_ok=True)
    max_frames = None
    if "--frames" in args:
        i = args.index("--frames"); max_frames = int(args[i + 1]); del args[i:i + 2]
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
    elif not use_serve and _cuda_available():
        backend = TorchBackend(char_dir)
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

    # Face tracking runs on its own thread so rendering never blocks on it;
    # the render rate becomes the effective fps.
    tracker = Tracker(landmarker, converter)
    if not tracker.opened:
        sys.exit("cannot open webcam (grant camera permission to the terminal)")
    tracker.start()

    fps_t, fps_n, total = time.time(), 0, 0
    headless = record_dir is not None and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    try:
        while True:
            disp = composite_on_bg(backend.render(tracker.pose()))
            if not headless:
                cv2.imshow("VPresentation (ESC/q)", disp)
            if record_dir:
                cv2.imwrite(os.path.join(record_dir, f"frame_{total:05d}.png"), disp)
            fps_n += 1
            total += 1
            if time.time() - fps_t > 2.0:
                print(f"[camera] {fps_n / (time.time() - fps_t):.1f} fps (render)")
                fps_t, fps_n = time.time(), 0
            if max_frames is not None and total >= max_frames:
                break
            if not headless and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
        backend.close()


if __name__ == "__main__":
    main()
