#!/usr/bin/env python3
"""VPresentation — real-time webcam VTuber (candle + Metal).

Webcam -> MediaPipe FaceLandmarker -> THA4 pose converter -> Rust `serve`
engine (student model) -> live character. ~2.8 fps on M3 (render-bound).

Run:  .venv/bin/python gui/vpresentation_camera.py [char_dir]
      char_dir defaults to data/character_models/lambda_00
Press ESC / q to quit.
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

# --- import THA4's pose converter (stub wx: only its UI panel needs it) ---
class _WxStub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return object

for _m in ("wx", "wx.lib", "wx.lib.newevent"):
    sys.modules[_m] = _WxStub(_m)
sys.path.insert(0, str(REPO / "third_party" / "tha4_src" / "src"))
from tha4.mocap.mediapipe_face_pose_converter_00 import MediaPoseFacePoseConverter00  # noqa: E402
from tha4.mocap.mediapipe_face_pose import MediaPipeFacePose  # noqa: E402


class Engine:
    """Persistent Rust serve process: pose (45 floats) -> rendered RGBA PNG.

    Student mode (fast, ~2.8fps) needs a distilled character dir. Teacher mode
    (slow, ~8s/frame) poses any preprocessed 512 image (e.g. char.png)."""

    def __init__(self, char_dir=None, teacher_image=None):
        if teacher_image:
            cmd = [str(SERVE_BIN), teacher_image]  # teacher: poses arbitrary image
        else:
            char_png = str(Path(char_dir) / "character.png")
            cmd = [str(SERVE_BIN), char_png, "--student", char_dir]
        self.proc = subprocess.Popen(
            cmd, cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"engine failed: {ready}")
        self.device = ready.split(" ", 1)[-1]
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
        return out

    def close(self):
        try:
            self.proc.stdin.write("quit\n"); self.proc.stdin.flush()
        except Exception:
            pass


def composite_on_bg(path, bg=(30, 30, 40)):
    """Load an RGBA PNG and composite over a solid BGR background for display."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGRA
    if img is None or img.shape[2] < 4:
        return None
    rgb = img[:, :, :3].astype(np.float32)
    a = (img[:, :, 3:4].astype(np.float32)) / 255.0
    bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
    out = rgb * a + bg_arr * (1 - a)
    return out.astype(np.uint8)


def main():
    args = sys.argv[1:]
    teacher_image = None
    if "--teacher" in args:
        i = args.index("--teacher")
        teacher_image = args[i + 1]
        del args[i:i + 2]
    char_dir = args[0] if args else str(REPO / "data/character_models/lambda_00")
    if not SERVE_BIN.exists():
        sys.exit("build engine: cargo build --release -p tha4 --bin serve")
    if not MODEL.exists():
        sys.exit(f"missing MediaPipe model: {MODEL}")

    converter = MediaPoseFacePoseConverter00()
    base = mp.tasks.BaseOptions(model_asset_path=str(MODEL))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("cannot open webcam (grant camera permission to the terminal)")

    if teacher_image:
        engine = Engine(teacher_image=teacher_image)
        print(f"[camera] TEACHER mode ({teacher_image}) on {engine.device} — ~8s/frame, be patient.")
    else:
        engine = Engine(char_dir=char_dir)
        print(f"[camera] engine on {engine.device}; look at the camera. ESC/q to quit.")

    last_pose = [0.0] * 45
    t_start = time.time()
    frame_i = 0
    fps_t, fps_n = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            small = cv2.resize(rgb, (256, 192))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
            ts = int((time.time() - t_start) * 1000)
            result = landmarker.detect_for_video(mp_image, ts)

            if result.face_blendshapes:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                xform = np.array(result.facial_transformation_matrixes[0], dtype=np.float32) \
                    if result.facial_transformation_matrixes else np.eye(4, dtype=np.float32)
                try:
                    last_pose = converter.convert(MediaPipeFacePose(bs, xform))
                except Exception as e:
                    print("convert error:", e, file=sys.stderr)

            out_path = engine.render(last_pose)
            disp = composite_on_bg(out_path)
            if disp is not None:
                cv2.imshow("VPresentation (ESC/q to quit)", disp)

            fps_n += 1
            if time.time() - fps_t > 2.0:
                print(f"[camera] {fps_n / (time.time() - fps_t):.1f} fps")
                fps_t, fps_n = time.time(), 0

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            frame_i += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()


if __name__ == "__main__":
    main()
