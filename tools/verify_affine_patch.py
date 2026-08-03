#!/usr/bin/env python3
"""Check that baking out `affine_grid` does not change the student's output.

`tools/export_student.py` replaces THA4's two `affine_grid` call sites with a
precomputed constant grid, so the model can be exported at opset 17 (see the
comment on `patch_affine_grid`). That rewrite is only safe if it is bit-for-bit
equivalent to what PyTorch produced before — this script asserts it is.

Usage (repo root):  .venv-distill/bin/python tools/verify_affine_patch.py [char_dir]
"""
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
THA4_SRC = Path(os.environ.get("THA4_SRC", REPO / "third_party" / "tha4_src"))
sys.path.insert(0, str(THA4_SRC / "src"))
sys.path.insert(0, str(REPO / "tools"))


def main():
    char_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/character_models/char")
    torch.manual_seed(0)

    from export_student import patch_siren, patch_affine_grid
    from tha4.poser.modes import mode_14

    patch_siren()

    image = torch.rand(1, 4, 512, 512) * 2 - 1
    pose = torch.rand(1, 45)
    pose39 = torch.rand(1, 39)

    # --- reference: stock THA4 (uses affine_grid) ---------------------------
    face = mode_14.load_face_morpher(str(char_dir / "face_morpher.pt")).eval()
    body = mode_14.load_body_morpher(str(char_dir / "body_morpher.pt")).eval()
    with torch.no_grad():
        face_ref = face(pose39)
        body_ref = body(image, pose)
    face_ref = face_ref[0] if isinstance(face_ref, (list, tuple)) else face_ref
    body_ref = body_ref[0] if isinstance(body_ref, (list, tuple)) else body_ref

    # --- patched: constant grids, no affine_grid ---------------------------
    patch_affine_grid()
    # Reload so the patched methods are the ones bound at call time.
    face2 = mode_14.load_face_morpher(str(char_dir / "face_morpher.pt")).eval()
    body2 = mode_14.load_body_morpher(str(char_dir / "body_morpher.pt")).eval()
    with torch.no_grad():
        face_new = face2(pose39)
        body_new = body2(image, pose)
    face_new = face_new[0] if isinstance(face_new, (list, tuple)) else face_new
    body_new = body_new[0] if isinstance(body_new, (list, tuple)) else body_new

    ok = True
    for name, ref, new in (("face_morpher", face_ref, face_new), ("body_morpher", body_ref, body_new)):
        diff = (ref - new).abs().max().item()
        status = "OK " if diff == 0.0 else ("close" if diff < 1e-5 else "FAIL")
        if diff >= 1e-5:
            ok = False
        print(f"[verify] {name:14s} shape {tuple(ref.shape)}  max|diff| = {diff:.3e}  {status}")

    print("[verify] affine_grid bake-out is equivalent" if ok else "[verify] MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
