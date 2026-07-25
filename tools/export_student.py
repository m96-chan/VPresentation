#!/usr/bin/env python3
"""Export THA4 *student* character-model nets (mode_14) to ONNX + reference I/O.

Student nets are per-character, SIREN-based, and much smaller/faster than the
teacher — this is THA4's real-time puppeteer path. PyTorch is convert/test-only.

Usage (repo root, .venv active):  python tools/export_student.py <char_dir>
  <char_dir> e.g. data/character_models/lambda_00
"""
import os
import sys
from pathlib import Path

import torch
import onnx
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
THA4_SRC = Path(os.environ.get("THA4_SRC", REPO / "third_party" / "tha4_src"))
sys.path.insert(0, str(THA4_SRC / "src"))

torch.manual_seed(0)


def patch_siren():
    from tha4.nn.siren.vanilla.siren import SineLinearLayer

    def sine_forward(self, x):
        omega = torch.as_tensor(self.omega_0, dtype=x.dtype, device=x.device)
        return torch.sin(self.linear(x) * omega)

    SineLinearLayer.forward = sine_forward


def export(module, name, dummies, out_dir):
    onnx_dir = out_dir / "onnx"
    ref_dir = out_dir / "reference"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        outputs = module(*dummies)
    out_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    onnx_path = onnx_dir / f"{name}.onnx"
    torch.onnx.export(
        module, tuple(dummies), str(onnx_path),
        input_names=[f"in{i}" for i in range(len(dummies))],
        output_names=[f"out{i}" for i in range(len(out_list))],
        opset_version=20, do_constant_folding=True, dynamo=False,
    )
    loaded = onnx.load(str(onnx_path))
    onnx.save_model(loaded, str(onnx_path), save_as_external_data=False)
    df = onnx_path.with_suffix(".onnx.data")
    if df.exists():
        df.unlink()
    ref = {f"in{i}": d.contiguous() for i, d in enumerate(dummies)}
    ref["out0"] = out_list[0].contiguous()
    save_file(ref, str(ref_dir / f"{name}.safetensors"))
    print(f"[student] {name}: {len(dummies)} inputs -> out0 {tuple(out_list[0].shape)}")


def main():
    char_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/character_models/lambda_00")
    patch_siren()
    from tha4.poser.modes import mode_14

    fm = mode_14.load_face_morpher(str(char_dir / "face_morpher.pt"))
    export(fm, "student_face_morpher", [torch.rand(1, 39)], char_dir)

    bm = mode_14.load_body_morpher(str(char_dir / "body_morpher.pt"))
    export(bm, "student_body_morpher", [torch.rand(1, 4, 512, 512), torch.rand(1, 45)], char_dir)


if __name__ == "__main__":
    main()
