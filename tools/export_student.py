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


_BASE_GRID_CACHE = {}


def _base_grid(h, w, dtype, device, align_corners):
    """The identity sampling grid, i.e. what `affine_grid` returns for the
    identity affine — but built directly, so no `affine_grid` node is traced.

    Layout matches `affine_grid`: `[..., 0]` is the x (width) coordinate and
    `[..., 1]` is y (height), both normalised to [-1, 1].
    """
    key = (h, w, str(dtype), str(device), bool(align_corners))
    grid = _BASE_GRID_CACHE.get(key)
    if grid is None:
        # Mirror ATen's `linspace_from_neg_one` exactly. The closed form
        # (2i + 1)/n - 1 is algebraically identical for align_corners=False but
        # rounds differently, and the resulting ~1e-7 grid error shows up as
        # ~3e-4 in the warped image.
        def axis(n):
            r = torch.linspace(-1.0, 1.0, n, dtype=dtype, device=device)
            return r if align_corners else r * (n - 1) / n

        xs, ys = axis(w), axis(h)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1, h, w, 2)
        # Pin the dtype. Constant folding during ONNX export otherwise happily
        # emits these as float64, and ORT's GridSample registration for
        # ai.onnx(16-19) constrains T2 to tensor(float) — a double grid makes
        # the node unresolvable ("Could not find an implementation").
        grid = grid.to(dtype).contiguous()
        _BASE_GRID_CACHE[key] = grid
    return grid


def patch_affine_grid():
    """Bake out `affine_grid`.

    torch.onnx only lowers `aten::affine_grid_generator` at opset >= 20, but
    ONNX Runtime Web's WebGPU EP only registers GridSample for ai.onnx(16-19).
    Both call sites here feed `affine_grid` a constant identity transform, so
    the grid depends on nothing but the spatial size — precomputing it removes
    the operator entirely and lets the model export at opset 17.
    """
    from torch.nn.functional import grid_sample
    from tha4.nn.image_processing_util import GridChangeApplier
    from tha4.nn.siren.morpher.siren_morpher_03 import SirenMorpher03

    def get_position_grid(self, n, image_size, device):
        grid = _base_grid(image_size, image_size, torch.float32, device, False)
        return grid.permute(0, 3, 1, 2).repeat(n, 1, 1, 1).to(torch.float32)

    SirenMorpher03.get_position_grid = get_position_grid

    def apply(self, grid_change, image, align_corners=False):
        n, c, h, w = image.shape
        grid_change = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        base_grid = _base_grid(h, w, grid_change.dtype, grid_change.device, align_corners)
        grid = (base_grid + grid_change).to(image.dtype)
        return grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=align_corners
        )

    GridChangeApplier.apply = apply


# ONNX Runtime Web's WebGPU EP registers GridSample for ai.onnx(16-19) only
# (js/web/docs/webgpu-operators.md). Exporting at opset 20 emits GridSample-20,
# which the WebGPU EP will not claim — the node silently falls back to the CPU
# kernel and the 512x512 warp dominates the frame time. Stay at 17.
DEFAULT_OPSET = 17


def export(module, name, dummies, out_dir, opset=DEFAULT_OPSET):
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
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    loaded = onnx.load(str(onnx_path))
    onnx.save_model(loaded, str(onnx_path), save_as_external_data=False)
    df = onnx_path.with_suffix(".onnx.data")
    if df.exists():
        df.unlink()
    ref = {f"in{i}": d.contiguous() for i, d in enumerate(dummies)}
    ref["out0"] = out_list[0].contiguous()
    save_file(ref, str(ref_dir / f"{name}.safetensors"))

    ops = sorted({n.op_type for n in loaded.graph.node})
    imported = {i.domain or "ai.onnx": i.version for i in loaded.opset_import}
    print(f"[student] {name}: {len(dummies)} inputs -> out0 {tuple(out_list[0].shape)}")
    print(f"          opset {imported}  ops: {', '.join(ops)}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a.split("=")[0]: a.split("=")[-1] for a in sys.argv[1:] if a.startswith("--")}
    char_dir = Path(args[0] if args else "data/character_models/lambda_00")
    opset = int(flags.get("--opset", DEFAULT_OPSET))

    patch_siren()
    if opset < 20:
        patch_affine_grid()
    from tha4.poser.modes import mode_14

    fm = mode_14.load_face_morpher(str(char_dir / "face_morpher.pt"))
    export(fm, "student_face_morpher", [torch.rand(1, 39)], char_dir, opset)

    bm = mode_14.load_body_morpher(str(char_dir / "body_morpher.pt"))
    export(bm, "student_body_morpher", [torch.rand(1, 4, 512, 512), torch.rand(1, 45)], char_dir, opset)


if __name__ == "__main__":
    main()
