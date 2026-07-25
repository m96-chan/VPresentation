#!/usr/bin/env python3
"""Export THA4 networks to ONNX + dump reference I/O tensors.

PyTorch is used here ONLY offline for conversion/testing — it is never a
runtime dependency of VPresentation (the runtime is Rust + candle). See
issue #4 / the project memory.

Usage (from repo root, with .venv active and weights fetched):
    python tools/export_onnx.py eyebrow_decomposer
    python tools/export_onnx.py all
"""
import sys
import os
from pathlib import Path

import torch
import onnx
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
THA4_SRC = REPO / "third_party" / "tha4_src"          # THA4 python source (see note below)
WEIGHTS = REPO / "data" / "tha4"
ONNX_DIR = WEIGHTS / "onnx"
REF_DIR = WEIGHTS / "reference"

# Allow overriding the THA4 source location via env (we clone it to scratch).
THA4_SRC = Path(os.environ.get("THA4_SRC", THA4_SRC))
sys.path.insert(0, str(THA4_SRC / "src"))

ONNX_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)

# (loader_name, weight_file, [input_shapes...]) per network.
# Input shapes/order match FiveStepPoserComputationProtocol.compute_output.
NETWORKS = {
    "eyebrow_decomposer": ("load_eyebrow_decomposer", "eyebrow_decomposer.pt",
                           [(1, 4, 128, 128)]),
    "eyebrow_morphing_combiner": ("load_eyebrow_morphing_combiner", "eyebrow_morphing_combiner.pt",
                                  [(1, 4, 128, 128), (1, 4, 128, 128), (1, 12)]),
    "face_morpher": ("load_face_morpher", "face_morpher.pt",
                     [(1, 4, 192, 192), (1, 27)]),
    "body_morpher": ("load_morpher_00", "body_morpher.pt",
                     [(1, 4, 256, 256), (1, 6)]),
    "upscaler": ("load_upscaler_02", "upscaler.pt",
                 [(1, 4, 512, 512), (1, 4, 512, 512), (1, 2, 512, 512), (1, 6)]),
}


def apply_export_patches():
    """Make THA4 forward passes exportable by the dynamo ONNX exporter.

    SIREN's `omega_0 * linear(x)` is a tensor*python-float (aten.mul.Scalar),
    which the dynamo exporter can't lower. Rewrite it as tensor*tensor.
    """
    from tha4.nn.siren.vanilla.siren import SineLinearLayer

    def sine_forward(self, x):
        omega = torch.as_tensor(self.omega_0, dtype=x.dtype, device=x.device)
        return torch.sin(self.linear(x) * omega)

    SineLinearLayer.forward = sine_forward


def export_one(name: str):
    apply_export_patches()
    from tha4.poser.modes import mode_07
    loader_name, weight_file, in_shapes = NETWORKS[name]
    loader = getattr(mode_07, loader_name)
    module = loader(str(WEIGHTS / weight_file))
    module.eval()

    dummies = [torch.rand(*s, dtype=torch.float32) for s in in_shapes]
    with torch.no_grad():
        outputs = module(*dummies)
    out_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    out0 = out_list[0]

    input_names = [f"in{i}" for i in range(len(dummies))]
    onnx_path = ONNX_DIR / f"{name}.onnx"
    torch.onnx.export(
        module, tuple(dummies), str(onnx_path),
        input_names=input_names,
        output_names=[f"out{i}" for i in range(len(out_list))],
        opset_version=20,  # legacy exporter: affine_grid needs >=20; handles SIREN scalar-mul
        do_constant_folding=True,
        dynamo=False,
    )
    # candle-onnx reads initializers inline only; collapse any external-data
    # tensors back into a single self-contained .onnx file.
    loaded = onnx.load(str(onnx_path))  # pulls in the sibling .onnx.data
    onnx.save_model(loaded, str(onnx_path), save_as_external_data=False)
    data_file = onnx_path.with_suffix(".onnx.data")
    if data_file.exists():
        data_file.unlink()
    # Save reference inputs + first output for candle numeric parity tests.
    ref = {f"in{i}": d.contiguous() for i, d in enumerate(dummies)}
    ref["out0"] = out0.contiguous()
    save_file(ref, str(REF_DIR / f"{name}.safetensors"))
    print(f"[export] {name}: onnx -> {onnx_path.name}, {len(dummies)} inputs, out0 {tuple(out0.shape)}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(NETWORKS) if which == "all" else [which]
    for n in names:
        export_one(n)


if __name__ == "__main__":
    main()
