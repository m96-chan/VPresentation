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

# (loader_name, weight_file, input_shape) per network.
NETWORKS = {
    "eyebrow_decomposer": ("load_eyebrow_decomposer", "eyebrow_decomposer.pt", (1, 4, 128, 128)),
}


def export_one(name: str):
    from tha4.poser.modes import mode_07
    loader_name, weight_file, in_shape = NETWORKS[name]
    loader = getattr(mode_07, loader_name)
    module = loader(str(WEIGHTS / weight_file))
    module.eval()

    dummy = torch.rand(*in_shape, dtype=torch.float32)
    with torch.no_grad():
        outputs = module(dummy)
    if isinstance(outputs, (list, tuple)):
        out0 = outputs[0]
    else:
        out0 = outputs

    onnx_path = ONNX_DIR / f"{name}.onnx"
    torch.onnx.export(
        module, (dummy,), str(onnx_path),
        input_names=["image"],
        output_names=[f"out{i}" for i in range(len(outputs) if isinstance(outputs, (list, tuple)) else 1)],
        opset_version=16,  # GridSample requires >=16
        do_constant_folding=True,
    )
    # candle-onnx reads initializers inline only; collapse any external-data
    # tensors back into a single self-contained .onnx file.
    loaded = onnx.load(str(onnx_path))  # pulls in the sibling .onnx.data
    onnx.save_model(loaded, str(onnx_path), save_as_external_data=False)
    data_file = onnx_path.with_suffix(".onnx.data")
    if data_file.exists():
        data_file.unlink()
    # Save reference input + first output for candle numeric parity tests.
    save_file({"image": dummy.contiguous(), "out0": out0.contiguous()},
              str(REF_DIR / f"{name}.safetensors"))
    print(f"[export] {name}: onnx -> {onnx_path.name}, ref out0 shape {tuple(out0.shape)}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(NETWORKS) if which == "all" else [which]
    for n in names:
        export_one(n)


if __name__ == "__main__":
    main()
