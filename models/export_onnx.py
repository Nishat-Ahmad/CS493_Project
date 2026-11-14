"""YOLOv11 ONNX model export module.

Downloads official YOLOv11 checkpoints (yolo11n.pt, yolo11s.pt) and exports them
to ONNX format with static input dimensions (1, 3, 640, 640) for both:
  - FP32 (half=False)
  - FP16 (half=True / converted to float16)
Outputs saved to:
  - weights/yolo11n_fp32.onnx
  - weights/yolo11n_fp16.onnx
  - weights/yolo11s_fp32.onnx
  - weights/yolo11s_fp16.onnx
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import onnx
from ultralytics import YOLO

try:
    from onnxconverter_common import float16
    HAS_ONNX_CONVERTER = True
except ImportError:
    HAS_ONNX_CONVERTER = False


def export_fp32_onnx(model_variant: str, weights_dir: Path) -> Path:
    """Export YOLOv11 variant to FP32 ONNX.

    Args:
        model_variant: 'yolo11n' or 'yolo11s'.
        weights_dir: Output directory path.

    Returns:
        output_path: Path to generated FP32 ONNX model.
    """
    target_onnx = weights_dir / f"{model_variant}_fp32.onnx"
    if target_onnx.exists():
        print(f"[Export] {target_onnx.name} already exists. Skipping FP32 export.")
        return target_onnx

    pt_name = f"{model_variant}.pt"
    print(f"[Export] Loading {pt_name}...")
    model = YOLO(pt_name)

    print(f"[Export] Exporting {pt_name} to FP32 ONNX (static 1x3x640x640)...")
    exported = model.export(
        format="onnx",
        imgsz=640,
        half=False,
        dynamic=False,
        simplify=True,
        batch=1
    )

    exported_path = Path(exported)
    if exported_path.resolve() != target_onnx.resolve():
        shutil.copyfile(exported_path, target_onnx)

    print(f"[Export] FP32 ONNX saved to {target_onnx}")
    return target_onnx


def export_fp16_onnx(model_variant: str, weights_dir: Path, fp32_onnx_path: Path) -> Path:
    """Export or convert YOLOv11 variant to FP16 ONNX.

    Args:
        model_variant: 'yolo11n' or 'yolo11s'.
        weights_dir: Output directory path.
        fp32_onnx_path: Path to the base FP32 ONNX model.

    Returns:
        output_path: Path to generated FP16 ONNX model.
    """
    target_onnx = weights_dir / f"{model_variant}_fp16.onnx"
    if target_onnx.exists():
        print(f"[Export] {target_onnx.name} already exists. Skipping FP16 export.")
        return target_onnx

    pt_name = f"{model_variant}.pt"
    exported_direct = False

    try:
        print(f"[Export] Attempting direct FP16 export for {pt_name}...")
        model = YOLO(pt_name)
        exported = model.export(
            format="onnx",
            imgsz=640,
            half=True,
            dynamic=False,
            simplify=True,
            batch=1
        )
        exp_path = Path(exported)
        if exp_path.exists():
            shutil.copyfile(exp_path, target_onnx)
            exported_direct = True
            print(f"[Export] Direct FP16 ONNX saved to {target_onnx}")
    except Exception as e:
        print(f"[Export] Direct FP16 export raised notice: {e}. Falling back to graph conversion.")

    if not exported_direct:
        print(f"[Export] Converting {fp32_onnx_path.name} to FP16 via onnxconverter_common...")
        if not HAS_ONNX_CONVERTER:
            raise RuntimeError(
                "onnxconverter-common is required to convert FP32 ONNX to FP16. Please install it."
            )
        onnx_model = onnx.load(str(fp32_onnx_path))
        onnx_model_fp16 = float16.convert_float_to_float16(
            onnx_model,
            keep_io_types=False
        )
        onnx.save(onnx_model_fp16, str(target_onnx))
        print(f"[Export] Converted FP16 ONNX saved to {target_onnx}")

    return target_onnx


def run_exports(output_dir: Optional[Path | str] = None) -> Dict[str, Path]:
    """Run full export pipeline for YOLOv11n and YOLOv11s across FP32 and FP16.

    Args:
        output_dir: Directory where exported models are stored.

    Returns:
        exported_models: Dictionary mapping configuration key to model Path.
    """
    weights_path = Path(output_dir) if output_dir else REPO_ROOT / "weights"
    weights_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Path] = {}
    variants = ["yolo11n", "yolo11s"]

    for var in variants:
        fp32_path = export_fp32_onnx(var, weights_path)
        results[f"{var}_fp32"] = fp32_path

        fp16_path = export_fp16_onnx(var, weights_path, fp32_path)
        results[f"{var}_fp16"] = fp16_path

    return results


if __name__ == "__main__":
    run_exports()