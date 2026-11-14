"""INT8 Post-Training Quantization module using ONNX Runtime.

Implements an ONNX Runtime CalibrationDataReader to calibrate YOLOv11 models on
representative traffic images from data/calibration/, then performs static
quantization with QDQ format, QInt8 activations and weights, and per-channel scaling.
Generates:
  - weights/yolo11n_int8.onnx
  - weights/yolo11s_int8.onnx
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod
)


class ImageCalibrationDataReader(CalibrationDataReader):
    """Feeds preprocessed calibration frames to ONNX Runtime quantizer."""

    def __init__(
        self,
        calibration_dir: Path | str,
        input_name: str = "images",
        target_size: tuple[int, int] = (640, 640)
    ) -> None:
        """Initialize calibration data reader.

        Args:
            calibration_dir: Directory containing calibration images.
            input_name: Input tensor name of the ONNX model graph.
            target_size: (width, height) expected by model.
        """
        super().__init__()
        self.calib_dir = Path(calibration_dir)
        self.input_name = input_name
        self.target_size = target_size

        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths: List[Path] = sorted(
            [p for p in self.calib_dir.iterdir() if p.suffix.lower() in img_exts]
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No calibration images found in {self.calib_dir}. "
                f"Run data/generate_mock_data.py first to create calibration frames."
            )

        self.current_idx = 0

    def preprocess(self, img_path: Path) -> np.ndarray:
        """Read image, convert BGR to RGB, resize, normalize to [0, 1], convert to NCHW float32.

        Args:
            img_path: Path to image file.

        Returns:
            tensor: Preprocessed array of shape (1, 3, 640, 640) float32.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image at {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW -> NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Return next input dictionary or None when calibration completes."""
        if self.current_idx >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.current_idx]
        self.current_idx += 1
        tensor = self.preprocess(img_path)
        return {self.input_name: tensor}

    def rewind(self) -> None:
        """Reset iteration index."""
        self.current_idx = 0


def quantize_model_int8(
    model_fp32_path: Path | str,
    output_int8_path: Path | str,
    calibration_dir: Path | str = "data/calibration"
) -> Path:
    """Quantize an FP32 ONNX model to INT8 using QDQ format and per-channel weights.

    Args:
        model_fp32_path: Path to input FP32 ONNX model.
        output_int8_path: Destination path for quantized INT8 ONNX model.
        calibration_dir: Path to directory containing calibration images.

    Returns:
        output_path: Path to generated INT8 model.
    """
    model_fp32_path = Path(model_fp32_path)
    output_int8_path = Path(output_int8_path)
    calibration_dir = Path(calibration_dir)

    if not model_fp32_path.exists():
        raise FileNotFoundError(f"Input model {model_fp32_path} does not exist.")

    output_int8_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine model input name dynamically from graph
    onnx_model = onnx.load(str(model_fp32_path))
    input_name = onnx_model.graph.input[0].name
    print(f"[Quantize] Identified model input name: '{input_name}'")

    data_reader = ImageCalibrationDataReader(
        calibration_dir=calibration_dir,
        input_name=input_name,
        target_size=(640, 640)
    )

    print(f"[Quantize] Quantizing {model_fp32_path.name} -> {output_int8_path.name}...")
    quantize_static(
        model_input=str(model_fp32_path),
        model_output=str(output_int8_path),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul"],
    )

    print(f"[Quantize] Successfully generated {output_int8_path}")
    return output_int8_path


def run_int8_quantization(
    weights_dir: Optional[Path | str] = None,
    calibration_dir: Optional[Path | str] = None
) -> Dict[str, Path]:
    """Execute INT8 quantization for both YOLOv11n and YOLOv11s.

    Args:
        weights_dir: Directory containing FP32 weights and target for INT8 weights.
        calibration_dir: Directory containing calibration images.

    Returns:
        results: Mapping of model key to output path.
    """
    weights_path = Path(weights_dir) if weights_dir else REPO_ROOT / "weights"
    calib_path = Path(calibration_dir) if calibration_dir else REPO_ROOT / "data" / "calibration"

    results: Dict[str, Path] = {}
    for var in ["yolo11n", "yolo11s"]:
        fp32_file = weights_path / f"{var}_fp32.onnx"
        int8_file = weights_path / f"{var}_int8.onnx"

        if not fp32_file.exists():
            print(f"[Quantize] Warning: {fp32_file} not found. Skipping {var} INT8 quantization.")
            continue

        quantize_model_int8(
            model_fp32_path=fp32_file,
            output_int8_path=int8_file,
            calibration_dir=calib_path
        )
        results[f"{var}_int8"] = int8_file

    return results


if __name__ == "__main__":
    run_int8_quantization()