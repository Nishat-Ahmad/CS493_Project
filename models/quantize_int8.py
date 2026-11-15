"""INT8 Post-Training Quantization module with Detection Head Protection.

Performs static QDQ INT8 quantization for YOLO26 models using ONNX Runtime.
Key Features:
  1. Calibration Extraction: Automatically extracts representative real-world
     traffic frames across day, dusk, night, low-light, and control sequences from TU-DAT.
  2. Detection Head Protection: Excludes the sensitive bounding-box regression and DFL
     layers (model.23 / Detect heads) from integer quantization to prevent coordinate
     clipping and restore mAP@50 localization accuracy.
  3. Per-Channel Quantization: Uses QInt8 weights with per-channel scaling and QInt8
     activations for CPUExecutionProvider edge execution.
Generates:
  - weights/yolo26n_int8.onnx
  - weights/yolo26s_int8.onnx
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

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


def extract_calibration_frames(
    tudat_dir: Path | str = "data/tudat",
    output_dir: Path | str = "data/calibration",
    target_count: int = 150
) -> Path:
    """Extract diverse, representative calibration frames from TU-DAT real video sequences.

    Samples frames evenly across positive accident sequences (day, dusk, night, weather)
    and negative normal traffic control sequences.

    Args:
        tudat_dir: Path to TU-DAT dataset root containing videos/.
        output_dir: Path to output directory for saving extracted calibration JPEGs.
        target_count: Target number of calibration frames to extract (default 150).

    Returns:
        output_dir: Path to populated calibration frames directory.
    """
    tudat_path = Path(tudat_dir)
    out_path = Path(output_dir)

    # If output directory already has >= 80 real frames, reuse it
    if out_path.exists():
        existing_imgs = [p for p in out_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        # Check if they are real frames (file size > 40KB) rather than mock frames
        if len(existing_imgs) >= 80 and existing_imgs[0].stat().st_size > 40000:
            print(f"[Calib] Found {len(existing_imgs)} existing real calibration frames in {out_path}. Reusing.")
            return out_path

    out_path.mkdir(parents=True, exist_ok=True)
    # Clear any old/synthetic calibration frames
    for old_file in out_path.glob("*.*"):
        if old_file.is_file():
            old_file.unlink()

    videos_dir = tudat_path / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"TU-DAT videos directory not found at {videos_dir}")

    pos_vids = sorted(list((videos_dir / "Positive_Videos").glob("*.mov")) + list((videos_dir / "Positive_Videos").glob("*.mp4")))
    neg_vids = sorted(list((videos_dir / "Negative_Videos").glob("*.mov")) + list((videos_dir / "Negative_Videos").glob("*.mp4")))

    total_videos = len(pos_vids) + len(neg_vids)
    if total_videos == 0:
        raise FileNotFoundError(f"No video files found in {videos_dir}")

    frames_per_pos = max(1, int((target_count * 0.75) / max(1, len(pos_vids))))
    frames_per_neg = max(1, int((target_count * 0.25) / max(1, len(neg_vids))))

    print(f"[Calib] Extracting ~{target_count} real calibration frames from {len(pos_vids)} positive and {len(neg_vids)} negative videos...")

    frame_counter = 0

    def sample_from_video(v_path: Path, n_samples: int) -> None:
        nonlocal frame_counter
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            print(f"[Calib] Warning: Could not open {v_path.name}")
            return

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_f <= 0:
            cap.release()
            return

        step = max(1, total_f // (n_samples + 1))
        target_indices = [step * (i + 1) for i in range(n_samples) if step * (i + 1) < total_f]

        for f_idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(f_idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                save_name = out_path / f"calib_{frame_counter:04d}_{v_path.stem}_f{f_idx}.jpg"
                cv2.imwrite(str(save_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                frame_counter += 1

        cap.release()

    for v in pos_vids:
        sample_from_video(v, frames_per_pos)
    for v in neg_vids:
        sample_from_video(v, frames_per_neg)

    print(f"[Calib] Extracted {frame_counter} high-quality real calibration frames into {out_path}.")
    return out_path


class ImageCalibrationDataReader(CalibrationDataReader):
    """Feeds preprocessed real roadside calibration frames to ONNX Runtime quantizer."""

    def __init__(
        self,
        calibration_dir: Path | str,
        input_name: str = "images",
        target_size: Tuple[int, int] = (640, 640),
        max_samples: int = 150
    ) -> None:
        """Initialize calibration data reader.

        Args:
            calibration_dir: Directory containing calibration images.
            input_name: Input tensor name of the ONNX model graph.
            target_size: (width, height) expected by model.
            max_samples: Maximum number of calibration samples to feed.
        """
        super().__init__()
        self.calib_dir = Path(calibration_dir)
        self.input_name = input_name
        self.target_size = target_size

        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_imgs = sorted([p for p in self.calib_dir.iterdir() if p.suffix.lower() in img_exts])
        if not all_imgs:
            raise FileNotFoundError(
                f"No calibration images found in {self.calib_dir}. "
                f"Run extract_calibration_frames() first."
            )

        self.image_paths: List[Path] = all_imgs[:max_samples]
        self.current_idx = 0

    def preprocess(self, img_path: Path) -> np.ndarray:
        """Read image, convert BGR to RGB, resize, normalize to [0, 1], convert to NCHW float32."""
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


def identify_detection_head_nodes(onnx_model: onnx.ModelProto) -> List[str]:
    """Inspect ONNX graph and identify detection head nodes to protect from quantization.

    In YOLOv11 and YOLOv8, 'model.23' corresponds to the Detect head (bounding-box regression
    convolutions, DFL projection, and class logits). Quantizing these nodes leads to integer
    clipping and coordinate jitter, collapsing mAP.

    Args:
        onnx_model: Loaded ONNX ModelProto.

    Returns:
        nodes_to_exclude: List of node names to preserve in FP32 precision.
    """
    head_indicators = ["model.23", "dfl", "detect"]
    excluded_nodes: List[str] = []

    for node in onnx_model.graph.node:
        name_lower = node.name.lower()
        if any(ind in name_lower for ind in head_indicators):
            excluded_nodes.append(node.name)

    return excluded_nodes


def quantize_model_int8(
    model_fp32_path: Path | str,
    output_int8_path: Path | str,
    calibration_dir: Path | str = "data/calibration",
    protect_detection_head: bool = True
) -> Path:
    """Quantize an FP32 ONNX model to INT8 using QDQ format with detection head protection.

    Args:
        model_fp32_path: Path to input FP32 ONNX model.
        output_int8_path: Destination path for quantized INT8 ONNX model.
        calibration_dir: Path to directory containing real calibration frames.
        protect_detection_head: If True, preserves detection head in FP32 to prevent mAP collapse.

    Returns:
        output_path: Path to generated INT8 model.
    """
    model_fp32_path = Path(model_fp32_path)
    output_int8_path = Path(output_int8_path)
    calibration_dir = Path(calibration_dir)

    if not model_fp32_path.exists():
        raise FileNotFoundError(f"Input model {model_fp32_path} does not exist.")

    output_int8_path.parent.mkdir(parents=True, exist_ok=True)

    # Load ONNX graph to inspect input and detection head nodes
    onnx_model = onnx.load(str(model_fp32_path))
    input_name = onnx_model.graph.input[0].name

    nodes_to_exclude = []
    if protect_detection_head:
        nodes_to_exclude = identify_detection_head_nodes(onnx_model)
        print(f"[Quantize] Protected {len(nodes_to_exclude)} detection head nodes from quantization.")

    data_reader = ImageCalibrationDataReader(
        calibration_dir=calibration_dir,
        input_name=input_name,
        target_size=(640, 640),
        max_samples=150
    )

    print(f"[Quantize] Quantizing {model_fp32_path.name} -> {output_int8_path.name}...")
    print(f"[Quantize] Calibration samples: {len(data_reader.image_paths)} | QDQ QInt8 Per-Channel")

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
        nodes_to_exclude=nodes_to_exclude if nodes_to_exclude else None,
    )

    fp32_size_mb = model_fp32_path.stat().st_size / (1024 * 1024)
    int8_size_mb = output_int8_path.stat().st_size / (1024 * 1024)
    print(
        f"[Quantize] Successfully generated {output_int8_path.name}: "
        f"{fp32_size_mb:.2f} MB -> {int8_size_mb:.2f} MB "
        f"({(1.0 - int8_size_mb / fp32_size_mb) * 100:.1f}% compression)"
    )
    return output_int8_path


def run_int8_quantization(
    weights_dir: Optional[Path | str] = None,
    tudat_dir: Optional[Path | str] = None,
    calibration_dir: Optional[Path | str] = None,
    protect_detection_head: bool = True
) -> Dict[str, Path]:
    """Execute INT8 calibration and quantization for YOLOv11n and YOLOv11s.

    Args:
        weights_dir: Directory containing FP32 weights and target for INT8 weights.
        tudat_dir: Path to TU-DAT dataset root.
        calibration_dir: Directory for calibration images.
        protect_detection_head: Preserves bounding-box heads in FP32.

    Returns:
        results: Mapping of model key to output path.
    """
    weights_path = Path(weights_dir) if weights_dir else REPO_ROOT / "weights"
    tudat_path = Path(tudat_dir) if tudat_dir else REPO_ROOT / "data" / "tudat"
    calib_path = Path(calibration_dir) if calibration_dir else REPO_ROOT / "data" / "calibration"

    # Step 1: Ensure real calibration frames are extracted
    if tudat_path.exists() and (tudat_path / "videos").exists():
        extract_calibration_frames(tudat_dir=tudat_path, output_dir=calib_path, target_count=150)
    elif not calib_path.exists() or len(list(calib_path.glob("*.jpg"))) == 0:
        raise FileNotFoundError(
            f"No calibration frames found and TU-DAT videos missing at {tudat_path}. "
            f"Place TU-DAT videos in {tudat_path}/videos/ first."
        )

    results: Dict[str, Path] = {}
    for var in ["yolo26n", "yolo26s"]:
        fp32_file = weights_path / f"{var}_fp32.onnx"
        int8_file = weights_path / f"{var}_int8.onnx"

        if not fp32_file.exists():
            print(f"[Quantize] Warning: {fp32_file} not found. Skipping {var} INT8 quantization.")
            continue

        quantize_model_int8(
            model_fp32_path=fp32_file,
            output_int8_path=int8_file,
            calibration_dir=calib_path,
            protect_detection_head=protect_detection_head
        )
        results[f"{var}_int8"] = int8_file

    return results


if __name__ == "__main__":
    run_int8_quantization()
