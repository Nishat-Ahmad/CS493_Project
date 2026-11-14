"""Benchmarking runner for edge collision anticipation models.

Evaluates YOLOv11 configurations (Nano vs. Small; FP32, FP16, INT8) under simulated
edge CPU constraints (4 threads, CPU execution provider).
Pipeline per frame:
  Single-shot detection -> SORT Kalman tracking -> Kinematic TTC forecasting

Metrics computed:
  - Throughput (FPS) & Latency (mean, p95, p99 ms)
  - Detection Accuracy (mAP@50)
  - Coordinate Jitter Variance
  - Resolution Time Margin (RTM, seconds before impact)

Outputs:
  - results/benchmark_summary.json
  - results/benchmark_summary.csv
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

from tracking.kalman_tracker import SortTracker
from tracking.collision_kinematics import CollisionPredictor
from data.dataset_loader import SequenceDataset
from benchmark.metrics import (
    compute_latency_metrics,
    compute_map50,
    compute_coordinate_jitter,
    compute_rtm
)


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) containing confidence scores.
        iou_threshold: Overlap threshold for box suppression.

    Returns:
        keep: List of indices of surviving boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess_yolo_output(
    output_tensor: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_classes: Optional[set[int]] = None
) -> np.ndarray:
    """Decode raw YOLOv11 output tensor to detected bounding boxes.

    YOLOv11 output is shape (1, 84, 8400) or (1, 8400, 84):
      - 4 bbox coordinates: [cx, cy, w, h]
      - 80 class probability scores

    Args:
        output_tensor: Model output array.
        conf_threshold: Minimum confidence score to retain detection.
        iou_threshold: NMS overlap threshold.
        target_classes: Optional set of class IDs to retain (e.g., COCO vehicle classes).

    Returns:
        detections: Array of shape (M, 6) with [x1, y1, x2, y2, score, class_id].
    """
    # Remove batch dimension if present
    if output_tensor.ndim == 3:
        output_tensor = output_tensor[0]

    # Ensure shape is (num_anchors, 84)
    if output_tensor.shape[0] < output_tensor.shape[1]:
        output_tensor = output_tensor.T

    boxes = output_tensor[:, :4]  # [cx, cy, w, h]
    scores_matrix = output_tensor[:, 4:]  # (8400, 80)

    class_ids = np.argmax(scores_matrix, axis=1)
    max_scores = scores_matrix[np.arange(len(class_ids)), class_ids]

    mask = max_scores >= conf_threshold
    if target_classes is not None:
        cls_mask = np.isin(class_ids, list(target_classes))
        mask = mask & cls_mask

    boxes = boxes[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float64)

    # Convert [cx, cy, w, h] -> [x1, y1, x2, y2]
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    boxes_xyxy = np.column_stack([x1, y1, x2, y2])
    keep_indices = nms_boxes(boxes_xyxy, max_scores, iou_threshold=iou_threshold)

    if not keep_indices:
        return np.empty((0, 6), dtype=np.float64)

    dets = np.column_stack([
        boxes_xyxy[keep_indices],
        max_scores[keep_indices],
        class_ids[keep_indices]
    ])
    return dets


class EdgeBenchmarkRunner:
    """Executes edge-constrained benchmarking across 6 model configurations."""

    TARGET_CONFIGURATIONS = [
        {"config": "yolo11n_fp32", "variant": "yolo11n", "precision": "FP32", "file": "yolo11n_fp32.onnx"},
        {"config": "yolo11n_fp16", "variant": "yolo11n", "precision": "FP16", "file": "yolo11n_fp16.onnx"},
        {"config": "yolo11n_int8", "variant": "yolo11n", "precision": "INT8", "file": "yolo11n_int8.onnx"},
        {"config": "yolo11s_fp32", "variant": "yolo11s", "precision": "FP32", "file": "yolo11s_fp32.onnx"},
        {"config": "yolo11s_fp16", "variant": "yolo11s", "precision": "FP16", "file": "yolo11s_fp16.onnx"},
        {"config": "yolo11s_int8", "variant": "yolo11s", "precision": "INT8", "file": "yolo11s_int8.onnx"},
    ]

    def __init__(
        self,
        weights_dir: Optional[Path | str] = None,
        dataset_path: Optional[Path | str] = None,
        results_dir: Optional[Path | str] = None,
        num_threads: int = 4,
        warmup_frames: int = 10
    ) -> None:
        self.weights_dir = Path(weights_dir) if weights_dir else REPO_ROOT / "weights"
        self.dataset_path = Path(dataset_path) if dataset_path else REPO_ROOT / "data" / "synthetic_sequence"
        self.results_dir = Path(results_dir) if results_dir else REPO_ROOT / "results"
        self.num_threads = num_threads
        self.warmup_frames = warmup_frames

        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _create_session(self, model_path: Path) -> ort.InferenceSession:
        """Create ONNX Runtime InferenceSession with simulated edge CPU thread constraints."""
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        return session

    def _preprocess_frame(self, frame: np.ndarray, is_fp16: bool = False) -> np.ndarray:
        """Preprocess OpenCV BGR image to NCHW input tensor."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)  # NCHW
        if is_fp16:
            tensor = tensor.astype(np.float16)
        return tensor

    def evaluate_configuration(
        self,
        config_meta: Dict[str, str],
        dataset: SequenceDataset
    ) -> Dict[str, Any]:
        """Execute complete pipeline benchmarking for one configuration."""
        config_name = config_meta["config"]
        model_file = self.weights_dir / config_meta["file"]

        print(f"\n{'='*70}")
        print(f"[Benchmark] Evaluating Configuration: {config_name}")
        print(f"            Model Path: {model_file}")
        print(f"{'='*70}")

        if not model_file.exists():
            print(f"[Benchmark] Error: Model file {model_file} not found. Skipping.")
            return {
                "config": config_name,
                "variant": config_meta["variant"],
                "precision": config_meta["precision"],
                "status": "MISSING_WEIGHTS",
            }

        try:
            session = self._create_session(model_file)
        except Exception as e:
            print(f"[Benchmark] Failed to initialize session for {model_file.name}: {e}")
            return {
                "config": config_name,
                "variant": config_meta["variant"],
                "precision": config_meta["precision"],
                "status": f"LOAD_ERROR: {e}",
            }

        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        is_fp16 = "float16" in input_type

        # Warm-up pass (10 frames)
        print(f"[Benchmark] Executing {self.warmup_frames}-frame warm-up pass...")
        dummy_tensor = np.zeros((1, 3, 640, 640), dtype=np.float16 if is_fp16 else np.float32)
        for _ in range(self.warmup_frames):
            _ = session.run(None, {input_name: dummy_tensor})

        # Pipeline components
        SortTracker.reset_counter()
        tracker = SortTracker(max_age=30, min_hits=1, iou_threshold=0.3)
        predictor = CollisionPredictor(
            tau_threshold=2.5,
            consecutive_frames_threshold=3,
            fps=dataset.fps
        )

        latencies_ms: List[float] = []
        predictions_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        ground_truth_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        track_trajectories: Dict[int, List[Tuple[float, float]]] = {}

        # Benchmark sequence loop
        print(f"[Benchmark] Running inference on sequence ({len(dataset)} frames)...")
        for frame_idx, frame_img, gt_objects in dataset:
            # Format ground truth for metrics
            ground_truth_by_frame[frame_idx] = [gt.to_dict() for gt in gt_objects]

            # 1. Preprocessing
            input_tensor = self._preprocess_frame(frame_img, is_fp16=is_fp16)

            # 2. Timed ONNX inference
            t_start = time.perf_counter()
            outputs = session.run(None, {input_name: input_tensor})
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            latencies_ms.append(t_elapsed_ms)

            # 3. Post-process bounding boxes across candidate foreground objects
            raw_out = outputs[0].astype(np.float32)
            detections = postprocess_yolo_output(
                output_tensor=raw_out,
                conf_threshold=0.20,
                iou_threshold=0.45,
                target_classes=None
            )

            # Record predictions for mAP calculation
            preds_list = []
            for d in detections:
                preds_list.append({
                    "bbox": d[:4].tolist(),
                    "score": float(d[4]),
                    "class_id": 0,
                })
            predictions_by_frame[frame_idx] = preds_list

            # 4. Kalman Multi-Object Tracking
            active_tracks = tracker.update(detections)

            # Record trajectories for jitter computation
            for trk in active_tracks:
                t_id = trk["id"]
                if t_id not in track_trajectories:
                    track_trajectories[t_id] = []
                track_trajectories[t_id].append(trk["center"])

            # 5. Kinematic TTC Collision Warning Evaluation
            predictor.evaluate_frame(frame_idx=frame_idx, tracks=active_tracks)

        # Compute benchmark metrics
        lat_stats = compute_latency_metrics(latencies_ms)
        map50 = compute_map50(
            predictions_by_frame=predictions_by_frame,
            ground_truth_by_frame=ground_truth_by_frame,
            iou_threshold=0.50,
            class_agnostic=True
        )
        jitter = compute_coordinate_jitter(track_trajectories)
        rtm = compute_rtm(
            accident_frame=dataset.accident_frame,
            first_sustained_alert_frame=predictor.first_sustained_frame,
            video_fps=dataset.fps
        )

        res_entry = {
            "config": config_name,
            "variant": config_meta["variant"],
            "precision": config_meta["precision"],
            "mean_latency_ms": round(lat_stats["mean_latency_ms"], 2),
            "median_latency_ms": round(lat_stats["median_latency_ms"], 2),
            "p95_latency_ms": round(lat_stats["p95_latency_ms"], 2),
            "p99_latency_ms": round(lat_stats["p99_latency_ms"], 2),
            "fps": round(lat_stats["fps"], 2),
            "map50": round(map50, 4),
            "jitter_variance": round(jitter, 4),
            "first_sustained_alert_frame": predictor.first_sustained_frame,
            "accident_frame": dataset.accident_frame,
            "rtm_seconds": round(rtm, 3),
            "status": "SUCCESS",
        }

        print(
            f"[Benchmark] Completed {config_name} -> "
            f"FPS: {res_entry['fps']} | "
            f"mAP@50: {res_entry['map50']} | "
            f"Jitter: {res_entry['jitter_variance']} | "
            f"RTM: {res_entry['rtm_seconds']}s",
            flush=True
        )
        return res_entry

    def run_all(self) -> List[Dict[str, Any]]:
        """Run benchmark across all 6 configurations and write summaries."""
        print(f"[Benchmark] Loading sequence dataset from {self.dataset_path}...")
        dataset = SequenceDataset(self.dataset_path)

        all_results: List[Dict[str, Any]] = []

        for cfg in self.TARGET_CONFIGURATIONS:
            res = self.evaluate_configuration(cfg, dataset)
            all_results.append(res)

        # Save JSON
        json_path = self.results_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[Benchmark] Summary JSON written to {json_path}")

        # Save CSV
        csv_path = self.results_dir / "benchmark_summary.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        print(f"[Benchmark] Summary CSV written to {csv_path}")

        return all_results


def main() -> None:
    runner = EdgeBenchmarkRunner()
    runner.run_all()


if __name__ == "__main__":
    main()