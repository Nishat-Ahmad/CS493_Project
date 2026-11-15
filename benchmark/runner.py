"""Benchmarking runner for edge collision anticipation models.

Evaluates YOLOv11 configurations (Nano vs. Small; FP32, FP16, INT8) under simulated
edge CPU constraints (4 threads, CPU execution provider).
Pipeline per frame:
  Single-shot detection -> SORT Kalman tracking -> Kinematic TTC forecasting

Metrics computed (Mean ± Std across sequences):
  - Throughput (FPS) & Latency (mean, p95 ms)
  - Detection Accuracy (mAP@50)
  - Coordinate Jitter Variance
  - Resolution Time Margin (RTM, seconds before impact)
  - False Positive Rate (FPR, on negative control videos)

Outputs:
  - results/benchmark_summary.json
  - results/benchmark_summary.csv
  - results/benchmark_sequences_detail.csv
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

from tracking.kalman_tracker import SortTracker
from tracking.collision_kinematics import CollisionPredictor, evaluate_negative_controls
from data.dataset_loader import (
    SequenceDataset,
    TUDATSequence,
    TUDATBenchmarkDataset,
    load_dataset
)
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
    conf_threshold: float = 0.20,
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
    if output_tensor.ndim == 3:
        output_tensor = output_tensor[0]

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
    """Executes multi-video edge-constrained benchmarking across 6 model configurations."""

    TARGET_CONFIGURATIONS = [
        {"config": "yolo26n_fp32", "variant": "yolo26n", "precision": "FP32", "file": "yolo26n_fp32.onnx"},
        {"config": "yolo26n_fp16", "variant": "yolo26n", "precision": "FP16", "file": "yolo26n_fp16.onnx"},
        {"config": "yolo26n_int8", "variant": "yolo26n", "precision": "INT8", "file": "yolo26n_int8.onnx"},
        {"config": "yolo26s_fp32", "variant": "yolo26s", "precision": "FP32", "file": "yolo26s_fp32.onnx"},
        {"config": "yolo26s_fp16", "variant": "yolo26s", "precision": "FP16", "file": "yolo26s_fp16.onnx"},
        {"config": "yolo26s_int8", "variant": "yolo26s", "precision": "INT8", "file": "yolo26s_int8.onnx"},
    ]

    # COCO vehicle classes: car (2), motorcycle (3), bus (5), truck (7)
    VEHICLE_CLASSES = {2, 3, 5, 7}

    def __init__(
        self,
        weights_dir: Optional[Path | str] = None,
        dataset_path: Optional[Path | str] = None,
        results_dir: Optional[Path | str] = None,
        num_threads: int = 4,
        warmup_frames: int = 10,
        max_eval_frames_per_sequence: Optional[int] = None,
        enforce_cpu_affinity: bool = False
    ) -> None:
        self.weights_dir = Path(weights_dir) if weights_dir else REPO_ROOT / "weights"
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.results_dir = Path(results_dir) if results_dir else REPO_ROOT / "results"
        self.num_threads = num_threads
        self.warmup_frames = warmup_frames
        self.max_eval_frames = max_eval_frames_per_sequence
        self.enforce_cpu_affinity = enforce_cpu_affinity

        if self.enforce_cpu_affinity:
            self._apply_cpu_affinity()

        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _apply_cpu_affinity(self) -> None:
        """Pin process CPU affinity to simulate 4-core edge hardware environment."""
        try:
            import psutil
            p = psutil.Process()
            target_cpus = list(range(min(self.num_threads, psutil.cpu_count(logical=True))))
            p.cpu_affinity(target_cpus)
            print(f"[Benchmark] Edge Hardware Simulation: CPU affinity strictly pinned to {len(target_cpus)} cores: {target_cpus}")
        except Exception as e:
            print(f"[Benchmark] Warning: Could not enforce CPU affinity: {e}")

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

    def evaluate_sequence(
        self,
        session: ort.InferenceSession,
        input_name: str,
        is_fp16: bool,
        sequence: Union[SequenceDataset, TUDATSequence],
        is_negative_control: bool = False
    ) -> Dict[str, Any]:
        """Execute detection, SORT tracking, and kinematics pipeline on a single sequence."""
        SortTracker.reset_counter()
        tracker = SortTracker(max_age=30, min_hits=1, iou_threshold=0.3)
        predictor = CollisionPredictor(
            tau_threshold=2.5,
            consecutive_frames_threshold=3,
            fps=sequence.fps,
            enable_lateral_filtering=True
        )

        latencies_ms: List[float] = []
        predictions_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        ground_truth_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        track_trajectories: Dict[int, List[Tuple[float, float]]] = {}

        # Determine frame range to evaluate
        total_available = len(sequence) if hasattr(sequence, "__len__") else sequence.total_frames
        start_f = 0
        if self.max_eval_frames is not None:
            end_f = min(total_available, self.max_eval_frames)
        elif is_negative_control:
            end_f = min(total_available, 200)
        elif sequence.accident_frame is not None:
            start_f = max(0, sequence.accident_frame - 150)
            end_f = min(total_available, sequence.accident_frame + 30)
        else:
            end_f = total_available

        annotated_keys = set(sequence.annotations_by_frame.keys()) if hasattr(sequence, "annotations_by_frame") else set()

        for f_idx in range(start_f, end_f):
            frame_img = sequence.get_frame(f_idx)
            if frame_img is None:
                break

            # 1. Preprocessing
            input_tensor = self._preprocess_frame(frame_img, is_fp16=is_fp16)

            # 2. Timed ONNX inference
            t_start = time.perf_counter()
            outputs = session.run(None, {input_name: input_tensor})
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            latencies_ms.append(t_elapsed_ms)

            # 3. Postprocess detections for vehicles
            raw_out = outputs[0].astype(np.float32)
            detections = postprocess_yolo_output(
                output_tensor=raw_out,
                conf_threshold=0.20,
                iou_threshold=0.45,
                target_classes=self.VEHICLE_CLASSES
            )

            # Record predictions on annotated frames for mAP evaluation
            if f_idx in annotated_keys:
                gt_objects = sequence.get_annotations(f_idx)
                ground_truth_by_frame[f_idx] = [gt.to_dict() for gt in gt_objects]
                preds_list = []
                for d in detections:
                    preds_list.append({
                        "bbox": d[:4].tolist(),
                        "score": float(d[4]),
                        "class_id": 0,
                    })
                predictions_by_frame[f_idx] = preds_list

            # 4. Kalman Multi-Object Tracking
            active_tracks = tracker.update(detections)

            # Trajectory recording for coordinate jitter variance
            for trk in active_tracks:
                t_id = trk["id"]
                if t_id not in track_trajectories:
                    track_trajectories[t_id] = []
                track_trajectories[t_id].append(trk["center"])

            # 5. Kinematic TTC Collision Warning Evaluation
            predictor.evaluate_frame(frame_idx=f_idx, tracks=active_tracks)

        # Compute per-sequence metrics
        lat_stats = compute_latency_metrics(latencies_ms)
        map50 = compute_map50(
            predictions_by_frame=predictions_by_frame,
            ground_truth_by_frame=ground_truth_by_frame,
            iou_threshold=0.50,
            class_agnostic=True
        )
        jitter = compute_coordinate_jitter(track_trajectories)
        rtm = compute_rtm(
            accident_frame=sequence.accident_frame,
            first_sustained_alert_frame=predictor.first_sustained_frame,
            video_fps=sequence.fps
        )

        seq_name = getattr(sequence, "sequence_name", Path(sequence.root_path).stem)
        tag = getattr(sequence, "scenario_tag", "standard")
        has_alert = predictor.first_sustained_frame is not None

        return {
            "sequence": seq_name,
            "scenario_tag": tag,
            "frames_evaluated": len(latencies_ms),
            "mean_latency_ms": lat_stats["mean_latency_ms"],
            "p95_latency_ms": lat_stats["p95_latency_ms"],
            "fps": lat_stats["fps"],
            "map50": map50,
            "jitter_variance": jitter,
            "alert_frame": predictor.first_sustained_frame,
            "accident_frame": sequence.accident_frame,
            "rtm_seconds": rtm,
            "has_sustained_alert": has_alert
        }

    def evaluate_configuration(
        self,
        config_meta: Dict[str, str],
        dataset: Union[SequenceDataset, TUDATBenchmarkDataset]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute benchmarking for one model configuration across all sequences."""
        config_name = config_meta["config"]
        model_file = self.weights_dir / config_meta["file"]

        print(f"\n{'='*75}")
        print(f"[Benchmark] Evaluating Configuration: {config_name}")
        print(f"            Model Checkpoint: {model_file.name}")
        print(f"{'='*75}")

        if not model_file.exists():
            print(f"[Benchmark] Error: Model file {model_file} not found. Skipping.")
            return {
                "config": config_name,
                "variant": config_meta["variant"],
                "precision": config_meta["precision"],
                "status": "MISSING_WEIGHTS",
            }, []

        try:
            session = self._create_session(model_file)
        except Exception as e:
            print(f"[Benchmark] Failed to initialize session for {model_file.name}: {e}")
            return {
                "config": config_name,
                "variant": config_meta["variant"],
                "precision": config_meta["precision"],
                "status": f"LOAD_ERROR: {e}",
            }, []

        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        is_fp16 = "float16" in input_type

        # Warm-up pass
        print(f"[Benchmark] Executing {self.warmup_frames}-frame warm-up pass...")
        dummy_tensor = np.zeros((1, 3, 640, 640), dtype=np.float16 if is_fp16 else np.float32)
        for _ in range(self.warmup_frames):
            _ = session.run(None, {input_name: dummy_tensor})

        # Separate positive vs negative sequences
        if isinstance(dataset, TUDATBenchmarkDataset):
            pos_sequences = dataset.positive_sequences
            neg_sequences = dataset.negative_sequences
        else:
            pos_sequences = [dataset]
            neg_sequences = []

        # 1. Evaluate Positive Collision Sequences
        print(f"[Benchmark] Evaluating {len(pos_sequences)} positive collision sequences...")
        pos_results: List[Dict[str, Any]] = []
        for seq in pos_sequences:
            t0 = time.perf_counter()
            seq_res = self.evaluate_sequence(
                session=session,
                input_name=input_name,
                is_fp16=is_fp16,
                sequence=seq,
                is_negative_control=False
            )
            seq_res["config"] = config_name
            seq_res["variant"] = config_meta["variant"]
            seq_res["precision"] = config_meta["precision"]
            pos_results.append(seq_res)
            dur = time.perf_counter() - t0
            print(
                f"  [{seq_res['sequence']:<5}] "
                f"FPS: {seq_res['fps']:5.1f} | "
                f"Lat: {seq_res['mean_latency_ms']:5.1f}ms | "
                f"mAP@50: {seq_res['map50']:5.3f} | "
                f"Jitter: {seq_res['jitter_variance']:5.2f} | "
                f"RTM: {seq_res['rtm_seconds']:5.2f}s | "
                f"({seq_res['frames_evaluated']} frames, {dur:.1f}s)",
                flush=True
            )

        # 2. Evaluate Negative Control Videos for False Positive Rate (FPR)
        print(f"[Benchmark] Evaluating {len(neg_sequences)} negative control sequences for FPR...")
        if isinstance(dataset, TUDATBenchmarkDataset) and len(neg_sequences) > 0:
            control_eval = evaluate_negative_controls(
                benchmark_dataset=dataset,
                tau_threshold=2.5,
                consecutive_frames_threshold=3,
                max_frames_per_sequence=250
            )
            fpr = float(control_eval["false_positive_rate"])
            neg_results = [
                {
                    "config": config_name,
                    "variant": config_meta["variant"],
                    "precision": config_meta["precision"],
                    "sequence": r["sequence"],
                    "scenario_tag": "negative_control",
                    "frames_evaluated": r["frames_evaluated"],
                    "has_sustained_alert": r["has_false_alarm"],
                    "alert_frame": r["first_false_alarm_frame"],
                    "accident_frame": None,
                    "rtm_seconds": 0.0,
                    "fps": summary_entry_fps if "summary_entry_fps" in locals() else float(np.mean([r["fps"] for r in pos_results])),
                    "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in pos_results])),
                    "p95_latency_ms": float(np.mean([r["p95_latency_ms"] for r in pos_results])),
                    "map50": 0.0,
                    "jitter_variance": 0.0
                }
                for r in control_eval.get("sequence_results", [])
            ]
            print(f"  Negative control evaluation: {control_eval['false_alarms']} false alarms across {control_eval['total_sequences']} videos (FPR: {fpr*100.0:.1f}%).")
        else:
            fpr = 0.0
            neg_results = []

        # 3. Calculate Aggregate Mean ± Std across positive collision sequences
        latencies = [r["mean_latency_ms"] for r in pos_results]
        p95_latencies = [r["p95_latency_ms"] for r in pos_results]
        fps_vals = [r["fps"] for r in pos_results]
        map_vals = [r["map50"] for r in pos_results]
        jitter_vals = [r["jitter_variance"] for r in pos_results]
        rtm_vals = [r["rtm_seconds"] for r in pos_results]

        summary_entry = {
            "config": config_name,
            "variant": config_meta["variant"],
            "precision": config_meta["precision"],
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "std_latency_ms": round(float(np.std(latencies)), 2),
            "p95_latency_ms": round(float(np.mean(p95_latencies)), 2),
            "std_p95_latency_ms": round(float(np.std(p95_latencies)), 2),
            "fps": round(float(np.mean(fps_vals)), 2),
            "std_fps": round(float(np.std(fps_vals)), 2),
            "map50": round(float(np.mean(map_vals)), 4),
            "std_map50": round(float(np.std(map_vals)), 4),
            "jitter_variance": round(float(np.mean(jitter_vals)), 4),
            "std_jitter_variance": round(float(np.std(jitter_vals)), 4),
            "rtm_seconds": round(float(np.mean(rtm_vals)), 3),
            "std_rtm_seconds": round(float(np.std(rtm_vals)), 3),
            "false_positive_rate": round(fpr * 100.0, 2),
            "positive_sequences_evaluated": len(pos_results),
            "control_sequences_evaluated": len(neg_sequences),
            "status": "SUCCESS",
        }

        print(
            f"\n[Summary] {config_name:<14} -> "
            f"FPS: {summary_entry['fps']} ± {summary_entry['std_fps']} | "
            f"Latency: {summary_entry['mean_latency_ms']} ± {summary_entry['std_latency_ms']} ms | "
            f"mAP@50: {summary_entry['map50']} ± {summary_entry['std_map50']} | "
            f"Jitter: {summary_entry['jitter_variance']} | "
            f"RTM: {summary_entry['rtm_seconds']} ± {summary_entry['std_rtm_seconds']}s | "
            f"FPR: {summary_entry['false_positive_rate']}%",
            flush=True
        )

        all_sequence_rows = pos_results + neg_results
        return summary_entry, all_sequence_rows

    def run_all(self) -> List[Dict[str, Any]]:
        """Run benchmark across all 6 configurations and write summaries."""
        print(f"[Benchmark] Loading benchmark dataset from {self.dataset_path or 'default auto-discovery'}...")
        dataset = load_dataset(self.dataset_path)
        print(f"[Benchmark] Loaded dataset: {dataset} (Positive: {len(getattr(dataset, 'positive_sequences', [dataset]))}, Control: {len(getattr(dataset, 'negative_sequences', []))})")

        all_summaries: List[Dict[str, Any]] = []
        all_sequence_details: List[Dict[str, Any]] = []

        for cfg in self.TARGET_CONFIGURATIONS:
            summary_entry, seq_rows = self.evaluate_configuration(cfg, dataset)
            all_summaries.append(summary_entry)
            all_sequence_details.extend(seq_rows)

        # 1. Save benchmark_summary.json
        json_path = self.results_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "configurations": all_summaries,
                "sequence_details": all_sequence_details
            }, f, indent=2)
        print(f"\n[Benchmark] Summary JSON written to {json_path}")

        # 2. Save benchmark_summary.csv
        csv_path = self.results_dir / "benchmark_summary.csv"
        df = pd.DataFrame(all_summaries)
        df.to_csv(csv_path, index=False)
        print(f"[Benchmark] Summary CSV written to {csv_path}")

        # 3. Save benchmark_sequences_detail.csv
        detail_csv_path = self.results_dir / "benchmark_sequences_detail.csv"
        df_detail = pd.DataFrame(all_sequence_details)
        df_detail.to_csv(detail_csv_path, index=False)
        print(f"[Benchmark] Detailed sequences CSV written to {detail_csv_path}")

        return all_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge Collision Benchmark Runner (4-Core CPU POSIX Simulation)")
    parser.add_argument("--weights-dir", type=str, default=None, help="Directory containing ONNX weights")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to TU-DAT or synthetic dataset")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory to save benchmark results")
    parser.add_argument("--threads", type=int, default=4, help="Number of intra-op CPU threads (default: 4)")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Number of warmup frames (default: 10)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per sequence")
    parser.add_argument("--enforce-cpu-affinity", action="store_true", help="Pin CPU affinity to simulate 4-core edge CPU")
    parser.add_argument("--config", type=str, default=None, help="Specific configuration name to run (e.g. yolo26n_fp32)")
    args = parser.parse_args()

    runner = EdgeBenchmarkRunner(
        weights_dir=args.weights_dir,
        dataset_path=args.dataset_path,
        results_dir=args.results_dir,
        num_threads=args.threads,
        warmup_frames=args.warmup_frames,
        max_eval_frames_per_sequence=args.max_frames,
        enforce_cpu_affinity=args.enforce_cpu_affinity
    )
    if args.config:
        cfg = next((c for c in runner.TARGET_CONFIGURATIONS if c["config"] == args.config), None)
        if not cfg:
            raise ValueError(f"Unknown configuration '{args.config}'. Choices: {[c['config'] for c in runner.TARGET_CONFIGURATIONS]}")
        dataset = load_dataset(runner.dataset_path)
        runner.evaluate_configuration(cfg, dataset)
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
