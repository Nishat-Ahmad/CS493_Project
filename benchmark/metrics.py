"""Evaluation metrics module for edge collision anticipation benchmark.

Implements mathematical formulations for:
  1. Latency & Throughput (FPS = 1000 / latency_ms)
  2. Spatial Localization Accuracy: mAP@50 (IoU threshold 0.50)
  3. Trajectory Coordinate Jitter Variance:
       Jitter = 1/(N-2) * sum(||c_{t+1} - 2*c_t + c_{t-1}||^2)
  4. Resolution Time Margin (RTM):
       RTM = max(0, F_accident - F_first_sustained_alert) / FPS_video
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


def compute_map50(
    predictions_by_frame: Dict[int, List[Dict[str, Any]]],
    ground_truth_by_frame: Dict[int, List[Dict[str, Any]]],
    iou_threshold: float = 0.50,
    class_agnostic: bool = True
) -> float:
    """Compute mean Average Precision at IoU=0.50 (mAP@50).

    Args:
        predictions_by_frame: Map from frame index to list of predictions:
            {'bbox': [x1, y1, x2, y2], 'score': float, 'class_id': int}
        ground_truth_by_frame: Map from frame index to list of GT objects:
            {'bbox': [x1, y1, x2, y2], 'class_id': int}
        iou_threshold: Intersection-over-union threshold for positive match.
        class_agnostic: If True, evaluates localization across all target vehicle objects.

    Returns:
        map50: Scalar mAP@50 in range [0.0, 1.0].
    """
    if class_agnostic:
        # Map all classes to a single target object category (0)
        p_remapped: Dict[int, List[Dict[str, Any]]] = {}
        for f_idx, preds in predictions_by_frame.items():
            p_remapped[f_idx] = [{**p, "class_id": 0} for p in preds]

        g_remapped: Dict[int, List[Dict[str, Any]]] = {}
        for f_idx, gts in ground_truth_by_frame.items():
            g_remapped[f_idx] = [{**g, "class_id": 0} for g in gts]

        return compute_map50(p_remapped, g_remapped, iou_threshold=iou_threshold, class_agnostic=False)

    # Group by class
    all_classes = set()
    for preds in predictions_by_frame.values():
        for p in preds:
            all_classes.add(p.get("class_id", 0))
    for gts in ground_truth_by_frame.values():
        for g in gts:
            all_classes.add(g.get("class_id", 0))

    if not all_classes:
        return 0.0

    class_aps: List[float] = []

    for cls in all_classes:
        # Collect all predictions for this class
        flat_preds: List[Tuple[int, np.ndarray, float]] = []
        for f_idx, preds in predictions_by_frame.items():
            for p in preds:
                if p.get("class_id", 0) == cls:
                    flat_preds.append((f_idx, np.asarray(p["bbox"], dtype=np.float64), float(p.get("score", 1.0))))

        # Collect ground truth for this class
        # Map: frame_idx -> array of bboxes, and matched mask
        gt_by_f: Dict[int, np.ndarray] = {}
        matched_gt_by_f: Dict[int, List[bool]] = {}
        total_gt = 0

        for f_idx, gts in ground_truth_by_frame.items():
            boxes = [g["bbox"] for g in gts if g.get("class_id", 0) == cls]
            if boxes:
                gt_by_f[f_idx] = np.asarray(boxes, dtype=np.float64)
                matched_gt_by_f[f_idx] = [False] * len(boxes)
                total_gt += len(boxes)

        if total_gt == 0:
            if len(flat_preds) == 0:
                continue
            else:
                class_aps.append(0.0)
                continue

        if len(flat_preds) == 0:
            class_aps.append(0.0)
            continue

        # Sort predictions by descending confidence score
        flat_preds.sort(key=lambda x: x[2], reverse=True)

        tp = np.zeros(len(flat_preds), dtype=np.float64)
        fp = np.zeros(len(flat_preds), dtype=np.float64)

        for i, (f_idx, p_box, _) in enumerate(flat_preds):
            if f_idx not in gt_by_f:
                fp[i] = 1.0
                continue

            gt_boxes = gt_by_f[f_idx]
            ious = np.array([compute_iou(p_box, gb) for gb in gt_boxes])
            max_idx = int(np.argmax(ious)) if len(ious) > 0 else -1
            max_iou = float(ious[max_idx]) if max_idx >= 0 else 0.0

            if max_iou >= iou_threshold and not matched_gt_by_f[f_idx][max_idx]:
                tp[i] = 1.0
                matched_gt_by_f[f_idx][max_idx] = True
            else:
                fp[i] = 1.0

        # Cumulative precision and recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / float(total_gt)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

        # 11-point interpolation AP calculation
        ap = 0.0
        for r_thresh in np.linspace(0.0, 1.0, 11):
            prec_at_r = precisions[recalls >= r_thresh]
            p_val = np.max(prec_at_r) if len(prec_at_r) > 0 else 0.0
            ap += p_val / 11.0

        class_aps.append(float(ap))

    return float(np.mean(class_aps)) if class_aps else 0.0


def compute_coordinate_jitter(track_trajectories: Dict[int, List[Tuple[float, float]]]) -> float:
    """Compute discrete acceleration jitter variance across track trajectories.

    Formula:
        Jitter = 1/(N-2) * sum_{t=1}^{N-2} ||c_{t+1} - 2*c_t + c_{t-1}||^2

    Args:
        track_trajectories: Mapping from track_id to sequence of center coordinates [(x_t, y_t), ...].

    Returns:
        mean_jitter: Average jitter variance across all qualifying tracks (length >= 3).
    """
    track_jitters: List[float] = []

    for trk_id, coords in track_trajectories.items():
        if len(coords) < 3:
            continue

        c = np.asarray(coords, dtype=np.float64)  # (N, 2)
        # Second-order finite difference: c_{t+1} - 2*c_t + c_{t-1}
        diff2 = c[2:] - 2.0 * c[1:-1] + c[:-2]
        # Squared Euclidean norm per step: dx^2 + dy^2
        sq_norms = np.sum(diff2 ** 2, axis=1)
        jitter_val = float(np.mean(sq_norms))
        track_jitters.append(jitter_val)

    if not track_jitters:
        return 0.0
    return float(np.mean(track_jitters))


def compute_rtm(
    accident_frame: int,
    first_sustained_alert_frame: Optional[int],
    video_fps: float = 30.0
) -> float:
    """Compute Resolution Time Margin (RTM) in seconds before impact.

    Formula:
        RTM = max(0, F_accident - F_first_sustained_alert) / FPS_video

    Args:
        accident_frame: Ground truth impact frame index.
        first_sustained_alert_frame: Frame where sustained alert condition was achieved.
        video_fps: Video framerate.

    Returns:
        rtm: Resolution Time Margin in seconds.
    """
    if first_sustained_alert_frame is None or first_sustained_alert_frame > accident_frame:
        return 0.0

    margin_frames = max(0, accident_frame - first_sustained_alert_frame)
    return float(margin_frames) / float(video_fps)


def compute_latency_metrics(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute statistical latency and throughput metrics.

    Args:
        latencies_ms: List of inference times in milliseconds.

    Returns:
        metrics: Dictionary containing mean, std, median, p95, p99, and FPS.
    """
    if not latencies_ms:
        return {
            "mean_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "fps": 0.0,
        }

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_lat = float(np.mean(arr))
    fps = 1000.0 / mean_lat if mean_lat > 1e-4 else 0.0

    return {
        "mean_latency_ms": mean_lat,
        "std_latency_ms": float(np.std(arr)),
        "median_latency_ms": float(np.median(arr)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": float(np.percentile(arr, 99)),
        "fps": fps,
    }

