"""Kalman-filter multi-object tracking implementation (SORT algorithm).

Implements 2D bounding box tracking with a 7-dimensional state vector:
    x = [u, v, s, r, u_dot, v_dot, s_dot]^T
where:
    u, v: Center coordinates of the bounding box
    s: Scale (area) of the bounding box
    r: Aspect ratio (width / height)
    u_dot, v_dot, s_dot: Velocities of center coordinates and scale
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert bounding box [x1, y1, x2, y2] to measurement vector z [u, v, s, r].

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        z: Measurement array of shape (4, 1).
    """
    w = max(0.0, float(bbox[2] - bbox[0]))
    h = max(0.0, float(bbox[3] - bbox[1]))
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    return np.array([[u], [v], [s], [r]], dtype=np.float64)


def z_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert state vector x [u, v, s, r, ...] to bounding box [x1, y1, x2, y2].

    Args:
        x: State vector containing at least [u, v, s, r].

    Returns:
        bbox: Bounding box array [x1, y1, x2, y2].
    """
    x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
    u = float(x_flat[0])
    v = float(x_flat[1])
    s = max(0.0, float(x_flat[2]))
    r = max(1e-6, float(x_flat[3]))

    w = np.sqrt(s * r)
    h = s / (w + 1e-6)

    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute Intersection over Union (IoU) matrix between two sets of bounding boxes.

    Args:
        boxes_a: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes_b: Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        iou_matrix: Array of shape (N, M) containing pairwise IoU values.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    # Compute intersection
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-6)

    return inter_area / union_area


class KalmanBoxTracker:
    """Single-object Kalman Filter state estimator tracking a 2D bounding box."""

    _count: int = 0

    def __init__(self, bbox: np.ndarray, class_id: int = 0, score: float = 1.0) -> None:
        """Initialize Kalman filter for tracking.

        Args:
            bbox: Initial detection [x1, y1, x2, y2].
            class_id: Detection class identifier.
            score: Confidence score.
        """
        # State vector: [u, v, s, r, u_dot, v_dot, s_dot]^T
        self.dim_x = 7
        self.dim_z = 4

        # State transition matrix F (constant velocity model)
        self.F = np.eye(self.dim_x, dtype=np.float64)
        self.F[0, 4] = 1.0  # u += u_dot
        self.F[1, 5] = 1.0  # v += v_dot
        self.F[2, 6] = 1.0  # s += s_dot

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # State estimation covariance P
        self.P = np.eye(self.dim_x, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # High initial velocity uncertainty

        # Process noise covariance Q
        self.Q = np.eye(self.dim_x, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

        # Initialize state with detection
        z = bbox_to_z(bbox)
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.x[:self.dim_z] = z

        self.time_since_update: int = 0
        self.id: int = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.history: List[np.ndarray] = []
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0
        self.class_id: int = class_id
        self.score: float = score

    def predict(self) -> np.ndarray:
        """Advance state estimation by one timestep using transition model F.

        Returns:
            bbox: Predicted bounding box [x1, y1, x2, y2].
        """
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0

        # x = F * x
        self.x = np.dot(self.F, self.x)
        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred_box = z_to_bbox(self.x)
        self.history.append(pred_box)
        return pred_box

    def update(self, bbox: np.ndarray, score: float = 1.0) -> None:
        """Update state estimate with new measurement bbox.

        Args:
            bbox: Measured bounding box [x1, y1, x2, y2].
            score: Detection confidence score.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score

        z = bbox_to_z(bbox)
        # Innovation y = z - H * x
        y = z - np.dot(self.H, self.x)
        # S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Kalman Gain K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # x = x + K * y
        self.x = self.x + np.dot(K, y)
        # P = (I - K * H) * P
        I = np.eye(self.dim_x, dtype=np.float64)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self) -> np.ndarray:
        """Return current bounding box estimate [x1, y1, x2, y2]."""
        return z_to_bbox(self.x)

    def get_velocity(self) -> Tuple[float, float]:
        """Return center velocity (u_dot, v_dot)."""
        return float(self.x[4, 0]), float(self.x[5, 0])

    def get_center(self) -> Tuple[float, float]:
        """Return center coordinate (u, v)."""
        return float(self.x[0, 0]), float(self.x[1, 0])


class SortTracker:
    """Multi-object SORT tracker managing KalmanBoxTracker instances."""

    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_threshold: float = 0.3) -> None:
        """Initialize SORT tracker.

        Args:
            max_age: Maximum frames to retain a lost track before deletion.
            min_hits: Minimum hits before a track is published.
            iou_threshold: Minimum IoU overlap for detection-to-track association.
        """
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.iou_threshold: float = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    @classmethod
    def reset_counter(cls) -> None:
        """Reset global track ID counter for deterministic runs."""
        KalmanBoxTracker._count = 0

    def update(self, detections: np.ndarray) -> List[Dict[str, Any]]:
        """Update tracker with detections for the current frame.

        Args:
            detections: Array of shape (N, 5) or (N, 6) in format:
                [x1, y1, x2, y2, score] or [x1, y1, x2, y2, score, class_id]

        Returns:
            active_tracks: List of track dicts with keys:
                'id': Track ID
                'bbox': [x1, y1, x2, y2]
                'center': (u, v)
                'velocity': (vx, vy)
                'score': Confidence score
                'class_id': Class identifier
        """
        self.frame_count += 1

        # Predict positions of existing trackers
        predicted_boxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                to_del.append(i)
            else:
                predicted_boxes.append(pos)

        for i in reversed(to_del):
            self.trackers.pop(i)

        if len(detections) > 0:
            det_boxes = detections[:, :4]
            det_scores = detections[:, 4] if detections.shape[1] >= 5 else np.ones(len(detections))
            det_classes = detections[:, 5].astype(int) if detections.shape[1] >= 6 else np.zeros(len(detections), dtype=int)
        else:
            det_boxes = np.empty((0, 4))
            det_scores = np.empty((0,))
            det_classes = np.empty((0,), dtype=int)

        # Associate detections to predicted tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            predicted_boxes=np.array(predicted_boxes),
            detections=det_boxes,
            iou_threshold=self.iou_threshold
        )

        # Update matched trackers
        for trk_idx, det_idx in matched:
            self.trackers[trk_idx].update(det_boxes[det_idx], score=float(det_scores[det_idx]))
            self.trackers[trk_idx].class_id = int(det_classes[det_idx])

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(
                bbox=det_boxes[det_idx],
                class_id=int(det_classes[det_idx]),
                score=float(det_scores[det_idx])
            )
            self.trackers.append(new_trk)

        # Extract active tracks and prune dead tracks
        active_tracks: List[Dict[str, Any]] = []
        remaining_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            # Keep tracks that haven't expired
            if trk.time_since_update <= self.max_age:
                remaining_trackers.append(trk)
                # Output track if it has sufficient observations or is recent
                if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    bbox = trk.get_state()
                    center = trk.get_center()
                    velocity = trk.get_velocity()
                    active_tracks.append({
                        "id": trk.id,
                        "bbox": bbox,
                        "center": center,
                        "velocity": velocity,
                        "score": trk.score,
                        "class_id": trk.class_id,
                        "hits": trk.hits,
                        "age": trk.age,
                    })

        self.trackers = remaining_trackers
        return active_tracks

    def _associate(
        self,
        predicted_boxes: np.ndarray,
        detections: np.ndarray,
        iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Assign detections to tracked targets using Hungarian algorithm.

        Args:
            predicted_boxes: Array of shape (N, 4).
            detections: Array of shape (M, 4).
            iou_threshold: Minimum IoU for valid assignment.

        Returns:
            matched: List of tuples (track_idx, det_idx).
            unmatched_detections: List of detection indices.
            unmatched_trackers: List of tracker indices.
        """
        if len(predicted_boxes) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(predicted_boxes)))

        iou_matrix = compute_iou_matrix(predicted_boxes, detections)

        # Cost matrix = -IoU
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(predicted_boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matched.append((r, c))
                if c in unmatched_dets:
                    unmatched_dets.remove(c)
                if r in unmatched_trks:
                    unmatched_trks.remove(r)

        return matched, unmatched_dets, unmatched_trks