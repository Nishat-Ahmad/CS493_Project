"""Kinematic Time-to-Collision (TTC) trajectory forecasting module.

Evaluates inter-vehicle collision risk based on relative kinematics and 2D trajectory intersection:
    1. Displacement Vector:      d_0 = p2 - p1 = [x2 - x1, y2 - y1]
    2. Relative Velocity Vector: v_rel = v1 - v2 = [vx1 - vx2, vy1 - vy2] (in px/sec)
    3. Closing Speed:            v_c = (d_0 . v_rel) / (||d_0|| + 1e-6)
    4. Time to Closest Approach: t_cpa = (d_0 . v_rel) / (||v_rel||^2 + 1e-6)
    5. Lateral Miss Distance:    d_min = ||d_0 - v_rel * t_cpa||
    6. Effective Collision Radii: R_coll = 0.5 * (diag1 + diag2) * margin_factor

Key Safety Rule (Lateral Trajectory Filtering):
    Vehicles in opposite or adjacent lanes often have v_c > 0 as they approach, but their
    lateral miss distance d_min > R_coll ensures they pass safely without colliding.
    An impending collision is ONLY flagged when:
        - The closing speed is positive (v_c > 1.0 px/s)
        - Time to closest approach is within warning horizon (0 < t_cpa < tau_threshold)
        - The lateral miss distance is within physical collision radius (d_min <= R_coll)
        - Alert persists across consecutive_frames_threshold consecutive frames.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
import numpy as np


class TTCResult(tuple):
    """Rich tuple returning (ttc, v_c, distance) with trajectory telemetry attributes.

    Maintains full backwards compatibility with standard 3-tuple unpacking:
        ttc, v_c, dist = compute_ttc_pair(...)
    while exposing detailed lateral trajectory metrics:
        res.is_collision_course, res.d_min, res.t_cpa, res.r_coll
    """

    def __new__(
        cls,
        ttc: float,
        v_c: float,
        distance: float,
        is_collision_course: bool = True,
        d_min: float = 0.0,
        t_cpa: float = 0.0,
        r_coll: float = 50.0
    ):
        return super().__new__(cls, (ttc, v_c, distance))

    def __init__(
        self,
        ttc: float,
        v_c: float,
        distance: float,
        is_collision_course: bool = True,
        d_min: float = 0.0,
        t_cpa: float = 0.0,
        r_coll: float = 50.0
    ) -> None:
        self.ttc: float = ttc
        self.v_c: float = v_c
        self.distance: float = distance
        self.is_collision_course: bool = is_collision_course
        self.d_min: float = d_min
        self.t_cpa: float = t_cpa
        self.r_coll: float = r_coll


def compute_boxes_intersection(
    box1: List[float] | np.ndarray,
    box2: List[float] | np.ndarray
) -> bool:
    """Check if two bounding boxes [x1, y1, x2, y2] intersect."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    return (x2 > x1) and (y2 > y1)


def compute_ttc_pair(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    fps: float = 30.0,
    bbox1: Optional[List[float] | np.ndarray] = None,
    bbox2: Optional[List[float] | np.ndarray] = None,
    lateral_margin_factor: float = 1.0,
    enable_lateral_filtering: bool = True
) -> TTCResult:
    """Compute Time-To-Collision (TTC) and lateral miss distance between two objects.

    Args:
        p1: Center position of object 1 (x1, y1).
        p2: Center position of object 2 (x2, y2).
        v1: Velocity vector of object 1 (vx1, vy1) in pixels/frame.
        v2: Velocity vector of object 2 (vx2, vy2) in pixels/frame.
        fps: Video framerate used to convert frame-step velocities to seconds.
        bbox1: Optional bounding box [x1, y1, x2, y2] for object 1.
        bbox2: Optional bounding box [x1, y1, x2, y2] for object 2.
        lateral_margin_factor: Multiplier for effective collision radius tolerance.
        enable_lateral_filtering: If True, suppresses alerts for vehicles passing in separate lanes.

    Returns:
        TTCResult: 3-tuple (ttc, v_c, distance) with attributes .is_collision_course, .d_min, .t_cpa.
    """
    # Displacement vector: d_0 = p2 - p1
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    distance = math.sqrt(dx * dx + dy * dy)

    # Convert velocities from pixels/frame to pixels/second
    vx1_s = float(v1[0]) * fps
    vy1_s = float(v1[1]) * fps
    vx2_s = float(v2[0]) * fps
    vy2_s = float(v2[1]) * fps

    # Relative velocity: v_rel = v1 - v2
    v_rel_x = vx1_s - vx2_s
    v_rel_y = vy1_s - vy2_s
    v_rel_sq = v_rel_x * v_rel_x + v_rel_y * v_rel_y
    v_rel_mag = math.sqrt(v_rel_sq)

    # 1. Line-of-sight closing speed
    dot_product = dx * v_rel_x + dy * v_rel_y
    v_c = dot_product / (distance + 1e-6)

    # If relative speed is negligible or vehicles are diverging along line-of-sight
    if v_rel_mag < 1e-3 or dot_product <= 0:
        return TTCResult(
            ttc=float("inf"),
            v_c=v_c,
            distance=distance,
            is_collision_course=False,
            d_min=distance,
            t_cpa=float("inf"),
            r_coll=0.0
        )

    # 2. Time to Closest Point of Approach (t_cpa)
    t_cpa = dot_product / (v_rel_sq + 1e-6)

    # 3. Lateral Miss Distance at closest approach (d_min)
    d_cpa_x = dx - v_rel_x * t_cpa
    d_cpa_y = dy - v_rel_y * t_cpa
    d_min = math.sqrt(d_cpa_x * d_cpa_x + d_cpa_y * d_cpa_y)

    # 4. Calculate dynamic collision radius based on vehicle physical sizes
    if bbox1 is not None and bbox2 is not None:
        w1 = max(10.0, float(bbox1[2] - bbox1[0]))
        h1 = max(10.0, float(bbox1[3] - bbox1[1]))
        w2 = max(10.0, float(bbox2[2] - bbox2[0]))
        h2 = max(10.0, float(bbox2[3] - bbox2[1]))
        size1 = math.sqrt(w1 * w1 + h1 * h1)
        size2 = math.sqrt(w2 * w2 + h2 * h2)
        r_coll = 0.5 * (size1 + size2) * lateral_margin_factor
    else:
        # Fallback radius when bounding box dimensions are not provided
        r_coll = 60.0 * lateral_margin_factor

    # 5. Evaluate Trajectory Intersection & Lateral Clearance
    if enable_lateral_filtering and (bbox1 is not None and bbox2 is not None):
        # A collision course requires the lateral miss distance to be within collision radius
        is_collision = (d_min <= r_coll) and (t_cpa > 0.0)

        # Immediate contact override: if boxes are already overlapping currently
        if compute_boxes_intersection(bbox1, bbox2):
            is_collision = True
    else:
        # Standard line-of-sight check (backwards compatibility for basic tests)
        is_collision = (v_c > 1e-3)

    if is_collision and t_cpa > 0.0:
        ttc = t_cpa
    elif is_collision and v_c > 1e-3:
        ttc = distance / v_c
    else:
        ttc = float("inf")

    return TTCResult(
        ttc=ttc,
        v_c=v_c,
        distance=distance,
        is_collision_course=is_collision,
        d_min=d_min,
        t_cpa=t_cpa,
        r_coll=r_coll
    )


class CollisionPredictor:
    """Monitors active multi-object tracks and evaluates persistent collision warnings.

    Incorporates lateral clearance filtering to suppress premature false alarms
    caused by oncoming vehicles passing safely in separate lanes.
    """

    def __init__(
        self,
        tau_threshold: float = 2.5,
        consecutive_frames_threshold: int = 3,
        fps: float = 30.0,
        enable_lateral_filtering: bool = True,
        lateral_margin_factor: float = 1.0,
        min_closing_speed: float = 1.0
    ) -> None:
        """Initialize collision predictor.

        Args:
            tau_threshold: Time-to-collision warning threshold in seconds (default 2.5s).
            consecutive_frames_threshold: Consecutive alert frames required to trigger sustained alert (k >= 3).
            fps: Video framerate.
            enable_lateral_filtering: If True, suppresses false alerts when lateral clearance d_min > R_coll.
            lateral_margin_factor: Multiplier tuning physical collision radius sensitivity.
            min_closing_speed: Minimum closing speed in px/s to consider (filters jitter).
        """
        self.tau_threshold: float = tau_threshold
        self.consecutive_frames_threshold: int = consecutive_frames_threshold
        self.fps: float = fps
        self.enable_lateral_filtering: bool = enable_lateral_filtering
        self.lateral_margin_factor: float = lateral_margin_factor
        self.min_closing_speed: float = min_closing_speed

        # Pair streak tracker: key = (min(id1, id2), max(id1, id2)) -> consecutive alert count
        self.pair_streaks: Dict[Tuple[int, int], int] = {}
        # Sustained alerts: set of pairs that have achieved sustained alert status
        self.sustained_pairs: set[Tuple[int, int]] = set()
        self.first_sustained_frame: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset internal state for a new video sequence."""
        self.pair_streaks.clear()
        self.sustained_pairs.clear()
        self.first_sustained_frame = None
        self.history.clear()

    def evaluate_frame(
        self,
        frame_idx: int,
        tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate collision risk across all track pairs in the current frame.

        Args:
            frame_idx: Current video frame index.
            tracks: List of active track dictionaries from tracker.
                    Each dict must contain 'id', 'center', 'velocity', and optionally 'bbox'.

        Returns:
            alerts: List of collision event dicts for pairs meeting the collision criteria.
        """
        num_tracks = len(tracks)
        active_pair_keys: set[Tuple[int, int]] = set()
        alerts: List[Dict[str, Any]] = []

        for i in range(num_tracks):
            for j in range(i + 1, num_tracks):
                trk_i = tracks[i]
                trk_j = tracks[j]
                id_i = int(trk_i["id"])
                id_j = int(trk_j["id"])

                pair_key = (min(id_i, id_j), max(id_i, id_j))
                active_pair_keys.add(pair_key)

                bbox_i = trk_i.get("bbox")
                bbox_j = trk_j.get("bbox")

                res = compute_ttc_pair(
                    p1=trk_i["center"],
                    p2=trk_j["center"],
                    v1=trk_i["velocity"],
                    v2=trk_j["velocity"],
                    fps=self.fps,
                    bbox1=bbox_i,
                    bbox2=bbox_j,
                    lateral_margin_factor=self.lateral_margin_factor,
                    enable_lateral_filtering=self.enable_lateral_filtering
                )

                # Valid alert criteria:
                # 1. Finite positive TTC below tau_threshold
                # 2. Confirmed trajectory intersection / collision course (within lateral clearance)
                # 3. Closing speed exceeds minimum threshold
                is_alert = (
                    math.isfinite(res.ttc)
                    and (res.ttc < self.tau_threshold)
                    and res.is_collision_course
                    and (res.v_c >= self.min_closing_speed)
                )

                if is_alert:
                    self.pair_streaks[pair_key] = self.pair_streaks.get(pair_key, 0) + 1
                else:
                    self.pair_streaks[pair_key] = 0

                streak = self.pair_streaks[pair_key]
                sustained = streak >= self.consecutive_frames_threshold

                if sustained and pair_key not in self.sustained_pairs:
                    self.sustained_pairs.add(pair_key)
                    if self.first_sustained_frame is None:
                        self.first_sustained_frame = frame_idx

                if is_alert:
                    alert_record = {
                        "frame_idx": frame_idx,
                        "pair": pair_key,
                        "ttc": res.ttc,
                        "closing_speed": res.v_c,
                        "distance": res.distance,
                        "lateral_miss_distance": res.d_min,
                        "collision_radius": res.r_coll,
                        "streak": streak,
                        "is_sustained": sustained,
                    }
                    alerts.append(alert_record)
                    self.history.append(alert_record)

        # Reset streak for any pairs no longer simultaneously tracked
        existing_keys = list(self.pair_streaks.keys())
        for k in existing_keys:
            if k not in active_pair_keys:
                self.pair_streaks[k] = 0

        return alerts


def evaluate_negative_controls(
    benchmark_dataset,
    tau_threshold: float = 2.5,
    consecutive_frames_threshold: int = 3,
    max_frames_per_sequence: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate collision predictor across negative control traffic videos to measure False Positive Rate.

    Args:
        benchmark_dataset: TUDATBenchmarkDataset instance containing negative control sequences.
        tau_threshold: Collision warning horizon in seconds.
        consecutive_frames_threshold: Consecutive alert frames required for sustained alert.
        max_frames_per_sequence: Optional cap on frames evaluated per video.

    Returns:
        results: Dict containing total control sequences, false alarm count, and false_positive_rate.
    """
    neg_sequences = benchmark_dataset.negative_sequences
    if not neg_sequences:
        return {
            "total_sequences": 0,
            "false_alarms": 0,
            "false_positive_rate": 0.0,
            "sequence_results": []
        }

    seq_reports = []
    false_alarm_count = 0

    for seq in neg_sequences:
        predictor = CollisionPredictor(
            tau_threshold=tau_threshold,
            consecutive_frames_threshold=consecutive_frames_threshold,
            fps=seq.fps,
            enable_lateral_filtering=True
        )

        n_frames = min(len(seq), max_frames_per_sequence) if max_frames_per_sequence else len(seq)
        total_alerts = 0

        # We evaluate tracking pairs using the ground truth boxes across frames
        for f_idx in range(n_frames):
            gt_objects = seq.get_annotations(f_idx)
            if len(gt_objects) < 2:
                continue

            tracks = []
            for obj in gt_objects:
                tracks.append({
                    "id": obj.track_id,
                    "center": obj.center,
                    # Fallback velocity estimate
                    "velocity": (1.0, 0.0),
                    "bbox": obj.bbox
                })

            frame_alerts = predictor.evaluate_frame(f_idx, tracks)
            total_alerts += len(frame_alerts)

        has_false_alarm = predictor.first_sustained_frame is not None
        if has_false_alarm:
            false_alarm_count += 1

        seq_reports.append({
            "sequence": seq.sequence_name,
            "frames_evaluated": n_frames,
            "has_false_alarm": has_false_alarm,
            "first_false_alarm_frame": predictor.first_sustained_frame,
            "raw_alert_frames_count": total_alerts
        })

    fpr = float(false_alarm_count) / max(1, len(neg_sequences))
    return {
        "total_sequences": len(neg_sequences),
        "false_alarms": false_alarm_count,
        "false_positive_rate": fpr,
        "sequence_results": seq_reports
    }

