"""Kinematic Time-to-Collision (TTC) trajectory forecasting module.

Evaluates inter-vehicle collision risk based on relative kinematics:
    Displacement vector: d = [x2 - x1, y2 - y1]
    Relative velocity:   v_rel = [vx1 - vx2, vy1 - vy2]
    Closing speed:       v_c = (d . v_rel) / (||d|| + 1e-6)
    Time-to-Collision:   TTC = ||d|| / v_c (if v_c > 0 else infinity)

An alert is triggered when TTC falls below tau_threshold and persists
for consecutive_frames_threshold consecutive frames.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np


def compute_ttc_pair(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    fps: float = 30.0
) -> Tuple[float, float, float]:
    """Compute Time-To-Collision (TTC) between two objects.

    Args:
        p1: Center position of object 1 (x1, y1).
        p2: Center position of object 2 (x2, y2).
        v1: Velocity vector of object 1 (vx1, vy1) in pixels/frame.
        v2: Velocity vector of object 2 (vx2, vy2) in pixels/frame.
        fps: Video framerate used to convert frame-step velocities to seconds.

    Returns:
        ttc: Time-to-collision in seconds (float('inf') if diverging/parallel).
        v_c: Closing speed in pixels/second.
        distance: Euclidean displacement magnitude ||d|| in pixels.
    """
    # Displacement vector: d = p2 - p1
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

    # Closing speed v_c = (d . v_rel) / (||d|| + 1e-6)
    dot_product = dx * v_rel_x + dy * v_rel_y
    v_c = dot_product / (distance + 1e-6)

    if v_c > 1e-3:
        ttc = distance / v_c
    else:
        ttc = float("inf")

    return ttc, v_c, distance


class CollisionPredictor:
    """Monitors active tracks and evaluates persistent collision warnings."""

    def __init__(
        self,
        tau_threshold: float = 2.5,
        consecutive_frames_threshold: int = 3,
        fps: float = 30.0
    ) -> None:
        """Initialize collision predictor.

        Args:
            tau_threshold: Time-to-collision warning threshold in seconds.
            consecutive_frames_threshold: Consecutive alert frames required to trigger sustained alert.
            fps: Video framerate.
        """
        self.tau_threshold: float = tau_threshold
        self.consecutive_frames_threshold: int = consecutive_frames_threshold
        self.fps: float = fps

        # Pair streak tracker: key = (min(id1, id2), max(id1, id2)) -> consecutive alert count
        self.pair_streaks: Dict[Tuple[int, int], int] = {}
        # Sustained alerts: set of pairs that have achieved sustained alert status
        self.sustained_pairs: set[Tuple[int, int]] = set()
        self.first_sustained_frame: Optional[int] = None

    def reset(self) -> None:
        """Reset internal state for a new video sequence."""
        self.pair_streaks.clear()
        self.sustained_pairs.clear()
        self.first_sustained_frame = None

    def evaluate_frame(
        self,
        frame_idx: int,
        tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate collision risk across all track pairs in the current frame.

        Args:
            frame_idx: Current video frame index.
            tracks: List of active track dictionaries from tracker.

        Returns:
            alerts: List of collision event dicts for pairs with TTC < tau_threshold.
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

                ttc, v_c, dist = compute_ttc_pair(
                    p1=trk_i["center"],
                    p2=trk_j["center"],
                    v1=trk_i["velocity"],
                    v2=trk_j["velocity"],
                    fps=self.fps
                )

                is_alert = (ttc < self.tau_threshold) and math.isfinite(ttc)

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
                    alerts.append({
                        "frame_idx": frame_idx,
                        "pair": pair_key,
                        "ttc": ttc,
                        "closing_speed": v_c,
                        "distance": dist,
                        "streak": streak,
                        "is_sustained": sustained,
                    })

        # Reset streak for any pairs no longer simultaneously visible
        existing_keys = list(self.pair_streaks.keys())
        for k in existing_keys:
            if k not in active_pair_keys:
                self.pair_streaks[k] = 0

        return alerts

