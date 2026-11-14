"""Comprehensive test suite for edge collision benchmark pipeline.

Verifies:
  1. Kalman Filter & SORT tracker lifecycle (initiation, trajectory updates, track deletion).
  2. Kinematic TTC calculation (converging, diverging, parallel trajectories, sustained alert streak).
  3. Metric calculations (Throughput FPS, RTM, coordinate jitter variance, mAP@50).
  4. Mock data generation and dataset loader parsing.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from tracking.kalman_tracker import (
    SortTracker,
    KalmanBoxTracker,
    bbox_to_z,
    z_to_bbox,
    compute_iou_matrix
)
from tracking.collision_kinematics import (
    compute_ttc_pair,
    CollisionPredictor
)
from benchmark.metrics import (
    compute_latency_metrics,
    compute_rtm,
    compute_coordinate_jitter,
    compute_map50,
    compute_iou
)
from data.generate_mock_data import generate_mock_dataset
from data.dataset_loader import (
    SequenceDataset,
    FrameAnnotation,
    TUDATSequence,
    TUDATBenchmarkDataset,
    load_dataset
)


# -----------------------------------------------------------------------------
# 1. Kalman Filter & SORT Lifecycle Tests
# -----------------------------------------------------------------------------

def test_kalman_lifecycle():
    """Verify SORT tracker handles initiation, continuous state updates, and lost track pruning."""
    SortTracker.reset_counter()
    tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)

    # 1. Track initiation
    det1 = np.array([[100.0, 100.0, 160.0, 160.0, 0.95, 2]])
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1, "Expected 1 active track on first detection"
    track_id = tracks1[0]["id"]
    init_bbox = tracks1[0]["bbox"]
    assert np.allclose(init_bbox, [100.0, 100.0, 160.0, 160.0], atol=2.0)

    # 2. Continuous trajectory updates (vehicle moves rightwards)
    for step in range(1, 6):
        x_shift = float(step * 5.0)
        det_step = np.array([[100.0 + x_shift, 100.0, 160.0 + x_shift, 160.0, 0.95, 2]])
        tracks = tracker.update(det_step)
        assert len(tracks) == 1
        assert tracks[0]["id"] == track_id, "Track ID should remain consistent across matched frames"
        # Velocity in X direction should be positive
        vx, vy = tracks[0]["velocity"]
        assert vx > 0.0, f"Expected positive X velocity, got vx={vx}"

    # 3. Track deletion after max_age missed frames
    empty_det = np.empty((0, 6))
    for _ in range(tracker.max_age + 2):
        _ = tracker.update(empty_det)

    assert len(tracker.trackers) == 0, "Tracker instances should be pruned after exceeding max_age"


def test_bbox_and_z_conversions():
    """Verify coordinate round-trip transformations between bbox [x1, y1, x2, y2] and z [u, v, s, r]."""
    orig_bbox = np.array([50.0, 60.0, 150.0, 200.0])  # w=100, h=140, u=100, v=130, s=14000, r=100/140
    z = bbox_to_z(orig_bbox)
    recon_bbox = z_to_bbox(z)
    assert np.allclose(orig_bbox, recon_bbox, atol=1e-4)


# -----------------------------------------------------------------------------
# 2. Kinematic Time-To-Collision (TTC) Tests
# -----------------------------------------------------------------------------

def test_ttc_diverging_and_converging():
    """Verify TTC returns finite positive values for converging trajectories and infinity for diverging/parallel."""
    fps = 30.0

    # Case A: Directly converging head-on along X axis
    # Vehicle 1 at x=100 moving right (+2 px/frame), Vehicle 2 at x=400 moving left (-2 px/frame)
    p1 = (100.0, 200.0)
    p2 = (400.0, 200.0)
    v1 = (2.0, 0.0)
    v2 = (-2.0, 0.0)

    ttc, v_c, dist = compute_ttc_pair(p1, p2, v1, v2, fps=fps)
    assert math.isfinite(ttc), "TTC should be finite for head-on converging paths"
    assert ttc > 0.0, "TTC must be strictly positive before impact"
    # Expected: distance = 300 px; closing speed = (2 - (-2)) * 30 = 120 px/s; TTC = 300 / 120 = 2.5 s
    assert pytest.approx(ttc, rel=1e-3) == 2.5
    assert pytest.approx(v_c, rel=1e-3) == 120.0
    assert pytest.approx(dist, rel=1e-3) == 300.0

    # Case B: Diverging trajectories (moving away from each other)
    v1_div = (-2.0, 0.0)
    v2_div = (2.0, 0.0)
    ttc_div, v_c_div, _ = compute_ttc_pair(p1, p2, v1_div, v2_div, fps=fps)
    assert math.isinf(ttc_div), "TTC should be infinity for diverging paths"
    assert v_c_div <= 0.0, "Closing speed should be <= 0 for diverging paths"

    # Case C: Parallel trajectories (same speed, parallel heading along Y)
    v1_par = (0.0, 5.0)
    v2_par = (0.0, 5.0)
    ttc_par, v_c_par, _ = compute_ttc_pair(p1, p2, v1_par, v2_par, fps=fps)
    assert math.isinf(ttc_par), "TTC should be infinity for parallel equal-speed trajectories"


def test_sustained_collision_alert():
    """Verify CollisionPredictor triggers sustained alert only after >= 3 consecutive frames."""
    predictor = CollisionPredictor(tau_threshold=2.5, consecutive_frames_threshold=3, fps=30.0)

    # Frame tracks yielding TTC ~ 2.0s (< 2.5s threshold)
    def make_converging_tracks():
        return [
            {"id": 1, "center": (100.0, 200.0), "velocity": (2.0, 0.0)},
            {"id": 2, "center": (340.0, 200.0), "velocity": (-2.0, 0.0)},
        ]

    # Frame 1: Alert 1 (streak=1, not sustained)
    alerts1 = predictor.evaluate_frame(1, make_converging_tracks())
    assert len(alerts1) == 1
    assert alerts1[0]["streak"] == 1
    assert not alerts1[0]["is_sustained"]
    assert predictor.first_sustained_frame is None

    # Frame 2: Alert 2 (streak=2, not sustained)
    alerts2 = predictor.evaluate_frame(2, make_converging_tracks())
    assert alerts2[0]["streak"] == 2
    assert not alerts2[0]["is_sustained"]
    assert predictor.first_sustained_frame is None

    # Frame 3: Alert 3 (streak=3 -> sustained alert achieved)
    alerts3 = predictor.evaluate_frame(3, make_converging_tracks())
    assert alerts3[0]["streak"] == 3
    assert alerts3[0]["is_sustained"]
    assert predictor.first_sustained_frame == 3


def test_lateral_trajectory_filter_suppresses_opposite_lane_false_alarms():
    """Verify that lateral trajectory filtering prevents false alarms for opposite lane vehicles."""
    # Case: Two vehicles approaching each other on parallel lanes separated by 120 pixels laterally
    # Lane 1 (y=200, moving rightwards +2 px/frame): bbox size 60x40
    # Lane 2 (y=320, moving leftwards -2 px/frame):  bbox size 60x40
    p1 = (100.0, 200.0)
    p2 = (300.0, 320.0)
    v1 = (2.0, 0.0)
    v2 = (-2.0, 0.0)
    bbox1 = [70.0, 180.0, 130.0, 220.0]
    bbox2 = [270.0, 300.0, 330.0, 340.0]

    # Without lateral filtering (naive line-of-sight closing speed)
    res_naive = compute_ttc_pair(
        p1=p1, p2=p2, v1=v1, v2=v2, fps=30.0,
        bbox1=bbox1, bbox2=bbox2, enable_lateral_filtering=False
    )
    # Naive filter sees closing distance and erroneously reports finite TTC
    assert math.isfinite(res_naive.ttc), "Naive filter without lateral clearance triggers false alert"
    assert res_naive.ttc < 3.0

    # With lateral trajectory filtering enabled (our fix)
    res_filtered = compute_ttc_pair(
        p1=p1, p2=p2, v1=v1, v2=v2, fps=30.0,
        bbox1=bbox1, bbox2=bbox2, enable_lateral_filtering=True
    )
    # Filter correctly computes lateral miss distance d_min = 120px > R_coll (~72px)
    assert res_filtered.d_min > res_filtered.r_coll, f"Expected d_min ({res_filtered.d_min}) > r_coll ({res_filtered.r_coll})"
    assert not res_filtered.is_collision_course, "Parallel lane vehicles must not be flagged as collision course"
    assert math.isinf(res_filtered.ttc), "Filtered TTC should be infinity for safe lane pass"

    # Verify that CollisionPredictor suppresses alerts across multiple frames
    predictor = CollisionPredictor(tau_threshold=2.5, consecutive_frames_threshold=3, fps=30.0, enable_lateral_filtering=True)
    opposite_lane_tracks = [
        {"id": 1, "center": p1, "velocity": v1, "bbox": bbox1},
        {"id": 2, "center": p2, "velocity": v2, "bbox": bbox2},
    ]
    for f in range(1, 10):
        alerts = predictor.evaluate_frame(f, opposite_lane_tracks)
        assert len(alerts) == 0, f"Frame {f} should have 0 alerts with lateral trajectory filtering"
    assert predictor.first_sustained_frame is None


def test_negative_control_videos_zero_false_positive_rate():
    """Verify that normal traffic control sequences yield 0.0 False Positive Rate."""
    from tracking.collision_kinematics import evaluate_negative_controls

    repo_root = Path(__file__).resolve().parent.parent
    tudat_candidates = [
        repo_root / "data" / "tudat",
        repo_root / "TU-DAT-main"
    ]
    tudat_dir = next((c for c in tudat_candidates if (c / "videos").exists()), None)
    if tudat_dir is None:
        pytest.skip("TU-DAT videos directory not found, skipping negative control FPR test.")

    benchmark = TUDATBenchmarkDataset(base_dir=tudat_dir, target_resolution=(640, 640))
    if len(benchmark.negative_sequences) == 0:
        pytest.skip("No negative control sequences found.")

    results = evaluate_negative_controls(benchmark, tau_threshold=2.5, consecutive_frames_threshold=3)
    assert results["total_sequences"] == 5, f"Expected 5 negative control sequences, got {results['total_sequences']}"
    assert results["false_alarms"] == 0, f"Expected 0 false alarms, got {results['false_alarms']}"
    assert results["false_positive_rate"] == 0.0, f"Expected 0.0 FPR, got {results['false_positive_rate']}"



# -----------------------------------------------------------------------------
# 3. Metric Calculation Tests
# -----------------------------------------------------------------------------

def test_metrics_calculation():
    """Verify mathematical properties of throughput FPS, RTM, jitter variance, and mAP@50."""
    # 1. Latency & FPS
    latencies = [20.0, 25.0, 30.0]  # mean = 25.0 ms -> FPS = 1000 / 25 = 40.0
    lat_stats = compute_latency_metrics(latencies)
    assert pytest.approx(lat_stats["mean_latency_ms"]) == 25.0
    assert pytest.approx(lat_stats["fps"]) == 40.0

    # 2. Resolution Time Margin (RTM)
    # Accident at F=180, sustained alert at F=120, FPS=30 -> RTM = (180 - 120) / 30 = 2.0 s
    assert pytest.approx(compute_rtm(180, 120, 30.0)) == 2.0
    # No sustained alert before impact -> RTM = 0.0
    assert compute_rtm(180, None, 30.0) == 0.0
    assert compute_rtm(180, 200, 30.0) == 0.0  # Alert after collision

    # 3. Coordinate Jitter Variance
    # Perfectly linear constant velocity motion -> second discrete difference is identically 0
    smooth_traj = {1: [(float(t * 10), float(t * 5)) for t in range(10)]}
    zero_jitter = compute_coordinate_jitter(smooth_traj)
    assert pytest.approx(zero_jitter, abs=1e-7) == 0.0

    # Jittery trajectory with alternating oscillations: c_t = (t, (-1)^t)
    jitter_traj = {1: [(float(t), 1.0 if t % 2 == 0 else -1.0) for t in range(5)]}
    # For t=1, 2, 3: c_{t+1} - 2*c_t + c_{t-1} = (0, 1 - 2(-1) + 1) = (0, 4) -> norm^2 = 16
    noisy_jitter = compute_coordinate_jitter(jitter_traj)
    assert pytest.approx(noisy_jitter, rel=1e-4) == 16.0

    # 4. mAP@50 test with identical bounding boxes
    gt_frame = {0: [{"bbox": [10.0, 10.0, 50.0, 50.0], "class_id": 2}]}
    pred_perfect = {0: [{"bbox": [10.0, 10.0, 50.0, 50.0], "score": 0.99, "class_id": 2}]}
    map_perfect = compute_map50(pred_perfect, gt_frame, iou_threshold=0.50)
    assert pytest.approx(map_perfect, rel=1e-3) == 1.0

    pred_disjoint = {0: [{"bbox": [200.0, 200.0, 300.0, 300.0], "score": 0.99, "class_id": 2}]}
    map_disjoint = compute_map50(pred_disjoint, gt_frame, iou_threshold=0.50)
    assert pytest.approx(map_disjoint, abs=1e-7) == 0.0


# -----------------------------------------------------------------------------
# 4. Synthetic Data Generation & Loading Tests
# -----------------------------------------------------------------------------

def test_mock_data_generation():
    """Verify synthetic frame synthesis, bounding box annotations, and dataset loader."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Generate a mini test sequence of 15 frames
        metadata = generate_mock_dataset(
            output_base_dir=tmp_path,
            num_frames=15,
            fps=30,
            accident_frame=10,
            width=320,
            height=320
        )

        assert metadata["num_frames"] == 15
        assert metadata["accident_frame"] == 10

        seq_dir = tmp_path / "synthetic_sequence"
        assert (seq_dir / "annotations.json").exists()
        assert (seq_dir / "sequence.mp4").exists()
        assert len(list((seq_dir / "frames").glob("*.jpg"))) == 15
        assert len(list((tmp_path / "calibration").glob("*.jpg"))) > 0

        # Validate sequence dataset loading
        dataset = SequenceDataset(sequence_path=seq_dir)
        assert len(dataset) == 15
        assert dataset.fps == 30.0
        assert dataset.accident_frame == 10

        # Read sample frame and annotation
        f_idx, frame_img, annos = next(iter(dataset))
        assert f_idx == 0
        assert frame_img is not None
        assert frame_img.shape == (320, 320, 3)
        assert len(annos) == 2, "Expected 2 vehicles in frame 0"


def test_tudat_benchmark_dataset_discovery():
    """Verify TU-DAT real-world roadside anomaly sequence loading and annotation parsing."""
    repo_root = Path(__file__).resolve().parent.parent
    tudat_candidates = [
        repo_root / "data" / "tudat",
        repo_root / "TU-DAT-main"
    ]
    tudat_dir = next((c for c in tudat_candidates if (c / "videos").exists()), None)
    if tudat_dir is None:
        pytest.skip("TU-DAT videos directory not found, skipping real-world data test.")

    benchmark = TUDATBenchmarkDataset(base_dir=tudat_dir, target_resolution=(640, 640))
    assert len(benchmark.positive_sequences) == 10, f"Expected 10 positive videos, got {len(benchmark.positive_sequences)}"
    assert len(benchmark.negative_sequences) == 5, f"Expected 5 negative videos, got {len(benchmark.negative_sequences)}"
    assert len(benchmark) == 15

    # Test sample positive crash sequence
    seq_v1 = benchmark["v1"]
    assert seq_v1.is_anomaly is True
    assert seq_v1.fps > 50.0
    assert seq_v1.accident_frame is not None and seq_v1.accident_frame > 0
    assert len(seq_v1.annotations_by_frame) > 0

    # Test sample challenging adverse lighting sequence
    seq_v36 = benchmark["v36"]
    assert seq_v36.scenario_tag == "challenging_adverse_light"
    assert seq_v36.accident_frame == 166

    # Test sample negative control sequence
    seq_v47 = benchmark["v47"]
    assert seq_v47.is_anomaly is False
    assert seq_v47.accident_frame is None
    assert seq_v47.scenario_tag == "negative_control"

    # Test summary catalog output
    summary = benchmark.summary()
    assert len(summary) == 15
    assert all("sequence" in row and "fps" in row and "accident_frame" in row for row in summary)


