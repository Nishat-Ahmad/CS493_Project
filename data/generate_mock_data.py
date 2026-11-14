"""Synthetic video and ground-truth annotation generator for collision benchmarking.

Generates a 300-frame 30 FPS video sequence (640x640) depicting two vehicles
approaching each other on a roadway and colliding at frame F=180.
Outputs:
  - data/synthetic_sequence/frames/ (JPEG frames)
  - data/synthetic_sequence/sequence.mp4 (Video file)
  - data/synthetic_sequence/annotations.json (Ground-truth bounding boxes and accident frame)
  - data/calibration/ (30 representative calibration frames for INT8 quantization)
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np


def draw_vehicle(
    canvas: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    color_bgr: Tuple[int, int, int],
    heading_right: bool = True,
    sprite: Optional[np.ndarray] = None
) -> Tuple[int, int, int, int]:
    """Draw or blend a vehicle onto the canvas.

    Args:
        canvas: Image array to draw on (BGR).
        center_x: Center X coordinate.
        center_y: Center Y coordinate.
        width: Vehicle width.
        height: Vehicle height.
        color_bgr: Main body color fallback.
        heading_right: True if vehicle is travelling towards the right.
        sprite: Optional realistic vehicle image patch.

    Returns:
        bbox: Integer bounding box [x1, y1, x2, y2].
    """
    w_int = int(round(width))
    h_int = int(round(height))
    x1 = int(round(center_x - w_int / 2.0))
    y1 = int(round(center_y - h_int / 2.0))
    x2 = x1 + w_int
    y2 = y1 + h_int

    H, W = canvas.shape[:2]
    x1_c = max(0, min(W, x1))
    y1_c = max(0, min(H, y1))
    x2_c = max(0, min(W, x2))
    y2_c = max(0, min(H, y2))

    if x2_c <= x1_c or y2_c <= y1_c:
        return x1, y1, x2, y2

    if sprite is not None:
        veh_patch = cv2.resize(sprite, (w_int, h_int))
        if not heading_right:
            veh_patch = cv2.flip(veh_patch, 1)

        src_x1 = x1_c - x1
        src_y1 = y1_c - y1
        src_x2 = src_x1 + (x2_c - x1_c)
        src_y2 = src_y1 + (y2_c - y1_c)

        canvas[y1_c:y2_c, x1_c:x2_c] = veh_patch[src_y1:src_y2, src_x1:src_x2]
        return x1, y1, x2, y2

    # Fallback procedural rendering
    cv2.rectangle(canvas, (x1_c, y1_c), (x2_c, y2_c), color_bgr, -1)
    cv2.rectangle(canvas, (x1_c, y1_c), (x2_c, y2_c), (30, 30, 30), 2)

    # Windshield / Cabin
    cabin_w = int(width * 0.45)
    cabin_h = int(height * 0.6)
    cx = int(center_x - (cabin_w * 0.2 if heading_right else -cabin_w * 0.2))
    cy = int(center_y)
    cw1 = max(x1_c, cx - cabin_w // 2)
    ch1 = max(y1_c, cy - cabin_h // 2)
    cw2 = min(x2_c, cx + cabin_w // 2)
    ch2 = min(y2_c, cy + cabin_h // 2)
    if cw2 > cw1 and ch2 > ch1:
        cv2.rectangle(canvas, (cw1, ch1), (cw2, ch2), (210, 210, 210), -1)
        cv2.rectangle(canvas, (cw1, ch1), (cw2, ch2), (60, 60, 60), 1)

    # Headlights
    hl_color = (0, 255, 255)
    hl_size = max(2, int(height * 0.15))
    if heading_right:
        hl_x = min(W - 2, x2_c - 3)
        cv2.circle(canvas, (hl_x, y1_c + hl_size + 2), hl_size, hl_color, -1)
        cv2.circle(canvas, (hl_x, y2_c - hl_size - 2), hl_size, hl_color, -1)
    else:
        hl_x = max(2, x1_c + 3)
        cv2.circle(canvas, (hl_x, y1_c + hl_size + 2), hl_size, hl_color, -1)
        cv2.circle(canvas, (hl_x, y2_c - hl_size - 2), hl_size, hl_color, -1)

    return x1, y1, x2, y2


def create_road_background(width: int = 640, height: int = 640, frame_idx: int = 0) -> np.ndarray:
    """Create a realistic roadway background with asphalt, curbs, and lane markings."""
    # Asphalt surface
    bg = np.full((height, width, 3), (55, 55, 55), dtype=np.uint8)

    # Sidewalks / Curbs
    curb_height = int(height * 0.22)
    bg[:curb_height, :] = (40, 110, 40)   # Green foliage / verge
    bg[-curb_height:, :] = (40, 110, 40)
    bg[curb_height:curb_height + 8, :] = (180, 180, 180)  # Concrete curb top
    bg[-curb_height - 8:-curb_height, :] = (180, 180, 180)  # Concrete curb bottom

    # Road shoulder lines (continuous solid white)
    cv2.line(bg, (0, curb_height + 12), (width, curb_height + 12), (240, 240, 240), 3)
    cv2.line(bg, (0, height - curb_height - 12), (width, height - curb_height - 12), (240, 240, 240), 3)

    # Center lane line (dashed yellow)
    center_y = height // 2
    dash_length = 30
    gap_length = 20
    for x in range(0, width, dash_length + gap_length):
        cv2.line(bg, (x, center_y), (min(width, x + dash_length), center_y), (0, 215, 255), 3)

    return bg


def generate_mock_dataset(
    output_base_dir: Optional[Path | str] = None,
    num_frames: int = 300,
    fps: int = 30,
    accident_frame: int = 180,
    width: int = 640,
    height: int = 640
) -> Dict[str, Any]:
    """Generate synthetic collision dataset and calibration samples.

    Args:
        output_base_dir: Root directory for datasets.
        num_frames: Total number of frames in sequence.
        fps: Framerate.
        accident_frame: Frame index where impact occurs.
        width: Video width.
        height: Video height.

    Returns:
        metadata: Dictionary containing generation statistics and file paths.
    """
    base_dir = Path(output_base_dir) if output_base_dir else REPO_ROOT / "data"
    seq_dir = base_dir / "synthetic_sequence"
    frames_dir = seq_dir / "frames"
    calib_dir = base_dir / "calibration"

    frames_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    video_path = seq_dir / "sequence.mp4"
    anno_path = seq_dir / "annotations.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))

    # Vehicle motion trajectory parameters
    # Vehicle 1: Moves Left-to-Right along lower lane (y ~ 390)
    v1_start_x = 70.0
    v1_target_x = 320.0
    v1_y = 390.0
    v1_w = 90.0
    v1_h = 52.0
    v1_color = (200, 60, 40)  # Blue-ish BGR

    # Vehicle 2: Moves Right-to-Left, veering slightly across center line into lane 1
    v2_start_x = 570.0
    v2_target_x = 320.0
    v2_start_y = 260.0
    v2_target_y = 390.0  # Veers into lower lane
    v2_w = 90.0
    v2_h = 52.0
    v2_color = (40, 40, 220)  # Red BGR

    # Load optional realistic vehicle sprite if present
    sprite_path = base_dir / "vehicle_sprite.png"
    veh_sprite = cv2.imread(str(sprite_path)) if sprite_path.exists() else None

    annotations_list: List[Dict[str, Any]] = []
    calib_step = max(1, num_frames // 30)

    for f in range(num_frames):
        canvas = create_road_background(width=width, height=height, frame_idx=f)
        frame_objs: List[Dict[str, Any]] = []

        if f < accident_frame:
            progress = float(f) / float(accident_frame)
            v1_x = v1_start_x + (v1_target_x - v1_w * 0.45 - v1_start_x) * progress
            v2_x = v2_start_x + (v2_target_x + v2_w * 0.45 - v2_start_x) * progress
            v2_y = v2_start_y + (v2_target_y - v2_start_y) * (progress ** 1.5)
        else:
            # Post-accident: vehicles deform and halt near center
            v1_x = v1_target_x - v1_w * 0.45
            v2_x = v2_target_x + v2_w * 0.45
            v2_y = v2_target_y

            # Visual impact debris / flash at collision frame
            if f < accident_frame + 15:
                cv2.circle(canvas, (int(v1_target_x), int(v2_target_y)), int((f - accident_frame + 5) * 4), (0, 165, 255), 2)

        # Draw Vehicle 1 (COCO class_id 2 = 'car')
        bx1_1, by1_1, bx2_1, by2_1 = draw_vehicle(
            canvas, v1_x, v1_y, v1_w, v1_h, v1_color, heading_right=True, sprite=veh_sprite
        )
        frame_objs.append({
            "id": 1,
            "class_id": 2,
            "class_name": "car",
            "bbox": [bx1_1, by1_1, bx2_1, by2_1],
            "center": [float(v1_x), float(v1_y)]
        })

        # Draw Vehicle 2
        bx1_2, by1_2, bx2_2, by2_2 = draw_vehicle(
            canvas, v2_x, v2_y, v2_w, v2_h, v2_color, heading_right=False, sprite=veh_sprite
        )
        frame_objs.append({
            "id": 2,
            "class_id": 2,
            "class_name": "car",
            "bbox": [bx1_2, by1_2, bx2_2, by2_2],
            "center": [float(v2_x), float(v2_y)]
        })

        # Save frame image
        frame_filename = f"frame_{f:04d}.jpg"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), canvas)

        # Append to video stream
        video_writer.write(canvas)

        # Save sample for INT8 calibration
        if f % calib_step == 0 and (len(list(calib_dir.glob("*.jpg"))) < 30):
            calib_frame_path = calib_dir / f"calib_{f:04d}.jpg"
            cv2.imwrite(str(calib_frame_path), canvas)

        annotations_list.append({
            "frame": f,
            "timestamp_sec": float(f) / float(fps),
            "objects": frame_objs
        })

    video_writer.release()

    metadata = {
        "sequence_name": "synthetic_head_on_collision",
        "num_frames": num_frames,
        "fps": fps,
        "accident_frame": accident_frame,
        "resolution": [width, height],
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "calibration_dir": str(calib_dir),
        "annotations": annotations_list
    }

    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[MockData] Successfully synthesized {num_frames} frames to {seq_dir}")
    print(f"[MockData] Generated 30 calibration frames in {calib_dir}")
    print(f"[MockData] Collision impact set at Frame F={accident_frame}")
    return metadata


if __name__ == "__main__":
    generate_mock_dataset()
