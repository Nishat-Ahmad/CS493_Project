"""Dataset loader for TU-DAT and synthetic collision benchmark sequences.

Supports parsing video sequences (MP4/AVI) or image frame directories, alongside
ground-truth annotations containing bounding boxes, track IDs, and ground-truth
accident impact frame F_accident.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import cv2
import numpy as np


class FrameAnnotation:
    """Ground truth annotation for a single object in a frame."""

    def __init__(
        self,
        track_id: int,
        bbox: List[float] | Tuple[float, float, float, float] | np.ndarray,
        class_id: int = 2,
        class_name: str = "vehicle"
    ) -> None:
        self.track_id: int = int(track_id)
        self.bbox: np.ndarray = np.asarray(bbox, dtype=np.float64)  # [x1, y1, x2, y2]
        self.class_id: int = int(class_id)
        self.class_name: str = class_name

    @property
    def center(self) -> Tuple[float, float]:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return float(self.bbox[0] + w / 2.0), float(self.bbox[1] + h / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.track_id,
            "bbox": self.bbox.tolist(),
            "center": list(self.center),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class SequenceDataset:
    """Represents a sequential driving video dataset with collision ground truth."""

    def __init__(
        self,
        sequence_path: Path | str,
        annotations_path: Optional[Path | str] = None,
        default_fps: float = 30.0,
        default_accident_frame: int = 180
    ) -> None:
        """Initialize sequence dataset loader.

        Args:
            sequence_path: Path to sequence folder, video file (.mp4), or frame directory.
            annotations_path: Optional explicit path to annotations JSON or text file.
            default_fps: Fallback FPS if not specified in metadata.
            default_accident_frame: Fallback accident frame if not specified.
        """
        self.root_path = Path(sequence_path)
        self.default_fps = default_fps
        self.fps = default_fps
        self.accident_frame = default_accident_frame
        self.annotations_by_frame: Dict[int, List[FrameAnnotation]] = {}
        self.frame_paths: List[Path] = []
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.resolution: Tuple[int, int] = (640, 640)

        self._discover_and_load(annotations_path)

    def _discover_and_load(self, explicit_anno_path: Optional[Path | str] = None) -> None:
        """Locate frames, video files, and parse annotation files."""
        # Check if root is a video file
        if self.root_path.is_file() and self.root_path.suffix.lower() in [".mp4", ".avi", ".mkv"]:
            self._load_from_video_file(self.root_path)
            anno_candidate = self.root_path.with_name("annotations.json")
            if explicit_anno_path:
                self._load_annotations(Path(explicit_anno_path))
            elif anno_candidate.exists():
                self._load_annotations(anno_candidate)
            return

        # Check directory structure
        if self.root_path.is_dir():
            frames_dir = self.root_path / "frames"
            if frames_dir.exists() and frames_dir.is_dir():
                target_dir = frames_dir
            else:
                target_dir = self.root_path

            # Collect image frames
            img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            all_imgs = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in img_exts])
            if all_imgs:
                self.frame_paths = all_imgs
                self.total_frames = len(all_imgs)
                # Sample first frame to get resolution
                sample = cv2.imread(str(all_imgs[0]))
                if sample is not None:
                    self.resolution = (sample.shape[1], sample.shape[0])
            else:
                # Check for video file inside directory
                video_files = list(self.root_path.glob("*.mp4")) + list(self.root_path.glob("*.avi"))
                if video_files:
                    self._load_from_video_file(video_files[0])

            # Look for annotations
            if explicit_anno_path:
                self._load_annotations(Path(explicit_anno_path))
            else:
                for cand in [
                    self.root_path / "annotations.json",
                    self.root_path / "labels.json",
                    self.root_path / "gt.json"
                ]:
                    if cand.exists():
                        self._load_annotations(cand)
                        break

    def _load_from_video_file(self, video_path: Path) -> None:
        """Initialize video stream reader."""
        self.video_cap = cv2.VideoCapture(str(video_path))
        if not self.video_cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if v_fps > 0:
            self.fps = v_fps
        w = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self.resolution = (w, h)

    def _load_annotations(self, anno_path: Path) -> None:
        """Parse annotations JSON (synthetic or TU-DAT standard)."""
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "fps" in data:
            self.fps = float(data["fps"])
        if "accident_frame" in data:
            self.accident_frame = int(data["accident_frame"])
        if "resolution" in data:
            self.resolution = (int(data["resolution"][0]), int(data["resolution"][1]))

        # Parse frame objects
        anno_entries = data.get("annotations", [])
        if isinstance(anno_entries, list):
            for entry in anno_entries:
                f_idx = entry.get("frame", 0)
                objs = entry.get("objects", [])
                obj_list = []
                for obj in objs:
                    obj_list.append(
                        FrameAnnotation(
                            track_id=obj.get("id", 0),
                            bbox=obj.get("bbox", [0, 0, 0, 0]),
                            class_id=obj.get("class_id", 2),
                            class_name=obj.get("class_name", "vehicle")
                        )
                    )
                self.annotations_by_frame[f_idx] = obj_list
        elif isinstance(anno_entries, dict):
            for f_str, objs in anno_entries.items():
                f_idx = int(f_str)
                obj_list = [
                    FrameAnnotation(
                        track_id=o.get("id", 0),
                        bbox=o.get("bbox", [0, 0, 0, 0]),
                        class_id=o.get("class_id", 2)
                    ) for o in objs
                ]
                self.annotations_by_frame[f_idx] = obj_list

    def __len__(self) -> int:
        return self.total_frames

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Read frame at given index as BGR uint8 image."""
        if index < 0 or index >= self.total_frames:
            return None

        if self.frame_paths and index < len(self.frame_paths):
            return cv2.imread(str(self.frame_paths[index]))

        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            ret, frame = self.video_cap.read()
            if ret:
                return frame

        return None

    def get_annotations(self, index: int) -> List[FrameAnnotation]:
        """Get ground truth annotations for given frame index."""
        return self.annotations_by_frame.get(index, [])

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, List[FrameAnnotation]]]:
        """Iterate through sequence yielding (frame_idx, frame_image, annotations)."""
        for i in range(self.total_frames):
            frame = self.get_frame(i)
            if frame is None:
                break
            annos = self.get_annotations(i)
            yield i, frame, annos


def load_dataset(sequence_dir: Path | str = "data/synthetic_sequence") -> SequenceDataset:
    """Convenience helper to load a sequence dataset."""
    return SequenceDataset(sequence_path=sequence_dir)

