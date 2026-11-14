"""Dataset loader for TU-DAT and synthetic collision benchmark sequences.

Supports:
  1. TU-DAT real-world roadside video sequences (.mov/.mp4) and official JSON annotations
     (Positive crash events, Negative control clips, and Challenging adverse conditions).
  2. Multi-video benchmark dataset collection (TUDATBenchmarkDataset) managing batch evaluation.
  3. Synthetic procedural benchmark sequences (data/synthetic_sequence) for quick smoke tests.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union
import cv2
import numpy as np


class FrameAnnotation:
    """Ground-truth bounding box annotation for a single object in a frame."""

    def __init__(
        self,
        track_id: int,
        bbox: List[float] | Tuple[float, float, float, float] | np.ndarray,
        class_id: int = 2,
        class_name: str = "vehicle",
        confidence: float = 1.0
    ) -> None:
        self.track_id: int = int(track_id)
        self.bbox: np.ndarray = np.asarray(bbox, dtype=np.float64)  # [x1, y1, x2, y2]
        self.class_id: int = int(class_id)
        self.class_name: str = class_name
        self.confidence: float = float(confidence)

    @property
    def width(self) -> float:
        return max(0.0, float(self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return max(0.0, float(self.bbox[3] - self.bbox[1]))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.bbox[0] + self.width / 2.0), float(self.bbox[1] + self.height / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.track_id,
            "bbox": self.bbox.tolist(),
            "center": list(self.center),
            "area": self.area,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }


class SequenceDataset:
    """Represents a single driving video sequence with collision ground truth."""

    def __init__(
        self,
        sequence_path: Path | str,
        annotations_path: Optional[Path | str] = None,
        default_fps: float = 30.0,
        default_accident_frame: Optional[int] = 180,
        target_resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        """Initialize single sequence dataset loader.

        Args:
            sequence_path: Path to video file (.mov, .mp4, .avi) or frame directory.
            annotations_path: Optional explicit path to annotations JSON file.
            default_fps: Fallback FPS if not specified in metadata.
            default_accident_frame: Fallback accident frame if not specified.
            target_resolution: Optional (width, height) to scale ground-truth bounding boxes
                               to match model input (e.g., (640, 640)).
        """
        self.root_path = Path(sequence_path)
        self.default_fps = default_fps
        self.fps = default_fps
        self.accident_frame: Optional[int] = default_accident_frame
        self.is_anomaly: bool = True
        self.target_resolution: Optional[Tuple[int, int]] = target_resolution
        self.annotations_by_frame: Dict[int, List[FrameAnnotation]] = {}
        self.frame_paths: List[Path] = []
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.resolution: Tuple[int, int] = (640, 640)
        self.metadata: Dict[str, Any] = {}

        self._discover_and_load(annotations_path)

    def _discover_and_load(self, explicit_anno_path: Optional[Path | str] = None) -> None:
        """Locate frames, video files, and parse annotations."""
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

        # 1. Direct video file
        if self.root_path.is_file() and self.root_path.suffix.lower() in video_exts:
            self._load_from_video_file(self.root_path)

            if explicit_anno_path:
                self._load_annotations(Path(explicit_anno_path))
            else:
                # Look for matching annotation file
                candidates = [
                    self.root_path.with_name(f"{self.root_path.stem}_annotation.json"),
                    self.root_path.with_name("annotations.json"),
                    self.root_path.with_name("gt.json"),
                ]
                for cand in candidates:
                    if cand.exists():
                        self._load_annotations(cand)
                        break
            return

        # 2. Directory structure
        if self.root_path.is_dir():
            frames_dir = self.root_path / "frames"
            target_dir = frames_dir if (frames_dir.exists() and frames_dir.is_dir()) else self.root_path

            img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            all_imgs = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in img_exts])
            if all_imgs:
                self.frame_paths = all_imgs
                self.total_frames = len(all_imgs)
                sample = cv2.imread(str(all_imgs[0]))
                if sample is not None:
                    self.resolution = (sample.shape[1], sample.shape[0])
            else:
                video_files = [p for p in self.root_path.iterdir() if p.suffix.lower() in video_exts]
                if video_files:
                    self._load_from_video_file(video_files[0])

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
        """Initialize OpenCV video stream reader."""
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
        """Parse annotations JSON (supports both TU-DAT format and synthetic format)."""
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})

        # Parse FPS
        if "fps" in self.metadata:
            self.fps = float(self.metadata["fps"])
        elif "fps" in data:
            self.fps = float(data["fps"])

        # Parse resolution
        if "resolution" in self.metadata:
            res = self.metadata["resolution"]
            self.resolution = (int(res[0]), int(res[1]))
        elif "resolution" in data:
            self.resolution = (int(data["resolution"][0]), int(data["resolution"][1]))

        # Calculate bounding box scale factors if target_resolution is requested
        scale_x = 1.0
        scale_y = 1.0
        if self.target_resolution is not None and self.resolution[0] > 0 and self.resolution[1] > 0:
            scale_x = float(self.target_resolution[0]) / float(self.resolution[0])
            scale_y = float(self.target_resolution[1]) / float(self.resolution[1])

        # Parse label and accident frame
        label = data.get("label", "").lower()
        if label == "negative":
            self.is_anomaly = False
            self.accident_frame = None
        else:
            self.is_anomaly = True
            # Method A: First anomaly event peak/start frame
            events = data.get("events", [])
            if events and isinstance(events, list) and len(events) > 0:
                first_event = events[0]
                self.accident_frame = int(first_event.get("peak_frame", first_event.get("start_frame", 0)))
            # Method B: Impact detection
            elif "impact_detection" in data and data["impact_detection"].get("detected", False):
                prim_frame = int(data["impact_detection"].get("primary_impact_frame", 0))
                self.accident_frame = prim_frame if prim_frame > 0 else int(self.total_frames * 0.6)
            # Method C: Explicit accident_frame
            elif "accident_frame" in data:
                self.accident_frame = int(data["accident_frame"])
            else:
                self.accident_frame = int(self.total_frames * 0.65) if self.total_frames > 0 else 180

        # Parse Bounding Boxes
        # Case A: TU-DAT format -> "bounding_boxes_sample"
        if "bounding_boxes_sample" in data:
            samples = data["bounding_boxes_sample"]
            for sample in samples:
                f_idx = int(sample.get("frame", 0))
                objs = sample.get("objects", [])
                obj_list = []
                for idx, obj in enumerate(objs):
                    raw_bbox = obj.get("bbox", [0, 0, 0, 0])
                    # TU-DAT format: [x, y, w, h]
                    x, y, w, h = float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])

                    # Filter out whole-frame background boxes (spurious full-screen artifacts)
                    if self.resolution[0] > 0 and self.resolution[1] > 0:
                        if w >= self.resolution[0] * 0.85 and h >= self.resolution[1] * 0.85:
                            continue

                    # Convert [x, y, w, h] to [x1, y1, x2, y2]
                    x1 = x * scale_x
                    y1 = y * scale_y
                    x2 = (x + w) * scale_x
                    y2 = (y + h) * scale_y

                    obj_list.append(
                        FrameAnnotation(
                            track_id=idx + 1,
                            bbox=[x1, y1, x2, y2],
                            class_id=2,  # COCO car / vehicle
                            class_name=obj.get("class", "vehicle"),
                            confidence=1.0
                        )
                    )
                self.annotations_by_frame[f_idx] = obj_list

        # Case B: Synthetic format -> "annotations"
        elif "annotations" in data:
            anno_entries = data["annotations"]
            if isinstance(anno_entries, list):
                for entry in anno_entries:
                    f_idx = int(entry.get("frame", 0))
                    objs = entry.get("objects", [])
                    obj_list = []
                    for obj in objs:
                        bbox = obj.get("bbox", [0, 0, 0, 0])
                        x1 = float(bbox[0]) * scale_x
                        y1 = float(bbox[1]) * scale_y
                        x2 = float(bbox[2]) * scale_x
                        y2 = float(bbox[3]) * scale_y
                        obj_list.append(
                            FrameAnnotation(
                                track_id=obj.get("id", 0),
                                bbox=[x1, y1, x2, y2],
                                class_id=obj.get("class_id", 2),
                                class_name=obj.get("class_name", "vehicle")
                            )
                        )
                    self.annotations_by_frame[f_idx] = obj_list
            elif isinstance(anno_entries, dict):
                for f_str, objs in anno_entries.items():
                    f_idx = int(f_str)
                    obj_list = []
                    for obj in objs:
                        bbox = obj.get("bbox", [0, 0, 0, 0])
                        x1 = float(bbox[0]) * scale_x
                        y1 = float(bbox[1]) * scale_y
                        x2 = float(bbox[2]) * scale_x
                        y2 = float(bbox[3]) * scale_y
                        obj_list.append(
                            FrameAnnotation(
                                track_id=obj.get("id", 0),
                                bbox=[x1, y1, x2, y2],
                                class_id=obj.get("class_id", 2)
                            )
                        )
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
            if ret and frame is not None:
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

    def close(self) -> None:
        """Release underlying video capture resource."""
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None


class TUDATSequence(SequenceDataset):
    """Specialized sequence loader for TU-DAT roadside video anomalies."""

    def __init__(
        self,
        video_path: Path | str,
        annotation_path: Path | str,
        category: str = "positive",
        scenario_tag: str = "standard",
        target_resolution: Optional[Tuple[int, int]] = (640, 640)
    ) -> None:
        self.video_file = Path(video_path)
        self.annotation_file = Path(annotation_path)
        self.category = category
        self.scenario_tag = scenario_tag
        self.sequence_name = self.video_file.stem

        super().__init__(
            sequence_path=self.video_file,
            annotations_path=self.annotation_file,
            target_resolution=target_resolution
        )

    def __repr__(self) -> str:
        status = f"Accident@F={self.accident_frame}" if self.is_anomaly else "NormalTraffic"
        return (
            f"<TUDATSequence {self.sequence_name} [{self.category}/{self.scenario_tag}] "
            f"Frames={self.total_frames} FPS={self.fps:.1f} {status}>"
        )


class TUDATBenchmarkDataset:
    """Manages multi-sequence TU-DAT benchmark evaluation split."""

    def __init__(
        self,
        base_dir: Optional[Path | str] = None,
        target_resolution: Optional[Tuple[int, int]] = (640, 640)
    ) -> None:
        if base_dir is None:
            # Check default candidate directories
            candidates = [
                Path("data/tudat"),
                Path(__file__).resolve().parent / "tudat",
                Path("TU-DAT-main"),
            ]
            self.base_dir = Path("data/tudat")
            for c in candidates:
                if c.exists() and (c / "videos").exists():
                    self.base_dir = c
                    break
        else:
            self.base_dir = Path(base_dir)

        self.target_resolution = target_resolution
        self.positive_sequences: List[TUDATSequence] = []
        self.negative_sequences: List[TUDATSequence] = []
        self._sequence_index: Dict[str, TUDATSequence] = {}

        self._discover_sequences()

    def _discover_sequences(self) -> None:
        """Discover and pair video files with their corresponding JSON annotations."""
        videos_dir = self.base_dir / "videos"
        annotations_dir = self.base_dir / "annotations"

        if not videos_dir.exists() or not annotations_dir.exists():
            return

        # 1. Discover Positive Accident Videos
        pos_vids_dir = videos_dir / "Positive_Videos"
        pos_annos_dir = annotations_dir / "positive"
        if pos_vids_dir.exists() and pos_annos_dir.exists():
            for v_path in sorted(pos_vids_dir.glob("*.mov")) + sorted(pos_vids_dir.glob("*.mp4")):
                v_name = v_path.stem
                anno_path = pos_annos_dir / f"{v_name}_annotation.json"
                if anno_path.exists():
                    # Classify scenario tag based on video ID
                    tag = "standard"
                    try:
                        v_num = int("".join([c for c in v_name if c.isdigit()]))
                        if 29 <= v_num <= 41:
                            tag = "challenging_adverse_light"
                    except ValueError:
                        pass

                    seq = TUDATSequence(
                        video_path=v_path,
                        annotation_path=anno_path,
                        category="positive",
                        scenario_tag=tag,
                        target_resolution=self.target_resolution
                    )
                    self.positive_sequences.append(seq)
                    self._sequence_index[v_name] = seq

        # 2. Discover Negative Control Videos
        neg_vids_dir = videos_dir / "Negative_Videos"
        neg_annos_dir = annotations_dir / "negative"
        if neg_vids_dir.exists() and neg_annos_dir.exists():
            for v_path in sorted(neg_vids_dir.glob("*.mov")) + sorted(neg_vids_dir.glob("*.mp4")):
                v_name = v_path.stem
                anno_path = neg_annos_dir / f"{v_name}_annotation.json"
                if anno_path.exists():
                    seq = TUDATSequence(
                        video_path=v_path,
                        annotation_path=anno_path,
                        category="negative",
                        scenario_tag="negative_control",
                        target_resolution=self.target_resolution
                    )
                    self.negative_sequences.append(seq)
                    self._sequence_index[v_name] = seq

    @property
    def all_sequences(self) -> List[TUDATSequence]:
        return self.positive_sequences + self.negative_sequences

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __iter__(self) -> Iterator[TUDATSequence]:
        return iter(self.all_sequences)

    def __getitem__(self, key: Union[int, str]) -> TUDATSequence:
        if isinstance(key, int):
            return self.all_sequences[key]
        if isinstance(key, str):
            if key in self._sequence_index:
                return self._sequence_index[key]
            raise KeyError(f"Sequence '{key}' not found in TU-DAT benchmark.")
        raise TypeError(f"Invalid key type: {type(key)}")

    def summary(self) -> List[Dict[str, Any]]:
        """Return structured summary list across all loaded sequences."""
        rows = []
        for seq in self.all_sequences:
            rows.append({
                "sequence": seq.sequence_name,
                "category": seq.category,
                "scenario_tag": seq.scenario_tag,
                "total_frames": seq.total_frames,
                "fps": round(seq.fps, 2),
                "resolution": f"{seq.resolution[0]}x{seq.resolution[1]}",
                "accident_frame": seq.accident_frame,
                "is_anomaly": seq.is_anomaly,
                "annotated_frames_count": len(seq.annotations_by_frame)
            })
        return rows


def load_dataset(
    sequence_dir: Optional[Path | str] = None,
    target_resolution: Tuple[int, int] = (640, 640)
) -> Union[SequenceDataset, TUDATBenchmarkDataset]:
    """Convenience helper to load a sequence dataset or the full TU-DAT benchmark.

    If sequence_dir is None, automatically detects TU-DAT-main and returns a
    TUDATBenchmarkDataset if available, falling back to data/synthetic_sequence.
    """
    if sequence_dir is not None:
        p = Path(sequence_dir)
        if p.is_file() or (p.is_dir() and (p / "annotations.json").exists()):
            return SequenceDataset(sequence_path=p, target_resolution=target_resolution)

    # Check for TU-DAT repository in data/tudat or TU-DAT-main
    candidates = [
        Path("data/tudat"),
        Path(__file__).resolve().parent / "tudat",
        Path("TU-DAT-main"),
    ]
    for c in candidates:
        if c.exists() and (c / "videos").exists():
            bench = TUDATBenchmarkDataset(base_dir=c, target_resolution=target_resolution)
            if len(bench) > 0:
                return bench

    # Fallback to synthetic sequence
    fallback = Path("data/synthetic_sequence")
    return SequenceDataset(sequence_path=fallback, target_resolution=target_resolution)
