from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


FRAME_W = 640
FRAME_H = 360
BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}
AUTO_BACKENDS = ("dshow", "msmf", "any")
WINDOW_NAME = "CrossCamReID MVP"


@dataclass(frozen=True)
class UiButton:
    label: str
    action: str
    rect: tuple[int, int, int, int]
    primary: bool = False


@dataclass
class UiState:
    buttons: list[UiButton] = field(default_factory=list)
    pending_action: Optional[str] = None
    pending_canvas_click: Optional[tuple[int, int]] = None
    event_scroll_offset: int = 0
    max_event_scroll: int = 0


@dataclass
class Detection:
    camera_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: float
    score: float
    feature: np.ndarray
    crop: np.ndarray
    target_similarity: Optional[float] = None
    is_target_match: bool = False
    target_choice: Optional[str] = None
    target_distance: Optional[float] = None


@dataclass
class Track:
    camera_id: int
    local_id: int
    global_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    feature: np.ndarray
    last_seen: float
    velocity: tuple[float, float] = (0.0, 0.0)
    missed: int = 0
    last_similarity: Optional[float] = None
    last_target_similarity: Optional[float] = None
    last_target_choice: Optional[str] = None
    last_target_distance: Optional[float] = None
    history: Deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=24))


@dataclass
class LostIdentity:
    global_id: int
    camera_id: int
    feature: np.ndarray
    bbox: tuple[int, int, int, int]
    last_seen: float


@dataclass
class TargetProfile:
    feature: Optional[np.ndarray] = None
    samples: int = 0
    saved_path: Optional[Path] = None
    max_templates: int = 6
    templates: list[np.ndarray] = field(default_factory=list)
    saved_sample_count: int = 0
    last_sample_saved_at: float = 0.0
    last_camera_id: Optional[int] = None
    last_bbox: Optional[tuple[int, int, int, int]] = None

    @property
    def active(self) -> bool:
        return self.feature is not None

    def register_from_detection(self, detection: Detection) -> None:
        self._reset(detection.feature)
        self.note_accepted(detection)

    def register_from_crop(self, crop: np.ndarray) -> None:
        h, w = crop.shape[:2]
        self._reset(extract_feature(crop, (w, h), float(w * h)))

    def similarity(self, feature: np.ndarray) -> Optional[float]:
        if self.feature is None:
            return None
        scores = [feature_similarity(self.feature, feature)]
        scores.extend(feature_similarity(template, feature) for template in self.templates)
        return max(scores)

    def update_from_detection(self, detection: Detection, alpha: float) -> None:
        if self.feature is None or alpha <= 0:
            return
        alpha = max(0.0, min(1.0, alpha))
        self.feature = normalize_vector((1.0 - alpha) * self.feature + alpha * detection.feature)
        self.samples += 1
        self._append_template(detection.feature)

    def _reset(self, feature: np.ndarray) -> None:
        self.feature = feature.copy()
        self.samples = 1
        self.templates = [self.feature.copy()]
        self.saved_sample_count = 0
        self.last_sample_saved_at = 0.0
        self.last_camera_id = None
        self.last_bbox = None

    def _append_template(self, feature: np.ndarray) -> None:
        if self.max_templates <= 1:
            self.templates = [self.templates[0]] if self.templates else []
            return
        self.templates.append(feature.copy())
        if len(self.templates) > self.max_templates:
            self.templates = [self.templates[0], *self.templates[-(self.max_templates - 1) :]]

    def can_save_sample(
        self,
        now: float,
        similarity: Optional[float],
        min_similarity: float,
        min_interval: float,
        max_samples: int,
    ) -> bool:
        if max_samples <= 0 or self.saved_sample_count >= max_samples:
            return False
        if similarity is not None and similarity < min_similarity:
            return False
        if self.saved_sample_count > 0 and now - self.last_sample_saved_at < min_interval:
            return False
        return True

    def note_sample_saved(self, now: float) -> None:
        self.saved_sample_count += 1
        self.last_sample_saved_at = now

    def note_accepted(self, detection: Detection) -> None:
        self.note_bbox(detection.camera_id, detection.bbox)

    def note_bbox(self, camera_id: int, bbox: tuple[int, int, int, int]) -> None:
        self.last_camera_id = camera_id
        self.last_bbox = bbox

    def distance_from_last(self, detection: Detection) -> Optional[float]:
        if self.last_camera_id != detection.camera_id or self.last_bbox is None:
            return None
        last_x, last_y, last_w, last_h = self.last_bbox
        last_center = (last_x + last_w / 2.0, last_y + last_h / 2.0)
        dx = last_center[0] - detection.center[0]
        dy = last_center[1] - detection.center[1]
        return math.hypot(dx, dy)


class EventLogger:
    def __init__(self, enabled: bool, log_dir: Path) -> None:
        self.enabled = enabled
        self.csv_path: Optional[Path] = None
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        if not enabled:
            return

        log_dir.mkdir(parents=True, exist_ok=True)
        run_name = time.strftime("%Y%m%d-%H%M%S")
        self.csv_path = log_dir / f"{run_name}-events.csv"
        self._file = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "time",
                "event_type",
                "camera",
                "global_id",
                "local_id",
                "similarity",
                "target_similarity",
                "target_choice",
                "target_distance",
                "bbox",
                "message",
            ],
        )
        self._writer.writeheader()
        self._file.flush()

    def write(
        self,
        now: float,
        event_type: str,
        camera_id: int,
        global_id: int,
        message: str,
        local_id: Optional[int] = None,
        similarity: Optional[float] = None,
        target_similarity: Optional[float] = None,
        target_choice: Optional[str] = None,
        target_distance: Optional[float] = None,
        bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        if self._writer is None or self._file is None:
            return
        self._writer.writerow(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "event_type": event_type,
                "camera": camera_id + 1,
                "global_id": f"G{global_id:03d}",
                "local_id": "" if local_id is None else local_id,
                "similarity": "" if similarity is None else f"{similarity:.4f}",
                "target_similarity": "" if target_similarity is None else f"{target_similarity:.4f}",
                "target_choice": "" if target_choice is None else target_choice,
                "target_distance": "" if target_distance is None else f"{target_distance:.2f}",
                "bbox": "" if bbox is None else f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "message": message,
            }
        )
        self._file.flush()

    def write_run_config(self, now: float, message: str) -> None:
        if self._writer is None or self._file is None:
            return
        self._writer.writerow(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "event_type": "run_config",
                "camera": "",
                "global_id": "",
                "local_id": "",
                "similarity": "",
                "target_similarity": "",
                "target_choice": "",
                "target_distance": "",
                "bbox": "",
                "message": message,
            }
        )
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class VideoFileSource:
    def __init__(self, path: Path, loop: bool = False) -> None:
        self.path = path.resolve()
        self.loop = loop
        if not self.path.is_file():
            raise RuntimeError(f"离线视频不存在：{self.path}")

        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"无法打开离线视频：{self.path}")

        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if math.isfinite(fps) and fps > 0.0 else 30.0
        self.frame_count = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.capture.read()
        if ok and frame is not None:
            return True, frame
        if not self.loop:
            return False, None

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        if ok and frame is not None:
            return True, frame
        return False, None

    def release(self) -> None:
        self.capture.release()


class SyntheticCamera:
    """Small deterministic synthetic scene used for repeatable verification."""

    def __init__(self, camera_id: int, width: int = FRAME_W, height: int = FRAME_H) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_index = 0

    def read(self) -> tuple[bool, np.ndarray]:
        self.frame_index += 1
        frame = np.full((self.height, self.width, 3), (38, 41, 45), dtype=np.uint8)
        cv2.rectangle(frame, (0, self.height - 68), (self.width, self.height), (54, 58, 64), -1)
        cv2.putText(
            frame,
            f"Synthetic camera {self.camera_id}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        x, y, visible = self._object_state()
        if visible:
            self._draw_pencil(frame, x, y)
        return True, frame

    def release(self) -> None:
        return None

    def _object_state(self) -> tuple[int, int, bool]:
        t = self.frame_index
        if self.camera_id == 0:
            if t < 25 or t > 115:
                return 0, 0, False
            progress = (t - 25) / 90.0
            return int(80 + progress * 430), 185, True
        start = 140 + (self.camera_id - 1) * 115
        end = start + 90
        if t < start or t > end:
            return 0, 0, False
        progress = (t - start) / 90.0
        return int(80 + progress * 430), 185, True

    def _draw_pencil(self, frame: np.ndarray, x: int, y: int) -> None:
        length = 170
        thickness = 18
        body_color = (35, 210, 245)
        edge_color = (20, 145, 190)
        tip_color = (55, 70, 85)
        eraser_color = (165, 90, 210)

        p1 = (x, y)
        p2 = (x + length, y + 18)
        cv2.line(frame, p1, p2, edge_color, thickness + 6, cv2.LINE_AA)
        cv2.line(frame, p1, p2, body_color, thickness, cv2.LINE_AA)

        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        tip = (int(p2[0] + math.cos(angle) * 28), int(p2[1] + math.sin(angle) * 28))
        cv2.line(frame, p2, tip, tip_color, thickness, cv2.LINE_AA)
        cv2.circle(frame, p1, 12, eraser_color, -1, cv2.LINE_AA)


class MotionDetector:
    def __init__(
        self,
        camera_id: int,
        min_area: int = 900,
        warmup_frames: int = 20,
        roi: Optional[tuple[int, int, int, int]] = None,
        target_mode: str = "general",
        single_object: bool = False,
        max_area_ratio: float = 0.65,
        max_shape_ratio: float = 1.0,
        min_long_side: int = 0,
        max_short_side: int = 0,
        max_detections: int = 4,
    ) -> None:
        self.camera_id = camera_id
        self.min_area = min_area
        self.warmup_frames = warmup_frames
        self.roi = roi
        self.target_mode = target_mode
        self.single_object = single_object
        self.max_area_ratio = max_area_ratio
        self.max_shape_ratio = max_shape_ratio
        self.min_long_side = min_long_side
        self.max_short_side = max_short_side
        self.max_detections = max_detections
        self.frame_count = 0
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=30,
            detectShadows=False,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.frame_count += 1
        view, offset_x, offset_y = crop_roi(frame, self.roi)
        mask = self.subtractor.apply(view)
        if self.frame_count <= self.warmup_frames:
            return []

        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
            if bbox_area > view.shape[0] * view.shape[1] * self.max_area_ratio:
                continue
            if w < 15 or h < 15:
                continue
            long_side = max(w, h)
            short_side = min(w, h)
            shape_ratio = short_side / max(1, long_side)
            if self.min_long_side > 0 and long_side < self.min_long_side:
                continue
            if self.max_short_side > 0 and short_side > self.max_short_side:
                continue
            if self.max_shape_ratio < 1.0 and shape_ratio > self.max_shape_ratio:
                continue
            crop = view[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            feature = extract_feature(crop, (w, h), area)
            bbox = (x + offset_x, y + offset_y, w, h)
            score = detection_score(area, w, h, self.target_mode)
            detections.append(
                Detection(
                    camera_id=self.camera_id,
                    bbox=bbox,
                    center=(bbox[0] + w / 2.0, bbox[1] + h / 2.0),
                    area=area,
                    score=score,
                    feature=feature,
                    crop=crop,
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        limit = 1 if self.single_object else self.max_detections
        return detections[:limit]


class YoloDetector:
    def __init__(
        self,
        camera_id: int,
        model_path: str,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str = "",
        classes: Optional[list[int]] = None,
        roi: Optional[tuple[int, int, int, int]] = None,
        single_object: bool = False,
        max_detections: int = 20,
        max_area_ratio: float = 0.65,
        max_shape_ratio: float = 1.0,
        min_long_side: int = 0,
        max_short_side: int = 0,
        model: Optional[object] = None,
    ) -> None:
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO detector requires ultralytics. Install it with: "
                    "python -m pip install ultralytics"
                ) from exc
            model = YOLO(model_path)

        self.camera_id = camera_id
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device
        self.classes = classes
        self.roi = roi
        self.single_object = single_object
        self.max_detections = max_detections
        self.max_area_ratio = max_area_ratio
        self.max_shape_ratio = max_shape_ratio
        self.min_long_side = min_long_side
        self.max_short_side = max_short_side
        self.model = model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        view, offset_x, offset_y = crop_roi(frame, self.roi)
        results = self.model.predict(
            source=view,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            device=self.device or None,
            classes=self.classes,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        detections: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)

        for box, confidence in zip(xyxy, confs):
            x1, y1, x2, y2 = box
            x = max(0, int(round(x1)))
            y = max(0, int(round(y1)))
            w = max(1, int(round(x2 - x1)))
            h = max(1, int(round(y2 - y1)))
            x, y, w, h = clamp_roi((x, y, w, h), view.shape[1], view.shape[0])
            bbox_area = w * h
            if bbox_area > view.shape[0] * view.shape[1] * self.max_area_ratio:
                continue
            long_side = max(w, h)
            short_side = min(w, h)
            shape_ratio = short_side / max(1, long_side)
            if self.min_long_side > 0 and long_side < self.min_long_side:
                continue
            if self.max_short_side > 0 and short_side > self.max_short_side:
                continue
            if self.max_shape_ratio < 1.0 and shape_ratio > self.max_shape_ratio:
                continue
            crop = view[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            bbox = (x + offset_x, y + offset_y, w, h)
            area = float(bbox_area)
            detections.append(
                Detection(
                    camera_id=self.camera_id,
                    bbox=bbox,
                    center=(bbox[0] + w / 2.0, bbox[1] + h / 2.0),
                    area=area,
                    score=float(confidence),
                    feature=extract_feature(crop, (w, h), area),
                    crop=crop,
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        limit = 1 if self.single_object else self.max_detections
        return detections[:limit]


class RfDetrDetector:
    MODEL_CLASSES = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "base": "RFDETRBase",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
        "xlarge": "RFDETRXLarge",
        "2xlarge": "RFDETR2XLarge",
    }

    def __init__(
        self,
        camera_id: int,
        model_size: str = "nano",
        weights: str = "",
        num_classes: int = 0,
        confidence: float = 0.35,
        classes: Optional[list[int]] = None,
        class_id_mode: str = "auto",
        category_id_offset: int = 1,
        roi: Optional[tuple[int, int, int, int]] = None,
        single_object: bool = False,
        max_detections: int = 20,
        max_area_ratio: float = 0.65,
        max_shape_ratio: float = 1.0,
        min_long_side: int = 0,
        max_short_side: int = 0,
        optimize: bool = False,
        model: Optional[object] = None,
    ) -> None:
        if model_size not in self.MODEL_CLASSES:
            raise ValueError(f"Unsupported RF-DETR model size: {model_size}")

        created_model = model is None
        if created_model:
            try:
                import rfdetr
            except ImportError as exc:
                raise RuntimeError(
                    "RF-DETR detector requires rfdetr. Install it with: "
                    "python -m pip install -r requirements-rfdetr.txt"
                ) from exc

            model_class = getattr(rfdetr, self.MODEL_CLASSES[model_size], None)
            if model_class is None:
                raise RuntimeError(f"Installed rfdetr does not provide {self.MODEL_CLASSES[model_size]}.")
            model_kwargs = {"num_classes": num_classes} if num_classes > 0 else {}
            if weights:
                model_kwargs["pretrain_weights"] = weights
            model = model_class(**model_kwargs)

        self.camera_id = camera_id
        self.model_size = model_size
        self.weights = weights
        self.num_classes = num_classes
        self.confidence = confidence
        self.classes = set(classes) if classes else None
        self.class_id_mode = class_id_mode
        self.category_id_offset = category_id_offset
        self.roi = roi
        self.single_object = single_object
        self.max_detections = max_detections
        self.max_area_ratio = max_area_ratio
        self.max_shape_ratio = max_shape_ratio
        self.min_long_side = min_long_side
        self.max_short_side = max_short_side

        self.model = model
        if created_model and optimize and hasattr(self.model, "optimize_for_inference"):
            optimized = self.model.optimize_for_inference()
            if optimized is not None:
                self.model = optimized

    @staticmethod
    def _map_class_id(raw_class_id: int, mode: str, category_id_offset: int) -> int:
        if mode == "zero":
            return raw_class_id
        if mode == "category":
            return raw_class_id - category_id_offset
        if raw_class_id >= category_id_offset:
            return raw_class_id - category_id_offset
        return raw_class_id

    @staticmethod
    def _prediction_arrays(predictions) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        xyxy = np.asarray(getattr(predictions, "xyxy", []), dtype=np.float32)
        if xyxy.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), None
        if xyxy.ndim == 1:
            if xyxy.size % 4 != 0:
                raise RuntimeError(f"RF-DETR xyxy 输出长度不是 4 的倍数：{xyxy.size}")
            xyxy = xyxy.reshape(-1, 4)
        elif xyxy.ndim == 2:
            if xyxy.shape[1] < 4:
                raise RuntimeError(f"RF-DETR xyxy 输出列数不足 4：{xyxy.shape[1]}")
            if xyxy.shape[1] > 4:
                xyxy = xyxy[:, :4]
        else:
            raise RuntimeError(f"RF-DETR xyxy 输出维度异常：{xyxy.ndim}")

        box_count = len(xyxy)
        confidences = getattr(predictions, "confidence", None)
        if confidences is None:
            confidences_array = np.ones(box_count, dtype=np.float32)
        else:
            confidences_array = np.asarray(confidences, dtype=np.float32).reshape(-1)
            if len(confidences_array) < box_count:
                confidences_array = np.pad(confidences_array, (0, box_count - len(confidences_array)), constant_values=1.0)

        class_ids = getattr(predictions, "class_id", None)
        if class_ids is None:
            class_id_array = None
        else:
            class_id_array = np.asarray(class_ids, dtype=np.int32).reshape(-1)
            if len(class_id_array) < box_count:
                class_id_array = np.pad(class_id_array, (0, box_count - len(class_id_array)), constant_values=0)
        return xyxy, confidences_array[:box_count], None if class_id_array is None else class_id_array[:box_count]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        view, offset_x, offset_y = crop_roi(frame, self.roi)
        rgb_view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        predictions = self.model.predict(rgb_view, threshold=self.confidence)

        xyxy, confidences_array, class_id_array = self._prediction_arrays(predictions)
        if len(xyxy) == 0:
            return []

        detections: list[Detection] = []
        for index, (box, confidence) in enumerate(zip(xyxy, confidences_array)):
            raw_class_id = 0 if class_id_array is None else int(class_id_array[index])
            class_id = self._map_class_id(raw_class_id, self.class_id_mode, self.category_id_offset)
            if class_id < 0:
                continue
            if self.classes is not None and class_id not in self.classes:
                continue

            x1, y1, x2, y2 = box
            x = max(0, int(round(x1)))
            y = max(0, int(round(y1)))
            w = max(1, int(round(x2 - x1)))
            h = max(1, int(round(y2 - y1)))
            x, y, w, h = clamp_roi((x, y, w, h), view.shape[1], view.shape[0])
            bbox_area = w * h
            if bbox_area > view.shape[0] * view.shape[1] * self.max_area_ratio:
                continue
            long_side = max(w, h)
            short_side = min(w, h)
            shape_ratio = short_side / max(1, long_side)
            if self.min_long_side > 0 and long_side < self.min_long_side:
                continue
            if self.max_short_side > 0 and short_side > self.max_short_side:
                continue
            if self.max_shape_ratio < 1.0 and shape_ratio > self.max_shape_ratio:
                continue
            crop = view[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            bbox = (x + offset_x, y + offset_y, w, h)
            area = float(bbox_area)
            detections.append(
                Detection(
                    camera_id=self.camera_id,
                    bbox=bbox,
                    center=(bbox[0] + w / 2.0, bbox[1] + h / 2.0),
                    area=area,
                    score=float(confidence),
                    feature=extract_feature(crop, (w, h), area),
                    crop=crop,
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        limit = 1 if self.single_object else self.max_detections
        return detections[:limit]


class HybridDetector:
    def __init__(self, primary: YoloDetector, fallback: RfDetrDetector, fallback_interval: int = 15) -> None:
        if primary.camera_id != fallback.camera_id:
            raise ValueError("Hybrid detector camera ids must match.")
        if fallback_interval <= 0:
            raise ValueError("Hybrid fallback interval must be greater than 0.")
        self.camera_id = primary.camera_id
        self.primary = primary
        self.fallback = fallback
        self.fallback_interval = fallback_interval
        self.frame_index = 0
        self.last_fallback_frame = -fallback_interval
        self.fallback_calls = 0
        self.fallback_candidates = 0
        self.fallback_accepted = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.frame_index += 1
        return self.primary.detect(frame)

    def fallback_due(self) -> bool:
        return self.frame_index - self.last_fallback_frame >= self.fallback_interval

    def detect_fallback(self, frame: np.ndarray) -> list[Detection]:
        if not self.fallback_due():
            return []
        self.last_fallback_frame = self.frame_index
        self.fallback_calls += 1
        detections = self.fallback.detect(frame)
        self.fallback_candidates += len(detections)
        return detections

    def note_fallback_accepted(self) -> None:
        self.fallback_accepted += 1


def hybrid_detector_stats(detectors: list[object]) -> dict[str, int]:
    hybrid_detectors = [detector for detector in detectors if isinstance(detector, HybridDetector)]
    return {
        "calls": sum(detector.fallback_calls for detector in hybrid_detectors),
        "candidates": sum(detector.fallback_candidates for detector in hybrid_detectors),
        "accepted": sum(detector.fallback_accepted for detector in hybrid_detectors),
    }


class CrossCameraTracker:
    def __init__(
        self,
        camera_count: int = 2,
        match_distance: float = 115.0,
        max_missed: int = 14,
        lost_ttl: float = 8.0,
        cross_threshold: float = 0.72,
        prediction_horizon: float = 0.35,
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        self.match_distance = match_distance
        self.max_missed = max_missed
        self.lost_ttl = lost_ttl
        self.cross_threshold = cross_threshold
        self.prediction_horizon = max(0.0, prediction_horizon)
        self.camera_count = camera_count
        self.next_local_id = [1 for _ in range(camera_count)]
        self.next_global_id = 1
        self.active: dict[int, list[Track]] = {camera_id: [] for camera_id in range(camera_count)}
        self.lost: list[LostIdentity] = []
        self.events: Deque[str] = deque(maxlen=80)
        self.event_logger = event_logger
        self.registered_target_id: Optional[int] = None
        self.pending_initial_target_camera: Optional[int] = None
        self.global_seen_cameras: dict[int, set[int]] = {}
        self.cross_camera_match_observed = False

    def activate_registered_target(self, target_global_id: int = 1) -> None:
        self.registered_target_id = target_global_id
        self.pending_initial_target_camera = None
        self.next_global_id = max(self.next_global_id, target_global_id + 1)
        self.next_local_id = [1 for _ in range(self.camera_count)]
        self.active = {camera_id: [] for camera_id in range(self.camera_count)}
        self.lost = []
        self.global_seen_cameras = {}
        self.cross_camera_match_observed = False

    def note_target_registered(
        self,
        now: float,
        camera_id: int,
        bbox: tuple[int, int, int, int],
        saved_path: Optional[Path],
    ) -> None:
        target_id = self.registered_target_id if self.registered_target_id is not None else 1
        self.pending_initial_target_camera = camera_id
        message = f"摄像头{camera_id + 1}：已注册目标为 G{target_id:03d}"
        if saved_path is not None:
            message += f" ({saved_path})"
        self._log(
            now,
            "target_registered",
            camera_id,
            target_id,
            message,
            similarity=1.0,
            target_similarity=1.0,
            target_choice="register",
            target_distance=0.0,
            bbox=bbox,
        )
        self._note_seen(target_id, camera_id)

    def update(self, camera_id: int, detections: list[Detection], now: float) -> list[Track]:
        tracks = self.active[camera_id]
        self._gate_registered_target_claims(detections, now)
        assigned_detection_indexes: set[int] = set()
        assigned_track_indexes: set[int] = set()

        pairs: list[tuple[float, int, int, float]] = []
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                if self.registered_target_id is not None:
                    track_is_target = track.global_id == self.registered_target_id
                    if track_is_target and not detection.is_target_match:
                        continue
                    if not track_is_target and detection.is_target_match:
                        continue
                match = track_detection_match(track, detection, self.match_distance, now, self.prediction_horizon)
                if match is not None:
                    cost, similarity = match
                    pairs.append((cost, track_index, detection_index, similarity))

        for _, track_index, detection_index, similarity in sorted(pairs, key=lambda item: item[0]):
            if track_index in assigned_track_indexes or detection_index in assigned_detection_indexes:
                continue
            track = tracks[track_index]
            detection = detections[detection_index]
            self._refresh_track(track, detection, now, similarity)
            assigned_track_indexes.add(track_index)
            assigned_detection_indexes.add(detection_index)

        for track_index, track in enumerate(list(tracks)):
            if track_index in assigned_track_indexes:
                continue
            track.missed += 1
            if track.missed > self.max_missed:
                self._move_to_lost(track, now)
                tracks.remove(track)

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detection_indexes:
                continue
            global_id, similarity, event_type = self._resolve_global_id(detection, now)
            if global_id is None:
                global_id = self.next_global_id
                self.next_global_id += 1
                event_type = "new"
                self._log(
                    now,
                    event_type,
                    camera_id,
                    global_id,
                    f"摄像头{camera_id + 1}：发现新目标 G{global_id:03d}",
                    similarity=similarity,
                    target_similarity=detection.target_similarity,
                    target_choice=detection.target_choice,
                    target_distance=detection.target_distance,
                    bbox=detection.bbox,
                )
            else:
                self._log(
                    now,
                    event_type,
                    camera_id,
                    global_id,
                    self._matched_message(
                        camera_id,
                        global_id,
                        event_type,
                        similarity,
                        detection.target_similarity,
                        detection.target_choice,
                        detection.target_distance,
                    ),
                    similarity=similarity,
                    target_similarity=detection.target_similarity,
                    target_choice=detection.target_choice,
                    target_distance=detection.target_distance,
                    bbox=detection.bbox,
                )

            local_id = self.next_local_id[camera_id]
            self.next_local_id[camera_id] += 1
            track = Track(
                camera_id=camera_id,
                local_id=local_id,
                global_id=global_id,
                bbox=detection.bbox,
                center=detection.center,
                feature=detection.feature,
                last_seen=now,
                last_similarity=similarity,
                last_target_similarity=detection.target_similarity,
                last_target_choice=detection.target_choice,
                last_target_distance=detection.target_distance,
            )
            if self.event_logger is not None:
                self.event_logger.write(
                    now,
                    "track_created",
                    camera_id,
                    global_id,
                    f"摄像头{camera_id + 1}：轨迹 L{local_id} 使用 G{global_id:03d}",
                    local_id=local_id,
                    similarity=similarity,
                    target_similarity=detection.target_similarity,
                    target_choice=detection.target_choice,
                    target_distance=detection.target_distance,
                    bbox=detection.bbox,
                )
            track.history.append((int(detection.center[0]), int(detection.center[1])))
            tracks.append(track)
            self._note_seen(global_id, camera_id)

        self._expire_lost(now)
        return list(tracks)

    def _resolve_global_id(
        self,
        detection: Detection,
        now: float,
    ) -> tuple[Optional[int], Optional[float], str]:
        target_similarity = self._resolve_registered_target(detection, now)
        if target_similarity is not None:
            return self.registered_target_id, target_similarity, "target_matched"

        global_id, similarity = self._match_lost_identity(detection, now)
        if global_id is not None:
            return global_id, similarity, "matched"
        return None, None, "new"

    def _gate_registered_target_claims(self, detections: list[Detection], now: float) -> None:
        if self.registered_target_id is None:
            return
        for detection in detections:
            if detection.is_target_match and not self._can_claim_registered_target(detection, now):
                detection.is_target_match = False

    def _can_claim_registered_target(self, detection: Detection, now: float) -> bool:
        if self.registered_target_id is None:
            return False
        if self.pending_initial_target_camera == detection.camera_id:
            return True
        if self._active_registered_target_camera() == detection.camera_id:
            return True
        return self._find_lost_registered_target(detection, now) is not None

    def _resolve_registered_target(self, detection: Detection, now: float) -> Optional[float]:
        if self.registered_target_id is None or not detection.is_target_match:
            return None
        if self.pending_initial_target_camera == detection.camera_id:
            self.pending_initial_target_camera = None
            return detection.target_similarity

        lost_item, lost_similarity = self._find_lost_registered_target(detection, now) or (None, None)
        if lost_item is not None:
            self.lost.remove(lost_item)
            return detection.target_similarity if detection.target_similarity is not None else lost_similarity
        return None

    def _active_registered_target_camera(self) -> Optional[int]:
        if self.registered_target_id is None:
            return None
        for camera_tracks in self.active.values():
            for track in camera_tracks:
                if track.global_id == self.registered_target_id:
                    return track.camera_id
        return None

    def active_registered_target_camera(self) -> Optional[int]:
        return self._active_registered_target_camera()

    def _find_lost_registered_target(
        self,
        detection: Detection,
        now: float,
    ) -> Optional[tuple[LostIdentity, float]]:
        if self.registered_target_id is None:
            return None
        best_item: Optional[LostIdentity] = None
        best_score = 0.0
        for item in self.lost:
            if item.global_id != self.registered_target_id:
                continue
            if item.camera_id == detection.camera_id:
                continue
            if now - item.last_seen > self.lost_ttl:
                continue
            score = feature_similarity(item.feature, detection.feature)
            if score > best_score:
                best_item = item
                best_score = score
        if best_item is not None and best_score >= self.cross_threshold:
            return best_item, best_score
        return None

    def _matched_message(
        self,
        camera_id: int,
        global_id: int,
        event_type: str,
        similarity: Optional[float],
        target_similarity: Optional[float],
        target_choice: Optional[str],
        target_distance: Optional[float],
    ) -> str:
        if event_type == "target_matched":
            sim_text = "" if target_similarity is None else f"，目标相似度={target_similarity:.2f}"
            choice_text = "" if target_choice is None else f"，选择={target_choice}"
            distance_text = "" if target_distance is None else f"，距上次={target_distance:.0f}px"
            return f"摄像头{camera_id + 1}：匹配到目标 G{global_id:03d}{sim_text}{choice_text}{distance_text}"
        sim_text = "" if similarity is None else f"，相似度={similarity:.2f}"
        return f"摄像头{camera_id + 1}：匹配到 G{global_id:03d}{sim_text}"

    def _note_seen(self, global_id: int, camera_id: int) -> None:
        cameras = self.global_seen_cameras.setdefault(global_id, set())
        cameras.add(camera_id)
        if len(cameras) >= 2:
            self.cross_camera_match_observed = True

    def _refresh_track(
        self,
        track: Track,
        detection: Detection,
        now: float,
        similarity: float,
    ) -> None:
        dt = max(1.0 / 60.0, min(now - track.last_seen, 1.0))
        observed_velocity = (
            (detection.center[0] - track.center[0]) / dt,
            (detection.center[1] - track.center[1]) / dt,
        )
        track.velocity = (
            0.65 * track.velocity[0] + 0.35 * observed_velocity[0],
            0.65 * track.velocity[1] + 0.35 * observed_velocity[1],
        )
        track.bbox = detection.bbox
        track.center = detection.center
        track.feature = normalize_vector(0.82 * track.feature + 0.18 * detection.feature)
        track.last_seen = now
        track.missed = 0
        track.last_similarity = similarity
        track.last_target_similarity = detection.target_similarity
        track.last_target_choice = detection.target_choice
        track.last_target_distance = detection.target_distance
        track.history.append((int(detection.center[0]), int(detection.center[1])))
        if (
            self.event_logger is not None
            and self.registered_target_id is not None
            and track.global_id == self.registered_target_id
            and detection.is_target_match
        ):
            self.event_logger.write(
                now,
                "target_refreshed",
                track.camera_id,
                track.global_id,
                f"摄像头{track.camera_id + 1}：G{track.global_id:03d} 持续锁定",
                local_id=track.local_id,
                similarity=similarity,
                target_similarity=detection.target_similarity,
                target_choice=detection.target_choice,
                target_distance=detection.target_distance,
                bbox=detection.bbox,
            )

    def _move_to_lost(self, track: Track, now: float) -> None:
        self.lost.append(
            LostIdentity(
                global_id=track.global_id,
                camera_id=track.camera_id,
                feature=track.feature.copy(),
                bbox=track.bbox,
                last_seen=track.last_seen,
            )
        )
        self._log(
            now,
            "left",
            track.camera_id,
            track.global_id,
            f"摄像头{track.camera_id + 1}：G{track.global_id:03d} 离开画面",
            local_id=track.local_id,
            bbox=track.bbox,
        )

    def _match_lost_identity(
        self,
        detection: Detection,
        now: float,
    ) -> tuple[Optional[int], Optional[float]]:
        best_item: Optional[LostIdentity] = None
        best_score = 0.0
        for item in self.lost:
            if item.camera_id == detection.camera_id:
                continue
            if now - item.last_seen > self.lost_ttl:
                continue
            score = feature_similarity(item.feature, detection.feature)
            if score > best_score:
                best_item = item
                best_score = score

        if best_item is not None and best_score >= self.cross_threshold:
            self.lost.remove(best_item)
            return best_item.global_id, best_score
        return None, None

    def _expire_lost(self, now: float) -> None:
        self.lost = [item for item in self.lost if now - item.last_seen <= self.lost_ttl]

    def _log(
        self,
        now: float,
        event_type: str,
        camera_id: int,
        global_id: int,
        message: str,
        local_id: Optional[int] = None,
        similarity: Optional[float] = None,
        target_similarity: Optional[float] = None,
        target_choice: Optional[str] = None,
        target_distance: Optional[float] = None,
        bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        self.events.appendleft(f"{time.strftime('%H:%M:%S', time.localtime(now))} {message}")
        if self.event_logger is not None:
            self.event_logger.write(
                now,
                event_type,
                camera_id,
                global_id,
                message,
                local_id=local_id,
                similarity=similarity,
                target_similarity=target_similarity,
                target_choice=target_choice,
                target_distance=target_distance,
                bbox=bbox,
            )


def extract_feature(crop: np.ndarray, size: tuple[int, int], area: float) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    hist = normalize_vector(cv2.normalize(hist, hist).flatten())

    resized_hsv = cv2.resize(hsv, (48, 48), interpolation=cv2.INTER_AREA)
    layout_parts = []
    grid = 3
    cell_h = resized_hsv.shape[0] // grid
    cell_w = resized_hsv.shape[1] // grid
    for gy in range(grid):
        for gx in range(grid):
            cell = resized_hsv[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            layout_parts.append(np.mean(cell.reshape(-1, 3), axis=0) / np.array([180.0, 255.0, 255.0]))
    color_layout = normalize_vector(np.concatenate(layout_parts).astype(np.float32))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    edge_hist, _ = np.histogram(angle, bins=8, range=(0, 180), weights=magnitude)
    edge_hist = normalize_vector(edge_hist.astype(np.float32))
    texture_hist = local_binary_pattern_histogram(gray)

    w, h = size
    aspect = min(w, h) / max(w, h)
    fill_ratio = min(1.0, area / max(1.0, w * h))
    size_hint = np.array(
        [
            min(1.0, max(w, h) / 240.0),
            min(1.0, min(w, h) / 120.0),
        ],
        dtype=np.float32,
    )
    mean_color = np.mean(hsv.reshape(-1, 3), axis=0) / np.array([180.0, 255.0, 255.0])

    feature = np.concatenate(
        [
            hist.astype(np.float32) * 0.50,
            color_layout.astype(np.float32) * 0.30,
            edge_hist.astype(np.float32) * 0.25,
            texture_hist.astype(np.float32) * 0.22,
            np.array([aspect, fill_ratio], dtype=np.float32) * 0.18,
            size_hint * 0.12,
            mean_color.astype(np.float32) * 0.15,
        ]
    )
    return normalize_vector(feature).astype(np.float32)


def local_binary_pattern_histogram(gray: np.ndarray) -> np.ndarray:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros(256, dtype=np.float32)

    center = gray[1:-1, 1:-1]
    code = np.zeros(center.shape, dtype=np.uint8)
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)

    hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    return normalize_vector(hist)


def normalize_vector(feature: np.ndarray) -> np.ndarray:
    feature = feature.astype(np.float32, copy=False)
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm
    return feature


def feature_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    cosine = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, cosine))


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    inter_w = max(0, right - left)
    inter_h = max(0, bottom - top)
    intersection = float(inter_w * inter_h)
    union = float(aw * ah + bw * bh) - intersection
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


def track_detection_match(
    track: Track,
    detection: Detection,
    match_distance: float,
    now: float,
    prediction_horizon: float,
) -> Optional[tuple[float, float]]:
    predicted_center = predict_track_center(track, now, prediction_horizon)
    distance = euclidean(predicted_center, detection.center)
    iou = bbox_iou(track.bbox, detection.bbox)
    similarity = feature_similarity(track.feature, detection.feature)

    if distance > match_distance and iou < 0.03:
        return None
    if similarity < 0.20 and iou < 0.12:
        return None

    distance_cost = min(distance / max(1.0, match_distance), 1.5)
    cost = 0.55 * distance_cost + 0.30 * (1.0 - iou) + 0.15 * (1.0 - similarity)
    return cost, similarity


def predict_track_center(track: Track, now: float, horizon: float) -> tuple[float, float]:
    age = min(max(0.0, now - track.last_seen), max(0.0, horizon))
    return (
        track.center[0] + track.velocity[0] * age,
        track.center[1] + track.velocity[1] * age,
    )


def detection_score(area: float, width: int, height: int, target_mode: str) -> float:
    long_side = max(width, height)
    short_side = max(1, min(width, height))
    elongation = long_side / short_side
    fill_ratio = area / max(1.0, width * height)

    if target_mode == "pencil":
        # A pencil-like motion blob is usually long, narrow, and not a huge full-body region.
        elongation_bonus = min(elongation, 8.0)
        compactness_penalty = max(0.25, min(1.0, fill_ratio))
        return area * elongation_bonus * compactness_penalty

    return area


def apply_target_profile(
    detections: list[Detection],
    target_profile: TargetProfile,
    threshold: float,
    update_alpha: float,
    keep_all: bool = False,
    stick_distance: float = 120.0,
    switch_margin: float = 0.08,
    log_dir: Optional[Path] = None,
    now: Optional[float] = None,
    sample_min_similarity: float = 0.72,
    sample_min_interval: float = 0.8,
    sample_max_count: int = 12,
    can_claim_target: Optional[Callable[[Detection], bool]] = None,
) -> list[Detection]:
    if not target_profile.active:
        return detections

    accepted: list[Detection] = []
    for detection in detections:
        detection.target_similarity = target_profile.similarity(detection.feature)
        detection.is_target_match = detection.target_similarity is not None and detection.target_similarity >= threshold
        if detection.is_target_match:
            accepted.append(detection)

    if not accepted:
        return detections if keep_all else []

    best = select_stable_target_match(accepted, target_profile, stick_distance, switch_margin)
    if can_claim_target is not None and not can_claim_target(best):
        for detection in detections:
            detection.is_target_match = False
            if detection is best:
                detection.target_choice = "blocked"
            else:
                detection.target_choice = None
                detection.target_distance = None
        return detections if keep_all else []

    for detection in detections:
        detection.is_target_match = detection is best
        if detection is not best:
            detection.target_choice = None
            detection.target_distance = None
    if not keep_all:
        target_profile.update_from_detection(best, update_alpha)
    target_profile.note_accepted(best)
    if (
        log_dir is not None
        and now is not None
        and target_profile.can_save_sample(
            now,
            best.target_similarity,
            sample_min_similarity,
            sample_min_interval,
            sample_max_count,
        )
    ):
        saved_path = save_target_sample(best.crop, log_dir, best.camera_id, now, "match", best.target_similarity)
        if saved_path is not None:
            target_profile.note_sample_saved(now)
    if keep_all:
        return detections
    return [best]


def hybrid_fallback_camera_order(
    detectors: list[object],
    target_profile: TargetProfile,
    tracker: CrossCameraTracker,
) -> list[int]:
    camera_ids = [camera_id for camera_id, detector in enumerate(detectors) if isinstance(detector, HybridDetector)]
    active_camera = tracker.active_registered_target_camera()
    if active_camera in camera_ids:
        return [active_camera]

    last_camera = target_profile.last_camera_id
    if last_camera not in camera_ids:
        return camera_ids
    return [camera_id for camera_id in camera_ids if camera_id != last_camera] + [last_camera]


def apply_target_profile_with_hybrid_fallback(
    raw_detections_by_camera: list[list[Detection]],
    frames: list[np.ndarray],
    detectors: list[object],
    target_profile: TargetProfile,
    tracker: CrossCameraTracker,
    args: argparse.Namespace,
    log_dir: Path,
    now: float,
) -> list[list[Detection]]:
    def apply_profile(detections: list[Detection], keep_all: bool) -> list[Detection]:
        return apply_target_profile(
            detections,
            target_profile,
            args.target_threshold,
            args.target_update_alpha,
            keep_all,
            stick_distance=args.target_stick_distance,
            switch_margin=args.target_switch_margin,
            log_dir=log_dir,
            now=now,
            sample_min_similarity=args.target_sample_min_similarity,
            sample_min_interval=args.target_sample_min_interval,
            sample_max_count=args.target_sample_max_count,
            can_claim_target=lambda detection: tracker._can_claim_registered_target(detection, now),
        )

    filtered_by_camera = [
        apply_profile(detections, args.track_all_after_register)
        for detections in raw_detections_by_camera
    ]
    if not target_profile.active:
        return filtered_by_camera
    if any(detection.is_target_match for detections in filtered_by_camera for detection in detections):
        return filtered_by_camera

    for camera_id in hybrid_fallback_camera_order(detectors, target_profile, tracker):
        detector = detectors[camera_id]
        if not isinstance(detector, HybridDetector):
            continue
        calls_before = detector.fallback_calls
        fallback_detections = detector.detect_fallback(frames[camera_id])
        if detector.fallback_calls == calls_before:
            continue

        if tracker.active_registered_target_camera() == camera_id:
            fallback_detections = [
                detection
                for detection in fallback_detections
                if (distance := target_profile.distance_from_last(detection)) is not None
                and distance <= args.target_stick_distance
            ]

        accepted = apply_profile(fallback_detections, False)
        if accepted:
            detector.note_fallback_accepted()
            if args.track_all_after_register:
                filtered_by_camera[camera_id].extend(accepted)
            else:
                filtered_by_camera[camera_id] = accepted
            detection = accepted[0]
            global_id = tracker.registered_target_id or 1
            message = f"摄像头{camera_id + 1}：YOLO 未匹配目标，RF-DETR 补检恢复 G{global_id:03d}"
            tracker.events.appendleft(f"{time.strftime('%H:%M:%S')} {message}")
            if tracker.event_logger is not None:
                tracker.event_logger.write(
                    now,
                    "hybrid_fallback",
                    camera_id,
                    global_id,
                    message,
                    target_similarity=detection.target_similarity,
                    target_choice=detection.target_choice,
                    target_distance=detection.target_distance,
                    bbox=detection.bbox,
                )
        break
    return filtered_by_camera


def select_stable_target_match(
    accepted: list[Detection],
    target_profile: TargetProfile,
    stick_distance: float,
    switch_margin: float,
) -> Detection:
    accepted.sort(key=lambda item: (item.target_similarity or 0.0, item.score), reverse=True)
    for detection in accepted:
        detection.target_distance = target_profile.distance_from_last(detection)
    best = accepted[0]
    if stick_distance <= 0 or target_profile.last_bbox is None:
        best.target_choice = "best"
        return best

    nearby = []
    for detection in accepted:
        distance = detection.target_distance
        if distance is not None and distance <= stick_distance:
            nearby.append(detection)
    if not nearby:
        best.target_choice = "best"
        return best

    sticky = min(
        nearby,
        key=lambda item: (
            (item.target_distance or 0.0) / max(1.0, stick_distance),
            -(item.target_similarity or 0.0),
            -item.score,
        ),
    )
    if sticky is best:
        sticky.target_choice = "near"
        return sticky

    best_similarity = best.target_similarity or 0.0
    sticky_similarity = sticky.target_similarity or 0.0
    if sticky_similarity + max(0.0, switch_margin) >= best_similarity:
        sticky.target_choice = "sticky"
        return sticky
    best.target_choice = "switch"
    return best


def save_target_sample(
    crop: np.ndarray,
    log_dir: Path,
    camera_id: int,
    now: float,
    source: str,
    target_similarity: Optional[float],
) -> Optional[Path]:
    target_dir = log_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    millis = int((now - int(now)) * 1000)
    path = target_dir / f"{timestamp}-{millis:03d}-cam{camera_id + 1}-{source}.jpg"
    ok = cv2.imwrite(str(path), crop)
    if not ok:
        return None
    append_target_sample_index(target_dir / "target_samples.csv", now, camera_id, source, target_similarity, path)
    return path


def append_target_sample_index(
    csv_path: Path,
    now: float,
    camera_id: int,
    source: str,
    target_similarity: Optional[float],
    image_path: Path,
) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time", "camera", "source", "target_similarity", "image"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "camera": camera_id + 1,
                "source": source,
                "target_similarity": "" if target_similarity is None else f"{target_similarity:.4f}",
                "image": str(image_path),
            }
        )


def register_target(
    target_profile: TargetProfile,
    tracker: CrossCameraTracker,
    event_logger: EventLogger,
    crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    camera_id: int,
    log_dir: Path,
    now: float,
) -> None:
    target_profile.register_from_crop(crop)
    target_profile.note_bbox(camera_id, bbox)
    target_profile.saved_path = save_target_sample(crop, log_dir, camera_id, now, "register", 1.0)
    if target_profile.saved_path is not None:
        target_profile.note_sample_saved(now)
    tracker.activate_registered_target(1)
    tracker.note_target_registered(now, camera_id, bbox, target_profile.saved_path)


def register_target_from_detection(
    target_profile: TargetProfile,
    tracker: CrossCameraTracker,
    event_logger: EventLogger,
    detection: Detection,
    log_dir: Path,
    now: float,
) -> None:
    target_profile.register_from_detection(detection)
    target_profile.saved_path = save_target_sample(
        detection.crop,
        log_dir,
        detection.camera_id,
        now,
        "register",
        1.0,
    )
    if target_profile.saved_path is not None:
        target_profile.note_sample_saved(now)
    tracker.activate_registered_target(1)
    tracker.note_target_registered(now, detection.camera_id, detection.bbox, target_profile.saved_path)


def register_best_detection(
    target_profile: TargetProfile,
    tracker: CrossCameraTracker,
    event_logger: EventLogger,
    detections: list[Detection],
    camera_id: int,
    log_dir: Path,
    now: float,
) -> bool:
    if not detections:
        tracker.events.appendleft(
            f"{time.strftime('%H:%M:%S', time.localtime(now))} "
            f"摄像头{camera_id + 1}：没有可注册的运动目标"
        )
        return False
    register_target_from_detection(target_profile, tracker, event_logger, detections[0], log_dir, now)
    return True


def displayed_camera_index(view_order: list[int] | tuple[int, ...], side: str) -> int:
    return view_order[0] if side == "left" else view_order[1]


def view_order_label(view_order: list[int] | tuple[int, ...]) -> str:
    return "".join(chr(ord("A") + camera_id) for camera_id in view_order)


def action_from_key(key: int) -> Optional[str]:
    if key in (27, ord("q")):
        return "quit"
    key_actions = {
        ord("r"): "register_left",
        ord("1"): "register_left",
        ord("t"): "register_right",
        ord("2"): "register_right",
        ord("3"): "register_third",
        ord("m"): "manual_left",
        ord("4"): "manual_left",
        ord("n"): "manual_right",
        ord("5"): "manual_right",
        ord("6"): "manual_third",
        ord("7"): "flip_third",
    }
    return key_actions.get(key)


def select_target_from_frame(
    frame: np.ndarray,
    camera_id: int,
) -> Optional[tuple[np.ndarray, tuple[int, int, int, int]]]:
    window_name = f"Select target - Camera {camera_id + 1}"
    x, y, w, h = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
    if w <= 0 or h <= 0:
        return None
    bbox = clamp_roi((int(x), int(y), int(w), int(h)), frame.shape[1], frame.shape[0])
    cx, cy, cw, ch = bbox
    crop = frame[cy : cy + ch, cx : cx + cw]
    if crop.size == 0:
        return None
    return crop, bbox


def crop_roi(
    frame: np.ndarray,
    roi: Optional[tuple[int, int, int, int]],
) -> tuple[np.ndarray, int, int]:
    if roi is None:
        return frame, 0, 0

    x, y, w, h = clamp_roi(roi, frame.shape[1], frame.shape[0])
    return frame[y : y + h, x : x + w], x, y


def clamp_roi(
    roi: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(1, min(w, frame_width - x))
    h = max(1, min(h, frame_height - y))
    return x, y, w, h


def trail_direction_arrow(
    points: Iterable[tuple[int, int]],
    min_displacement: float = 6.0,
    arrow_length: float = 30.0,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    recent_points = list(points)[-8:]
    if len(recent_points) < 2:
        return None

    current = recent_points[-1]
    threshold = max(1.0, float(min_displacement))
    for previous in reversed(recent_points[:-1]):
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        distance = math.hypot(dx, dy)
        if distance < threshold:
            continue
        scale = max(8.0, float(arrow_length)) / distance
        arrow_tip = (
            int(round(current[0] + dx * scale)),
            int(round(current[1] + dy * scale)),
        )
        return current, arrow_tip
    return None


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    camera_id: int,
    roi: Optional[tuple[int, int, int, int]] = None,
    show_trails: bool = False,
) -> np.ndarray:
    output = frame.copy()
    if roi is not None:
        x, y, w, h = clamp_roi(roi, output.shape[1], output.shape[0])
        cv2.rectangle(output, (x, y), (x + w, y + h), (120, 120, 120), 1)
        cv2.putText(
            output,
            "ROI",
            (x + 8, max(22, y + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"Camera {camera_id + 1}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    for track in tracks:
        x, y, w, h = track.bbox
        color = id_color(track.global_id)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        label = f"G{track.global_id:03d} L{track.local_id}"
        if track.last_similarity is not None:
            label += f" s={track.last_similarity:.2f}"
        if track.last_target_similarity is not None:
            label += f" t={track.last_target_similarity:.2f}"
        cv2.putText(
            output,
            label,
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        if show_trails:
            points = list(track.history)
            for i in range(1, len(points)):
                cv2.line(output, points[i - 1], points[i], color, 2, cv2.LINE_AA)
            direction_arrow = trail_direction_arrow(points)
            if direction_arrow is not None:
                cv2.arrowedLine(
                    output,
                    direction_arrow[0],
                    direction_arrow[1],
                    color,
                    3,
                    cv2.LINE_AA,
                    tipLength=0.32,
                )
    return output


def pad_canvas_width(frame: np.ndarray, min_width: int) -> np.ndarray:
    if frame.shape[1] >= min_width:
        return frame
    pad_width = min_width - frame.shape[1]
    padding = np.full((frame.shape[0], pad_width, 3), (18, 20, 24), dtype=np.uint8)
    return np.hstack([frame, padding])


_FONT_CACHE: dict[int, object] = {}


def get_ui_font(size: int):
    if ImageFont is None:
        return None
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            _FONT_CACHE[size] = ImageFont.truetype(str(path), size)
            return _FONT_CACHE[size]
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 18,
) -> None:
    if Image is None or ImageDraw is None:
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return

    font = get_ui_font(size)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    drawer = ImageDraw.Draw(pil_image)
    drawer.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_button(panel: np.ndarray, button: UiButton) -> None:
    x, y, w, h = button.rect
    fill = (48, 112, 180) if button.primary else (58, 64, 72)
    border = (110, 175, 240) if button.primary else (96, 105, 116)
    cv2.rectangle(panel, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), border, 1)
    draw_text(panel, button.label, (x + 14, y + 9), (245, 248, 252), 17)


def draw_event_panel(
    frame: np.ndarray,
    events: Iterable[str],
    scroll_offset: int = 0,
    view_order: list[int] | tuple[int, ...] = (0, 1),
    flip_horizontal: Optional[list[bool]] = None,
    target_status: str = "",
    target_status_color: tuple[int, int, int] = (170, 184, 198),
) -> tuple[np.ndarray, list[UiButton], int, int]:
    panel_height = 300
    panel = np.full((panel_height, frame.shape[1], 3), (24, 27, 31), dtype=np.uint8)
    draw_text(panel, "事件日志", (16, 14), (235, 235, 235), 19)
    draw_text(
        panel,
        "点击画面中的检测框可注册某根管子；也可用按钮注册最佳目标或手动框选。日志区可滚动。",
        (110, 16),
        (170, 184, 198),
        16,
    )
    flip_horizontal = flip_horizontal or [False, False]
    flip_status = " ".join(
        f"{chr(ord('A') + camera_id)}:{'开' if flip_horizontal[camera_id] else '关'}" for camera_id in view_order
    )
    view_status = (
        f"顺序 {view_order_label(view_order)} | "
        f"翻转 {flip_status}"
    )
    draw_text(panel, view_status, (frame.shape[1] - 380, 16), (170, 184, 198), 16)

    buttons = [UiButton("注册左侧目标", "register_left", (16, 48, 142, 38), True)]
    if len(view_order) >= 2:
        buttons.append(UiButton("注册右侧目标", "register_right", (168, 48, 142, 38), True))
    next_x = 320
    if len(view_order) >= 3:
        buttons.append(UiButton("注册第三路目标", "register_third", (next_x, 48, 154, 38), True))
        next_x += 164
    buttons.append(UiButton("手动框选左侧", "manual_left", (next_x, 48, 142, 38)))
    next_x += 152
    if len(view_order) >= 2:
        buttons.append(UiButton("手动框选右侧", "manual_right", (next_x, 48, 142, 38)))
        next_x += 152
    if len(view_order) >= 3:
        buttons.append(UiButton("手动框选第三路", "manual_third", (next_x, 48, 154, 38)))
        next_x += 164
    if len(view_order) >= 2:
        buttons.append(UiButton("交换左右", "swap_views", (next_x, 48, 106, 38)))
        next_x += 116
    buttons.append(UiButton("翻转左侧", "flip_left", (next_x, 48, 106, 38)))
    next_x += 116
    if len(view_order) >= 2:
        buttons.append(UiButton("翻转右侧", "flip_right", (next_x, 48, 106, 38)))
        next_x += 116
    if len(view_order) >= 3:
        buttons.append(UiButton("翻转第三路", "flip_third", (next_x, 48, 122, 38)))
        next_x += 132
    buttons.append(UiButton("退出", "quit", (next_x, 48, 82, 38)))
    for button in buttons:
        draw_button(panel, button)

    if target_status:
        draw_text(panel, target_status, (16, 94), target_status_color, 16)

    all_events = list(events)
    visible_count = max(1, (panel_height - 136) // 22)
    max_scroll = max(0, len(all_events) - visible_count)
    scroll_offset = max(0, min(scroll_offset, max_scroll))
    visible_events = all_events[scroll_offset : scroll_offset + visible_count]

    log_x = 16
    log_y = 122
    log_w = frame.shape[1] - 32
    log_h = panel_height - log_y - 14
    cv2.rectangle(panel, (log_x, log_y), (log_x + log_w, log_y + log_h), (31, 35, 41), -1)
    cv2.rectangle(panel, (log_x, log_y), (log_x + log_w, log_y + log_h), (65, 72, 82), 1)

    status = f"显示 {len(visible_events)} / {len(all_events)} 条"
    if max_scroll > 0:
        status += f"，滚动位置 {scroll_offset + 1}-{scroll_offset + len(visible_events)}"
    draw_text(panel, status, (frame.shape[1] - 230, 68), (165, 180, 195), 15)

    y = log_y + 12
    for event in visible_events:
        draw_text(panel, event, (16, y), (210, 220, 230), 15)
        y += 22
    if max_scroll > 0:
        bar_x = log_x + log_w - 10
        bar_top = log_y + 4
        bar_h = log_h - 8
        thumb_h = max(24, int(bar_h * visible_count / max(visible_count, len(all_events))))
        thumb_y = bar_top + int((bar_h - thumb_h) * scroll_offset / max(1, max_scroll))
        cv2.rectangle(panel, (bar_x, bar_top), (bar_x + 4, bar_top + bar_h), (48, 54, 63), -1)
        cv2.rectangle(panel, (bar_x, thumb_y), (bar_x + 4, thumb_y + thumb_h), (126, 144, 166), -1)
    return np.vstack([frame, panel]), buttons, max_scroll, visible_count


def target_status_for_ui(
    target_profile: TargetProfile,
    tracker: CrossCameraTracker,
    tracks_by_camera: list[list[Track]],
) -> tuple[str, tuple[int, int, int]]:
    if not target_profile.active or tracker.registered_target_id is None:
        return "锁定状态：未锁定目标，点击检测框或使用按钮注册一根管子", (120, 190, 255)

    target_id = tracker.registered_target_id
    matched_parts: list[str] = []
    for camera_id, tracks in enumerate(tracks_by_camera):
        for track in tracks:
            if track.global_id != target_id:
                continue
            sim_text = "" if track.last_target_similarity is None else f" t={track.last_target_similarity:.2f}"
            matched_parts.append(f"摄像头{camera_id + 1}{sim_text}")

    if matched_parts:
        return f"锁定状态：G{target_id:03d} 正在匹配 | {', '.join(matched_parts)}", (120, 220, 150)
    return f"锁定状态：G{target_id:03d} 已注册，当前暂未匹配，等待目标出现或跨镜头接力", (80, 190, 230)


def offset_buttons(buttons: list[UiButton], offset_y: int) -> list[UiButton]:
    return [
        UiButton(
            button.label,
            button.action,
            (button.rect[0], button.rect[1] + offset_y, button.rect[2], button.rect[3]),
            button.primary,
        )
        for button in buttons
    ]


def button_at(buttons: list[UiButton], x: int, y: int) -> Optional[UiButton]:
    for button in buttons:
        bx, by, bw, bh = button.rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return button
    return None


def detection_at(detections: Iterable[Detection], x: int, y: int) -> Optional[Detection]:
    candidates: list[tuple[float, float, Detection]] = []
    for detection in detections:
        bx, by, bw, bh = detection.bbox
        if bx <= x <= bx + bw and by <= y <= by + bh:
            area = float(bw * bh)
            distance = euclidean((float(x), float(y)), detection.center)
            candidates.append((area, distance, detection))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def on_mouse(event: int, x: int, y: int, _flags: int, userdata) -> None:
    if not isinstance(userdata, UiState):
        return
    if event == cv2.EVENT_MOUSEWHEEL:
        delta = cv2.getMouseWheelDelta(_flags)
        step = -3 if delta > 0 else 3
        userdata.event_scroll_offset = max(
            0,
            min(userdata.event_scroll_offset + step, userdata.max_event_scroll),
        )
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        button = button_at(userdata.buttons, x, y)
        if button is not None:
            userdata.pending_action = button.action
            return
        userdata.pending_canvas_click = (x, y)


def id_color(global_id: int) -> tuple[int, int, int]:
    palette = [
        (70, 210, 255),
        (95, 230, 150),
        (230, 160, 95),
        (210, 120, 230),
        (250, 220, 90),
    ]
    return palette[(global_id - 1) % len(palette)]


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h with integers.") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h.")
    if parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive.")
    return parts[0], parts[1], parts[2], parts[3]


def parse_class_ids(value: str, label: str = "class ids") -> Optional[list[int]]:
    if not value:
        return None
    try:
        classes = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated class ids, such as 0,1,2.") from exc
    return classes or None


def parse_yolo_classes(value: str) -> Optional[list[int]]:
    return parse_class_ids(value, "YOLO classes")


def parse_camera_index(value: str) -> Optional[int]:
    text = str(value).strip().lower()
    if text == "auto":
        return None
    try:
        index = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Camera index must be an integer or auto.") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("Camera index must be >= 0, or auto.")
    return index


def parse_camera_index_list(value: str) -> list[int]:
    indexes: list[int] = []
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Camera indexes must be comma-separated integers.") from exc
        if index < 0:
            raise argparse.ArgumentTypeError("Camera indexes must be >= 0.")
        if index in indexes:
            raise argparse.ArgumentTypeError("Camera indexes must not contain duplicates.")
        indexes.append(index)
    if not 1 <= len(indexes) <= 3:
        raise argparse.ArgumentTypeError("Camera indexes must contain 1 to 3 indexes.")
    return indexes


def parse_view_order(value: str) -> tuple[int, int]:
    text = str(value).strip().lower().replace(",", "").replace("|", "").replace(" ", "")
    aliases = {
        "ab": (0, 1),
        "12": (0, 1),
        "lr": (0, 1),
        "ba": (1, 0),
        "21": (1, 0),
        "rl": (1, 0),
    }
    if text not in aliases:
        raise argparse.ArgumentTypeError("View order must be AB or BA.")
    return aliases[text]


def build_initial_view_order(camera_count: int, requested_order: Iterable[int]) -> list[int]:
    view_order: list[int] = []
    for camera_id in requested_order:
        if 0 <= camera_id < camera_count and camera_id not in view_order:
            view_order.append(camera_id)
    for camera_id in range(camera_count):
        if camera_id not in view_order:
            view_order.append(camera_id)
    return view_order


def parse_camera_scan_order(value: str) -> list[int]:
    indexes: list[int] = []
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Camera scan order must be comma-separated integers.") from exc
        if index < 0:
            raise argparse.ArgumentTypeError("Camera scan order indexes must be >= 0.")
        if index not in indexes:
            indexes.append(index)
    if not indexes:
        raise argparse.ArgumentTypeError("Camera scan order cannot be empty.")
    return indexes


def backend_names(selected: str) -> tuple[str, ...]:
    if selected == "auto":
        return AUTO_BACKENDS
    return (selected,)


def open_camera(index: int, backend: str) -> tuple[Optional[cv2.VideoCapture], Optional[str]]:
    for backend_name in backend_names(backend):
        cap = cv2.VideoCapture(index, BACKENDS[backend_name])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, backend_name
        cap.release()
    return None, None


def ordered_camera_indexes(max_index: int, preferred_order: list[int]) -> list[int]:
    indexes: list[int] = []
    for index in preferred_order:
        if index <= max_index and index not in indexes:
            indexes.append(index)
    for index in range(max_index + 1):
        if index not in indexes:
            indexes.append(index)
    return indexes


def find_available_camera_indexes(
    max_index: int,
    backend: str,
    preferred_order: list[int],
    needed: int = 0,
) -> list[int]:
    available: list[int] = []
    for index in ordered_camera_indexes(max_index, preferred_order):
        cap, _backend_name = open_camera(index, backend)
        if cap is not None:
            available.append(index)
            cap.release()
            if needed > 0 and len(available) >= needed:
                break
    return available


def resolve_camera_indexes(args: argparse.Namespace) -> list[int]:
    if args.camera_indexes is not None:
        return list(args.camera_indexes)

    cam_a = args.cam_a
    cam_b = args.cam_b
    if cam_a is not None and cam_b is not None:
        if cam_a == cam_b:
            raise RuntimeError("两个摄像头索引不能相同。请使用不同索引，或使用 --cam-a auto --cam-b auto。")
        return [cam_a, cam_b]

    fixed_indexes = [index for index in (cam_a, cam_b) if index is not None]
    needed = 2 - len(fixed_indexes)
    available = find_available_camera_indexes(
        args.probe_max,
        args.backend,
        args.camera_scan_order,
        needed=needed + len(fixed_indexes),
    )

    auto_candidates = [index for index in available if index not in fixed_indexes]
    if cam_a is None and auto_candidates:
        cam_a = auto_candidates.pop(0)
    if cam_b is None and auto_candidates:
        cam_b = auto_candidates.pop(0)

    if cam_a is None or cam_b is None:
        raise RuntimeError(
            "自动选择摄像头失败：当前可用摄像头少于 2 个。"
            f" 可用索引={available}，可以先用 --probe 检查。"
        )
    if cam_a == cam_b:
        raise RuntimeError("两个摄像头索引不能相同。请使用不同索引，或使用 --cam-a auto --cam-b auto。")
    return [cam_a, cam_b]


def synthetic_camera_count(args: argparse.Namespace) -> int:
    if args.camera_indexes is not None:
        return len(args.camera_indexes)
    return 2


def synthetic_sources(count: int) -> list[SyntheticCamera]:
    if not 1 <= count <= 3:
        raise RuntimeError("内置模拟演示只支持 1 到 3 路摄像头。")
    return [SyntheticCamera(camera_id) for camera_id in range(count)]


def resolve_video_paths(args: argparse.Namespace) -> list[Path]:
    values = [
        str(getattr(args, "video_a", "") or "").strip(),
        str(getattr(args, "video_b", "") or "").strip(),
        str(getattr(args, "video_c", "") or "").strip(),
    ]
    if not any(values):
        return []
    if not values[0]:
        raise RuntimeError("使用离线视频时必须先指定 --video-a。")
    if values[2] and not values[1]:
        raise RuntimeError("指定 --video-c 时也必须指定 --video-b。")

    paths: list[Path] = []
    for value in values:
        if not value:
            break
        paths.append(Path(value).expanduser())
    return paths


def open_video_sources(args: argparse.Namespace, paths: list[Path]) -> list[VideoFileSource]:
    sources: list[VideoFileSource] = []
    try:
        for path in paths:
            sources.append(VideoFileSource(path, loop=args.loop_videos))
    except Exception:
        for source in sources:
            source.release()
        raise

    fps_values = [source.fps for source in sources]
    if max(fps_values) - min(fps_values) > 0.1:
        print(
            "离线视频帧率不一致，将按帧序号同步读取："
            + "，".join(f"{source.path.name}={source.fps:.2f}fps" for source in sources)
        )
    labels = [
        f"{chr(ord('A') + index)}={source.path} ({source.fps:.2f}fps, {source.frame_count}帧)"
        for index, source in enumerate(sources)
    ]
    print("已打开离线视频：" + "，".join(labels))
    return sources


def video_frame_interval(sources: list[object]) -> Optional[float]:
    if not sources or not all(isinstance(source, VideoFileSource) for source in sources):
        return None
    return 1.0 / max(1.0, sources[0].fps)


def open_sources(args: argparse.Namespace):
    video_paths = resolve_video_paths(args)
    if video_paths:
        if args.demo:
            raise RuntimeError("--demo 不能和离线视频参数同时使用。")
        if args.fallback_demo:
            raise RuntimeError("离线视频模式不能使用 --fallback-demo。")
        if args.camera_indexes is not None or args.cam_a is not None or args.cam_b is not None:
            raise RuntimeError("离线视频参数不能和物理摄像头索引同时使用。")
        return open_video_sources(args, video_paths)

    if args.demo:
        return synthetic_sources(synthetic_camera_count(args))

    try:
        camera_indexes = resolve_camera_indexes(args)
    except RuntimeError:
        if args.fallback_demo:
            print("自动选择物理摄像头失败，已切换到内置模拟演示。")
            return synthetic_sources(2)
        raise
    sources = []
    opened_labels = []
    for source_id, camera_index in enumerate(camera_indexes):
        cap, backend_name = open_camera(camera_index, args.backend)
        if cap is None:
            for source in sources:
                source.release()
            if args.fallback_demo:
                print(f"无法同时打开 {len(camera_indexes)} 个物理摄像头，已切换到内置模拟演示。")
                return synthetic_sources(len(camera_indexes))
            raise RuntimeError(
                f"无法打开摄像头索引 {camera_index}。已选择索引={camera_indexes}。"
                " 可以先用 --probe 检查索引，或使用 --backend auto / --fallback-demo。"
            )
        sources.append(cap)
        opened_labels.append(f"{chr(ord('A') + source_id)} 索引={camera_index} 后端={backend_name}")

    print("已打开摄像头：" + "，".join(opened_labels))
    return sources


def roi_for_camera(args: argparse.Namespace, camera_id: int) -> Optional[tuple[int, int, int, int]]:
    if camera_id == 0:
        return args.roi_a
    if camera_id == 1:
        return args.roi_b
    return args.roi_c


def create_yolo_detector(
    args: argparse.Namespace,
    camera_id: int,
    classes: Optional[list[int]],
    model: Optional[object] = None,
) -> YoloDetector:
    return YoloDetector(
        camera_id,
        args.yolo_model,
        confidence=args.yolo_conf,
        iou=args.yolo_iou,
        image_size=args.yolo_imgsz,
        device=args.yolo_device,
        classes=classes,
        roi=roi_for_camera(args, camera_id),
        single_object=args.single_object,
        max_detections=args.max_detections,
        max_area_ratio=args.max_area_ratio,
        max_shape_ratio=args.max_shape_ratio,
        min_long_side=args.min_long_side,
        max_short_side=args.max_short_side,
        model=model,
    )


def validate_rfdetr_args(args: argparse.Namespace) -> None:
    if args.rfdetr_num_classes < 0:
        raise RuntimeError("--rfdetr-num-classes 不能小于 0。")
    if not 0.0 <= args.rfdetr_conf <= 1.0:
        raise RuntimeError("--rfdetr-conf 必须在 0 到 1 之间。")
    if args.rfdetr_category_id_offset < 0:
        raise RuntimeError("--rfdetr-category-id-offset 不能小于 0。")


def create_rfdetr_detector(
    args: argparse.Namespace,
    camera_id: int,
    classes: Optional[list[int]],
    model: Optional[object] = None,
) -> RfDetrDetector:
    return RfDetrDetector(
        camera_id,
        model_size=args.rfdetr_size,
        weights=args.rfdetr_weights,
        num_classes=args.rfdetr_num_classes,
        confidence=args.rfdetr_conf,
        classes=classes,
        class_id_mode=args.rfdetr_class_id_mode,
        category_id_offset=args.rfdetr_category_id_offset,
        roi=roi_for_camera(args, camera_id),
        single_object=args.single_object,
        max_detections=args.max_detections,
        max_area_ratio=args.max_area_ratio,
        max_shape_ratio=args.max_shape_ratio,
        min_long_side=args.min_long_side,
        max_short_side=args.max_short_side,
        optimize=args.rfdetr_optimize,
        model=model,
    )


def build_detectors(args: argparse.Namespace, camera_count: int):
    if args.detector == "motion":
        return [
            MotionDetector(
                camera_id,
                args.min_area,
                args.warmup_frames,
                roi_for_camera(args, camera_id),
                target_mode=args.target_mode,
                single_object=args.single_object,
                max_area_ratio=args.max_area_ratio,
                max_shape_ratio=args.max_shape_ratio,
                min_long_side=args.min_long_side,
                max_short_side=args.max_short_side,
                max_detections=args.max_detections,
            )
            for camera_id in range(camera_count)
        ]

    if args.detector == "yolo":
        classes = parse_yolo_classes(args.yolo_classes)
        return [create_yolo_detector(args, camera_id, classes) for camera_id in range(camera_count)]

    if args.detector == "rfdetr":
        validate_rfdetr_args(args)
        classes = parse_class_ids(args.rfdetr_classes, "RF-DETR classes")
        return [create_rfdetr_detector(args, camera_id, classes) for camera_id in range(camera_count)]

    if args.detector == "hybrid":
        if args.hybrid_fallback_interval <= 0:
            raise RuntimeError("--hybrid-fallback-interval 必须大于 0。")
        validate_rfdetr_args(args)
        yolo_classes = parse_yolo_classes(args.yolo_classes)
        rfdetr_classes = parse_class_ids(args.rfdetr_classes, "RF-DETR classes")
        detectors: list[HybridDetector] = []
        shared_yolo_model: Optional[object] = None
        shared_rfdetr_model: Optional[object] = None
        for camera_id in range(camera_count):
            primary = create_yolo_detector(args, camera_id, yolo_classes, shared_yolo_model)
            fallback = create_rfdetr_detector(args, camera_id, rfdetr_classes, shared_rfdetr_model)
            shared_yolo_model = primary.model
            shared_rfdetr_model = fallback.model
            detectors.append(HybridDetector(primary, fallback, args.hybrid_fallback_interval))
        return detectors

    raise ValueError(f"Unsupported detector: {args.detector}")


def run_config_message(
    args: argparse.Namespace,
    camera_count: int,
    sources: Optional[list[object]] = None,
) -> str:
    if resolve_video_paths(args):
        source_mode = "video"
    elif args.demo:
        source_mode = "demo"
    else:
        source_mode = "camera"
    parts = [
        f"source_mode={source_mode}",
        f"detector={args.detector}",
        f"camera_count={camera_count}",
        f"backend={args.backend}",
        f"max_detections={args.max_detections}",
        f"single_object={int(args.single_object)}",
        f"track_all_after_register={int(args.track_all_after_register)}",
        f"target_threshold={args.target_threshold:.3f}",
        f"target_update_alpha={args.target_update_alpha:.3f}",
        f"cross_threshold={args.cross_threshold:.3f}",
    ]
    if source_mode == "video":
        parts.extend(
            [
                f"video_loop={int(args.loop_videos)}",
                f"video_playback_rate={args.video_playback_rate:.3f}",
            ]
        )
        for camera_id, source in enumerate(sources or []):
            if not isinstance(source, VideoFileSource):
                continue
            label = chr(ord("a") + camera_id)
            parts.extend(
                [
                    f"video_{label}_path={source.path}",
                    f"video_{label}_fps={source.fps:.3f}",
                    f"video_{label}_frames={source.frame_count}",
                ]
            )
    if args.detector in ("yolo", "hybrid"):
        parts.extend(
            [
                f"yolo_model={args.yolo_model}",
                f"yolo_conf={args.yolo_conf:.3f}",
                f"yolo_iou={args.yolo_iou:.3f}",
                f"yolo_imgsz={args.yolo_imgsz}",
                f"yolo_device={args.yolo_device or 'auto'}",
                f"yolo_classes={args.yolo_classes or 'all'}",
            ]
        )
    if args.detector in ("rfdetr", "hybrid"):
        parts.extend(
            [
                f"rfdetr_size={args.rfdetr_size}",
                f"rfdetr_weights={args.rfdetr_weights or 'default'}",
                f"rfdetr_num_classes={args.rfdetr_num_classes}",
                f"rfdetr_classes={args.rfdetr_classes or 'all'}",
                f"rfdetr_conf={args.rfdetr_conf:.3f}",
                f"rfdetr_class_id_mode={args.rfdetr_class_id_mode}",
            ]
        )
    if args.detector == "hybrid":
        parts.append(f"hybrid_fallback_interval={args.hybrid_fallback_interval}")
    return "; ".join(parts)


def write_video_run_manifest(
    log_dir: Path,
    args: argparse.Namespace,
    sources: list[object],
    started_at: float,
    status: str,
    processed_frames: int,
    cross_camera_match_observed: bool,
    event_log: Optional[Path],
    detectors: Optional[list[object]] = None,
) -> Optional[Path]:
    video_sources = [source for source in sources if isinstance(source, VideoFileSource)]
    if args.no_log or len(video_sources) != len(sources):
        return None

    detector: dict[str, object] = {
        "type": args.detector,
        "max_detections": args.max_detections,
    }
    if args.detector in ("yolo", "hybrid"):
        detector["yolo"] = {
            "model": str(Path(args.yolo_model).resolve()) if Path(args.yolo_model).exists() else args.yolo_model,
            "confidence": args.yolo_conf,
            "iou": args.yolo_iou,
            "image_size": args.yolo_imgsz,
            "device": args.yolo_device or "auto",
        }
    if args.detector in ("rfdetr", "hybrid"):
        detector["rfdetr"] = {
            "size": args.rfdetr_size,
            "weights": args.rfdetr_weights or "default",
            "confidence": args.rfdetr_conf,
            "num_classes": args.rfdetr_num_classes,
        }
    if args.detector == "hybrid":
        detector["fallback_interval"] = args.hybrid_fallback_interval
        detector["fallback_stats"] = hybrid_detector_stats(detectors or [])

    payload = {
        "schema_version": 1,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "status": status,
        "processed_frames": processed_frames,
        "cross_camera_match_observed": cross_camera_match_observed,
        "event_log": None if event_log is None else str(event_log.resolve()),
        "source": {
            "mode": "video",
            "loop": args.loop_videos,
            "playback_rate": args.video_playback_rate,
            "videos": [
                {
                    "camera": chr(ord("A") + camera_id),
                    "path": str(source.path),
                    "fps": source.fps,
                    "frame_count": source.frame_count,
                }
                for camera_id, source in enumerate(video_sources)
            ],
        },
        "detector": detector,
        "target": {
            "threshold": args.target_threshold,
            "update_alpha": args.target_update_alpha,
            "cross_threshold": args.cross_threshold,
            "stick_distance": args.target_stick_distance,
            "switch_margin": args.target_switch_margin,
        },
    }
    manifest_path = log_dir / "run_manifest.json"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"无法写入离线视频运行清单：{exc}")
        return None
    return manifest_path


def probe_cameras(max_index: int, backend: str) -> None:
    print("正在探测摄像头索引...")
    for index in range(max_index + 1):
        cap, backend_name = open_camera(index, backend)
        if cap is not None:
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"  索引 {index}: 可用，后端={backend_name}，画面={frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"  索引 {index}: 不可用")
            cap.release()
        else:
            print(f"  索引 {index}: 不可用")


def run(args: argparse.Namespace) -> int:
    if args.video_playback_rate <= 0:
        raise RuntimeError("--video-playback-rate 必须大于 0。")
    if args.detector == "hybrid" and args.hybrid_fallback_interval <= 0:
        raise RuntimeError("--hybrid-fallback-interval 必须大于 0。")
    video_paths = resolve_video_paths(args)
    if video_paths and args.probe:
        raise RuntimeError("--probe 不能和离线视频参数同时使用。")
    if args.probe:
        probe_cameras(args.probe_max, args.backend)
        return 0

    log_dir = Path(args.log_dir)
    event_logger = EventLogger(not args.no_log, log_dir)
    sources = open_sources(args)
    camera_count = len(sources)
    detectors = build_detectors(args, camera_count)
    print(f"检测模式：{args.detector}")
    run_started_at = time.time()
    event_logger.write_run_config(run_started_at, run_config_message(args, camera_count, sources))
    tracker = CrossCameraTracker(
        camera_count=camera_count,
        max_missed=args.max_missed,
        lost_ttl=args.lost_ttl,
        cross_threshold=args.cross_threshold,
        prediction_horizon=args.prediction_horizon,
        event_logger=event_logger,
    )
    target_profile = TargetProfile(max_templates=args.target_template_limit)
    ui_state = UiState()
    view_order = build_initial_view_order(camera_count, args.view_order)
    flip_horizontal = [False for _ in range(camera_count)]
    if camera_count >= 1:
        flip_horizontal[0] = args.flip_a
    if camera_count >= 2:
        flip_horizontal[1] = args.flip_b
    if camera_count >= 3:
        flip_horizontal[2] = args.flip_c
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse, ui_state)

    processed = 0
    matched_seen = False
    run_finished = False
    manifest_path = write_video_run_manifest(
        log_dir,
        args,
        sources,
        run_started_at,
        "running",
        processed,
        matched_seen,
        event_logger.csv_path,
        detectors,
    )
    frame_interval = video_frame_interval(sources)
    wait_delay_ms = (
        max(1, int(round(frame_interval * 1000.0 / args.video_playback_rate)))
        if frame_interval is not None
        else 1
    )
    try:
        while True:
            frames: list[np.ndarray] = []
            stopped = False
            for camera_id, source in enumerate(sources):
                ok, frame = source.read()
                if not ok or frame is None:
                    stopped = True
                    break
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                if flip_horizontal[camera_id]:
                    frame = cv2.flip(frame, 1)
                frames.append(frame)
            if stopped:
                if frame_interval is not None:
                    print("离线视频播放结束：至少一路视频已到结尾。")
                else:
                    print("有一个视频源停止输出画面。")
                break

            now = run_started_at + processed * frame_interval if frame_interval is not None else time.time()

            raw_detections_by_camera = [detector.detect(frames[camera_id]) for camera_id, detector in enumerate(detectors)]
            if args.auto_register_first and not target_profile.active and raw_detections_by_camera[0]:
                register_target_from_detection(
                    target_profile,
                    tracker,
                    event_logger,
                    raw_detections_by_camera[0][0],
                    log_dir,
                    now,
                )
            detections_by_camera = apply_target_profile_with_hybrid_fallback(
                raw_detections_by_camera,
                frames,
                detectors,
                target_profile,
                tracker,
                args,
                log_dir,
                now,
            )
            tracks_by_camera = [
                tracker.update(camera_id, detections_by_camera[camera_id], now)
                for camera_id in range(camera_count)
            ]

            if tracker.cross_camera_match_observed:
                matched_seen = True

            if not args.headless:
                frames_by_camera = {
                    camera_id: (
                        frames[camera_id],
                        tracks_by_camera[camera_id],
                        roi_for_camera(args, camera_id),
                        raw_detections_by_camera[camera_id],
                    )
                    for camera_id in range(camera_count)
                }
                displayed_frames = [
                    draw_tracks(
                        frames_by_camera[camera_id][0],
                        frames_by_camera[camera_id][1],
                        camera_id,
                        frames_by_camera[camera_id][2],
                        args.show_trails,
                    )
                    for camera_id in view_order
                ]
                canvas = np.hstack(displayed_frames)
                canvas = pad_canvas_width(canvas, 1080)
                content_height = canvas.shape[0]
                target_status, target_status_color = target_status_for_ui(
                    target_profile,
                    tracker,
                    tracks_by_camera,
                )
                canvas, buttons, max_event_scroll, _visible_events = draw_event_panel(
                    canvas,
                    tracker.events,
                    ui_state.event_scroll_offset,
                    view_order,
                    flip_horizontal,
                    target_status,
                    target_status_color,
                )
                ui_state.max_event_scroll = max_event_scroll
                ui_state.event_scroll_offset = min(ui_state.event_scroll_offset, max_event_scroll)
                ui_state.buttons = offset_buttons(buttons, content_height)
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(wait_delay_ms) & 0xFF
                action = ui_state.pending_action
                canvas_click = ui_state.pending_canvas_click
                ui_state.pending_action = None
                ui_state.pending_canvas_click = None
                key_action = action_from_key(key)
                if key_action is not None:
                    action = key_action

                if action == "quit":
                    break
                if action == "swap_views":
                    view_order[0], view_order[1] = view_order[1], view_order[0]
                    tracker.events.appendleft(f"{time.strftime('%H:%M:%S')} 已交换左右显示顺序为 {view_order_label(view_order)}")
                if action == "flip_left":
                    camera_id = displayed_camera_index(view_order, "left")
                    flip_horizontal[camera_id] = not flip_horizontal[camera_id]
                    state = "开启" if flip_horizontal[camera_id] else "关闭"
                    tracker.events.appendleft(f"{time.strftime('%H:%M:%S')} 已{state}左侧画面水平翻转")
                if action == "flip_right" and len(view_order) >= 2:
                    camera_id = displayed_camera_index(view_order, "right")
                    flip_horizontal[camera_id] = not flip_horizontal[camera_id]
                    state = "开启" if flip_horizontal[camera_id] else "关闭"
                    tracker.events.appendleft(f"{time.strftime('%H:%M:%S')} 已{state}右侧画面水平翻转")
                if action == "flip_third" and len(view_order) >= 3:
                    camera_id = view_order[2]
                    flip_horizontal[camera_id] = not flip_horizontal[camera_id]
                    state = "开启" if flip_horizontal[camera_id] else "关闭"
                    tracker.events.appendleft(f"{time.strftime('%H:%M:%S')} 已{state}第三路画面水平翻转")
                if action == "register_left":
                    camera_id = displayed_camera_index(view_order, "left")
                    register_best_detection(
                        target_profile,
                        tracker,
                        event_logger,
                        frames_by_camera[camera_id][3],
                        camera_id,
                        log_dir,
                        now,
                    )
                if action == "register_right" and len(view_order) >= 2:
                    camera_id = displayed_camera_index(view_order, "right")
                    register_best_detection(
                        target_profile,
                        tracker,
                        event_logger,
                        frames_by_camera[camera_id][3],
                        camera_id,
                        log_dir,
                        now,
                    )
                if action == "register_third" and len(view_order) >= 3:
                    camera_id = view_order[2]
                    register_best_detection(
                        target_profile,
                        tracker,
                        event_logger,
                        frames_by_camera[camera_id][3],
                        camera_id,
                        log_dir,
                        now,
                    )
                if action == "manual_left":
                    camera_id = displayed_camera_index(view_order, "left")
                    selected = select_target_from_frame(frames_by_camera[camera_id][0], camera_id)
                    if selected is not None:
                        crop, bbox = selected
                        register_target(target_profile, tracker, event_logger, crop, bbox, camera_id, log_dir, now)
                if action == "manual_right" and len(view_order) >= 2:
                    camera_id = displayed_camera_index(view_order, "right")
                    selected = select_target_from_frame(frames_by_camera[camera_id][0], camera_id)
                    if selected is not None:
                        crop, bbox = selected
                        register_target(target_profile, tracker, event_logger, crop, bbox, camera_id, log_dir, now)
                if action == "manual_third" and len(view_order) >= 3:
                    camera_id = view_order[2]
                    selected = select_target_from_frame(frames_by_camera[camera_id][0], camera_id)
                    if selected is not None:
                        crop, bbox = selected
                        register_target(target_profile, tracker, event_logger, crop, bbox, camera_id, log_dir, now)
                if canvas_click is not None and canvas_click[1] < FRAME_H:
                    click_x, click_y = canvas_click
                    display_slot = click_x // FRAME_W
                    if 0 <= display_slot < len(view_order):
                        camera_id = view_order[display_slot]
                        clicked_detection = detection_at(
                            frames_by_camera[camera_id][3],
                            click_x - display_slot * FRAME_W,
                            click_y,
                        )
                        if clicked_detection is not None:
                            register_target_from_detection(
                                target_profile,
                                tracker,
                                event_logger,
                                clicked_detection,
                                log_dir,
                                now,
                            )
                        else:
                            tracker.events.appendleft(
                                f"{time.strftime('%H:%M:%S')} 未点中检测框，请点击目标框内部或使用手动框选"
                            )

            processed += 1
            if args.frames > 0 and processed >= args.frames:
                break
        run_finished = True
    finally:
        if manifest_path is not None:
            write_video_run_manifest(
                log_dir,
                args,
                sources,
                run_started_at,
                "completed" if run_finished else "interrupted",
                processed,
                matched_seen,
                event_logger.csv_path,
                detectors,
            )
        for source in sources:
            source.release()
        event_logger.close()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"已处理帧数：{processed}")
    print(f"是否观察到跨摄像头匹配：{'是' if matched_seen else '否'}")
    if args.detector == "hybrid":
        fallback_stats = hybrid_detector_stats(detectors)
        print(
            "混合补检统计："
            f"调用={fallback_stats['calls']}，候选={fallback_stats['candidates']}，"
            f"恢复={fallback_stats['accepted']}"
        )
    if event_logger.csv_path is not None:
        print(f"事件日志：{event_logger.csv_path}")
    if manifest_path is not None:
        print(f"运行清单：{manifest_path}")
    if tracker.events:
        print("最近事件：")
        for event in reversed(list(tracker.events)):
            print(f"  {event}")

    if args.require_match and not matched_seen:
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-camera OpenCV MVP for cross-camera object Re-ID.",
    )
    parser.add_argument(
        "--camera-indexes",
        type=parse_camera_index_list,
        default=None,
        help="Comma-separated camera indexes to open, 1 to 3 cameras, such as 1,3 or 1,2,3.",
    )
    parser.add_argument("--cam-a", type=parse_camera_index, default=None, help="Camera A index, or auto.")
    parser.add_argument("--cam-b", type=parse_camera_index, default=None, help="Camera B index, or auto.")
    parser.add_argument("--video-a", default="", help="Offline video path for source A.")
    parser.add_argument("--video-b", default="", help="Optional offline video path for source B.")
    parser.add_argument("--video-c", default="", help="Optional offline video path for source C.")
    parser.add_argument("--loop-videos", action="store_true", help="Loop offline videos after reaching the end.")
    parser.add_argument(
        "--video-playback-rate",
        type=float,
        default=1.0,
        help="GUI playback speed for offline videos, such as 0.5 or 2.0.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", *BACKENDS.keys()),
        default="auto",
        help="OpenCV camera backend. auto tries dshow, msmf, then any.",
    )
    parser.add_argument("--demo", action="store_true", help="Use synthetic 1-3 camera demo; default is two cameras.")
    parser.add_argument(
        "--fallback-demo",
        action="store_true",
        help="Use the synthetic demo if selected physical cameras cannot be opened.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable GUI window.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames; 0 means manual stop.")
    parser.add_argument("--view-order", type=parse_view_order, default=(0, 1), help="GUI display order: AB or BA.")
    parser.add_argument("--flip-a", action="store_true", help="Horizontally flip camera A frames.")
    parser.add_argument("--flip-b", action="store_true", help="Horizontally flip camera B frames.")
    parser.add_argument("--flip-c", action="store_true", help="Horizontally flip camera C frames.")
    parser.add_argument(
        "--show-trails",
        action="store_true",
        help="Draw track center history trails with a current direction arrow.",
    )
    parser.add_argument("--probe", action="store_true", help="List available camera indexes.")
    parser.add_argument("--probe-max", type=int, default=5, help="Max camera index for --probe.")
    parser.add_argument(
        "--camera-scan-order",
        type=parse_camera_scan_order,
        default=parse_camera_scan_order("1,3,2,0,4,5"),
        help="Preferred camera indexes for auto selection, such as 1,3,2,0,4,5.",
    )
    parser.add_argument(
        "--detector",
        choices=("motion", "yolo", "rfdetr", "hybrid"),
        default="motion",
        help="Detection backend. hybrid uses YOLO normally and RF-DETR only for registered-target fallback.",
    )
    parser.add_argument("--min-area", type=int, default=900, help="Minimum moving contour area.")
    parser.add_argument(
        "--target-mode",
        choices=("general", "pencil"),
        default="general",
        help="Detection scoring mode. pencil favors long, narrow moving blobs.",
    )
    parser.add_argument(
        "--single-object",
        action="store_true",
        help="Keep only the best detection per camera frame for cleaner demos.",
    )
    parser.add_argument("--max-detections", type=int, default=4, help="Max detections per camera frame.")
    parser.add_argument("--max-area-ratio", type=float, default=0.65, help="Reject boxes larger than this ROI ratio.")
    parser.add_argument(
        "--max-shape-ratio",
        type=float,
        default=1.0,
        help="Reject boxes whose short_side/long_side exceeds this value; lower favors thin objects.",
    )
    parser.add_argument("--min-long-side", type=int, default=0, help="Reject boxes with long side below this size.")
    parser.add_argument("--max-short-side", type=int, default=0, help="Reject boxes with short side above this size.")
    parser.add_argument("--warmup-frames", type=int, default=20, help="Frames used to stabilize background.")
    parser.add_argument("--roi-a", type=parse_roi, help="Camera A ROI as x,y,w,h after resize to 640x360.")
    parser.add_argument("--roi-b", type=parse_roi, help="Camera B ROI as x,y,w,h after resize to 640x360.")
    parser.add_argument("--roi-c", type=parse_roi, help="Camera C ROI as x,y,w,h after resize to 640x360.")
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="Ultralytics model path/name for --detector yolo. Use a trained pipe/pencil model later.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--yolo-device", default="", help="YOLO device, such as cpu, 0, or cuda:0. Empty lets Ultralytics choose.")
    parser.add_argument(
        "--yolo-classes",
        default="",
        help="Optional comma-separated YOLO class ids to keep, for example 0,1. Leave empty for custom pipe models.",
    )
    parser.add_argument(
        "--rfdetr-size",
        choices=("nano", "small", "base", "medium", "large", "xlarge", "2xlarge"),
        default="nano",
        help="RF-DETR model size for --detector rfdetr.",
    )
    parser.add_argument(
        "--rfdetr-weights",
        default="",
        help="Optional RF-DETR checkpoint path, usually output/checkpoint_best_ema.pth after fine-tuning.",
    )
    parser.add_argument(
        "--rfdetr-num-classes",
        type=int,
        default=0,
        help="Optional RF-DETR class count, useful when loading a fine-tuned one-class checkpoint.",
    )
    parser.add_argument("--rfdetr-conf", type=float, default=0.35, help="RF-DETR confidence threshold.")
    parser.add_argument(
        "--rfdetr-classes",
        default="",
        help="Optional comma-separated YOLO-style class ids to keep after RF-DETR class-id mapping. Empty keeps all classes.",
    )
    parser.add_argument(
        "--rfdetr-class-id-mode",
        choices=("auto", "zero", "category"),
        default="auto",
        help="How to map RF-DETR class ids before --rfdetr-classes filtering.",
    )
    parser.add_argument(
        "--rfdetr-category-id-offset",
        type=int,
        default=1,
        help="COCO category id offset used when mapping RF-DETR category ids back to YOLO class ids.",
    )
    parser.add_argument(
        "--rfdetr-optimize",
        action="store_true",
        help="Call RF-DETR optimize_for_inference() when the installed rfdetr package supports it.",
    )
    parser.add_argument(
        "--hybrid-fallback-interval",
        type=int,
        default=15,
        help="Minimum primary frames between RF-DETR fallback attempts per camera in hybrid mode.",
    )
    parser.add_argument("--log-dir", default="runs", help="Directory for per-run CSV event logs.")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV event logging.")
    parser.add_argument("--max-missed", type=int, default=14, help="Frames before a track becomes lost.")
    parser.add_argument("--lost-ttl", type=float, default=8.0, help="Seconds to keep lost IDs matchable.")
    parser.add_argument(
        "--prediction-horizon",
        type=float,
        default=0.35,
        help="Seconds of short-term motion prediction for single-camera tracking.",
    )
    parser.add_argument(
        "--cross-threshold",
        type=float,
        default=0.72,
        help="Minimum cross-camera feature similarity.",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=0.58,
        help="Minimum similarity to the manually registered target template.",
    )
    parser.add_argument(
        "--target-update-alpha",
        type=float,
        default=0.04,
        help="Small feature update rate for accepted target matches; 0 disables adaptation.",
    )
    parser.add_argument(
        "--target-template-limit",
        type=int,
        default=6,
        help="Max reliable templates kept for the registered target.",
    )
    parser.add_argument(
        "--target-stick-distance",
        type=float,
        default=120.0,
        help="Prefer the previous registered-target location within this pixel distance; 0 disables sticky selection.",
    )
    parser.add_argument(
        "--target-switch-margin",
        type=float,
        default=0.08,
        help="Only switch away from the nearby registered target when another candidate is this much more similar.",
    )
    parser.add_argument(
        "--target-sample-max-count",
        type=int,
        default=12,
        help="Max registered-target crop samples saved under log_dir/targets.",
    )
    parser.add_argument(
        "--target-sample-min-similarity",
        type=float,
        default=0.72,
        help="Minimum target similarity for saving a matched target crop sample.",
    )
    parser.add_argument(
        "--target-sample-min-interval",
        type=float,
        default=0.8,
        help="Minimum seconds between saved target crop samples.",
    )
    parser.add_argument(
        "--track-all-after-register",
        action="store_true",
        help="Keep non-target detections visible after registering a target; useful for dense pipe piles.",
    )
    parser.add_argument(
        "--auto-register-first",
        action="store_true",
        help="Register the first Camera A detection as the target; mainly useful for demo/headless tests.",
    )
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="Exit with code 2 if no cross-camera match was observed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
