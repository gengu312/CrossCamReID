from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterable, Optional

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


@dataclass
class Track:
    camera_id: int
    local_id: int
    global_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    feature: np.ndarray
    last_seen: float
    missed: int = 0
    last_similarity: Optional[float] = None
    last_target_similarity: Optional[float] = None
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

    @property
    def active(self) -> bool:
        return self.feature is not None

    def register_from_detection(self, detection: Detection) -> None:
        self.feature = detection.feature.copy()
        self.samples = 1

    def register_from_crop(self, crop: np.ndarray) -> None:
        h, w = crop.shape[:2]
        self.feature = extract_feature(crop, (w, h), float(w * h))
        self.samples = 1

    def similarity(self, feature: np.ndarray) -> Optional[float]:
        if self.feature is None:
            return None
        return feature_similarity(self.feature, feature)

    def update_from_detection(self, detection: Detection, alpha: float) -> None:
        if self.feature is None or alpha <= 0:
            return
        alpha = max(0.0, min(1.0, alpha))
        self.feature = normalize_vector((1.0 - alpha) * self.feature + alpha * detection.feature)
        self.samples += 1


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
                "bbox": "" if bbox is None else f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "message": message,
            }
        )
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class SyntheticCamera:
    """Small deterministic two-camera scene used for repeatable verification."""

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
        if t < 140 or t > 230:
            return 0, 0, False
        progress = (t - 140) / 90.0
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


class CrossCameraTracker:
    def __init__(
        self,
        match_distance: float = 115.0,
        max_missed: int = 14,
        lost_ttl: float = 8.0,
        cross_threshold: float = 0.72,
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        self.match_distance = match_distance
        self.max_missed = max_missed
        self.lost_ttl = lost_ttl
        self.cross_threshold = cross_threshold
        self.next_local_id = [1, 1]
        self.next_global_id = 1
        self.active: dict[int, list[Track]] = {0: [], 1: []}
        self.lost: list[LostIdentity] = []
        self.events: Deque[str] = deque(maxlen=80)
        self.event_logger = event_logger
        self.registered_target_id: Optional[int] = None
        self.global_seen_cameras: dict[int, set[int]] = {}
        self.cross_camera_match_observed = False

    def activate_registered_target(self, target_global_id: int = 1) -> None:
        self.registered_target_id = target_global_id
        self.next_global_id = max(self.next_global_id, target_global_id + 1)
        self.next_local_id = [1, 1]
        self.active = {0: [], 1: []}
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
            bbox=bbox,
        )
        self._note_seen(target_id, camera_id)

    def update(self, camera_id: int, detections: list[Detection], now: float) -> list[Track]:
        tracks = self.active[camera_id]
        assigned_detection_indexes: set[int] = set()
        assigned_track_indexes: set[int] = set()

        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                dist = euclidean(track.center, detection.center)
                if dist <= self.match_distance:
                    pairs.append((dist, track_index, detection_index))

        for _, track_index, detection_index in sorted(pairs, key=lambda item: item[0]):
            if track_index in assigned_track_indexes or detection_index in assigned_detection_indexes:
                continue
            track = tracks[track_index]
            detection = detections[detection_index]
            similarity = feature_similarity(track.feature, detection.feature)
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
                    bbox=detection.bbox,
                )
            else:
                self._log(
                    now,
                    event_type,
                    camera_id,
                    global_id,
                    self._matched_message(camera_id, global_id, event_type, similarity, detection.target_similarity),
                    similarity=similarity,
                    target_similarity=detection.target_similarity,
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
        if self.registered_target_id is not None and detection.target_similarity is not None:
            return self.registered_target_id, detection.target_similarity, "target_matched"

        global_id, similarity = self._match_lost_identity(detection, now)
        if global_id is not None:
            return global_id, similarity, "matched"
        return None, None, "new"

    def _matched_message(
        self,
        camera_id: int,
        global_id: int,
        event_type: str,
        similarity: Optional[float],
        target_similarity: Optional[float],
    ) -> str:
        if event_type == "target_matched":
            sim_text = "" if target_similarity is None else f"，目标相似度={target_similarity:.2f}"
            return f"摄像头{camera_id + 1}：匹配到目标 G{global_id:03d}{sim_text}"
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
        track.bbox = detection.bbox
        track.center = detection.center
        track.feature = 0.82 * track.feature + 0.18 * detection.feature
        track.last_seen = now
        track.missed = 0
        track.last_similarity = similarity
        track.last_target_similarity = detection.target_similarity
        track.history.append((int(detection.center[0]), int(detection.center[1])))

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
            np.array([aspect, fill_ratio], dtype=np.float32) * 0.18,
            size_hint * 0.12,
            mean_color.astype(np.float32) * 0.15,
        ]
    )
    return normalize_vector(feature).astype(np.float32)


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
) -> list[Detection]:
    if not target_profile.active:
        return detections

    accepted: list[Detection] = []
    for detection in detections:
        detection.target_similarity = target_profile.similarity(detection.feature)
        if detection.target_similarity is not None and detection.target_similarity >= threshold:
            accepted.append(detection)

    accepted.sort(key=lambda item: (item.target_similarity or 0.0, item.score), reverse=True)
    if accepted:
        target_profile.update_from_detection(accepted[0], update_alpha)
    return accepted


def save_target_crop(crop: np.ndarray, log_dir: Path, camera_id: int) -> Optional[Path]:
    target_dir = log_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-cam{camera_id + 1}-target.jpg"
    ok = cv2.imwrite(str(path), crop)
    return path if ok else None


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
    target_profile.saved_path = save_target_crop(crop, log_dir, camera_id)
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
    target_profile.saved_path = save_target_crop(detection.crop, log_dir, detection.camera_id)
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


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    camera_id: int,
    roi: Optional[tuple[int, int, int, int]] = None,
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
        points = list(track.history)
        for i in range(1, len(points)):
            cv2.line(output, points[i - 1], points[i], color, 2, cv2.LINE_AA)
    return output


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
) -> tuple[np.ndarray, list[UiButton], int, int]:
    panel_height = 300
    panel = np.full((panel_height, frame.shape[1], 3), (24, 27, 31), dtype=np.uint8)
    draw_text(panel, "事件日志", (16, 14), (235, 235, 235), 19)
    draw_text(
        panel,
        "先让目标轻微移动并出现检测框，再点击注册按钮。日志区可用鼠标滚轮翻看。",
        (110, 16),
        (170, 184, 198),
        16,
    )

    buttons = [
        UiButton("注册左侧目标", "register_left", (16, 48, 142, 38), True),
        UiButton("注册右侧目标", "register_right", (168, 48, 142, 38), True),
        UiButton("手动框选左侧", "manual_left", (320, 48, 142, 38)),
        UiButton("手动框选右侧", "manual_right", (472, 48, 142, 38)),
        UiButton("退出", "quit", (624, 48, 82, 38)),
    ]
    for button in buttons:
        draw_button(panel, button)

    all_events = list(events)
    visible_count = max(1, (panel_height - 118) // 22)
    max_scroll = max(0, len(all_events) - visible_count)
    scroll_offset = max(0, min(scroll_offset, max_scroll))
    visible_events = all_events[scroll_offset : scroll_offset + visible_count]

    log_x = 16
    log_y = 104
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


def open_sources(args: argparse.Namespace):
    if args.demo:
        return SyntheticCamera(0), SyntheticCamera(1)

    cap_a, backend_a = open_camera(args.cam_a, args.backend)
    cap_b, backend_b = open_camera(args.cam_b, args.backend)

    if cap_a is None or cap_b is None:
        if cap_a is not None:
            cap_a.release()
        if cap_b is not None:
            cap_b.release()
        if args.fallback_demo:
            print("无法同时打开两个物理摄像头，已切换到内置模拟演示。")
            return SyntheticCamera(0), SyntheticCamera(1)
        raise RuntimeError(
            "无法同时打开两个摄像头。可以先用 --probe 检查索引，"
            "或使用 --backend auto / --fallback-demo。"
        )
    print(f"已打开摄像头：A 索引={args.cam_a} 后端={backend_a}，B 索引={args.cam_b} 后端={backend_b}")
    return cap_a, cap_b


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
    if args.probe:
        probe_cameras(args.probe_max, args.backend)
        return 0

    log_dir = Path(args.log_dir)
    event_logger = EventLogger(not args.no_log, log_dir)
    sources = open_sources(args)
    detectors = [
        MotionDetector(
            0,
            args.min_area,
            args.warmup_frames,
            args.roi_a,
            target_mode=args.target_mode,
            single_object=args.single_object,
            max_area_ratio=args.max_area_ratio,
            max_shape_ratio=args.max_shape_ratio,
            min_long_side=args.min_long_side,
            max_short_side=args.max_short_side,
            max_detections=args.max_detections,
        ),
        MotionDetector(
            1,
            args.min_area,
            args.warmup_frames,
            args.roi_b,
            target_mode=args.target_mode,
            single_object=args.single_object,
            max_area_ratio=args.max_area_ratio,
            max_shape_ratio=args.max_shape_ratio,
            min_long_side=args.min_long_side,
            max_short_side=args.max_short_side,
            max_detections=args.max_detections,
        ),
    ]
    tracker = CrossCameraTracker(
        max_missed=args.max_missed,
        lost_ttl=args.lost_ttl,
        cross_threshold=args.cross_threshold,
        event_logger=event_logger,
    )
    target_profile = TargetProfile()
    ui_state = UiState()
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse, ui_state)

    processed = 0
    matched_seen = False
    try:
        while True:
            ok_a, frame_a = sources[0].read()
            ok_b, frame_b = sources[1].read()
            if not ok_a or not ok_b:
                print("有一个视频源停止输出画面。")
                break

            frame_a = cv2.resize(frame_a, (FRAME_W, FRAME_H))
            frame_b = cv2.resize(frame_b, (FRAME_W, FRAME_H))
            now = time.time()

            raw_detections_a = detectors[0].detect(frame_a)
            raw_detections_b = detectors[1].detect(frame_b)
            if args.auto_register_first and not target_profile.active and raw_detections_a:
                register_target_from_detection(
                    target_profile,
                    tracker,
                    event_logger,
                    raw_detections_a[0],
                    log_dir,
                    now,
                )
            detections_a = apply_target_profile(
                raw_detections_a,
                target_profile,
                args.target_threshold,
                args.target_update_alpha,
            )
            detections_b = apply_target_profile(
                raw_detections_b,
                target_profile,
                args.target_threshold,
                args.target_update_alpha,
            )
            tracks_a = tracker.update(0, detections_a, now)
            tracks_b = tracker.update(1, detections_b, now)

            if tracker.cross_camera_match_observed:
                matched_seen = True

            if not args.headless:
                canvas = np.hstack(
                    [
                        draw_tracks(frame_a, tracks_a, 0, args.roi_a),
                        draw_tracks(frame_b, tracks_b, 1, args.roi_b),
                    ]
                )
                content_height = canvas.shape[0]
                canvas, buttons, max_event_scroll, _visible_events = draw_event_panel(
                    canvas,
                    tracker.events,
                    ui_state.event_scroll_offset,
                )
                ui_state.max_event_scroll = max_event_scroll
                ui_state.event_scroll_offset = min(ui_state.event_scroll_offset, max_event_scroll)
                ui_state.buttons = offset_buttons(buttons, content_height)
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                action = ui_state.pending_action
                ui_state.pending_action = None
                if key in (27, ord("q")):
                    action = "quit"
                elif key in (ord("r"), ord("1")):
                    action = "register_left"
                elif key in (ord("t"), ord("2")):
                    action = "register_right"
                elif key in (ord("m"), ord("3")):
                    action = "manual_left"
                elif key in (ord("n"), ord("4")):
                    action = "manual_right"

                if action == "quit":
                    break
                if action == "register_left":
                    register_best_detection(
                        target_profile,
                        tracker,
                        event_logger,
                        raw_detections_a,
                        0,
                        log_dir,
                        time.time(),
                    )
                if action == "register_right":
                    register_best_detection(
                        target_profile,
                        tracker,
                        event_logger,
                        raw_detections_b,
                        1,
                        log_dir,
                        time.time(),
                    )
                if action == "manual_left":
                    selected = select_target_from_frame(frame_a, 0)
                    if selected is not None:
                        crop, bbox = selected
                        register_target(target_profile, tracker, event_logger, crop, bbox, 0, log_dir, time.time())
                if action == "manual_right":
                    selected = select_target_from_frame(frame_b, 1)
                    if selected is not None:
                        crop, bbox = selected
                        register_target(target_profile, tracker, event_logger, crop, bbox, 1, log_dir, time.time())

            processed += 1
            if args.frames > 0 and processed >= args.frames:
                break
    finally:
        sources[0].release()
        sources[1].release()
        event_logger.close()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"已处理帧数：{processed}")
    print(f"是否观察到跨摄像头匹配：{'是' if matched_seen else '否'}")
    if event_logger.csv_path is not None:
        print(f"事件日志：{event_logger.csv_path}")
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
    parser.add_argument("--cam-a", type=int, default=0, help="Camera A index.")
    parser.add_argument("--cam-b", type=int, default=1, help="Camera B index.")
    parser.add_argument(
        "--backend",
        choices=("auto", *BACKENDS.keys()),
        default="auto",
        help="OpenCV camera backend. auto tries dshow, msmf, then any.",
    )
    parser.add_argument("--demo", action="store_true", help="Use synthetic two-camera demo.")
    parser.add_argument(
        "--fallback-demo",
        action="store_true",
        help="Use the synthetic demo if both physical cameras cannot be opened.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable GUI window.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames; 0 means manual stop.")
    parser.add_argument("--probe", action="store_true", help="List available camera indexes.")
    parser.add_argument("--probe-max", type=int, default=5, help="Max camera index for --probe.")
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
    parser.add_argument("--log-dir", default="runs", help="Directory for per-run CSV event logs.")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV event logging.")
    parser.add_argument("--max-missed", type=int, default=14, help="Frames before a track becomes lost.")
    parser.add_argument("--lost-ttl", type=float, default=8.0, help="Seconds to keep lost IDs matchable.")
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
