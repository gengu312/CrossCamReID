from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create first-pass YOLO labels for green pencil/pipe photos.")
    parser.add_argument("--images", default="dataset_raw/to_label_20260626/images", help="Input image directory.")
    parser.add_argument("--labels", default="dataset_raw/to_label_20260626/labels", help="Output YOLO label directory.")
    parser.add_argument("--previews", default="dataset_raw/to_label_20260626/previews", help="Output preview directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty label files.")
    parser.add_argument("--preview", action="store_true", help="Save preview images with boxes.")
    parser.add_argument("--negative-prefix", default="negative", help="Images with this prefix get empty labels.")
    parser.add_argument("--min-area-ratio", type=float, default=0.0008, help="Minimum box area as ratio of image area.")
    return parser.parse_args()


def green_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Green pencils are saturated teal/green; the wooden desk is yellow/brown and stays outside this hue range.
    mask = cv2.inRange(hsv, np.array([35, 35, 20]), np.array([100, 255, 245]))
    mask = cv2.medianBlur(mask, 5)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return mask


def dominant_angle(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return None
    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    _ = mean
    vx, vy = eigenvectors[0]
    angle = math.degrees(math.atan2(float(vy), float(vx)))
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return angle


def rotate_mask(mask: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, matrix


def row_intervals(rotated_mask: np.ndarray) -> list[tuple[int, int]]:
    projection = (rotated_mask > 0).sum(axis=1).astype(np.float32)
    if projection.max(initial=0) <= 0:
        return []
    smooth = cv2.GaussianBlur(projection.reshape(-1, 1), (1, 31), 0).ravel()
    threshold = max(25.0, float(smooth.max()) * 0.18)
    active = smooth >= threshold

    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(active) - 1))

    merged: list[tuple[int, int]] = []
    for begin, end in intervals:
        if end - begin + 1 < 8:
            continue
        if merged and begin - merged[-1][1] <= 10:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))
    return merged


def inverse_box_from_rotated(
    rotated_mask: np.ndarray,
    matrix: np.ndarray,
    interval: tuple[int, int],
    image_shape: tuple[int, int],
) -> Box | None:
    h, w = image_shape
    begin, end = interval
    pad_y = 18
    y1 = max(0, begin - pad_y)
    y2 = min(rotated_mask.shape[0] - 1, end + pad_y)
    ys, xs = np.where(rotated_mask[y1 : y2 + 1, :] > 0)
    if len(xs) < 80:
        return None
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32) + y1
    x1 = max(0.0, float(xs.min()) - 24.0)
    x2 = min(float(rotated_mask.shape[1] - 1), float(xs.max()) + 24.0)

    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float32,
    )
    original = corners @ inverse.T
    bx1 = int(np.floor(np.clip(original[:, 0].min(), 0, w - 1)))
    by1 = int(np.floor(np.clip(original[:, 1].min(), 0, h - 1)))
    bx2 = int(np.ceil(np.clip(original[:, 0].max(), 0, w - 1)))
    by2 = int(np.ceil(np.clip(original[:, 1].max(), 0, h - 1)))
    if bx2 <= bx1 or by2 <= by1:
        return None
    return Box(bx1, by1, bx2, by2)


def inverse_rect_from_rotated(
    matrix: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_shape: tuple[int, int],
) -> Box | None:
    h, w = image_shape
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float32,
    )
    original = corners @ inverse.T
    bx1 = int(np.floor(np.clip(original[:, 0].min(), 0, w - 1)))
    by1 = int(np.floor(np.clip(original[:, 1].min(), 0, h - 1)))
    bx2 = int(np.ceil(np.clip(original[:, 0].max(), 0, w - 1)))
    by2 = int(np.ceil(np.clip(original[:, 1].max(), 0, h - 1)))
    if bx2 <= bx1 or by2 <= by1:
        return None
    return Box(bx1, by1, bx2, by2)


def robust_bounds(rotated_mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(rotated_mask > 0)
    if len(xs) < 100:
        return None
    x1, x2 = np.percentile(xs, [1.0, 99.0])
    y1, y2 = np.percentile(ys, [1.0, 99.0])
    return float(x1), float(y1), float(x2), float(y2)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def estimate_count(stem: str, rotated_height: float) -> int:
    if stem.startswith("single"):
        return 1
    if stem.startswith("small_stack"):
        return clamp(int(round(rotated_height / 145.0)), 2, 5)
    if stem.startswith("stack_many"):
        return clamp(int(round(rotated_height / 135.0)), 5, 10)
    if stem.startswith("hand_move"):
        return clamp(int(round(rotated_height / 220.0)), 1, 4)
    return max(1, int(round(rotated_height / 150.0)))


def split_rotated_group(
    rotated_mask: np.ndarray,
    matrix: np.ndarray,
    stem: str,
    image_shape: tuple[int, int],
    min_area: int,
) -> list[Box]:
    bounds = robust_bounds(rotated_mask)
    if bounds is None:
        return []
    group_x1, group_y1, group_x2, group_y2 = bounds
    group_height = max(1.0, group_y2 - group_y1)
    count = estimate_count(stem, group_height)

    boxes: list[Box] = []
    band_height = group_height / count
    for index in range(count):
        y1 = group_y1 + index * band_height
        y2 = group_y1 + (index + 1) * band_height
        pad_y = max(12.0, band_height * 0.18)
        band_y1 = max(0.0, y1 - pad_y)
        band_y2 = min(float(rotated_mask.shape[0] - 1), y2 + pad_y)
        ys, xs = np.where(rotated_mask[int(band_y1) : int(band_y2) + 1, :] > 0)
        if len(xs) >= 80:
            band_x1, band_x2 = np.percentile(xs, [1.0, 99.0])
            band_x1 = min(float(band_x1), group_x1)
            band_x2 = max(float(band_x2), group_x2)
        else:
            band_x1, band_x2 = group_x1, group_x2
        pad_x = 28.0
        box = inverse_rect_from_rotated(
            matrix,
            max(0.0, float(band_x1) - pad_x),
            band_y1,
            min(float(rotated_mask.shape[1] - 1), float(band_x2) + pad_x),
            band_y2,
            image_shape,
        )
        if box and box.area >= min_area:
            boxes.append(box)
    return boxes


def contour_fallback(mask: np.ndarray, image_shape: tuple[int, int], min_area: int) -> list[Box]:
    h, w = image_shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area * 0.35:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        box = Box(max(0, x - 10), max(0, y - 10), min(w - 1, x + bw + 10), min(h - 1, y + bh + 10))
        if box.area >= min_area:
            boxes.append(box)
    return boxes


def non_max_suppression(boxes: list[Box], threshold: float = 0.55) -> list[Box]:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.area, reverse=True):
        duplicate = False
        for other in kept:
            ix1 = max(box.x1, other.x1)
            iy1 = max(box.y1, other.y1)
            ix2 = min(box.x2, other.x2)
            iy2 = min(box.y2, other.y2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = box.area + other.area - intersection
            if union > 0 and intersection / union > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return sorted(kept, key=lambda item: (item.y1, item.x1))


def detect_boxes(image: np.ndarray, min_area_ratio: float, stem: str = "") -> list[Box]:
    h, w = image.shape[:2]
    min_area = int(w * h * min_area_ratio)
    mask = green_mask(image)
    angle = dominant_angle(mask)
    boxes: list[Box] = []
    if angle is not None:
        rotated, matrix = rotate_mask(mask, angle)
        boxes = split_rotated_group(rotated, matrix, stem, (h, w), min_area)
        if boxes:
            return boxes
    if not boxes:
        boxes = contour_fallback(mask, (h, w), min_area)
    return non_max_suppression(boxes)


def yolo_line(box: Box, width: int, height: int) -> str:
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    bw = (box.x2 - box.x1) / width
    bh = (box.y2 - box.y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def save_preview(image: np.ndarray, boxes: list[Box], path: Path) -> None:
    preview = image.copy()
    for index, box in enumerate(boxes, start=1):
        cv2.rectangle(preview, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 4)
        cv2.putText(
            preview,
            str(index),
            (box.x1, max(35, box.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    preview_dir = Path(args.previews)
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    total_boxes = 0
    written = 0
    skipped = 0
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip() and not args.overwrite:
            skipped += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARN unreadable image: {image_path}")
            continue

        if image_path.stem.startswith(args.negative_prefix):
            boxes: list[Box] = []
        else:
            boxes = detect_boxes(image, args.min_area_ratio, image_path.stem)

        label_text = "\n".join(yolo_line(box, image.shape[1], image.shape[0]) for box in boxes)
        label_path.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
        if args.preview:
            save_preview(image, boxes, preview_dir / image_path.name)
        written += 1
        total_boxes += len(boxes)
        print(f"{image_path.name}: boxes={len(boxes)}")

    print(f"done images={len(images)} written={written} skipped={skipped} boxes={total_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
