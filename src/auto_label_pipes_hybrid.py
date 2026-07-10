from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from auto_label_pipes import Box, IMAGE_EXTENSIONS, save_preview, yolo_line


@dataclass(frozen=True)
class LineCandidate:
    angle: float
    rho: float
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create geometry-assisted YOLO labels for the July pencil dataset."
    )
    parser.add_argument("--images", default="dataset_raw/to_label_next/images")
    parser.add_argument("--labels", default="dataset_raw/to_label_next/labels")
    parser.add_argument("--previews", default="dataset_raw/to_label_next/previews")
    parser.add_argument("--report", default="dataset_raw/to_label_next/hybrid_label_report.csv")
    parser.add_argument("--negative-prefix", default="negative")
    parser.add_argument("--single-prefix", default="single_front")
    parser.add_argument("--expected-count", type=int, default=7)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--hue-min", type=int, default=88)
    parser.add_argument("--hue-max", type=int, default=103)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def angle_distance(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def contiguous_interval(
    values: np.ndarray,
    segment_begin: float,
    segment_end: float,
    max_gap: int,
) -> tuple[int, int] | None:
    if len(values) == 0:
        return None
    rounded = np.unique(np.round(values).astype(np.int32))
    groups: list[tuple[int, int]] = []
    begin = previous = int(rounded[0])
    for raw_value in rounded[1:]:
        value = int(raw_value)
        if value - previous > max_gap:
            groups.append((begin, previous))
            begin = value
        previous = value
    groups.append((begin, previous))

    def group_score(group: tuple[int, int]) -> tuple[float, float, int]:
        overlap = max(0.0, min(group[1], segment_end) - max(group[0], segment_begin))
        midpoint_distance = abs(
            (group[0] + group[1]) / 2.0 - (segment_begin + segment_end) / 2.0
        )
        return overlap, -midpoint_distance, group[1] - group[0]

    return max(groups, key=group_score)


def strict_green_mask(image: np.ndarray, hue_min: int, hue_max: int) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([hue_min, 50, 12]),
        np.array([hue_max, 255, 235]),
    )
    mask = cv2.medianBlur(mask, 5)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def line_candidates(
    image: np.ndarray,
    hue_min: int,
    hue_max: int,
) -> tuple[list[LineCandidate], tuple[int, int]]:
    height, width = image.shape[:2]
    short_side = min(height, width)
    mask = strict_green_mask(image, hue_min, hue_max)
    edges = cv2.Canny(mask, 35, 110)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        max(20, int(short_side * 0.025)),
        minLineLength=max(55, int(short_side * 0.09)),
        maxLineGap=max(15, int(short_side * 0.025)),
    )
    if lines is None:
        return [], (height, width)

    mask_y, mask_x = np.where(mask > 0)
    candidates: list[LineCandidate] = []
    band_width = max(6, short_side * 0.009)
    projection_gap = max(5, int(short_side * 0.007))
    min_span = short_side * 0.08

    for x1, y1, x2, y2 in lines[:, 0, :]:
        delta_x = float(x2 - x1)
        delta_y = float(y2 - y1)
        length = math.hypot(delta_x, delta_y)
        if length < short_side * 0.09:
            continue
        unit_x = delta_x / length
        unit_y = delta_y / length
        normal_x = -unit_y
        normal_y = unit_x
        distance = np.abs((mask_x - x1) * normal_x + (mask_y - y1) * normal_y)
        close = distance <= band_width
        if int(close.sum()) < 50:
            continue
        projection = (mask_x[close] - x1) * unit_x + (mask_y[close] - y1) * unit_y
        interval = contiguous_interval(projection, 0.0, length, projection_gap)
        if interval is None:
            continue
        begin, end = interval
        span = float(end - begin)
        if span < min_span:
            continue
        angle = (math.degrees(math.atan2(delta_y, delta_x)) + 180.0) % 180.0
        midpoint_x = x1 + (begin + end) * 0.5 * unit_x
        midpoint_y = y1 + (begin + end) * 0.5 * unit_y
        rho = midpoint_x * normal_x + midpoint_y * normal_y
        candidates.append(
            LineCandidate(
                angle=angle,
                rho=rho,
                score=span,
                x1=x1 + begin * unit_x,
                y1=y1 + begin * unit_y,
                x2=x1 + end * unit_x,
                y2=y1 + end * unit_y,
            )
        )
    return candidates, (height, width)


def select_lines(
    candidates: list[LineCandidate],
    expected_count: int,
    image_shape: tuple[int, int],
) -> list[LineCandidate]:
    short_side = min(image_shape)
    rho_tolerance = max(9.0, short_side * 0.018)
    selected: list[LineCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = any(
            angle_distance(candidate.angle, other.angle) < 7.0
            and abs(candidate.rho - other.rho) < rho_tolerance
            for other in selected
        )
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= expected_count:
            break
    return selected


def detect_hybrid_boxes(
    image: np.ndarray,
    expected_count: int,
    max_side: int,
    hue_min: int,
    hue_max: int,
) -> tuple[list[Box], int]:
    original_height, original_width = image.shape[:2]
    scale = min(1.0, max_side / max(original_height, original_width))
    resized = cv2.resize(
        image,
        (max(1, int(original_width * scale)), max(1, int(original_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    candidates, resized_shape = line_candidates(resized, hue_min, hue_max)
    selected = select_lines(candidates, expected_count, resized_shape)
    padding = max(10, int(min(resized_shape) * 0.014))

    boxes: list[Box] = []
    for line in selected:
        resized_x1 = max(0, int(min(line.x1, line.x2) - padding))
        resized_y1 = max(0, int(min(line.y1, line.y2) - padding))
        resized_x2 = min(resized_shape[1] - 1, int(max(line.x1, line.x2) + padding))
        resized_y2 = min(resized_shape[0] - 1, int(max(line.y1, line.y2) + padding))
        box = Box(
            x1=max(0, int(round(resized_x1 / scale))),
            y1=max(0, int(round(resized_y1 / scale))),
            x2=min(original_width - 1, int(round(resized_x2 / scale))),
            y2=min(original_height - 1, int(round(resized_y2 / scale))),
        )
        if box.area > 0:
            boxes.append(box)
    boxes = largest_spatial_cluster(boxes, original_width, original_height)
    return sorted(boxes, key=lambda item: (item.y1, item.x1)), len(candidates)


def largest_spatial_cluster(boxes: list[Box], width: int, height: int) -> list[Box]:
    if len(boxes) <= 1:
        return boxes
    adjacency: list[set[int]] = [set() for _ in boxes]
    for first in range(len(boxes)):
        first_center_x = (boxes[first].x1 + boxes[first].x2) / 2.0 / width
        first_center_y = (boxes[first].y1 + boxes[first].y2) / 2.0 / height
        for second in range(first + 1, len(boxes)):
            second_center_x = (boxes[second].x1 + boxes[second].x2) / 2.0 / width
            second_center_y = (boxes[second].y1 + boxes[second].y2) / 2.0 / height
            center_distance = math.hypot(
                first_center_x - second_center_x,
                first_center_y - second_center_y,
            )
            if box_iou(boxes[first], boxes[second]) > 0.0 or center_distance <= 0.35:
                adjacency[first].add(second)
                adjacency[second].add(first)

    components: list[list[int]] = []
    unseen = set(range(len(boxes)))
    while unseen:
        start = unseen.pop()
        component = [start]
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(component)
    chosen = max(
        components,
        key=lambda component: (len(component), sum(boxes[index].area for index in component)),
    )
    return [boxes[index] for index in chosen]


def expected_for_stem(stem: str, args: argparse.Namespace) -> int:
    if stem.startswith(args.negative_prefix):
        return 0
    if stem.startswith(args.single_prefix):
        return 1
    return args.expected_count


def box_iou(first: Box, second: Box) -> float:
    intersection_width = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def geometry_metrics(boxes: list[Box], width: int, height: int) -> tuple[float, float]:
    image_area = max(1, width * height)
    max_area_ratio = max((box.area / image_area for box in boxes), default=0.0)
    max_pair_iou = max(
        (
            box_iou(boxes[first], boxes[second])
            for first in range(len(boxes))
            for second in range(first + 1, len(boxes))
        ),
        default=0.0,
    )
    return max_area_ratio, max_pair_iou


def review_status(
    stem: str,
    generated: int,
    expected: int,
    candidate_count: int,
    max_area_ratio: float,
    max_pair_iou: float,
) -> str:
    if expected == 0:
        return "trusted" if generated == 0 else "review"
    if stem.startswith("partial_visible"):
        if not 3 <= generated <= expected:
            return "review"
    elif generated != expected:
        return "review"
    if candidate_count < generated:
        return "review"
    if max_area_ratio > 0.45 or max_pair_iou > 0.95:
        return "review"
    return "trusted"


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    preview_dir = Path(args.previews)
    report_path = Path(args.report)
    if not image_dir.is_dir():
        print(f"混合预标注失败：图片目录不存在：{image_dir}")
        return 2
    if args.expected_count <= 0 or args.max_side <= 0:
        print("混合预标注失败：--expected-count 和 --max-side 必须大于 0。")
        return 2
    if not 0 <= args.hue_min <= args.hue_max <= 179:
        print("混合预标注失败：HSV 色相范围应在 0 到 179 之间。")
        return 2

    label_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"混合预标注失败：图片目录中没有图片：{image_dir}")
        return 2

    rows: list[dict[str, object]] = []
    total_boxes = 0
    unreadable = 0
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            print(f"混合预标注失败：标签已存在，请使用 --overwrite：{label_path}")
            return 2
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable += 1
            rows.append(
                {
                    "image": image_path.name,
                    "expected_count": "",
                    "generated_count": 0,
                    "line_candidates": 0,
                    "max_area_ratio": "",
                    "max_pair_iou": "",
                    "status": "unreadable",
                }
            )
            continue

        expected = expected_for_stem(image_path.stem, args)
        if expected == 0:
            boxes: list[Box] = []
            candidate_count = 0
        else:
            boxes, candidate_count = detect_hybrid_boxes(
                image,
                expected,
                args.max_side,
                args.hue_min,
                args.hue_max,
            )
        label_text = "\n".join(yolo_line(box, image.shape[1], image.shape[0]) for box in boxes)
        label_path.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
        if args.preview:
            save_preview(image, boxes, preview_dir / image_path.name)
        max_area_ratio, max_pair_iou = geometry_metrics(boxes, image.shape[1], image.shape[0])
        status = review_status(
            image_path.stem,
            len(boxes),
            expected,
            candidate_count,
            max_area_ratio,
            max_pair_iou,
        )
        rows.append(
            {
                "image": image_path.name,
                "expected_count": expected,
                "generated_count": len(boxes),
                "line_candidates": candidate_count,
                "max_area_ratio": f"{max_area_ratio:.6f}",
                "max_pair_iou": f"{max_pair_iou:.6f}",
                "status": status,
            }
        )
        total_boxes += len(boxes)

    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "expected_count",
                "generated_count",
                "line_candidates",
                "max_area_ratio",
                "max_pair_iou",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    trusted = sum(row["status"] == "trusted" for row in rows)
    review = sum(row["status"] == "review" for row in rows)
    print(
        f"混合预标注完成：images={len(images)} boxes={total_boxes} "
        f"trusted={trusted} review={review} unreadable={unreadable}"
    )
    print(f"标签目录：{label_dir}")
    print(f"质量报告：{report_path}")
    return 0 if unreadable == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
