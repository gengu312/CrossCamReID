from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class YoloBox:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        )


@dataclass
class ImageResult:
    stem: str
    gt_count: int
    pred_count: int
    matched_count: int
    false_positive_count: int
    false_negative_count: int
    avg_iou: Optional[float]
    oversized_count: int


@dataclass
class EvalSummary:
    image_count: int
    gt_count: int
    pred_count: int
    matched_count: int
    false_positive_count: int
    false_negative_count: int
    images_with_false_positive: int
    images_with_false_negative: int
    images_without_predictions: int
    oversized_count: int
    avg_iou: Optional[float]
    precision: float
    recall: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO prediction labels against a prepared dataset split.")
    parser.add_argument("--dataset-root", default="datasets/pipe_yolo", help="Prepared YOLO dataset root.")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Dataset split to analyze.")
    parser.add_argument("--images-dir", help="Override image directory.")
    parser.add_argument("--labels-dir", help="Override ground-truth label directory.")
    parser.add_argument(
        "--pred-labels",
        default="runs_yolo_eval/pipe_yolov8n_eval_predict/labels",
        help="Directory containing YOLO predict save_txt labels.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU threshold for a matched target.")
    parser.add_argument("--conf-threshold", type=float, default=0.0, help="Ignore predictions below this confidence.")
    parser.add_argument(
        "--oversize-ratio",
        type=float,
        default=1.8,
        help="Count a matched prediction as too large if pred_area / gt_area exceeds this ratio.",
    )
    parser.add_argument("--report-csv", help="Optional path to save per-image CSV report.")
    parser.add_argument("--max-examples", type=int, default=12, help="Max issue examples printed to the console.")
    parser.add_argument("--min-precision", type=float, help="Fail if precision is below this value.")
    parser.add_argument("--min-recall", type=float, help="Fail if recall is below this value.")
    parser.add_argument("--max-false-positives", type=int, help="Fail if false positives exceed this number.")
    parser.add_argument("--max-false-negatives", type=int, help="Fail if false negatives exceed this number.")
    parser.add_argument("--require-predictions", action="store_true", help="Fail if prediction label directory is missing.")
    return parser.parse_args()


def read_yolo_labels(path: Path, conf_threshold: float = 0.0) -> list[YoloBox]:
    if not path.exists():
        return []

    boxes: list[YoloBox] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) not in (5, 6):
            raise ValueError(f"{path}:{line_no} YOLO label should have 5 or 6 columns.")
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(value) for value in parts[1:5])
        conf = float(parts[5]) if len(parts) == 6 else 1.0
        if conf < conf_threshold:
            continue
        boxes.append(YoloBox(cls=cls, cx=cx, cy=cy, w=w, h=h, conf=conf))
    return boxes


def box_iou(a: YoloBox, b: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


def image_stems(images_dir: Path) -> list[str]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory was not found: {images_dir}")
    stems = sorted(path.stem for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not stems:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return stems


def match_image(
    stem: str,
    labels_dir: Path,
    pred_labels_dir: Path,
    iou_threshold: float,
    conf_threshold: float,
    oversize_ratio: float,
) -> tuple[ImageResult, list[float]]:
    gt_boxes = read_yolo_labels(labels_dir / f"{stem}.txt")
    pred_boxes = sorted(
        read_yolo_labels(pred_labels_dir / f"{stem}.txt", conf_threshold=conf_threshold),
        key=lambda box: box.conf,
        reverse=True,
    )

    unmatched_gt = set(range(len(gt_boxes)))
    matched_ious: list[float] = []
    false_positive_count = 0
    oversized_count = 0

    for pred in pred_boxes:
        best_gt_index: Optional[int] = None
        best_iou = 0.0
        for gt_index in unmatched_gt:
            gt = gt_boxes[gt_index]
            if pred.cls != gt.cls:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            gt = gt_boxes[best_gt_index]
            unmatched_gt.remove(best_gt_index)
            matched_ious.append(best_iou)
            if gt.area > 0 and pred.area / gt.area > oversize_ratio:
                oversized_count += 1
        else:
            false_positive_count += 1

    false_negative_count = len(unmatched_gt)
    avg_iou = sum(matched_ious) / len(matched_ious) if matched_ious else None
    return (
        ImageResult(
            stem=stem,
            gt_count=len(gt_boxes),
            pred_count=len(pred_boxes),
            matched_count=len(matched_ious),
            false_positive_count=false_positive_count,
            false_negative_count=false_negative_count,
            avg_iou=avg_iou,
            oversized_count=oversized_count,
        ),
        matched_ious,
    )


def summarize(results: list[ImageResult], matched_ious: list[float]) -> EvalSummary:
    gt_count = sum(item.gt_count for item in results)
    pred_count = sum(item.pred_count for item in results)
    matched_count = sum(item.matched_count for item in results)
    false_positive_count = sum(item.false_positive_count for item in results)
    false_negative_count = sum(item.false_negative_count for item in results)
    precision = matched_count / pred_count if pred_count else 0.0
    recall = matched_count / gt_count if gt_count else 0.0
    return EvalSummary(
        image_count=len(results),
        gt_count=gt_count,
        pred_count=pred_count,
        matched_count=matched_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        images_with_false_positive=sum(1 for item in results if item.false_positive_count > 0),
        images_with_false_negative=sum(1 for item in results if item.false_negative_count > 0),
        images_without_predictions=sum(1 for item in results if item.gt_count > 0 and item.pred_count == 0),
        oversized_count=sum(item.oversized_count for item in results),
        avg_iou=sum(matched_ious) / len(matched_ious) if matched_ious else None,
        precision=precision,
        recall=recall,
    )


def write_report(path: Path, results: list[ImageResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "gt",
                "pred",
                "matched",
                "false_positive",
                "false_negative",
                "avg_iou",
                "oversized",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "image": item.stem,
                    "gt": item.gt_count,
                    "pred": item.pred_count,
                    "matched": item.matched_count,
                    "false_positive": item.false_positive_count,
                    "false_negative": item.false_negative_count,
                    "avg_iou": "" if item.avg_iou is None else f"{item.avg_iou:.4f}",
                    "oversized": item.oversized_count,
                }
            )


def print_summary(summary: EvalSummary, results: list[ImageResult], max_examples: int) -> None:
    print(f"图片数：{summary.image_count}")
    print(f"标注目标数：{summary.gt_count}")
    print(f"预测目标数：{summary.pred_count}")
    print(f"匹配成功数：{summary.matched_count}")
    print(f"误检数：{summary.false_positive_count}")
    print(f"漏检数：{summary.false_negative_count}")
    print(f"有误检的图片数：{summary.images_with_false_positive}")
    print(f"有漏检的图片数：{summary.images_with_false_negative}")
    print(f"有标注但没有预测的图片数：{summary.images_without_predictions}")
    print(f"框偏大匹配数：{summary.oversized_count}")
    print(f"平均 IoU：{'无' if summary.avg_iou is None else f'{summary.avg_iou:.3f}'}")
    print(f"Precision：{summary.precision:.3f}")
    print(f"Recall：{summary.recall:.3f}")

    issue_rows = [
        item
        for item in results
        if item.false_positive_count > 0 or item.false_negative_count > 0 or item.oversized_count > 0
    ]
    if not issue_rows:
        print("问题样例：无")
        return

    print("问题样例：")
    for item in issue_rows[:max(0, max_examples)]:
        avg_iou = "无" if item.avg_iou is None else f"{item.avg_iou:.2f}"
        print(
            f"- {item.stem}: 标注={item.gt_count}, 预测={item.pred_count}, "
            f"匹配={item.matched_count}, 误检={item.false_positive_count}, "
            f"漏检={item.false_negative_count}, 框偏大={item.oversized_count}, 平均IoU={avg_iou}"
        )


def quality_failures(summary: EvalSummary, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.min_precision is not None and summary.precision < args.min_precision:
        failures.append(f"Precision {summary.precision:.3f} 低于阈值 {args.min_precision:.3f}。")
    if args.min_recall is not None and summary.recall < args.min_recall:
        failures.append(f"Recall {summary.recall:.3f} 低于阈值 {args.min_recall:.3f}。")
    if args.max_false_positives is not None and summary.false_positive_count > args.max_false_positives:
        failures.append(f"误检数 {summary.false_positive_count} 超过阈值 {args.max_false_positives}。")
    if args.max_false_negatives is not None and summary.false_negative_count > args.max_false_negatives:
        failures.append(f"漏检数 {summary.false_negative_count} 超过阈值 {args.max_false_negatives}。")
    return failures


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    images_dir = Path(args.images_dir) if args.images_dir else dataset_root / "images" / args.split
    labels_dir = Path(args.labels_dir) if args.labels_dir else dataset_root / "labels" / args.split
    pred_labels_dir = Path(args.pred_labels)

    if not labels_dir.exists():
        raise FileNotFoundError(f"Ground-truth label directory was not found: {labels_dir}")
    if not pred_labels_dir.exists():
        message = f"预测标签目录不存在：{pred_labels_dir}"
        if args.require_predictions:
            print(message)
            return 2
        print(message)
        print("请先运行：.\\evaluate_pipe_yolo.bat -PredictOnly")
        return 0

    results: list[ImageResult] = []
    all_ious: list[float] = []
    for stem in image_stems(images_dir):
        result, ious = match_image(
            stem,
            labels_dir,
            pred_labels_dir,
            args.iou_threshold,
            args.conf_threshold,
            args.oversize_ratio,
        )
        results.append(result)
        all_ious.extend(ious)

    summary = summarize(results, all_ious)
    print_summary(summary, results, args.max_examples)
    if args.report_csv:
        report_path = Path(args.report_csv)
        write_report(report_path, results)
        print(f"CSV 报告：{report_path}")

    failures = quality_failures(summary, args)
    if failures:
        print("检测质量验收：未通过")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print("检测质量验收：通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
