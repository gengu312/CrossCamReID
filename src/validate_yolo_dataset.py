from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitReport:
    name: str
    image_dir: Path
    label_dir: Path
    image_count: int = 0
    label_count: int = 0
    object_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset before training.")
    parser.add_argument("--dataset-root", default="datasets/pipe_yolo", help="Dataset root containing images/ and labels/.")
    parser.add_argument("--class-count", type=int, default=1, help="Allowed class id range is 0..class-count-1.")
    parser.add_argument("--min-train", type=int, default=1, help="Minimum train image count.")
    parser.add_argument("--min-val", type=int, default=1, help="Minimum val image count.")
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Allow images without .txt labels. Prefer empty .txt files for negative samples.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def list_labels(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt"}


def validate_label_file(path: Path, class_count: int, report: SplitReport) -> int:
    object_count = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8-sig").splitlines()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            report.errors.append(f"{path}:{line_number} 应为 5 列：class x_center y_center width height")
            continue
        try:
            class_id = int(parts[0])
            values = [float(part) for part in parts[1:]]
        except ValueError:
            report.errors.append(f"{path}:{line_number} 包含非数字字段")
            continue
        if class_id < 0 or class_id >= class_count:
            report.errors.append(f"{path}:{line_number} 类别 {class_id} 超出范围 0..{class_count - 1}")
        for value in values:
            if value < 0.0 or value > 1.0:
                report.errors.append(f"{path}:{line_number} 坐标 {value} 不在 0..1 之间")
        if values[2] <= 0.0 or values[3] <= 0.0:
            report.errors.append(f"{path}:{line_number} width/height 必须大于 0")
        object_count += 1
    return object_count


def validate_split(
    root: Path,
    split: str,
    class_count: int,
    allow_missing_labels: bool,
    min_images: int,
) -> SplitReport:
    report = SplitReport(
        name=split,
        image_dir=root / "images" / split,
        label_dir=root / "labels" / split,
    )
    images = list_images(report.image_dir)
    labels = list_labels(report.label_dir)
    report.image_count = len(images)
    report.label_count = len(labels)

    if report.image_count < min_images:
        report.errors.append(f"{split} 图片数量 {report.image_count} 小于要求 {min_images}")

    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    if missing_labels:
        message = f"{split} 有 {len(missing_labels)} 张图片没有同名标签文件"
        if allow_missing_labels:
            report.warnings.append(message)
        else:
            report.errors.append(message + "；负样本建议放空 .txt，或显式使用 --allow-missing-labels")

    if orphan_labels:
        report.errors.append(f"{split} 有 {len(orphan_labels)} 个标签文件没有同名图片")

    for label_path in labels.values():
        report.object_count += validate_label_file(label_path, class_count, report)

    return report


def print_report(reports: list[SplitReport]) -> None:
    for report in reports:
        print(
            f"{report.name}: 图片={report.image_count}, 标签={report.label_count}, "
            f"标注目标={report.object_count}"
        )
        for warning in report.warnings:
            print(f"  WARN: {warning}")
        for error in report.errors:
            print(f"  ERROR: {error}")


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root)
    reports = [
        validate_split(root, "train", args.class_count, args.allow_missing_labels, args.min_train),
        validate_split(root, "val", args.class_count, args.allow_missing_labels, args.min_val),
    ]
    print_report(reports)

    error_count = sum(len(report.errors) for report in reports)
    if error_count:
        print(f"数据集校验失败：{error_count} 个错误。")
        return 2
    print("数据集校验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
