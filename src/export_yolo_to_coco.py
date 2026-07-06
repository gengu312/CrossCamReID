from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from validate_yolo_dataset import IMAGE_EXTENSIONS, list_images


@dataclass(frozen=True)
class ExportStats:
    split: str
    image_count: int
    annotation_count: int
    output_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a prepared YOLO dataset to RF-DETR COCO format.")
    parser.add_argument("--yolo-root", default="datasets/pipe_yolo", help="Prepared YOLO dataset root.")
    parser.add_argument("--output-root", default="datasets/pipe_rfdetr", help="Output RF-DETR/COCO dataset root.")
    parser.add_argument(
        "--class-names",
        default="pipe",
        help="Comma-separated class names. YOLO class 0 maps to the first name.",
    )
    parser.add_argument(
        "--category-id-offset",
        type=int,
        default=1,
        help="COCO category id offset. Default maps YOLO class 0 to COCO category 1.",
    )
    parser.add_argument("--clean", action="store_true", help="Clean output train/valid/test directories first.")
    parser.add_argument(
        "--test-from-val",
        action="store_true",
        help="Also copy the YOLO val split to RF-DETR test. Useful when no separate test split exists yet.",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Treat images without .txt labels as negative samples.",
    )
    return parser.parse_args()


def parse_class_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise RuntimeError("--class-names 不能为空。")
    if len(set(names)) != len(names):
        raise RuntimeError("--class-names 不能包含重复类别名。")
    return names


def safe_clean_output(root: Path) -> None:
    root_resolved = root.resolve()
    for split in ("train", "valid", "test"):
        split_dir = (root / split).resolve()
        if root_resolved not in split_dir.parents:
            raise RuntimeError(f"拒绝清理输出目录之外的路径：{split_dir}")
        if split_dir.exists():
            shutil.rmtree(split_dir)


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def read_yolo_labels(
    label_path: Path,
    width: int,
    height: int,
    class_names: list[str],
    category_id_offset: int,
) -> list[dict]:
    if not label_path.exists():
        return []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()

    annotations = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise RuntimeError(f"{label_path}:{line_number} 应为 5 列：class x_center y_center width height")
        try:
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = [float(part) for part in parts[1:]]
        except ValueError as exc:
            raise RuntimeError(f"{label_path}:{line_number} 包含非数字字段") from exc
        if class_id < 0 or class_id >= len(class_names):
            raise RuntimeError(f"{label_path}:{line_number} 类别 {class_id} 超出 class-names 范围")
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, box_width, box_height)):
            raise RuntimeError(f"{label_path}:{line_number} 坐标不在 0..1 范围内")
        if box_width <= 0.0 or box_height <= 0.0:
            raise RuntimeError(f"{label_path}:{line_number} width/height 必须大于 0")

        pixel_width = box_width * width
        pixel_height = box_height * height
        x_min = (x_center * width) - pixel_width / 2.0
        y_min = (y_center * height) - pixel_height / 2.0
        x_min = max(0.0, min(float(width), x_min))
        y_min = max(0.0, min(float(height), y_min))
        pixel_width = max(1.0, min(pixel_width, float(width) - x_min))
        pixel_height = max(1.0, min(pixel_height, float(height) - y_min))

        annotations.append(
            {
                "category_id": class_id + category_id_offset,
                "bbox": [
                    round(x_min, 3),
                    round(y_min, 3),
                    round(pixel_width, 3),
                    round(pixel_height, 3),
                ],
                "area": round(pixel_width * pixel_height, 3),
                "iscrowd": 0,
                "segmentation": [],
            }
        )
    return annotations


def export_split(
    yolo_root: Path,
    yolo_split: str,
    output_root: Path,
    coco_split: str,
    class_names: list[str],
    category_id_offset: int,
    allow_missing_labels: bool,
) -> ExportStats:
    image_dir = yolo_root / "images" / yolo_split
    label_dir = yolo_root / "labels" / yolo_split
    if not image_dir.exists():
        raise RuntimeError(f"图片目录不存在：{image_dir}")
    if not label_dir.exists():
        raise RuntimeError(f"标签目录不存在：{label_dir}")

    images = list_images(image_dir)
    output_dir = output_root / coco_split
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    annotation_id = 1

    for image_id, image_path in enumerate(sorted(images.values(), key=lambda item: item.name), start=1):
        output_image = output_dir / image_path.name
        shutil.copy2(image_path, output_image)
        width, height = read_image_size(image_path)
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists() and not allow_missing_labels:
            raise RuntimeError(f"缺少标签文件：{label_path}")
        labels = read_yolo_labels(label_path, width, height, class_names, category_id_offset)
        for label in labels:
            coco_annotations.append({"id": annotation_id, "image_id": image_id, **label})
            annotation_id += 1

    categories = [
        {
            "id": index + category_id_offset,
            "name": name,
            "supercategory": "object",
        }
        for index, name in enumerate(class_names)
    ]
    output_json = output_dir / "_annotations.coco.json"
    output_json.write_text(
        json.dumps(
            {
                "info": {
                    "description": "CrossCamReID RF-DETR export from YOLO labels",
                    "version": "1.0",
                },
                "licenses": [],
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": categories,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ExportStats(coco_split, len(coco_images), len(coco_annotations), output_json)


def export_empty_split(output_root: Path, split: str, class_names: list[str], category_id_offset: int) -> ExportStats:
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = [
        {
            "id": index + category_id_offset,
            "name": name,
            "supercategory": "object",
        }
        for index, name in enumerate(class_names)
    ]
    output_json = output_dir / "_annotations.coco.json"
    output_json.write_text(
        json.dumps(
            {
                "info": {
                    "description": "Empty CrossCamReID RF-DETR test split",
                    "version": "1.0",
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": categories,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ExportStats(split, 0, 0, output_json)


def main() -> int:
    args = parse_args()
    yolo_root = Path(args.yolo_root)
    output_root = Path(args.output_root)

    try:
        class_names = parse_class_names(args.class_names)
        if args.category_id_offset < 0:
            raise RuntimeError("--category-id-offset 不能小于 0。")
        if args.clean:
            safe_clean_output(output_root)
        stats = [
            export_split(
                yolo_root,
                "train",
                output_root,
                "train",
                class_names,
                args.category_id_offset,
                args.allow_missing_labels,
            ),
            export_split(
                yolo_root,
                "val",
                output_root,
                "valid",
                class_names,
                args.category_id_offset,
                args.allow_missing_labels,
            ),
        ]
        if args.test_from_val:
            stats.append(
                export_split(
                    yolo_root,
                    "val",
                    output_root,
                    "test",
                    class_names,
                    args.category_id_offset,
                    args.allow_missing_labels,
                )
            )
        else:
            stats.append(export_empty_split(output_root, "test", class_names, args.category_id_offset))
    except RuntimeError as exc:
        print(f"RF-DETR 数据集导出失败：{exc}")
        return 2

    for item in stats:
        print(f"{item.split}: 图片={item.image_count}, 标注={item.annotation_count}, JSON={item.output_json}")
    print(f"RF-DETR/COCO 数据集输出目录：{output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
