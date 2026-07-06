from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from validate_yolo_dataset import IMAGE_EXTENSIONS


@dataclass
class DatasetItem:
    stem: str
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare exported YOLO labels into train/val folders.")
    parser.add_argument("--source-images", required=True, help="Directory containing exported images.")
    parser.add_argument("--source-labels", required=True, help="Directory containing exported YOLO .txt labels.")
    parser.add_argument("--dataset-root", default="datasets/pipe_yolo", help="Output dataset root.")
    parser.add_argument("--class-names", default="pipe", help="Comma-separated class names for data.yaml.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--clean", action="store_true", help="Clean existing train/val images and labels first.")
    parser.add_argument(
        "--allow-negative",
        action="store_true",
        help="Create empty label files for images without labels. Use this only for intentional negative samples.",
    )
    parser.add_argument(
        "--drop-confidence-column",
        action="store_true",
        help="Convert YOLO prediction labels from 6 columns to 5 training columns.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    duplicate_stems: dict[str, list[str]] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in images:
            duplicate_stems.setdefault(path.stem, [images[path.stem].name]).append(path.name)
            continue
        images[path.stem] = path
    if duplicate_stems:
        preview = "; ".join(f"{stem}: {', '.join(names)}" for stem, names in list(duplicate_stems.items())[:3])
        raise RuntimeError(f"图片目录中存在同名不同后缀图片，请先去重：{preview}")
    return images


def list_labels(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt"}


def parse_class_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise RuntimeError("--class-names 不能为空。")
    if len(set(names)) != len(names):
        raise RuntimeError("--class-names 不能包含重复类别名。")
    return names


def write_data_yaml(root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {json.dumps(root.resolve().as_posix(), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(class_names))
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_output(root: Path) -> None:
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_file():
                path.unlink()


def ensure_output_dirs(root: Path) -> None:
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        keep = directory / ".gitkeep"
        if not keep.exists():
            keep.write_text("\n", encoding="utf-8")


def collect_items(source_images: Path, source_labels: Path, allow_negative: bool) -> list[DatasetItem]:
    images = list_images(source_images)
    labels = list_labels(source_labels)
    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))

    if orphan_labels:
        preview = ", ".join(orphan_labels[:5])
        raise RuntimeError(f"有 {len(orphan_labels)} 个标签没有同名图片：{preview}")
    if missing_labels and not allow_negative:
        preview = ", ".join(missing_labels[:5])
        raise RuntimeError(
            f"有 {len(missing_labels)} 张图片没有同名标签：{preview}。"
            "如果这些是负样本，请加 --allow-negative。"
        )

    items: list[DatasetItem] = []
    for stem, image_path in sorted(images.items()):
        label_path = labels.get(stem)
        if label_path is None:
            label_path = source_labels / f"{stem}.txt"
        items.append(DatasetItem(stem=stem, image_path=image_path, label_path=label_path))
    return items


def split_items(items: list[DatasetItem], val_ratio: float, seed: int) -> tuple[list[DatasetItem], list[DatasetItem]]:
    if not 0.0 < val_ratio < 1.0:
        raise RuntimeError("--val-ratio 必须在 0 到 1 之间。")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_items = sorted(shuffled[:val_count], key=lambda item: item.stem)
    train_items = sorted(shuffled[val_count:], key=lambda item: item.stem)
    return train_items, val_items


def write_training_label(source: Path, output: Path, drop_confidence_column: bool) -> None:
    if not drop_confidence_column:
        shutil.copy2(source, output)
        return

    converted_lines: list[str] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) == 6:
            parts = parts[:5]
        elif len(parts) != 5:
            raise RuntimeError(f"{source}:{line_number} YOLO 标签列数应为 5 或 6，实际为 {len(parts)}。")
        converted_lines.append(" ".join(parts))
    output.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")


def copy_item(
    item: DatasetItem,
    root: Path,
    split: str,
    allow_negative: bool,
    drop_confidence_column: bool,
) -> None:
    image_output = root / "images" / split / item.image_path.name
    label_output = root / "labels" / split / f"{item.stem}.txt"
    shutil.copy2(item.image_path, image_output)
    if item.label_path.exists():
        write_training_label(item.label_path, label_output, drop_confidence_column)
    elif allow_negative:
        label_output.write_text("", encoding="utf-8")
    else:
        raise RuntimeError(f"缺少标签文件：{item.label_path}")


def digest_split(items: list[DatasetItem]) -> str:
    text = "\n".join(item.stem for item in items)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def main() -> int:
    args = parse_args()
    source_images = Path(args.source_images)
    source_labels = Path(args.source_labels)
    root = Path(args.dataset_root)

    if not source_images.is_dir():
        print(f"图片目录不存在或不是文件夹：{source_images}")
        return 2
    if not source_labels.is_dir():
        print(f"标签目录不存在或不是文件夹：{source_labels}")
        return 2

    try:
        class_names = parse_class_names(args.class_names)
        items = collect_items(source_images, source_labels, args.allow_negative)
        if not items:
            raise RuntimeError("没有找到可整理的图片。")
        train_items, val_items = split_items(items, args.val_ratio, args.seed)
        if not train_items or not val_items:
            raise RuntimeError("train/val 不能为空；请增加图片数量或调整 --val-ratio。")
        ensure_output_dirs(root)
        if args.clean:
            clean_output(root)
        for item in train_items:
            copy_item(item, root, "train", args.allow_negative, args.drop_confidence_column)
        for item in val_items:
            copy_item(item, root, "val", args.allow_negative, args.drop_confidence_column)
        write_data_yaml(root, class_names)
    except RuntimeError as exc:
        print(f"数据集整理失败：{exc}")
        return 2

    print(f"整理完成：train={len(train_items)}, val={len(val_items)}")
    print(f"train split digest: {digest_split(train_items)}")
    print(f"val split digest: {digest_split(val_items)}")
    print(f"输出目录：{root}")
    print(f"data.yaml：{root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
