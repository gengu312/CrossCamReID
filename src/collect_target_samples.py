from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


REQUIRED_FIELDS = {"time", "camera", "source", "target_similarity", "image"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect saved registered-target samples for review.")
    parser.add_argument("--samples-csv", default="runs/targets/target_samples.csv", help="target_samples.csv path.")
    parser.add_argument("--output-dir", default="runs/target_sample_review", help="Output review directory.")
    parser.add_argument("--max-count", type=int, default=120, help="Max samples to copy.")
    parser.add_argument("--source", choices=("all", "register", "match"), default="all", help="Sample source filter.")
    parser.add_argument("--min-similarity", type=float, help="Only copy samples with target_similarity at least this value.")
    parser.add_argument("--min-count", type=int, help="Exit with code 2 if copied samples are below this number.")
    parser.add_argument("--clean", action="store_true", help="Remove old generated review outputs first.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 if any selected sample image is missing.")
    parser.add_argument("--preview", action="store_true", help="Generate a contact-sheet preview image.")
    parser.add_argument("--preview-cols", type=int, default=4, help="Columns in the contact-sheet preview.")
    return parser.parse_args()


def parse_optional_float(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"目标样本索引不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} 缺少字段：{', '.join(sorted(missing))}")
        return list(reader)


def resolve_image_path(raw_path: str, csv_path: Path) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    if image_path.exists():
        return image_path
    return csv_path.parent / image_path


def clean_generated_outputs(output_dir: Path) -> None:
    output_root = output_dir.resolve()
    images_dir = (output_dir / "images").resolve()
    if output_root not in images_dir.parents:
        raise RuntimeError(f"拒绝清理异常路径：{images_dir}")
    if images_dir.exists():
        shutil.rmtree(images_dir)
    summary_csv = output_dir / "target_samples.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    preview_image = output_dir / "target_samples_preview.jpg"
    if preview_image.exists():
        preview_image.unlink()


def selected_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        source = row.get("source", "")
        if args.source != "all" and source != args.source:
            continue
        similarity = parse_optional_float(row.get("target_similarity", ""))
        if args.min_similarity is not None and (similarity is None or similarity < args.min_similarity):
            continue
        selected.append(row)
        if len(selected) >= max(0, args.max_count):
            break
    return selected


def safe_part(value: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value.strip())
    return cleaned or fallback


def output_image_path(output_dir: Path, row: dict[str, str], index: int, source_image: Path) -> Path:
    source = safe_part(row.get("source", ""), "sample")
    camera = safe_part(row.get("camera", ""), "cam")
    name = f"{index:04d}_cam{camera}_{source}_{source_image.name}"
    return output_dir / "images" / source / name


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time", "camera", "source", "target_similarity", "source_image", "copied_image"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_thumbnail(image_path: Path, size: int, label: str) -> Optional[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tile = np.full((size + 28, size, 3), 245, dtype=np.uint8)
    x = (size - resized_w) // 2
    y = (size - resized_h) // 2
    tile[y : y + resized_h, x : x + resized_w] = resized
    cv2.putText(tile, label[:28], (6, size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return tile


def write_preview(path: Path, rows: list[dict[str, str]], cols: int, thumb_size: int = 160) -> bool:
    if not rows:
        return False
    cols = max(1, min(cols, len(rows)))
    tiles: list[np.ndarray] = []
    for row in rows:
        label = f"cam{row['camera']} {row['source']} s={row['target_similarity']}"
        thumb = make_thumbnail(Path(row["copied_image"]), thumb_size, label)
        if thumb is not None:
            tiles.append(thumb)
    if not tiles:
        return False

    tile_h, tile_w = tiles[0].shape[:2]
    row_count = (len(tiles) + cols - 1) // cols
    sheet = np.full((row_count * tile_h, cols * tile_w, 3), 235, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        y = row * tile_h
        x = col * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), sheet))


def main() -> int:
    args = parse_args()
    samples_csv = Path(args.samples_csv)
    output_dir = Path(args.output_dir)

    if args.clean and output_dir.exists():
        try:
            clean_generated_outputs(output_dir)
        except RuntimeError as exc:
            print(f"目标样本收集失败：{exc}")
            return 2

    try:
        rows = selected_rows(read_rows(samples_csv), args)
    except RuntimeError as exc:
        print(f"目标样本收集失败：{exc}")
        return 2

    copied_rows: list[dict[str, str]] = []
    missing_images = 0
    for index, row in enumerate(rows, start=1):
        raw_image = row.get("image", "")
        source_image = resolve_image_path(raw_image, samples_csv)
        if not source_image.exists():
            missing_images += 1
            continue
        copied_image = output_image_path(output_dir, row, index, source_image)
        copied_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, copied_image)
        copied_rows.append(
            {
                "time": row.get("time", ""),
                "camera": row.get("camera", ""),
                "source": row.get("source", ""),
                "target_similarity": row.get("target_similarity", ""),
                "source_image": str(source_image),
                "copied_image": str(copied_image),
            }
        )

    write_summary(output_dir / "target_samples.csv", copied_rows)
    preview_path = output_dir / "target_samples_preview.jpg"
    preview_written = write_preview(preview_path, copied_rows, args.preview_cols) if args.preview else False

    print(f"目标样本输出目录：{output_dir}")
    print(f"样本记录数：{len(rows)}，已复制：{len(copied_rows)}，缺失图片：{missing_images}")
    print(f"汇总 CSV：{output_dir / 'target_samples.csv'}")
    if args.preview:
        print(f"预览拼图：{preview_path if preview_written else '未生成'}")

    if args.strict and missing_images:
        print("目标样本校验：未通过")
        return 2
    if args.min_count is not None and len(copied_rows) < args.min_count:
        print(f"目标样本校验：未通过，已复制 {len(copied_rows)} 低于 {args.min_count}")
        return 2
    print("目标样本校验：通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
