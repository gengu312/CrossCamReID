from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

FIRST_BATCH_SCENARIO_MINIMUMS = {
    "single": 20,
    "stack": 70,
    "hand_move": 30,
    "negative": 20,
}
FIRST_BATCH_CAMERA_MINIMUMS = {
    "cam1": 40,
    "cam2": 40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CrossCamReID capture_manifest.csv.")
    parser.add_argument("--manifest", default="dataset_raw/capture_manifest.csv", help="Capture manifest CSV path.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 if any image path is missing.")
    parser.add_argument("--first-batch-plan", action="store_true", help="Print first-batch capture shortages.")
    parser.add_argument(
        "--require-first-batch",
        action="store_true",
        help="Exit with code 2 if first-batch capture minimums are not met.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_by(rows: list[dict[str, str]], field: str) -> Counter[str]:
    return Counter(row.get(field, "") or "未填写" for row in rows)


def image_exists(raw_path: str, manifest: Path) -> bool:
    path = Path(raw_path)
    if path.exists():
        return True
    if not path.is_absolute() and (manifest.parent / path).exists():
        return True
    return False


def missing_paths(rows: list[dict[str, str]], manifest: Path) -> list[str]:
    missing: list[str] = []
    for row in rows:
        raw_path = row.get("path", "")
        if raw_path and not image_exists(raw_path, manifest):
            missing.append(raw_path)
    return missing


def print_counter(title: str, counter: Counter[str]) -> None:
    if not counter:
        print(f"{title}：无")
        return
    print(f"{title}：")
    for name, count in sorted(counter.items()):
        print(f"  {name}: {count}")


def first_batch_shortages(rows: list[dict[str, str]]) -> list[str]:
    shortages: list[str] = []
    scenario_counts = count_by(rows, "scenario")
    camera_counts = count_by(rows, "camera_label")

    for scenario, minimum in FIRST_BATCH_SCENARIO_MINIMUMS.items():
        count = scenario_counts.get(scenario, 0)
        if count < minimum:
            shortages.append(f"场景 {scenario}: 当前 {count}，建议至少 {minimum}，还差 {minimum - count}")

    for camera_label, minimum in FIRST_BATCH_CAMERA_MINIMUMS.items():
        count = camera_counts.get(camera_label, 0)
        if count < minimum:
            shortages.append(f"摄像头 {camera_label}: 当前 {count}，建议至少 {minimum}，还差 {minimum - count}")

    return shortages


def print_first_batch_plan(rows: list[dict[str, str]], shortages: list[str]) -> None:
    print("第一批采集建议：")
    print("  single >= 20, stack >= 70, hand_move >= 30, negative >= 20")
    print("  cam1 >= 40, cam2 >= 40")
    if not shortages:
        print("  结果：已达到第一批最低采集建议。")
        return
    print("  缺口：")
    for item in shortages:
        print(f"  - {item}")


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"采集记录不存在：{manifest}")
        return 2

    rows = read_manifest(manifest)
    capture_ids = {row.get("capture_id", "") for row in rows if row.get("capture_id", "")}
    missing = missing_paths(rows, manifest)
    shortages = first_batch_shortages(rows)

    print(f"采集记录：{manifest}")
    print(f"图片记录数：{len(rows)}")
    print(f"采集次数：{len(capture_ids)}")
    print(f"缺失图片数：{len(missing)}")
    print_counter("按场景统计", count_by(rows, "scenario"))
    print_counter("按摄像头统计", count_by(rows, "camera_label"))
    print_counter("按动作统计", count_by(rows, "action"))
    if missing:
        print("缺失图片样例：")
        for path in missing[:10]:
            print(f"  {path}")

    if args.first_batch_plan or args.require_first_batch:
        print_first_batch_plan(rows, shortages)

    if args.strict and missing:
        print("采集记录校验：未通过")
        return 2
    if args.require_first_batch and shortages:
        print("第一批采集校验：未通过")
        return 2
    print("采集记录校验：通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
