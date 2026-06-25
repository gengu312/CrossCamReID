from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HandoffResult:
    log_path: Path
    total_events: int
    registered_id: Optional[str]
    registered_camera: Optional[str]
    left_index: Optional[int]
    handoff_index: Optional[int]
    handoff_camera: Optional[str]
    target_match_count: int
    cross_camera_ids: list[str]

    @property
    def handoff_success(self) -> bool:
        return (
            self.registered_id is not None
            and self.registered_camera is not None
            and self.left_index is not None
            and self.handoff_index is not None
            and self.handoff_camera is not None
            and self.handoff_camera != self.registered_camera
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CrossCamReID event CSV logs.")
    parser.add_argument("--log", help="Specific event CSV path. Defaults to latest runs/*-events.csv.")
    parser.add_argument("--log-dir", default="runs", help="Directory containing event CSV logs.")
    parser.add_argument(
        "--require-handoff",
        action="store_true",
        help="Exit with code 2 if registered target handoff was not observed.",
    )
    return parser.parse_args()


def latest_log(log_dir: Path) -> Optional[Path]:
    candidates = sorted(log_dir.glob("*-events.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def first_nonempty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def analyze_rows(path: Path, rows: list[dict[str, str]]) -> HandoffResult:
    registered_id: Optional[str] = None
    registered_camera: Optional[str] = None
    left_index: Optional[int] = None
    handoff_index: Optional[int] = None
    handoff_camera: Optional[str] = None
    target_match_count = 0
    seen_cameras_by_id: dict[str, set[str]] = {}

    for index, row in enumerate(rows):
        event_type = row.get("event_type", "")
        global_id = row.get("global_id", "")
        camera = row.get("camera", "")

        if global_id and camera and event_type in {"target_registered", "target_matched", "matched", "track_created"}:
            seen_cameras_by_id.setdefault(global_id, set()).add(camera)

        if event_type == "target_registered" and registered_id is None:
            registered_id = global_id
            registered_camera = camera

        if event_type == "target_matched":
            target_match_count += 1

        if registered_id is None or global_id != registered_id:
            continue

        if event_type == "left" and camera == registered_camera and left_index is None:
            left_index = index
            continue

        if (
            left_index is not None
            and event_type == "target_matched"
            and camera
            and camera != registered_camera
            and handoff_index is None
        ):
            handoff_index = index
            handoff_camera = camera

    cross_camera_ids = sorted(
        global_id for global_id, cameras in seen_cameras_by_id.items() if len(cameras) >= 2
    )
    return HandoffResult(
        log_path=path,
        total_events=len(rows),
        registered_id=registered_id,
        registered_camera=registered_camera,
        left_index=left_index,
        handoff_index=handoff_index,
        handoff_camera=handoff_camera,
        target_match_count=target_match_count,
        cross_camera_ids=cross_camera_ids,
    )


def print_result(result: HandoffResult) -> None:
    print(f"日志文件：{result.log_path}")
    print(f"事件总数：{result.total_events}")
    print(f"注册目标：{first_nonempty(result.registered_id, '未发现')}")
    print(f"注册摄像头：{first_nonempty(result.registered_camera, '未发现')}")
    print(f"目标匹配事件数：{result.target_match_count}")
    if result.cross_camera_ids:
        print(f"跨摄像头出现过的 ID：{', '.join(result.cross_camera_ids)}")
    else:
        print("跨摄像头出现过的 ID：无")

    if result.handoff_success:
        print(
            f"接力结果：成功，{result.registered_id} 从摄像头{result.registered_camera} "
            f"离开后，在摄像头{result.handoff_camera} 接上。"
        )
    else:
        print("接力结果：未确认成功。")
        if result.registered_id is None:
            print("原因：日志里没有 target_registered。")
        elif result.left_index is None:
            print("原因：注册目标没有在原摄像头产生 left 事件。")
        elif result.handoff_index is None:
            print("原因：注册目标离开后，没有在另一个摄像头产生 target_matched。")


def main() -> int:
    args = parse_args()
    log_path = Path(args.log) if args.log else latest_log(Path(args.log_dir))
    if log_path is None:
        print(f"没有找到事件日志：{Path(args.log_dir) / '*-events.csv'}")
        return 2
    if not log_path.exists():
        print(f"日志文件不存在：{log_path}")
        return 2

    rows = read_rows(log_path)
    result = analyze_rows(log_path, rows)
    print_result(result)
    if args.require_handoff and not result.handoff_success:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
