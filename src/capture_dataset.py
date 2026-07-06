from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from crosscam_mvp import (
    FRAME_H,
    FRAME_W,
    draw_text,
    find_available_camera_indexes,
    open_camera,
    parse_camera_index,
    parse_camera_scan_order,
)


SCENARIO_DIRS = {
    "single": ("cam1_single", "cam2_single"),
    "stack": ("cam1_stack", "cam2_stack"),
    "hand_move": ("cam1_hand_move", "cam2_hand_move"),
    "negative": ("negative", "negative"),
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
MANIFEST_FIELDS = [
    "capture_id",
    "saved_at",
    "scenario",
    "action",
    "camera_label",
    "camera_index",
    "path",
    "view_order",
    "flip_a",
    "flip_b",
    "flip_both",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture dual-camera photos for the pipe YOLO dataset.",
    )
    parser.add_argument("--cam-a", type=parse_camera_index, default=None, help="Camera A index, or auto.")
    parser.add_argument("--cam-b", type=parse_camera_index, default=None, help="Camera B index, or auto.")
    parser.add_argument(
        "--camera-scan-order",
        type=parse_camera_scan_order,
        default=parse_camera_scan_order("1,3,2,0,4,5"),
        help="Preferred camera indexes for auto selection, such as 1,3,2,0,4,5.",
    )
    parser.add_argument("--probe-max", type=int, default=10, help="Max camera index to probe when using auto.")
    parser.add_argument(
        "--backend",
        choices=("auto", "dshow", "msmf", "any"),
        default="dshow",
        help="OpenCV camera backend.",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIO_DIRS.keys()),
        default="stack",
        help="Capture scene type; stack is the main pipe-pile scenario.",
    )
    parser.add_argument("--output-root", default="dataset_raw", help="Raw photo output directory.")
    parser.add_argument("--prefix", default="", help="Optional filename prefix.")
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=0.0,
        help="Automatically save both cameras every N seconds; 0 disables auto capture.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after saving N image pairs; 0 means manual stop.")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print resolved camera indexes and output directories, then exit without opening the capture window.",
    )
    parser.add_argument("--flip-a", action="store_true", help="Horizontally flip camera A before preview and saving.")
    parser.add_argument("--flip-b", action="store_true", help="Horizontally flip camera B before preview and saving.")
    parser.add_argument("--flip-both", action="store_true", help="Horizontally flip both cameras before preview and saving.")
    parser.add_argument("--view-order", choices=("AB", "BA"), default="AB", help="Capture/display order for the two cameras.")
    return parser.parse_args()


def resolve_camera_pair(args: argparse.Namespace) -> list[int]:
    cam_a = args.cam_a
    cam_b = args.cam_b
    fixed_indexes = [index for index in (cam_a, cam_b) if index is not None]

    needed = 2 - len(fixed_indexes)
    if needed > 0:
        available = find_available_camera_indexes(
            args.probe_max,
            args.backend,
            args.camera_scan_order,
            needed=needed + len(fixed_indexes),
        )
        auto_candidates = [index for index in available if index not in fixed_indexes]
        if cam_a is None and auto_candidates:
            cam_a = auto_candidates.pop(0)
        if cam_b is None and auto_candidates:
            cam_b = auto_candidates.pop(0)

    if cam_a is None or cam_b is None:
        raise RuntimeError(
            "自动选择采集摄像头失败：当前可用摄像头少于 2 个。"
            f" 可先运行 .\\run_crosscam.bat -Probe -ProbeMax {args.probe_max} 检查。"
        )
    if cam_a == cam_b:
        raise RuntimeError("两个摄像头索引不能相同。请使用不同索引，或使用 auto。")
    return [cam_a, cam_b]


def capture_order(camera_indexes: list[int], view_order: str) -> list[int]:
    if view_order == "BA" and len(camera_indexes) >= 2:
        return [camera_indexes[1], camera_indexes[0], *camera_indexes[2:]]
    return list(camera_indexes)


def scenario_paths(root: Path, scenario: str) -> tuple[Path, Path]:
    left_name, right_name = SCENARIO_DIRS[scenario]
    return root / left_name, root / right_name


def image_count(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def image_counts(left_dir: Path, right_dir: Path) -> tuple[int, int]:
    left_count = image_count(left_dir)
    if left_dir == right_dir:
        return left_count, left_count
    return left_count, image_count(right_dir)


def existing_images_message(left_dir: Path, right_dir: Path, counts: tuple[int, int] | None = None) -> str:
    left_count, right_count = counts if counts is not None else image_counts(left_dir, right_dir)
    if left_dir == right_dir:
        return f"已有图片：{left_dir}={left_count}"
    return f"已有图片：左={left_count}，右={right_count}"


def new_images_message(left_dir: Path, right_dir: Path, start_counts: tuple[int, int]) -> str:
    end_left, end_right = image_counts(left_dir, right_dir)
    left_delta = max(0, end_left - start_counts[0])
    right_delta = max(0, end_right - start_counts[1])
    if left_dir == right_dir:
        return f"本次新增图片：{left_dir}={left_delta}"
    return f"本次新增图片：左={left_delta}，右={right_delta}"


def ensure_dirs(root: Path, scenario: str) -> tuple[Path, Path]:
    left_dir, right_dir = scenario_paths(root, scenario)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    return left_dir, right_dir


def append_manifest_rows(manifest: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def capture_manifest_rows(
    capture_id: str,
    scenario: str,
    action: str,
    camera_paths: list[tuple[str, int, Optional[Path]]],
    view_order: str,
    flip_a: bool,
    flip_b: bool,
    flip_both: bool,
) -> list[dict[str, str]]:
    saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, str]] = []
    for camera_label, camera_index, path in camera_paths:
        if path is None:
            continue
        rows.append(
            {
                "capture_id": capture_id,
                "saved_at": saved_at,
                "scenario": scenario,
                "action": action,
                "camera_label": camera_label,
                "camera_index": str(camera_index),
                "path": str(path),
                "view_order": view_order,
                "flip_a": str(bool(flip_a)),
                "flip_b": str(bool(flip_b)),
                "flip_both": str(bool(flip_both)),
            }
        )
    return rows


def new_capture_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"


def next_path(directory: Path, prefix: str, camera_label: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    millis = int((time.time() % 1) * 1000)
    safe_prefix = f"{prefix}_" if prefix else ""
    return directory / f"{safe_prefix}{camera_label}_{stamp}-{millis:03d}.jpg"


def save_frame(frame: np.ndarray, directory: Path, prefix: str, camera_label: str) -> Optional[Path]:
    path = next_path(directory, prefix, camera_label)
    ok = cv2.imwrite(str(path), frame)
    return path if ok else None


def apply_capture_flips(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    flip_a: bool,
    flip_b: bool,
    flip_both: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if flip_a or flip_both:
        frame_a = cv2.flip(frame_a, 1)
    if flip_b or flip_both:
        frame_b = cv2.flip(frame_b, 1)
    return frame_a, frame_b


def render_canvas(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    scenario: str,
    saved_pairs: int,
    existing_message: str,
    last_message: str,
) -> np.ndarray:
    left = cv2.resize(frame_a, (FRAME_W, FRAME_H))
    right = cv2.resize(frame_b, (FRAME_W, FRAME_H))
    canvas = np.hstack([left, right])
    panel_h = 136
    panel = np.full((panel_h, canvas.shape[1], 3), (25, 29, 35), dtype=np.uint8)
    draw_text(panel, f"采集场景：{scenario}    已保存组数：{saved_pairs}", (16, 12), (235, 240, 245), 20)
    draw_text(panel, existing_message, (16, 42), (190, 205, 220), 17)
    draw_text(panel, "按 Space/B 保存双摄；按 1 只存左侧；按 2 只存右侧；按 Q 退出。", (16, 70), (190, 205, 220), 17)
    draw_text(panel, last_message, (16, 100), (120, 220, 155), 17)
    cv2.putText(canvas, "Camera A", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Camera B", (FRAME_W + 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack([canvas, panel])


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    cam_a, cam_b = capture_order(resolve_camera_pair(args), args.view_order)[:2]

    if args.print_only:
        left_dir, right_dir = scenario_paths(output_root, args.scenario)
        print("采集配置：")
        print(f"  摄像头：A={cam_a} B={cam_b} 后端={args.backend}")
        print(f"  场景：{args.scenario}")
        print(f"  保存目录：{left_dir} / {right_dir}")
        print(f"  {existing_images_message(left_dir, right_dir)}")
        if args.auto_interval > 0:
            print(f"  自动采集间隔：{args.auto_interval} 秒")
        if args.limit > 0:
            print(f"  保存组数上限：{args.limit}")
        print("PrintOnly：未打开采集窗口，未创建目录或写入图片。")
        return 0

    left_dir, right_dir = ensure_dirs(output_root, args.scenario)

    cap_a, backend_a = open_camera(cam_a, args.backend)
    cap_b, backend_b = open_camera(cam_b, args.backend)
    if cap_a is None or cap_b is None:
        if cap_a is not None:
            cap_a.release()
        if cap_b is not None:
            cap_b.release()
        raise RuntimeError("无法同时打开两个摄像头。请先关闭占用摄像头的软件，或检查 --cam-a / --cam-b。")

    print(f"已打开摄像头：A={cam_a}({backend_a}) B={cam_b}({backend_b})")
    print(f"保存目录：{left_dir} / {right_dir}")
    manifest = output_root / "capture_manifest.csv"
    print(f"采集记录：{manifest}")
    start_counts = image_counts(left_dir, right_dir)
    existing_message = existing_images_message(left_dir, right_dir, start_counts)
    print(existing_message)

    window = "CrossCamReID Dataset Capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    saved_pairs = 0
    last_auto = time.time()
    last_message = "准备采集。优先多拍堆叠、遮挡、只露端面或侧面的管子。"

    try:
        while True:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or not ok_b:
                print("有一个摄像头停止输出画面。")
                return 2
            frame_a, frame_b = apply_capture_flips(frame_a, frame_b, args.flip_a, args.flip_b, args.flip_both)

            now = time.time()
            should_auto_save = args.auto_interval > 0 and now - last_auto >= args.auto_interval
            if should_auto_save:
                capture_id = new_capture_id()
                path_a = save_frame(frame_a, left_dir, args.prefix, "cam1")
                path_b = save_frame(frame_b, right_dir, args.prefix, "cam2")
                append_manifest_rows(
                    manifest,
                    capture_manifest_rows(
                        capture_id,
                        args.scenario,
                        "auto_pair",
                        [("cam1", cam_a, path_a), ("cam2", cam_b, path_b)],
                        args.view_order,
                        args.flip_a,
                        args.flip_b,
                        args.flip_both,
                    ),
                )
                saved_pairs += 1
                last_auto = now
                last_message = f"自动保存：{path_a} / {path_b}"

            cv2.imshow(
                window,
                render_canvas(frame_a, frame_b, args.scenario, saved_pairs, existing_message, last_message),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord(" "), ord("b"), ord("B")):
                capture_id = new_capture_id()
                path_a = save_frame(frame_a, left_dir, args.prefix, "cam1")
                path_b = save_frame(frame_b, right_dir, args.prefix, "cam2")
                append_manifest_rows(
                    manifest,
                    capture_manifest_rows(
                        capture_id,
                        args.scenario,
                        "manual_pair",
                        [("cam1", cam_a, path_a), ("cam2", cam_b, path_b)],
                        args.view_order,
                        args.flip_a,
                        args.flip_b,
                        args.flip_both,
                    ),
                )
                saved_pairs += 1
                last_message = f"保存双摄：{path_a} / {path_b}"
            elif key == ord("1"):
                capture_id = new_capture_id()
                path_a = save_frame(frame_a, left_dir, args.prefix, "cam1")
                append_manifest_rows(
                    manifest,
                    capture_manifest_rows(
                        capture_id,
                        args.scenario,
                        "manual_left",
                        [("cam1", cam_a, path_a)],
                        args.view_order,
                        args.flip_a,
                        args.flip_b,
                        args.flip_both,
                    ),
                )
                last_message = f"保存左侧：{path_a}"
            elif key == ord("2"):
                capture_id = new_capture_id()
                path_b = save_frame(frame_b, right_dir, args.prefix, "cam2")
                append_manifest_rows(
                    manifest,
                    capture_manifest_rows(
                        capture_id,
                        args.scenario,
                        "manual_right",
                        [("cam2", cam_b, path_b)],
                        args.view_order,
                        args.flip_a,
                        args.flip_b,
                        args.flip_both,
                    ),
                )
                last_message = f"保存右侧：{path_b}"

            if args.limit > 0 and saved_pairs >= args.limit:
                break
    finally:
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()

    print(f"采集结束，已保存双摄组数：{saved_pairs}")
    print(new_images_message(left_dir, right_dir, start_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
