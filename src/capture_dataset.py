from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from crosscam_mvp import FRAME_H, FRAME_W, draw_text, open_camera


SCENARIO_DIRS = {
    "single": ("cam1_single", "cam2_single"),
    "stack": ("cam1_stack", "cam2_stack"),
    "hand_move": ("cam1_hand_move", "cam2_hand_move"),
    "negative": ("negative", "negative"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture dual-camera photos for the pipe YOLO dataset.",
    )
    parser.add_argument("--cam-a", type=int, default=0, help="Camera A index.")
    parser.add_argument("--cam-b", type=int, default=2, help="Camera B index.")
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
    return parser.parse_args()


def ensure_dirs(root: Path, scenario: str) -> tuple[Path, Path]:
    left_name, right_name = SCENARIO_DIRS[scenario]
    left_dir = root / left_name
    right_dir = root / right_name
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    return left_dir, right_dir


def next_path(directory: Path, prefix: str, camera_label: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    millis = int((time.time() % 1) * 1000)
    safe_prefix = f"{prefix}_" if prefix else ""
    return directory / f"{safe_prefix}{camera_label}_{stamp}-{millis:03d}.jpg"


def save_frame(frame: np.ndarray, directory: Path, prefix: str, camera_label: str) -> Optional[Path]:
    path = next_path(directory, prefix, camera_label)
    ok = cv2.imwrite(str(path), frame)
    return path if ok else None


def render_canvas(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    scenario: str,
    saved_pairs: int,
    last_message: str,
) -> np.ndarray:
    left = cv2.resize(frame_a, (FRAME_W, FRAME_H))
    right = cv2.resize(frame_b, (FRAME_W, FRAME_H))
    canvas = np.hstack([left, right])
    panel_h = 112
    panel = np.full((panel_h, canvas.shape[1], 3), (25, 29, 35), dtype=np.uint8)
    draw_text(panel, f"采集场景：{scenario}    已保存组数：{saved_pairs}", (16, 12), (235, 240, 245), 20)
    draw_text(panel, "按 Space/B 保存双摄；按 1 只存左侧；按 2 只存右侧；按 Q 退出。", (16, 42), (190, 205, 220), 17)
    draw_text(panel, last_message, (16, 72), (120, 220, 155), 17)
    cv2.putText(canvas, "Camera A", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Camera B", (FRAME_W + 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack([canvas, panel])


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    left_dir, right_dir = ensure_dirs(output_root, args.scenario)

    cap_a, backend_a = open_camera(args.cam_a, args.backend)
    cap_b, backend_b = open_camera(args.cam_b, args.backend)
    if cap_a is None or cap_b is None:
        if cap_a is not None:
            cap_a.release()
        if cap_b is not None:
            cap_b.release()
        raise RuntimeError("无法同时打开两个摄像头。请先关闭占用摄像头的软件，或检查 --cam-a / --cam-b。")

    print(f"已打开摄像头：A={args.cam_a}({backend_a}) B={args.cam_b}({backend_b})")
    print(f"保存目录：{left_dir} / {right_dir}")

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

            now = time.time()
            should_auto_save = args.auto_interval > 0 and now - last_auto >= args.auto_interval
            if should_auto_save:
                path_a = save_frame(frame_a, left_dir, args.prefix, "cam1")
                path_b = save_frame(frame_b, right_dir, args.prefix, "cam2")
                saved_pairs += 1
                last_auto = now
                last_message = f"自动保存：{path_a} / {path_b}"

            cv2.imshow(window, render_canvas(frame_a, frame_b, args.scenario, saved_pairs, last_message))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord(" "), ord("b"), ord("B")):
                path_a = save_frame(frame_a, left_dir, args.prefix, "cam1")
                path_b = save_frame(frame_b, right_dir, args.prefix, "cam2")
                saved_pairs += 1
                last_message = f"保存双摄：{path_a} / {path_b}"
            elif key == ord("1"):
                path_a = save_frame(frame_a, left_dir, args.prefix, "cam1")
                last_message = f"保存左侧：{path_a}"
            elif key == ord("2"):
                path_b = save_frame(frame_b, right_dir, args.prefix, "cam2")
                last_message = f"保存右侧：{path_b}"

            if args.limit > 0 and saved_pairs >= args.limit:
                break
    finally:
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()

    print(f"采集结束，已保存双摄组数：{saved_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
