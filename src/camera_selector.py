from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

try:
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk
except ImportError:
    tk = None
    messagebox = None
    Image = None
    ImageTk = None


if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


@dataclass
class CameraProbe:
    index: int
    width: int
    height: int
    frame_bgr: object


def probe_cameras(max_index: int, backend: str) -> list[CameraProbe]:
    probes: list[CameraProbe] = []
    backend_id = BACKENDS[backend]
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, backend_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        ok = False
        frame = None
        if cap.isOpened():
            time.sleep(0.25)
            for _ in range(8):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)
        if ok and frame is not None:
            height, width = frame.shape[:2]
            cv2.putText(
                frame,
                f"Index {index}  {width}x{height}",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            probes.append(CameraProbe(index=index, width=width, height=height, frame_bgr=frame.copy()))
        cap.release()
        time.sleep(0.25)
    return probes


def parse_preferred_indexes(value: str) -> list[int]:
    indexes: list[int] = []
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError:
            continue
        if index >= 0 and index not in indexes:
            indexes.append(index)
    return indexes


def default_selected_indexes(probes: list[CameraProbe], preferred_indexes: list[int]) -> set[int]:
    available = {probe.index for probe in probes}
    selected = [index for index in preferred_indexes if index in available]
    for probe in probes:
        if len(selected) >= 2:
            break
        if probe.index not in selected:
            selected.append(probe.index)
    return set(selected[:3])


def start_crosscam(
    repo_root: Path,
    script: Path,
    selected_indexes: list[int],
    pipe_mode: bool,
    flip_both: bool,
    show_trails: bool,
) -> None:
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-CameraIndexes",
        ",".join(str(index) for index in selected_indexes),
    ]
    if pipe_mode:
        command.append("-PipeMode")
    if flip_both:
        command.append("-FlipBoth")
    if show_trails:
        command.append("-ShowTrails")

    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(command, cwd=str(repo_root), creationflags=creationflags)


def build_selector(args: argparse.Namespace, probes: list[CameraProbe]) -> int:
    if tk is None:
        print("缺少 tkinter，无法打开摄像头选择窗口。")
        return 1

    repo_root = Path.cwd()
    script = (repo_root / args.script).resolve()

    root = tk.Tk()
    root.title("选择摄像头 - CrossCamReID")
    root.geometry("820x760")

    title = tk.Label(root, text="选择要使用的摄像头", font=("Microsoft YaHei UI", 16, "bold"))
    title.pack(anchor="w", padx=16, pady=(14, 4))

    hint = tk.Label(
        root,
        text="勾选 1 到 3 个摄像头后启动。双摄是当前主要追踪模式；三摄会显示三路画面，可点击任意画面中的检测框注册目标。",
        font=("Microsoft YaHei UI", 10),
        wraplength=760,
        justify="left",
    )
    hint.pack(anchor="w", padx=16, pady=(0, 10))

    options_frame = tk.Frame(root)
    options_frame.pack(fill="x", padx=16, pady=(0, 10))
    pipe_var = tk.BooleanVar(value=args.pipe_mode)
    flip_var = tk.BooleanVar(value=args.flip_both)
    trails_var = tk.BooleanVar(value=args.show_trails)
    tk.Checkbutton(options_frame, text="使用训练后的管子模型 PipeMode", variable=pipe_var).pack(side="left", padx=(0, 18))
    tk.Checkbutton(options_frame, text="启动时水平翻转两路画面", variable=flip_var).pack(side="left", padx=(0, 18))
    tk.Checkbutton(options_frame, text="显示轨迹线", variable=trails_var).pack(side="left")

    list_frame = tk.Frame(root)
    list_frame.pack(fill="both", expand=True, padx=16, pady=8)

    selected_vars: dict[int, tk.BooleanVar] = {}
    default_selected = default_selected_indexes(probes, parse_preferred_indexes(args.preferred_indexes))
    image_refs = []
    if not probes:
        tk.Label(list_frame, text="没有探测到可用摄像头。请检查连接后重新打开。").pack(anchor="w")
    for position, probe in enumerate(probes):
        selected_vars[probe.index] = tk.BooleanVar(value=probe.index in default_selected)
        item = tk.Frame(list_frame, relief="groove", borderwidth=1)
        item.pack(fill="x", pady=6)

        check = tk.Checkbutton(
            item,
            text=f"索引 {probe.index}    {probe.width}x{probe.height}",
            variable=selected_vars[probe.index],
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        check.pack(anchor="w", padx=10, pady=(8, 4))

        if Image is not None and ImageTk is not None:
            rgb = cv2.cvtColor(probe.frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((360, 200))
            photo = ImageTk.PhotoImage(image)
            image_refs.append(photo)
            tk.Label(item, image=photo).pack(anchor="w", padx=10, pady=(0, 10))
    root.image_refs = image_refs

    button_frame = tk.Frame(root)
    button_frame.pack(fill="x", padx=16, pady=14)

    def on_start() -> None:
        selected = [index for index, var in selected_vars.items() if var.get()]
        if not 1 <= len(selected) <= 3:
            messagebox.showwarning("选择数量不正确", "请选择 1 到 3 个摄像头。")
            return
        start_crosscam(repo_root, script, selected, pipe_var.get(), flip_var.get(), trails_var.get())
        root.destroy()

    def on_close() -> None:
        root.destroy()

    tk.Button(button_frame, text="启动", command=on_start, width=14, height=2).pack(side="right", padx=(10, 0))
    tk.Button(button_frame, text="取消", command=on_close, width=10, height=2).pack(side="right")

    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrossCamReID camera selection UI.")
    parser.add_argument("--backend", choices=tuple(BACKENDS.keys()), default="dshow")
    parser.add_argument("--probe-max", type=int, default=10)
    parser.add_argument("--preferred-indexes", default="1,3")
    parser.add_argument("--script", default="scripts/run_crosscam.ps1")
    parser.add_argument("--pipe-mode", action="store_true")
    parser.add_argument("--flip-both", action="store_true")
    parser.add_argument("--show-trails", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probes = probe_cameras(args.probe_max, args.backend)
    if args.probe_only:
        for probe in probes:
            print(f"索引 {probe.index}: 可用，画面={probe.width}x{probe.height}")
        return 0
    return build_selector(args, probes)


if __name__ == "__main__":
    raise SystemExit(main())
