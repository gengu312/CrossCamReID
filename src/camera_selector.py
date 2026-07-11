from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

try:
    from detector_backend_status import collect_backend_statuses, selector_status_text
except ImportError:
    from src.detector_backend_status import collect_backend_statuses, selector_status_text

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

DETECTORS = ("motion", "yolo", "rfdetr")
PIPE_DETECTORS = ("yolo", "rfdetr")
CAPTURE_SCENARIOS = ("stack", "single", "hand_move", "negative")
RFDETR_SIZES = ("nano", "small", "base", "medium", "large", "xlarge", "2xlarge")


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


def ordered_selected_indexes(selected_indexes: list[int], preferred_indexes: list[int], probes: list[CameraProbe]) -> list[int]:
    selected = set(selected_indexes)
    ordered: list[int] = []
    for index in preferred_indexes:
        if index in selected and index not in ordered:
            ordered.append(index)
    for probe in probes:
        if probe.index in selected and probe.index not in ordered:
            ordered.append(probe.index)
    return ordered


def start_crosscam(
    repo_root: Path,
    script: Path,
    selected_indexes: list[int],
    backend: str,
    pipe_mode: bool,
    flip_both: bool,
    show_trails: bool,
    view_order: str,
    extra_args: list[str],
) -> None:
    command = build_crosscam_command(
        script,
        selected_indexes,
        backend,
        pipe_mode,
        flip_both,
        show_trails,
        view_order,
        extra_args,
    )
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(command, cwd=str(repo_root), creationflags=creationflags)


def start_capture(
    repo_root: Path,
    script: Path,
    selected_indexes: list[int],
    backend: str,
    scenario: str,
    output_root: str,
    flip_both: bool,
    view_order: str,
) -> None:
    command = build_capture_command(script, selected_indexes, backend, scenario, output_root, flip_both, view_order)
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(command, cwd=str(repo_root), creationflags=creationflags)


def build_crosscam_command(
    script: Path,
    selected_indexes: list[int],
    backend: str,
    pipe_mode: bool,
    flip_both: bool,
    show_trails: bool,
    view_order: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Backend",
        backend,
        "-ViewOrder",
        view_order,
    ]
    if selected_indexes:
        command.extend(["-CameraIndexes", ",".join(str(index) for index in selected_indexes)])
    command.extend(extra_args or [])
    if pipe_mode:
        command.append("-PipeMode")
    if flip_both:
        command.append("-FlipBoth")
    if show_trails:
        command.append("-ShowTrails")
    return command


def build_capture_command(
    script: Path,
    selected_indexes: list[int],
    backend: str,
    scenario: str,
    output_root: str,
    flip_both: bool = False,
    view_order: str = "AB",
) -> list[str]:
    capture_indexes = capture_order(selected_indexes, view_order)
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Backend",
        backend,
        "-Scenario",
        scenario,
        "-OutputRoot",
        output_root,
    ]
    if len(capture_indexes) >= 1:
        command.extend(["-CamA", str(capture_indexes[0])])
    if len(capture_indexes) >= 2:
        command.extend(["-CamB", str(capture_indexes[1])])
    if flip_both:
        command.append("-FlipBoth")
    return command


def capture_order(selected_indexes: list[int], view_order: str) -> list[int]:
    if view_order == "BA" and len(selected_indexes) >= 2:
        return [selected_indexes[1], selected_indexes[0], *selected_indexes[2:]]
    return list(selected_indexes)


def extra_arg_value(extra_args: list[str], name: str, default: str) -> str:
    key = name.lower()
    for index, value in enumerate(extra_args[:-1]):
        if value.lower() == key:
            return extra_args[index + 1]
    return default


def set_extra_arg_pair(extra_args: list[str], name: str, value: str) -> list[str]:
    key = name.lower()
    cleaned: list[str] = []
    index = 0
    while index < len(extra_args):
        if extra_args[index].lower() == key:
            index += 2
            continue
        cleaned.append(extra_args[index])
        index += 1
    cleaned.extend([name, value])
    return cleaned


def set_optional_extra_arg_pair(extra_args: list[str], name: str, value: str) -> list[str]:
    key = name.lower()
    cleaned: list[str] = []
    index = 0
    while index < len(extra_args):
        if extra_args[index].lower() == key:
            index += 2
            continue
        cleaned.append(extra_args[index])
        index += 1
    value = value.strip()
    if value:
        cleaned.extend([name, value])
    return cleaned


def extra_arg_flag(extra_args: list[str], name: str) -> bool:
    key = name.lower()
    return any(value.lower() == key for value in extra_args)


def set_extra_arg_flag(extra_args: list[str], name: str, enabled: bool) -> list[str]:
    key = name.lower()
    cleaned = [value for value in extra_args if value.lower() != key]
    if enabled:
        cleaned.append(name)
    return cleaned


def detector_for_selector(extra_args: list[str], pipe_mode: bool) -> str:
    detector = extra_arg_value(extra_args, "-Detector", "motion")
    if detector not in DETECTORS:
        detector = "motion"
    if pipe_mode and detector not in PIPE_DETECTORS:
        return "yolo"
    return detector


def selection_error(selected_count: int, require_handoff: bool, fallback_demo: bool = False) -> str | None:
    if selected_count == 0 and fallback_demo and not require_handoff:
        return None
    if not 1 <= selected_count <= 3:
        return "请选择 1 到 3 个摄像头。"
    if require_handoff and selected_count < 2:
        return "接力成功验收至少需要 2 个摄像头。"
    return None


def target_review_flags(
    target_lock_gate: bool,
    collect_target_samples: bool,
    target_sample_preview: bool,
) -> tuple[bool, bool]:
    if target_lock_gate:
        return True, True
    return collect_target_samples, target_sample_preview


def selector_extra_args(
    extra_args: list[str],
    detector: str,
    fallback_demo: bool,
    analyze_after_run: bool,
    require_handoff: bool,
    target_lock_gate: bool,
    collect_target_samples: bool,
    target_sample_preview: bool,
    rfdetr_size: str | None = None,
    rfdetr_weights: str | None = None,
    rfdetr_num_classes: str | None = None,
    rfdetr_conf: str | None = None,
) -> list[str]:
    collect_target_samples, target_sample_preview = target_review_flags(
        target_lock_gate,
        collect_target_samples,
        target_sample_preview,
    )
    updated = set_extra_arg_pair(extra_args, "-Detector", detector)
    updated = set_extra_arg_flag(updated, "-FallbackDemo", fallback_demo)
    updated = set_extra_arg_flag(updated, "-AnalyzeAfterRun", analyze_after_run or require_handoff or target_lock_gate)
    updated = set_extra_arg_flag(updated, "-AnalyzeRequireHandoff", require_handoff)
    updated = set_extra_arg_flag(updated, "-AnalyzeTargetLockGate", target_lock_gate)
    updated = set_extra_arg_flag(updated, "-CollectTargetSamplesAfterRun", collect_target_samples)
    updated = set_extra_arg_flag(updated, "-TargetSamplePreview", target_sample_preview)
    if rfdetr_size is not None:
        updated = set_extra_arg_pair(updated, "-RfDetrSize", rfdetr_size)
    if rfdetr_weights is not None:
        updated = set_optional_extra_arg_pair(updated, "-RfDetrWeights", rfdetr_weights)
    if rfdetr_num_classes is not None:
        updated = set_extra_arg_pair(updated, "-RfDetrNumClasses", rfdetr_num_classes)
    if rfdetr_conf is not None:
        updated = set_extra_arg_pair(updated, "-RfDetrConf", rfdetr_conf)
    return updated


def mousewheel_units(delta: int, button_num: int | None = None) -> int:
    if button_num == 4:
        return -1
    if button_num == 5:
        return 1
    if delta == 0:
        return 0
    units = int(-delta / 120)
    if units == 0:
        return -1 if delta > 0 else 1
    return units


def build_selector(args: argparse.Namespace, probes: list[CameraProbe]) -> int:
    if tk is None:
        print("缺少 tkinter，无法打开摄像头选择窗口。")
        return 1

    repo_root = Path.cwd()
    script = (repo_root / args.script).resolve()
    capture_script = (repo_root / args.capture_script).resolve()

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
    swap_var = tk.BooleanVar(value=args.view_order == "BA")
    fallback_var = tk.BooleanVar(value=extra_arg_flag(args.extra_arg, "-FallbackDemo"))
    analyze_var = tk.BooleanVar(value=extra_arg_flag(args.extra_arg, "-AnalyzeAfterRun"))
    require_handoff_var = tk.BooleanVar(value=extra_arg_flag(args.extra_arg, "-AnalyzeRequireHandoff"))
    target_lock_gate_var = tk.BooleanVar(value=extra_arg_flag(args.extra_arg, "-AnalyzeTargetLockGate"))
    collect_samples_var = tk.BooleanVar(value=extra_arg_flag(args.extra_arg, "-CollectTargetSamplesAfterRun"))
    sample_preview_var = tk.BooleanVar(value=extra_arg_flag(args.extra_arg, "-TargetSamplePreview"))
    detector_var = tk.StringVar(value=detector_for_selector(args.extra_arg, args.pipe_mode))
    rfdetr_size_var = tk.StringVar(value=extra_arg_value(args.extra_arg, "-RfDetrSize", "nano"))
    rfdetr_weights_var = tk.StringVar(value=extra_arg_value(args.extra_arg, "-RfDetrWeights", ""))
    rfdetr_num_classes_var = tk.StringVar(value=extra_arg_value(args.extra_arg, "-RfDetrNumClasses", "0"))
    rfdetr_conf_var = tk.StringVar(value=extra_arg_value(args.extra_arg, "-RfDetrConf", "0.35"))
    capture_scenario_var = tk.StringVar(value=args.capture_scenario)
    capture_output_root_var = tk.StringVar(value=args.capture_output_root)
    backend_status_var = tk.StringVar()

    detector_frame = tk.Frame(options_frame)
    detector_frame.pack(fill="x", pady=(0, 6))
    tk.Label(detector_frame, text="检测后端").pack(side="left", padx=(0, 6))
    tk.OptionMenu(detector_frame, detector_var, *DETECTORS).pack(side="left", padx=(0, 18))
    tk.Checkbutton(detector_frame, text="使用训练后的管子模型 PipeMode", variable=pipe_var).pack(side="left")

    rfdetr_frame = tk.Frame(options_frame)
    rfdetr_frame.pack(fill="x", pady=(0, 6))
    tk.Label(rfdetr_frame, text="RF-DETR").pack(side="left", padx=(0, 6))
    tk.OptionMenu(rfdetr_frame, rfdetr_size_var, *RFDETR_SIZES).pack(side="left", padx=(0, 10))
    tk.Label(rfdetr_frame, text="权重").pack(side="left", padx=(0, 6))
    tk.Entry(rfdetr_frame, textvariable=rfdetr_weights_var, width=28).pack(side="left", padx=(0, 10))
    tk.Label(rfdetr_frame, text="类别数").pack(side="left", padx=(0, 6))
    tk.Entry(rfdetr_frame, textvariable=rfdetr_num_classes_var, width=6).pack(side="left", padx=(0, 10))
    tk.Label(rfdetr_frame, text="置信度").pack(side="left", padx=(0, 6))
    tk.Entry(rfdetr_frame, textvariable=rfdetr_conf_var, width=6).pack(side="left")

    backend_status_label = tk.Label(
        options_frame,
        textvariable=backend_status_var,
        font=("Microsoft YaHei UI", 9),
        wraplength=760,
        justify="left",
    )
    backend_status_label.pack(fill="x", pady=(0, 6))

    display_frame = tk.Frame(options_frame)
    display_frame.pack(fill="x", pady=(0, 6))
    tk.Checkbutton(display_frame, text="启动时水平翻转两路画面", variable=flip_var).pack(side="left", padx=(0, 18))
    tk.Checkbutton(display_frame, text="交换左右显示", variable=swap_var).pack(side="left", padx=(0, 18))
    tk.Checkbutton(display_frame, text="显示轨迹线", variable=trails_var).pack(side="left")

    run_frame = tk.Frame(options_frame)
    run_frame.pack(fill="x")
    tk.Checkbutton(run_frame, text="摄像头打开失败时切到演示兜底", variable=fallback_var).pack(side="left", padx=(0, 18))
    tk.Checkbutton(run_frame, text="退出后自动分析日志", variable=analyze_var).pack(side="left", padx=(0, 18))
    tk.Checkbutton(
        run_frame,
        text="要求跨摄像头接力成功",
        variable=require_handoff_var,
    ).pack(side="left", padx=(0, 18))
    tk.Checkbutton(run_frame, text="目标锁定质量门槛", variable=target_lock_gate_var).pack(side="left")

    review_frame = tk.Frame(options_frame)
    review_frame.pack(fill="x", pady=(6, 0))
    tk.Checkbutton(review_frame, text="退出后整理目标样本", variable=collect_samples_var).pack(side="left", padx=(0, 18))
    tk.Checkbutton(review_frame, text="生成样本预览图", variable=sample_preview_var).pack(side="left")

    capture_frame = tk.Frame(options_frame)
    capture_frame.pack(fill="x", pady=(6, 0))
    tk.Label(capture_frame, text="采集场景").pack(side="left", padx=(0, 6))
    tk.OptionMenu(capture_frame, capture_scenario_var, *CAPTURE_SCENARIOS).pack(side="left", padx=(0, 18))
    tk.Label(capture_frame, text="保存目录").pack(side="left", padx=(0, 6))
    tk.Entry(capture_frame, textvariable=capture_output_root_var, width=32).pack(side="left")

    def update_backend_status(*_args) -> None:
        statuses = collect_backend_statuses(
            repo_root,
            yolo_model=extra_arg_value(args.extra_arg, "-YoloModel", ""),
            rfdetr_size=rfdetr_size_var.get(),
            rfdetr_weights=rfdetr_weights_var.get().strip(),
        )
        status = statuses[detector_var.get()]
        backend_status_var.set(selector_status_text(status))
        backend_status_label.configure(fg="#176b3a" if status.project_ready else "#9a5a00")

    def on_pipe_mode_change(*_args) -> None:
        if pipe_var.get() and detector_var.get() not in PIPE_DETECTORS:
            detector_var.set("yolo")
        update_backend_status()

    def on_detector_change(*_args) -> None:
        if detector_var.get() not in PIPE_DETECTORS and pipe_var.get():
            pipe_var.set(False)
        update_backend_status()

    def on_target_lock_gate_change(*_args) -> None:
        if target_lock_gate_var.get():
            collect_samples_var.set(True)
            sample_preview_var.set(True)

    pipe_var.trace_add("write", on_pipe_mode_change)
    detector_var.trace_add("write", on_detector_change)
    rfdetr_size_var.trace_add("write", update_backend_status)
    rfdetr_weights_var.trace_add("write", update_backend_status)
    target_lock_gate_var.trace_add("write", on_target_lock_gate_change)
    update_backend_status()

    list_outer = tk.Frame(root)
    list_outer.pack(fill="both", expand=True, padx=16, pady=8)
    list_canvas = tk.Canvas(list_outer, highlightthickness=0)
    list_scrollbar = tk.Scrollbar(list_outer, orient="vertical", command=list_canvas.yview)
    list_frame = tk.Frame(list_canvas)
    list_window = list_canvas.create_window((0, 0), window=list_frame, anchor="nw")

    def on_list_configure(_event) -> None:
        list_canvas.configure(scrollregion=list_canvas.bbox("all"))

    def on_canvas_configure(event) -> None:
        list_canvas.itemconfigure(list_window, width=event.width)

    def on_mousewheel(event) -> None:
        units = mousewheel_units(getattr(event, "delta", 0), getattr(event, "num", None))
        if units:
            list_canvas.yview_scroll(units, "units")

    def bind_mousewheel(_event) -> None:
        root.bind_all("<MouseWheel>", on_mousewheel)
        root.bind_all("<Button-4>", on_mousewheel)
        root.bind_all("<Button-5>", on_mousewheel)

    def unbind_mousewheel(_event) -> None:
        root.unbind_all("<MouseWheel>")
        root.unbind_all("<Button-4>")
        root.unbind_all("<Button-5>")

    list_frame.bind("<Configure>", on_list_configure)
    list_canvas.bind("<Configure>", on_canvas_configure)
    list_outer.bind("<Enter>", bind_mousewheel)
    list_outer.bind("<Leave>", unbind_mousewheel)
    list_canvas.configure(yscrollcommand=list_scrollbar.set)
    list_canvas.pack(side="left", fill="both", expand=True)
    list_scrollbar.pack(side="right", fill="y")

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
        selected = ordered_selected_indexes(
            [index for index, var in selected_vars.items() if var.get()],
            parse_preferred_indexes(args.preferred_indexes),
            probes,
        )
        error = selection_error(len(selected), require_handoff_var.get(), fallback_var.get())
        if error is not None:
            messagebox.showwarning("选择数量不正确", error)
            return
        view_order = "BA" if swap_var.get() else "AB"
        extra_args = selector_extra_args(
            args.extra_arg,
            detector_var.get(),
            fallback_var.get(),
            analyze_var.get(),
            require_handoff_var.get(),
            target_lock_gate_var.get(),
            collect_samples_var.get(),
            sample_preview_var.get(),
            rfdetr_size_var.get(),
            rfdetr_weights_var.get(),
            rfdetr_num_classes_var.get(),
            rfdetr_conf_var.get(),
        )
        start_crosscam(
            repo_root,
            script,
            selected,
            args.backend,
            pipe_var.get(),
            flip_var.get(),
            trails_var.get(),
            view_order,
            extra_args,
        )
        root.destroy()

    def on_capture() -> None:
        selected = ordered_selected_indexes(
            [index for index, var in selected_vars.items() if var.get()],
            parse_preferred_indexes(args.preferred_indexes),
            probes,
        )
        if len(selected) != 2:
            messagebox.showwarning("选择数量不正确", "采集训练照片需要刚好选择 2 个摄像头。")
            return
        output_root = capture_output_root_var.get().strip()
        if not output_root:
            messagebox.showwarning("保存目录为空", "采集照片保存目录不能为空。")
            return
        view_order = "BA" if swap_var.get() else "AB"
        start_capture(
            repo_root,
            capture_script,
            selected,
            args.backend,
            capture_scenario_var.get(),
            output_root,
            flip_var.get(),
            view_order,
        )
        root.destroy()

    def on_close() -> None:
        root.destroy()

    tk.Button(button_frame, text="启动", command=on_start, width=14, height=2).pack(side="right", padx=(10, 0))
    tk.Button(button_frame, text="采集训练照片", command=on_capture, width=14, height=2).pack(side="right", padx=(10, 0))
    tk.Button(button_frame, text="取消", command=on_close, width=10, height=2).pack(side="right")

    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrossCamReID camera selection UI.")
    parser.add_argument("--backend", choices=tuple(BACKENDS.keys()), default="dshow")
    parser.add_argument("--probe-max", type=int, default=10)
    parser.add_argument("--preferred-indexes", default="1,3")
    parser.add_argument("--script", default="scripts/run_crosscam.ps1")
    parser.add_argument("--capture-script", default="scripts/capture_dataset.ps1")
    parser.add_argument("--capture-scenario", choices=CAPTURE_SCENARIOS, default="stack")
    parser.add_argument("--capture-output-root", default="dataset_raw")
    parser.add_argument("--pipe-mode", action="store_true")
    parser.add_argument("--flip-both", action="store_true")
    parser.add_argument("--show-trails", action="store_true")
    parser.add_argument("--view-order", choices=("AB", "BA"), default="AB")
    parser.add_argument("--extra-arg", action="append", default=[])
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
