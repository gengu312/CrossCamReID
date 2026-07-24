from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
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
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
except ImportError:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None
    Image = None
    ImageTk = None


if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}

DETECTORS = ("motion", "yolo", "rfdetr", "hybrid")
PIPE_DETECTORS = ("yolo", "rfdetr", "hybrid")
CAPTURE_SCENARIOS = ("stack", "single", "hand_move", "negative")
RFDETR_SIZES = ("nano", "small", "base", "medium", "large", "xlarge", "2xlarge")
DETECTOR_LABELS = {
    "motion": "运动检测（演示）",
    "yolo": "YOLO 管子识别（推荐）",
    "rfdetr": "RF-DETR 单独检测（实验）",
    "hybrid": "YOLO + RF-DETR 补检（实验）",
}
DETECTOR_DESCRIPTIONS = {
    "motion": "只检测发生移动的区域，适合验证摄像头和基础流程。",
    "yolo": "两个摄像头依次使用同一份 YOLO 模型，适合日常识别与演示。",
    "rfdetr": "两个摄像头依次使用同一份 RF-DETR 模型，主要用于效果对比。",
    "hybrid": "YOLO 负责每帧检测；注册目标漏检时，RF-DETR 才按间隔补检。",
}
BACKEND_LABELS = {
    "dshow": "Windows 推荐（DirectShow）",
    "msmf": "Windows 兼容（MSMF）",
    "any": "自动选择",
}
CAPTURE_SCENARIO_LABELS = {
    "stack": "堆叠场景",
    "single": "单根目标",
    "hand_move": "手持移动",
    "negative": "无目标负样本",
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


def detector_label(detector: str) -> str:
    return DETECTOR_LABELS.get(detector, detector)


def detector_from_label(label: str) -> str:
    for detector, detector_text in DETECTOR_LABELS.items():
        if detector_text == label:
            return detector
    return "motion"


def backend_from_label(label: str) -> str:
    for backend, backend_text in BACKEND_LABELS.items():
        if backend_text == label:
            return backend
    return "dshow"


def capture_scenario_from_label(label: str) -> str:
    for scenario, scenario_text in CAPTURE_SCENARIO_LABELS.items():
        if scenario_text == label:
            return scenario
    return "stack"


def validate_detector_settings(
    detector: str,
    rfdetr_num_classes: str,
    rfdetr_conf: str,
    hybrid_fallback_interval: str,
) -> str | None:
    if detector not in ("rfdetr", "hybrid"):
        return None
    try:
        num_classes = int(rfdetr_num_classes)
    except ValueError:
        return "识别类别数必须是整数。"
    if num_classes < 0:
        return "识别类别数不能小于 0。"
    try:
        confidence = float(rfdetr_conf)
    except ValueError:
        return "最低置信度必须是数字。"
    if not 0.0 <= confidence <= 1.0:
        return "最低置信度必须在 0 到 1 之间。"
    if detector == "hybrid":
        try:
            interval = int(hybrid_fallback_interval)
        except ValueError:
            return "漏检补检间隔必须是整数。"
        if interval <= 0:
            return "漏检补检间隔必须大于 0。"
    return None


def selected_camera_summary(selected_indexes: list[int], swap: bool = False) -> str:
    if not selected_indexes:
        return "尚未选择摄像头"
    ordered = list(selected_indexes)
    if swap and len(ordered) >= 2:
        ordered[0], ordered[1] = ordered[1], ordered[0]
    positions = ("左侧", "右侧", "第三路")
    parts = [f"{positions[position]}：摄像头 {index}" for position, index in enumerate(ordered[:3])]
    return f"已选择 {len(ordered)} 路 · " + " · ".join(parts)


def camera_preview_layout(camera_count: int) -> tuple[int, tuple[int, int]]:
    if camera_count >= 3:
        return 3, (285, 140)
    return max(1, camera_count), (340, 155)


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
    hybrid_fallback_interval: str | None = None,
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
    if hybrid_fallback_interval is not None:
        updated = set_extra_arg_pair(updated, "-HybridFallbackInterval", hybrid_fallback_interval)
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
    if tk is None or ttk is None:
        print("缺少 tkinter，无法打开摄像头选择窗口。")
        return 1

    repo_root = Path.cwd()
    script = (repo_root / args.script).resolve()
    capture_script = (repo_root / args.capture_script).resolve()
    preferred_indexes = parse_preferred_indexes(args.preferred_indexes)
    current_probes = list(probes)

    root = tk.Tk()
    root.title("CrossCamReID - 摄像头与识别模式")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = max(860, min(1040, screen_width - 64))
    window_height = max(680, min(900, screen_height - 100))
    window_x = 20
    window_y = 20
    root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
    root.minsize(min(860, window_width), min(650, window_height))

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
    style.configure("Subtitle.TLabel", font=("Microsoft YaHei UI", 10), foreground="#4b5563")
    style.configure("Section.TLabelframe.Label", font=("Microsoft YaHei UI", 10, "bold"))
    style.configure("Mode.TLabel", font=("Microsoft YaHei UI", 10), foreground="#374151")
    style.configure("Ready.TLabel", font=("Microsoft YaHei UI", 9), foreground="#176b3a")
    style.configure("Warning.TLabel", font=("Microsoft YaHei UI", 9), foreground="#9a5a00")
    style.configure("Error.TLabel", font=("Microsoft YaHei UI", 9), foreground="#a12622")
    style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(14, 8))
    style.configure("Action.TButton", padding=(12, 8))
    style.configure("Camera.TFrame", relief="solid", borderwidth=1)

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
    detector_display_var = tk.StringVar(value=detector_label(detector_var.get()))
    detector_description_var = tk.StringVar(value=DETECTOR_DESCRIPTIONS[detector_var.get()])
    backend_display_var = tk.StringVar(value=BACKEND_LABELS[args.backend])
    backend_var = tk.StringVar(value=args.backend)
    rfdetr_size_var = tk.StringVar(value=extra_arg_value(args.extra_arg, "-RfDetrSize", "nano"))
    rfdetr_weights_var = tk.StringVar(value=extra_arg_value(args.extra_arg, "-RfDetrWeights", ""))
    rfdetr_num_classes_var = tk.StringVar(value=extra_arg_value(args.extra_arg, "-RfDetrNumClasses", "0"))
    rfdetr_conf_var = tk.StringVar(value=extra_arg_value(args.extra_arg, "-RfDetrConf", "0.35"))
    hybrid_fallback_interval_var = tk.StringVar(
        value=extra_arg_value(args.extra_arg, "-HybridFallbackInterval", "15")
    )
    capture_scenario_display_var = tk.StringVar(value=CAPTURE_SCENARIO_LABELS[args.capture_scenario])
    capture_output_root_var = tk.StringVar(value=args.capture_output_root)
    backend_status_var = tk.StringVar()
    selection_summary_var = tk.StringVar(value="尚未选择摄像头")
    launch_status_var = tk.StringVar(value="请选择摄像头。")
    probing_var = tk.BooleanVar(value=False)
    backend_statuses = {}
    reprobe_results: queue.Queue[tuple[list[CameraProbe] | None, str | None]] = queue.Queue()
    selected_vars: dict[int, tk.BooleanVar] = {}
    image_refs: list[object] = []

    header = ttk.Frame(root, padding=(18, 10, 18, 6))
    header.pack(fill="x")
    ttk.Label(header, text="摄像头与识别模式", style="Title.TLabel").pack(anchor="w")
    ttk.Label(header, text="选择画面来源和识别模式后即可启动。", style="Subtitle.TLabel").pack(
        anchor="w", pady=(2, 0)
    )

    notebook = ttk.Notebook(root)
    recognition_tab = ttk.Frame(notebook, padding=10)
    camera_tab = ttk.Frame(notebook, padding=10)
    operation_tab = ttk.Frame(notebook, padding=10)
    notebook.add(recognition_tab, text="识别与显示")
    notebook.add(camera_tab, text="摄像头")
    notebook.add(operation_tab, text="测试与拍照")
    recognition_tab.columnconfigure(0, weight=1)
    operation_tab.columnconfigure(0, weight=1)

    mode_group = ttk.LabelFrame(
        recognition_tab,
        text="识别模式",
        padding=10,
        style="Section.TLabelframe",
    )
    mode_group.grid(row=0, column=0, sticky="ew")
    mode_group.columnconfigure(2, weight=1)
    ttk.Label(mode_group, text="模式").grid(row=0, column=0, sticky="w", padx=(0, 8))
    detector_combo = ttk.Combobox(
        mode_group,
        textvariable=detector_display_var,
        values=[DETECTOR_LABELS[name] for name in DETECTORS],
        state="readonly",
        width=34,
    )
    detector_combo.grid(row=0, column=1, sticky="w")
    pipe_check = ttk.Checkbutton(mode_group, text="使用已训练的铅笔/管子模型", variable=pipe_var)
    pipe_check.grid(row=0, column=2, sticky="w", padx=(18, 0))
    ttk.Label(mode_group, textvariable=detector_description_var, style="Mode.TLabel", wraplength=860).grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(7, 3)
    )
    backend_status_label = ttk.Label(mode_group, textvariable=backend_status_var, wraplength=880)
    backend_status_label.grid(row=2, column=0, columnspan=3, sticky="w")

    rfdetr_group = ttk.LabelFrame(
        recognition_tab,
        text="RF-DETR 实验参数",
        padding=10,
        style="Section.TLabelframe",
    )
    rfdetr_group.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    ttk.Label(rfdetr_group, text="模型大小").grid(row=0, column=0, sticky="w", padx=(0, 6))
    ttk.Combobox(
        rfdetr_group,
        textvariable=rfdetr_size_var,
        values=RFDETR_SIZES,
        state="readonly",
        width=10,
    ).grid(row=0, column=1, sticky="w", padx=(0, 16))
    ttk.Label(rfdetr_group, text="识别类别数").grid(row=0, column=2, sticky="w", padx=(0, 6))
    ttk.Entry(rfdetr_group, textvariable=rfdetr_num_classes_var, width=8).grid(
        row=0, column=3, sticky="w", padx=(0, 16)
    )
    ttk.Label(rfdetr_group, text="最低置信度").grid(row=0, column=4, sticky="w", padx=(0, 6))
    ttk.Entry(rfdetr_group, textvariable=rfdetr_conf_var, width=8).grid(row=0, column=5, sticky="w")
    ttk.Label(rfdetr_group, text="模型文件").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
    weights_entry = ttk.Entry(rfdetr_group, textvariable=rfdetr_weights_var)
    weights_entry.configure(width=58)
    weights_entry.grid(row=1, column=1, columnspan=4, sticky="ew", padx=(0, 8), pady=(8, 0))
    choose_weights_button = ttk.Button(rfdetr_group, text="选择...", width=9)
    choose_weights_button.grid(row=1, column=5, sticky="e", pady=(8, 0))
    hybrid_interval_label = ttk.Label(rfdetr_group, text="漏检后每隔")
    hybrid_interval_entry = ttk.Entry(rfdetr_group, textvariable=hybrid_fallback_interval_var, width=8)
    hybrid_interval_unit = ttk.Label(rfdetr_group, text="帧尝试一次")
    hybrid_interval_label.grid(row=2, column=0, sticky="w", pady=(8, 0))
    hybrid_interval_entry.grid(row=2, column=1, sticky="w", pady=(8, 0))
    hybrid_interval_unit.grid(row=2, column=2, sticky="w", pady=(8, 0))
    hybrid_interval_widgets = (hybrid_interval_label, hybrid_interval_entry, hybrid_interval_unit)
    ttk.Label(
        rfdetr_group,
        text="一般保持默认；本机没有独立显卡时使用 Nano，只有做对比实验时再修改。",
        style="Subtitle.TLabel",
    ).grid(row=3, column=0, columnspan=6, sticky="w", pady=(8, 0))

    display_group = ttk.LabelFrame(
        recognition_tab,
        text="画面设置",
        padding=10,
        style="Section.TLabelframe",
    )
    display_group.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    ttk.Label(display_group, text="读取方式").grid(row=0, column=0, sticky="w", padx=(0, 6))
    ttk.Combobox(
        display_group,
        textvariable=backend_display_var,
        values=list(BACKEND_LABELS.values()),
        state="readonly",
        width=22,
    ).grid(row=0, column=1, sticky="w", padx=(0, 22))
    ttk.Checkbutton(display_group, text="镜像画面", variable=flip_var).grid(row=0, column=2, sticky="w", padx=(0, 18))
    ttk.Checkbutton(display_group, text="交换左右", variable=swap_var).grid(row=0, column=3, sticky="w", padx=(0, 18))
    ttk.Checkbutton(display_group, text="显示轨迹和方向箭头", variable=trails_var).grid(
        row=0,
        column=4,
        sticky="w",
    )

    run_group = ttk.LabelFrame(
        operation_tab,
        text="运行与验收",
        padding=10,
        style="Section.TLabelframe",
    )
    run_group.grid(row=0, column=0, sticky="ew")
    ttk.Checkbutton(run_group, text="摄像头打不开时进入演示画面", variable=fallback_var).grid(
        row=0, column=0, sticky="w", padx=(0, 22)
    )
    ttk.Checkbutton(run_group, text="结束后自动检查运行结果", variable=analyze_var).grid(
        row=0, column=1, sticky="w", padx=(0, 22)
    )
    ttk.Checkbutton(run_group, text="检查跨摄像头接力是否成功", variable=require_handoff_var).grid(
        row=0, column=2, sticky="w"
    )
    ttk.Checkbutton(run_group, text="检查识别框是否频繁跳动", variable=target_lock_gate_var).grid(
        row=1, column=0, sticky="w", padx=(0, 22), pady=(8, 0)
    )
    ttk.Checkbutton(run_group, text="保存本次锁定目标样本", variable=collect_samples_var).grid(
        row=1, column=1, sticky="w", padx=(0, 22), pady=(8, 0)
    )
    sample_preview_check = ttk.Checkbutton(run_group, text="生成目标样本拼图", variable=sample_preview_var)
    sample_preview_check.grid(row=1, column=2, sticky="w", pady=(8, 0))

    capture_group = ttk.LabelFrame(
        operation_tab,
        text="训练素材采集",
        padding=10,
        style="Section.TLabelframe",
    )
    capture_group.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    capture_group.columnconfigure(3, weight=1)
    ttk.Label(capture_group, text="拍摄内容").grid(row=0, column=0, sticky="w", padx=(0, 6))
    ttk.Combobox(
        capture_group,
        textvariable=capture_scenario_display_var,
        values=list(CAPTURE_SCENARIO_LABELS.values()),
        state="readonly",
        width=18,
    ).grid(row=0, column=1, sticky="w", padx=(0, 18))
    ttk.Label(capture_group, text="保存目录").grid(row=0, column=2, sticky="w", padx=(0, 6))
    ttk.Entry(capture_group, textvariable=capture_output_root_var).grid(row=0, column=3, sticky="ew", padx=(0, 8))
    choose_capture_dir_button = ttk.Button(capture_group, text="选择...", width=9)
    choose_capture_dir_button.grid(row=0, column=4, sticky="e")

    camera_group = ttk.Frame(camera_tab, padding=(4, 2))
    camera_group.pack(fill="both", expand=True)
    camera_header = ttk.Frame(camera_group)
    camera_header.pack(fill="x", pady=(0, 6))
    selection_summary_label = ttk.Label(camera_header, textvariable=selection_summary_var, style="Mode.TLabel")
    selection_summary_label.pack(side="left")
    reprobe_button = ttk.Button(camera_header, text="重新检测摄像头", width=15)
    reprobe_button.pack(side="right")

    list_outer = ttk.Frame(camera_group)
    list_outer.pack(fill="both", expand=True)
    list_canvas = tk.Canvas(list_outer, highlightthickness=0, borderwidth=0)
    list_scrollbar = ttk.Scrollbar(list_outer, orient="vertical", command=list_canvas.yview)
    list_frame = ttk.Frame(list_canvas)
    list_window = list_canvas.create_window((0, 0), window=list_frame, anchor="nw")
    list_canvas.configure(yscrollcommand=list_scrollbar.set)
    list_canvas.pack(side="left", fill="both", expand=True)
    list_scrollbar.pack(side="right", fill="y")

    footer = ttk.Frame(root, padding=(18, 4, 18, 14))
    launch_status_label = ttk.Label(footer, textvariable=launch_status_var, wraplength=500)
    launch_status_label.pack(side="left", fill="x", expand=True)

    start_button = None
    capture_button = None

    def selected_indexes() -> list[int]:
        return ordered_selected_indexes(
            [index for index, var in selected_vars.items() if var.get()],
            preferred_indexes,
            current_probes,
        )

    def update_backend_status() -> None:
        nonlocal backend_statuses
        backend_statuses = collect_backend_statuses(
            repo_root,
            yolo_model=extra_arg_value(args.extra_arg, "-YoloModel", ""),
            rfdetr_size=rfdetr_size_var.get(),
            rfdetr_weights=rfdetr_weights_var.get().strip(),
        )
        status = backend_statuses[detector_var.get()]
        readiness = "可用" if status.project_ready else "未就绪"
        backend_status_var.set(f"模式状态：{readiness} · {status.detail}")
        backend_status_label.configure(style="Ready.TLabel" if status.project_ready else "Warning.TLabel")

    def refresh_state(*_args) -> None:
        chosen = selected_indexes()
        selection_summary_var.set(selected_camera_summary(chosen, swap_var.get()))
        settings_error = validate_detector_settings(
            detector_var.get(),
            rfdetr_num_classes_var.get().strip(),
            rfdetr_conf_var.get().strip(),
            hybrid_fallback_interval_var.get().strip(),
        )
        select_error = selection_error(len(chosen), require_handoff_var.get(), fallback_var.get())
        status = backend_statuses.get(detector_var.get())
        if probing_var.get():
            launch_status_var.set("正在重新检测摄像头...")
            launch_status_label.configure(style="Warning.TLabel")
        elif select_error is not None:
            launch_status_var.set(select_error)
            launch_status_label.configure(style="Error.TLabel")
        elif settings_error is not None:
            launch_status_var.set(settings_error)
            launch_status_label.configure(style="Error.TLabel")
        elif status is not None and not status.project_ready:
            launch_status_var.set("当前模式尚未准备好；请确认模型组件和模型文件。")
            launch_status_label.configure(style="Warning.TLabel")
        else:
            launch_status_var.set(f"可以启动 · {detector_label(detector_var.get())}")
            launch_status_label.configure(style="Ready.TLabel")

        start_allowed = not probing_var.get() and select_error is None and settings_error is None
        capture_allowed = (
            not probing_var.get()
            and len(chosen) == 2
            and bool(capture_output_root_var.get().strip())
        )
        if start_button is not None:
            start_button.state(["!disabled"] if start_allowed else ["disabled"])
        if capture_button is not None:
            capture_button.state(["!disabled"] if capture_allowed else ["disabled"])
        reprobe_button.state(["disabled"] if probing_var.get() else ["!disabled"])
        sample_preview_check.state(["!disabled"] if collect_samples_var.get() else ["disabled"])

    def update_mode_controls(*_args) -> None:
        detector = detector_from_label(detector_display_var.get())
        detector_var.set(detector)
        detector_description_var.set(DETECTOR_DESCRIPTIONS[detector])
        if detector == "motion":
            pipe_var.set(False)
            pipe_check.state(["disabled"])
        else:
            pipe_var.set(True)
            pipe_check.state(["!disabled"])
        if detector in ("rfdetr", "hybrid"):
            rfdetr_group.grid()
        else:
            rfdetr_group.grid_remove()
        for widget in hybrid_interval_widgets:
            if detector == "hybrid":
                widget.grid()
            else:
                widget.grid_remove()
        update_backend_status()
        refresh_state()

    def update_backend_choice(*_args) -> None:
        backend_var.set(backend_from_label(backend_display_var.get()))
        refresh_state()

    def on_target_lock_gate_change(*_args) -> None:
        if target_lock_gate_var.get():
            analyze_var.set(True)
            collect_samples_var.set(True)
            sample_preview_var.set(True)
        refresh_state()

    def on_require_handoff_change(*_args) -> None:
        if require_handoff_var.get():
            analyze_var.set(True)
        refresh_state()

    def on_collect_samples_change(*_args) -> None:
        if not collect_samples_var.get() and not target_lock_gate_var.get():
            sample_preview_var.set(False)
        refresh_state()

    def choose_weights() -> None:
        if filedialog is None:
            return
        selected_path = filedialog.askopenfilename(
            parent=root,
            title="选择 RF-DETR 模型文件",
            initialdir=str((repo_root / "runs_rfdetr").resolve()),
            filetypes=(("RF-DETR 模型文件", "*.pth"), ("所有文件", "*.*")),
        )
        if not selected_path:
            return
        weight_path = Path(selected_path).resolve()
        try:
            value = str(weight_path.relative_to(repo_root))
        except ValueError:
            value = str(weight_path)
        rfdetr_weights_var.set(value)

    def choose_capture_directory() -> None:
        if filedialog is None:
            return
        initial_path = Path(capture_output_root_var.get().strip() or "dataset_raw")
        if not initial_path.is_absolute():
            initial_path = repo_root / initial_path
        selected_path = filedialog.askdirectory(
            parent=root,
            title="选择训练素材保存目录",
            initialdir=str(initial_path.resolve()),
        )
        if not selected_path:
            return
        output_path = Path(selected_path).resolve()
        try:
            value = str(output_path.relative_to(repo_root))
        except ValueError:
            value = str(output_path)
        capture_output_root_var.set(value)

    def on_list_configure(_event=None) -> None:
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

    def enforce_camera_limit(index: int) -> None:
        if sum(1 for value in selected_vars.values() if value.get()) > 3:
            selected_vars[index].set(False)
            messagebox.showwarning("最多选择三路", "当前程序最多同时显示 3 个摄像头。")
        refresh_state()

    def toggle_camera(index: int) -> None:
        variable = selected_vars[index]
        variable.set(not variable.get())
        enforce_camera_limit(index)

    def render_camera_cards(new_probes: list[CameraProbe]) -> None:
        nonlocal current_probes
        current_probes = list(new_probes)
        for child in list_frame.winfo_children():
            child.destroy()
        selected_vars.clear()
        image_refs.clear()
        default_selected = default_selected_indexes(current_probes, preferred_indexes)
        column_count, preview_size = camera_preview_layout(len(current_probes))
        for column in range(3):
            list_frame.columnconfigure(
                column,
                weight=1 if column < column_count else 0,
                uniform="camera" if column < column_count else "",
            )
        if not current_probes:
            ttk.Label(list_frame, text="没有检测到可用摄像头。检查连接后点击“重新检测摄像头”。").grid(
                row=0, column=0, columnspan=column_count, sticky="w", padx=8, pady=18
            )
        for position, probe in enumerate(current_probes):
            selected_vars[probe.index] = tk.BooleanVar(value=probe.index in default_selected)
            card = ttk.Frame(list_frame, style="Camera.TFrame", padding=8)
            card.grid(
                row=position // column_count,
                column=position % column_count,
                sticky="nsew",
                padx=5,
                pady=5,
            )
            card.columnconfigure(0, weight=1)
            header_row = ttk.Frame(card)
            header_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
            check = ttk.Checkbutton(
                header_row,
                text=f"摄像头 {probe.index}  ·  {probe.width}×{probe.height}",
                variable=selected_vars[probe.index],
                command=lambda index=probe.index: enforce_camera_limit(index),
            )
            check.pack(side="left")
            if probe.index in preferred_indexes:
                ttk.Label(header_row, text="默认", style="Ready.TLabel").pack(side="right")
            if Image is not None and ImageTk is not None:
                rgb = cv2.cvtColor(probe.frame_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail(preview_size)
                photo = ImageTk.PhotoImage(image)
                image_refs.append(photo)
                preview = ttk.Label(card, image=photo, cursor="hand2")
                preview.grid(row=1, column=0, sticky="w")
                preview.bind("<Button-1>", lambda _event, index=probe.index: toggle_camera(index))
        root.image_refs = image_refs
        root.after_idle(on_list_configure)
        refresh_state()

    def finish_reprobe(new_probes: list[CameraProbe], error: str | None = None) -> None:
        probing_var.set(False)
        if error is not None:
            messagebox.showerror("摄像头检测失败", error)
        else:
            render_camera_cards(new_probes)
        refresh_state()

    def on_reprobe() -> None:
        if probing_var.get():
            return
        probing_var.set(True)
        refresh_state()
        selected_backend = backend_var.get()

        def worker() -> None:
            try:
                result = probe_cameras(args.probe_max, selected_backend)
            except Exception as exc:  # pragma: no cover - hardware-specific path
                reprobe_results.put((None, str(exc)))
                return
            reprobe_results.put((result, None))

        def poll_result() -> None:
            try:
                result, error = reprobe_results.get_nowait()
            except queue.Empty:
                root.after(100, poll_result)
                return
            finish_reprobe(result or [], error)

        threading.Thread(target=worker, daemon=True).start()
        root.after(100, poll_result)

    def on_start() -> None:
        selected = selected_indexes()
        error = selection_error(len(selected), require_handoff_var.get(), fallback_var.get())
        if error is not None:
            messagebox.showwarning("无法启动", error)
            return
        settings_error = validate_detector_settings(
            detector_var.get(),
            rfdetr_num_classes_var.get().strip(),
            rfdetr_conf_var.get().strip(),
            hybrid_fallback_interval_var.get().strip(),
        )
        if settings_error is not None:
            messagebox.showwarning("参数不正确", settings_error)
            return
        status = backend_statuses.get(detector_var.get())
        if status is not None and not status.project_ready:
            proceed = messagebox.askyesno(
                "后端尚未就绪",
                selector_status_text(status) + "\n\n仍要继续启动吗？",
            )
            if not proceed:
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
            rfdetr_size_var.get().strip(),
            rfdetr_weights_var.get().strip(),
            rfdetr_num_classes_var.get().strip(),
            rfdetr_conf_var.get().strip(),
            hybrid_fallback_interval_var.get().strip(),
        )
        try:
            start_crosscam(
                repo_root,
                script,
                selected,
                backend_var.get(),
                pipe_var.get(),
                flip_var.get(),
                trails_var.get(),
                view_order,
                extra_args,
            )
        except OSError as exc:
            messagebox.showerror("启动失败", str(exc))
            return
        root.destroy()

    def on_capture() -> None:
        selected = selected_indexes()
        if len(selected) != 2:
            messagebox.showwarning("无法采集", "采集训练照片需要刚好选择 2 个摄像头。")
            return
        output_root = capture_output_root_var.get().strip()
        if not output_root:
            messagebox.showwarning("保存目录为空", "采集照片保存目录不能为空。")
            return
        try:
            start_capture(
                repo_root,
                capture_script,
                selected,
                backend_var.get(),
                capture_scenario_from_label(capture_scenario_display_var.get()),
                output_root,
                flip_var.get(),
                "BA" if swap_var.get() else "AB",
            )
        except OSError as exc:
            messagebox.showerror("启动采集失败", str(exc))
            return
        root.destroy()

    def on_close() -> None:
        root.destroy()

    ttk.Button(footer, text="取消", command=on_close, width=10, style="Action.TButton").pack(
        side="right", padx=(8, 0)
    )
    capture_button = ttk.Button(
        footer,
        text="采集训练照片",
        command=on_capture,
        width=14,
        style="Action.TButton",
    )
    capture_button.pack(side="right", padx=(8, 0))
    start_button = ttk.Button(
        footer,
        text="启动识别",
        command=on_start,
        width=14,
        style="Primary.TButton",
    )
    start_button.pack(side="right", padx=(8, 0))
    footer.pack(side="bottom", fill="x")
    notebook.pack(side="top", fill="both", expand=True, padx=18, pady=(0, 8))

    choose_weights_button.configure(command=choose_weights)
    choose_capture_dir_button.configure(command=choose_capture_directory)
    reprobe_button.configure(command=on_reprobe)
    list_frame.bind("<Configure>", on_list_configure)
    list_canvas.bind("<Configure>", on_canvas_configure)
    list_outer.bind("<Enter>", bind_mousewheel)
    list_outer.bind("<Leave>", unbind_mousewheel)
    detector_display_var.trace_add("write", update_mode_controls)
    backend_display_var.trace_add("write", update_backend_choice)
    rfdetr_size_var.trace_add("write", lambda *_args: (update_backend_status(), refresh_state()))
    rfdetr_weights_var.trace_add("write", lambda *_args: (update_backend_status(), refresh_state()))
    rfdetr_num_classes_var.trace_add("write", refresh_state)
    rfdetr_conf_var.trace_add("write", refresh_state)
    hybrid_fallback_interval_var.trace_add("write", refresh_state)
    swap_var.trace_add("write", refresh_state)
    fallback_var.trace_add("write", refresh_state)
    require_handoff_var.trace_add("write", on_require_handoff_change)
    target_lock_gate_var.trace_add("write", on_target_lock_gate_change)
    collect_samples_var.trace_add("write", on_collect_samples_change)
    capture_output_root_var.trace_add("write", refresh_state)
    root.protocol("WM_DELETE_WINDOW", on_close)

    update_mode_controls()
    render_camera_cards(current_probes)
    refresh_state()
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
