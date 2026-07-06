from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TargetSampleSummary:
    path: Optional[Path]
    exists: bool
    sample_count: int
    register_sample_count: int
    match_sample_count: int
    missing_image_count: int
    camera_counts: dict[str, int]
    similarity_count: int
    min_similarity: Optional[float]
    avg_similarity: Optional[float]
    max_similarity: Optional[float]


@dataclass
class HandoffResult:
    log_path: Path
    total_events: int
    run_config: dict[str, str]
    run_config_message: Optional[str]
    registered_id: Optional[str]
    registered_camera: Optional[str]
    left_index: Optional[int]
    handoff_index: Optional[int]
    handoff_camera: Optional[str]
    target_match_count: int
    cross_camera_ids: list[str]
    unique_global_ids: list[str]
    new_event_count: int
    matched_event_count: int
    left_event_count: int
    track_created_count: int
    new_after_register_count: int
    new_by_camera: dict[str, int]
    registered_left_count: int
    target_similarity_count: int
    min_target_similarity: Optional[float]
    avg_target_similarity: Optional[float]
    max_target_similarity: Optional[float]
    target_choice_counts: dict[str, int]
    blocked_target_candidate_count: int
    target_distance_count: int
    avg_target_distance: Optional[float]
    max_target_distance: Optional[float]
    target_samples: TargetSampleSummary

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
    parser.add_argument(
        "--target-lock-gate",
        action="store_true",
        help="Apply practical default gates for a locked-target demo run.",
    )
    parser.add_argument("--max-new-ids", type=int, help="Fail if new object events exceed this number.")
    parser.add_argument("--max-new-after-register", type=int, help="Fail if new events after registration exceed this.")
    parser.add_argument("--max-unique-ids", type=int, help="Fail if unique global IDs exceed this number.")
    parser.add_argument("--min-target-matches", type=int, help="Fail if target_matched events are below this number.")
    parser.add_argument(
        "--min-target-similarity",
        type=float,
        help="Fail if any registered-target similarity is below this value.",
    )
    parser.add_argument(
        "--max-registered-lefts",
        type=int,
        help="Fail if registered target left events exceed this number.",
    )
    parser.add_argument(
        "--max-target-switches",
        type=int,
        help="Fail if registered-target switch choices exceed this number.",
    )
    parser.add_argument(
        "--max-target-distance",
        type=float,
        help="Fail if registered-target same-camera movement exceeds this pixel distance.",
    )
    parser.add_argument(
        "--max-blocked-target-candidates",
        type=int,
        help="Fail if protected similar target candidates exceed this number.",
    )
    parser.add_argument(
        "--min-cross-camera-ids",
        type=int,
        help="Fail if fewer global IDs appeared in both cameras.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write a machine-readable analysis summary.",
    )
    parser.add_argument(
        "--summary-md",
        help="Optional path to write a short human-readable Markdown summary.",
    )
    parser.add_argument(
        "--target-samples-csv",
        default="",
        help="Target sample CSV path. Defaults to log_dir/targets/target_samples.csv when present.",
    )
    parser.add_argument("--min-target-samples", type=int, help="Fail if saved target samples are below this number.")
    parser.add_argument("--min-match-samples", type=int, help="Fail if saved matched target samples are below this number.")
    parser.add_argument(
        "--min-sample-cameras",
        type=int,
        help="Fail if saved target samples come from fewer cameras.",
    )
    return parser.parse_args()


def apply_target_lock_gate_defaults(args: argparse.Namespace) -> None:
    if not args.target_lock_gate:
        return
    if args.min_target_matches is None:
        args.min_target_matches = 1
    if args.max_target_switches is None:
        args.max_target_switches = 0
    if args.max_blocked_target_candidates is None:
        args.max_blocked_target_candidates = 0
    if args.max_target_distance is None:
        args.max_target_distance = 120.0
    if args.min_target_samples is None:
        args.min_target_samples = 2
    if args.min_match_samples is None:
        args.min_match_samples = 1


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


def parse_optional_float(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_run_config_message(message: str) -> dict[str, str]:
    config: dict[str, str] = {}
    for part in message.split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            config[key] = value.strip()
    return config


def format_run_config(config: dict[str, str]) -> str:
    if not config:
        return "未记录"
    preferred_keys = [
        "detector",
        "camera_count",
        "backend",
        "yolo_model",
        "yolo_conf",
        "rfdetr_size",
        "rfdetr_weights",
        "rfdetr_conf",
        "rfdetr_num_classes",
    ]
    parts = [f"{key}={config[key]}" for key in preferred_keys if config.get(key)]
    if not parts:
        parts = [f"{key}={value}" for key, value in sorted(config.items()) if value]
    return ", ".join(parts) if parts else "未记录"


def default_target_samples_path(log_path: Path) -> Path:
    return log_path.parent / "targets" / "target_samples.csv"


def resolve_indexed_image_path(raw_path: str, csv_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return csv_path.parent / path


def analyze_target_samples(path: Optional[Path]) -> TargetSampleSummary:
    if path is None:
        return TargetSampleSummary(
            path=None,
            exists=False,
            sample_count=0,
            register_sample_count=0,
            match_sample_count=0,
            missing_image_count=0,
            camera_counts={},
            similarity_count=0,
            min_similarity=None,
            avg_similarity=None,
            max_similarity=None,
        )
    if not path.exists():
        return TargetSampleSummary(
            path=path,
            exists=False,
            sample_count=0,
            register_sample_count=0,
            match_sample_count=0,
            missing_image_count=0,
            camera_counts={},
            similarity_count=0,
            min_similarity=None,
            avg_similarity=None,
            max_similarity=None,
        )

    rows = read_rows(path)
    camera_counts: dict[str, int] = {}
    similarities: list[float] = []
    missing_image_count = 0
    register_sample_count = 0
    match_sample_count = 0
    for row in rows:
        camera = row.get("camera", "")
        if camera:
            camera_counts[camera] = camera_counts.get(camera, 0) + 1
        source = row.get("source", "")
        if source == "register":
            register_sample_count += 1
        elif source == "match":
            match_sample_count += 1
        similarity = parse_optional_float(row.get("target_similarity", ""))
        if similarity is not None:
            similarities.append(similarity)
        image = row.get("image", "")
        if image and not resolve_indexed_image_path(image, path).exists():
            missing_image_count += 1

    return TargetSampleSummary(
        path=path,
        exists=True,
        sample_count=len(rows),
        register_sample_count=register_sample_count,
        match_sample_count=match_sample_count,
        missing_image_count=missing_image_count,
        camera_counts=dict(sorted(camera_counts.items())),
        similarity_count=len(similarities),
        min_similarity=min(similarities) if similarities else None,
        avg_similarity=sum(similarities) / len(similarities) if similarities else None,
        max_similarity=max(similarities) if similarities else None,
    )


def analyze_rows(path: Path, rows: list[dict[str, str]], target_samples: TargetSampleSummary) -> HandoffResult:
    registered_id: Optional[str] = None
    registered_camera: Optional[str] = None
    left_index: Optional[int] = None
    handoff_index: Optional[int] = None
    handoff_camera: Optional[str] = None
    target_match_count = 0
    seen_cameras_by_id: dict[str, set[str]] = {}
    unique_global_ids: set[str] = set()
    new_event_count = 0
    matched_event_count = 0
    left_event_count = 0
    track_created_count = 0
    new_after_register_count = 0
    new_by_camera: dict[str, int] = {}
    registered_left_count = 0
    target_similarities: list[float] = []
    target_choice_counts: dict[str, int] = {}
    blocked_target_candidate_count = 0
    target_distances: list[float] = []
    run_config_message: Optional[str] = None
    run_config: dict[str, str] = {}

    for index, row in enumerate(rows):
        event_type = row.get("event_type", "")
        global_id = row.get("global_id", "")
        camera = row.get("camera", "")

        if event_type == "run_config":
            message = row.get("message", "")
            if message:
                run_config_message = message
                run_config = parse_run_config_message(message)
            continue

        if global_id:
            unique_global_ids.add(global_id)

        if global_id and camera and event_type in {"target_registered", "target_matched", "matched", "track_created"}:
            seen_cameras_by_id.setdefault(global_id, set()).add(camera)

        if event_type == "target_registered" and registered_id is None:
            registered_id = global_id
            registered_camera = camera

        if event_type == "new":
            new_event_count += 1
            if registered_id is not None:
                new_after_register_count += 1
            if camera:
                new_by_camera[camera] = new_by_camera.get(camera, 0) + 1

        if event_type == "matched":
            matched_event_count += 1

        if event_type == "left":
            left_event_count += 1
            if registered_id is not None and global_id == registered_id:
                registered_left_count += 1

        if event_type == "track_created":
            track_created_count += 1

        if event_type == "target_matched":
            target_match_count += 1
            target_choice = row.get("target_choice", "")
            if target_choice:
                target_choice_counts[target_choice] = target_choice_counts.get(target_choice, 0) + 1
            target_distance = parse_optional_float(row.get("target_distance", ""))
            if target_distance is not None:
                target_distances.append(target_distance)
        elif event_type == "new" and row.get("target_choice", "") == "blocked":
            blocked_target_candidate_count += 1

        if event_type in {"target_registered", "target_matched"}:
            target_similarity = parse_optional_float(row.get("target_similarity", ""))
            if target_similarity is not None:
                target_similarities.append(target_similarity)

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
        run_config=run_config,
        run_config_message=run_config_message,
        registered_id=registered_id,
        registered_camera=registered_camera,
        left_index=left_index,
        handoff_index=handoff_index,
        handoff_camera=handoff_camera,
        target_match_count=target_match_count,
        cross_camera_ids=cross_camera_ids,
        unique_global_ids=sorted(unique_global_ids),
        new_event_count=new_event_count,
        matched_event_count=matched_event_count,
        left_event_count=left_event_count,
        track_created_count=track_created_count,
        new_after_register_count=new_after_register_count,
        new_by_camera=dict(sorted(new_by_camera.items())),
        registered_left_count=registered_left_count,
        target_similarity_count=len(target_similarities),
        min_target_similarity=min(target_similarities) if target_similarities else None,
        avg_target_similarity=sum(target_similarities) / len(target_similarities) if target_similarities else None,
        max_target_similarity=max(target_similarities) if target_similarities else None,
        target_choice_counts=dict(sorted(target_choice_counts.items())),
        blocked_target_candidate_count=blocked_target_candidate_count,
        target_distance_count=len(target_distances),
        avg_target_distance=sum(target_distances) / len(target_distances) if target_distances else None,
        max_target_distance=max(target_distances) if target_distances else None,
        target_samples=target_samples,
    )


def print_result(result: HandoffResult) -> None:
    print(f"日志文件：{result.log_path}")
    print(f"事件总数：{result.total_events}")
    print(f"运行配置：{format_run_config(result.run_config)}")
    print(f"注册目标：{first_nonempty(result.registered_id, '未发现')}")
    print(f"注册摄像头：{first_nonempty(result.registered_camera, '未发现')}")
    print(f"目标匹配事件数：{result.target_match_count}")
    print(f"新建目标事件数：{result.new_event_count}")
    print(f"注册后新建目标事件数：{result.new_after_register_count}")
    print(f"普通跨摄像头匹配事件数：{result.matched_event_count}")
    print(f"离开画面事件数：{result.left_event_count}")
    print(f"注册目标离开次数：{result.registered_left_count}")
    print(f"轨迹创建事件数：{result.track_created_count}")
    print(f"唯一全局 ID 数：{len(result.unique_global_ids)}")
    if result.target_similarity_count:
        print(
            "注册目标相似度："
            f"次数={result.target_similarity_count}, "
            f"最低={result.min_target_similarity:.3f}, "
            f"平均={result.avg_target_similarity:.3f}, "
            f"最高={result.max_target_similarity:.3f}"
        )
    else:
        print("注册目标相似度：无")
    if result.target_choice_counts:
        choice_parts = [f"{choice}={count}" for choice, count in result.target_choice_counts.items()]
        print(f"注册目标选择：{', '.join(choice_parts)}")
    else:
        print("注册目标选择：无")
    print(f"被保护逻辑拦截的相似候选数：{result.blocked_target_candidate_count}")
    if result.target_distance_count:
        print(
            "注册目标距上次位置："
            f"次数={result.target_distance_count}, "
            f"平均={result.avg_target_distance:.1f}px, "
            f"最大={result.max_target_distance:.1f}px"
        )
    else:
        print("注册目标距上次位置：无")
    print_target_sample_summary(result.target_samples)
    if result.new_by_camera:
        camera_parts = [f"摄像头{camera}={count}" for camera, count in result.new_by_camera.items()]
        print(f"新建目标按摄像头：{', '.join(camera_parts)}")
    else:
        print("新建目标按摄像头：无")
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
    for diagnosis in diagnosis_messages(result):
        print(f"诊断：{diagnosis}")


def target_lock_status(result: HandoffResult, failures: list[str]) -> str:
    candidate_events = result.new_event_count + result.track_created_count
    if result.registered_id is None:
        return "needs_registration" if candidate_events > 0 else "needs_detection"
    if result.blocked_target_candidate_count > 0:
        return "blocked_candidate"
    if result.target_match_count == 0:
        return "registered_not_matched"
    if result.target_choice_counts.get("switch", 0) > 0:
        return "target_jump_risk"
    if result.max_target_distance is not None and result.max_target_distance > 120:
        return "target_jump_risk"
    if result.target_samples.missing_image_count > 0:
        return "sample_issue"
    if result.target_samples.sample_count < 2 or result.target_samples.match_sample_count < 1:
        return "sample_issue"
    if failures:
        return "failed_quality_gate"
    if result.handoff_success:
        return "handoff_ok"
    return "lock_ok"


def target_lock_status_label(status: str) -> str:
    labels = {
        "needs_detection": "还没有检测到候选目标",
        "needs_registration": "已经检测到目标，但还没有注册锁定目标",
        "blocked_candidate": "相似候选被保护逻辑拦截，需要确认是否满足接力条件",
        "registered_not_matched": "已经注册目标，但后续没有稳定匹配到",
        "target_jump_risk": "目标可能发生跳框，需要复查注册框或运动过程",
        "sample_issue": "目标样本不足或样本文件异常",
        "failed_quality_gate": "没有通过当前质量门槛",
        "handoff_ok": "跨摄像头接力成功",
        "lock_ok": "目标锁定基本正常",
    }
    return labels.get(status, status)


def print_target_sample_summary(samples: TargetSampleSummary) -> None:
    if samples.path is None:
        print("目标样本索引：未指定")
        return
    if not samples.exists:
        print(f"目标样本索引：未找到 ({samples.path})")
        return
    print(f"目标样本索引：{samples.path}")
    print(
        "目标样本："
        f"总数={samples.sample_count}, "
        f"注册={samples.register_sample_count}, "
        f"匹配={samples.match_sample_count}, "
        f"缺失图片={samples.missing_image_count}"
    )
    if samples.camera_counts:
        camera_parts = [f"摄像头{camera}={count}" for camera, count in samples.camera_counts.items()]
        print(f"目标样本按摄像头：{', '.join(camera_parts)}")
    else:
        print("目标样本按摄像头：无")
    if samples.similarity_count:
        print(
            "目标样本相似度："
            f"次数={samples.similarity_count}, "
            f"最低={samples.min_similarity:.3f}, "
            f"平均={samples.avg_similarity:.3f}, "
            f"最高={samples.max_similarity:.3f}"
        )
    else:
        print("目标样本相似度：无")


def diagnosis_messages(result: HandoffResult) -> list[str]:
    messages: list[str] = []
    candidate_events = result.new_event_count + result.track_created_count
    if result.registered_id is None:
        if candidate_events > 0:
            messages.append("已经检测到候选目标，但还没有注册锁定目标；请先点击检测框或使用注册按钮。")
        else:
            messages.append("没有检测到候选目标；请先检查检测后端、模型权重、阈值或画面 ROI。")
    elif result.target_match_count == 0:
        messages.append("已经注册目标，但没有后续 target_matched；可能是目标特征不稳定、阈值过高或检测框跳变。")
    elif result.target_samples.exists and result.target_samples.match_sample_count == 0:
        messages.append("已经匹配到目标，但没有保存匹配样本；后续 ReID 模板素材可能不够。")
    if result.blocked_target_candidate_count > 0:
        messages.append(
            "有相似候选被保护逻辑拦截：它像 G001，但原目标尚未离开或跨摄像头接力条件未满足。"
        )
    return messages


def recommended_actions(result: HandoffResult, failures: list[str]) -> list[str]:
    actions: list[str] = []
    candidate_events = result.new_event_count + result.track_created_count
    if result.registered_id is None:
        if candidate_events > 0:
            actions.append("先点击目标检测框或使用注册按钮，确认日志出现 target_registered。")
        else:
            actions.append("先检查检测后端、模型权重、阈值和 ROI，让日志至少出现候选目标事件。")
        return actions

    if result.target_match_count == 0:
        actions.append("重新注册更干净的目标框，必要时降低匹配阈值或补充目标样本。")
    if result.target_choice_counts.get("switch", 0) > 0:
        actions.append("出现明显跳框，优先检查注册框背景占比，并补拍相似目标/遮挡场景再训练。")
    if result.max_target_distance is not None and result.max_target_distance > 120:
        actions.append("同摄像头内目标位移过大，优先复查是否跳到旁边目标。")
    if result.blocked_target_candidate_count > 0:
        actions.append("有相似候选被拦截；若确实是跨摄像头接力，先让原摄像头中的 G001 离开画面。")
    if result.target_samples.exists and result.target_samples.match_sample_count == 0:
        actions.append("已注册但缺少匹配样本，演示后打开目标样本预览图检查注册目标是否干净。")
    if result.target_samples.sample_count < 2 or result.target_samples.match_sample_count < 1:
        actions.append("开启退出后整理目标样本，至少保留注册样本和一次匹配样本用于复盘。")
    if result.target_samples.missing_image_count > 0:
        actions.append("目标样本索引里有图片缺失，先清理旧索引或重新收集样本。")
    if failures and not actions:
        actions.append("先按质量验收失败项逐条复查日志和目标样本。")
    if not failures and not actions:
        actions.append("本次日志通过当前质量门槛，可作为一次可复盘的演示记录。")
    return actions


def quality_failures(result: HandoffResult, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.require_handoff and not result.handoff_success:
        failures.append("未观察到注册目标跨摄像头接力。")
    if args.max_new_ids is not None and result.new_event_count > args.max_new_ids:
        failures.append(f"新建目标事件数 {result.new_event_count} 超过阈值 {args.max_new_ids}。")
    if args.max_new_after_register is not None and result.new_after_register_count > args.max_new_after_register:
        failures.append(
            f"注册后新建目标事件数 {result.new_after_register_count} 超过阈值 {args.max_new_after_register}。"
        )
    if args.max_unique_ids is not None and len(result.unique_global_ids) > args.max_unique_ids:
        failures.append(f"唯一全局 ID 数 {len(result.unique_global_ids)} 超过阈值 {args.max_unique_ids}。")
    if args.min_target_matches is not None and result.target_match_count < args.min_target_matches:
        failures.append(f"目标匹配事件数 {result.target_match_count} 低于阈值 {args.min_target_matches}。")
    if args.min_target_similarity is not None:
        if result.min_target_similarity is None:
            failures.append("没有可用的注册目标相似度。")
        elif result.min_target_similarity < args.min_target_similarity:
            failures.append(
                f"注册目标最低相似度 {result.min_target_similarity:.3f} 低于阈值 {args.min_target_similarity:.3f}。"
            )
    if args.max_registered_lefts is not None and result.registered_left_count > args.max_registered_lefts:
        failures.append(f"注册目标离开次数 {result.registered_left_count} 超过阈值 {args.max_registered_lefts}。")
    if args.max_target_switches is not None:
        switch_count = result.target_choice_counts.get("switch", 0)
        if switch_count > args.max_target_switches:
            failures.append(f"注册目标明显切换次数 {switch_count} 超过阈值 {args.max_target_switches}。")
    if args.max_target_distance is not None:
        if result.max_target_distance is None:
            failures.append("没有可用的注册目标移动距离。")
        elif result.max_target_distance > args.max_target_distance:
            failures.append(
                f"注册目标最大移动距离 {result.max_target_distance:.1f}px 超过阈值 {args.max_target_distance:.1f}px。"
            )
    if (
        args.max_blocked_target_candidates is not None
        and result.blocked_target_candidate_count > args.max_blocked_target_candidates
    ):
        failures.append(
            "被保护逻辑拦截的相似候选数 "
            f"{result.blocked_target_candidate_count} 超过阈值 {args.max_blocked_target_candidates}。"
        )
    if args.min_cross_camera_ids is not None and len(result.cross_camera_ids) < args.min_cross_camera_ids:
        failures.append(f"跨摄像头 ID 数 {len(result.cross_camera_ids)} 低于阈值 {args.min_cross_camera_ids}。")
    if args.min_target_samples is not None and result.target_samples.sample_count < args.min_target_samples:
        failures.append(
            f"目标样本数 {result.target_samples.sample_count} 低于阈值 {args.min_target_samples}。"
        )
    if args.min_match_samples is not None and result.target_samples.match_sample_count < args.min_match_samples:
        failures.append(
            f"目标匹配样本数 {result.target_samples.match_sample_count} 低于阈值 {args.min_match_samples}。"
        )
    if args.min_sample_cameras is not None and len(result.target_samples.camera_counts) < args.min_sample_cameras:
        failures.append(
            f"目标样本摄像头数 {len(result.target_samples.camera_counts)} 低于阈值 {args.min_sample_cameras}。"
        )
    if result.target_samples.missing_image_count > 0:
        failures.append(f"目标样本索引中有 {result.target_samples.missing_image_count} 张图片缺失。")
    return failures


def write_summary_json(path: Path, result: HandoffResult, failures: list[str]) -> None:
    actions = recommended_actions(result, failures)
    status = target_lock_status(result, failures)
    payload = {
        "passed": not failures,
        "failures": failures,
        "target_lock_status": status,
        "target_lock_status_label": target_lock_status_label(status),
        "log_path": str(result.log_path),
        "total_events": result.total_events,
        "run_config": result.run_config,
        "run_config_message": result.run_config_message,
        "registered_id": result.registered_id,
        "registered_camera": result.registered_camera,
        "handoff_success": result.handoff_success,
        "handoff_camera": result.handoff_camera,
        "target_match_count": result.target_match_count,
        "cross_camera_ids": result.cross_camera_ids,
        "unique_global_ids": result.unique_global_ids,
        "new_event_count": result.new_event_count,
        "matched_event_count": result.matched_event_count,
        "left_event_count": result.left_event_count,
        "track_created_count": result.track_created_count,
        "new_after_register_count": result.new_after_register_count,
        "new_by_camera": result.new_by_camera,
        "registered_left_count": result.registered_left_count,
        "target_similarity_count": result.target_similarity_count,
        "min_target_similarity": result.min_target_similarity,
        "avg_target_similarity": result.avg_target_similarity,
        "max_target_similarity": result.max_target_similarity,
        "target_choice_counts": result.target_choice_counts,
        "blocked_target_candidate_count": result.blocked_target_candidate_count,
        "target_distance_count": result.target_distance_count,
        "avg_target_distance": result.avg_target_distance,
        "max_target_distance": result.max_target_distance,
        "target_samples_path": None if result.target_samples.path is None else str(result.target_samples.path),
        "target_samples_exists": result.target_samples.exists,
        "target_sample_count": result.target_samples.sample_count,
        "register_sample_count": result.target_samples.register_sample_count,
        "match_sample_count": result.target_samples.match_sample_count,
        "target_sample_missing_image_count": result.target_samples.missing_image_count,
        "target_sample_camera_counts": result.target_samples.camera_counts,
        "target_sample_similarity_count": result.target_samples.similarity_count,
        "target_sample_min_similarity": result.target_samples.min_similarity,
        "target_sample_avg_similarity": result.target_samples.avg_similarity,
        "target_sample_max_similarity": result.target_samples.max_similarity,
        "diagnosis": diagnosis_messages(result),
        "recommended_actions": actions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_markdown(path: Path, result: HandoffResult, failures: list[str]) -> None:
    actions = recommended_actions(result, failures)
    diagnoses = diagnosis_messages(result)
    status = target_lock_status(result, failures)
    passed_text = "通过" if not failures else "未通过"
    handoff_text = "成功" if result.handoff_success else "未确认成功"
    registered_text = result.registered_id if result.registered_id is not None else "未注册"
    camera_text = result.registered_camera if result.registered_camera is not None else "无"
    target_sample_path = result.target_samples.path if result.target_samples.path is not None else "未指定"
    min_similarity = "无" if result.min_target_similarity is None else f"{result.min_target_similarity:.3f}"
    max_distance = "无" if result.max_target_distance is None else f"{result.max_target_distance:.1f}px"
    lines = [
        "# CrossCamReID 运行摘要",
        "",
        f"- 日志文件：`{result.log_path}`",
        f"- 目标样本索引：`{target_sample_path}`",
        f"- 运行配置：{format_run_config(result.run_config)}",
        f"- 质量验收：{passed_text}",
        f"- 目标锁定：{target_lock_status_label(status)} (`{status}`)",
        f"- 注册目标：{registered_text}",
        f"- 注册摄像头：{camera_text}",
        f"- 目标匹配次数：{result.target_match_count}",
        f"- 注册目标最低相似度：{min_similarity}",
        f"- 注册目标最大位移：{max_distance}",
        f"- 跨摄像头接力：{handoff_text}",
        (
            "- 目标样本："
            f"总数 {result.target_samples.sample_count}，"
            f"匹配样本 {result.target_samples.match_sample_count}，"
            f"缺失图片 {result.target_samples.missing_image_count}"
        ),
        "",
        "## 诊断",
        "",
    ]
    if diagnoses:
        lines.extend(f"- {diagnosis}" for diagnosis in diagnoses)
    else:
        lines.append("- 暂无异常诊断。")

    lines.extend([
        "",
        "## 建议",
        "",
    ])
    if actions:
        lines.extend(f"- {action}" for action in actions)
    else:
        lines.append("- 暂无额外建议。")

    if failures:
        lines.extend(["", "## 未通过原因", ""])
        lines.extend(f"- {failure}" for failure in failures)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    apply_target_lock_gate_defaults(args)
    log_path = Path(args.log) if args.log else latest_log(Path(args.log_dir))
    if log_path is None:
        print(f"没有找到事件日志：{Path(args.log_dir) / '*-events.csv'}")
        return 2
    if not log_path.exists():
        print(f"日志文件不存在：{log_path}")
        return 2

    samples_path = Path(args.target_samples_csv) if args.target_samples_csv else default_target_samples_path(log_path)
    target_samples = analyze_target_samples(samples_path)
    rows = read_rows(log_path)
    result = analyze_rows(log_path, rows, target_samples)
    print_result(result)
    failures = quality_failures(result, args)
    status = target_lock_status(result, failures)
    print(f"目标锁定状态：{status}（{target_lock_status_label(status)}）")
    for action in recommended_actions(result, failures):
        print(f"建议：{action}")
    if args.summary_json:
        summary_path = Path(args.summary_json)
        write_summary_json(summary_path, result, failures)
        print(f"分析摘要：{summary_path}")
    if args.summary_md:
        summary_md_path = Path(args.summary_md)
        write_summary_markdown(summary_md_path, result, failures)
        print(f"运行摘要：{summary_md_path}")
    if failures:
        print("质量验收：未通过")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print("质量验收：通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
