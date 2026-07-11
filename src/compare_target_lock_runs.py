from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two target-lock analysis summaries.")
    parser.add_argument("--baseline", required=True, help="Baseline latest-summary.json path.")
    parser.add_argument("--candidate", required=True, help="Candidate latest-summary.json path.")
    parser.add_argument("--output-json", help="Optional comparison JSON output path.")
    parser.add_argument("--output-md", help="Optional comparison Markdown output path.")
    parser.add_argument("--min-match-ratio", type=float, default=0.98)
    parser.add_argument("--max-switch-increase", type=int, default=0)
    parser.add_argument("--max-average-distance-ratio", type=float, default=1.10)
    parser.add_argument("--max-maximum-distance-ratio", type=float, default=1.10)
    parser.add_argument("--max-new-id-increase", type=int, default=0)
    parser.add_argument("--max-registered-left-increase", type=int, default=0)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"分析摘要不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取分析摘要 {path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"分析摘要不是 JSON 对象：{path}")
    return payload


def number(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"分析摘要字段 {key} 不是数字。")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"分析摘要字段 {key} 不是有限数字。")
    return result


def switch_count(summary: dict[str, Any]) -> int:
    choices = summary.get("target_choice_counts") or {}
    if not isinstance(choices, dict):
        raise RuntimeError("分析摘要字段 target_choice_counts 不是对象。")
    value = choices.get("switch", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("分析摘要字段 target_choice_counts.switch 不是数字。")
    return int(value)


def safe_ratio(candidate: float, baseline: float) -> float:
    if baseline <= 1e-9:
        return 1.0 if candidate <= 1e-9 else math.inf
    return candidate / baseline


def normalized_video_path(summary: dict[str, Any]) -> str:
    config = summary.get("run_config") or {}
    if not isinstance(config, dict):
        return ""
    raw_path = str(config.get("video_a_path") or "").strip()
    if not raw_path:
        return ""
    return os.path.normcase(os.path.normpath(raw_path))


def compare(baseline: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    baseline_matches = number(baseline, "target_match_count")
    candidate_matches = number(candidate, "target_match_count")
    baseline_avg_distance = number(baseline, "avg_target_distance")
    candidate_avg_distance = number(candidate, "avg_target_distance")
    baseline_max_distance = number(baseline, "max_target_distance")
    candidate_max_distance = number(candidate, "max_target_distance")
    baseline_new_ids = number(baseline, "new_after_register_count")
    candidate_new_ids = number(candidate, "new_after_register_count")
    baseline_lefts = number(baseline, "registered_left_count")
    candidate_lefts = number(candidate, "registered_left_count")
    baseline_switches = switch_count(baseline)
    candidate_switches = switch_count(candidate)

    match_ratio = safe_ratio(candidate_matches, baseline_matches)
    avg_distance_ratio = safe_ratio(candidate_avg_distance, baseline_avg_distance)
    max_distance_ratio = safe_ratio(candidate_max_distance, baseline_max_distance)
    failures: list[str] = []

    baseline_video = normalized_video_path(baseline)
    candidate_video = normalized_video_path(candidate)
    if baseline_video and candidate_video and baseline_video != candidate_video:
        failures.append("两次运行使用的 video_a_path 不一致。")
    if match_ratio < args.min_match_ratio:
        failures.append(f"目标匹配比例 {match_ratio:.3f} 低于门槛 {args.min_match_ratio:.3f}。")
    if candidate_switches > baseline_switches + args.max_switch_increase:
        failures.append(
            f"目标切换次数从 {baseline_switches} 增加到 {candidate_switches}，"
            f"超过允许增量 {args.max_switch_increase}。"
        )
    if avg_distance_ratio > args.max_average_distance_ratio:
        failures.append(
            f"平均位移比例 {avg_distance_ratio:.3f} 超过门槛 {args.max_average_distance_ratio:.3f}。"
        )
    if max_distance_ratio > args.max_maximum_distance_ratio:
        failures.append(
            f"最大位移比例 {max_distance_ratio:.3f} 超过门槛 {args.max_maximum_distance_ratio:.3f}。"
        )
    if candidate_new_ids > baseline_new_ids + args.max_new_id_increase:
        failures.append(
            f"注册后新 ID 数从 {int(baseline_new_ids)} 增加到 {int(candidate_new_ids)}，"
            f"超过允许增量 {args.max_new_id_increase}。"
        )
    if candidate_lefts > baseline_lefts + args.max_registered_left_increase:
        failures.append(
            f"注册目标离开次数从 {int(baseline_lefts)} 增加到 {int(candidate_lefts)}，"
            f"超过允许增量 {args.max_registered_left_increase}。"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "baseline": {
            "target_matches": int(baseline_matches),
            "target_switches": baseline_switches,
            "average_distance": baseline_avg_distance,
            "maximum_distance": baseline_max_distance,
            "new_after_register": int(baseline_new_ids),
            "registered_lefts": int(baseline_lefts),
        },
        "candidate": {
            "target_matches": int(candidate_matches),
            "target_switches": candidate_switches,
            "average_distance": candidate_avg_distance,
            "maximum_distance": candidate_max_distance,
            "new_after_register": int(candidate_new_ids),
            "registered_lefts": int(candidate_lefts),
        },
        "ratios": {
            "target_matches": match_ratio,
            "average_distance": avg_distance_ratio,
            "maximum_distance": max_distance_ratio,
        },
    }


def markdown_report(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    candidate = result["candidate"]
    lines = [
        "# 目标锁定回归对比",
        "",
        f"结果：{'通过' if result['passed'] else '未通过'}",
        "",
        "| 指标 | 基线 | 候选 |",
        "| --- | ---: | ---: |",
        f"| 目标匹配次数 | {baseline['target_matches']} | {candidate['target_matches']} |",
        f"| 目标切换次数 | {baseline['target_switches']} | {candidate['target_switches']} |",
        f"| 平均位移 | {baseline['average_distance']:.2f}px | {candidate['average_distance']:.2f}px |",
        f"| 最大位移 | {baseline['maximum_distance']:.2f}px | {candidate['maximum_distance']:.2f}px |",
        f"| 注册后新 ID 数 | {baseline['new_after_register']} | {candidate['new_after_register']} |",
        f"| 注册目标离开次数 | {baseline['registered_lefts']} | {candidate['registered_lefts']} |",
    ]
    if result["failures"]:
        lines.extend(["", "## 未通过原因", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    return "\n".join(lines) + "\n"


def write_optional(path_value: str | None, content: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.min_match_ratio < 0:
        print("目标锁定回归对比失败：--min-match-ratio 不能小于 0。")
        return 2
    for name in ("max_average_distance_ratio", "max_maximum_distance_ratio"):
        if getattr(args, name) < 0:
            print(f"目标锁定回归对比失败：--{name.replace('_', '-')} 不能小于 0。")
            return 2
    try:
        result = compare(read_summary(Path(args.baseline)), read_summary(Path(args.candidate)), args)
    except RuntimeError as exc:
        print(f"目标锁定回归对比失败：{exc}")
        return 2

    report = markdown_report(result)
    write_optional(args.output_json, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_optional(args.output_md, report)
    print(report, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
