from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


COUNT_FIELDS = ("gt", "pred", "matched", "false_positive", "false_negative", "oversized")


@dataclass(frozen=True)
class EvalRow:
    image: str
    gt: int
    pred: int
    matched: int
    false_positive: int
    false_negative: int
    oversized: int
    avg_iou: Optional[float]

    @property
    def issue_score(self) -> int:
        return self.false_positive + self.false_negative + self.oversized


@dataclass(frozen=True)
class EvalSummary:
    image_count: int
    gt: int
    pred: int
    matched: int
    false_positive: int
    false_negative: int
    oversized: int
    avg_iou: Optional[float]
    precision: float
    recall: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two detector analysis CSV files.")
    parser.add_argument("--left-csv", required=True, help="First analysis.csv, usually YOLO.")
    parser.add_argument("--right-csv", required=True, help="Second analysis.csv, usually RF-DETR.")
    parser.add_argument("--left-name", default="YOLO")
    parser.add_argument("--right-name", default="RF-DETR")
    parser.add_argument("--output-csv", help="Optional per-image comparison CSV.")
    parser.add_argument("--summary-json", help="Optional path to write a machine-readable comparison summary.")
    parser.add_argument("--summary-md", help="Optional path to write a short human-readable Markdown summary.")
    parser.add_argument("--max-examples", type=int, default=12, help="Max changed image examples to print.")
    return parser.parse_args()


def parse_int(value: str, field: str, path: Path, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{path}:{line_number} 字段 {field} 不是整数：{value}") from exc


def parse_optional_float(value: str, field: str, path: Path, line_number: int) -> Optional[float]:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise RuntimeError(f"{path}:{line_number} 字段 {field} 不是数字：{value}") from exc


def read_rows(path: Path) -> dict[str, EvalRow]:
    if not path.exists():
        raise RuntimeError(f"评估 CSV 不存在：{path}")

    rows: dict[str, EvalRow] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_fields = {"image", *COUNT_FIELDS, "avg_iou"} - set(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise RuntimeError(f"{path} 缺少字段：{missing}")

        for line_number, row in enumerate(reader, start=2):
            image = (row.get("image") or "").strip()
            if not image:
                raise RuntimeError(f"{path}:{line_number} image 不能为空")
            if image in rows:
                raise RuntimeError(f"{path}:{line_number} 重复图片：{image}")

            avg_iou = parse_optional_float(row.get("avg_iou") or "", "avg_iou", path, line_number)
            rows[image] = EvalRow(
                image=image,
                gt=parse_int(row["gt"], "gt", path, line_number),
                pred=parse_int(row["pred"], "pred", path, line_number),
                matched=parse_int(row["matched"], "matched", path, line_number),
                false_positive=parse_int(row["false_positive"], "false_positive", path, line_number),
                false_negative=parse_int(row["false_negative"], "false_negative", path, line_number),
                oversized=parse_int(row["oversized"], "oversized", path, line_number),
                avg_iou=avg_iou,
            )
    return rows


def detector_eval_command(name: str, path: Path) -> str:
    text = f"{name} {path}".lower()
    if "rf-detr" in text or "rfdetr" in text:
        return ".\\evaluate_rfdetr.bat"
    if "yolo" in text:
        return ".\\evaluate_pipe_yolo.bat"
    return "重新运行对应检测器的评估脚本"


def missing_csv_messages(left_csv: Path, right_csv: Path, left_name: str, right_name: str) -> list[str]:
    messages: list[str] = []
    for label, path in ((left_name, left_csv), (right_name, right_csv)):
        if path.exists():
            continue
        command = detector_eval_command(label, path)
        messages.append(f"缺少 {label} 分析 CSV：{path}")
        messages.append(f"请先生成该结果：{command}")
    return messages


def summarize(rows: dict[str, EvalRow]) -> EvalSummary:
    gt = sum(item.gt for item in rows.values())
    pred = sum(item.pred for item in rows.values())
    matched = sum(item.matched for item in rows.values())
    false_positive = sum(item.false_positive for item in rows.values())
    false_negative = sum(item.false_negative for item in rows.values())
    oversized = sum(item.oversized for item in rows.values())

    weighted_iou = 0.0
    weighted_count = 0
    for item in rows.values():
        if item.avg_iou is None or item.matched <= 0:
            continue
        weighted_iou += item.avg_iou * item.matched
        weighted_count += item.matched

    return EvalSummary(
        image_count=len(rows),
        gt=gt,
        pred=pred,
        matched=matched,
        false_positive=false_positive,
        false_negative=false_negative,
        oversized=oversized,
        avg_iou=weighted_iou / weighted_count if weighted_count else None,
        precision=matched / pred if pred else 0.0,
        recall=matched / gt if gt else 0.0,
    )


def format_optional(value: Optional[float]) -> str:
    return "无" if value is None else f"{value:.3f}"


def print_summary(name: str, summary: EvalSummary) -> None:
    print(f"{name}:")
    print(f"  图片数：{summary.image_count}")
    print(f"  标注目标数：{summary.gt}")
    print(f"  预测目标数：{summary.pred}")
    print(f"  匹配成功数：{summary.matched}")
    print(f"  误检数：{summary.false_positive}")
    print(f"  漏检数：{summary.false_negative}")
    print(f"  框偏大匹配数：{summary.oversized}")
    print(f"  平均 IoU：{format_optional(summary.avg_iou)}")
    print(f"  Precision：{summary.precision:.3f}")
    print(f"  Recall：{summary.recall:.3f}")


def summary_to_dict(summary: EvalSummary) -> dict[str, object]:
    return {
        "image_count": summary.image_count,
        "gt": summary.gt,
        "pred": summary.pred,
        "matched": summary.matched,
        "false_positive": summary.false_positive,
        "false_negative": summary.false_negative,
        "oversized": summary.oversized,
        "avg_iou": summary.avg_iou,
        "precision": summary.precision,
        "recall": summary.recall,
    }


def diff_to_dict(left_summary: EvalSummary, right_summary: EvalSummary) -> dict[str, object]:
    return {
        "precision_delta_right_minus_left": right_summary.precision - left_summary.precision,
        "recall_delta_right_minus_left": right_summary.recall - left_summary.recall,
        "false_positive_delta_right_minus_left": right_summary.false_positive - left_summary.false_positive,
        "false_negative_delta_right_minus_left": right_summary.false_negative - left_summary.false_negative,
        "oversized_delta_right_minus_left": right_summary.oversized - left_summary.oversized,
    }


def comparison_verdict(left_name: str, right_name: str, left_summary: EvalSummary, right_summary: EvalSummary) -> dict[str, str]:
    left_issues = left_summary.false_positive + left_summary.false_negative + left_summary.oversized
    right_issues = right_summary.false_positive + right_summary.false_negative + right_summary.oversized
    precision_delta = right_summary.precision - left_summary.precision
    recall_delta = right_summary.recall - left_summary.recall

    if left_issues == right_issues and abs(precision_delta) < 0.001 and abs(recall_delta) < 0.001:
        return {
            "winner": "tie",
            "label": "基本持平",
            "reason": "两侧问题数、Precision 和 Recall 基本一致。",
        }

    if right_summary.false_negative < left_summary.false_negative:
        return {
            "winner": "right",
            "label": f"{right_name} 更优",
            "reason": f"{right_name} 漏检更少。",
        }
    if left_summary.false_negative < right_summary.false_negative:
        return {
            "winner": "left",
            "label": f"{left_name} 更优",
            "reason": f"{left_name} 漏检更少。",
        }
    if right_issues < left_issues:
        return {
            "winner": "right",
            "label": f"{right_name} 更优",
            "reason": f"{right_name} 总问题数更少。",
        }
    if left_issues < right_issues:
        return {
            "winner": "left",
            "label": f"{left_name} 更优",
            "reason": f"{left_name} 总问题数更少。",
        }
    if precision_delta > 0.01 and recall_delta >= -0.01:
        return {
            "winner": "right",
            "label": f"{right_name} 略优",
            "reason": f"{right_name} Precision 更高且 Recall 未明显下降。",
        }
    if precision_delta < -0.01 and recall_delta <= 0.01:
        return {
            "winner": "left",
            "label": f"{left_name} 略优",
            "reason": f"{left_name} Precision 更高且 Recall 未明显下降。",
        }
    return {
        "winner": "mixed",
        "label": "结果混合",
        "reason": "Precision、Recall 或问题数各有优劣，需要结合问题样例判断。",
    }


def write_comparison(
    path: Path,
    left_name: str,
    right_name: str,
    left_rows: dict[str, EvalRow],
    right_rows: dict[str, EvalRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(set(left_rows) | set(right_rows))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                f"{left_name}_gt",
                f"{right_name}_gt",
                f"{left_name}_pred",
                f"{right_name}_pred",
                f"{left_name}_matched",
                f"{right_name}_matched",
                f"{left_name}_issues",
                f"{right_name}_issues",
                "issue_delta_right_minus_left",
                "note",
            ],
        )
        writer.writeheader()
        for image in images:
            left = left_rows.get(image)
            right = right_rows.get(image)
            if left is None:
                writer.writerow({"image": image, "note": f"only_in_{right_name}"})
                continue
            if right is None:
                writer.writerow({"image": image, "note": f"only_in_{left_name}"})
                continue
            writer.writerow(
                {
                    "image": image,
                    f"{left_name}_gt": left.gt,
                    f"{right_name}_gt": right.gt,
                    f"{left_name}_pred": left.pred,
                    f"{right_name}_pred": right.pred,
                    f"{left_name}_matched": left.matched,
                    f"{right_name}_matched": right.matched,
                    f"{left_name}_issues": left.issue_score,
                    f"{right_name}_issues": right.issue_score,
                    "issue_delta_right_minus_left": right.issue_score - left.issue_score,
                    "note": "",
                }
            )


def changed_examples(
    left_name: str,
    right_name: str,
    left_rows: dict[str, EvalRow],
    right_rows: dict[str, EvalRow],
    max_examples: int,
) -> list[dict[str, object]]:
    changed = []
    for image in sorted(set(left_rows) & set(right_rows)):
        left = left_rows[image]
        right = right_rows[image]
        delta = right.issue_score - left.issue_score
        if delta != 0 or left.matched != right.matched or left.pred != right.pred:
            changed.append((abs(delta), image, left, right, delta))

    examples: list[dict[str, object]] = []
    for _, image, left, right, delta in sorted(changed, reverse=True)[: max(0, max_examples)]:
        direction = "worse" if delta > 0 else "better" if delta < 0 else "changed"
        examples.append(
            {
                "image": image,
                "direction": direction,
                "issue_delta_right_minus_left": delta,
                "left_matched": left.matched,
                "left_pred": left.pred,
                "left_issues": left.issue_score,
                "right_matched": right.matched,
                "right_pred": right.pred,
                "right_issues": right.issue_score,
                "left_name": left_name,
                "right_name": right_name,
            }
        )
    return examples


def print_changed_examples(examples: list[dict[str, object]]) -> None:
    if not examples:
        print("差异样例：无")
        return

    print("差异样例：")
    for example in examples:
        direction = {"worse": "更差", "better": "更好"}.get(str(example["direction"]), "变化")
        print(
            f"- {example['image']}: {example['right_name']} 相比 {example['left_name']} {direction}，"
            f"{example['left_name']}=匹配{example['left_matched']}/预测{example['left_pred']}/问题{example['left_issues']}，"
            f"{example['right_name']}=匹配{example['right_matched']}/预测{example['right_pred']}/问题{example['right_issues']}"
        )


def write_summary_json(
    path: Path,
    left_csv: Path,
    right_csv: Path,
    left_name: str,
    right_name: str,
    left_summary: EvalSummary,
    right_summary: EvalSummary,
    missing_left: list[str],
    missing_right: list[str],
    examples: list[dict[str, object]],
) -> None:
    verdict = comparison_verdict(left_name, right_name, left_summary, right_summary)
    payload = {
        "left_name": left_name,
        "right_name": right_name,
        "left_csv": str(left_csv),
        "right_csv": str(right_csv),
        "left": summary_to_dict(left_summary),
        "right": summary_to_dict(right_summary),
        "diff": diff_to_dict(left_summary, right_summary),
        "verdict": verdict,
        "only_in_left": missing_right,
        "only_in_right": missing_left,
        "changed_examples": examples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary_markdown(
    path: Path,
    left_csv: Path,
    right_csv: Path,
    left_name: str,
    right_name: str,
    left_summary: EvalSummary,
    right_summary: EvalSummary,
    missing_left: list[str],
    missing_right: list[str],
    examples: list[dict[str, object]],
) -> None:
    diff = diff_to_dict(left_summary, right_summary)
    verdict = comparison_verdict(left_name, right_name, left_summary, right_summary)
    lines = [
        "# 检测后端对比摘要",
        "",
        f"- 左侧结果：{left_name} (`{left_csv}`)",
        f"- 右侧结果：{right_name} (`{right_csv}`)",
        f"- 对比结论：{verdict['label']}（{verdict['reason']}）",
        f"- {left_name}：Precision {left_summary.precision:.3f}，Recall {left_summary.recall:.3f}，误检 {left_summary.false_positive}，漏检 {left_summary.false_negative}",
        f"- {right_name}：Precision {right_summary.precision:.3f}，Recall {right_summary.recall:.3f}，误检 {right_summary.false_positive}，漏检 {right_summary.false_negative}",
        (
            "- 差值（右侧减左侧）："
            f"Precision {diff['precision_delta_right_minus_left']:+.3f}，"
            f"Recall {diff['recall_delta_right_minus_left']:+.3f}，"
            f"误检 {diff['false_positive_delta_right_minus_left']:+d}，"
            f"漏检 {diff['false_negative_delta_right_minus_left']:+d}，"
            f"框偏大 {diff['oversized_delta_right_minus_left']:+d}"
        ),
        f"- 只在 {left_name} 中出现的图片数：{len(missing_right)}",
        f"- 只在 {right_name} 中出现的图片数：{len(missing_left)}",
        "",
        "## 差异样例",
        "",
    ]
    if examples:
        for example in examples:
            direction = {"worse": "更差", "better": "更好"}.get(str(example["direction"]), "变化")
            lines.append(
                f"- {example['image']}：{right_name} 相比 {left_name} {direction}，"
                f"{left_name}=匹配{example['left_matched']}/预测{example['left_pred']}/问题{example['left_issues']}，"
                f"{right_name}=匹配{example['right_matched']}/预测{example['right_pred']}/问题{example['right_issues']}"
            )
    else:
        lines.append("- 无明显差异样例。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    left_csv = Path(args.left_csv)
    right_csv = Path(args.right_csv)
    missing_messages = missing_csv_messages(left_csv, right_csv, args.left_name, args.right_name)
    if missing_messages:
        print("检测结果对比失败：缺少评估输入。")
        for message in missing_messages:
            print(f"- {message}")
        return 2

    try:
        left_rows = read_rows(left_csv)
        right_rows = read_rows(right_csv)
    except RuntimeError as exc:
        print(f"检测结果对比失败：{exc}")
        return 2

    left_summary = summarize(left_rows)
    right_summary = summarize(right_rows)
    verdict = comparison_verdict(args.left_name, args.right_name, left_summary, right_summary)
    print_summary(args.left_name, left_summary)
    print_summary(args.right_name, right_summary)
    print("总体差异：")
    print(f"  Precision 差值：{right_summary.precision - left_summary.precision:+.3f}")
    print(f"  Recall 差值：{right_summary.recall - left_summary.recall:+.3f}")
    print(f"  误检差值：{right_summary.false_positive - left_summary.false_positive:+d}")
    print(f"  漏检差值：{right_summary.false_negative - left_summary.false_negative:+d}")
    print(f"  框偏大差值：{right_summary.oversized - left_summary.oversized:+d}")
    print(f"  对比结论：{verdict['label']}（{verdict['reason']}）")
    examples = changed_examples(args.left_name, args.right_name, left_rows, right_rows, args.max_examples)
    print_changed_examples(examples)

    missing_left = sorted(set(right_rows) - set(left_rows))
    missing_right = sorted(set(left_rows) - set(right_rows))
    if missing_left:
        print(f"只在 {args.right_name} 中出现的图片数：{len(missing_left)}")
    if missing_right:
        print(f"只在 {args.left_name} 中出现的图片数：{len(missing_right)}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        write_comparison(output_path, args.left_name, args.right_name, left_rows, right_rows)
        print(f"对比 CSV：{output_path}")

    if args.summary_json:
        summary_json = Path(args.summary_json)
        write_summary_json(
            summary_json,
            left_csv,
            right_csv,
            args.left_name,
            args.right_name,
            left_summary,
            right_summary,
            missing_left,
            missing_right,
            examples,
        )
        print(f"对比 JSON 摘要：{summary_json}")

    if args.summary_md:
        summary_md = Path(args.summary_md)
        write_summary_markdown(
            summary_md,
            left_csv,
            right_csv,
            args.left_name,
            args.right_name,
            left_summary,
            right_summary,
            missing_left,
            missing_right,
            examples,
        )
        print(f"对比 Markdown 摘要：{summary_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
