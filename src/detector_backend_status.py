from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DetectorBackendStatus:
    detector: str
    dependency_available: bool
    project_weights: str
    project_ready: bool
    detail: str


def module_available(name: str, available_modules: set[str] | None = None) -> bool:
    if available_modules is not None:
        return name in available_modules
    return importlib.util.find_spec(name) is not None


def resolve_existing_path(repo_root: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [repo_root / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_yolo_pipe_weights(repo_root: Path, requested_model: str = "") -> Path | None:
    requested = resolve_existing_path(repo_root, requested_model)
    if requested is not None and requested.name.lower() not in {"yolov8n.pt", "yolo11n.pt"}:
        return requested
    candidates = [
        repo_root / "runs_yolo" / "pipe_yolov8n_hybrid_0710_v2" / "weights" / "best.pt",
        repo_root / "runs_yolo" / "pipe_yolov8n" / "weights" / "best.pt",
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def find_rfdetr_pipe_weights(repo_root: Path, model_size: str, requested_weights: str = "") -> Path | None:
    requested = resolve_existing_path(repo_root, requested_weights)
    if requested is not None:
        return requested
    output_dir = repo_root / "runs_rfdetr" / f"pipe_rfdetr_{model_size}"
    preferred = (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
    )
    for name in preferred:
        candidate = output_dir / name
        if candidate.is_file():
            return candidate.resolve()
    best_matches = sorted(output_dir.glob("checkpoint_best*.pth")) if output_dir.is_dir() else []
    if best_matches:
        return best_matches[0].resolve()
    checkpoints = sorted(output_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True) if output_dir.is_dir() else []
    return checkpoints[0].resolve() if checkpoints else None


def collect_backend_statuses(
    repo_root: Path,
    yolo_model: str = "",
    rfdetr_size: str = "nano",
    rfdetr_weights: str = "",
    available_modules: Iterable[str] | None = None,
) -> dict[str, DetectorBackendStatus]:
    root = repo_root.resolve()
    module_set = set(available_modules) if available_modules is not None else None
    yolo_dependency = module_available("ultralytics", module_set)
    rfdetr_dependency = module_available("rfdetr", module_set)
    yolo_weights = find_yolo_pipe_weights(root, yolo_model)
    rfdetr_pipe_weights = find_rfdetr_pipe_weights(root, rfdetr_size, rfdetr_weights)

    return {
        "motion": DetectorBackendStatus(
            detector="motion",
            dependency_available=True,
            project_weights="",
            project_ready=True,
            detail="内置可用，只适合运动区域检测，静止目标不会持续识别。",
        ),
        "yolo": DetectorBackendStatus(
            detector="yolo",
            dependency_available=yolo_dependency,
            project_weights="" if yolo_weights is None else str(yolo_weights),
            project_ready=yolo_dependency and yolo_weights is not None,
            detail=(
                "项目 YOLO 已就绪。"
                if yolo_dependency and yolo_weights is not None
                else "YOLO 尚未就绪，需要安装 ultralytics 并准备项目训练权重。"
            ),
        ),
        "rfdetr": DetectorBackendStatus(
            detector="rfdetr",
            dependency_available=rfdetr_dependency,
            project_weights="" if rfdetr_pipe_weights is None else str(rfdetr_pipe_weights),
            project_ready=rfdetr_dependency and rfdetr_pipe_weights is not None,
            detail=(
                "项目 RF-DETR 已就绪。"
                if rfdetr_dependency and rfdetr_pipe_weights is not None
                else "RF-DETR 只是可选入口；需要安装 rfdetr 并训练项目专用权重。"
            ),
        ),
    }


def selector_status_text(status: DetectorBackendStatus) -> str:
    if status.detector == "motion":
        return f"后端状态：motion {status.detail}"
    dependency = "依赖已安装" if status.dependency_available else "依赖未安装"
    weights = f"项目权重={status.project_weights}" if status.project_weights else "项目权重未找到"
    readiness = "可用于项目" if status.project_ready else "当前未就绪"
    return f"后端状态：{status.detector} {readiness}；{dependency}；{weights}。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local detector backend readiness.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--rfdetr-size", default="nano")
    parser.add_argument("--rfdetr-weights", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    statuses = collect_backend_statuses(
        Path(args.repo_root),
        yolo_model=args.yolo_model,
        rfdetr_size=args.rfdetr_size,
        rfdetr_weights=args.rfdetr_weights,
    )
    if args.json:
        print(json.dumps({name: asdict(status) for name, status in statuses.items()}, ensure_ascii=False, indent=2))
    else:
        for name in ("motion", "yolo", "rfdetr"):
            print(selector_status_text(statuses[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
