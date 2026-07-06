from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
}


@dataclass(frozen=True)
class Prediction:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR predictions and export labels compatible with analyze_yolo_eval.py."
    )
    parser.add_argument("--dataset-root", default="datasets/pipe_yolo", help="Prepared YOLO dataset root.")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Dataset split to evaluate.")
    parser.add_argument("--source", help="Override image source directory.")
    parser.add_argument(
        "--output-labels",
        default="runs_rfdetr_eval/pipe_rfdetr_nano_eval/labels",
        help="Directory for exported YOLO-format prediction labels.",
    )
    parser.add_argument("--model-size", choices=tuple(MODEL_CLASSES.keys()), default="nano")
    parser.add_argument("--weights", default="", help="Optional RF-DETR checkpoint path.")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Class count for RF-DETR model initialization. Defaults to data.yaml nc or label scan.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="RF-DETR confidence threshold.")
    parser.add_argument(
        "--classes",
        default="",
        help="Optional comma-separated YOLO class ids to keep after class-id mapping.",
    )
    parser.add_argument(
        "--class-id-mode",
        choices=("auto", "zero", "category"),
        default="auto",
        help="How to map RF-DETR class ids to YOLO class ids.",
    )
    parser.add_argument(
        "--category-id-offset",
        type=int,
        default=1,
        help="COCO category id offset used during export_yolo_to_coco.py.",
    )
    parser.add_argument("--optimize", action="store_true", help="Call optimize_for_inference() when available.")
    parser.add_argument("--clean", action="store_true", help="Remove old .txt files from output-labels first.")
    parser.add_argument("--check-only", action="store_true", help="Validate paths only; do not import or run RF-DETR.")
    parser.add_argument("--print-only", action="store_true", help="Print configuration only; do not run RF-DETR.")
    return parser.parse_args()


def parse_class_ids(value: str) -> set[int] | None:
    if not value.strip():
        return None
    classes: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            class_id = int(part)
        except ValueError as exc:
            raise RuntimeError(f"类别列表包含非整数：{part}") from exc
        if class_id < 0:
            raise RuntimeError(f"类别列表不能包含负数：{class_id}")
        classes.add(class_id)
    return classes or None


def list_images(source: Path) -> list[Path]:
    if not source.exists():
        raise RuntimeError(f"图片目录不存在：{source}")
    images = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"图片目录中没有可评估图片：{source}")
    return images


def infer_num_classes(dataset_root: Path) -> int | None:
    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("nc:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError as exc:
                    raise RuntimeError(f"{data_yaml} 的 nc 不是整数。") from exc

    max_class_id = -1
    labels_root = dataset_root / "labels"
    for label_path in labels_root.glob("*/*.txt"):
        for raw_line in label_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                class_id = int(float(line.split()[0]))
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"{label_path} 中存在无效类别。") from exc
            max_class_id = max(max_class_id, class_id)
    return max_class_id + 1 if max_class_id >= 0 else None


def validate_eval_args(args: argparse.Namespace) -> None:
    if args.num_classes < 0:
        raise RuntimeError("--num-classes 不能小于 0。")
    if not 0.0 <= args.conf <= 1.0:
        raise RuntimeError("--conf 必须在 0 到 1 之间。")
    if args.category_id_offset < 0:
        raise RuntimeError("--category-id-offset 不能小于 0。")
    if args.weights and not Path(args.weights).exists():
        raise RuntimeError(f"RF-DETR 权重不存在：{args.weights}")


def clean_old_labels(output_labels: Path) -> None:
    output_labels.mkdir(parents=True, exist_ok=True)
    for label_path in output_labels.glob("*.txt"):
        label_path.unlink()


def map_class_id(raw_class_id: int, mode: str, category_id_offset: int) -> int:
    if mode == "zero":
        return raw_class_id
    if mode == "category":
        return raw_class_id - category_id_offset
    if raw_class_id >= category_id_offset:
        return raw_class_id - category_id_offset
    return raw_class_id


def load_model_class(model_size: str):
    try:
        import rfdetr
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR evaluation requires rfdetr. Install it with: "
            "python -m pip install -r requirements-rfdetr.txt"
        ) from exc

    class_name = MODEL_CLASSES[model_size]
    model_class = getattr(rfdetr, class_name, None)
    if model_class is None:
        raise RuntimeError(f"Installed rfdetr does not provide {class_name}.")
    return model_class


def prediction_arrays(predictions: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    xyxy = np.asarray(getattr(predictions, "xyxy", []), dtype=np.float32)
    if xyxy.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), None
    if xyxy.ndim == 1:
        if xyxy.size % 4 != 0:
            raise RuntimeError(f"RF-DETR xyxy 输出长度不是 4 的倍数：{xyxy.size}")
        xyxy = xyxy.reshape(-1, 4)
    elif xyxy.ndim == 2:
        if xyxy.shape[1] < 4:
            raise RuntimeError(f"RF-DETR xyxy 输出列数不足 4：{xyxy.shape[1]}")
        if xyxy.shape[1] > 4:
            xyxy = xyxy[:, :4]
    else:
        raise RuntimeError(f"RF-DETR xyxy 输出维度异常：{xyxy.ndim}")

    box_count = len(xyxy)
    confidence = getattr(predictions, "confidence", None)
    if confidence is None:
        confidence_array = np.ones(box_count, dtype=np.float32)
    else:
        confidence_array = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if len(confidence_array) < box_count:
            confidence_array = np.pad(confidence_array, (0, box_count - len(confidence_array)), constant_values=1.0)

    class_ids = getattr(predictions, "class_id", None)
    if class_ids is None:
        class_id_array = None
    else:
        class_id_array = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        if len(class_id_array) < box_count:
            class_id_array = np.pad(class_id_array, (0, box_count - len(class_id_array)), constant_values=0)
    return xyxy, confidence_array[:box_count], None if class_id_array is None else class_id_array[:box_count]


def predict_image(
    model: Any,
    image_path: Path,
    confidence: float,
    allowed_classes: set[int] | None,
    class_id_mode: str,
    category_id_offset: int,
) -> list[Prediction]:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        predictions = model.predict(np.asarray(rgb_image), threshold=confidence)

    xyxy, confidences, class_ids = prediction_arrays(predictions)
    rows: list[Prediction] = []
    for index, box in enumerate(xyxy):
        raw_class_id = 0 if class_ids is None else int(class_ids[index])
        class_id = map_class_id(raw_class_id, class_id_mode, category_id_offset)
        if class_id < 0:
            continue
        if allowed_classes is not None and class_id not in allowed_classes:
            continue

        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if box_width <= 0.0 or box_height <= 0.0:
            continue

        rows.append(
            Prediction(
                cls=class_id,
                cx=((x1 + x2) / 2.0) / width,
                cy=((y1 + y2) / 2.0) / height,
                w=box_width / width,
                h=box_height / height,
                conf=float(confidences[index]),
            )
        )
    rows.sort(key=lambda item: item.conf, reverse=True)
    return rows


def write_prediction_label(path: Path, predictions: list[Prediction]) -> None:
    lines = [
        f"{item.cls} {item.cx:.6f} {item.cy:.6f} {item.w:.6f} {item.h:.6f} {item.conf:.6f}"
        for item in predictions
    ]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def print_config(
    args: argparse.Namespace,
    source: Path,
    output_labels: Path,
    image_count: int,
    num_classes: int | None,
) -> None:
    print("RF-DETR 评估配置：")
    print(f"  dataset_root={args.dataset_root}")
    print(f"  split={args.split}")
    print(f"  source={source}")
    print(f"  output_labels={output_labels}")
    print(f"  model_size={args.model_size}")
    print(f"  weights={args.weights or '默认预训练权重'}")
    print(f"  num_classes={num_classes if num_classes is not None else '未指定'}")
    print(f"  conf={args.conf}")
    print(f"  class_id_mode={args.class_id_mode}")
    print(f"  category_id_offset={args.category_id_offset}")
    print(f"  image_count={image_count}")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    source = Path(args.source) if args.source else dataset_root / "images" / args.split
    output_labels = Path(args.output_labels)

    try:
        validate_eval_args(args)
        allowed_classes = parse_class_ids(args.classes)
        images = list_images(source)
        inferred_classes = infer_num_classes(dataset_root)
        num_classes = args.num_classes if args.num_classes > 0 else inferred_classes
        if allowed_classes is not None and num_classes is not None:
            invalid_classes = sorted(class_id for class_id in allowed_classes if class_id >= num_classes)
            if invalid_classes:
                preview = ", ".join(str(class_id) for class_id in invalid_classes[:5])
                raise RuntimeError(f"--classes 包含超出类别数 {num_classes} 的类别：{preview}")
        print_config(args, source, output_labels, len(images), num_classes)
        if args.check_only:
            print("CheckOnly：未启动 RF-DETR 预测。")
            return 0
        if args.print_only:
            print("PrintOnly：未启动 RF-DETR 预测。")
            return 0

        if args.clean:
            clean_old_labels(output_labels)
        else:
            output_labels.mkdir(parents=True, exist_ok=True)

        model_class = load_model_class(args.model_size)
        model_kwargs: dict[str, Any] = {}
        if num_classes is not None:
            model_kwargs["num_classes"] = num_classes
        if args.weights:
            model_kwargs["pretrain_weights"] = args.weights
        model = model_class(**model_kwargs)
        if args.optimize and hasattr(model, "optimize_for_inference"):
            optimized = model.optimize_for_inference()
            if optimized is not None:
                model = optimized

        predicted_count = 0
        for image_path in images:
            predictions = predict_image(
                model,
                image_path,
                args.conf,
                allowed_classes,
                args.class_id_mode,
                args.category_id_offset,
            )
            predicted_count += len(predictions)
            write_prediction_label(output_labels / f"{image_path.stem}.txt", predictions)
    except RuntimeError as exc:
        print(f"RF-DETR 评估失败：{exc}")
        return 2

    print(f"RF-DETR 预测完成：图片={len(images)}，预测框={predicted_count}，标签目录={output_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
