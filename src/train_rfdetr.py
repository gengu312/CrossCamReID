from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RF-DETR model on the exported CrossCamReID COCO dataset.")
    parser.add_argument("--dataset-dir", default="datasets/pipe_rfdetr", help="RF-DETR COCO dataset directory.")
    parser.add_argument("--output-dir", default="runs_rfdetr/pipe_rfdetr_nano", help="Training output directory.")
    parser.add_argument("--model-size", choices=tuple(MODEL_CLASSES.keys()), default="nano")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Override class count. Defaults to the COCO category count.",
    )
    parser.add_argument("--pretrain-weights", default="", help="Optional initial checkpoint path.")
    parser.add_argument("--resume", default="", help="Optional checkpoint.pth path to resume training.")
    parser.add_argument("--check-only", action="store_true", help="Only validate dataset structure; do not train.")
    parser.add_argument("--print-only", action="store_true", help="Print the training configuration; do not train.")
    return parser.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"缺少 COCO 标注文件：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"COCO JSON 无法解析：{path}") from exc
    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise RuntimeError(f"{path} 缺少列表字段：{key}")
    return data


def validate_dataset(dataset_dir: Path) -> int:
    train = load_coco(dataset_dir / "train" / "_annotations.coco.json")
    valid = load_coco(dataset_dir / "valid" / "_annotations.coco.json")
    test_path = dataset_dir / "test" / "_annotations.coco.json"
    test = None
    if test_path.exists():
        test = load_coco(test_path)

    if not train["images"]:
        raise RuntimeError("train split 没有图片。")
    if not valid["images"]:
        raise RuntimeError("valid split 没有图片。")
    if not train["categories"]:
        raise RuntimeError("COCO categories 不能为空。")

    expected_categories = {item.get("id") for item in train["categories"]}
    if any(category_id is None for category_id in expected_categories):
        raise RuntimeError("train categories 中存在缺少 id 的类别。")

    split_data = [("train", train), ("valid", valid)]
    if test is not None:
        split_data.append(("test", test))

    for split_name, data in split_data:
        split_dir = dataset_dir / split_name
        missing = [item["file_name"] for item in data["images"] if not (split_dir / item["file_name"]).exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(f"{split_name} 有 {len(missing)} 张 JSON 图片文件不存在：{preview}")
        category_ids = {item.get("id") for item in data["categories"]}
        if category_ids != expected_categories:
            raise RuntimeError(f"{split_name} categories 和 train 不一致。")
        image_ids = {item.get("id") for item in data["images"]}
        if any(image_id is None for image_id in image_ids):
            raise RuntimeError(f"{split_name} images 中存在缺少 id 的图片。")
        for annotation in data["annotations"]:
            if annotation.get("image_id") not in image_ids:
                raise RuntimeError(f"{split_name} annotation 引用了不存在的 image_id：{annotation.get('image_id')}")
            if annotation.get("category_id") not in expected_categories:
                raise RuntimeError(f"{split_name} annotation 引用了不存在的 category_id：{annotation.get('category_id')}")

    print(
        "RF-DETR 数据集检查通过："
        f"train={len(train['images'])}/{len(train['annotations'])}，"
        f"valid={len(valid['images'])}/{len(valid['annotations'])}，"
        f"classes={len(train['categories'])}"
    )
    return len(train["categories"])


def validate_training_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise RuntimeError("--epochs 必须大于 0。")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size 必须大于 0。")
    if args.grad_accum_steps <= 0:
        raise RuntimeError("--grad-accum-steps 必须大于 0。")
    if args.lr <= 0:
        raise RuntimeError("--lr 必须大于 0。")
    if args.num_classes < 0:
        raise RuntimeError("--num-classes 不能小于 0。")
    if args.pretrain_weights and not Path(args.pretrain_weights).exists():
        raise RuntimeError(f"pretrain 权重不存在：{args.pretrain_weights}")
    if args.resume and not Path(args.resume).exists():
        raise RuntimeError(f"resume 权重不存在：{args.resume}")


def print_config(args: argparse.Namespace, num_classes: int) -> None:
    print("RF-DETR 训练配置：")
    print(f"  dataset_dir={args.dataset_dir}")
    print(f"  output_dir={args.output_dir}")
    print(f"  model_size={args.model_size}")
    print(f"  num_classes={num_classes}")
    print(f"  epochs={args.epochs}")
    print(f"  batch_size={args.batch_size}")
    print(f"  grad_accum_steps={args.grad_accum_steps}")
    print(f"  lr={args.lr}")
    if args.pretrain_weights:
        print(f"  pretrain_weights={args.pretrain_weights}")
    if args.resume:
        print(f"  resume={args.resume}")


def load_model_class(model_size: str):
    try:
        import rfdetr
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR training requires rfdetr. Install it with: "
            "python -m pip install -r requirements-rfdetr.txt"
        ) from exc

    class_name = MODEL_CLASSES[model_size]
    model_class = getattr(rfdetr, class_name, None)
    if model_class is None:
        raise RuntimeError(f"Installed rfdetr does not provide {class_name}.")
    return model_class


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    try:
        validate_training_args(args)
        dataset_classes = validate_dataset(dataset_dir)
        num_classes = args.num_classes if args.num_classes > 0 else dataset_classes
        if num_classes < dataset_classes:
            raise RuntimeError(f"--num-classes={num_classes} 小于数据集类别数 {dataset_classes}。")
        print_config(args, num_classes)
        if args.check_only:
            print("CheckOnly：未启动 RF-DETR 训练。")
            return 0
        if args.print_only:
            print("PrintOnly：未启动 RF-DETR 训练。")
            return 0

        model_class = load_model_class(args.model_size)
        model_kwargs: dict[str, Any] = {"num_classes": num_classes}
        if args.pretrain_weights:
            model_kwargs["pretrain_weights"] = args.pretrain_weights
        model = model_class(**model_kwargs)
        train_kwargs: dict[str, Any] = {
            "dataset_dir": str(dataset_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "output_dir": str(output_dir),
        }
        if args.resume:
            train_kwargs["resume"] = args.resume
        model.train(**train_kwargs)
    except RuntimeError as exc:
        print(f"RF-DETR 训练入口失败：{exc}")
        return 2

    print(f"RF-DETR 训练完成，输出目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
