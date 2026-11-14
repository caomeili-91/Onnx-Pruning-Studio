r"""VOC 验证集基线评估入口：加载未剪枝 FCN，打印 mIoU 与每类 IoU。

运行（必须使用 FCN 环境 + GPU）：
    D:\anaconda\envs\FCN\python.exe src/eval/run_baseline.py --split val.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluator import evaluate_detailed
from src.models.fcn_loader import build_fcn, load_checkpoint, resolve_path
from utils.my_dataset import VOCData
from utils.segmentation_transforms import get_transform

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="未剪枝 FCN 的 VOC 基线 mIoU")
    parser.add_argument("--weights", default="weights/fcn_resnet50_coco.pth")
    parser.add_argument("--voc-root", default="VOCdevkit")
    parser.add_argument("--split", default="val.txt", help="ImageSets/Segmentation 下的划分")
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=1, help="1 可避免不同尺寸 padding 造成的算力浪费")
    parser.add_argument("--workers", type=int, default=0, help="Windows 下建议 0，避免多进程启动开销")
    parser.add_argument("--max-batches", type=int, default=0, help="只跑前 N 个 batch（0=全部）")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--aux-loss", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    weights_path = resolve_path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"权重文件不存在: {weights_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        print(f"设备: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"设备: {device}（未使用 GPU）")

    state_dict = load_checkpoint(weights_path)
    aux_keys = [key for key in state_dict if key.startswith("aux_classifier.")]
    aux_loss = bool(aux_keys) if args.aux_loss == "auto" else args.aux_loss == "on"

    model = build_fcn(num_classes=args.num_classes, aux_loss=aux_loss)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"权重: {weights_path}")
    print(f"aux_loss: {aux_loss}（checkpoint 中 aux_classifier.*: {len(aux_keys)}）")
    print(f"missing: {len(missing)}，unexpected: {len(unexpected)}")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise SystemExit(f"strict=True 加载失败:\n{exc}") from exc
    print("strict=True 加载成功")
    model.to(device).eval()

    dataset = VOCData(
        str(resolve_path(args.voc_root)),
        transforms=get_transform(train=False),
        txt_name=args.split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    print("预处理: utils/segmentation_transforms.get_transform(train=False)（短边 520 + ImageNet 归一化）")
    print(f"验证集: {args.split}，{len(dataset)} 张，batch_size={args.batch_size}，workers={args.workers}")
    print("指标计算: 只使用主输出 out，忽略 aux")

    result = evaluate_detailed(
        model,
        dataloader,
        device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        progress=not args.no_progress,
        desc=f"VOC {args.split}",
        max_batches=(args.max_batches or None),
    )

    per_class_iou = result["per_class_iou"]
    print("per-class IoU:")
    for class_id, class_name in enumerate(VOC_CLASSES[: args.num_classes]):
        value = float(per_class_iou[class_id])
        if math.isnan(value):
            print(f"  {class_id:2d} {class_name:12s}    n/a")
        else:
            print(f"  {class_id:2d} {class_name:12s} {value * 100:6.2f}%")

    elapsed = result["elapsed_sec"]
    throughput = result["num_images"] / elapsed if elapsed > 0 else 0.0
    print(f"有效像素: {result['num_pixels']}")
    print(f"Pixel Accuracy: {result['pixel_accuracy'] * 100:.2f}%")
    print(f"mIoU: {result['miou'] * 100:.2f}%  （{result['num_images']} 张，{elapsed:.1f}s，{throughput:.1f} 张/s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
