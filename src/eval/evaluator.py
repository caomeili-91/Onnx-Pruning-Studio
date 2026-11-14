r"""VOC 语义分割评估工具：混淆矩阵累加，返回 mIoU。

模型前向输出的兼容处理（严格只用主输出）：
- dict / OrderedDict：取 ["out"]
- tuple / list：取 [0]，例如 (out, aux)
- 单个 Tensor：直接使用

aux 输出永远不会参与混淆矩阵。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm

from src.eval.metrics import ConfusionMatrix


def extract_main_logits(output: Any) -> torch.Tensor:
    """从模型输出中取出主分割 logits，忽略 aux。"""
    if isinstance(output, dict):
        if "out" not in output:
            raise KeyError(f"模型输出 dict 中没有 'out'，实际 keys: {list(output.keys())}")
        logits = output["out"]
    elif isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise ValueError("模型输出为空 tuple/list")
        logits = output[0]  # (out, aux) -> out
    elif torch.is_tensor(output):
        logits = output
    else:
        raise TypeError(f"无法识别的模型输出类型: {type(output)!r}")

    if not torch.is_tensor(logits):
        raise TypeError(f"主输出不是 Tensor: {type(logits)!r}")
    return logits


@torch.no_grad()
def evaluate_detailed(
    model: torch.nn.Module,
    dataloader,
    device,
    num_classes: int = 21,
    ignore_index: int = 255,
    progress: bool = True,
    desc: str = "evaluate",
    max_batches: int | None = None,
) -> dict:
    """遍历 dataloader，用 21x21 混淆矩阵累加计算指标。

    Returns:
        dict: miou（0~1）、per_class_iou（Tensor）、pixel_accuracy、
              num_images、num_pixels、confusion_matrix、elapsed_sec
    """
    device = torch.device(device)
    model.eval()
    confusion = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)

    iterator = tqdm(dataloader, desc=desc, unit="batch", mininterval=2.0) if progress else dataloader
    num_images = 0
    start = time.perf_counter()

    for batch_idx, (images, targets) in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = extract_main_logits(model(images))
        if logits.shape[1] != num_classes:
            raise ValueError(
                f"主输出通道数 {logits.shape[1]} 与 num_classes={num_classes} 不一致"
            )

        predictions = logits.argmax(dim=1)
        confusion.update(predictions, targets)
        num_images += int(images.shape[0])

        if progress:
            iterator.set_postfix(imgs=num_images)

    elapsed = time.perf_counter() - start
    per_class_iou, miou = confusion.compute_miou()
    return {
        "miou": miou,
        "per_class_iou": per_class_iou,
        "pixel_accuracy": confusion.pixel_accuracy,
        "num_images": num_images,
        "num_pixels": confusion.num_pixels,
        "confusion_matrix": confusion.matrix,
        "elapsed_sec": elapsed,
    }


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device,
    num_classes: int = 21,
    ignore_index: int = 255,
    progress: bool = True,
) -> float:
    """遍历验证集，返回 mIoU（0~1 比例），只使用主输出 out。"""
    result = evaluate_detailed(
        model,
        dataloader,
        device,
        num_classes=num_classes,
        ignore_index=ignore_index,
        progress=progress,
    )
    return result["miou"]
