"""语义分割指标：混淆矩阵与 mIoU（纯 PyTorch 实现，不依赖 torchmetrics）。

IoU 定义（类别 c）：
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
其中 TP/FP/FN 由整个验证集的 21x21 混淆矩阵累加后得到，
不是"逐张图算 IoU 再平均"，也不是 pixel accuracy。

标签中等于 ignore_index（默认 255）的像素不参与任何统计。
"""

from __future__ import annotations

from typing import Tuple

import torch


def compute_miou(confusion_matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """由混淆矩阵计算 per-class IoU 与 mIoU。

    Args:
        confusion_matrix: (C, C) 混淆矩阵，行是 GT，列是预测，整数计数。

    Returns:
        per_class_iou: (C,) float64 张量，取值 0~1；
                       数据集中既无 GT 也无预测的类别记为 NaN
        miou: 忽略 NaN 后的平均 IoU，取值 0~1（打印时乘 100 显示为百分比）
    """
    if confusion_matrix.dim() != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(f"混淆矩阵必须是方阵，收到 {tuple(confusion_matrix.shape)}")

    # float64 避免大像素计数下的精度问题
    cm = confusion_matrix.to(dtype=torch.float64, device="cpu")
    num_classes = cm.shape[0]

    true_positive = torch.diag(cm)                 # TP
    false_positive = cm.sum(dim=0) - true_positive  # FP：预测成 c 但不是 c
    false_negative = cm.sum(dim=1) - true_positive  # FN：是 c 但没预测成 c
    union = true_positive + false_positive + false_negative

    per_class_iou = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    valid = union > 0
    per_class_iou[valid] = true_positive[valid] / union[valid]

    miou = float(per_class_iou[valid].mean()) if bool(valid.any()) else float("nan")
    return per_class_iou, miou


class ConfusionMatrix:
    """C x C 混淆矩阵，默认 21 类（VOC：背景 + 20 个前景类），忽略 255。

    用法::

        cm = ConfusionMatrix(num_classes=21, ignore_index=255)
        for images, targets in dataloader:
            preds = model(images)["out"].argmax(dim=1)
            cm.update(preds, targets)
        per_class_iou, miou = cm.compute_miou()
    """

    def __init__(self, num_classes: int = 21, ignore_index: int = 255) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes 必须为正整数，收到 {num_classes}")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @property
    def num_pixels(self) -> int:
        """参与统计的有效像素总数（已排除 ignore_index）。"""
        return int(self.matrix.sum().item())

    @property
    def pixel_accuracy(self) -> float:
        """全局像素准确率（仅用于 sanity check，不能替代 mIoU）。"""
        total = self.num_pixels
        if total == 0:
            return 0.0
        return float(torch.diag(self.matrix).sum().item() / total)

    @property
    def support(self) -> torch.Tensor:
        """每个类别的 GT 像素数（混淆矩阵行和）。"""
        return self.matrix.sum(dim=1)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """累加一个 batch 的预测结果。

        Args:
            pred: (N, H, W) 或 (H, W) 的预测类别 id
            target: 与 pred 同形状的 GT 类别 id，255 表示忽略
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred 与 target 形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}")

        pred = pred.reshape(-1).to(dtype=torch.int64)
        target = target.reshape(-1).to(dtype=torch.int64)

        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (pred >= 0)
            & (pred < self.num_classes)
        )
        if not bool(valid.any()):
            return

        indices = target[valid] * self.num_classes + pred[valid]
        histogram = torch.bincount(indices, minlength=self.num_classes ** 2)
        # 混淆矩阵始终留在 CPU，避免占用显存
        self.matrix += histogram.reshape(self.num_classes, self.num_classes).to(dtype=torch.int64).cpu()

    def reset(self) -> None:
        self.matrix.zero_()

    def compute_iou(self) -> torch.Tensor:
        """返回 per-class IoU（0~1，未见类别为 NaN）。"""
        per_class_iou, _ = compute_miou(self.matrix)
        return per_class_iou

    def compute_miou(self) -> Tuple[torch.Tensor, float]:
        """返回 (per-class IoU, mIoU)，与模块级 compute_miou 一致。"""
        return compute_miou(self.matrix)

    def __repr__(self) -> str:
        return (
            f"ConfusionMatrix(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, num_pixels={self.num_pixels})"
        )
