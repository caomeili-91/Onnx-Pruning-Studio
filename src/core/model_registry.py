"""内置语义分割模型注册表。

新增内置模型只需在这里加一个 builder：
    builder(num_classes, aux_loss) -> nn.Module
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

Builder = Callable[..., nn.Module]


def _build_fcn(num_classes: int, aux_loss: bool = True) -> nn.Module:
    from src.models.fcn_loader import build_fcn

    return build_fcn(num_classes=num_classes, aux_loss=aux_loss)


def _build_deeplab(num_classes: int, aux_loss: bool = True) -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    return deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        aux_loss=aux_loss,
    )


def _build_unet(num_classes: int, aux_loss: bool = True) -> nn.Module:
    from src.models.unet import UNet

    return UNet(num_classes=num_classes)


BUILTIN_MODELS: Dict[str, Builder] = {
    "FCN-ResNet50 (VOC 21类)": _build_fcn,
    "DeepLabV3-ResNet50 (torchvision)": _build_deeplab,
    "UNet (轻量, 需权重)": _build_unet,
}


def list_builtin_models() -> list[str]:
    return list(BUILTIN_MODELS.keys())


def build_builtin_model(name: str, num_classes: int, aux_loss: bool = True) -> nn.Module:
    if name not in BUILTIN_MODELS:
        raise KeyError(f"未知内置模型: {name}")
    return BUILTIN_MODELS[name](num_classes=num_classes, aux_loss=aux_loss)
