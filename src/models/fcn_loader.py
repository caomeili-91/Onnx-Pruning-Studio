"""加载 VOC 21 类的 FCN-ResNet50 权重，并用项目自带 transforms 做一次前向推理。

结论（脚本会根据 checkpoint 自动判断并打印）：

- weights/fcn_resnet50_coco.pth 是 torchvision `fcn_resnet50(weights=COCO)` 的权重，
  21 类（VOC 的 20 个前景类 + 背景）。
- checkpoint 里有 8 个 `aux_classifier.*` 参数，说明导出时启用了辅助分类器，
  等价于 torchvision 的 `aux_loss=True`。项目里的 `fcn_resnet50()` 既没有
  `aux_loss` 参数，也没有辅助分支，所以直接 `strict=True` 加载会报 8 个
  unexpected keys；本脚本用项目自己的 `resnet50` / `IntermediateLayerGetter` /
  `FCNHead` / `FCN` 拼出带 aux 分支的版本，从而让 `strict=True` 成功加载。

脚本只做「构建 -> 严格加载 -> 单张前向」，不包含任何剪枝逻辑。
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.backbone import resnet50
from model.fcn_model import FCN, FCNHead, IntermediateLayerGetter
from utils.my_dataset import VOCData

NUM_CLASSES = 21
OUT_INPLANES = 2048
AUX_INPLANES = 1024
BASE_SIZE = 520
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FCNWithAux(FCN):
    """在项目 FCN 的基础上补上 torchvision 同款辅助分类器分支。

    Args:
        backbone: 已用 IntermediateLayerGetter 包装、能返回 out 和 aux 的骨干网络
        classifier: 主分类器
        aux_classifier: 辅助分类器（checkpoint 内不含该分支时传 None）
    """

    def __init__(self, backbone, classifier, aux_classifier=None):
        super().__init__(backbone, classifier)
        self.aux_classifier = aux_classifier

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        out = self.classifier(features["out"])
        result["out"] = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)

        if self.aux_classifier is not None:
            aux = self.aux_classifier(features["aux"])
            result["aux"] = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
        return result


def build_fcn(num_classes: int = NUM_CLASSES, aux_loss: bool = True) -> FCNWithAux:
    """构建与 fcn_resnet50_coco.pth 结构一致的 FCN。

    与项目 model.fcn_model.fcn_resnet50 的唯一区别是：当 aux_loss=True 时，
    额外把 layer3 特征暴露出来并挂一个 FCNHead(1024, num_classes) 的辅助分类器。
    """
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    return_layers = {"layer4": "out"}
    aux_classifier = None
    if aux_loss:
        return_layers["layer3"] = "aux"
        aux_classifier = FCNHead(AUX_INPLANES, num_classes)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = FCNHead(OUT_INPLANES, num_classes)
    return FCNWithAux(backbone, classifier, aux_classifier)


class TorchvisionFreeEvalTransform:
    """缺少 torchvision 时，与 utils.segmentation_transforms.SegmentationPresetEval(520)
    等价的纯 PIL/torch 预处理（缩放到短边 520 -> ToTensor -> Normalize）。

    仅在当前 Python 环境没有 torchvision（无法导入项目 transforms）时作为兜底使用，
    脚本会明确打印提示，不会把它伪装成项目 transforms。
    """

    def __init__(self, base_size: int = BASE_SIZE, mean=MEAN, std=STD):
        self.base_size = base_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image, target: Image.Image):
        width, height = image.size
        # 与 torchvision F.resize(int) 一致：短边缩放到 base_size，长边按比例截断取整
        if width <= height:
            new_size = (self.base_size, int(self.base_size * height / width))
        else:
            new_size = (int(self.base_size * width / height), self.base_size)
        image = image.resize(new_size, Image.BILINEAR)
        target = target.resize(new_size, Image.NEAREST)

        array = np.asarray(image, dtype=np.uint8)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        return tensor, torch.as_tensor(np.array(target, dtype=np.int64))


def resolve_path(path: str | Path) -> Path:
    """相对路径先按当前工作目录、再按项目根目录解析。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), PROJECT_ROOT):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return PROJECT_ROOT / candidate


def load_checkpoint(path: Path) -> "OrderedDict[str, torch.Tensor]":
    """torch.load 权重；兼容 state_dict 包裹和旧版 torch。"""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 没有 weights_only 参数
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"无法识别的 checkpoint 类型: {type(checkpoint)!r}")
    return checkpoint


def get_transform(train: bool, allow_fallback: bool):
    """优先使用项目自带 transforms；缺少 torchvision 时按需兜底。"""
    try:
        from utils.segmentation_transforms import get_transform as project_get_transform

        return project_get_transform(train=train), "project"
    except ModuleNotFoundError as exc:
        if not str(exc.name or "").startswith("torchvision"):
            raise
        if not allow_fallback:
            raise SystemExit(
                "当前环境没有安装 torchvision，无法导入 utils.segmentation_transforms，"
                "项目 transforms 不可用。"
            ) from exc
        print(
            "[WARN] 当前 Python 环境缺少 torchvision，无法导入 utils/segmentation_transforms.py。\n"
            "       已改用等价的纯 PIL/torch 预处理（短边缩放到 520 -> ToTensor -> Normalize）；\n"
            "       这不是项目 transforms 本身，仅用于让 base 环境也能完成前向 shape 验证。"
        )
        return TorchvisionFreeEvalTransform(BASE_SIZE), "fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加载 fcn_resnet50_coco.pth 并前向一次")
    parser.add_argument("--weights", default="weights/fcn_resnet50_coco.pth", help="checkpoint 路径")
    parser.add_argument("--voc-root", default="VOCdevkit", help="VOCdevkit 根目录")
    parser.add_argument("--split", default="test.txt", help="ImageSets/Segmentation 下的划分文件")
    parser.add_argument("--index", type=int, default=0, help="取第几个样本")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="类别数（VOC=21）")
    parser.add_argument(
        "--aux-loss",
        choices=("auto", "on", "off"),
        default="auto",
        help="是否构建辅助分类器；auto=按 checkpoint 中是否存在 aux_classifier.* 判断",
    )
    parser.add_argument("--train", action="store_true", help="使用训练版 transforms（默认 eval 版）")
    parser.add_argument(
        "--require-project-transforms",
        action="store_true",
        help="缺少 torchvision 时直接报错，而不是使用等价兜底预处理",
    )
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

    print(f"权重文件: {weights_path}")
    print(f"设备: {device}")

    # 1) 读取 checkpoint，据 aux_classifier.* 判断是否需要 aux_loss
    state_dict = load_checkpoint(weights_path)
    aux_keys = [key for key in state_dict if key.startswith("aux_classifier.")]
    expects_aux = bool(aux_keys)
    if args.aux_loss == "auto":
        aux_loss = expects_aux
        print(f"checkpoint key 总数: {len(state_dict)}，其中 aux_classifier.*: {len(aux_keys)}")
        print(f"=> 需要 aux_loss: {aux_loss}（辅助分类器分支{'存在' if expects_aux else '不存在'}）")
    else:
        aux_loss = args.aux_loss == "on"
        print(f"手动指定 aux_loss={aux_loss}（checkpoint 中 aux_classifier.*: {len(aux_keys)}）")
        if aux_loss != expects_aux:
            print("[WARN] 手动选择与 checkpoint 结构不一致，strict=True 大概率会失败。")

    # 2) 构建 FCN 并严格加载
    model = build_fcn(num_classes=args.num_classes, aux_loss=aux_loss)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f} M")
    print(
        "主分类器 classifier.4.weight: "
        f"{tuple(model.classifier[4].weight.shape)}"
        + (
            f"；辅助分类器 aux_classifier.4.weight: {tuple(model.aux_classifier[4].weight.shape)}"
            if model.aux_classifier is not None
            else "；无辅助分类器"
        )
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"缺失的 key（missing）: {len(missing)}")
    for key in missing:
        print(f"  - {key}")
    print(f"多余的 key（unexpected）: {len(unexpected)}")
    for key in unexpected:
        print(f"  - {key}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise SystemExit(f"strict=True 加载失败:\n{exc}") from exc
    print("strict=True 加载成功（missing=0, unexpected=0）")

    model.to(device).eval()

    # 3) 用项目 transforms 预处理 VOC 图片并前向一次
    transform, transform_source = get_transform(train=args.train, allow_fallback=not args.require_project_transforms)
    print(f"预处理: {transform_source}" + ("（utils.segmentation_transforms）" if transform_source == "project" else ""))

    dataset = VOCData(str(resolve_path(args.voc_root)), transforms=transform, txt_name=args.split)
    if not 0 <= args.index < len(dataset):
        raise SystemExit(f"index 越界: {args.index}，数据集大小 {len(dataset)}")
    image, _target = dataset[args.index]
    sample_path = dataset.images[args.index]
    print(f"样本: {sample_path}")
    print(f"  原始尺寸: {Image.open(sample_path).size}（宽, 高）")
    print(f"  预处理后 tensor: {tuple(image.shape)}")

    batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(batch)

    print("前向输出:")
    for name, tensor in output.items():
        print(f"  {name}: {tuple(tensor.shape)}")
    print(f"预测类别数（out 通道）: {output['out'].shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
