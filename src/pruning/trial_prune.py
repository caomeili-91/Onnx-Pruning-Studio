r"""对已加载的 FCN 做一次保守的结构化试剪（torch_pruning / DepGraph）。

策略：
- 只让 backbone.layer1.0.conv1 这一个 Conv2d 成为剪枝候选，剪枝率 0.1；
- ignored_layers 中放入 FCN 的最终分割输出层 classifier.4（以及 aux_classifier.4），
  同时把其余 Conv2d 一并忽略，从而保证候选集合里只有目标 Conv2d；
- pruner.step(interactive=True) 只取第一个 group 并 prune；
- 打印参数量变化并前向一次，不做完整评估。

注意：本脚本依赖 torch_pruning。当前 base 环境已安装（1.6.0），FCN 环境未安装。
运行：
    D:\anaconda\python.exe src/pruning/trial_prune.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_pruning as tp
except ImportError as exc:  # 给 FCN 等未安装 torch_pruning 的环境一个明确提示
    raise SystemExit(
        "当前 Python 环境没有 torch_pruning。请先安装 torch-pruning，"
        "或改用已安装该包的 base 环境运行。"
    ) from exc

from src.models.fcn_loader import build_fcn, load_checkpoint, resolve_path

DEFAULT_TARGET = "backbone.layer1.0.conv1"


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_module(model: nn.Module, dotted_path: str) -> nn.Module:
    """按 'backbone.layer1.0.conv1' 这样的路径取子模块（支持数字下标）。"""
    module: nn.Module = model
    for part in dotted_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def final_score_layers(model: nn.Module) -> list[nn.Conv2d]:
    """FCN 的最终分割输出层（classifier 与 aux_classifier 的最后一个 Conv2d）。"""
    layers: list[nn.Conv2d] = []
    for head_name in ("classifier", "aux_classifier"):
        head = getattr(model, head_name, None)
        if head is not None and len(head) > 0 and isinstance(head[-1], nn.Conv2d):
            layers.append(head[-1])
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FCN 单层保守试剪（torch_pruning）")
    parser.add_argument("--weights", default="weights/fcn_resnet50_coco.pth", help="checkpoint 路径")
    parser.add_argument("--num-classes", type=int, default=21, help="类别数（VOC=21）")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="只剪这一个 backbone Conv2d")
    parser.add_argument("--ratio", type=float, default=0.1, help="剪枝率")
    parser.add_argument("--height", type=int, default=520, help="前向输入高度")
    parser.add_argument("--width", type=int, default=693, help="前向输入宽度")
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

    state_dict = load_checkpoint(weights_path)
    aux_keys = [key for key in state_dict if key.startswith("aux_classifier.")]
    aux_loss = bool(aux_keys) if args.aux_loss == "auto" else args.aux_loss == "on"

    model = build_fcn(num_classes=args.num_classes, aux_loss=aux_loss)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(f"加载异常: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    target = resolve_module(model, args.target)
    if not isinstance(target, nn.Conv2d):
        raise SystemExit(f"目标不是 Conv2d: {args.target} -> {type(target)}")
    backbone_conv_ids = {id(module) for module in model.backbone.modules() if isinstance(module, nn.Conv2d)}
    if id(target) not in backbone_conv_ids:
        raise SystemExit(f"目标不是 backbone 中的 Conv2d: {args.target}")

    final_layers = final_score_layers(model)
    # 其余 Conv2d 全部忽略，使唯一的剪枝候选就是 target
    ignored_layers = list(final_layers)
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module is not target and module not in ignored_layers:
            ignored_layers.append(module)

    print(f"torch {torch.__version__} | torch_pruning {tp.__version__} | device {device}")
    print(f"权重: {weights_path}")
    print(f"目标层: {args.target} -> {target}")
    print(f"候选剪枝层数量: 1（其余 {len(ignored_layers)} 个 Conv2d 均在 ignored_layers 中）")
    print("最终分割输出层已保护: " + ", ".join(
        f"{name}[-1] out_channels={layer.out_channels}"
        for name, layer in zip(("classifier", "aux_classifier"), final_layers)
    ))

    parameters_before = count_parameters(model)
    target_out_before = target.out_channels

    example_inputs = torch.randn(1, 3, args.height, args.width, device=device)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=args.ratio,
        global_pruning=False,
        ignored_layers=ignored_layers,
    )

    groups = pruner.step(interactive=True)
    try:
        first_group = next(groups)
    except StopIteration:
        raise SystemExit("pruner 没有产生任何可剪 group，请检查 target / ignored_layers。") from None
    print(f"pruner.step(interactive=True) 产生 group，仅处理第一个：")
    print(first_group.details())
    first_group.prune()
    groups.close()

    parameters_after = count_parameters(model)
    removed = parameters_before - parameters_after
    print(f"参数量: {parameters_before:,} -> {parameters_after:,} "
          f"（减少 {removed:,}，{100 * removed / parameters_before:.4f}%）")
    print(f"目标层 out_channels: {target_out_before} -> {target.out_channels}")
    print("最终分割输出层通道（应保持 21）: " + ", ".join(
        f"{name}[-1]={layer.out_channels}"
        for name, layer in zip(("classifier", "aux_classifier"), final_layers)
    ))

    # 前向一次，确认结构化剪枝后的依赖仍然自洽
    forward_input = torch.randn(1, 3, args.height, args.width, device=device)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(forward_input)
    elapsed = time.perf_counter() - start

    print(f"前向成功（{elapsed:.1f}s），输出:")
    for name, tensor in output.items():
        print(f"  {name}: {tuple(tensor.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
