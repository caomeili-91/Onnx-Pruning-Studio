r"""交互式逐组剪枝 + 即时评估（torch_pruning MagnitudePruner）。

流程：
1. 加载未剪枝 FCN（strict=True），使用 cuda；
2. ignored_layers 保护最终分割头 classifier[-1] / aux_classifier[-1]；
3. 从 VOC val 固定随机抽取 N 张（默认 200），构建一个复用的 DataLoader；
4. pruner.step(interactive=True) 惰性遍历 group：
   prune 一个 group -> 评估一次 -> empty_cache -> 写 CSV；
5. 不做微调，只报告剪枝后的即时精度。

CSV: outputs/step_prune_log.csv
列: group_index, module_name, params_count, current_miou, miou_drop
其中 current_miou 是百分比（0-100），miou_drop = baseline - current（百分点）。
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_pruning as tp
except ImportError as exc:
    raise SystemExit("当前环境没有 torch_pruning，请先在 FCN 环境安装。") from exc

from src.eval.evaluator import evaluate_detailed
from src.models.fcn_loader import build_fcn, load_checkpoint, resolve_path
from utils.my_dataset import VOCData
from utils.segmentation_transforms import get_transform


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_ignored_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    ignored: list[torch.nn.Module] = []
    for head_name in ("classifier", "aux_classifier"):
        head = getattr(model, head_name, None)
        if head is not None and len(head) > 0:
            ignored.append(head[-1])
    return ignored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交互式逐组剪枝 + 即时评估")
    parser.add_argument("--weights", default="weights/fcn_resnet50_coco.pth")
    parser.add_argument("--voc-root", default="VOCdevkit")
    parser.add_argument("--split", default="val.txt")
    parser.add_argument("--num-samples", type=int, default=200, help="固定随机抽取的验证集数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.1, help="每个 group 的剪枝率")
    parser.add_argument(
        "--global-pruning",
        action="store_true",
        help="开启全局排序剪枝（默认关闭；逐组交互模式建议关闭）",
    )
    parser.add_argument("--max-groups", type=int, default=0, help="只处理前 N 个 group（0=全部）")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--output", default="outputs/step_prune_log.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 cuda，但当前环境不可用")
    device = torch.device(args.device)

    weights_path = resolve_path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"权重文件不存在: {weights_path}")

    state_dict = load_checkpoint(weights_path)
    aux_keys = [key for key in state_dict if key.startswith("aux_classifier.")]
    aux_loss = bool(aux_keys)

    model = build_fcn(num_classes=21, aux_loss=aux_loss)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(f"加载异常: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    ignored_layers = build_ignored_layers(model)
    print(f"设备: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"权重: {weights_path}")
    print(f"aux_loss: {aux_loss}")
    head_names = ["classifier", "aux_classifier"][: len(ignored_layers)]
    print("ignored_layers: " + ", ".join(
        f"{name}[-1](out_channels={layer.out_channels})"
        for name, layer in zip(head_names, ignored_layers)
    ))

    params_baseline = count_parameters(model)
    print(f"基线参数量: {params_baseline:,}")

    full_dataset = VOCData(
        str(resolve_path(args.voc_root)),
        transforms=get_transform(train=False),
        txt_name=args.split,
    )
    num_samples = min(args.num_samples, len(full_dataset))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(full_dataset)), num_samples)
    subset = Subset(full_dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=full_dataset.collate_fn,
    )
    print(f"评估子集: {args.split} 中固定随机 {num_samples} 张（seed={args.seed}），batch_size={args.batch_size}")

    baseline_result = evaluate_detailed(model, dataloader, device, progress=False)
    baseline_miou = baseline_result["miou"]
    print(f"剪枝前基线 mIoU: {baseline_miou * 100:.2f}%（{num_samples} 张，"
          f"{baseline_result['elapsed_sec']:.1f}s）")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["group_index", "module_name", "params_count", "current_miou", "miou_drop"])
    csv_file.flush()

    example_inputs = torch.randn(1, 3, 520, 693, device=device)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=args.ratio,
        global_pruning=args.global_pruning,
        ignored_layers=ignored_layers,
    )

    module_names = {id(module): name for name, module in model.named_modules()}
    groups = pruner.step(interactive=True)
    processed = 0
    last_miou = baseline_miou
    start_time = time.perf_counter()

    try:
        for group_index, group in enumerate(groups):
            if args.max_groups and processed >= args.max_groups:
                break

            affected = []
            for dependency, _idxs in group:
                module = dependency.target.module
                name = module_names.get(id(module))
                if name is not None and name not in affected:
                    affected.append(name)
            root_module = group[0][0].target.module
            root_name = module_names.get(id(root_module), type(root_module).__name__)

            params_before = count_parameters(model)
            print(f"\n[group {group_index}] root={root_name}（deps={len(group)}）")
            print("  被剪层: " + (", ".join(affected) if affected else "(未命名层)"))

            group.prune()
            processed += 1
            params_after = count_parameters(model)

            if device.type == "cuda":
                torch.cuda.synchronize()
            eval_start = time.perf_counter()
            result = evaluate_detailed(model, dataloader, device, progress=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            eval_time = time.perf_counter() - eval_start

            current_miou = result["miou"]
            drop = baseline_miou - current_miou
            last_miou = current_miou

            print(f"  参数量: {params_before:,} -> {params_after:,}（-{params_before - params_after:,}）")
            print(f"  mIoU: {current_miou * 100:.2f}%  drop_vs_baseline: {drop * 100:+.2f} pp  "
                  f"（评估 {eval_time:.1f}s）")

            writer.writerow([
                group_index,
                root_name,
                params_after,
                f"{current_miou * 100:.2f}",
                f"{drop * 100:.2f}",
            ])
            csv_file.flush()

            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        try:
            groups.close()
        except Exception:
            pass
        csv_file.close()

    elapsed = time.perf_counter() - start_time
    final_params = count_parameters(model)
    print("\n" + "=" * 60)
    print(f"处理 group 数: {processed}")
    print(f"基线 mIoU: {baseline_miou * 100:.2f}%")
    print(f"最终 mIoU: {last_miou * 100:.2f}%（相对基线 {last_miou * 100 - baseline_miou * 100:+.2f} pp）")
    print(f"最终参数量: {final_params:,}（基线 {params_baseline:,}，减少 {params_baseline - final_params:,}）")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
