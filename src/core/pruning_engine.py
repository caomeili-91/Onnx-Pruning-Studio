r"""FCN 剪枝引擎：把逐组剪枝 + 即时评估封装成可复用的类。

对外主要接口：
    engine = FCNPruningEngine()
    engine.load_model("weights/fcn_resnet50_coco.pth")
    engine.evaluate("subset")                  # 固定 200 张子集
    for event in engine.run_interactive_pruning(0.1, strategy="local"):
        ...                                    # 实时事件（baseline / group / done）
    engine.export_onnx("outputs/fcn.onnx")

说明：
- 基线/逐组评估默认使用固定随机 200 张验证集，以控制 UI 响应时间；
  evaluate("baseline") 仍可跑完整 val.txt（582 张）。
- 评估严格只使用主输出 out，忽略 aux。
- 剪枝策略 local 对应 global_pruning=False（每个 group 各自 10%），
  global 对应 global_pruning=True（全局排序）。
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import inspect
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_pruning as tp
except ImportError:  # 允许只加载/评估模型而不剪枝
    tp = None

from src.core.model_registry import build_builtin_model, list_builtin_models
from src.eval.evaluator import evaluate_detailed
from src.models.fcn_loader import load_checkpoint
from utils.my_dataset import VOCData
from utils.segmentation_transforms import get_transform

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def list_model_classes(py_path: str | Path) -> List[str]:
    """静态解析 .py（不执行），列出其中定义的类名。"""
    path = Path(py_path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Python 语法错误: {exc}") from exc
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _output_dim(module: nn.Module) -> int | None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return int(module.out_channels)
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    return None


def _import_custom_module(file_path: Path):
    name = f"_custom_seg_model_{hashlib.md5(str(file_path).encode('utf-8')).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模型文件: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent = str(file_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec.loader.exec_module(module)
    return module


def build_custom_model(
    py_path: str | Path,
    class_name: str,
    num_classes: int,
    aux_loss: bool = True,
) -> nn.Module:
    """从用户上传的 .py 中实例化指定模型类。

    约定：
    - 类需要是 nn.Module 子类；
    - 如果 __init__ 有 num_classes / n_classes / num_class / classes 参数，自动传入；
    - 如果 __init__ 有 aux_loss 参数，自动传入；
    - 其他参数需有默认值，或使用模块级工厂函数。
    """
    path = Path(py_path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    module = _import_custom_module(path)
    cls = getattr(module, class_name, None)
    if cls is None:
        available = [name for name in dir(module) if isinstance(getattr(module, name), type)]
        raise AttributeError(f"模型文件中没有类 {class_name}；可用类: {available}")
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise TypeError(f"{class_name} 不是 torch.nn.Module 子类")

    signature = inspect.signature(cls)
    kwargs = {}
    for key in ("num_classes", "n_classes", "num_class", "classes"):
        if key in signature.parameters:
            kwargs[key] = num_classes
            break
    if "aux_loss" in signature.parameters:
        kwargs["aux_loss"] = aux_loss

    try:
        model = cls(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"实例化 {class_name} 失败（需要额外参数？请看 __init__ 签名 {signature}）: {exc}"
        ) from exc
    if not isinstance(model, nn.Module):
        raise TypeError(f"{class_name} 实例化结果不是 nn.Module")
    return model


def get_app_dir() -> Path:
    """返回外部资源（VOCdevkit、outputs、weights）应所在的目录。

    - PyInstaller 打包后：可执行文件（.exe）所在目录；
    - 源码运行：项目根目录。
    注意不要用 sys._MEIPASS 存外部数据，那是 onefile 的临时解压目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return PROJECT_ROOT


class _MainLogits(torch.nn.Module):
    """ONNX 导出用的包装：只返回主输出 out。"""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, dict):
            return output["out"]
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


class FCNPruningEngine:
    def __init__(
        self,
        voc_root: str = "VOCdevkit",
        split: str = "val.txt",
        num_samples: int = 200,
        seed: int = 42,
        device: str = "auto",
        output_csv: str = "outputs/step_prune_log.csv",
        num_classes: int = 21,
        ignore_index: int = 255,
        base_dir: str | Path | None = None,
    ) -> None:
        self.voc_root = voc_root
        self.split = split
        self.num_samples = num_samples
        self.seed = seed
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else get_app_dir()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.output_csv = self._absolute(output_csv)

        self.model: torch.nn.Module | None = None
        self.aux_loss = False
        self.baseline_miou: float | None = None
        self.last_miou: float | None = None
        self.last_params: int | None = None
        self.pruned = False

        self._full_dataset = None
        self._full_loader = None
        self._subset_loader = None
        self._subset_indices: List[int] = []
        self._module_names: Dict[int, str] = {}

        self.model_spec: Dict[str, Any] | None = None
        self.model_name = ""
        self.ignored_layers: List[nn.Module] = []
        self.protected_layer_names: List[str] = []
        self.ignore_mode = "auto"
        self.ignore_fallback = False

    # ------------------------------------------------------------------
    # 基础状态
    # ------------------------------------------------------------------
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters())

    def _absolute(self, path: str | Path) -> Path:
        """把相对路径统一解析到 base_dir（源码根目录或 .exe 所在目录）。"""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.base_dir / candidate).resolve()

    def _collect_ignored_layers(self, num_classes: int, mode: str = "auto", last_n: int = 1) -> List[nn.Module]:
        """自动选择需要保护的输出层。

        mode="auto"   : 保护所有 out_channels/out_features == num_classes 的层；
        mode="last_n" : 保护最后 N 个 Conv2d/ConvTranspose2d/Linear。
        """
        if self.model is None:
            return []
        candidates = [
            (name, module)
            for name, module in self.model.named_modules()
            if _output_dim(module) is not None
        ]
        if not candidates:
            self.protected_layer_names = []
            return []

        self.ignore_fallback = False
        if mode == "last_n":
            count = max(1, int(last_n))
            selected = candidates[-count:]
        else:
            selected = [
                (name, module)
                for name, module in candidates
                if _output_dim(module) == int(num_classes)
            ]
            if not selected:
                self.ignore_fallback = True
                selected = candidates[-max(1, int(last_n)):]

        self.protected_layer_names = [name for name, _ in selected]
        return [module for _, module in selected]

    @staticmethod
    def list_model_classes(py_path: str | Path) -> List[str]:
        return list_model_classes(py_path)

    def _ensure_dataset(self) -> None:
        if self._full_dataset is not None:
            return
        root = self._absolute(self.voc_root)
        self._full_dataset = VOCData(
            str(root),
            transforms=get_transform(train=False),
            txt_name=self.split,
        )
        num_samples = min(self.num_samples, len(self._full_dataset))
        rng = random.Random(self.seed)
        self._subset_indices = rng.sample(range(len(self._full_dataset)), num_samples)

        common = dict(batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
                      collate_fn=self._full_dataset.collate_fn)
        self._subset_loader = DataLoader(Subset(self._full_dataset, self._subset_indices), **common)
        self._full_loader = DataLoader(self._full_dataset, **common)

    # ------------------------------------------------------------------
    # 模型加载 / 评估
    # ------------------------------------------------------------------
    def _build_model(self, model_spec: Dict[str, Any], aux_loss: bool) -> nn.Module:
        kind = model_spec.get("kind", "builtin")
        if kind == "builtin":
            name = model_spec.get("name")
            if name not in list_builtin_models():
                raise KeyError(f"未知内置模型: {name}；可选: {list_builtin_models()}")
            return build_builtin_model(
                name=name,
                num_classes=self.num_classes,
                aux_loss=aux_loss,
            )
        if kind == "file":
            py_path = self._absolute(model_spec["path"])
            class_name = model_spec.get("class_name")
            if not class_name:
                raise ValueError("上传模型时必须选择模型类名")
            return build_custom_model(
                py_path=py_path,
                class_name=class_name,
                num_classes=self.num_classes,
                aux_loss=aux_loss,
            )
        raise ValueError(f"未知模型来源: {kind}")

    @staticmethod
    def _model_label(model_spec: Dict[str, Any]) -> str:
        if model_spec.get("kind") == "builtin":
            return str(model_spec.get("name", "builtin"))
        return f"{Path(model_spec.get('path', '?')).name}:{model_spec.get('class_name', '?')}"

    def load_model(
        self,
        pth_path: str | Path,
        model_spec: Dict[str, Any] | None = None,
        num_classes: int | None = None,
        ignored_mode: str = "auto",
        last_n: int = 1,
    ) -> Dict[str, Any]:
        """加载 .pth；支持内置模型或用户上传 .py 中的模型类。

        Args:
            pth_path: 权重文件
            model_spec: {"kind":"builtin","name":...} 或
                        {"kind":"file","path":...,"class_name":...}
            num_classes: 类别数（默认沿用引擎构造时的 21）
            ignored_mode: "auto"（out_channels == num_classes）或 "last_n"
            last_n: ignored_mode="last_n" 时保护最后 N 个 Conv/Linear
        """
        try:
            path = self._absolute(pth_path)
            if not path.exists():
                return {"ok": False, "message": f"文件不存在: {path}"}
            if num_classes is not None:
                self.num_classes = int(num_classes)
            if model_spec is None:
                model_spec = {"kind": "builtin", "name": "FCN-ResNet50 (VOC 21类)"}

            state_dict = load_checkpoint(path)
            aux_present = any(key.startswith("aux_classifier.") for key in state_dict)
            model = self._build_model(model_spec, aux_loss=aux_present)
            if not isinstance(model, nn.Module):
                raise TypeError("模型工厂返回的不是 nn.Module")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            non_aux_missing = [key for key in missing if not key.startswith("aux_classifier.")]
            non_aux_unexpected = [key for key in unexpected if not key.startswith("aux_classifier.")]
            if non_aux_missing or non_aux_unexpected:
                return {
                    "ok": False,
                    "message": (
                        f"权重与模型不匹配（非 aux 部分）：missing={non_aux_missing[:8]} "
                        f"(共 {len(non_aux_missing)})，unexpected={non_aux_unexpected[:8]} "
                        f"(共 {len(non_aux_unexpected)})。请确认模型架构/类别数与 checkpoint 一致。"
                    ),
                }
            aux_mismatch = bool(missing or unexpected)
            if not aux_mismatch:
                model.load_state_dict(state_dict, strict=True)
            model.to(self.device).eval()

            self.model = model
            self.model_spec = model_spec
            self.model_name = self._model_label(model_spec)
            self.aux_loss = getattr(model, "aux_classifier", None) is not None
            self.ignore_mode = ignored_mode
            self._module_names = {id(module): name for name, module in model.named_modules()}
            self.ignored_layers = self._collect_ignored_layers(
                num_classes=self.num_classes,
                mode=ignored_mode,
                last_n=last_n,
            )
            if not self.ignored_layers:
                self.model = None
                return {
                    "ok": False,
                    "message": "没有找到可保护的 Conv/Linear 输出层，请改用 last_n 模式或检查模型。",
                }

            self.baseline_miou = None
            self.last_miou = None
            self.last_params = self.count_parameters(model)
            self.pruned = False

            protected = ", ".join(self.protected_layer_names) or "(none)"
            aux_note = ""
            if aux_mismatch:
                aux_note = (
                    f" | aux 权重不完全匹配（missing={len(missing)}, unexpected={len(unexpected)}），"
                    "已忽略 aux 差异"
                )
            message = (
                f"加载成功: {self.model_name} | {path.name} | {self.last_params:,} params | "
                f"aux={self.aux_loss} | device={self.device} | 保护层: {protected}{aux_note}"
            )
            return {
                "ok": True,
                "message": message,
                "params": self.last_params,
                "aux_loss": self.aux_loss,
                "aux_mismatch": aux_mismatch,
                "device": str(self.device),
                "model_name": self.model_name,
                "protected": list(self.protected_layer_names),
                "ignore_fallback": self.ignore_fallback,
                "num_classes": self.num_classes,
            }
        except Exception as exc:
            self.model = None
            return {"ok": False, "message": f"加载失败: {type(exc).__name__}: {exc}"}

    def evaluate(self, mode: str = "baseline") -> float:
        """mode='baseline' 跑完整 val.txt；mode='subset' 跑固定 200 张子集。"""
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")
        if mode not in ("baseline", "subset"):
            raise ValueError("mode 只能是 'baseline' 或 'subset'")

        self._ensure_dataset()
        loader = self._full_loader if mode == "baseline" else self._subset_loader
        result = evaluate_detailed(
            self.model,
            loader,
            self.device,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            progress=False,
        )
        return result["miou"]

    # ------------------------------------------------------------------
    # 逐组剪枝（生成器，实时 yield 状态）
    # ------------------------------------------------------------------
    def run_interactive_pruning(
        self,
        ratio: float,
        strategy: str = "local",
        max_groups: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")
        if tp is None:
            raise RuntimeError("当前环境没有 torch_pruning，无法剪枝")
        if not 0.0 < ratio < 1.0:
            raise ValueError("ratio 必须在 (0, 1) 内")
        if strategy not in ("local", "global"):
            raise ValueError("strategy 只能是 'local' 或 'global'")

        self._ensure_dataset()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(self.output_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["group_index", "module_name", "params_count", "current_miou", "miou_drop"])
        csv_file.flush()

        params_baseline = self.count_parameters(self.model)
        ignored_layers = list(self.ignored_layers)
        if not ignored_layers:
            raise RuntimeError("没有可保护的输出层，无法剪枝")

        yield {
            "type": "baseline_start",
            "message": (
                f"正在评估基线（{self.model_name}，固定 {len(self._subset_indices)} 张，"
                f"策略={strategy}，ratio={ratio}）..."
            ),
            "model_name": self.model_name,
            "protected": list(self.protected_layer_names),
        }

        baseline_start = time.perf_counter()
        self.baseline_miou = self.evaluate("subset")
        baseline_elapsed = time.perf_counter() - baseline_start
        yield {
            "type": "baseline",
            "miou": self.baseline_miou,
            "elapsed_sec": baseline_elapsed,
            "num_samples": len(self._subset_indices),
            "params": params_baseline,
        }

        pruner = tp.pruner.MagnitudePruner(
            self.model,
            torch.randn(1, 3, 520, 693, device=self.device),
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=ratio,
            global_pruning=(strategy == "global"),
            ignored_layers=ignored_layers,
        )
        groups = pruner.step(interactive=True)

        processed = 0
        last_miou = self.baseline_miou
        start_time = time.perf_counter()

        try:
            for group_index, group in enumerate(groups):
                if max_groups and processed >= max_groups:
                    break

                affected: List[str] = []
                for dependency, _idxs in group:
                    module = dependency.target.module
                    name = self._module_names.get(id(module))
                    if name is not None and name not in affected:
                        affected.append(name)

                root_module = group[0][0].target.module
                root_name = self._module_names.get(id(root_module), type(root_module).__name__)
                params_before = self.count_parameters(self.model)

                group.prune()
                processed += 1
                params_after = self.count_parameters(self.model)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                eval_start = time.perf_counter()
                current_miou = self.evaluate("subset")
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                eval_time = time.perf_counter() - eval_start
                last_miou = current_miou
                drop = self.baseline_miou - current_miou

                writer.writerow([
                    group_index,
                    root_name,
                    params_after,
                    f"{current_miou * 100:.2f}",
                    f"{drop * 100:.2f}",
                ])
                csv_file.flush()

                yield {
                    "type": "group",
                    "group_index": group_index,
                    "module_name": root_name,
                    "affected": affected,
                    "params_before": params_before,
                    "params": params_after,
                    "miou": current_miou,
                    "drop": drop,
                    "eval_sec": eval_time,
                }

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            try:
                groups.close()
            except Exception:
                pass
            csv_file.close()

        elapsed = time.perf_counter() - start_time
        self.pruned = processed > 0
        self.last_miou = last_miou
        self.last_params = self.count_parameters(self.model)

        yield {
            "type": "done",
            "processed": processed,
            "baseline_miou": self.baseline_miou,
            "final_miou": last_miou,
            "params_baseline": params_baseline,
            "final_params": self.last_params,
            "elapsed_sec": elapsed,
            "csv_path": str(self.output_csv),
            "strategy": strategy,
            "ratio": ratio,
            "model_name": self.model_name,
            "protected": list(self.protected_layer_names),
        }

    # ------------------------------------------------------------------
    # ONNX 导出
    # ------------------------------------------------------------------
    def export_onnx(self, save_path: str | Path = "outputs/fcn_pruned.onnx") -> Dict[str, Any]:
        if self.model is None:
            return {"ok": False, "message": "模型尚未加载"}
        try:
            import onnx  # noqa: F401
        except ImportError:
            return {
                "ok": False,
                "message": "缺少 onnx 包，请先执行: pip install onnx",
            }

        path = self._absolute(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        original_device = next(self.model.parameters()).device
        try:
            self.model.cpu().eval()
            wrapper = _MainLogits(self.model).eval()
            dummy = torch.randn(1, 3, 520, 693)
            torch.onnx.export(
                wrapper,
                dummy,
                str(path),
                input_names=["input"],
                output_names=["out"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "out": {0: "batch", 2: "height", 3: "width"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        except Exception as exc:
            return {"ok": False, "message": f"ONNX 导出失败: {type(exc).__name__}: {exc}"}
        finally:
            self.model.to(original_device)

        return {"ok": True, "path": str(path), "message": f"ONNX 已导出: {path}"}
