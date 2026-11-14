"""VOC segmentation evaluation on ONNX Runtime (float32, CPU by default)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    import psutil
except ImportError:  # optional; the app still works without it
    psutil = None

ort.set_default_logger_severity(3)


def rss_mb() -> float | None:
    if psutil is None:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None

INPUT_SIZE = (520, 520)
NUM_CLASSES = 21
IGNORE_INDEX = 255
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class VOCSegDataset:
    def __init__(self, voc_root: str | Path, split: str = "val.txt") -> None:
        root = Path(voc_root) / "VOC2012"
        self.images_dir = root / "JPEGImages"
        self.masks_dir = root / "SegmentationClass"
        names_path = root / "ImageSets" / "Segmentation" / split
        if not names_path.exists():
            raise FileNotFoundError(f"找不到验证集划分: {names_path}")
        self.names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = Image.open(self.images_dir / f"{name}.jpg").convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
        mask = Image.open(self.masks_dir / f"{name}.png").resize(INPUT_SIZE, Image.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = ((array.transpose(2, 0, 1) - MEAN) / STD).astype(np.float32)
        target = np.asarray(mask, dtype=np.int64)
        return tensor[None], target


def subset_indices(dataset_size: int, subset_size: int = 50, seed: int = 42) -> List[int]:
    size = min(int(subset_size), int(dataset_size))
    return random.Random(int(seed)).sample(range(dataset_size), size)


def _metrics(confusion: np.ndarray) -> Dict[str, Any]:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    union = tp + fp + fn
    iou = np.full(confusion.shape[0], np.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = tp[valid] / union[valid]
    total = confusion.sum()
    return {
        "miou": float(np.nanmean(iou)) if valid.any() else float("nan"),
        "per_class_iou": iou.tolist(),
        "pixel_accuracy": float(tp.sum() / total) if total else 0.0,
        "valid_pixels": int(total),
        "valid_classes": int(valid.sum()),
    }


class OnnxSegmentationEvaluator:
    def __init__(self, model_bytes: bytes, providers: Optional[Sequence[str]] = None) -> None:
        available = ort.get_available_providers()
        if providers is None:
            providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available[:1]
        providers = [provider for provider in providers if provider in available] or available[:1]
        self.model_bytes = model_bytes
        self.session = ort.InferenceSession(model_bytes, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = list(self.session.get_inputs()[0].shape)
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.providers = self.session.get_providers()

    def evaluate(
        self,
        dataset: VOCSegDataset,
        indices: Iterable[int],
        output_name: str,
        batch_size: int = 1,
        progress_callback=None,
    ) -> Dict[str, Any]:
        if output_name not in self.output_names:
            raise ValueError(f"ONNX 输出不存在: {output_name}")
        indices = list(indices)
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        rss_before = rss_mb()
        start = time.perf_counter()
        processed = 0

        batch_tensors: List[np.ndarray] = []
        batch_targets: List[np.ndarray] = []

        def flush() -> None:
            nonlocal processed
            if not batch_tensors:
                return
            inputs = np.concatenate(batch_tensors, axis=0)
            logits = self.session.run([output_name], {self.input_name: inputs})[0]
            predictions = np.argmax(logits, axis=1)
            for pred, target in zip(predictions, batch_targets):
                valid = (target != IGNORE_INDEX) & (target >= 0) & (target < NUM_CLASSES)
                np.add.at(confusion, (target[valid], pred[valid]), 1)
            processed += len(batch_tensors)
            batch_tensors.clear()
            batch_targets.clear()
            if progress_callback:
                progress_callback(processed, len(indices))

        for index in indices:
            image, target = dataset[index]
            batch_tensors.append(image)
            batch_targets.append(target)
            if len(batch_tensors) >= batch_size:
                flush()
        flush()
        elapsed = time.perf_counter() - start
        rss_after = rss_mb()
        metrics = _metrics(confusion)
        metrics.update({
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "rss_delta_mb": (rss_after - rss_before) if rss_before is not None and rss_after is not None else None,
            "num_images": processed,
            "elapsed_sec": elapsed,
            "ms_per_image": (elapsed / processed * 1000.0) if processed else 0.0,
            "providers": self.providers,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "output_name": output_name,
            "confusion_matrix": confusion.tolist(),
        })
        return metrics

    def close(self) -> None:
        self.session = None
        self.model_bytes = b""


def save_baseline(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_baseline(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
