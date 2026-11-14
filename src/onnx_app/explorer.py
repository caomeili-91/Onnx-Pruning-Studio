"""Cumulative rollback exploration on an ONNX graph.

For every nominal per-group ratio, groups are visited in importance order.
A candidate is always rebuilt from the original folded ONNX plus all accepted
masks; nothing is written to disk. Evaluation is ONNX Runtime float32 mIoU.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Set

import numpy as np
import onnx
from onnx import numpy_helper

from src.onnx_app.evaluator import OnnxSegmentationEvaluator, VOCSegDataset, rss_mb, subset_indices
from src.onnx_app.grouper import Grouper
from src.onnx_app.preflight import _estimate_resources, _shape_map
from src.onnx_app.pruner import apply_masks

CSV_FIELDS = [
    "step", "group_id", "group_nodes", "nominal_ratio", "actual_channel_ratio",
    "cumulative_channel_ratio", "cumulative_param_ratio", "round_to",
    "miou", "delta_vs_baseline", "params", "flops_est", "peak_act_est",
    "status", "error_msg", "elapsed_ms",
    "profile_name", "profile_version", "model_hash", "input_signature", "provider",
    "sort_strategy", "sort_seed",
]


def count_parameters(model: onnx.ModelProto) -> int:
    total = 0
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        if array.dtype.kind in "fiub":
            total += int(array.size)
    return total


class Explorer:
    def __init__(
        self,
        folded_model_bytes: bytes,
        profile: Dict[str, Any],
        input_signature: Dict[str, Any],
        output_name: str,
        model_hash: str,
        voc_root: str | Path = "VOCdevkit",
        subset_size: int = 50,
        seed: int = 42,
        ratios: Sequence[float] = (0.1, 0.2),
        providers: Optional[Sequence[str]] = None,
        csv_path: str | Path = "outputs/curve.csv",
        baseline_path: str | Path = "outputs/baseline.json",
        max_groups: int = 0,
        sort_strategy: str = "l1_asc",
        sort_seed: Optional[int] = None,
    ) -> None:
        self.model_bytes = folded_model_bytes
        self.profile = profile
        self.profile_name = profile["profile_name"]
        self.profile_version = profile["profile_version"]
        self.input_signature = input_signature
        self.output_name = output_name
        self.model_hash = model_hash
        self.voc_root = voc_root
        self.subset_size = int(subset_size)
        self.seed = int(seed)
        self.ratios = [float(r) for r in ratios]
        self.providers = providers
        self.csv_path = Path(csv_path)
        self.baseline_path = Path(baseline_path)
        self.max_groups = int(max_groups)
        if sort_strategy not in {"l1_asc", "random", "depth_asc"}:
            raise ValueError(f"unknown sort_strategy: {sort_strategy}")
        self.sort_strategy = sort_strategy
        self.sort_seed = int(seed if sort_seed is None else sort_seed)
        self.dataset = VOCSegDataset(voc_root)
        self.rss_log_path = self.csv_path.with_name("rss.log")

    def _baseline(self) -> Dict[str, Any]:
        indices = subset_indices(len(self.dataset), self.subset_size, self.seed)
        evaluator = OnnxSegmentationEvaluator(self.model_bytes, self.providers)
        try:
            metrics = evaluator.evaluate(self.dataset, indices, self.output_name)
        finally:
            evaluator.close()
        model = onnx.load_model_from_string(self.model_bytes)
        shapes = _shape_map(model)
        resources = _estimate_resources(model, shapes)
        payload = {
            "model_hash": self.model_hash,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "input_signature": self.input_signature,
            "selected_output": self.output_name,
            "subset_size": len(indices),
            "seed": self.seed,
            "sort_strategy": self.sort_strategy,
            "sort_seed": self.sort_seed,
            "provider": metrics["providers"],
            "single_image_ms": metrics["ms_per_image"],
            "rss_before_mb": metrics.get("rss_before_mb"),
            "rss_after_mb": metrics.get("rss_after_mb"),
            "onnx_miou": metrics["miou"],
            "per_class_iou": metrics["per_class_iou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "params": int(resources["params"]),
            "flops_est": int(resources["flops"]),
            "peak_act_est": int(resources["peak_act_bytes"]),
            "baseline_source": "onnxruntime_float32",
        }
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _order_groups(self, groups, model: onnx.ModelProto):
        candidates = [group for group in groups if not group.protected and not group.unresolved]
        if self.sort_strategy == "random":
            ordered = list(candidates)
            random.Random(self.sort_seed).shuffle(ordered)
            return ordered
        if self.sort_strategy == "depth_asc":
            node_order = {node.name: index for index, node in enumerate(model.graph.node)}
            def depth(group):
                return min((node_order.get(node, 10 ** 9) for node in group.nodes), default=10 ** 9)
            return sorted(candidates, key=depth)
        return sorted(
            candidates,
            key=lambda group: float(group.importance.mean()) if group.importance is not None else 0.0,
        )

    def run(self) -> Generator[Dict[str, Any], None, None]:
        start = time.perf_counter()
        rss_start = rss_mb()
        rss_lines = [f"start,rss_mb={rss_start}"]
        baseline = self._baseline()
        rss_lines.append(f"baseline,rss_before_mb={baseline.get('rss_before_mb')},rss_after_mb={baseline.get('rss_after_mb')}")
        yield {"type": "baseline", "baseline": baseline}

        base_params = int(baseline["params"])
        model = onnx.load_model_from_string(self.model_bytes)
        grouper = Grouper(model, self.output_name, round_to=int(self.profile["channel_round_to"]))
        summary = grouper.summary()
        groups = [group for group in grouper.groups if group.active]
        ordered_groups = self._order_groups(groups, model)
        summary["sort_strategy"] = self.sort_strategy
        summary["sort_seed"] = self.sort_seed
        summary["ordered_group_indices"] = [
            groups.index(group) for group in ordered_groups
        ]
        yield {"type": "groups", "summary": summary}

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            step = 0

            def write_row(group_id, group_nodes, nominal_ratio, actual_channel_ratio,
                          cumulative_channel_ratio, cumulative_param_ratio, miou,
                          delta, params, flops, peak, status, error, elapsed_ms):
                nonlocal step
                writer.writerow({
                    "step": step,
                    "group_id": group_id,
                    "group_nodes": group_nodes,
                    "nominal_ratio": f"{nominal_ratio:.4f}",
                    "actual_channel_ratio": f"{actual_channel_ratio:.4f}",
                    "cumulative_channel_ratio": f"{cumulative_channel_ratio:.4f}",
                    "cumulative_param_ratio": f"{cumulative_param_ratio:.4f}",
                    "round_to": self.profile["channel_round_to"],
                    "miou": "" if miou is None else f"{miou:.4f}",
                    "delta_vs_baseline": "" if delta is None else f"{delta:.4f}",
                    "params": params,
                    "flops_est": flops,
                    "peak_act_est": peak,
                    "status": status,
                    "error_msg": error,
                    "elapsed_ms": f"{elapsed_ms:.1f}",
                    "profile_name": self.profile_name,
                    "profile_version": self.profile_version,
                    "model_hash": self.model_hash,
                    "input_signature": json.dumps(self.input_signature, ensure_ascii=False),
                    "provider": ",".join(baseline["provider"]),
                    "sort_strategy": self.sort_strategy,
                    "sort_seed": self.sort_seed,
                })
                csv_file.flush()
                step += 1

            write_row("BASELINE", "", 0.0, 0.0, 0.0, 0.0,
                      baseline["onnx_miou"] * 100, 0.0, base_params,
                      baseline["flops_est"], baseline["peak_act_est"],
                      "OK", "", 0.0)

            ordered = self._order_groups(groups, model)
            if self.max_groups:
                ordered = ordered[: self.max_groups]

            for ratio in self.ratios:
                masks: Dict[str, Set[int]] = {}
                total_pruned = 0
                for group_id, group in enumerate(ordered):
                    if group.protected or group.unresolved:
                        continue
                    k = group.channels_to_prune(ratio, int(self.profile["channel_round_to"]))
                    if k <= 0:
                        continue
                    if group.importance is None:
                        chosen = np.arange(k, dtype=np.int64)
                    else:
                        chosen = np.argsort(group.importance)[:k]
                    added: Dict[str, Set[int]] = {}
                    for tensor, indices in group.tensors.items():
                        for local in chosen:
                            added.setdefault(tensor, set()).add(int(indices[int(local)]))
                    for tensor, values in added.items():
                        masks.setdefault(tensor, set()).update(values)

                    t0 = time.perf_counter()
                    try:
                        candidate = apply_masks(self.model_bytes, masks)
                        evaluator = OnnxSegmentationEvaluator(candidate, self.providers)
                        try:
                            metrics = evaluator.evaluate(
                                self.dataset,
                                subset_indices(len(self.dataset), self.subset_size, self.seed),
                                self.output_name,
                            )
                        finally:
                            evaluator.close()
                        candidate_model = onnx.load_model_from_string(candidate)
                        shapes = _shape_map(candidate_model)
                        resources = _estimate_resources(candidate_model, shapes)
                        params = int(resources["params"])
                        miou = metrics["miou"] * 100.0
                        delta = miou - baseline["onnx_miou"] * 100.0
                        total_pruned += k
                        cumulative_channels = total_pruned / max(1, sum(g.size for g in ordered))
                        cumulative_params = (base_params - params) / max(1, base_params)
                        status = "CLAMPED" if ratio > group.max_prunable_ratio(int(self.profile["channel_round_to"])) else "OK"
                        elapsed = (time.perf_counter() - t0) * 1000
                        write_row(group_id, ";".join(group.nodes), ratio, k / group.size,
                                  cumulative_channels, cumulative_params, miou, delta,
                                  params, int(resources["flops"]), int(resources["peak_act_bytes"]),
                                  status, "", elapsed)
                        rss_lines.append(
                            f"step={step},ratio={ratio:.4f},group={group_id},"
                            f"rss_before_mb={metrics.get('rss_before_mb')},"
                            f"rss_after_mb={metrics.get('rss_after_mb')}"
                        )
                        yield {
                            "type": "step", "ratio": ratio, "group_id": group_id,
                            "group_nodes": list(group.nodes), "miou": miou, "delta": delta,
                            "params": params, "status": status, "k": k,
                        }
                    except Exception as exc:
                        for tensor, values in added.items():
                            masks[tensor].difference_update(values)
                            if not masks[tensor]:
                                del masks[tensor]
                        elapsed = (time.perf_counter() - t0) * 1000
                        write_row(group_id, ";".join(group.nodes), ratio, k / group.size,
                                  0.0, 0.0, None, None, base_params,
                                  baseline["flops_est"], baseline["peak_act_est"],
                                  "ERROR", f"{type(exc).__name__}: {exc}", elapsed)
                        yield {
                            "type": "error", "ratio": ratio, "group_id": group_id,
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    finally:
                        gc.collect()

                yield {"type": "ratio_done", "ratio": ratio}

        rss_lines.append(f"end,rss_mb={rss_mb()}")
        self.rss_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rss_log_path.write_text("\n".join(rss_lines) + "\n", encoding="utf-8")
        yield {
            "type": "done",
            "csv_path": str(self.csv_path),
            "baseline_path": str(self.baseline_path),
            "rss_log_path": str(self.rss_log_path),
            "elapsed_sec": time.perf_counter() - start,
            "baseline_miou": baseline["onnx_miou"] * 100.0,
        }
