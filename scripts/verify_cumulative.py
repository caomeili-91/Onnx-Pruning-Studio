r"""Verify cumulative D1 semantics: 3 groups x 2 ratios on 20 fixed images.

Run:
    D:\anaconda\envs\FCN\python.exe scripts\verify_cumulative.py
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import onnx
from onnx import numpy_helper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.onnx_app.evaluator import OnnxSegmentationEvaluator, VOCSegDataset, subset_indices
from src.onnx_app.explorer import CSV_FIELDS
from src.onnx_app.grouper import Grouper
from src.onnx_app.preflight import _estimate_resources, _shape_map, preflight_onnx
from src.onnx_app.profile_loader import load_profile
from src.onnx_app.pruner import apply_masks

MODEL = PROJECT_ROOT / "models" / "fcn_resnet50_voc_520.onnx"
BASELINE = PROJECT_ROOT / "outputs" / "baseline.json"
CURVE = PROJECT_ROOT / "outputs" / "verify_curve.csv"
JSON_OUT = PROJECT_ROOT / "outputs" / "verify_cumulative.json"
DATASET = PROJECT_ROOT / "VOCdevkit"


def channel_map(model: onnx.ModelProto):
    """tensor -> ('out'|'in', initializer_name, dim_index) for Conv weights."""
    arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    result = {}
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight = arrays.get(node.input[1])
        if weight is None or weight.ndim != 4:
            continue
        result[node.output[0]] = ("out", node.input[1], 0)
        result[node.input[0]] = ("in", node.input[1], 1)
    return result


def assert_prior_masks(candidate_bytes: bytes, prior: Dict[str, Set[int]], original_model: onnx.ModelProto) -> None:
    if not prior:
        return
    original_shapes = _shape_map(original_model)
    candidate = onnx.load_model_from_string(candidate_bytes)
    candidate_shapes = _shape_map(candidate)
    convs = channel_map(candidate)
    candidate_arrays = {init.name: numpy_helper.to_array(init) for init in candidate.graph.initializer}
    checked = 0
    for tensor, removed in prior.items():
        if not removed:
            continue
        expected = None
        if tensor in original_shapes and len(original_shapes[tensor]) == 4:
            expected = original_shapes[tensor][1] - len(removed)
        if expected is not None and tensor in candidate_shapes and len(candidate_shapes[tensor]) == 4:
            actual = candidate_shapes[tensor][1]
        elif tensor in convs:
            _kind, weight_name, axis = convs[tensor]
            actual = int(candidate_arrays[weight_name].shape[axis])
            if expected is None:
                expected = actual + len(removed)
        else:
            continue
        assert actual == expected, f"prior mask not applied for {tensor}: {actual} != {expected}"
        checked += 1
    print(f"  ASSERT prior masks applied: {checked} tensors checked")


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    original_bytes = MODEL.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    pf = preflight_onnx(original_bytes, "cpu_ort")
    assert pf["status"] in ("OK", "WARN"), pf["reasons"]
    profile = load_profile("cpu_ort")
    round_to = int(profile["channel_round_to"])
    dataset = VOCSegDataset(DATASET)
    indices = subset_indices(len(dataset), int(baseline["subset_size"]), int(baseline["seed"]))

    base_eval = OnnxSegmentationEvaluator(pf["folded_model_bytes"])
    try:
        base = base_eval.evaluate(dataset, indices, pf["selected_output"])
    finally:
        base_eval.close()
    print(f"baseline 20-image mIoU: {base['miou'] * 100:.2f}%")
    assert abs(base["miou"] - baseline["onnx_miou"]) < 0.01

    original_model = onnx.load_model_from_string(pf["folded_model_bytes"])
    grouper = Grouper(original_model, pf["selected_output"], round_to=round_to)
    groups = [g for g in grouper.groups if g.active and not g.protected and not g.unresolved][:3]
    assert len(groups) == 3, f"expected 3 candidate groups, got {len(groups)}"
    total_group_channels = sum(g.size for g in groups)
    print("groups:", [(g.size, g.nodes[:1]) for g in groups])

    ratios = (0.1, 0.2)
    rows: List[Dict[str, object]] = []
    assertions: List[str] = []
    step = 0
    base_params = baseline["params"]

    def write_row(**kw):
        nonlocal step
        row = {field: kw.get(field, "") for field in CSV_FIELDS}
        row.update(kw)
        row["step"] = step
        row["profile_name"] = "cpu_ort"
        row["profile_version"] = pf["profile_version"]
        row["model_hash"] = pf["model_hash"]
        row["input_signature"] = json.dumps(pf["input_signature"], ensure_ascii=False)
        row["provider"] = ",".join(base["providers"])
        rows.append(row)
        writer.writerow(row)
        step += 1

    CURVE.parent.mkdir(parents=True, exist_ok=True)
    with open(CURVE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        write_row(
            group_id="BASELINE", group_nodes="", nominal_ratio=0.0,
            actual_channel_ratio=0.0, cumulative_channel_ratio=0.0,
            cumulative_param_ratio=0.0, round_to=round_to,
            miou=f"{base['miou'] * 100:.4f}", delta_vs_baseline="0.0000",
            params=base_params, flops_est=baseline["flops_est"],
            peak_act_est=baseline["peak_act_est"], status="OK", error_msg="",
            elapsed_ms=f"{base['ms_per_image'] * len(indices):.1f}",
        )

        for ratio in ratios:
            accepted: Dict[str, Set[int]] = {}
            total_pruned = 0
            last_cum = 0.0
            for group_id, group in enumerate(groups):
                k = group.channels_to_prune(ratio, round_to)
                if k <= 0:
                    continue
                chosen = np.argsort(group.importance)[:k]
                added: Dict[str, Set[int]] = {}
                for tensor, channel_indices in group.tensors.items():
                    for local in chosen:
                        added.setdefault(tensor, set()).add(int(channel_indices[int(local)]))

                # The k-th candidate is built from the original ONNX plus all
                # previous accepted masks plus this group's new mask.
                expected = {tensor: set(values) for tensor, values in accepted.items()}
                for tensor, values in added.items():
                    expected.setdefault(tensor, set()).update(values)
                candidate = apply_masks(pf["folded_model_bytes"], expected)
                assert_prior_masks(candidate, accepted, original_model)
                assertions.append(f"ratio={ratio} step={step} prior_masks_applied=True")

                t0 = time.perf_counter()
                try:
                    evaluator = OnnxSegmentationEvaluator(candidate)
                    try:
                        metrics = evaluator.evaluate(dataset, indices, pf["selected_output"])
                    finally:
                        evaluator.close()
                    candidate_model = onnx.load_model_from_string(candidate)
                    resources = _estimate_resources(candidate_model, _shape_map(candidate_model))
                    accepted = expected
                    total_pruned += k
                    cumulative_channels = total_pruned / max(1, total_group_channels)
                    assert cumulative_channels > last_cum, "cumulative_channel_ratio must increase"
                    last_cum = cumulative_channels
                    cumulative_params = (base_params - resources["params"]) / max(1, base_params)
                    elapsed = (time.perf_counter() - t0) * 1000
                    status = "CLAMPED" if ratio > group.max_prunable_ratio(round_to) else "OK"
                    write_row(
                        group_id=group_id, group_nodes=";".join(group.nodes),
                        nominal_ratio=f"{ratio:.4f}", actual_channel_ratio=f"{k / group.size:.4f}",
                        cumulative_channel_ratio=f"{cumulative_channels:.4f}",
                        cumulative_param_ratio=f"{cumulative_params:.4f}", round_to=round_to,
                        miou=f"{metrics['miou'] * 100:.4f}",
                        delta_vs_baseline=f"{(metrics['miou'] - base['miou']) * 100:.4f}",
                        params=int(resources["params"]), flops_est=int(resources["flops"]),
                        peak_act_est=int(resources["peak_act_bytes"]), status=status,
                        error_msg="", elapsed_ms=f"{elapsed:.1f}",
                    )
                    print(
                        f"  ratio={ratio:.2f} group={group_id} k={k} "
                        f"mIoU={metrics['miou'] * 100:.2f}% cumulative={cumulative_channels * 100:.2f}%"
                    )
                except Exception as exc:
                    elapsed = (time.perf_counter() - t0) * 1000
                    write_row(
                        group_id=group_id, group_nodes=";".join(group.nodes),
                        nominal_ratio=f"{ratio:.4f}", actual_channel_ratio=f"{k / group.size:.4f}",
                        cumulative_channel_ratio=f"{last_cum:.4f}",
                        cumulative_param_ratio="", round_to=round_to, miou="", delta_vs_baseline="",
                        params=base_params, flops_est=baseline["flops_est"],
                        peak_act_est=baseline["peak_act_est"], status="ERROR",
                        error_msg=f"{type(exc).__name__}: {exc}", elapsed_ms=f"{elapsed:.1f}",
                    )
                    assertions.append(f"ratio={ratio} step={step} rollback=True error={exc}")
                    print(f"  ERROR group={group_id}: {exc} (accepted masks unchanged)")
                finally:
                    gc.collect()
                assert hashlib.sha256(original_bytes).hexdigest() == original_hash, "original ONNX mutated!"
            print(f"ratio {ratio}: accepted masks cover {len(accepted)} tensors; original ONNX untouched")

    # Explicit rollback probe: a deliberately invalid candidate must not mutate
    # the original bytes or the accepted mask state.
    probe_group = groups[0]
    probe = {
        tensor: set(range(probe_group.size))
        for tensor in probe_group.tensors
    }
    probe_error = None
    try:
        probe_candidate = apply_masks(pf["folded_model_bytes"], probe)
        # The mask is deliberately invalid for ORT (zero channels). Building
        # the session forces the candidate to fail; accepted state must remain
        # empty and the original bytes must remain untouched.
        OnnxSegmentationEvaluator(probe_candidate).close()
    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
    assert hashlib.sha256(original_bytes).hexdigest() == original_hash, "original ONNX mutated by rollback probe!"
    print(f"  rollback probe: {probe_error or 'candidate built without exception; no accepted state changed'}")
    assertions.append(f"rollback_probe={probe_error or 'no_exception'} original_untouched=True")

    assertions.append("cumulative_channel_ratio_monotonic=True")
    assertions.append("step_k_contains_prior_masks=True")
    payload = {
        "baseline_miou": base["miou"],
        "subset_size": len(indices),
        "seed": int(baseline["seed"]),
        "profiles": {"name": "cpu_ort", "round_to": round_to},
        "groups": [{"size": g.size, "nodes": g.nodes} for g in groups],
        "assertions": assertions,
        "curve_path": str(CURVE),
        "rows": rows,
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {CURVE}")
    print(f"wrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
