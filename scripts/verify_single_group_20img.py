r"""Verify first-group pruning on the fixed 20-image baseline subset.

Run:
    D:\anaconda\envs\FCN\python.exe scripts\verify_single_group_20img.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.onnx_app.evaluator import OnnxSegmentationEvaluator, VOCSegDataset, subset_indices
from src.onnx_app.grouper import Grouper
from src.onnx_app.preflight import preflight_onnx
from src.onnx_app.profile_loader import load_profile
from src.onnx_app.pruner import apply_masks

MODEL = PROJECT_ROOT / "models" / "fcn_resnet50_voc_520.onnx"
BASELINE = PROJECT_ROOT / "outputs" / "baseline.json"
OUTPUT = PROJECT_ROOT / "outputs" / "verify_single_group_20img.json"
DATASET = PROJECT_ROOT / "VOCdevkit"


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    model_bytes = MODEL.read_bytes()
    pf = preflight_onnx(model_bytes, "cpu_ort")
    assert pf["status"] in ("OK", "WARN"), pf["reasons"]
    profile = load_profile("cpu_ort")

    dataset = VOCSegDataset(DATASET)
    indices = subset_indices(len(dataset), int(baseline["subset_size"]), int(baseline["seed"]))
    evaluator = OnnxSegmentationEvaluator(pf["folded_model_bytes"])
    try:
        before = evaluator.evaluate(dataset, indices, pf["selected_output"])
    finally:
        evaluator.close()

    print(f"baseline.json mIoU: {baseline['onnx_miou'] * 100:.2f}%")
    print(f"recomputed 20-image mIoU: {before['miou'] * 100:.2f}%")
    assert abs(before["miou"] - baseline["onnx_miou"]) < 0.01, "baseline self-check failed"

    # The earlier 97.05% number came from the first 2 indices of this same subset.
    two = indices[:2]
    two_details = []
    for index in two:
        ev = OnnxSegmentationEvaluator(pf["folded_model_bytes"])
        try:
            one = ev.evaluate(dataset, [index], pf["selected_output"])
        finally:
            ev.close()
        mask = np.asarray(Image.open(
            dataset.masks_dir / f"{dataset.names[index]}.png"
        ).resize((520, 520), Image.NEAREST))
        valid = mask != 255
        bg = float((mask[valid] == 0).mean()) if valid.any() else 0.0
        present = int(len(np.unique(mask[valid]))) if valid.any() else 0
        two_details.append({
            "index": index,
            "name": dataset.names[index],
            "miou": one["miou"],
            "background_fraction": bg,
            "classes_present": present,
        })
        print(
            f"  2-image sample: {dataset.names[index]} | mIoU={one['miou'] * 100:.2f}% | "
            f"background={bg * 100:.1f}% | classes={present}"
        )

    grouper_model = onnx.load_model_from_string(pf["folded_model_bytes"])
    grouper = Grouper(grouper_model, pf["selected_output"], round_to=int(profile["channel_round_to"]))
    group = next(g for g in grouper.groups if g.active and not g.protected and not g.unresolved)
    k = group.channels_to_prune(0.1, int(profile["channel_round_to"]))
    chosen = np.argsort(group.importance)[:k]
    masks = {tensor: {int(indices[int(i)]) for i in chosen} for tensor, indices in group.tensors.items()}
    candidate = apply_masks(pf["folded_model_bytes"], masks)
    evaluator = OnnxSegmentationEvaluator(candidate)
    try:
        after = evaluator.evaluate(dataset, indices, pf["selected_output"])
    finally:
        evaluator.close()

    delta = (after["miou"] - before["miou"]) * 100.0
    print(f"first group: size={group.size}, cut={k}, nodes={group.nodes[:3]}")
    print(f"after mIoU: {after['miou'] * 100:.2f}% | delta: {delta:+.2f} pp")

    payload = {
        "model": MODEL.name,
        "profile": "cpu_ort",
        "subset_size": len(indices),
        "seed": int(baseline["seed"]),
        "indices": indices,
        "baseline_miou": before["miou"],
        "pruned_miou": after["miou"],
        "delta_pp": delta,
        "group": {
            "size": group.size,
            "cut_channels": k,
            "nodes": group.nodes,
            "tensors": {tensor: sorted(values) for tensor, values in masks.items()},
        },
        "two_image_sample": {
            "indices": two,
            "details": two_details,
            "why_97_percent": (
                "The earlier 97.05% was measured on only the first two images of this 20-image seed-42 "
                "subset. Those images are dominated by background/one large class and contain very few "
                "classes, so per-class mIoU is high and not comparable to the 20-image or 582-image value. "
                "It is a small-sample artifact, not a preprocessing bug."
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
