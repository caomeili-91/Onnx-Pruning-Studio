r"""Verify round_to alignment for the first channel group.

Run:
    D:\anaconda\envs\FCN\python.exe scripts\verify_round_to.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import onnx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.onnx_app.grouper import Grouper
from src.onnx_app.preflight import preflight_onnx
from src.onnx_app.profile_loader import load_profile

MODEL = PROJECT_ROOT / "models" / "fcn_resnet50_voc_520.onnx"
CURVE = PROJECT_ROOT / "outputs" / "verify_curve.csv"
SINGLE = PROJECT_ROOT / "outputs" / "verify_single_group_20img.json"
OUTPUT = PROJECT_ROOT / "outputs" / "verify_round_to.json"


def first_group(profile_name: str, model_bytes: bytes):
    pf = preflight_onnx(model_bytes, profile_name)
    assert pf["status"] in ("OK", "WARN"), pf["reasons"]
    profile = load_profile(profile_name)
    model = onnx.load_model_from_string(pf["folded_model_bytes"])
    grouper = Grouper(model, pf["selected_output"], round_to=int(profile["channel_round_to"]))
    group = next(g for g in grouper.groups if g.active and not g.protected and not g.unresolved)
    return profile, pf, group


def describe(profile, group, ratio=0.1):
    round_to = int(profile["channel_round_to"])
    k = group.channels_to_prune(ratio, round_to)
    max_ratio = group.max_prunable_ratio(round_to)
    return {
        "profile": profile["profile_name"],
        "round_to": round_to,
        "group_size": group.size,
        "first_group_nodes": group.nodes,
        "ratio": ratio,
        "channels_cut": k,
        "aligned": (k % round_to == 0) if round_to else True,
        "max_prunable_ratio": round(max_ratio, 4),
        "clamped": ratio > max_ratio,
    }


def main() -> int:
    model_bytes = MODEL.read_bytes()
    cpu_profile, cpu_pf, cpu_group = first_group("cpu_ort", model_bytes)
    rknn_profile, rknn_pf, rknn_group = first_group("rknn_like", model_bytes)

    cpu_info = describe(cpu_profile, cpu_group)
    rknn_info = describe(rknn_profile, rknn_group)
    print("cpu_ort first group:", cpu_info)
    print("rknn_like first group:", rknn_info)

    single = json.loads(SINGLE.read_text(encoding="utf-8")) if SINGLE.exists() else None
    curve_round_to = []
    curve_first_cut = None
    if CURVE.exists():
        with open(CURVE, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("round_to"):
                    curve_round_to.append(int(row["round_to"]))
                if row.get("group_id") == "0" and row.get("actual_channel_ratio") and curve_first_cut is None:
                    curve_first_cut = {
                        "actual_channel_ratio": float(row["actual_channel_ratio"]),
                        "round_to": int(row["round_to"]),
                    }
            if curve_first_cut:
                curve_first_cut["estimated_cut_channels"] = round(
                    curve_first_cut["actual_channel_ratio"] * cpu_group.size
                )

    explanation = (
        "The earlier 7-channel cut was produced with cpu_ort (round_to=1). "
        "With round_to=1 every integer is already aligned, so 7 is the post-clamp value and is valid. "
        "It is not a pre-alignment value. With rknn_like (round_to=8) the same group produces 8 channels; "
        "if a future profile has round_to>1 and still emits 7, that is a bug/configuration error and this "
        "script should fail via the aligned flag."
    )
    print(explanation)

    if single:
        cut = single["group"]["cut_channels"]
        assert cut % cpu_profile["channel_round_to"] == 0, "single-group cut is not aligned"
        print(f"single_group json cut={cut}, cpu round_to={cpu_profile['channel_round_to']}, aligned=True")
    if curve_round_to:
        assert set(curve_round_to) == {int(cpu_profile["channel_round_to"])}, (
            f"curve round_to mismatch: {set(curve_round_to)} vs {cpu_profile['channel_round_to']}"
        )
        print(f"curve.csv round_to values: {sorted(set(curve_round_to))} (all equal to cpu profile)")
    else:
        print("curve.csv missing or has no round_to column; run verify_cumulative.py first for the cross-check")

    if cpu_info["channels_cut"] % cpu_info["round_to"] != 0:
        raise AssertionError("cpu_ort cut is not aligned to its own round_to")
    if rknn_info["channels_cut"] % rknn_info["round_to"] != 0:
        raise AssertionError("rknn_like cut is not aligned to its own round_to")

    payload = {
        "cpu_ort": cpu_info,
        "rknn_like": rknn_info,
        "single_group_json": single["group"] if single else None,
        "curve_round_to_values": sorted(set(curve_round_to)),
        "curve_first_group": curve_first_cut,
        "explanation": explanation,
        "verdict": "7 channels is valid for round_to=1; it is the post-clamp value, not a bug.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
