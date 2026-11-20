"""Constant folding/simplification through onnxsim.

The previous hand-written shape folder was a fallback for an environment without
onnxsim. v1.1 uses the maintained tool: onnxsim performs the shape/int folding
and graph simplification, then the caller keeps its existing checker/shape
validation gates.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import onnx

try:
    import onnxsim
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "onnxsim is required for the v1.1 graph backend. "
        "Install it with: python -m pip install onnxsim==0.7.3"
    ) from exc


def fold_constants(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    """Simplify the model with onnxsim and return a checked model.

    The function keeps the old public return shape so preflight.py and the
    verification scripts do not need to change.
    """
    before_nodes = len(model.graph.node)
    before_inits = len(model.graph.initializer)
    copied = onnx.ModelProto()
    copied.CopyFrom(model)

    simplified, ok = onnxsim.simplify(copied)
    if not ok:
        raise RuntimeError("onnxsim.simplify returned a non-equivalent model check")

    onnx.checker.check_model(simplified)
    report = {
        "backend": "onnxsim",
        "ok": True,
        "nodes_before": before_nodes,
        "nodes_after": len(simplified.graph.node),
        "initializers_before": before_inits,
        "initializers_after": len(simplified.graph.initializer),
    }
    return simplified, report
