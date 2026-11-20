"""Apply channel masks through onnx-graphsurgeon.

v1.1 replaces the previous raw protobuf graph surgery with GraphSurgeon. The
product code still decides the masks; GraphSurgeon performs the initializer
slicing, graph cleanup, and export.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def _keep_indices(size: int, removed: Iterable[int]) -> List[int]:
    removed = {int(value) for value in removed}
    return [index for index in range(size) if index not in removed]


def _constant_values(tensor):
    values = getattr(tensor, "values", None)
    if values is None:
        return None
    return np.asarray(values)


def _take_constant(tensor, axis: int, removed: Set[int]) -> None:
    if tensor is None or not removed:
        return
    values = _constant_values(tensor)
    if values is None or values.ndim <= axis:
        return
    keep = _keep_indices(values.shape[axis], removed)
    tensor.values = np.take(values, keep, axis=axis)


def apply_masks(model_bytes: bytes, masks: Dict[str, Set[int]]) -> bytes:
    """Return a new ONNX model with the requested channel masks applied."""
    model = onnx.load_model_from_string(model_bytes)
    graph = gs.import_onnx(model)

    for node in graph.nodes:
        if node.op == "Conv":
            out_mask = masks.get(node.outputs[0].name, set()) if node.outputs else set()
            in_mask = masks.get(node.inputs[0].name, set()) if node.inputs else set()
            if len(node.inputs) > 1:
                weight = node.inputs[1]
                values = _constant_values(weight)
                if values is not None and values.ndim == 4:
                    if in_mask:
                        _take_constant(weight, 1, in_mask)
                    if out_mask:
                        _take_constant(weight, 0, out_mask)
            if len(node.inputs) > 2:
                _take_constant(node.inputs[2], 0, out_mask)
        elif node.op == "BatchNormalization":
            mask = masks.get(node.outputs[0].name, set()) if node.outputs else set()
            if not mask and node.inputs:
                mask = masks.get(node.inputs[0].name, set())
            for tensor in node.inputs[1:5]:
                _take_constant(tensor, 0, mask)
        # Elementwise channel constants are not part of the v1 anchor. The
        # grouper marks such regions UNRESOLVABLE rather than guessing.

    graph.cleanup().toposort()
    exported = gs.export_onnx(graph)
    del exported.graph.value_info[:]
    onnx.checker.check_model(exported)
    inferred = onnx.shape_inference.infer_shapes(exported, strict_mode=False)
    return inferred.SerializeToString()
