"""Apply channel masks to an ONNX ModelProto (in memory)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import onnx
from onnx import numpy_helper


def _keep_indices(size: int, removed: Iterable[int]) -> List[int]:
    removed = set(int(i) for i in removed)
    return [i for i in range(size) if i not in removed]


def apply_masks(model_bytes: bytes, masks: Dict[str, Set[int]]) -> bytes:
    model = onnx.load_model_from_string(model_bytes)
    arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    def slice_param(name: str, axis: int, removed: Set[int]) -> None:
        if name not in arrays or not removed:
            return
        array = arrays[name]
        if array.ndim <= axis:
            return
        keep = _keep_indices(array.shape[axis], removed)
        arrays[name] = np.take(array, keep, axis=axis)

    for node in model.graph.node:
        op = node.op_type
        if op == "Conv":
            out_mask = masks.get(node.output[0], set())
            in_mask = masks.get(node.input[0], set()) if node.input else set()
            weight_name = node.input[1] if len(node.input) > 1 else ""
            if weight_name in arrays:
                weight = arrays[weight_name]
                if weight.ndim == 4:
                    if out_mask:
                        weight = np.take(weight, _keep_indices(weight.shape[0], out_mask), axis=0)
                    if in_mask:
                        weight = np.take(weight, _keep_indices(weight.shape[1], in_mask), axis=1)
                    arrays[weight_name] = weight
            if len(node.input) > 2 and node.input[2] and out_mask:
                slice_param(node.input[2], 0, out_mask)
        elif op == "BatchNormalization":
            mask = masks.get(node.output[0], set()) or masks.get(node.input[0], set())
            for param in node.input[1:5]:
                if param:
                    slice_param(param, 0, mask)
        elif op in {"Add", "Sub", "Mul"}:
            # v1 anchor has no channel-broadcast constants in these nodes.
            # A non-scalar channel constant here would make the graph invalid;
            # such a region is marked UNRESOLVABLE by the grouper.
            continue

    del model.graph.initializer[:]
    model.graph.initializer.extend(numpy_helper.from_array(array, name) for name, array in arrays.items())
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    return inferred.SerializeToString()
