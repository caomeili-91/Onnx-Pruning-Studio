"""Static constant folding for ONNX shape subgraphs.

This is a small replacement for onnxsim: it only folds shape/int arithmetic
(Shape/Slice/Concat/Cast/Gather/Unsqueeze/Squeeze/Reshape/Identity and
integer Add/Sub/Mul/Div). It is intentionally conservative: if anything is
dynamic or unsupported, the node is left untouched and preflight will reject it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _shape_map(model: onnx.ModelProto) -> Dict[str, Optional[Tuple[int, ...]]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes: Dict[str, Optional[Tuple[int, ...]]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor_type = value.type.tensor_type
        dims = []
        ok = tensor_type.HasField("shape")
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                ok = False
                break
        shapes[value.name] = tuple(dims) if ok else None
    for init in inferred.graph.initializer:
        shapes[init.name] = tuple(int(d) for d in init.dims)
    return shapes


def _attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def _slice_array(data: np.ndarray, starts, ends, axes, steps) -> np.ndarray:
    slices = [slice(None)] * data.ndim
    for axis, start, end, step in zip(axes, starts, ends, steps):
        axis = int(axis)
        if axis < 0:
            axis += data.ndim
        start = int(start)
        end = int(end)
        step = int(step)
        if step <= 0:
            raise ValueError("negative Slice step is not supported")
        dim = data.shape[axis]
        if start < 0:
            start += dim
        if end < 0:
            end += dim
        start = max(0, min(start, dim))
        end = max(0, min(end, dim))
        slices[axis] = slice(start, end, step)
    return data[tuple(slices)]


def _try_fold(node: onnx.NodeProto, values: Dict[str, np.ndarray], shapes: Dict[str, Optional[Tuple[int, ...]]]):
    op = node.op_type
    inputs = list(node.input)

    if op == "Constant":
        value = _attr(node, "value")
        if isinstance(value, onnx.TensorProto):
            return {node.output[0]: numpy_helper.to_array(value)}
        return None

    if op == "Shape":
        shape = shapes.get(inputs[0])
        if shape is None:
            return None
        return {node.output[0]: np.asarray(shape, dtype=np.int64)}

    if all(inp in values or inp == "" for inp in inputs):
        arrays = [values.get(inp) for inp in inputs]
    else:
        return None

    try:
        if op == "Concat":
            axis = int(_attr(node, "axis", 0))
            return {node.output[0]: np.concatenate(arrays, axis=axis)}
        if op == "Slice":
            data = arrays[0]
            rest = arrays[1:]
            if len(rest) == 3:
                starts, ends, axes = rest
                steps = np.asarray([1], dtype=np.int64)
            elif len(rest) == 4:
                starts, ends, axes, steps = rest
            else:
                return None
            return {node.output[0]: _slice_array(data, starts, ends, axes, steps)}
        if op == "Cast":
            return {node.output[0]: arrays[0].astype(np.int64)}
        if op == "Unsqueeze":
            axes = [int(a) for a in np.asarray(arrays[1]).reshape(-1)]
            return {node.output[0]: np.expand_dims(arrays[0], axis=tuple(axes))}
        if op == "Squeeze":
            if len(arrays) > 1:
                axes = [int(a) for a in np.asarray(arrays[1]).reshape(-1)]
                return {node.output[0]: np.squeeze(arrays[0], axis=tuple(axes))}
            return {node.output[0]: np.squeeze(arrays[0])}
        if op == "Gather":
            axis = int(_attr(node, "axis", 0))
            return {node.output[0]: np.take(arrays[0], arrays[1].astype(np.int64), axis=axis)}
        if op == "Reshape":
            shape = tuple(int(x) for x in np.asarray(arrays[1]).reshape(-1))
            return {node.output[0]: arrays[0].reshape(shape)}
        if op == "Identity":
            return {node.output[0]: arrays[0]}
        if op == "Transpose":
            perm = _attr(node, "perm", None)
            return {node.output[0]: np.transpose(arrays[0], perm)}
        if op in ("Add", "Sub", "Mul", "Div"):
            if op == "Add":
                out = arrays[0] + arrays[1]
            elif op == "Sub":
                out = arrays[0] - arrays[1]
            elif op == "Mul":
                out = arrays[0] * arrays[1]
            else:
                out = arrays[0] // arrays[1]
            return {node.output[0]: out}
    except Exception:
        return None
    return None


def fold_constants(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    """Return a folded copy of the model and a small report."""
    copied = onnx.ModelProto()
    copied.CopyFrom(model)
    model = copied
    shapes = _shape_map(model)
    values: Dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        values[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant":
            folded = _try_fold(node, values, shapes)
            if folded:
                values.update(folded)

    graph_outputs = {out.name for out in model.graph.output}
    folded_nodes = []
    new_constants = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue
        if any(out in graph_outputs for out in node.output):
            continue
        folded = _try_fold(node, values, shapes)
        if not folded:
            continue
        # Do not fold anything whose output is a float feature tensor.
        if any(np.asarray(arr).dtype.kind == "f" for arr in folded.values()):
            continue
        folded_nodes.append(node)
        values.update(folded)
        for name, array in folded.items():
            new_constants.append(
                helper.make_node(
                    "Constant",
                    [],
                    [name],
                    name=f"folded_{name}",
                    value=numpy_helper.from_array(np.asarray(array)),
                )
            )

    if not folded_nodes:
        return model, {"folded_nodes": 0}

    folded_ids = {id(node) for node in folded_nodes}
    folded_inputs = {inp for node in folded_nodes for inp in node.input}
    remaining = [node for node in model.graph.node if id(node) not in folded_ids and node.op_type != "Constant"]
    # keep original constants that are still referenced
    original_constants = [node for node in model.graph.node if node.op_type == "Constant" and id(node) not in folded_ids]
    nodes = new_constants + original_constants + remaining

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]

    used = {inp for node in model.graph.node for inp in node.input if inp}
    used |= {out.name for out in model.graph.output}
    used |= {inp.name for inp in model.graph.input}
    keep_inits = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_inits)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    onnx.checker.check_model(model)
    return model, {"folded_nodes": len(folded_nodes)}
