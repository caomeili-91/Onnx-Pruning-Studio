"""Channel dependency grouping for static CNN segmentation ONNX graphs.

Scope is deliberately conservative: Conv(groups=1), Relu, MaxPool, Add,
Resize, BatchNormalization and channel-preserving elementwise ops. Anything
else (Flatten/Reshape/Gemm/attention/grouped Conv) marks the affected region
as UNRESOLVABLE instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import onnx
from onnx import numpy_helper

NUM_CLASSES = 21


def _shape_map(model: onnx.ModelProto) -> Dict[str, tuple]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    result = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = []
        ok = value.type.tensor_type.HasField("shape")
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                ok = False
                break
        if ok:
            result[value.name] = tuple(dims)
    return result


@dataclass
class ChannelGroup:
    size: int
    tensors: Dict[str, List[int]] = field(default_factory=dict)
    importance: Optional[np.ndarray] = None
    protected: bool = False
    unresolved: bool = False
    active: bool = True
    nodes: List[str] = field(default_factory=list)

    def add_tensor(self, name: str, indices: List[int]) -> bool:
        if name in self.tensors:
            return self.tensors[name] == list(indices)
        self.tensors[name] = list(indices)
        return True

    def merge(self, other: "ChannelGroup") -> bool:
        if self.size != other.size:
            return False
        for name, indices in other.tensors.items():
            if not self.add_tensor(name, indices):
                return False
        if self.importance is None:
            self.importance = other.importance
        elif other.importance is not None:
            self.importance = (self.importance + other.importance) / 2.0
        self.protected = self.protected or other.protected
        self.unresolved = self.unresolved or other.unresolved
        self.nodes.extend(other.nodes)
        other.active = False
        return True

    def max_prunable_ratio(self, round_to: int) -> float:
        # Keep at least one aligned block. For C=64 and round_to=8 that means
        # at most 56 channels can be removed, i.e. max ratio 0.875.
        if self.size <= round_to:
            return 0.0
        return max(0.0, (self.size - round_to) / self.size)

    def channels_to_prune(self, ratio: float, round_to: int) -> int:
        target_keep = int(self.size * (1.0 - max(0.0, min(0.95, ratio))))
        target_keep = (target_keep // round_to) * round_to
        target_keep = max(round_to, target_keep)
        target_keep = min(self.size - round_to, target_keep)
        return max(0, self.size - target_keep)

    def to_dict(self, round_to: int) -> Dict[str, Any]:
        return {
            "size": self.size,
            "tensors": {name: indices for name, indices in self.tensors.items()},
            "nodes": self.nodes,
            "protected": self.protected,
            "unresolved": self.unresolved,
            "max_prunable_ratio": round(self.max_prunable_ratio(round_to), 4),
        }


class Grouper:
    def __init__(self, model: onnx.ModelProto, output_name: str, round_to: int = 1) -> None:
        self.model = model
        self.output_name = output_name
        self.round_to = int(round_to)
        self.shapes = _shape_map(model)
        self.initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        self.tensor_group: Dict[str, ChannelGroup] = {}
        self.groups: List[ChannelGroup] = []
        self.unresolved_nodes: List[Dict[str, str]] = []
        self._build()

    def _channel_count(self, name: str) -> Optional[int]:
        shape = self.shapes.get(name)
        if shape is None or len(shape) != 4:
            return None
        return int(shape[1])

    def _new_group(self, tensor: str, size: int, importance=None, node: str = "") -> ChannelGroup:
        group = ChannelGroup(size=size, importance=importance, nodes=[node] if node else [])
        group.add_tensor(tensor, list(range(size)))
        self.groups.append(group)
        self.tensor_group[tensor] = group
        return group

    def _pass(self, input_name: str, output_name: str, node: str) -> None:
        group = self.tensor_group.get(input_name)
        if group is None:
            count = self._channel_count(output_name)
            if count is not None:
                self._new_group(output_name, count, node=node)
            return
        if not group.add_tensor(output_name, group.tensors[input_name]):
            group.unresolved = True
            self.unresolved_nodes.append({"node": node, "reason": "tensor already in incompatible group"})
        self.tensor_group[output_name] = group
        if node not in group.nodes:
            group.nodes.append(node)

    def _unresolve(self, node: str, tensors: List[str], reason: str) -> None:
        self.unresolved_nodes.append({"node": node, "reason": reason})
        for tensor in tensors:
            group = self.tensor_group.get(tensor)
            if group is not None:
                group.unresolved = True

    def _build(self) -> None:
        for node in self.model.graph.node:
            op = node.op_type
            inputs = [name for name in node.input if name]
            outputs = [name for name in node.output if name]

            if op == "Conv":
                out_name = outputs[0]
                out_count = self._channel_count(out_name)
                if out_count is None:
                    self._unresolve(node.name, outputs + inputs, "Conv output is not 4D")
                    continue
                groups = 1
                dilations = [1, 1]
                for attr in node.attribute:
                    if attr.name == "groups":
                        groups = int(attr.i)
                    if attr.name == "dilations":
                        dilations = list(attr.ints)
                if groups != 1:
                    self._unresolve(node.name, outputs + inputs, "grouped/depthwise Conv is not supported in v1")
                    continue
                weight = self.initializers.get(node.input[1]) if len(node.input) > 1 else None
                if weight is None or weight.ndim != 4:
                    self._unresolve(node.name, outputs + inputs, "Conv weight initializer missing")
                    continue
                importance = np.abs(weight).sum(axis=(1, 2, 3)).astype(np.float64)
                if importance.max() > 0:
                    importance = importance / importance.max()
                self._new_group(out_name, out_count, importance=importance, node=node.name)
                continue

            if op == "BatchNormalization":
                self._pass(inputs[0], outputs[0], node.name)
                continue

            if op in {"Relu", "LeakyRelu", "Clip", "MaxPool", "AveragePool", "GlobalAveragePool",
                      "Resize", "Upsample", "Sigmoid", "Softmax", "Dropout", "Identity"}:
                if op == "Resize" and self.shapes.get(outputs[0]) is None:
                    self._unresolve(node.name, outputs + inputs, "Resize output shape unknown")
                    continue
                self._pass(inputs[0], outputs[0], node.name)
                continue

            if op in {"Add", "Sub", "Mul"}:
                groups = [self.tensor_group.get(name) for name in inputs]
                feature_groups = [group for group in groups if group is not None]
                if not feature_groups:
                    self._unresolve(node.name, outputs + inputs, "elementwise inputs are not grouped")
                    continue
                group = feature_groups[0]
                for other in feature_groups[1:]:
                    if group is other:
                        continue
                    if not group.merge(other):
                        self._unresolve(node.name, outputs + inputs, "elementwise groups cannot be merged")
                        break
                    for tensor_name in other.tensors:
                        self.tensor_group[tensor_name] = group
                else:
                    reference = next(iter(group.tensors.values()))
                    if not group.add_tensor(outputs[0], reference):
                        self._unresolve(node.name, outputs + inputs, "elementwise output incompatible")
                    else:
                        self.tensor_group[outputs[0]] = group
                        if node.name not in group.nodes:
                            group.nodes.append(node.name)
                    continue
                continue

            if op == "Concat":
                self._unresolve(node.name, outputs + inputs, "feature Concat is not supported in v1")
                continue

            if op in {"Shape", "Slice", "Cast", "Unsqueeze", "Squeeze", "Gather", "Constant", "Flatten", "Transpose"}:
                continue

            if op in {"Gemm", "MatMul", "Reshape", "ConvTranspose", "Split", "ReduceMean", "LayerNormalization"}:
                self._unresolve(node.name, outputs + inputs, f"{op} is not supported in v1")
                continue

            self._unresolve(node.name, outputs + inputs, f"operator {op} has no grouping rule")

        # protect graph output groups and all num_classes heads
        for out in self.model.graph.output:
            group = self.tensor_group.get(out.name)
            if group is not None:
                group.protected = True
        for tensor, group in self.tensor_group.items():
            if group.size == NUM_CLASSES:
                group.protected = True

    def summary(self) -> Dict[str, Any]:
        active = [group for group in self.groups if group.active]
        return {
            "group_count": len(active),
            "protected_groups": sum(1 for group in active if group.protected),
            "unresolved_groups": sum(1 for group in active if group.unresolved),
            "unresolved_nodes": self.unresolved_nodes,
            "groups": [group.to_dict(self.round_to) for group in active],
        }
