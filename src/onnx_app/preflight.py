"""ONNX preflight: target-profile rule checks before any pruning.

The product promise is deliberately narrow: static-shape CNN segmentation
models, evaluated with mIoU. Anything dynamic/unsupported is rejected here,
before baseline or grouping.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from src.onnx_app.constant_fold import fold_constants
from src.onnx_app.profile_loader import load_profile

INPUT_SIZE = (520, 520)  # v1 platform contract; documented in baseline.json
NUM_CLASSES = 21

STRUCTURAL_OPS = {
    "Constant", "Shape", "Gather", "Slice", "Concat", "Unsqueeze", "Squeeze",
    "Cast", "Identity", "Flatten", "Transpose",
}
CONTROL_OPS = {"If", "Loop", "Scan"}
QDQ_OPS = {
    "QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul",
    "ConvInteger", "MatMulInteger", "DynamicQuantizeLinear",
}
CONST_PARAM_OPS = {"Slice", "Pad", "Transpose", "Split", "Reshape", "Unsqueeze", "Squeeze", "Gather"}


def _reason(
    category: str,
    node: str = "",
    op_type: str = "",
    violated_field: str = "",
    expected: Any = "",
    actual: Any = "",
    next_action: str = "",
    severity: str = "ERROR",
) -> Dict[str, Any]:
    return {
        "category": category,
        "node_name": node or "(graph)",
        "op_type": op_type or "-",
        "violated_field": violated_field,
        "expected": expected,
        "actual": actual,
        "next_action": next_action,
        "severity": severity,
    }


def _dims(value: onnx.ValueInfoProto) -> List[int | str]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    result: List[int | str] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(dim.dim_param or "?")
    return result


def _tensor_type(value: onnx.ValueInfoProto) -> int:
    return int(value.type.tensor_type.elem_type)


def _normalise_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_normalise_attr(item) for item in value]
    return value


def _shape_map(model: onnx.ModelProto) -> Dict[str, Tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes: Dict[str, Tuple[int, ...]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = _dims(value)
        if dims and all(isinstance(dim, int) and dim > 0 for dim in dims):
            shapes[value.name] = tuple(int(dim) for dim in dims)
    return shapes


def _constant_names(model: onnx.ModelProto) -> set[str]:
    names = {init.name for init in model.graph.initializer}
    names |= {node.output[0] for node in model.graph.node if node.op_type == "Constant"}
    return names


def _estimate_resources(model: onnx.ModelProto, shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, float]:
    params = 0
    weight_arrays = {}
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        weight_arrays[init.name] = array
        if array.dtype.kind in "fiub":
            params += int(array.size)

    flops = 0
    for node in model.graph.node:
        out_shape = shapes.get(node.output[0]) if node.output else None
        if out_shape is None:
            continue
        if node.op_type == "Conv":
            if not node.input or node.input[1] not in weight_arrays:
                continue
            weight = weight_arrays[node.input[1]]
            if weight.ndim != 4:
                continue
            cout, cin, kh, kw = weight.shape
            groups = 1
            for attr in node.attribute:
                if attr.name == "groups":
                    groups = int(attr.i)
            out_elems = int(np.prod(out_shape))
            flops += int(2 * out_elems * cin * kh * kw / max(groups, 1))
        elif node.op_type in {"Add", "Sub", "Mul", "Div"}:
            flops += int(np.prod(out_shape))
        elif node.op_type in {"Relu", "Sigmoid", "LeakyRelu", "Clip"}:
            flops += int(np.prod(out_shape))

    peak = 0
    for shape in shapes.values():
        peak = max(peak, int(np.prod(shape)) * 4)
    return {
        "params": params,
        "flops": flops,
        "peak_act_bytes": peak,
        "params_mb": params * 4 / (1024 * 1024),
        "flops_g": flops / 1e9,
        "peak_act_mb": peak / (1024 * 1024),
    }


def _check_profile_attr(
    node: onnx.NodeProto,
    profile: Dict[str, Any],
    reasons: List[Dict[str, Any]],
) -> None:
    spec = profile["ops"].get(node.op_type)
    if spec is None:
        return
    allowed = spec.get("allowed_attributes", {}) or {}
    for attr in node.attribute:
        if attr.name not in allowed:
            continue
        allowed_values = allowed[attr.name]
        if allowed_values is None:
            continue
        actual = _normalise_attr(onnx.helper.get_attribute_value(attr))
        if actual not in allowed_values:
            reasons.append(_reason(
                "ATTRIBUTE", node.name, node.op_type, attr.name,
                allowed_values, actual,
                f"将该 {node.op_type} 的 {attr.name} 改为 profile 允许的取值，或换 target profile",
            ))


def preflight_onnx(
    model_bytes: bytes,
    profile_name: str = "cpu_ort",
    output_name: Optional[str] = None,
) -> Dict[str, Any]:
    profile = load_profile(profile_name)
    reasons: List[Dict[str, Any]] = []
    model_hash = hashlib.sha256(model_bytes).hexdigest()

    try:
        model = onnx.load_model_from_string(model_bytes)
        onnx.checker.check_model(model)
    except Exception as exc:
        return {
            "status": "REJECT",
            "profile_name": profile_name,
            "profile_version": profile.get("profile_version", "?"),
            "model_hash": model_hash,
            "reasons": [_reason(
                "INPUT_SIGNATURE", "", "", "onnx", "valid ONNX", str(exc),
                "用 onnx.checker 修复模型后重新上传",
            )],
            "warnings": [],
            "input_signature": None,
            "outputs": [],
            "op_histogram": {},
        }

    try:
        folded, fold_report = fold_constants(model)
    except Exception as exc:
        return {
            "status": "REJECT",
            "profile_name": profile_name,
            "profile_version": profile.get("profile_version", "?"),
            "model_hash": model_hash,
            "reasons": [_reason(
                "STRUCTURAL_OP", "", "", "constant_fold", "success", str(exc),
                "先修复/简化模型中的 shape 子图，确保常量折叠可完成",
            )],
            "warnings": [],
            "input_signature": None,
            "outputs": [],
            "op_histogram": {},
        }

    try:
        onnx.checker.check_model(folded)
        shapes = _shape_map(folded)
    except Exception as exc:
        return {
            "status": "REJECT",
            "profile_name": profile_name,
            "profile_version": profile.get("profile_version", "?"),
            "model_hash": model_hash,
            "reasons": [_reason(
                "STRUCTURAL_OP", "", "", "shape_inference", "success", str(exc),
                "修复模型 shape 后重试",
            )],
            "warnings": [],
            "input_signature": None,
            "outputs": [],
            "op_histogram": {},
        }

    # IR / opset
    if not (profile["ir_version_range"][0] <= folded.ir_version <= profile["ir_version_range"][1]):
        reasons.append(_reason(
            "OPSET_IR", "", "", "ir_version",
            profile["ir_version_range"], folded.ir_version,
            "用 profile 支持的 opset 重新导出 ONNX",
        ))
    for opset in folded.opset_import:
        if opset.domain not in ("", "ai.onnx") and opset.domain != "ai.onnx.ml":
            reasons.append(_reason(
                "OPSET_IR", "", "", "domain",
                "ai.onnx", opset.domain,
                "移除自定义 domain，或换支持它的 target profile",
            ))
        elif not (profile["opset_range"][0] <= opset.version <= profile["opset_range"][1]):
            reasons.append(_reason(
                "OPSET_IR", "", "", "opset_version",
                profile["opset_range"], opset.version,
                "用 profile 支持的 opset 重新导出 ONNX",
            ))

    # input contract
    if len(folded.graph.input) != 1:
        reasons.append(_reason(
            "INPUT_SIGNATURE", "", "", "graph.input",
            "1 image input", len(folded.graph.input),
            "v1 只接受单图像输入模型",
        ))
    input_signature = None
    if folded.graph.input:
        graph_input = folded.graph.input[0]
        dims = _dims(graph_input)
        input_signature = {"name": graph_input.name, "shape": dims, "dtype": _tensor_type(graph_input)}
        if dims != [1, 3, INPUT_SIZE[0], INPUT_SIZE[1]]:
            reasons.append(_reason(
                "INPUT_SIGNATURE", "", "", "input_shape",
                [1, 3, INPUT_SIZE[0], INPUT_SIZE[1]], dims,
                "v1 固定契约要求 [1,3,520,520]；请以静态 batch=1、520x520 重新导出",
            ))
        if _tensor_type(graph_input) != onnx.TensorProto.FLOAT:
            reasons.append(_reason(
                "INPUT_SIGNATURE", "", "", "input_dtype",
                "float32", _tensor_type(graph_input),
                "使用 float32 输入重新导出；v1 不支持 float16/QDQ",
            ))

    # output selection
    output_names = [out.name for out in folded.graph.output]
    if len(output_names) > 1 and not output_name:
        reasons.append(_reason(
            "OUTPUT_SELECT", "", "", "output_name",
            "user selected output", output_names,
            "在 UI 选择评估输出名（必须是 [1,21,520,520] 的分割 logits）",
        ))
    selected = output_name or (output_names[0] if output_names else None)
    selected_value = next((out for out in folded.graph.output if out.name == selected), None)
    if selected_value is None:
        reasons.append(_reason(
            "OUTPUT_SELECT", "", "", "output_name",
            "existing output", selected,
            "选择一个存在的输出名",
        ))
    else:
        dims = _dims(selected_value)
        if dims != [1, NUM_CLASSES, INPUT_SIZE[0], INPUT_SIZE[1]]:
            reasons.append(_reason(
                "OUTPUT_SELECT", "", "", "output_shape",
                [1, NUM_CLASSES, INPUT_SIZE[0], INPUT_SIZE[1]], dims,
                "v1 只评估 [1,21,520,520] 的语义分割 logits",
            ))
        if _tensor_type(selected_value) != onnx.TensorProto.FLOAT:
            reasons.append(_reason(
                "OUTPUT_SELECT", "", "", "output_dtype",
                "float32", _tensor_type(selected_value),
                "使用 float32 logits 输出",
            ))

    # intermediate dynamic shapes
    for value in folded.graph.value_info:
        dims = _dims(value)
        if not dims:
            continue
        if any(isinstance(dim, str) or (isinstance(dim, int) and dim <= 0) for dim in dims):
            reasons.append(_reason(
                "DYNAMIC_SHAPE", value.name, "", "intermediate_dims",
                "static dims", dims,
                "消除该中间张量的动态维，或重新导出静态 shape 模型",
            ))
            break

    # operators
    const_names = _constant_names(folded)
    structural = set(profile.get("structural_ops", [])) | STRUCTURAL_OPS
    op_counter = Counter(node.op_type for node in folded.graph.node)
    for node in folded.graph.node:
        if node.op_type in QDQ_OPS:
            reasons.append(_reason(
                "QDQ", node.name, node.op_type, "op_type",
                "float32 graph", node.op_type,
                "v1 不支持 QDQ；请上传 float32 ONNX",
            ))
            continue
        if node.op_type in CONTROL_OPS:
            reasons.append(_reason(
                "COMPUTE_OP", node.name, node.op_type, "op_type",
                "whitelist", node.op_type,
                "移除控制流，或换支持控制流的 target（v1 不支持）",
            ))
            continue
        if node.domain not in ("", "ai.onnx"):
            reasons.append(_reason(
                "COMPUTE_OP", node.name, node.op_type, "domain",
                "ai.onnx", node.domain,
                "移除自定义 domain 算子",
            ))
            continue
        if node.op_type == "NonMaxSuppression":
            reasons.append(_reason(
                "COMPUTE_OP", node.name, node.op_type, "op_type",
                "whitelist", node.op_type,
                "v1 不做检测；请上传分割 logits 模型",
            ))
            continue

        is_structural = node.op_type in structural and node.op_type not in profile["ops"]
        if node.op_type in CONST_PARAM_OPS and node.op_type in structural:
            for inp in node.input:
                if inp and inp not in const_names:
                    reasons.append(_reason(
                        "STRUCTURAL_OP", node.name, node.op_type, "parameter_input",
                        "initializer/Constant", inp,
                        f"将该 {node.op_type} 的参数改为 initializer 或 Constant",
                    ))
                    break
        if node.op_type == "Resize":
            for idx, inp in enumerate(node.input):
                if idx == 0 or not inp:
                    continue
                if inp not in const_names:
                    reasons.append(_reason(
                        "STRUCTURAL_OP", node.name, node.op_type, f"input[{idx}]",
                        "initializer/Constant", inp,
                        "将 Resize 的 scales/sizes 改为 initializer 或 Constant",
                    ))
        if node.op_type == "BatchNormalization":
            for attr in node.attribute:
                if attr.name == "training_mode" and int(attr.i) != 0:
                    reasons.append(_reason(
                        "ATTRIBUTE", node.name, node.op_type, "training_mode",
                        "0", int(attr.i),
                        "以 inference 模式导出 BatchNormalization（training_mode=0）",
                    ))

        if node.op_type not in profile["ops"] and not is_structural:
            reasons.append(_reason(
                "COMPUTE_OP", node.name, node.op_type, "op_type",
                list(profile["ops"].keys()), node.op_type,
                "移除该算子，或换 target profile",
            ))
            continue
        _check_profile_attr(node, profile, reasons)

    # resources
    resources = _estimate_resources(folded, shapes)
    resource_warnings = []
    for key, label in (
        ("params", "params_max"),
        ("flops", "flops_max"),
        ("peak_act_bytes", "peak_act_max"),
    ):
        limit = profile["resource_limits"].get(label)
        if not limit:
            continue
        actual = resources[key]
        if actual > limit["value"]:
            severity = limit.get("severity", profile["severity_default"])
            item = _reason(
                "RESOURCE", "", "", label, limit["value"], actual,
                f"该模型超出 {profile_name} 的 {label} 上限；换 profile 或换模型",
                severity=severity,
            )
            if severity == "REJECT":
                reasons.append(item)
            else:
                resource_warnings.append(item)

    errors = [item for item in reasons if item.get("severity", "ERROR") != "WARN"]
    status = "REJECT" if errors else ("WARN" if resource_warnings else "OK")
    return {
        "status": status,
        "profile_name": profile_name,
        "profile_version": profile.get("profile_version", "?"),
        "schema_version": profile.get("schema_version", "?"),
        "model_hash": model_hash,
        "ir_version": folded.ir_version,
        "opset": [(opset.domain, opset.version) for opset in folded.opset_import],
        "input_signature": input_signature,
        "outputs": [
            {"name": out.name, "shape": _dims(out), "dtype": _tensor_type(out)}
            for out in folded.graph.output
        ],
        "selected_output": selected,
        "op_histogram": dict(sorted(op_counter.items())),
        "params": int(resources["params"]),
        "flops_est": int(resources["flops"]),
        "peak_act_est": int(resources["peak_act_bytes"]),
        "params_mb": round(resources["params_mb"], 3),
        "flops_g": round(resources["flops_g"], 3),
        "peak_act_mb": round(resources["peak_act_mb"], 3),
        "reasons": reasons,
        "warnings": resource_warnings,
        "fold_report": fold_report,
        "folded_model_bytes": folded.SerializeToString(),
    }
