r"""Minimal ONNX pruning analysis UI (v1).

Only the presentation layer changed; preflight/grouper/pruner/explorer are
untouched. Target profile is fixed to cpu_ort; ordering is fixed to the
Explorer default (l1_asc); seed is fixed to 42; output is auto-selected.
"""

from __future__ import annotations

import csv
import html
import logging
import os
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windowed PyInstaller builds have no console. Redirect stdio to a file before
# importing Gradio/matplotlib so startup errors are not silent.
IS_FROZEN = getattr(sys, "frozen", False)
BASE_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parent
_LOG_DIR = BASE_DIR / "outputs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_STREAM = open(_LOG_DIR / "app.log", "w", encoding="utf-8", buffering=1)
sys.stdout = _LOG_STREAM
sys.stderr = _LOG_STREAM
logging.basicConfig(
    stream=_LOG_STREAM,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.info("starting OnnxPruneStudio; frozen=%s base=%s", IS_FROZEN, BASE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError:
    traceback.print_exc()
    raise

if not IS_FROZEN:
    sys.path.insert(0, str(BASE_DIR))

from src.onnx_app.explorer import Explorer
from src.onnx_app.preflight import preflight_onnx
from src.onnx_app.profile_loader import load_profile

URL = "http://127.0.0.1:7860"
FIXED_PROFILE = "cpu_ort"
FIXED_SEED = 42
HEADER = (
    "只接受静态 [1,3,520,520] 输入、[1,21,520,520] 输出的 ONNX 模型。"
    "评估指标为 VOC 21 类 mIoU，不运行厂商转换器，当前为纯 CPU 评估。"
)
STATE: Dict[str, Any] = {
    "onnx_path": None,
    "onnx_bytes": None,
    "preflight": None,
}


def resolve_base_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def open_browser_when_ready(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(URL, timeout=1.0)
            break
        except Exception:
            time.sleep(0.5)
    try:
        webbrowser.open(URL)
    except Exception:
        pass


def _placeholder_html() -> str:
    return (
        '<div style="padding:12px 14px;border:1px solid #d9d9d9;border-radius:6px;'
        'background:#fafafa;color:#555;font-size:15px;">请上传 ONNX 模型，上传后自动预检。</div>'
    )


def _reason_line(reason: Dict[str, Any]) -> str:
    return (
        f"<div style='margin:6px 0;line-height:1.45;'>"
        f"<b>{html.escape(str(reason.get('category', '')))}</b>"
        f" · {html.escape(str(reason.get('node_name', '')))}"
        f" · {html.escape(str(reason.get('violated_field', '')))}<br>"
        f"期望：{html.escape(str(reason.get('expected', '')))}"
        f"　实际：{html.escape(str(reason.get('actual', '')))}<br>"
        f"下一步：{html.escape(str(reason.get('next_action', '')))}"
        f"</div>"
    )


def _preflight_html(result: Dict[str, Any]) -> str:
    if result.get("status") == "REJECT":
        reasons = [r for r in result.get("reasons", []) if r.get("severity", "ERROR") != "WARN"]
        body = "".join(_reason_line(reason) for reason in reasons[:5])
        if len(reasons) > 5:
            body += f"<div style='color:#666;'>另有 {len(reasons) - 5} 条原因未展开。</div>"
        return (
            '<div style="padding:14px 16px;border-left:5px solid #c62828;'
            'background:#ffebee;color:#7f1010;border-radius:6px;">'
            '<div style="font-size:18px;font-weight:700;margin-bottom:6px;">✗ 预检拒绝</div>'
            f'<div style="font-size:14px;">{body}</div></div>'
        )

    status = result.get("status", "OK")
    if "params" in result:
        summary = (
            f"参数量 {result.get('params', 0):,} · 峰值激活约 {result.get('peak_act_mb', 0)} MB "
            f"· 输入 {result.get('input_signature', {}).get('shape', '?')} "
            f"· 输出 {result.get('selected_output', '?')}"
        )
    else:
        summary = "模型已通过基础检查。"
    warn = ""
    if status == "WARN" and result.get("warnings"):
        warn = "<div style='margin-top:6px;color:#7a5a00;font-size:13px;'>资源粗估有提示，不阻塞分析。</div>"
    label = "✓ 预检通过" if status == "OK" else "✓ 预检通过（有资源提示）"
    return (
        '<div style="padding:14px 16px;border-left:5px solid #2e7d32;'
        'background:#e8f5e9;color:#174d1b;border-radius:6px;">'
        f'<div style="font-size:18px;font-weight:700;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:14px;line-height:1.5;">{html.escape(summary)}</div>'
        f'{warn}</div>'
    )


def _upload_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _upload_path(value[0]) if value else None
    if isinstance(value, (bytes, bytearray)):
        return None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if hasattr(value, "path") and not isinstance(value, (str, os.PathLike)):
        value = value.path
    if not value:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _read_upload(value: Any) -> Optional[bytes]:
    """Accept path/bytes/FileData and avoid reading a Gradio temp file mid-copy."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, (list, tuple)):
        return _read_upload(value[0]) if value else None
    path = _upload_path(value)
    if path is None:
        return None
    last = b""
    previous_size = -1
    for _ in range(12):
        try:
            size = path.stat().st_size
            data = path.read_bytes()
        except (FileNotFoundError, OSError):
            time.sleep(0.2)
            continue
        last = data
        if data and len(data) == size and size == previous_size:
            return data
        previous_size = size
        time.sleep(0.2)
    return last or None


def do_preflight(upload_value: Any):
    if upload_value is None:
        return _placeholder_html()
    try:
        data = _read_upload(upload_value)
    except Exception as exc:
        data = None
        read_error = str(exc)
    else:
        read_error = ""
    if not data:
        return (
            '<div style="padding:12px 14px;border-left:5px solid #c62828;'
            'background:#ffebee;color:#7f1010;">'
            f'读取文件失败：{html.escape(read_error or "上传文件为空或未完成，请重新上传。")}'
            '</div>'
        )

    result = preflight_onnx(data, FIXED_PROFILE, None)
    # Multi-output models: auto-use the first declared output so that the
    # simplified UI does not need an output dropdown. Preflight still validates
    # that this output is [1,21,520,520]; if not, the red error is shown.
    if result.get("status") == "REJECT" and result.get("outputs"):
        categories = {r.get("category") for r in result.get("reasons", [])}
        if "OUTPUT_SELECT" in categories:
            for output in result["outputs"]:
                if output.get("shape") == [1, 21, 520, 520]:
                    result = preflight_onnx(data, FIXED_PROFILE, output["name"])
                    break

    STATE.update({
        "onnx_path": upload_value,
        "onnx_bytes": data,
        "preflight": result,
    })
    return _preflight_html(result)


def _read_curve(csv_path: Path):
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if not row.get("miou"):
                    continue
                try:
                    rows.append({
                        "ratio": float(row["nominal_ratio"]),
                        "x": float(row["cumulative_channel_ratio"]),
                        "miou": float(row["miou"]),
                    })
                except (ValueError, KeyError):
                    continue
    return rows


def make_plot(csv_path: Path):
    rows = _read_curve(csv_path)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    if not rows:
        ax.text(0.5, 0.5, "Waiting for analysis...", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#777777")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    ax.axhline(rows[0]["miou"], color="#777777", linestyle="--", linewidth=1.0, label="baseline")
    ratios = sorted({row["ratio"] for row in rows})
    for ratio in ratios:
        series = [row for row in rows if row["ratio"] == ratio]
        ax.plot([r["x"] * 100 for r in series], [r["miou"] for r in series],
                marker="o", markersize=3, linewidth=1.6, label=f"ratio {ratio:.2f}")
    ax.set_xlabel("Cumulative channel ratio (%)")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Cumulative mIoU", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _log(lines: List[str]) -> str:
    return "\n".join(lines[-200:])


def run_analysis(ratios_text: str, subset_size: float, max_groups: float, dataset_path: str):
    preflight = STATE.get("preflight")
    if not preflight:
        yield "", make_plot(Path("__missing__.csv")), "请先上传 ONNX 模型。", None
        return
    if preflight.get("status") == "REJECT":
        yield "", make_plot(Path("__missing__.csv")), "预检未通过，不能开始分析。", None
        return

    try:
        ratios = [float(x.strip()) for x in str(ratios_text).split(",") if x.strip()]
    except ValueError:
        yield "", make_plot(Path("__missing__.csv")), "剪枝率格式错误，请用 0.1。", None
        return
    if not ratios or any(not (0.0 < r < 0.9) for r in ratios):
        yield "", make_plot(Path("__missing__.csv")), "剪枝率必须在 (0, 0.9) 内。", None
        return

    profile = load_profile(FIXED_PROFILE)
    dataset_root = resolve_base_path(dataset_path)
    if not (dataset_root / "VOC2012").exists():
        yield "", make_plot(Path("__missing__.csv")), f"找不到数据集：{dataset_root}", None
        return

    csv_path = BASE_DIR / "outputs" / "ui_curve.csv"
    baseline_path = BASE_DIR / "outputs" / "ui_baseline.json"
    explorer = Explorer(
        folded_model_bytes=preflight["folded_model_bytes"],
        profile=profile,
        input_signature=preflight["input_signature"],
        output_name=preflight["selected_output"],
        model_hash=preflight["model_hash"],
        voc_root=dataset_root,
        subset_size=int(subset_size),
        seed=FIXED_SEED,
        ratios=ratios,
        csv_path=csv_path,
        baseline_path=baseline_path,
        max_groups=int(max_groups),
    )
    lines = [
        f"模型: {preflight['model_hash'][:16]}...",
        f"评估图片: {int(subset_size)} 张 | 剪枝率: {ratios} | 最大组数: {int(max_groups)}",
    ]
    yield _log(lines), make_plot(csv_path), "正在评估基线...", None

    try:
        for event in explorer.run():
            if event["type"] == "baseline":
                b = event["baseline"]
                lines.append(
                    f"基线 mIoU={b['onnx_miou'] * 100:.2f}% | "
                    f"单张 {b['single_image_ms']:.0f} ms"
                )
                yield _log(lines), make_plot(csv_path), "基线完成，开始逐组分析...", None
            elif event["type"] == "groups":
                s = event["summary"]
                lines.append(f"共 {s['group_count']} 组可分析。")
                yield _log(lines), make_plot(csv_path), "正在逐组分析...", None
            elif event["type"] == "step":
                lines.append(
                    f"group {event['group_id']} | 剪 {event['k']} 通道 | "
                    f"mIoU={event['miou']:.2f}% | {event['delta']:+.2f} pp"
                )
                yield _log(lines), make_plot(csv_path), "正在分析...", None
            elif event["type"] == "error":
                lines.append(f"group {event['group_id']} 失败，已回滚该步：{event['message']}")
                yield _log(lines), make_plot(csv_path), "有步骤失败，继续...", None
            elif event["type"] == "done":
                lines.append(f"完成，用时 {event['elapsed_sec']:.1f}s")
                yield _log(lines), make_plot(csv_path), "分析完成。", event["csv_path"]
    except Exception as exc:
        lines.append(f"[ERROR] {type(exc).__name__}: {exc}")
        yield _log(lines), make_plot(csv_path), "分析失败。", None


HEADER_HTML = (
    '<div style="font-size:20px;color:#111;line-height:1.65;'
    'margin:8px 0 18px 0;font-weight:400;">'
    + html.escape(HEADER)
    + "</div>"
)

with gr.Blocks(title="ONNX 剪枝分析") as demo:
    gr.HTML(HEADER_HTML)

    onnx_file = gr.File(
        label="上传 ONNX 模型",
        file_types=[".onnx"],
        type="filepath",
        file_count="single",
    )
    preflight_html = gr.HTML(_placeholder_html())
    start_button = gr.Button("开始分析", variant="primary")
    status_md = gr.Markdown("")
    plot_box = gr.Plot(label="累计 mIoU 曲线")

    with gr.Accordion("高级设置", open=False):
        dataset_box = gr.Textbox(value="VOCdevkit", label="数据集目录")
        with gr.Row():
            subset_slider = gr.Slider(1, 200, value=20, step=1, label="评估图片数")
            ratios_box = gr.Textbox(value="0.1", label="剪枝率（逗号分隔）")
            max_groups_box = gr.Number(value=0, precision=0, label="最大组数（0=全部）")
        csv_file = gr.File(label="下载结果 CSV")

    with gr.Accordion("运行日志", open=False):
        log_box = gr.Textbox(label="", lines=14, max_lines=20,
                             autoscroll=True, interactive=False)

    onnx_file.upload(do_preflight, [onnx_file], [preflight_html])
    start_button.click(
        run_analysis,
        [ratios_box, subset_slider, max_groups_box, dataset_box],
        [log_box, plot_box, status_md, csv_file],
    )


if __name__ == "__main__":
    try:
        if os.environ.get("FCN_NO_BROWSER", "0") != "1":
            threading.Thread(target=open_browser_when_ready, daemon=True).start()
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            inbrowser=False,
            max_file_size=500 * 1024 * 1024,
        )
    except Exception:
        logging.exception("startup failed")
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                "启动失败，请查看程序目录下的 outputs/app.log。",
                "OnnxPruneStudio",
                0x10,
            )
        except Exception:
            pass
        raise
