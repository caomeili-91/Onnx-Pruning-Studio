r"""PyInstaller 构建入口：绕过 Python 3.10.0 的 dis 缺陷并补运行时数据。

用法：
    python build.py
    python build.py --noconfirm --clean build.spec
"""

from __future__ import annotations

import dis
import runpy
import shutil
import sys
from pathlib import Path

_orig_get_const_info = dis._get_const_info


def _safe_get_const_info(arg, const_list):
    if arg >= len(const_list):
        return None, f"<invalid const {arg}>"
    return _orig_get_const_info(arg, const_list)


dis._get_const_info = _safe_get_const_info


def _dist_root(args) -> Path:
    root = Path("dist")
    for index, arg in enumerate(args):
        if arg == "--distpath" and index + 1 < len(args):
            root = Path(args[index + 1])
        elif arg.startswith("--distpath="):
            root = Path(arg.split("=", 1)[1])
    return root


def _postprocess(dist_root: Path) -> None:
    dist = dist_root / "OnnxPruneStudio"
    if not (dist / "OnnxPruneStudio.exe").exists():
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"

    dll_count = 0
    for libs_dir in site_packages.glob("*.libs"):
        for dll in libs_dir.glob("*.dll"):
            shutil.copy2(dll, dist / dll.name)
            dll_count += 1

    for package in ("safehttpx", "groovy"):
        source = site_packages / package
        if source.is_dir():
            for name in ("version.txt",):
                file = source / name
                if file.exists():
                    target = dist / package / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, target)

    gradio_src = site_packages / "gradio"
    source_count = 0
    if gradio_src.is_dir():
        for py_file in gradio_src.rglob("*.py"):
            rel = py_file.relative_to(gradio_src).with_suffix(".pyc")
            target = dist / "gradio" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, target)
            source_count += 1

    print(f"[postprocess] copied {dll_count} .libs DLLs, {source_count} gradio source files")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        args = ["--noconfirm", "--clean", "build.spec"]
    dist_root = _dist_root(args)
    sys.argv = ["pyinstaller", *args]
    try:
        runpy.run_module("PyInstaller", run_name="__main__")
    except SystemExit:
        _postprocess(dist_root)
        raise
    else:
        _postprocess(dist_root)


if __name__ == "__main__":
    main()
