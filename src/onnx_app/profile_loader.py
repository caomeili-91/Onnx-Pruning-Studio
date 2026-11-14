"""Target profile loading and validation (data-driven JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
REQUIRED_FIELDS = {
    "schema_version",
    "profile_version",
    "profile_name",
    "severity_default",
    "ir_version_range",
    "opset_range",
    "dtype",
    "input_layout",
    "channel_round_to",
    "hw_align",
    "ops",
    "structural_ops",
    "resource_limits",
    "resource_formula",
}


class ProfileError(ValueError):
    pass


def list_profile_files() -> List[Path]:
    return sorted(PROFILE_DIR.glob("*.json"))


def list_profiles() -> List[str]:
    return [path.stem for path in list_profile_files()]


def load_profile(name: str) -> Dict[str, Any]:
    path = PROFILE_DIR / f"{name}.json"
    if not path.exists():
        raise ProfileError(f"未知 target profile: {name}")
    with open(path, encoding="utf-8-sig") as file:
        profile = json.load(file)
    missing = REQUIRED_FIELDS - set(profile)
    if missing:
        raise ProfileError(f"profile {name} 缺少字段: {sorted(missing)}")
    if not isinstance(profile["ops"], dict) or not profile["ops"]:
        raise ProfileError(f"profile {name} 的 ops 必须是非空对象")
    if not isinstance(profile["resource_limits"], dict):
        raise ProfileError(f"profile {name} 的 resource_limits 必须是对象")
    return profile
