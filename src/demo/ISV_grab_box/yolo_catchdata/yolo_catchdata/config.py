"""Configuration loading and Phase-0 validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    data["_config_path"] = str(config_path)
    return data


def resolve_from_config(config: dict[str, Any], value: str) -> Path:
    config_path = Path(config["_config_path"])
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def validate_assets_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != 1:
        errors.append("assets schema_version must be 1")
    classes = config.get("classes")
    if not isinstance(classes, list) or not classes:
        return errors + ["classes must be a non-empty list"]

    ids = [item.get("class_id") for item in classes]
    names = [item.get("name") for item in classes]
    if len(ids) != len(set(ids)):
        errors.append("class_id values must be unique")
    if len(names) != len(set(names)):
        errors.append("class names must be unique")
    if sorted(ids) != list(range(len(ids))):
        errors.append("class_id values must be contiguous and start at zero")

    for item in classes:
        name = item.get("name", "<unknown>")
        path = item.get("usd")
        if not isinstance(path, str):
            errors.append(f"{name}: usd must be a path string")
        elif not resolve_from_config(config, path).is_file():
            errors.append(f"{name}: USD does not exist: {resolve_from_config(config, path)}")
        matrix = item.get("asset_from_canonical")
        if not (
            isinstance(matrix, list)
            and len(matrix) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in matrix)
        ):
            errors.append(f"{name}: asset_from_canonical must be 4x4")
    return errors

