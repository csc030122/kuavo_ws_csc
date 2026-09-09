"""Small Isaac/Replicator helpers shared by the active side-camera pipeline."""

from __future__ import annotations

import json
import math
import subprocess
from typing import Any

from yolo_catchdata.camera_geometry import (
    apply_opencv_roll_to_usd_camera,
    look_at_world_from_usd_camera,
)


def gpu_info() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = []
        for line in completed.stdout.strip().splitlines():
            name, driver, memory = (part.strip() for part in line.split(",", maxsplit=2))
            rows.append(
                {"name": name, "driver_version": driver, "memory_total_mib": int(memory)}
            )
        return {"available": bool(rows), "gpus": rows}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "gpus": [], "error": str(exc)}


def _as_list(matrix: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def gf_matrix_from_column_transform(transform: Any) -> Any:
    from pxr import Gf

    matrix = _as_list(transform)
    transposed = [[matrix[column][row] for column in range(4)] for row in range(4)]
    return Gf.Matrix4d(*[value for row in transposed for value in row])


def set_transform(prim: Any, transform: Any) -> None:
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        gf_matrix_from_column_transform(transform)
    )


def add_semantics(prim: Any, class_name: str) -> None:
    from pxr import Semantics

    semantic = Semantics.SemanticsAPI.Apply(prim, f"class_{class_name}")
    semantic.CreateSemanticTypeAttr().Set("class")
    semantic.CreateSemanticDataAttr().Set(class_name)


def extract_array(data: Any) -> Any:
    return data["data"] if isinstance(data, dict) and "data" in data else data


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def target_instance_ids(
    instance_output: dict[str, Any], class_name: str, prim_path: str
) -> list[int]:
    info = instance_output.get("info", {})
    semantics = info.get("idToSemantics", {})
    labels = info.get("idToLabels", {})
    result = []
    for raw_identifier in set(semantics) | set(labels):
        semantic = semantics.get(raw_identifier, semantics.get(str(raw_identifier), {}))
        label = labels.get(raw_identifier, labels.get(str(raw_identifier), ""))
        if class_name in json.dumps(json_safe(semantic), ensure_ascii=False) or prim_path in str(label):
            result.append(int(raw_identifier))
    return sorted(set(result))


def bbox_from_mask(mask: Any) -> list[int] | None:
    import numpy as np

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        return None
    rows, columns = np.nonzero(array)
    if not len(rows):
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]


def preview_camera_pose(
    sampling_center_world: list[float],
    rotation_world_from_sampling: list[list[float]],
    radius_m: float,
    theta_deg: float,
    phi_deg: float,
    psi_deg: float,
) -> tuple[list[float], tuple[tuple[float, ...], ...]]:
    """Place the external stable-pose preview camera; never used for collection."""

    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    offset_sampling = (
        radius_m * math.cos(phi) * math.cos(theta),
        radius_m * math.cos(phi) * math.sin(theta),
        radius_m * math.sin(phi),
    )
    offset_world = [
        sum(
            rotation_world_from_sampling[row][column] * offset_sampling[column]
            for column in range(3)
        )
        for row in range(3)
    ]
    position = [sampling_center_world[index] + offset_world[index] for index in range(3)]
    pose = look_at_world_from_usd_camera(position, sampling_center_world)
    return position, apply_opencv_roll_to_usd_camera(pose, psi_deg)


__all__ = [
    "add_semantics",
    "bbox_from_mask",
    "extract_array",
    "gf_matrix_from_column_transform",
    "gpu_info",
    "json_safe",
    "preview_camera_pose",
    "set_transform",
    "target_instance_ids",
]
