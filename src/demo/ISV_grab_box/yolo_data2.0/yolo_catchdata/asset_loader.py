"""Read-only USD inspection used by Phase 1.

The pxr imports intentionally live inside functions: this module can be imported
by ordinary Python tests, while USD inspection itself runs with Isaac Sim Python.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def unit_scale_for_scene(asset_meters_per_unit: float, scene_meters_per_unit: float) -> float:
    """Scale authored asset units into the target stage's authored units."""

    if asset_meters_per_unit <= 0.0 or scene_meters_per_unit <= 0.0:
        raise ValueError("metersPerUnit values must be positive")
    return float(asset_meters_per_unit) / float(scene_meters_per_unit)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vec(values: Any) -> list[float]:
    return [float(value) for value in values]


def _attribute_value(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    if not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
        return None
    value = attribute.Get()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return _vec(value)
    except TypeError:
        return str(value)


def inspect_usd(path: Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {path}")
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError(f"USD has no default prim: {path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    ).ComputeWorldBound(root).ComputeAlignedRange()
    minimum_units = _vec(bbox.GetMin())
    maximum_units = _vec(bbox.GetMax())
    center_units = _vec(bbox.GetMidpoint())
    size_units = _vec(bbox.GetSize())
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    xform_ops = []
    if root.IsA(UsdGeom.Xformable):
        xform_ops = [op.GetOpName() for op in UsdGeom.Xformable(root).GetOrderedXformOps()]

    canonical_keys = (
        "kuavo:canonicalFrameVersion",
        "kuavo:canonicalOrigin",
        "kuavo:canonicalExpectedCenterUnits",
        "kuavo:canonicalExpectedSizeUnits",
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "default_prim": str(root.GetPath()),
        "meters_per_unit": meters_per_unit,
        "up_axis": up_axis,
        "root_xform_ops": xform_ops,
        "bounds_units": {
            "min": minimum_units,
            "max": maximum_units,
            "center": center_units,
            "size": size_units,
        },
        "bounds_m": {
            "min": [value * meters_per_unit for value in minimum_units],
            "max": [value * meters_per_unit for value in maximum_units],
            "center": [value * meters_per_unit for value in center_units],
            "size": [value * meters_per_unit for value in size_units],
        },
        "mesh_count": len(meshes),
        "mesh_paths": [str(prim.GetPath()) for prim in meshes],
        "canonical_metadata": {key: _attribute_value(root, key) for key in canonical_keys},
    }


def inspect_prim(path: Path, prim_path: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {path}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"USD prim does not exist: {path}:{prim_path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    ).ComputeWorldBound(prim).ComputeAlignedRange()
    minimum_units = _vec(bbox.GetMin())
    maximum_units = _vec(bbox.GetMax())
    size_units = _vec(bbox.GetSize())
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return {
        "path": prim_path,
        "name": prim.GetName(),
        "type_name": prim.GetTypeName(),
        "world_transform": [
            [float(transform[row][column]) for column in range(4)] for row in range(4)
        ],
        "world_bounds_m": {
            "min": [value * meters_per_unit for value in minimum_units],
            "max": [value * meters_per_unit for value in maximum_units],
            "size": [value * meters_per_unit for value in size_units],
        },
    }


def scene_name_candidates(path: Path, terms: tuple[str, ...] = ("table", "box")) -> list[str]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {path}")
    lowered = tuple(term.lower() for term in terms)
    return [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if any(term in prim.GetName().lower() for term in lowered)
    ]


def add_reference_with_unit_scale(
    stage: Any,
    asset_path: Path,
    prim_path: str,
    asset_meters_per_unit: float,
    scene_meters_per_unit: float,
) -> Any:
    """Reference an asset and author unit conversion on the runtime instance root.

    This function only authors the supplied stage/layer. Source asset USD files
    remain untouched.
    """

    from pxr import Gf, UsdGeom

    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    prim.GetReferences().AddReference(str(asset_path))
    scale = unit_scale_for_scene(asset_meters_per_unit, scene_meters_per_unit)
    xformable = UsdGeom.Xformable(prim)
    scale_op = next(
        (
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeScale
        ),
        None,
    )
    if scale_op is None:
        scale_op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(scale, scale, scale))
    return prim
