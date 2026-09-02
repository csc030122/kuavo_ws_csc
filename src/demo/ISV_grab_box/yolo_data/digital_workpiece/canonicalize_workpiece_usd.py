#!/usr/bin/env python3
"""把四个工件 USD 的几何直接烘焙到 canonical_v1 根坐标系。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FRAME_DEFINITIONS = {
    "left": {
        "origin_old_root_units": (-459.8818359375, 54.12745551891794, 83.76587677001953),
        "old_root_from_canonical": (
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        "expected_size_units": (426.08935546875, 219.766357421875, 69.81707763671875),
    },
    "right": {
        "origin_old_root_units": (-492.6775360107422, 2594.5919189453125, 88.4495964050293),
        "old_root_from_canonical": (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        "expected_size_units": (421.8275451660156, 215.63381958007812, 99.726318359375),
    },
    "hose": {
        "origin_old_root_units": (-344.6284637451172, 530.0250221342717, 294.67606353759766),
        "old_root_from_canonical": (
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        "expected_size_units": (330.6038818359375, 151.112060546875, 74.70498657226562),
    },
    "outhandle": {
        "origin_old_root_units": (-305.74671936035156, 2365.8248291015625, 178.73741912841797),
        "old_root_from_canonical": (
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
        "expected_size_units": (72.441650390625, 66.93617248535156, 43.789947509765625),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="只打印根节点与子节点变换，不修改或保存 USD。",
    )
    parser.add_argument(
        "assets",
        nargs="*",
        type=Path,
        default=[SCRIPT_DIR / f"{name}.usd" for name in FRAME_DEFINITIONS],
        help="待规范化的 USD；文件名必须是 left/right/hose/outhandle.usd。",
    )
    args, _ = parser.parse_known_args()
    args.assets = [path.expanduser().resolve() for path in args.assets]
    return args


def _canonical_components(vector: Any, axes: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    """用 canonical 轴在旧根坐标系中的列向量投影一个向量。"""

    values = tuple(float(value) for value in vector)
    return tuple(
        sum(axes[row][column] * values[row] for row in range(3))
        for column in range(3)
    )


def _bounds(stage: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    from pxr import Usd, UsdGeom

    value_range = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    ).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    center = tuple(float(value) for value in value_range.GetMidpoint())
    size = tuple(float(value) for value in value_range.GetSize())
    return center, size


def validate_canonical_stage(stage: Any, class_name: str) -> dict[str, Any]:
    from pxr import UsdGeom

    definition = FRAME_DEFINITIONS[class_name]
    root = stage.GetDefaultPrim()
    center, size = _bounds(stage)
    version_attr = root.GetAttribute("kuavo:canonicalFrameVersion")
    version = (
        version_attr.Get()
        if version_attr.IsValid() and version_attr.HasAuthoredValueOpinion()
        else root.GetCustomDataByKey("kuavo:canonicalFrameVersion")
    )
    if version == "canonical_v2_user_adjusted":
        center_attr = root.GetAttribute("kuavo:canonicalExpectedCenterUnits")
        size_attr = root.GetAttribute("kuavo:canonicalExpectedSizeUnits")
        expected_center = tuple(
            float(value)
            for value in center_attr.Get()
        )
        expected = tuple(
            float(value)
            for value in size_attr.Get()
        )
    elif version == "canonical_v1":
        expected_center = (0.0, 0.0, 0.0)
        expected = definition["expected_size_units"]
    else:
        raise RuntimeError(f"{class_name} has unsupported canonical version: {version}")
    if any(
        not math.isclose(actual, target, rel_tol=0.0, abs_tol=2e-3)
        for actual, target in zip(center, expected_center)
    ):
        raise RuntimeError(
            f"{class_name} canonical center mismatch: actual={center}, expected={expected_center}"
        )
    if any(
        not math.isclose(actual, target, rel_tol=3e-5, abs_tol=3e-3)
        for actual, target in zip(size, expected)
    ):
        raise RuntimeError(
            f"{class_name} canonical size mismatch: actual={size}, expected={expected}"
        )
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError(f"{class_name} contains no Mesh")
    bindings = []
    for prim in meshes:
        targets = prim.GetRelationship("material:binding").GetTargets()
        if not targets:
            raise RuntimeError(f"{class_name} mesh lost its material binding: {prim.GetPath()}")
        bindings.extend(str(target) for target in targets)
    return {
        "class_name": class_name,
        "center_units": center,
        "size_units": size,
        "mesh_count": len(meshes),
        "material_bindings": bindings,
    }


def canonicalize_stage(stage: Any, class_name: str) -> dict[str, Any]:
    """原位规范化一个已打开的资产 Stage；调用者决定何时保存。"""

    from pxr import Gf, UsdGeom, Vt

    if class_name not in FRAME_DEFINITIONS:
        raise ValueError(f"Unsupported workpiece asset: {class_name}")
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError(f"{class_name} has no default Prim")
    version_attr = root.GetAttribute("kuavo:canonicalFrameVersion")
    version = (
        version_attr.Get()
        if version_attr.IsValid() and version_attr.HasAuthoredValueOpinion()
        else root.GetCustomDataByKey("kuavo:canonicalFrameVersion")
    )
    if version in {
        "canonical_v1",
        "canonical_v2_user_adjusted",
    }:
        return validate_canonical_stage(stage, class_name)
    if UsdGeom.Xformable(root).GetOrderedXformOps():
        raise RuntimeError(f"{class_name} default Prim unexpectedly has authored xform ops")

    definition = FRAME_DEFINITIONS[class_name]
    origin = definition["origin_old_root_units"]
    axes = definition["old_root_from_canonical"]
    cache = UsdGeom.XformCache()
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError(f"{class_name} contains no Mesh")

    baked: list[tuple[Any, Any, Any]] = []
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        transform = cache.GetLocalToWorldTransform(prim)
        points = mesh.GetPointsAttr().Get() or []
        canonical_points = []
        for point in points:
            old_root_point = transform.Transform(Gf.Vec3d(*point))
            delta = tuple(float(old_root_point[index]) - origin[index] for index in range(3))
            canonical_points.append(Gf.Vec3f(*_canonical_components(delta, axes)))

        normals = mesh.GetNormalsAttr().Get() or []
        canonical_normals = []
        for normal in normals:
            old_root_normal = transform.TransformDir(Gf.Vec3d(*normal))
            values = _canonical_components(old_root_normal, axes)
            length = math.sqrt(sum(value * value for value in values))
            if length <= 1e-12:
                raise RuntimeError(f"{class_name} contains a zero-length normal")
            canonical_normals.append(Gf.Vec3f(*(value / length for value in values)))
        baked.append(
            (
                mesh,
                Vt.Vec3fArray(canonical_points),
                Vt.Vec3fArray(canonical_normals),
            )
        )

    # 所有世界变换已在清理层级 xform 之前读取并烘焙到 Mesh 数据。
    for mesh, points, normals in baked:
        mesh.GetPointsAttr().Set(points)
        if len(normals):
            mesh.GetNormalsAttr().Set(normals)
        minimum = [min(float(point[index]) for point in points) for index in range(3)]
        maximum = [max(float(point[index]) for point in points) for index in range(3)]
        mesh.CreateExtentAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*minimum), Gf.Vec3f(*maximum)])
        )
    for prim in stage.Traverse():
        if prim == root or not prim.GetPath().HasPrefix(root.GetPath()):
            continue
        if prim.IsA(UsdGeom.Xformable):
            UsdGeom.Xformable(prim).ClearXformOpOrder()

    root.SetCustomDataByKey("kuavo:canonicalFrameVersion", "canonical_v1")
    root.SetCustomDataByKey("kuavo:canonicalOrigin", "asset_local_aabb_center")
    root.SetCustomDataByKey(
        "kuavo:sourceFrameTransform",
        json.dumps(
            {
                "origin_old_root_units": origin,
                "old_root_from_canonical": axes,
            },
            ensure_ascii=False,
        ),
    )
    return validate_canonical_stage(stage, class_name)


def canonicalize_path(path: Path) -> dict[str, Any]:
    from pxr import Usd

    class_name = path.stem
    if class_name not in FRAME_DEFINITIONS:
        raise ValueError(f"USD filename must identify a known workpiece: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {path}")
    report = canonicalize_stage(stage, class_name)
    stage.GetRootLayer().Save()
    report["path"] = str(path)
    return report


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "fast_shutdown": True})
    try:
        if args.inspect_only:
            from pxr import Usd, UsdGeom

            for path in args.assets:
                stage = Usd.Stage.Open(str(path))
                if stage is None:
                    raise RuntimeError(f"Could not open USD: {path}")
                root = stage.GetDefaultPrim()
                print(f"ASSET path={path} default_prim={root.GetPath()}", flush=True)
                for prim in stage.Traverse():
                    if not prim.GetPath().HasPrefix(root.GetPath()) or not prim.IsA(
                        UsdGeom.Xformable
                    ):
                        continue
                    xformable = UsdGeom.Xformable(prim)
                    ops = xformable.GetOrderedXformOps()
                    if not ops and prim != root:
                        continue
                    values = [(str(op.GetOpName()), str(op.Get())) for op in ops]
                    print(
                        f"XFORM path={prim.GetPath()} "
                        f"reset={int(xformable.GetResetXformStack())} ops={values}",
                        flush=True,
                    )
            print("INSPECT_ONLY=1", flush=True)
            return 0
        reports = []
        for path in args.assets:
            if not path.is_file():
                raise FileNotFoundError(path)
            report = canonicalize_path(path)
            reports.append(report)
            print(
                f"CANONICALIZED class={report['class_name']} "
                f"center={report['center_units']} size={report['size_units']} "
                f"materials={report['material_bindings']} path={path}",
                flush=True,
            )
        print(f"CANONICALIZED_ASSETS={len(reports)}", flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
