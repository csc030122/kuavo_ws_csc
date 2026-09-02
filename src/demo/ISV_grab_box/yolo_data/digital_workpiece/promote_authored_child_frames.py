#!/usr/bin/env python3
"""把 Isaac Sim 中手工调整的同名子节点坐标系提升为 USD 根坐标系。

约定：
* 默认 Prim 是最终需要输出 6D 位姿的工件根节点；
* 根节点下的同名 Xform 是用户在 Isaac Sim 中调整好的目标坐标系；
* Mesh 可以带用户为对齐 XYZ 方向而设置的局部变换。

脚本将所有 Mesh 顶点和法向量转换到目标子节点坐标系，再清空后代
XformOp。默认只预检；只有指定 --apply 才覆盖源 USD。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_NAMES = ("left", "right", "hose", "outhandle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "assets",
        nargs="*",
        type=Path,
        default=[SCRIPT_DIR / f"{name}.usd" for name in ASSET_NAMES],
    )
    parser.add_argument("--apply", action="store_true", help="备份后覆盖源 USD")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="备份目录；默认在 digital_workpiece/backups 下按时间创建",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=SCRIPT_DIR.parent / "configs/canonical_frame_user_adjustments.json",
    )
    args, _ = parser.parse_known_args()
    args.assets = [path.expanduser().resolve() for path in args.assets]
    args.report = args.report.expanduser().resolve()
    if args.backup_dir is not None:
        args.backup_dir = args.backup_dir.expanduser().resolve()
    return args


def matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(vector: Any, class_name: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in vector)
    length = math.sqrt(sum(value * value for value in values))
    if length <= 1e-12:
        raise RuntimeError(f"{class_name} contains a zero-length normal")
    return tuple(value / length for value in values)


def transformed_asset(stage: Any, class_name: str) -> dict[str, Any]:
    from pxr import Gf, UsdGeom, Vt

    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError(f"{class_name} has no default Prim")
    if root.GetName() != class_name:
        raise RuntimeError(
            f"Unexpected default Prim for {class_name}: {root.GetPath()}"
        )
    if UsdGeom.Xformable(root).GetOrderedXformOps():
        raise RuntimeError(f"Root Prim must remain identity: {root.GetPath()}")

    guide = stage.GetPrimAtPath(root.GetPath().AppendChild(class_name))
    if not guide.IsValid() or not guide.IsA(UsdGeom.Xformable):
        raise RuntimeError(f"Missing authored frame guide: {guide.GetPath()}")

    cache = UsdGeom.XformCache()
    guide_from_asset = cache.GetLocalToWorldTransform(guide)
    asset_from_guide = guide_from_asset.GetInverse()
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"{class_name} expected one Mesh, found {len(meshes)}")

    baked = []
    all_points: list[tuple[float, float, float]] = []
    mesh_reports = []
    bindings = []
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        mesh_from_asset = cache.GetLocalToWorldTransform(prim)
        points = mesh.GetPointsAttr().Get() or []
        output_points = []
        for point in points:
            asset_point = mesh_from_asset.Transform(Gf.Vec3d(*point))
            guide_point = asset_from_guide.Transform(asset_point)
            values = tuple(float(value) for value in guide_point)
            all_points.append(values)
            output_points.append(Gf.Vec3f(*values))

        output_normals = []
        for normal in mesh.GetNormalsAttr().Get() or []:
            asset_normal = mesh_from_asset.TransformDir(Gf.Vec3d(*normal))
            guide_normal = asset_from_guide.TransformDir(asset_normal)
            output_normals.append(Gf.Vec3f(*normalized(guide_normal, class_name)))

        targets = [str(target) for target in prim.GetRelationship("material:binding").GetTargets()]
        if not targets:
            raise RuntimeError(f"{class_name} Mesh lost its material binding: {prim.GetPath()}")
        bindings.extend(targets)
        baked.append((mesh, Vt.Vec3fArray(output_points), Vt.Vec3fArray(output_normals)))
        mesh_reports.append(
            {
                "path": str(prim.GetPath()),
                "asset_from_mesh_matrix": matrix_rows(mesh_from_asset),
                "material_bindings": targets,
            }
        )

    minimum = [min(point[index] for point in all_points) for index in range(3)]
    maximum = [max(point[index] for point in all_points) for index in range(3)]
    center = [(minimum[index] + maximum[index]) * 0.5 for index in range(3)]
    size = [maximum[index] - minimum[index] for index in range(3)]
    return {
        "root": root,
        "guide": guide,
        "baked": baked,
        "center_units": center,
        "size_units": size,
        "guide_from_asset_matrix": matrix_rows(guide_from_asset),
        "meshes": mesh_reports,
        "material_bindings": bindings,
    }


def apply_transformation(stage: Any, report: dict[str, Any]) -> None:
    from pxr import Gf, Sdf, UsdGeom, Vt

    print(f"APPLY_BEGIN class={report['class_name']}", flush=True)
    for mesh, points, normals in report.pop("baked"):
        mesh.GetPointsAttr().Set(points)
        if len(normals):
            mesh.GetNormalsAttr().Set(normals)
        minimum = [min(float(point[index]) for point in points) for index in range(3)]
        maximum = [max(float(point[index]) for point in points) for index in range(3)]
        mesh.CreateExtentAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*minimum), Gf.Vec3f(*maximum)])
        )
        print(f"APPLY_MESH class={report['class_name']} path={mesh.GetPath()}", flush=True)

    root = report.pop("root")
    report.pop("guide")
    xformables = []
    for prim in stage.Traverse():
        if prim == root or not prim.GetPath().HasPrefix(root.GetPath()):
            continue
        if prim.IsA(UsdGeom.Xformable):
            xformables.append(prim)
    for prim in xformables:
        print(f"CLEAR_XFORM_BEGIN class={report['class_name']} path={prim.GetPath()}", flush=True)
        UsdGeom.Xformable(prim).ClearXformOpOrder()
        print(f"CLEAR_XFORM_DONE class={report['class_name']} path={prim.GetPath()}", flush=True)

    print(f"SET_METADATA_BEGIN class={report['class_name']}", flush=True)
    root.CreateAttribute(
        "kuavo:canonicalFrameVersion", Sdf.ValueTypeNames.String, custom=True
    ).Set("canonical_v2_user_adjusted")
    print(f"SET_METADATA_VERSION class={report['class_name']}", flush=True)
    root.CreateAttribute("kuavo:canonicalOrigin", Sdf.ValueTypeNames.String, custom=True).Set(
        "user_authored_child_frame"
    )
    print(f"SET_METADATA_ORIGIN class={report['class_name']}", flush=True)
    root.CreateAttribute(
        "kuavo:userAuthoredFrame",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(
        json.dumps(
            {
                "guide_path": str(root.GetPath().AppendChild(root.GetName())),
                "guide_from_asset_matrix": report["guide_from_asset_matrix"],
                "meshes": report["meshes"],
            },
            ensure_ascii=False,
        ),
    )
    print(f"SET_METADATA_FRAME class={report['class_name']}", flush=True)
    root.CreateAttribute(
        "kuavo:canonicalExpectedSizeUnits", Sdf.ValueTypeNames.Double3, custom=True
    ).Set(Gf.Vec3d(*report["size_units"]))
    print(f"SET_METADATA_SIZE class={report['class_name']}", flush=True)
    root.CreateAttribute(
        "kuavo:canonicalExpectedCenterUnits", Sdf.ValueTypeNames.Double3, custom=True
    ).Set(Gf.Vec3d(*report["center_units"]))
    print(f"APPLY_READY_TO_SAVE class={report['class_name']}", flush=True)


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "fast_shutdown": True})
    try:
        from pxr import Usd

        reports = []
        stages = []
        for path in args.assets:
            if not path.is_file():
                raise FileNotFoundError(path)
            class_name = path.stem
            if class_name not in ASSET_NAMES:
                raise RuntimeError(f"Unsupported asset name: {path}")
            stage = Usd.Stage.Open(str(path))
            if stage is None:
                raise RuntimeError(f"Could not open USD: {path}")
            report = transformed_asset(stage, class_name)
            report.update(
                {
                    "class_name": class_name,
                    "path": str(path),
                    "source_sha256": file_sha256(path),
                }
            )
            stages.append((path, stage, report))
            reports.append(report)
            print(
                f"FRAME_PREVIEW class={class_name} center={tuple(round(v, 6) for v in report['center_units'])} "
                f"size={tuple(round(v, 6) for v in report['size_units'])} "
                f"materials={report['material_bindings']}",
                flush=True,
            )

        if not args.apply:
            print("DRY_RUN=1", flush=True)
            return 0

        backup_dir = args.backup_dir
        if backup_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = SCRIPT_DIR / "backups" / f"user_adjusted_before_promote_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path, _, _ in stages:
            shutil.copy2(path, backup_dir / path.name)

        serializable_reports = []
        for path, stage, report in stages:
            apply_transformation(stage, report)
            print(f"SAVE_BEGIN class={path.stem}", flush=True)
            stage.GetRootLayer().Save()
            print(f"SAVE_DONE class={path.stem}", flush=True)
            report["result_sha256"] = file_sha256(path)
            serializable_reports.append(report)
            print(f"FRAME_PROMOTED class={path.stem} path={path}", flush=True)

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "coordinate_units": "asset units (millimetres)",
                    "backup_dir": str(backup_dir),
                    "assets": serializable_reports,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"BACKUP_DIR={backup_dir}", flush=True)
        print(f"REPORT={args.report}", flush=True)
        print(f"PROMOTED_ASSETS={len(serializable_reports)}", flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
