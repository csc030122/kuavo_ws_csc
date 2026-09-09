#!/usr/bin/env python3
"""检查工件 USD 的层级、组合依赖以及 resetXformStack。"""

import json
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True, "fast_shutdown": True})


def main() -> int:
    from pxr import Gf, Sdf, Usd, UsdGeom

    path = Path(__file__).resolve().parents[1] / "yolo_scene/outhandle.usd"
    output_path = (
        Path(__file__).resolve().parents[1]
        / "output/manual_wrist/outhandle_hierarchy_report.json"
    )
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise RuntimeError(f"无法打开 layer：{path}")
    report = {
        "path": str(path),
        "layer": {
            "identifier": layer.identifier,
            "real_path": layer.realPath,
            "default_prim": layer.defaultPrim,
            "sub_layers": list(layer.subLayerPaths),
            "root_prims": [prim.name for prim in layer.rootPrims],
        },
        "prims": [],
    }
    stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"无法打开 {path}")
    print(f"STAGE={path}", flush=True)
    print(f"LAYER_DEFAULT_PRIM={layer.defaultPrim}", flush=True)
    print(f"LAYER_ROOT_PRIMS={[prim.name for prim in layer.rootPrims]}", flush=True)
    print(f"LAYER_SUBLAYERS={list(layer.subLayerPaths)}", flush=True)
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        xformable = UsdGeom.Xformable(prim)
        reset = xformable.GetResetXformStack()
        ops = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
        item = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "active": prim.IsActive(),
            "loaded": prim.IsLoaded(),
            "instance": prim.IsInstance(),
            "instance_proxy": prim.IsInstanceProxy(),
            "reset_xform_stack": bool(reset),
            "ordered_xform_ops": ops,
            "applied_schemas": list(prim.GetAppliedSchemas()),
            "properties": [prop.GetName() for prop in prim.GetProperties()],
            "xform_attributes": {
                attr.GetName(): {
                    "value": str(attr.Get()),
                    "time_samples": list(attr.GetTimeSamples()),
                }
                for attr in prim.GetAttributes()
                if attr.GetName().startswith("xformOp:")
            },
        }
        report["prims"].append(item)
        if reset or ops or prim == stage.GetDefaultPrim():
            print(
                f"PRIM={prim.GetPath()} TYPE={prim.GetTypeName()} "
                f"RESET={bool(reset)} OPS={ops}",
                flush=True,
            )
    root = stage.GetDefaultPrim()
    report["stage_default_prim"] = str(root.GetPath()) if root and root.IsValid() else None
    if root and root.IsValid():
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
        ).ComputeWorldBound(root).ComputeAlignedRange()
        report["bbox"] = {
            "min": list(bbox.GetMin()),
            "max": list(bbox.GetMax()),
        }

    # 最小组合实验：在米制匿名场景中给工件套一层世界位姿，再在
    # Asset 节点上施加 mm -> m 缩放。将 BBoxCache 与直接变换网格点对比，
    # 用于区分资产问题和运行时缓存/变换读取问题。
    composed = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(composed, 1.0)
    wrapper = UsdGeom.Xform.Define(composed, "/Test").GetPrim()
    wrapper_matrix = Gf.Matrix4d(1.0)
    wrapper_matrix.SetTranslateOnly(Gf.Vec3d(1.5959904194, 0.0589537096, 1.2493275309))
    UsdGeom.Xformable(wrapper).AddTransformOp().Set(wrapper_matrix)
    asset = UsdGeom.Xform.Define(composed, "/Test/Asset").GetPrim()
    asset.GetReferences().AddReference(str(path))
    UsdGeom.Xformable(asset).AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))

    composed_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    composed_range = composed_cache.ComputeWorldBound(asset).ComputeAlignedRange()
    direct_points = []
    for composed_prim in composed.Traverse():
        if not composed_prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(composed_prim)
        matrix = UsdGeom.Xformable(composed_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        direct_points.extend(matrix.Transform(Gf.Vec3d(point)) for point in mesh.GetPointsAttr().Get())
    if direct_points:
        direct_min = [min(float(point[axis]) for point in direct_points) for axis in range(3)]
        direct_max = [max(float(point[axis]) for point in direct_points) for axis in range(3)]
    else:
        direct_min = direct_max = None
    asset_world = UsdGeom.Xformable(asset).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    report["minimal_composition_check"] = {
        "asset_world_matrix_row_major": [
            [float(asset_world[row][column]) for column in range(4)]
            for row in range(4)
        ],
        "bbox_min": list(composed_range.GetMin()),
        "bbox_max": list(composed_range.GetMax()),
        "direct_vertex_min": direct_min,
        "direct_vertex_max": direct_max,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DEFAULT_PRIM={report['stage_default_prim']}", flush=True)
    print(f"PRIM_COUNT={len(report['prims'])}", flush=True)
    print(f"REPORT={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
