#!/usr/bin/env python3
"""定位真实右腕相机视野被哪个 USD Prim 遮挡。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_radius import (  # noqa: E402
    _add_semantics,
    _extract_array,
    _set_transform,
    _target_instance_ids,
)
from scripts.render_wrist_radius_extremes import (  # noqa: E402
    RIGHT_ARM_LINKS,
    RIGHT_WRIST_RGB,
    _camera_pose_data,
    _camera_pose_with_hand_roll_zero,
    _hide_detached_upstream_arm,
    _make_panel,
    _place_rigid_wrist,
    _prepare_rigid_wrist,
)
from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera.yaml")
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    parser.add_argument("--stable-poses", type=Path, default=CONFIG_DIR / "stable_poses.json")
    parser.add_argument("--class-name", choices=("left", "right", "hose", "outhandle"), default="left")
    parser.add_argument("--radius-key", choices=("r_max", "r_min"), default="r_max")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output/workspace_calibration/wrist_occlusion_diagnostic",
    )
    return parser.parse_args()


def _aligned_world_range(stage: Any, prim_path: str) -> tuple[list[float], list[float]] | None:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    ).ComputeWorldBound(prim).ComputeAlignedBox()
    if bbox.IsEmpty():
        return None
    return [float(value) for value in bbox.GetMin()], [float(value) for value in bbox.GetMax()]


def _point_aabb_relation(point: Any, bounds: tuple[list[float], list[float]] | None) -> dict[str, Any]:
    import numpy as np

    if bounds is None:
        return {"bounds_world_m": None, "camera_inside_aabb": False, "distance_to_aabb_m": None}
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    point = np.asarray(point, dtype=float)
    outside = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    return {
        "bounds_world_m": [lower.tolist(), upper.tolist()],
        "camera_inside_aabb": bool(np.all(point >= lower) and np.all(point <= upper)),
        "distance_to_aabb_m": float(np.linalg.norm(outside)),
    }


def _renderable_descendants(stage: Any, root_path: str) -> list[str]:
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    return [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.IsValid() and prim.IsA(UsdGeom.Gprim)
    ]


def _group_by_first_child(root_path: str, paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    prefix = root_path.rstrip("/") + "/"
    for path in paths:
        relative = path[len(prefix):] if path.startswith(prefix) else path
        first = relative.split("/", 1)[0]
        groups.setdefault(first, []).append(path)
    return groups


@contextmanager
def _temporarily_hidden(stage: Any, prim_paths: list[str]):
    from pxr import UsdGeom

    states = []
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            continue
        attribute = UsdGeom.Imageable(prim).GetVisibilityAttr()
        states.append((attribute, attribute.Get()))
        attribute.Set(UsdGeom.Tokens.invisible)
    try:
        yield
    finally:
        for attribute, original in states:
            attribute.Set(original or UsdGeom.Tokens.inherited)


def _make_sheet(items: list[tuple[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    columns = 2
    panel_size = (640, 360)
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel_size[0], rows * panel_size[1]), "white")
    for index, (_, panel) in enumerate(items):
        sheet.paste(
            panel.resize(panel_size, Image.Resampling.LANCZOS),
            ((index % columns) * panel_size[0], (index // columns) * panel_size[1]),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    args = parse_args()
    assets = load_yaml(args.assets_config)
    camera_config = load_yaml(args.camera_config)["camera"]
    collection = load_yaml(args.collection_config)
    workspace = collection["workspace_calibration"]
    stable = json.loads(args.stable_poses.read_text(encoding="utf-8"))
    scene_path = resolve_from_config(assets, assets["scene"]["usd"])
    radius = float(workspace["radius_range_m"]["max" if args.radius_key == "r_max" else "min"])
    viewpoint = workspace["radius_min_selection"]["baseline_viewpoint"]

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": True,
            "multi_gpu": False,
            "renderer": "RaytracedLighting",
            "width": int(camera_config["resolution"][0]),
            "height": int(camera_config["resolution"][1]),
        }
    )
    annotators = []
    product = None
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import UsdGeom

        from yolo_catchdata.asset_loader import add_reference_with_unit_scale, inspect_usd

        context = omni.usd.get_context()
        if not context.open_stage(str(scene_path)):
            raise RuntimeError(f"无法打开场景：{scene_path}")
        for _ in range(20):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        omni.timeline.get_timeline_interface().stop()

        rig = _prepare_rigid_wrist(stage, scene_path)
        hidden_upstream = _hide_detached_upstream_arm(stage)
        definition = next(item for item in assets["classes"] if item["name"] == args.class_name)
        asset_path = resolve_from_config(assets, definition["usd"])
        inspection = inspect_usd(asset_path)
        root_path = f"/World/SyntheticData/WristOcclusionDiagnostic/{args.class_name}"
        root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        _set_transform(root, stable["classes"][args.class_name]["pose"]["settled_world_from_asset"])
        asset_prim_path = f"{root_path}/Asset"
        asset_prim = add_reference_with_unit_scale(
            stage,
            asset_path,
            asset_prim_path,
            inspection["meters_per_unit"],
            float(UsdGeom.GetStageMetersPerUnit(stage)),
        )
        _add_semantics(asset_prim, args.class_name)
        center = np.asarray(
            stable["classes"][args.class_name]["pose"]["sampling_center_world_m"], dtype=float
        )
        _, world_from_camera, camera_psi, finger_direction = _camera_pose_with_hand_roll_zero(
            rig,
            center,
            assets["sampling_frame"]["rotation_world_from_sampling"],
            radius,
            float(viewpoint["theta_deg"]),
            float(viewpoint["phi_deg"]),
        )
        _place_rigid_wrist(rig, world_from_camera)
        _, camera_position, camera_forward, _, _ = _camera_pose_data(
            stage, rig["camera_path"]
        )

        r7_path = rig["root_path"]
        r7_renderables = _renderable_descendants(stage, r7_path)
        r7_groups = _group_by_first_child(r7_path, r7_renderables)
        table_path = assets["table_box"]["scene_prim_path"]
        table_renderables = _renderable_descendants(stage, table_path)

        width, height = (int(value) for value in camera_config["resolution"])
        product = rep.create.render_product(
            rig["camera_path"], (width, height), name="WristOcclusionDiagnostic"
        )
        rgb_annotator = rep.annotators.get("rgb", device="cpu")
        instance_annotator = rep.annotators.get(
            "instance_segmentation_fast", init_params={"colorize": False}, device="cpu"
        )
        depth_annotator = rep.annotators.get("distance_to_image_plane", device="cpu")
        annotators = [rgb_annotator, instance_annotator, depth_annotator]
        for annotator in annotators:
            annotator.attach(product)

        cases: list[tuple[str, list[str]]] = [("baseline", [])]
        cases.extend((f"hide_r7_group_{name}", paths) for name, paths in sorted(r7_groups.items()))
        cases.extend(
            [
                ("hide_all_r7_geometry", r7_renderables),
                ("hide_table_box", table_renderables),
                ("hide_r7_and_table_box", r7_renderables + table_renderables),
            ]
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = args.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        records = []
        panels = []
        for case_name, hidden_paths in cases:
            with _temporarily_hidden(stage, hidden_paths):
                rep.orchestrator.step(rt_subframes=4)
                rgb = np.asarray(_extract_array(rgb_annotator.get_data()))[..., :3].astype(np.uint8)
                instance_output = instance_annotator.get_data()
                instance = np.asarray(_extract_array(instance_output))
                depth = np.asarray(_extract_array(depth_annotator.get_data()), dtype=np.float32)
                target_ids = _target_instance_ids(instance_output, args.class_name, asset_prim_path)
                mask = np.isin(instance, target_ids)
                finite = depth[np.isfinite(depth)]
                record = {
                    "case": case_name,
                    "hidden_prim_count": len(hidden_paths),
                    "hidden_prims": hidden_paths,
                    "mask_pixels": int(mask.sum()),
                    "rgb_mean": float(rgb.mean()),
                    "rgb_dark_pixel_ratio": float(np.all(rgb < 8, axis=2).mean()),
                    "depth_min_m": float(finite.min()) if finite.size else None,
                    "depth_median_m": float(np.median(finite)) if finite.size else None,
                    "depth_below_3cm_ratio": float((finite < 0.03).mean()) if finite.size else None,
                }
                records.append(record)
                title = f"{case_name}｜隐藏 {len(hidden_paths)} 个 Gprim"
                subtitle = (
                    f"Mask={record['mask_pixels']}  深度中位={record['depth_median_m']:.3f}m  "
                    f"近于3cm={record['depth_below_3cm_ratio']:.3f}  暗像素={record['rgb_dark_pixel_ratio']:.3f}"
                )
                panel = _make_panel(rgb, mask, title, subtitle)
                panels.append((case_name, panel))
                Image.fromarray(rgb, mode="RGB").save(image_dir / f"{case_name}.png")
                panel.save(image_dir / f"{case_name}_reference.png")
            rep.orchestrator.step(rt_subframes=2)
            print(
                f"CASE={case_name} mask={record['mask_pixels']} depth_median={record['depth_median_m']:.4f} "
                f"near3cm={record['depth_below_3cm_ratio']:.4f} dark={record['rgb_dark_pixel_ratio']:.4f}",
                flush=True,
            )

        sheet_path = args.output_dir / "occlusion_diagnostic_contact_sheet.png"
        _make_sheet(panels, sheet_path)
        geometry_bounds = {}
        for path in [r7_path, table_path, *r7_groups.keys()]:
            actual_path = path if path.startswith("/") else f"{r7_path}/{path}"
            geometry_bounds[actual_path] = _point_aabb_relation(
                camera_position, _aligned_world_range(stage, actual_path)
            )
        result = {
            "status": "diagnostic_complete",
            "scene": str(scene_path),
            "class": args.class_name,
            "radius_key": args.radius_key,
            "radius_m": radius,
            "camera_prim": rig["camera_path"],
            "camera_source_prim": RIGHT_WRIST_RGB,
            "camera_position_world_m": camera_position.tolist(),
            "camera_forward_world": camera_forward.tolist(),
            "sampling_center_world_m": center.tolist(),
            "camera_psi_at_roll_zero_deg": camera_psi,
            "finger_direction_world": finger_direction.tolist(),
            "hidden_upstream_arm_prims": hidden_upstream,
            "geometry_bounds": geometry_bounds,
            "r7_groups": r7_groups,
            "cases": records,
            "contact_sheet": str(sheet_path),
        }
        json_path = args.output_dir / "occlusion_diagnostic.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"RESULT={json_path}", flush=True)
        print(f"CONTACT_SHEET={sheet_path}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        for annotator in annotators:
            try:
                annotator.detach(product)
            except Exception:
                pass
        if product is not None:
            try:
                product.destroy()
            except Exception:
                pass
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
