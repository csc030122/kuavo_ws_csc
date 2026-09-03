#!/usr/bin/env python3
"""阶段三前置：在真实 table_box 中物理沉降单类工件并保存稳定姿态。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_radius import (  # noqa: E402
    _add_semantics,
    _bbox_from_mask,
    _camera_pose,
    _extract_array,
    _gpu_info,
    _set_transform,
    _target_instance_ids,
)
from yolo_catchdata.camera_geometry import make_transform, matrix_multiply  # noqa: E402
from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera.yaml")
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    return parser.parse_args()


def _rotation_xyz(euler_deg: list[float]) -> tuple[tuple[float, ...], ...]:
    roll, pitch, yaw = (math.radians(float(value)) for value in euler_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _initial_transform(
    bounds_m: dict[str, list[float]],
    table_center_xy: list[float],
    floor_z_m: float,
    drop_height_m: float,
    rotation: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    minimum, maximum = bounds_m["min"], bounds_m["max"]
    rotated = []
    for x in (minimum[0], maximum[0]):
        for y in (minimum[1], maximum[1]):
            for z in (minimum[2], maximum[2]):
                rotated.append(
                    tuple(
                        sum(rotation[row][column] * point for column, point in enumerate((x, y, z)))
                        for row in range(3)
                    )
                )
    rotated_min = [min(point[index] for point in rotated) for index in range(3)]
    rotated_max = [max(point[index] for point in rotated) for index in range(3)]
    translation = (
        table_center_xy[0] - (rotated_min[0] + rotated_max[0]) * 0.5,
        table_center_xy[1] - (rotated_min[1] + rotated_max[1]) * 0.5,
        floor_z_m + drop_height_m - rotated_min[2],
    )
    return make_transform(rotation, translation)


def _column_transform_from_gf(matrix: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(matrix[column][row]) for column in range(4)) for row in range(4)
    )


def _as_list(matrix: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _norm(values: Any) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _pose_change(initial: Any, final: Any) -> tuple[float, float]:
    translation = math.sqrt(
        sum((float(final[index][3]) - float(initial[index][3])) ** 2 for index in range(3))
    )
    relative_trace = sum(
        sum(float(initial[k][row]) * float(final[k][row]) for k in range(3))
        for row in range(3)
    )
    cosine = max(-1.0, min(1.0, (relative_trace - 1.0) * 0.5))
    return translation, math.degrees(math.acos(cosine))


def _world_bounds(stage: Any, prim: Any) -> dict[str, list[float]]:
    from pxr import Usd, UsdGeom

    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    ).ComputeWorldBound(prim).ComputeAlignedRange()
    return {
        "min": [float(value) for value in bbox.GetMin()],
        "max": [float(value) for value in bbox.GetMax()],
        "center": [float(value) for value in bbox.GetMidpoint()],
        "size": [float(value) for value in bbox.GetSize()],
    }


def _visual_mesh_vertex_bounds(stage: Any, root: Any) -> dict[str, Any]:
    """使用实际可视 Mesh 顶点计算世界边界，避免旋转 AABB 空角点误报穿透。"""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points_world: list[list[float]] = []
    mesh_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        transform = cache.GetLocalToWorldTransform(prim)
        for point in UsdGeom.Mesh(prim).GetPointsAttr().Get() or []:
            world = transform.Transform(point)
            points_world.append([float(world[index]) for index in range(3)])
    if not points_world:
        raise RuntimeError(f"{root.GetPath()} 下没有可用于边界检查的 Mesh 顶点")
    minimum = [min(point[index] for point in points_world) for index in range(3)]
    maximum = [max(point[index] for point in points_world) for index in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "center": [(minimum[index] + maximum[index]) * 0.5 for index in range(3)],
        "size": [maximum[index] - minimum[index] for index in range(3)],
        "method": "exact_transformed_visual_mesh_vertices",
        "mesh_count": mesh_count,
        "vertex_count": len(points_world),
    }


def _estimated_support_contacts(
    bounds: dict[str, Any], table: dict[str, Any], threshold_m: float
) -> dict[str, float]:
    """按实际网格边界与箱底/内壁的距离记录可能的支撑接触。"""
    inner_min = table["inner_bounds_world_m"]["min_xy"]
    inner_max = table["inner_bounds_world_m"]["max_xy"]
    distances = {
        "floor": abs(float(bounds["min"][2]) - float(table["floor_z_m"])),
        "x_min_wall": abs(float(bounds["min"][0]) - float(inner_min[0])),
        "x_max_wall": abs(float(inner_max[0]) - float(bounds["max"][0])),
        "y_min_wall": abs(float(bounds["min"][1]) - float(inner_min[1])),
        "y_max_wall": abs(float(inner_max[1]) - float(bounds["max"][1])),
    }
    return {name: distance for name, distance in distances.items() if distance <= threshold_m}


def _save_preview(path: Path, rgb: Any, mask: Any, class_name: str) -> tuple[Path, Path, Path]:
    import numpy as np
    from PIL import Image, ImageDraw

    rgb_array = np.asarray(rgb, dtype=np.uint8)[..., :3]
    mask_array = np.asarray(mask, dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_name(f"{class_name}_settled_rgb.png")
    mask_path = path.with_name(f"{class_name}_settled_mask.png")
    Image.fromarray(rgb_array, mode="RGB").save(raw_path)
    Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L").save(mask_path)

    interior = mask_array.copy()
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    interior[1:-1, 1:-1] &= (
        mask_array[:-2, 1:-1]
        & mask_array[2:, 1:-1]
        & mask_array[1:-1, :-2]
        & mask_array[1:-1, 2:]
    )
    boundary = mask_array & ~interior
    thick = boundary.copy()
    thick[:-1, :] |= boundary[1:, :]
    thick[1:, :] |= boundary[:-1, :]
    thick[:, :-1] |= boundary[:, 1:]
    thick[:, 1:] |= boundary[:, :-1]
    annotated = rgb_array.copy()
    annotated[thick] = (0, 255, 255)
    image = Image.fromarray(annotated, mode="RGB")
    draw = ImageDraw.Draw(image)
    bbox = _bbox_from_mask(mask_array)
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline=(255, 215, 0), width=2)
    draw.rectangle((8, 8, 760, 42), fill=(0, 0, 0))
    draw.text(
        (16, 15),
        f"physics-settled in real table_box | {class_name} | cyan=mask contour",
        fill=(255, 255, 255),
    )
    image.save(path)
    return raw_path, mask_path, path


def _write_report(result: dict[str, Any], path: Path) -> None:
    pose = result["pose"]
    status = "通过" if result["status"] == "pass" else "失败"
    lines = [
        f"# {result['class']} 物理稳定姿态报告",
        "",
        f"状态：**{status}**",
        "",
        "## 方法",
        "",
        "- 工件在机器人面前的真实 `table_box` 中从指定高度释放。",
        "- RGB、Mask 使用原始 USD 可视网格，没有抽稀或减面。",
        f"- 动态碰撞体近似：`{result['physics']['dynamic_collision_approximation']}`。",
        f"- `table_box` 静态碰撞近似：`{result['physics']['table_collision_approximation']}`。",
        "- 碰撞近似仅影响沉降，不参与 RGB、Mask、Depth 或尺度计算。",
        "",
        "## 稳定判定",
        "",
        f"- 仿真帧数：`{pose['settled_frame']}`。",
        f"- 连续稳定帧数：`{pose['stable_consecutive_frames']}`。",
        f"- 最终线速度：`{pose['final_linear_speed_m_s']:.6f} m/s`。",
        f"- 最终角速度：`{pose['final_angular_speed_deg_s']:.6f} °/s`。",
        f"- 沉降平移量：`{pose['translation_change_m']:.6f} m`。",
        f"- 沉降旋转变化：`{pose['rotation_change_deg']:.6f}°`。",
        f"- 最终世界包围盒最低点：`{pose['final_world_bounds_m']['min'][2]:.6f} m`。",
        f"- 边界计算方法：`{pose['final_world_bounds_m']['method']}`。",
        f"- 估计支撑接触：`{', '.join(pose['estimated_support_contacts_m']) or '未识别'}`。",
        "- 工件与侧壁接触并形成静止支撑属于有效自然稳定姿态，不作为失败条件。",
        "",
        "## 输出",
        "",
        f"- 分片：`{result['outputs']['shard']}`",
        f"- 原始 RGB：`{result['outputs']['rgb']}`",
        f"- 二值 Mask：`{result['outputs']['mask']}`",
        f"- 轮廓参考图：`{result['outputs']['preview']}`",
    ]
    if result["issues"]:
        lines.extend(["", "## 错误", ""])
        lines.extend(f"- {item}" for item in result["issues"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assets_config = load_yaml(args.assets_config)
    camera_config = load_yaml(args.camera_config)["camera"]
    collection_config = load_yaml(args.collection_config)
    stable_config = collection_config["stable_pose"]
    definition = next(
        (item for item in assets_config["classes"] if item["name"] == args.class_name),
        None,
    )
    if definition is None:
        raise ValueError(f"未知类别：{args.class_name}")
    scene_path = resolve_from_config(assets_config, assets_config["scene"]["usd"])
    asset_path = resolve_from_config(assets_config, definition["usd"])
    report_dir = resolve_from_config(collection_config, stable_config["report_dir"])
    shard_path = report_dir / "shards" / f"{args.class_name}.json"
    report_path = report_dir / "shards" / f"{args.class_name}_report.md"
    preview_path = report_dir / "preview" / f"{args.class_name}_settled_reference.png"

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": True,
            "renderer": stable_config["renderer"],
            "width": int(camera_config["resolution"][0]),
            "height": int(camera_config["resolution"][1]),
        }
    )
    annotators = []
    render_product = None
    physics_interface = None
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from omni.physics.core import (
            get_physics_interaction_interface,
            get_physics_simulation_interface,
        )
        from omni.physx.scripts import utils as physics_utils
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils

        from yolo_catchdata.asset_loader import (
            add_reference_with_unit_scale,
            inspect_usd,
            sha256_file,
        )

        context = omni.usd.get_context()
        if not context.open_stage(str(scene_path)):
            raise RuntimeError(f"无法打开实验室场景：{scene_path}")
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("无法取得实验室场景 Stage")
        # 运行时修改只写到匿名 Session Layer，源 USD 不保存、不修改。
        stage.SetEditTarget(stage.GetSessionLayer())
        scene_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
        table = assets_config["table_box"]
        table_min = table["inner_bounds_world_m"]["min_xy"]
        table_max = table["inner_bounds_world_m"]["max_xy"]
        table_center = [(table_min[index] + table_max[index]) * 0.5 for index in range(2)]
        table_prim = stage.GetPrimAtPath(table["scene_prim_path"])
        if not table_prim.IsValid():
            raise RuntimeError(f"真实 table_box Prim 不存在：{table['scene_prim_path']}")
        table_colliders_before = sum(
            prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(table_prim)
        )
        physics_utils.setStaticCollider(table_prim, UsdPhysics.Tokens.none)
        table_colliders_after = sum(
            prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(table_prim)
        )
        print(
            f"REAL_TABLE_COLLIDERS before={table_colliders_before} after={table_colliders_after}",
            flush=True,
        )

        inspection = inspect_usd(asset_path)
        initial_rotation = _rotation_xyz(stable_config["initial_euler_xyz_deg"])
        initial_world_from_asset = _initial_transform(
            inspection["bounds_m"],
            table_center,
            float(table["floor_z_m"]),
            float(stable_config["drop_height_m"]),
            initial_rotation,
        )

        root_path = f"/World/SyntheticData/StablePose/{args.class_name}"
        root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        _set_transform(root, initial_world_from_asset)
        asset_prim_path = f"{root_path}/Asset"
        asset_prim = add_reference_with_unit_scale(
            stage,
            asset_path,
            asset_prim_path,
            inspection["meters_per_unit"],
            scene_mpu,
        )
        _add_semantics(asset_prim, args.class_name)
        visual_mesh_count = sum(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(asset_prim))
        authored_collision_count = sum(
            prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(asset_prim)
        )
        physics_utils.setRigidBody(
            root,
            stable_config["dynamic_collision_approximation"],
            False,
        )
        decomposition_config = stable_config["convex_decomposition"]
        if stable_config["dynamic_collision_approximation"] == "convexDecomposition":
            for prim in Usd.PrimRange(asset_prim):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomposition.CreateMaxConvexHullsAttr().Set(
                    int(decomposition_config["max_convex_hulls"])
                )
                decomposition.CreateHullVertexLimitAttr().Set(
                    int(decomposition_config["hull_vertex_limit"])
                )
                decomposition.CreateVoxelResolutionAttr().Set(
                    int(decomposition_config["voxel_resolution"])
                )
                decomposition.CreateErrorPercentageAttr().Set(
                    float(decomposition_config["error_percentage"])
                )
                decomposition.CreateShrinkWrapAttr().Set(
                    bool(decomposition_config["shrink_wrap"])
                )
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr().Set(float(stable_config["mass_kg"]))
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        physx_body.CreateLinearDampingAttr().Set(0.15)
        physx_body.CreateAngularDampingAttr().Set(0.15)
        runtime_collision_count = sum(
            prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(asset_prim)
        )

        physics_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        if physics_scenes:
            physics_scene = UsdPhysics.Scene(physics_scenes[0])
        else:
            physics_scene = UsdPhysics.Scene.Define(
                stage, "/World/SyntheticData/StablePosePhysicsScene"
            )
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        for _ in range(6):
            app.update()

        cache = UsdUtils.StageCache.Get()
        cache.Insert(stage)
        stage_id = cache.GetId(stage).ToLongInt()
        physics_interface = get_physics_simulation_interface()
        interaction = get_physics_interaction_interface()
        physics_interface.initialize(stage_id)
        dt = float(stable_config["physics_dt_s"])
        stable_required = int(stable_config["stable_consecutive_frames"])
        stable_count = 0
        settled_frame = None
        final_debug = None
        for frame in range(1, int(stable_config["maximum_frames"]) + 1):
            physics_interface.simulate(dt, frame * dt)
            debug = interaction.get_prim_debug_data(root_path)
            if not debug or "Linear velocity" not in debug or "Angular velocity" not in debug:
                continue
            linear_speed = _norm(debug["Linear velocity"]["value"])
            angular_speed_rad = _norm(debug["Angular velocity"]["value"])
            final_debug = debug
            below = (
                linear_speed <= float(stable_config["linear_speed_threshold_m_s"])
                and math.degrees(angular_speed_rad)
                <= float(stable_config["angular_speed_threshold_deg_s"])
            )
            if frame >= int(stable_config["minimum_frames"]) and below:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= stable_required:
                settled_frame = frame
                break
            if frame % 120 == 0:
                print(
                    f"SETTLE_PROGRESS class={args.class_name} frame={frame} "
                    f"linear={linear_speed:.6f} angular_deg={math.degrees(angular_speed_rad):.6f}",
                    flush=True,
                )
        physics_interface.close()
        physics_interface = None

        issues = []
        if settled_frame is None:
            issues.append("在最大仿真帧数内未达到连续稳定阈值")
            settled_frame = int(stable_config["maximum_frames"])
        if final_debug is None:
            issues.append("PhysX 未返回目标刚体调试数据")
            final_linear_speed = float("inf")
            final_angular_speed_deg = float("inf")
        else:
            final_linear_speed = _norm(final_debug["Linear velocity"]["value"])
            final_angular_speed_deg = math.degrees(
                _norm(final_debug["Angular velocity"]["value"])
            )

        final_world_from_asset = _column_transform_from_gf(
            UsdGeom.XformCache().GetLocalToWorldTransform(root)
        )
        translation_change_m, rotation_change_deg = _pose_change(
            initial_world_from_asset, final_world_from_asset
        )
        final_bounds = _visual_mesh_vertex_bounds(stage, root)
        floor_z = float(table["floor_z_m"])
        if final_bounds["min"][2] < floor_z - 0.01:
            issues.append("实际可视网格顶点穿透 table_box 地板超过 1 cm")
        if final_bounds["min"][2] > floor_z + 0.05:
            issues.append("沉降后工件未落到 table_box 地板附近")
        if not (
            table_min[0] <= final_bounds["center"][0] <= table_max[0]
            and table_min[1] <= final_bounds["center"][1] <= table_max[1]
        ):
            issues.append("沉降后工件中心离开 table_box 内部区域")
        support_contacts = _estimated_support_contacts(final_bounds, table, 0.015)
        print(
            f"EXACT_VISUAL_BOUNDS={final_bounds} SUPPORT_CONTACTS={support_contacts}",
            flush=True,
        )

        # 物理完成后关闭刚体，再用真实实验室场景渲染稳定姿态预览。
        UsdPhysics.RigidBodyAPI(root).GetRigidBodyEnabledAttr().Set(False)
        asset_from_object = tuple(
            tuple(float(value) for value in row) for row in definition["asset_from_canonical"]
        )
        world_from_object = matrix_multiply(final_world_from_asset, asset_from_object)
        sampling = definition["sampling_center_canonical_m"]
        sampling_center_world = [
            sum(world_from_object[row][column] * sampling[column] for column in range(3))
            + world_from_object[row][3]
            for row in range(3)
        ]
        size = inspection["bounds_m"]["size"]
        hfov = float(camera_config["intrinsics"]["horizontal_fov_degrees"])
        radius = max(0.16, min(0.85, max(size[0], size[1]) / (0.42 * 2.0 * math.tan(math.radians(hfov) / 2.0))))
        _, world_from_camera = _camera_pose(
            sampling_center_world,
            assets_config["sampling_frame"]["rotation_world_from_sampling"],
            radius,
            0.0,
            60.0,
            0.0,
        )
        camera_path = str(camera_config["prim_path"])
        UsdGeom.Xform.Define(stage, str(Path(camera_path).parent))
        camera = UsdGeom.Camera.Define(stage, camera_path)
        focal_length = 10.0
        vfov = float(camera_config["intrinsics"]["vertical_fov_degrees"])
        camera.CreateFocalLengthAttr().Set(focal_length)
        camera.CreateHorizontalApertureAttr().Set(
            2.0 * focal_length * math.tan(math.radians(hfov) * 0.5)
        )
        camera.CreateVerticalApertureAttr().Set(
            2.0 * focal_length * math.tan(math.radians(vfov) * 0.5)
        )
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*camera_config["clipping_range_m"]))
        _set_transform(camera.GetPrim(), world_from_camera)
        dome = UsdLux.DomeLight.Define(stage, "/World/SyntheticData/StablePoseDome")
        dome.CreateIntensityAttr().Set(500.0)
        width, height = (int(value) for value in camera_config["resolution"])
        render_product = rep.create.render_product(camera_path, (width, height), name="StablePose")
        rgb_annotator = rep.annotators.get("rgb", device="cpu")
        instance_annotator = rep.annotators.get(
            "instance_segmentation_fast", init_params={"colorize": False}, device="cpu"
        )
        annotators = [rgb_annotator, instance_annotator]
        for annotator in annotators:
            annotator.attach(render_product)
        for _ in range(int(stable_config["warmup_frames"])):
            app.update()
        rep.orchestrator.step(rt_subframes=int(stable_config["rt_subframes"]))
        rgb = np.asarray(_extract_array(rgb_annotator.get_data()))
        instance_output = instance_annotator.get_data()
        instance = np.asarray(_extract_array(instance_output))
        target_ids = _target_instance_ids(instance_output, args.class_name, asset_prim_path)
        mask = np.isin(instance, target_ids)
        if not target_ids or not np.any(mask):
            issues.append("稳定姿态预览中未找到目标实例 Mask")
        rgb_path, mask_path, annotated_path = _save_preview(
            preview_path, rgb, mask, args.class_name
        )

        result = {
            "schema_version": 1,
            "artifact_kind": "stable_pose_shard",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if not issues else "fail",
            "class": args.class_name,
            "class_id": int(definition["class_id"]),
            "pose_id": stable_config["pose_id"],
            "asset": {
                "path": str(asset_path),
                "sha256": sha256_file(asset_path),
                "meters_per_unit": inspection["meters_per_unit"],
                "visual_geometry": stable_config["visual_geometry"],
                "visual_mesh_count": visual_mesh_count,
                "authored_collision_count_before_runtime": authored_collision_count,
                "runtime_collision_count": runtime_collision_count,
            },
            "scene": {
                "path": str(scene_path),
                "table_box_prim_path": table["scene_prim_path"],
                "table_colliders_before_runtime": table_colliders_before,
                "table_colliders_after_runtime": table_colliders_after,
                "floor_z_m": floor_z,
            },
            "physics": {
                "dynamic_collision_approximation": stable_config[
                    "dynamic_collision_approximation"
                ],
                "table_collision_approximation": stable_config[
                    "table_collision_approximation"
                ],
                "convex_decomposition": decomposition_config,
                "mass_kg": float(stable_config["mass_kg"]),
                "drop_height_m": float(stable_config["drop_height_m"]),
                "initial_euler_xyz_deg": [
                    float(value) for value in stable_config["initial_euler_xyz_deg"]
                ],
                "physics_dt_s": dt,
                "linear_speed_threshold_m_s": float(
                    stable_config["linear_speed_threshold_m_s"]
                ),
                "angular_speed_threshold_deg_s": float(
                    stable_config["angular_speed_threshold_deg_s"]
                ),
            },
            "pose": {
                "initial_world_from_asset": _as_list(initial_world_from_asset),
                "settled_world_from_asset": _as_list(final_world_from_asset),
                "settled_world_from_object": _as_list(world_from_object),
                "settled_frame": settled_frame,
                "stable_consecutive_frames": stable_count,
                "final_linear_speed_m_s": final_linear_speed,
                "final_angular_speed_deg_s": final_angular_speed_deg,
                "translation_change_m": translation_change_m,
                "rotation_change_deg": rotation_change_deg,
                "final_world_bounds_m": final_bounds,
                "estimated_support_contacts_m": support_contacts,
                "sampling_center_world_m": sampling_center_world,
            },
            "gpu": _gpu_info(),
            "issues": issues,
            "outputs": {
                "shard": str(shard_path),
                "report": str(report_path),
                "rgb": str(rgb_path),
                "mask": str(mask_path),
                "preview": str(annotated_path),
            },
        }
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_report(result, report_path)
        print(f"STABLE_POSE_SHARD={shard_path}", flush=True)
        print(
            f"CLASS={args.class_name} STATUS={result['status'].upper()} "
            f"frame={settled_frame} linear={final_linear_speed:.6f} "
            f"angular_deg={final_angular_speed_deg:.6f}",
            flush=True,
        )
        return 0 if result["status"] == "pass" else 1
    finally:
        if physics_interface is not None:
            try:
                physics_interface.close()
            except Exception:
                pass
        for annotator in annotators:
            try:
                annotator.detach(render_product)
            except Exception:
                pass
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
