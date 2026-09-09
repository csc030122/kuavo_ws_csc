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

from yolo_catchdata.isaac_utils import (  # noqa: E402
    add_semantics as _add_semantics,
    bbox_from_mask as _bbox_from_mask,
    extract_array as _extract_array,
    gpu_info as _gpu_info,
    preview_camera_pose as _preview_camera_pose,
    set_transform as _set_transform,
    target_instance_ids as _target_instance_ids,
)
from yolo_catchdata.camera_geometry import make_transform, matrix_multiply  # noqa: E402
from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402
from yolo_catchdata.support_pose_policy import (  # noqa: E402
    SUPPORT_NORMALS_LOCAL,
    rotation_for_support_face,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera.yaml")
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    parser.add_argument(
        "--support-face",
        choices=tuple(SUPPORT_NORMALS_LOCAL),
        help="覆盖配置中的初始姿态：让指定工件局部面朝向世界 -Z，并物理沉降。",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="支撑面内绕世界 Z 轴的 yaw；仅与 --support-face 一起使用。",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="输出分片和预览文件名后缀，例如 +x_yaw090；默认由支撑面和 yaw 生成。",
    )
    parser.add_argument(
        "--drop-xy-world-m",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="释放点的世界坐标 XY；省略时使用 table_box 内部中心。",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="批量采集只保存稳定姿态分片，跳过不参与训练的外部预览相机渲染。",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Isaac Sim renderer/PhysX 使用的物理 GPU 编号。",
    )
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


def _visual_mesh_vertex_bounds(stage: Any, root: Any) -> dict[str, Any]:
    """使用实际可视 Mesh 顶点计算世界边界，不读取 USD AABB。"""
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


def _initial_transform_from_visual_mesh(
    stage: Any,
    root: Any,
    drop_xy_world_m: list[float],
    table_min_xy: list[float],
    table_max_xy: list[float],
    floor_z_m: float,
    drop_height_m: float,
    boundary_margin_m: float,
    rotation: tuple[tuple[float, ...], ...],
) -> tuple[tuple[tuple[float, ...], ...], list[float], dict[str, Any]]:
    """Place a referenced asset using its transformed Mesh vertices.

    The temporary transform exposes the actual USD Mesh points at the requested
    orientation.  The final XY center is clamped only when necessary so every
    Mesh vertex stays inside the convex table-box interior.  This is an exact
    containment test for the axis-aligned box; no empty AABB corners are used.
    """

    _set_transform(root, make_transform(rotation, (0.0, 0.0, 0.0)))
    provisional = _visual_mesh_vertex_bounds(stage, root)
    half_x = 0.5 * (float(provisional["max"][0]) - float(provisional["min"][0]))
    half_y = 0.5 * (float(provisional["max"][1]) - float(provisional["min"][1]))
    center_x = 0.5 * (float(provisional["max"][0]) + float(provisional["min"][0]))
    center_y = 0.5 * (float(provisional["max"][1]) + float(provisional["min"][1]))
    lower_x = float(table_min_xy[0]) + float(boundary_margin_m) + half_x
    upper_x = float(table_max_xy[0]) - float(boundary_margin_m) - half_x
    lower_y = float(table_min_xy[1]) + float(boundary_margin_m) + half_y
    upper_y = float(table_max_xy[1]) - float(boundary_margin_m) - half_y
    if lower_x > upper_x or lower_y > upper_y:
        raise RuntimeError(
            "实际 Mesh 在当前支撑面/yaw 下无法放入 table_box 内部："
            f"x=[{lower_x:.6f},{upper_x:.6f}] y=[{lower_y:.6f},{upper_y:.6f}]"
        )
    actual_center = [
        min(max(float(drop_xy_world_m[0]), lower_x), upper_x),
        min(max(float(drop_xy_world_m[1]), lower_y), upper_y),
    ]
    translation = (
        actual_center[0] - center_x,
        actual_center[1] - center_y,
        float(floor_z_m) + float(drop_height_m) - float(provisional["min"][2]),
    )
    return make_transform(rotation, translation), actual_center, provisional


def _actual_support_face(
    stage: Any,
    root: Any,
    candidate_faces: list[str] | None = None,
) -> dict[str, Any]:
    """根据沉降后的根节点旋转，找最接近世界向下的局部轴向面。"""
    from pxr import UsdGeom

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(root)
    faces = list(candidate_faces or SUPPORT_NORMALS_LOCAL)
    scores: dict[str, float] = {}
    for face in faces:
        local = SUPPORT_NORMALS_LOCAL[face]
        world = tuple(
            sum(float(matrix[row][column]) * local[column] for column in range(3))
            for row in range(3)
        )
        scores[face] = -world[2]
    if not scores:
        raise ValueError("candidate_faces 不能为空")
    face = max(scores, key=scores.get)
    return {
        "face": face,
        "normal_down_cosine": float(scores[face]),
        "scores": {name: float(value) for name, value in scores.items()},
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
    Image.fromarray(rgb_array).save(raw_path)
    Image.fromarray(mask_array.astype(np.uint8) * 255).save(mask_path)

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
    image = Image.fromarray(annotated)
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
        f"- 最终实际 Mesh 顶点最低点：`{pose['final_world_bounds_m']['min'][2]:.6f} m`。",
        f"- 边界计算方法：`{pose['final_world_bounds_m']['method']}`。",
        f"- 估计支撑接触：`{', '.join(pose['estimated_support_contacts_m']) or '未识别'}`。",
        f"- 请求支撑面：`{pose.get('requested_support_face') or '配置初始姿态'}`。",
        f"- 实际支撑面：`{(pose.get('actual_support') or {}).get('face', '未执行面判定')}`。",
        "- 工件必须接触箱底，并与 table_box 四面侧壁保持配置的安全距离。",
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
    if args.gpu_index < 0:
        raise ValueError("gpu-index 不能为负数")
    assets_config = load_yaml(args.assets_config)
    camera_config = load_yaml(args.camera_config)["camera"]
    collection_config = load_yaml(args.collection_config)
    stable_config = collection_config["stable_pose"]
    approach_support_config = collection_config.get("approach_open", {}).get(
        "support_pose", {}
    )
    definition = next(
        (item for item in assets_config["classes"] if item["name"] == args.class_name),
        None,
    )
    if definition is None:
        raise ValueError(f"未知类别：{args.class_name}")
    scene_path = resolve_from_config(assets_config, assets_config["scene"]["usd"])
    asset_path = resolve_from_config(assets_config, definition["usd"])
    report_dir = resolve_from_config(collection_config, stable_config["report_dir"])
    suffix = ""
    if args.support_face is not None:
        suffix = args.output_suffix or f"{args.support_face}_yaw{args.yaw_deg:g}"
        suffix = "_" + suffix.replace("+", "p").replace("-", "m").replace(".", "d")
    shard_path = report_dir / "shards" / f"{args.class_name}{suffix}.json"
    report_path = report_dir / "shards" / f"{args.class_name}{suffix}_report.md"
    preview_path = report_dir / "preview" / f"{args.class_name}{suffix}_settled_reference.png"

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": True,
            "active_gpu": args.gpu_index,
            "physics_gpu": args.gpu_index,
            "multi_gpu": False,
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
        if args.support_face is None:
            initial_rotation = _rotation_xyz(stable_config["initial_euler_xyz_deg"])
        else:
            class_support = approach_support_config.get("classes", {}).get(
                args.class_name, {}
            )
            allowed_faces = class_support.get("allowed_support_faces", [])
            if args.support_face not in allowed_faces:
                raise ValueError(
                    f"{args.class_name} 不允许支撑面 {args.support_face}；"
                    f"允许集合为 {allowed_faces}"
                )
            initial_rotation = rotation_for_support_face(
                args.support_face, math.radians(float(args.yaw_deg))
            )
        root_path = f"/World/SyntheticData/StablePose/{args.class_name}"
        root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        drop_xy = [float(value) for value in (args.drop_xy_world_m or table_center)]
        asset_prim_path = f"{root_path}/Asset"
        # Instantiate the actual USD Mesh first. The release transform is then
        # solved from transformed Mesh vertices instead of inspection bounds or
        # any hand-written footprint rectangle.
        _set_transform(root, make_transform(initial_rotation, (0.0, 0.0, 0.0)))
        asset_prim = add_reference_with_unit_scale(
            stage,
            asset_path,
            asset_prim_path,
            inspection["meters_per_unit"],
            scene_mpu,
        )
        _add_semantics(asset_prim, args.class_name)
        initial_world_from_asset, actual_drop_xy, initial_mesh_bounds = (
            _initial_transform_from_visual_mesh(
                stage,
                root,
                drop_xy,
                table_min,
                table_max,
                float(table["floor_z_m"]),
                float(stable_config["drop_height_m"]),
                float(
                    collection_config.get("approach_open", {})
                    .get("object_translation", {})
                    .get("boundary_margin_m", 0.0)
                ),
                initial_rotation,
            )
        )
        _set_transform(root, initial_world_from_asset)
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
        validation_config = approach_support_config.get("validation", {})
        floor_tolerance = float(
            validation_config.get("floor_penetration_tolerance_m", 0.001)
        )
        floor_snap_m = 0.0
        floor_snap_enabled = bool(
            validation_config.get("floor_snap_after_settle", False)
        )
        floor_snap_max_m = float(validation_config.get("floor_snap_max_m", 0.0))
        penetration_m = floor_z - float(final_bounds["min"][2])
        if (
            floor_snap_enabled
            and penetration_m > floor_tolerance
            and penetration_m <= floor_snap_max_m
        ):
            snapped_transform = [list(row) for row in final_world_from_asset]
            snapped_transform[2][3] += penetration_m
            _set_transform(root, snapped_transform)
            floor_snap_m = penetration_m
            final_world_from_asset = _column_transform_from_gf(
                UsdGeom.XformCache().GetLocalToWorldTransform(root)
            )
            final_bounds = _visual_mesh_vertex_bounds(stage, root)
            translation_change_m, rotation_change_deg = _pose_change(
                initial_world_from_asset, final_world_from_asset
            )
        wall_tolerance = float(
            validation_config.get("wall_penetration_tolerance_m", 0.001)
        )
        if final_bounds["min"][2] < floor_z - floor_tolerance:
            issues.append(
                "实际可视网格顶点穿透 table_box 地板超过 "
                f"{floor_tolerance:.3f} m"
            )
        floor_contact_tolerance = float(
            validation_config.get("floor_contact_tolerance_m", 0.003)
        )
        if final_bounds["min"][2] > floor_z + floor_contact_tolerance:
            issues.append(
                "沉降后工件没有接触 table_box 箱底："
                f"间隙超过 {floor_contact_tolerance:.3f} m"
            )
        wall_clearance = float(
            validation_config.get(
                "wall_clearance_m", table.get("wall_clearance_m", 0.0)
            )
        )
        if (
            final_bounds["min"][0] < table_min[0] + wall_clearance - wall_tolerance
            or final_bounds["max"][0] > table_max[0] - wall_clearance + wall_tolerance
            or final_bounds["min"][1] < table_min[1] + wall_clearance - wall_tolerance
            or final_bounds["max"][1] > table_max[1] - wall_clearance + wall_tolerance
        ):
            issues.append(
                "实际可视网格距 table_box 侧壁不足 "
                f"{wall_clearance:.3f} m"
            )

        actual_support = None
        support_normal_tolerance_deg = float(
            validation_config.get("support_normal_tolerance_deg", 15.0)
        )
        if args.support_face is not None:
            allowed_faces = list(
                approach_support_config["classes"][args.class_name][
                    "allowed_support_faces"
                ]
            )
            actual_support = _actual_support_face(stage, root)
            actual_support["matches_requested_face"] = (
                actual_support["face"] == args.support_face
            )
            if bool(
                validation_config.get(
                    "require_allowed_support_face_after_settle", True
                )
            ):
                expected_cosine = math.cos(math.radians(support_normal_tolerance_deg))
                if actual_support["face"] not in allowed_faces:
                    issues.append(
                        f"实际最接近支撑面 {actual_support['face']} 不在允许集合 {allowed_faces}"
                    )
                if actual_support["normal_down_cosine"] < expected_cosine:
                    issues.append(
                        "沉降后支撑面法向偏离世界 -Z 超过 "
                        f"{support_normal_tolerance_deg:.1f}°"
                    )
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
        rgb_path = mask_path = annotated_path = None
        if not args.skip_preview:
            size = final_bounds["size"]
            hfov = float(camera_config["intrinsics"]["horizontal_fov_degrees"])
            radius = max(
                0.16,
                min(
                    0.85,
                    max(size[0], size[1])
                    / (0.42 * 2.0 * math.tan(math.radians(hfov) / 2.0)),
                ),
            )
            _, world_from_camera = _preview_camera_pose(
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
            camera.CreateClippingRangeAttr().Set(
                Gf.Vec2f(*camera_config["clipping_range_m"])
            )
            _set_transform(camera.GetPrim(), world_from_camera)
            dome = UsdLux.DomeLight.Define(stage, "/World/SyntheticData/StablePoseDome")
            dome.CreateIntensityAttr().Set(500.0)
            width, height = (int(value) for value in camera_config["resolution"])
            render_product = rep.create.render_product(
                camera_path, (width, height), name="StablePose"
            )
            rgb_annotator = rep.annotators.get("rgb", device="cpu")
            instance_annotator = rep.annotators.get(
                "instance_segmentation_fast",
                init_params={"colorize": False},
                device="cpu",
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
            target_ids = _target_instance_ids(
                instance_output, args.class_name, asset_prim_path
            )
            mask = np.isin(instance, target_ids)
            if rgb.ndim < 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
                issues.append("稳定姿态预览 RGB 为空")
            elif not target_ids or not np.any(mask):
                issues.append("稳定姿态预览中未找到目标实例 Mask")
            else:
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
                "floor_snap_m": floor_snap_m,
                "final_world_bounds_m": final_bounds,
                "estimated_support_contacts_m": support_contacts,
                "sampling_center_world_m": sampling_center_world,
                "requested_support_face": args.support_face,
                "requested_in_plane_yaw_deg": float(args.yaw_deg)
                if args.support_face is not None
                else None,
                "requested_drop_xy_world_m": drop_xy,
                "actual_drop_xy_world_m": actual_drop_xy,
                "initial_visual_mesh_bounds_at_origin_m": initial_mesh_bounds,
                "actual_support": actual_support,
            },
            "gpu": _gpu_info(),
            "issues": issues,
            "outputs": {
                "shard": str(shard_path),
                "report": str(report_path),
                "rgb": str(rgb_path) if rgb_path is not None else None,
                "mask": str(mask_path) if mask_path is not None else None,
                "preview": str(annotated_path) if annotated_path is not None else None,
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
