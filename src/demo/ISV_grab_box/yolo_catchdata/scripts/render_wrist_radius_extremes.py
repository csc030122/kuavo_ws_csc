#!/usr/bin/env python3
"""使用机器人真实右腕 D405，渲染四类工件在全局最大/最小半径处的图像。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
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
    _gf_matrix_from_column_transform,
    _set_transform,
    _target_instance_ids,
)
from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402


WAIST_PATH = (
    "/World/Robot/Geometry/base_link/base_to_joint/knee_link_1/leg_link/"
    "waist_link/waist_yaw_link"
)
RIGHT_ARM_LINKS = tuple(
    WAIST_PATH + "".join(f"/zarm_r{joint}_link" for joint in range(1, index + 1))
    for index in range(1, 8)
)
RIGHT_ARM_VISUAL_NAMES = (
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_forearm",
    "r_hand_yaw",
    "r_hand_roll",
)
RIGHT_WRIST_RGB = (
    RIGHT_ARM_LINKS[-1]
    + "/r_hand_tripod/r_hand_camera_link/right_wrist_d405_rgb"
)
JOINT_AXES = ("Y", "X", "Z", "Y", "Z", "X", "Y")
# 来自 yolo_data/scripts/render_right_wrist_capture_pose.py 的已验证腕部视角。
REFERENCE_JOINTS_DEG = (
    -86.348167,
    -48.620611,
    81.393436,
    -126.305789,
    24.460575,
    41.660079,
    0.316566,
)
# 基准姿态：相机光轴直接穿过 sampling center，不使用图像平面偏置。
TARGET_RAY_LOCAL_X = 0.0
TARGET_RAY_LOCAL_Y = 0.0
BASELINE_OPTICAL_ROLL_DEG = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera.yaml")
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    parser.add_argument("--stable-poses", type=Path, default=CONFIG_DIR / "stable_poses.json")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output/workspace_calibration/wrist_extremes"
    )
    parser.add_argument("--class-name", choices=("left", "right", "hose", "outhandle"))
    parser.add_argument("--radius-key", choices=("r_max", "r_min"))
    parser.add_argument("--theta-deg", type=float, help="覆盖配置中的基准 theta，仅用于预览")
    parser.add_argument("--phi-deg", type=float, help="覆盖配置中的基准 phi，仅用于预览")
    args = parser.parse_args()
    if (args.class_name is None) != (args.radius_key is None):
        parser.error("--class-name 与 --radius-key 必须同时提供")
    return args


def _set_reference_joint_orientations(stage: Any) -> None:
    from pxr import Gf, UsdGeom

    axes = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    for link_path, axis, degrees in zip(RIGHT_ARM_LINKS, JOINT_AXES, REFERENCE_JOINTS_DEG):
        link = stage.GetPrimAtPath(link_path)
        orient_ops = [
            op
            for op in UsdGeom.Xformable(link).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient
        ]
        if len(orient_ops) != 1:
            raise RuntimeError(f"{link_path} 应当且仅应当包含一个旋转操作")
        rotation = Gf.Rotation(axes[axis], float(degrees)).GetQuat()
        orient_ops[0].Set(
            Gf.Quatf(
                float(rotation.GetReal()),
                Gf.Vec3f(*(float(value) for value in rotation.GetImaginary())),
            )
        )
def _camera_pose_data(
    stage: Any, camera_prim_path: str = RIGHT_WRIST_RGB
) -> tuple[Any, Any, Any, Any, Any]:
    import numpy as np
    from pxr import Gf, UsdGeom

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(camera_prim_path)
    )
    position = np.asarray(tuple(matrix.ExtractTranslation()), dtype=float)
    forward = np.asarray(tuple(matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))), dtype=float)
    up = np.asarray(tuple(matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))), dtype=float)
    right = np.asarray(tuple(matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))), dtype=float)
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right /= np.linalg.norm(right)
    return matrix, position, forward, up, right


def _prepare_rigid_wrist(stage: Any, scene_path: Path) -> dict[str, Any]:
    """复制 r7 腕部为独立采集刚体，并完整保留真实手眼安装外参。

    原机器人 r7 是 articulation 的一部分。仅在 USD 层覆盖 r7 变换时，Camera
    Prim 会更新，但手部渲染网格仍可能被 PhysX/Fabric 保持在旧位姿。因此采集
    阶段使用同一场景中 r7 子树的引用副本，避免手和相机发生空间分离。
    """
    from pxr import Sdf, Usd, UsdGeom

    _set_reference_joint_orientations(stage)
    camera_world, _, _, _, _ = _camera_pose_data(stage)
    r7_prim = stage.GetPrimAtPath(RIGHT_ARM_LINKS[-1])
    r7_world = UsdGeom.XformCache().GetLocalToWorldTransform(r7_prim)
    camera_from_r7 = camera_world * r7_world.GetInverse()

    detached_root_path = "/World/SyntheticData/DetachedRightWrist"
    detached_root = stage.DefinePrim(detached_root_path, "Xform")
    detached_root.GetReferences().AddReference(str(scene_path), RIGHT_ARM_LINKS[-1])
    detached_camera_path = detached_root_path + RIGHT_WRIST_RGB[len(RIGHT_ARM_LINKS[-1]):]
    detached_camera = stage.GetPrimAtPath(detached_camera_path)
    if not detached_camera.IsValid() or not detached_camera.IsA(UsdGeom.Camera):
        raise RuntimeError(f"独立腕部未能加载真实 D405 相机：{detached_camera_path}")

    # 引用副本只承担成像可视化，不参与物理。显式关闭所有可能继承下来的
    # 刚体、碰撞和 articulation 开关，防止 Replicator step 把它拉回原机械臂姿态。
    for prim in Usd.PrimRange(detached_root):
        for attribute_name in (
            "physics:rigidBodyEnabled",
            "physics:collisionEnabled",
            "physics:articulationEnabled",
        ):
            prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Bool).Set(False)

    xformable = UsdGeom.Xformable(detached_root)
    xformable.ClearXformOpOrder()
    transform_op = xformable.AddTransformOp(
        UsdGeom.XformOp.PrecisionDouble,
        "wristRadiusOverride",
    )
    xformable.SetResetXformStack(True)

    # 原始 r7 的渲染网格仍由 articulation/Fabric 管理，必须逐个隐藏；相机
    # 本身不是 Gprim，不会因这些可见性覆盖而失效。
    hidden_original_wrist_prims = []
    for prim in Usd.PrimRange(r7_prim):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_original_wrist_prims.append(str(prim.GetPath()))
    return {
        "camera_from_r7": camera_from_r7,
        "transform_op": transform_op,
        "root_path": detached_root_path,
        "camera_path": detached_camera_path,
        "hidden_original_wrist_prims": hidden_original_wrist_prims,
    }


def _hide_detached_upstream_arm(stage: Any) -> list[str]:
    """隐藏不随末端刚体移动的 r1-r6 可视模型，避免其包住相机。"""
    from pxr import UsdGeom

    hidden = []
    for link_path, visual_name in zip(RIGHT_ARM_LINKS[:-1], RIGHT_ARM_VISUAL_NAMES):
        visual_prim = stage.GetPrimAtPath(f"{link_path}/{visual_name}")
        if visual_prim.IsValid() and visual_prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(visual_prim).MakeInvisible()
            hidden.append(str(visual_prim.GetPath()))
    return hidden


def _place_rigid_wrist(rig: dict[str, Any], world_from_camera: Any) -> Any:
    """保持真实安装外参，把相机刚体位姿直接设置为指定 look-at 位姿。"""
    desired_camera_world = _gf_matrix_from_column_transform(world_from_camera)
    desired_r7_world = rig["camera_from_r7"].GetInverse() * desired_camera_world
    rig["transform_op"].Set(desired_r7_world)
    return desired_camera_world.ExtractTranslation()


def _camera_pose_with_hand_roll_zero(
    rig: dict[str, Any],
    sampling_center: Any,
    sampling_rotation: Any,
    radius: float,
    theta_deg: float,
    phi_deg: float,
) -> tuple[Any, Any, float, Any]:
    """光轴对准 sampling center，并令四指方向尽量贴近世界 -Y。"""
    import numpy as np
    from pxr import Gf

    desired_finger_direction = np.asarray((0.0, -1.0, 0.0), dtype=float)
    best = None
    # 0.5° 搜索精度足以用于方向预览；正式随机化时在该零点上叠加 roll。
    for camera_psi_deg in np.arange(-180.0, 180.0, 0.5):
        camera_position, world_from_camera = _camera_pose(
            list(sampling_center),
            sampling_rotation,
            radius,
            theta_deg,
            phi_deg,
            float(camera_psi_deg),
        )
        camera_world = _gf_matrix_from_column_transform(world_from_camera)
        r7_world = rig["camera_from_r7"].GetInverse() * camera_world
        finger = np.asarray(
            tuple(r7_world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))), dtype=float
        )
        finger /= np.linalg.norm(finger)
        score = float(np.dot(finger, desired_finger_direction))
        if best is None or score > best[0]:
            best = (
                score,
                np.asarray(camera_position, dtype=float),
                world_from_camera,
                float(camera_psi_deg),
                finger,
            )
    if best is None:
        raise RuntimeError("无法建立 roll=0 腕部基准姿态")
    return best[1], best[2], best[3], best[4]


def _make_panel(rgb: Any, mask: Any, title: str, subtitle: str) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    rgb_array = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
    mask_array = np.asarray(mask, dtype=bool)
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
    thick[:-1] |= boundary[1:]
    thick[1:] |= boundary[:-1]
    thick[:, :-1] |= boundary[:, 1:]
    thick[:, 1:] |= boundary[:, :-1]
    rgb_array[thick] = (0, 255, 255)
    image = Image.fromarray(rgb_array, mode="RGB")
    draw = ImageDraw.Draw(image)
    bbox = _bbox_from_mask(mask_array)
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline=(255, 215, 0), width=3)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = ImageFont.truetype(str(font_path), 24) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 74), fill=(0, 0, 0))
    draw.text((15, 8), title, fill=(255, 255, 255), font=font)
    draw.text((15, 43), subtitle, fill=(220, 220, 220), font=small)
    return image


def _contact_sheet(panels: dict[tuple[str, str], Any], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    class_names = ("left", "right", "hose", "outhandle")
    column_names = ("r_max", "r_min")
    panel_width, panel_height = 640, 360
    header_height = 54
    row_label_width = 120
    sheet = Image.new(
        "RGB",
        (row_label_width + panel_width * 2, header_height + panel_height * 4),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = ImageFont.truetype(str(font_path), 24) if font_path.is_file() else ImageFont.load_default()
    for column, label in enumerate(("最大 r = 0.50 m", "最小 r = 0.12 m")):
        draw.text((row_label_width + column * panel_width + 15, 12), label, fill="black", font=font)
    for row, class_name in enumerate(class_names):
        draw.text((12, header_height + row * panel_height + 15), class_name, fill="black", font=font)
        for column, column_name in enumerate(column_names):
            panel = panels[(class_name, column_name)].resize(
                (panel_width, panel_height), Image.Resampling.LANCZOS
            )
            sheet.paste(panel, (row_label_width + column * panel_width, header_height + row * panel_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# 真实右腕相机最大/最小半径图像报告",
        "",
        f"状态：**{'完整通过' if result['status'] == 'pass' else '单张方向预览'}**",
        "",
        "## 方法",
        "",
        "- 使用实验室场景右侧灵巧手支架上的 `right_wrist_d405_rgb`，没有使用自由观察相机。",
        "- 为避免机器人 articulation 覆盖手部网格变换，采集时引用同一 `zarm_r7_link` 子树建立独立腕部刚体；几何和相机安装外参保持原值。",
        "- 不求解机械臂 IK；将 `zarm_r7_link`、灵巧手和腕部相机整体放置，使相机精确位于指定半径并指向 sampling center。",
        "- `roll=0°` 定义为：在保持光轴不变时，使四指方向最接近世界 `-Y` 的绕光轴角度。",
        "- 本轮按要求不检查机械臂、灵巧手或箱体碰撞，也不进行轨迹规划。",
        "- 工件采用阶段三得到的自然沉降姿态。",
        "- 青色为目标实例 Mask 轮廓，黄色为可见 Mask 包围框。",
        "",
        "## 图像指标",
        "",
        "| 类别 | 指定半径/m | 实际半径/m | 放置误差/m | 目标射线点积 | Mask 像素 | 可见 bbox 宽度比例 | Mask 深度有效率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["captures"]:
        lines.append(
            f"| {item['class']} | {item['requested_r_m']:.2f} | {item['actual_r_m']:.4f} | "
            f"{item['placement']['radius_error_m']:.6f} | {item['placement']['target_ray_dot']:.6f} | "
            f"{item['mask_pixels']} | {item['visible_bbox_width_ratio']:.3f} | "
            f"{item['depth_valid_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- 对比总图：`{result['outputs']['contact_sheet'] or '单张预览模式未生成'}`",
            f"- 元数据：`{result['outputs']['metadata']}`",
            f"- 单张图像目录：`{result['outputs']['image_dir']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assets = load_yaml(args.assets_config)
    camera_config = load_yaml(args.camera_config)["camera"]
    collection = load_yaml(args.collection_config)
    workspace = collection["workspace_calibration"]
    stable = json.loads(args.stable_poses.read_text(encoding="utf-8"))
    scene_path = resolve_from_config(assets, assets["scene"]["usd"])
    radius_max = float(workspace["radius_range_m"]["max"])
    radius_min = float(workspace["radius_range_m"]["min"])
    viewpoint = dict(workspace["radius_min_selection"]["baseline_viewpoint"])
    if args.theta_deg is not None:
        viewpoint["theta_deg"] = float(args.theta_deg)
    if args.phi_deg is not None:
        viewpoint["phi_deg"] = float(args.phi_deg)

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
    render_product = None
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Usd, UsdGeom

        from yolo_catchdata.asset_loader import add_reference_with_unit_scale, inspect_usd

        context = omni.usd.get_context()
        if not context.open_stage(str(scene_path)):
            raise RuntimeError(f"无法打开场景：{scene_path}")
        for _ in range(20):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        omni.timeline.get_timeline_interface().stop()
        if not stage.GetPrimAtPath(RIGHT_WRIST_RGB).IsValid():
            raise RuntimeError(f"缺少真实右腕 RGB 相机：{RIGHT_WRIST_RGB}")
        for path in RIGHT_ARM_LINKS:
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"缺少右臂运动链 Prim：{path}")

        rig = _prepare_rigid_wrist(stage, scene_path)
        hidden_upstream_arm_prims = _hide_detached_upstream_arm(stage)

        scene_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
        object_runtime: dict[str, dict[str, Any]] = {}
        selected_definitions = [
            definition
            for definition in assets["classes"]
            if args.class_name is None or definition["name"] == args.class_name
        ]
        selected_radii = (
            (radius_max, radius_min)
            if args.radius_key is None
            else ((radius_max,) if args.radius_key == "r_max" else (radius_min,))
        )
        for definition in selected_definitions:
            class_name = definition["name"]
            asset_path = resolve_from_config(assets, definition["usd"])
            inspection = inspect_usd(asset_path)
            root_path = f"/World/SyntheticData/WristRadiusExtremes/{class_name}"
            root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
            _set_transform(
                root,
                stable["classes"][class_name]["pose"]["settled_world_from_asset"],
            )
            asset_prim_path = f"{root_path}/Asset"
            asset_prim = add_reference_with_unit_scale(
                stage,
                asset_path,
                asset_prim_path,
                inspection["meters_per_unit"],
                scene_mpu,
            )
            _add_semantics(asset_prim, class_name)
            UsdGeom.Imageable(root).MakeInvisible()
            object_runtime[class_name] = {
                "root": root,
                "asset_prim_path": asset_prim_path,
                "sampling_center_world": np.asarray(
                    stable["classes"][class_name]["pose"]["sampling_center_world_m"],
                    dtype=float,
                ),
            }

        width, height = (int(value) for value in camera_config["resolution"])
        depth_min_m, depth_max_m = (
            float(value) for value in camera_config["clipping_range_m"]
        )
        render_product = rep.create.render_product(
            rig["camera_path"], (width, height), name="WristExtremes"
        )
        rgb_annotator = rep.annotators.get("rgb", device="cpu")
        instance_annotator = rep.annotators.get(
            "instance_segmentation_fast", init_params={"colorize": False}, device="cpu"
        )
        depth_annotator = rep.annotators.get("distance_to_image_plane", device="cpu")
        annotators = [rgb_annotator, instance_annotator, depth_annotator]
        for annotator in annotators:
            annotator.attach(render_product)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = args.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        panels: dict[tuple[str, str], Any] = {}
        captures = []
        for definition in selected_definitions:
            class_name = definition["name"]
            runtime = object_runtime[class_name]
            UsdGeom.Imageable(runtime["root"]).MakeVisible()
            for radius in selected_radii:
                if render_product is None:
                    render_product = rep.create.render_product(
                        rig["camera_path"], (width, height), name="WristExtremes"
                    )
                    rgb_annotator = rep.annotators.get("rgb", device="cpu")
                    instance_annotator = rep.annotators.get(
                        "instance_segmentation_fast",
                        init_params={"colorize": False},
                        device="cpu",
                    )
                    depth_annotator = rep.annotators.get(
                        "distance_to_image_plane", device="cpu"
                    )
                    annotators = [rgb_annotator, instance_annotator, depth_annotator]
                    for annotator in annotators:
                        annotator.attach(render_product)
                (
                    requested_camera_position,
                    world_from_camera,
                    camera_psi_at_roll_zero_deg,
                    finger_direction_world,
                ) = _camera_pose_with_hand_roll_zero(
                    rig,
                    runtime["sampling_center_world"],
                    assets["sampling_frame"]["rotation_world_from_sampling"],
                    radius,
                    float(viewpoint["theta_deg"]),
                    float(viewpoint["phi_deg"]),
                )
                _place_rigid_wrist(rig, world_from_camera)
                # 首次变换后先丢弃一帧，确保 Hydra/RTX 已消费独立腕部和相机的
                # 新世界变换。否则可能读到相机已移动、网格仍在旧位姿的混合帧。
                rep.orchestrator.step(rt_subframes=4)
                rep.orchestrator.step(rt_subframes=2)
                rgb = np.asarray(_extract_array(rgb_annotator.get_data()))
                instance_output = instance_annotator.get_data()
                instance = np.asarray(_extract_array(instance_output))
                depth = np.asarray(_extract_array(depth_annotator.get_data()), dtype=np.float32)
                target_ids = _target_instance_ids(
                    instance_output, class_name, runtime["asset_prim_path"]
                )
                mask = np.isin(instance, target_ids)
                bbox = _bbox_from_mask(mask)
                mask_pixels = int(mask.sum())
                valid_depth = (
                    mask
                    & np.isfinite(depth)
                    & (depth >= depth_min_m)
                    & (depth <= depth_max_m)
                )
                depth_valid_ratio = float(valid_depth.sum() / mask_pixels) if mask_pixels else 0.0
                bbox_width_ratio = (
                    float((bbox[2] - bbox[0] + 1) / width) if bbox is not None else 0.0
                )
                _, camera_position, camera_forward, camera_up, camera_right = _camera_pose_data(
                    stage, rig["camera_path"]
                )
                actual_r = float(np.linalg.norm(camera_position - runtime["sampling_center_world"]))
                actual_target_direction = runtime["sampling_center_world"] - camera_position
                actual_target_direction /= np.linalg.norm(actual_target_direction)
                target_ray_now = (
                    camera_forward
                    + TARGET_RAY_LOCAL_X * camera_right
                    + TARGET_RAY_LOCAL_Y * camera_up
                )
                target_ray_now /= np.linalg.norm(target_ray_now)
                target_ray_dot = float(np.dot(target_ray_now, actual_target_direction))
                position_error = float(np.linalg.norm(camera_position - requested_camera_position))
                radius_error = abs(actual_r - radius)
                if position_error > 1e-4 or radius_error > 1e-4 or target_ray_dot < 0.9999:
                    raise RuntimeError(
                        "刚体腕部放置发生漂移："
                        f"position_error={position_error:.6f}, radius_error={radius_error:.6f}, "
                        f"target_ray_dot={target_ray_dot:.6f}"
                    )
                key = "r_max" if radius == radius_max else "r_min"
                subtitle = (
                    f"实际 r={actual_r:.3f} m  Mask={mask_pixels}  "
                    f"bbox宽比={bbox_width_ratio:.3f}  深度有效={depth_valid_ratio:.3f}"
                )
                panel = _make_panel(
                    rgb,
                    mask,
                    f"真实右腕 D405｜{class_name}｜{key}｜roll=0°",
                    subtitle,
                )
                panel_path = image_dir / f"{class_name}_{key}_reference.png"
                raw_path = image_dir / f"{class_name}_{key}_rgb.png"
                mask_path = image_dir / f"{class_name}_{key}_mask.png"
                depth_path = image_dir / f"{class_name}_{key}_depth.npy"
                panel.save(panel_path)
                Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB").save(raw_path)
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
                np.save(depth_path, depth)
                panels[(class_name, key)] = panel
                captures.append(
                    {
                        "class": class_name,
                        "radius_key": key,
                        "requested_r_m": radius,
                        "actual_r_m": actual_r,
                        "optical_roll_deg": BASELINE_OPTICAL_ROLL_DEG,
                        "camera_psi_at_roll_zero_deg": camera_psi_at_roll_zero_deg,
                        "finger_direction_world": finger_direction_world.tolist(),
                        "sampling_center_world_m": runtime["sampling_center_world"].tolist(),
                        "camera_position_world_m": camera_position.tolist(),
                        "camera_forward_world": camera_forward.tolist(),
                        "reference_joint_positions_deg": list(REFERENCE_JOINTS_DEG),
                        "placement": {
                            "method": "detached_r7_wrist_rig_no_ik",
                            "position_error_m": position_error,
                            "radius_error_m": radius_error,
                            "target_ray_dot": target_ray_dot,
                        },
                        "mask_pixels": mask_pixels,
                        "visible_bbox_xyxy": bbox,
                        "visible_bbox_width_ratio": bbox_width_ratio,
                        "depth_valid_ratio": depth_valid_ratio,
                        "files": {
                            "reference": str(panel_path),
                            "rgb": str(raw_path),
                            "mask": str(mask_path),
                            "depth": str(depth_path),
                        },
                    }
                )
                print(
                    f"CAPTURE class={class_name} key={key} requested_r={radius:.2f} "
                    f"actual_r={actual_r:.4f} mask={mask_pixels} depth_valid={depth_valid_ratio:.3f}",
                    flush=True,
                )
                # Isaac Sim 6.0 中，移动带物理层级的相机后复用同一 SDG 图，
                # 第二帧可能丢失 DispatchSync。每张图独立释放采集通道可稳定规避。
                for annotator in annotators:
                    annotator.detach(render_product)
                render_product.destroy()
                annotators = []
                render_product = None
            UsdGeom.Imageable(runtime["root"]).MakeInvisible()

        full_run = args.class_name is None
        contact_sheet = (
            args.output_dir / "right_wrist_radius_extremes_contact_sheet.png"
            if full_run
            else None
        )
        if contact_sheet is not None:
            _contact_sheet(panels, contact_sheet)
        suffix = "" if full_run else "_preview"
        metadata_path = args.output_dir / f"right_wrist_radius_extremes{suffix}.json"
        report_path = args.output_dir / f"right_wrist_radius_extremes_report{suffix}.md"
        result = {
            "schema_version": 1,
            "artifact_kind": "right_wrist_radius_extremes",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if full_run else "preview",
            "scene": str(scene_path),
            "camera_prim": rig["camera_path"],
            "camera_source_prim": RIGHT_WRIST_RGB,
            "camera_is_attached_to_real_dexterous_hand": True,
            "wrist_orientation_source": str(
                PROJECT_ROOT.parent / "yolo_data/scripts/render_right_wrist_capture_pose.py"
            ),
            "placement_method": "detached_r7_wrist_rig_no_ik",
            "optical_roll_deg": BASELINE_OPTICAL_ROLL_DEG,
            "optical_axis_target": "workpiece_sampling_center",
            "roll_zero_definition": "four_finger_direction_closest_to_world_minus_y",
            "inverse_kinematics": False,
            "trajectory_planning": False,
            "collision_checking": False,
            "hidden_upstream_arm_prims": hidden_upstream_arm_prims,
            "hidden_original_wrist_prims": rig["hidden_original_wrist_prims"],
            "radius_range_m": {"min": radius_min, "max": radius_max},
            "viewpoint": viewpoint,
            "captures": captures,
            "outputs": {
                "contact_sheet": str(contact_sheet) if contact_sheet is not None else None,
                "metadata": str(metadata_path),
                "report": str(report_path),
                "image_dir": str(image_dir),
            },
        }
        metadata_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_report(result, report_path)
        if contact_sheet is not None:
            print(f"CONTACT_SHEET={contact_sheet}", flush=True)
        print(f"REPORT={report_path}", flush=True)
        print(
            f"STATUS={'PASS' if full_run else 'PREVIEW'} CAPTURES={len(captures)}",
            flush=True,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
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
