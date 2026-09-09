#!/usr/bin/env python3
"""Open the corrected S63 scene for manual wrist-camera pose inspection.

The script places all four workpieces (left/right/hose/outhandle) as independent
manual Xforms and creates an independent copy of the right r7 wrist subtree
(dexterous hand + D405 mount).
The copy is intentionally detached from the S63 articulation so the user can
translate/rotate it directly with the Isaac Sim transform gizmo without PhysX
or the robot joint drives snapping it back.

Normal workflow in Isaac Sim:
  1. Click ``Select Wrist`` in the Manual Wrist Control window.
  2. Press W to translate or E to rotate the selected wrist assembly.
  3. Click ``D405 View`` to inspect the actual wrist-camera image.
  4. Click ``Overview`` to return to the external inspection camera.
  5. Use the four workpiece buttons to select and move each part independently.

Selecting ``Select Camera Mount`` moves the camera/bracket relative to the hand.
Use that only when intentionally checking/changing the hand-eye mounting transform.
All runtime edits are authored in the anonymous Session Layer; source USD files
are never saved by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.hand_sides import hand_spec

DEFAULT_SCENE = PROJECT_ROOT / "yolo_scene/lab_yolo_grab_data_s63_mount_corrected.usd"
STABLE_POSES = PROJECT_ROOT / "configs/stable_poses.json"
DEFAULT_LIVE_POSE = PROJECT_ROOT / "output/manual_wrist/live_pose.json"
WORKPIECE_ASSETS = {
    "left": PROJECT_ROOT / "yolo_scene/left.usd",
    "right": PROJECT_ROOT / "yolo_scene/right.usd",
    "hose": PROJECT_ROOT / "yolo_scene/hose.usd",
    "outhandle": PROJECT_ROOT / "yolo_scene/outhandle.usd",
}
# Manual staging offsets relative to each class's measured stable pose. They are
# intentionally separated so all four parts are visible and independently draggable.
WORKPIECE_OFFSETS_M = {
    "left": (-0.52, -0.34, 0.24),
    "right": (0.00, -0.34, 0.24),
    "hose": (-0.52, 0.32, 0.24),
    "outhandle": (0.00, 0.32, 0.24),
}

WAIST_PATH = (
    "/World/Robot/Geometry/base_link/base_to_joint/knee_link_1/leg_link/"
    "waist_link/waist_yaw_link"
)
RIGHT_ARM_LINKS = tuple(
    WAIST_PATH + "".join(f"/zarm_r{j}_link" for j in range(1, index + 1))
    for index in range(1, 8)
)
LEFT_ARM_LINKS = tuple(
    WAIST_PATH + "".join(f"/zarm_l{j}_link" for j in range(1, index + 1))
    for index in range(1, 8)
)
RIGHT_R7 = RIGHT_ARM_LINKS[-1]
LEFT_R7 = LEFT_ARM_LINKS[-1]
JOINT_AXES = ("Y", "X", "Z", "Y", "Z", "X", "Y")

# Start from a wrist-visible pose.  The hand becomes independent immediately
# afterwards, so these angles only determine the initial location of the
# detached wrist and the visible upstream arm.
START_RIGHT_ARM_DEG = (-88.385, -13.870, 85.0, -112.054, 8.894, 75.0, 0.0)

RIGHT_HAND = hand_spec("right")
LEFT_HAND = hand_spec("left")
RIGHT_MANUAL_ROOT = str(RIGHT_HAND["manual_root"])
RIGHT_CAMERA_LINK = str(RIGHT_HAND["camera_link"])
RIGHT_D405 = str(RIGHT_HAND["rgb_camera"])
LEFT_MANUAL_ROOT = str(LEFT_HAND["manual_root"])
LEFT_CAMERA_LINK = str(LEFT_HAND["camera_link"])
LEFT_D405 = str(LEFT_HAND["rgb_camera"])
RIGHT_FINGER_JOINTS = RIGHT_HAND["finger_joints"]
LEFT_FINGER_JOINTS = LEFT_HAND["finger_joints"]
WORKPIECE_ROOTS = {
    name: f"/World/ManualInspection/Workpieces/{name}" for name in WORKPIECE_ASSETS
}
WORKPIECE_INSTANCES = {
    name: root + "/Asset" for name, root in WORKPIECE_ROOTS.items()
}
OVERVIEW_CAMERA = "/World/ManualInspection/OverviewCamera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--pose-output", type=Path, default=DEFAULT_LIVE_POSE)
    parser.add_argument(
        "--restore-pose-from",
        type=Path,
        help="启动时恢复上一次导出的左右腕与外层工件位姿。",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--exit-after-setup",
        action="store_true",
        help="Initialize, validate the manual scene, print poses, then exit.",
    )
    args, _ = parser.parse_known_args()
    args.scene = args.scene.expanduser().resolve()
    args.pose_output = args.pose_output.expanduser().resolve()
    if args.restore_pose_from is not None:
        args.restore_pose_from = args.restore_pose_from.expanduser().resolve()
    return args


def _set_right_arm_pose(stage: Any) -> None:
    from pxr import Gf, UsdGeom

    axes = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    for path, axis, degrees in zip(RIGHT_ARM_LINKS, JOINT_AXES, START_RIGHT_ARM_DEG):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing right-arm link: {path}")
        orient_ops = [
            op
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient
        ]
        if len(orient_ops) != 1:
            raise RuntimeError(f"Expected one orient op on {path}, got {len(orient_ops)}")
        quat = Gf.Rotation(axes[axis], float(degrees)).GetQuat()
        orient_ops[0].Set(
            Gf.Quatf(
                float(quat.GetReal()),
                Gf.Vec3f(*(float(v) for v in quat.GetImaginary())),
            )
        )


def _spawn_workpieces(stage: Any) -> dict[str, list[float]]:
    from pxr import Sdf, UsdGeom

    from yolo_catchdata.isaac_utils import set_transform as _set_transform
    from yolo_catchdata.asset_loader import add_reference_with_unit_scale, inspect_usd

    stable_classes = json.loads(STABLE_POSES.read_text(encoding="utf-8"))["classes"]
    centers: dict[str, list[float]] = {}
    for name, asset_path in WORKPIECE_ASSETS.items():
        stable = stable_classes[name]
        world_from_asset = [list(row) for row in stable["pose"]["settled_world_from_asset"]]
        offset = WORKPIECE_OFFSETS_M[name]
        for axis in range(3):
            world_from_asset[axis][3] += float(offset[axis])

        root_path = WORKPIECE_ROOTS[name]
        instance_path = WORKPIECE_INSTANCES[name]
        root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        _set_transform(root, world_from_asset)

        inspection = inspect_usd(asset_path)
        asset = add_reference_with_unit_scale(
            stage,
            asset_path,
            instance_path,
            float(inspection["meters_per_unit"]),
            float(UsdGeom.GetStageMetersPerUnit(stage)),
        )
        # 禁止在视口中误选并单独移动引用内部的 Mesh。用户点击模型时最多
        # 选中 Asset 实例根，手—物相对位姿导出不会再出现“原点没动、网格动了”。
        asset.SetInstanceable(True)
        asset.CreateAttribute(
            "kuavo:manualInspectionWorkpiece", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)
        asset.CreateAttribute(
            "kuavo:workpieceClass", Sdf.ValueTypeNames.String, custom=True
        ).Set(name)

        stable_center = [float(v) for v in stable["pose"]["sampling_center_world_m"]]
        centers[name] = [stable_center[i] + float(offset[i]) for i in range(3)]
    return centers


def _copy_material_bindings(stage: Any, source_root: Any, target_root_path: str) -> int:
    from pxr import Usd, UsdGeom, UsdShade

    copied = 0
    source_root_path = str(source_root.GetPath())
    for source in Usd.PrimRange(source_root):
        if not source.IsA(UsdGeom.Gprim):
            continue
        suffix = str(source.GetPath())[len(source_root_path) :]
        target = stage.GetPrimAtPath(target_root_path + suffix)
        if not target.IsValid():
            continue
        material, _ = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial()
        if not material or not material.GetPrim().IsValid():
            continue
        UsdShade.MaterialBindingAPI.Apply(target).Bind(material)
        copied += 1
    return copied


def _create_detached_wrist(stage: Any, scene_path: Path, side: str) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom

    if side == "right":
        source_path = RIGHT_R7
        target_root = RIGHT_MANUAL_ROOT
        camera_link = RIGHT_CAMERA_LINK
        d405 = RIGHT_D405
    elif side == "left":
        source_path = LEFT_R7
        target_root = LEFT_MANUAL_ROOT
        camera_link = LEFT_CAMERA_LINK
        d405 = LEFT_D405
    else:
        raise ValueError(f"Unsupported wrist side: {side}")

    source_r7 = stage.GetPrimAtPath(source_path)
    if not source_r7.IsValid():
        raise RuntimeError(f"Missing source wrist: {source_path}")

    source_world = UsdGeom.XformCache().GetLocalToWorldTransform(source_r7)
    root = UsdGeom.Xform.Define(stage, target_root).GetPrim()
    root.GetReferences().AddReference(str(scene_path), source_path)

    disabled = 0
    for prim in Usd.PrimRange(root):
        for attribute_name in (
            "physics:rigidBodyEnabled",
            "physics:collisionEnabled",
            "physics:articulationEnabled",
        ):
            prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Bool).Set(False)
        disabled += 1

    xformable = UsdGeom.Xformable(root)
    xformable.ClearXformOpOrder()
    pose_op = xformable.AddTransformOp(
        UsdGeom.XformOp.PrecisionDouble,
        "manualPose",
    )
    pose_op.Set(Gf.Matrix4d(source_world))
    xformable.SetResetXformStack(True)

    copied_materials = _copy_material_bindings(stage, source_r7, target_root)

    for required in (camera_link, d405):
        if not stage.GetPrimAtPath(required).IsValid():
            raise RuntimeError(f"Detached {side} wrist is missing required prim: {required}")

    root.CreateAttribute("kuavo:manualInspectionDetached", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    root.CreateAttribute("kuavo:sourceR7", Sdf.ValueTypeNames.String, custom=True).Set(source_path)
    return {
        "root": root,
        "pose_op": pose_op,
        "copied_material_bindings": copied_materials,
        "disabled_prim_count": disabled,
    }


def _hide_robot(stage: Any) -> None:
    from pxr import Sdf, Usd, UsdGeom

    robot = stage.GetPrimAtPath("/World/Robot")
    if not robot.IsValid():
        raise RuntimeError("Missing /World/Robot")

    # Keep the source prim hierarchy active because detached hand material
    # bindings can still resolve to Material prims authored under /World/Robot.
    # Making the robot inactive would invalidate those materials and turn the
    # detached hands back to default gray/white.  Instead, hide all rendering
    # and disable all relevant physics in the Session Layer only.
    UsdGeom.Imageable(robot).MakeInvisible()
    for prim in Usd.PrimRange(robot):
        for attribute_name in (
            "physics:rigidBodyEnabled",
            "physics:collisionEnabled",
            "physics:articulationEnabled",
        ):
            prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Bool).Set(False)


def _create_overview_camera(stage: Any, target: list[float]) -> None:
    from pxr import Gf, UsdGeom

    from yolo_catchdata.isaac_utils import set_transform as _set_transform
    from yolo_catchdata.camera_geometry import look_at_world_from_usd_camera

    camera = UsdGeom.Camera.Define(stage, OVERVIEW_CAMERA)
    # Outside the right side of the bin, high enough to see hand, camera mount,
    # workpiece, and the end of the real arm at the same time.
    eye = (1.10, 0.58, 1.62)
    world_from_camera = look_at_world_from_usd_camera(eye, target)
    _set_transform(camera.GetPrim(), world_from_camera)
    camera.CreateFocalLengthAttr().Set(28.0)
    camera.CreateHorizontalApertureAttr().Set(36.0)
    camera.CreateVerticalApertureAttr().Set(22.5)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))


def _prepare_finger_control(
    stage: Any, hand_side: str = "right"
) -> dict[str, dict[str, Any]]:
    from pxr import UsdGeom

    joints = hand_spec(hand_side)["finger_joints"]
    state: dict[str, dict[str, Any]] = {}
    for name, config in joints.items():
        path = str(config["path"])
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing detached finger body: {path}")
        orient_ops = [
            op
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient
        ]
        if len(orient_ops) != 1:
            raise RuntimeError(f"Expected one orient op on {path}, got {len(orient_ops)}")
        state[name] = {
            **config,
            "hand_side": hand_side,
            "orient_op": orient_ops[0],
            "rest_quat": orient_ops[0].Get(),
            "angle_deg": 0.0,
        }
    return state


def _set_finger_angle(finger_state: dict[str, dict[str, Any]], name: str, angle_deg: float) -> None:
    from pxr import Gf

    joint = finger_state[name]
    angle = max(float(joint["min_deg"]), min(float(joint["max_deg"]), float(angle_deg)))
    axis = Gf.Vec3d(*(float(v) for v in joint["axis"]))
    delta_d = Gf.Rotation(axis, angle).GetQuat()
    rest = joint["rest_quat"]

    if isinstance(rest, Gf.Quatf):
        delta = Gf.Quatf(
            float(delta_d.GetReal()),
            Gf.Vec3f(*(float(v) for v in delta_d.GetImaginary())),
        )
    elif isinstance(rest, Gf.Quatd):
        delta = Gf.Quatd(
            float(delta_d.GetReal()),
            Gf.Vec3d(*(float(v) for v in delta_d.GetImaginary())),
        )
    else:
        delta = type(rest)(
            float(delta_d.GetReal()),
            type(rest.GetImaginary())(*(float(v) for v in delta_d.GetImaginary())),
        )

    # MJCF hinge axes are local to the body, so post-multiply the rest orientation.
    joint["orient_op"].Set(rest * delta)
    joint["angle_deg"] = angle
    hand_side = str(joint.get("hand_side", "right")).upper()
    print(f"{hand_side}_{name}_DEG={angle:.2f}", flush=True)


def _world_pose(stage: Any, path: str) -> dict[str, Any]:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid pose path: {path}")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    # Gf.Transform 会先分解缩放/剪切。Asset 节点通常带 0.001 单位缩放，
    # 直接对原矩阵调用 ExtractRotationQuat() 会产生非单位且错误的四元数。
    decomposed = Gf.Transform(matrix)
    translation = decomposed.GetTranslation()
    quat = decomposed.GetRotation().GetQuat()
    return {
        "path": path,
        "translation_m": [float(v) for v in translation],
        "quaternion_wxyz": [
            float(quat.GetReal()),
            *(float(v) for v in quat.GetImaginary()),
        ],
    }


def _world_matrix_row_major(stage: Any, path: str) -> list[list[float]]:
    """保留 Gf 的完整世界矩阵，避免丢失 Asset 节点上的缩放信息。"""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid matrix path: {path}")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _workpiece_world_bounds(stage: Any) -> dict[str, dict[str, Any]]:
    """导出实际可渲染 Asset 的世界包围盒，而不是仅记录包装根节点。"""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    result: dict[str, dict[str, Any]] = {}
    for name, path in WORKPIECE_INSTANCES.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid workpiece Asset path: {path}")
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = [float(value) for value in aligned.GetMin()]
        maximum = [float(value) for value in aligned.GetMax()]
        result[name] = {
            "path": path,
            "min_m": minimum,
            "max_m": maximum,
            "center_m": [(minimum[i] + maximum[i]) * 0.5 for i in range(3)],
            "size_m": [maximum[i] - minimum[i] for i in range(3)],
        }
    return result


def _relative_pose(stage: Any, parent_path: str, child_path: str) -> dict[str, Any]:
    """返回 child 在 parent 坐标系下的位姿。"""
    from pxr import Gf, UsdGeom

    cache = UsdGeom.XformCache()
    parent_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
    child_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_path))
    # Gf.Matrix4d 使用行向量约定，因此 child->world->parent 的顺序如下。
    parent_from_child = child_world * parent_world.GetInverse()
    decomposed = Gf.Transform(parent_from_child)
    translation = decomposed.GetTranslation()
    quat = decomposed.GetRotation().GetQuat()
    return {
        "parent_path": parent_path,
        "child_path": child_path,
        "translation_m": [float(v) for v in translation],
        "quaternion_wxyz": [
            float(quat.GetReal()),
            *(float(v) for v in quat.GetImaginary()),
        ],
    }


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _column_transform_from_pose(entry: dict[str, Any]) -> list[list[float]]:
    """把导出的 wxyz 位姿恢复成列向量齐次矩阵。"""
    w, x, y, z = (float(value) for value in entry["quaternion_wxyz"])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("姿态四元数长度为零")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    tx, ty, tz = (float(value) for value in entry["translation_m"])
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), tx],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), ty],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _restore_exported_pose(
    stage: Any,
    payload: dict[str, Any],
    finger_state: dict[str, dict[str, Any]],
) -> None:
    """恢复可安全重建的外层 Prim；不尝试恢复旧会话中的内部 Mesh 误编辑。"""
    from yolo_catchdata.isaac_utils import set_transform as _set_transform

    for path, key in (
        (RIGHT_MANUAL_ROOT, "right_wrist"),
        (LEFT_MANUAL_ROOT, "left_wrist"),
    ):
        if key in payload:
            _set_transform(stage.GetPrimAtPath(path), _column_transform_from_pose(payload[key]))
    for name, path in WORKPIECE_ROOTS.items():
        entry = payload.get("workpieces", {}).get(name)
        if entry is not None:
            _set_transform(stage.GetPrimAtPath(path), _column_transform_from_pose(entry))
    for name, angle in payload.get("right_finger_joint_deg", {}).items():
        if name in finger_state:
            _set_finger_angle(finger_state, name, float(angle))


def _pose_payload(
    stage: Any,
    finger_state: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    right_palm_path = RIGHT_MANUAL_ROOT + "/r_palm"
    left_palm_path = LEFT_MANUAL_ROOT + "/l_palm"
    right_palm = _world_pose(stage, right_palm_path)
    left_palm = _world_pose(stage, left_palm_path)
    right_camera = _world_pose(stage, RIGHT_D405)
    left_camera = _world_pose(stage, LEFT_D405)
    workpieces = {
        name: _world_pose(stage, path) for name, path in WORKPIECE_ROOTS.items()
    }
    workpiece_assets = {
        name: {
            **_world_pose(stage, path),
            "world_matrix_row_major": _world_matrix_row_major(stage, path),
        }
        for name, path in WORKPIECE_INSTANCES.items()
    }
    workpiece_bounds = _workpiece_world_bounds(stage)
    workpiece_pose_integrity = {}
    for name, workpiece in workpiece_assets.items():
        center = workpiece_bounds[name]["center_m"]
        size = workpiece_bounds[name]["size_m"]
        origin_to_center = _distance(workpiece["translation_m"], center)
        bbox_diagonal = math.sqrt(sum(float(value) ** 2 for value in size))
        # 正常 CAD 原点可以不在几何中心，但不应离实际网格数个包围盒之外。
        valid = origin_to_center <= max(0.02, 2.0 * bbox_diagonal)
        workpiece_pose_integrity[name] = {
            "asset_origin_to_bbox_center_m": origin_to_center,
            "bbox_diagonal_m": bbox_diagonal,
            "valid": valid,
        }
    radii = {"right": {}, "left": {}}
    camera_distances = {"right": {}, "left": {}}
    palm_from_workpiece = {"right": {}, "left": {}}
    palm_from_workpiece_asset = {"right": {}, "left": {}}
    for name, workpiece in workpiece_assets.items():
        sampling_center = workpiece["translation_m"]
        radii["right"][name] = _distance(
            right_palm["translation_m"], sampling_center
        )
        radii["left"][name] = _distance(
            left_palm["translation_m"], sampling_center
        )
        camera_distances["right"][name] = _distance(
            right_camera["translation_m"], sampling_center
        )
        camera_distances["left"][name] = _distance(
            left_camera["translation_m"], sampling_center
        )
        palm_from_workpiece["right"][name] = _relative_pose(
            stage, right_palm_path, WORKPIECE_ROOTS[name]
        )
        palm_from_workpiece["left"][name] = _relative_pose(
            stage, left_palm_path, WORKPIECE_ROOTS[name]
        )
        palm_from_workpiece_asset["right"][name] = _relative_pose(
            stage, right_palm_path, WORKPIECE_INSTANCES[name]
        )
        palm_from_workpiece_asset["left"][name] = _relative_pose(
            stage, left_palm_path, WORKPIECE_INSTANCES[name]
        )

    nearest = {}
    for side in ("right", "left"):
        class_name = min(radii[side], key=radii[side].get)
        nearest[side] = {
            "class": class_name,
            "palm_to_sampling_center_r_m": radii[side][class_name],
            "camera_to_sampling_center_distance_m": camera_distances[side][class_name],
        }

    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "manual_wrist_inspection_session_layer",
        "radius_definition": "euclidean_distance_from_palm_prim_origin_to_workpiece_asset_origin",
        "sampling_center_definition": "workpiece_asset_origin",
        "right_wrist": _world_pose(stage, RIGHT_MANUAL_ROOT),
        "right_palm_reference": right_palm,
        "right_camera_link": _world_pose(stage, RIGHT_CAMERA_LINK),
        "right_d405_rgb": right_camera,
        "left_wrist": _world_pose(stage, LEFT_MANUAL_ROOT),
        "left_palm_reference": left_palm,
        "left_camera_link": _world_pose(stage, LEFT_CAMERA_LINK),
        "left_d405_rgb": left_camera,
        "workpieces": workpieces,
        "workpiece_assets": workpiece_assets,
        "workpiece_world_bounds": workpiece_bounds,
        "workpiece_pose_integrity": workpiece_pose_integrity,
        "palm_to_sampling_center_r_m": radii,
        "camera_to_sampling_center_distance_m": camera_distances,
        "palm_from_workpiece": palm_from_workpiece,
        "palm_from_workpiece_asset": palm_from_workpiece_asset,
        "nearest_workpiece_by_hand": nearest,
    }
    if finger_state is not None:
        payload["right_finger_joint_deg"] = {
            name: float(joint["angle_deg"]) for name, joint in finger_state.items()
        }
    try:
        import omni.usd

        payload["selected_prim_paths"] = list(
            omni.usd.get_context().get_selection().get_selected_prim_paths()
        )
    except Exception:
        payload["selected_prim_paths"] = []
    return payload


def _write_pose_file(
    stage: Any,
    path: Path,
    finger_state: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """原子更新实时姿态，避免外部读取到半个 JSON。"""
    payload = _pose_payload(stage, finger_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


def _print_pose_report(stage: Any, finger_state: dict[str, dict[str, Any]] | None = None) -> None:
    payload = _pose_payload(stage, finger_state)
    print("\n========== CURRENT MANUAL POSE ==========")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=========================================\n", flush=True)


def _make_control_window(
    stage: Any,
    viewport: Any,
    finger_state: dict[str, dict[str, Any]],
    pose_output: Path,
) -> Any:
    import omni.ui as ui
    import omni.usd

    selection = omni.usd.get_context().get_selection()

    def select(path: str) -> None:
        selection.set_selected_prim_paths([path], True)
        print(f"SELECTED={path}", flush=True)

    def set_view(path: str) -> None:
        viewport.camera_path = path
        print(f"VIEWPORT_CAMERA={path}", flush=True)

    finger_models: dict[str, Any] = {}
    for name, joint in finger_state.items():
        model = ui.SimpleFloatModel(float(joint["angle_deg"]))
        model.add_value_changed_fn(
            lambda value_model, joint_name=name: _set_finger_angle(
                finger_state, joint_name, value_model.as_float
            )
        )
        finger_models[name] = model

    def reset_fingers() -> None:
        for model in finger_models.values():
            model.set_value(0.0)

    def joint_row(label: str, joint_name: str) -> None:
        joint = finger_state[joint_name]
        with ui.HStack(height=30, spacing=6):
            ui.Label(label, width=126)
            ui.FloatSlider(
                model=finger_models[joint_name],
                min=float(joint["min_deg"]),
                max=float(joint["max_deg"]),
            )
            ui.FloatField(model=finger_models[joint_name], width=68)

    window = ui.Window("Manual Wrist Control", width=470, height=930)
    with window.frame:
        with ui.VStack(spacing=7, height=0):
            ui.Label("Two detached hands + wrist D405 cameras", height=24)
            ui.Label("W = Translate, E = Rotate", height=22)
            with ui.HStack(height=34, spacing=6):
                ui.Button("Select Left Wrist", clicked_fn=lambda: select(LEFT_MANUAL_ROOT))
                ui.Button("Select Right Wrist", clicked_fn=lambda: select(RIGHT_MANUAL_ROOT))
            with ui.HStack(height=34, spacing=6):
                ui.Button("Left D405", clicked_fn=lambda: set_view(LEFT_D405))
                ui.Button("Right D405", clicked_fn=lambda: set_view(RIGHT_D405))
                ui.Button("Overview", clicked_fn=lambda: set_view(OVERVIEW_CAMERA))

            ui.Separator(height=6)
            ui.Label("Workpieces - select then use W/E", height=24)
            with ui.HStack(height=34, spacing=6):
                ui.Button("Select LEFT", clicked_fn=lambda: select(WORKPIECE_ROOTS["left"]))
                ui.Button("Select RIGHT", clicked_fn=lambda: select(WORKPIECE_ROOTS["right"]))
            with ui.HStack(height=34, spacing=6):
                ui.Button("Select HOSE", clicked_fn=lambda: select(WORKPIECE_ROOTS["hose"]))
                ui.Button("Select OUTHANDLE", clicked_fn=lambda: select(WORKPIECE_ROOTS["outhandle"]))

            ui.Separator(height=6)
            ui.Label("Right Thumb", height=22)
            joint_row("Thumb CMC 0-90", "THUMB_CMC")
            joint_row("Thumb MCP 0-50", "THUMB_MCP")

            ui.Label("Right Index", height=22)
            joint_row("Index MCP 0-75", "INDEX_MCP")
            joint_row("Index PIP 0-120", "INDEX_PIP")

            ui.Label("Right Middle", height=22)
            joint_row("Middle MCP 0-75", "MIDDLE_MCP")
            joint_row("Middle PIP 0-120", "MIDDLE_PIP")

            ui.Label("Right Ring", height=22)
            joint_row("Ring MCP 0-75", "RING_MCP")
            joint_row("Ring PIP 0-120", "RING_PIP")

            ui.Label("Right Little", height=22)
            joint_row("Little MCP 0-75", "LITTLE_MCP")
            joint_row("Little PIP 0-120", "LITTLE_PIP")

            with ui.HStack(height=34, spacing=6):
                ui.Button("Reset Fingers", clicked_fn=reset_fingers)
                ui.Button(
                    "Print Poses",
                    clicked_fn=lambda: _print_pose_report(stage, finger_state),
                )
                ui.Button(
                    "Export Pose",
                    clicked_fn=lambda: _write_pose_file(
                        stage, pose_output, finger_state
                    ),
                )
            ui.Separator(height=6)
            ui.Label(
                "Robot body is hidden. Move Left/Right Wrist to tune hand-camera poses.",
                word_wrap=True,
                height=42,
            )
    return window


def main() -> int:
    args = parse_args()
    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    restore_payload = None
    if args.restore_pose_from is not None:
        if not args.restore_pose_from.is_file():
            raise FileNotFoundError(args.restore_pose_from)
        restore_payload = json.loads(args.restore_pose_from.read_text(encoding="utf-8"))

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": bool(args.headless),
            "fast_shutdown": True,
            "multi_gpu": False,
            "width": 1440,
            "height": 900,
        }
    )
    try:
        import omni.usd
        from pxr import Usd

        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene)):
            raise RuntimeError(f"Could not open corrected scene: {args.scene}")
        for _ in range(20):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Could not obtain opened USD stage")
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))

        _set_right_arm_pose(stage)
        for _ in range(3):
            app.update()

        workpiece_centers = _spawn_workpieces(stage)
        right_wrist_info = _create_detached_wrist(stage, args.scene, "right")
        left_wrist_info = _create_detached_wrist(stage, args.scene, "left")
        finger_state = _prepare_finger_control(stage)
        if restore_payload is not None:
            _restore_exported_pose(stage, restore_payload, finger_state)
        _hide_robot(stage)
        overview_target = [
            sum(center[axis] for center in workpiece_centers.values()) / len(workpiece_centers)
            for axis in range(3)
        ]
        _create_overview_camera(stage, overview_target)
        for _ in range(8):
            app.update()

        if not args.headless:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("No active Isaac Sim viewport")
            viewport.camera_path = OVERVIEW_CAMERA
            control_window = _make_control_window(
                stage, viewport, finger_state, args.pose_output
            )
            # Keep the object alive for the entire application lifetime.
            _ = control_window
            omni.usd.get_context().get_selection().set_selected_prim_paths([RIGHT_MANUAL_ROOT], True)

        print("\n========== MANUAL WRIST MODE READY ==========")
        print(f"SCENE={args.scene}")
        for name, path in WORKPIECE_INSTANCES.items():
            print(f"WORKPIECE_{name.upper()}={path}")
        print(f"RIGHT_WRIST_ROOT={RIGHT_MANUAL_ROOT}")
        print(f"RIGHT_D405={RIGHT_D405}")
        print(f"LEFT_WRIST_ROOT={LEFT_MANUAL_ROOT}")
        print(f"LEFT_D405={LEFT_D405}")
        print("RIGHT_THUMB_CMC_RANGE_DEG=0..90")
        print("RIGHT_THUMB_MCP_RANGE_DEG=0..50")
        print("RIGHT_INDEX_MCP_RANGE_DEG=0..75")
        print("RIGHT_INDEX_PIP_RANGE_DEG=0..120")
        print("RIGHT_MIDDLE_MCP_RANGE_DEG=0..75")
        print("RIGHT_MIDDLE_PIP_RANGE_DEG=0..120")
        print("RIGHT_RING_MCP_RANGE_DEG=0..75")
        print("RIGHT_RING_PIP_RANGE_DEG=0..120")
        print("RIGHT_LITTLE_MCP_RANGE_DEG=0..75")
        print("RIGHT_LITTLE_PIP_RANGE_DEG=0..120")
        print(f"OVERVIEW_CAMERA={OVERVIEW_CAMERA}")
        print(f"LIVE_POSE_FILE={args.pose_output}")
        print(f"RIGHT_COPIED_MATERIAL_BINDINGS={right_wrist_info['copied_material_bindings']}")
        print(f"LEFT_COPIED_MATERIAL_BINDINGS={left_wrist_info['copied_material_bindings']}")
        print("ROBOT_VISIBLE=0")
        print("SOURCE_USD_SAVED=0")
        print("Use W=translate, E=rotate. Close Isaac Sim to exit.")
        print("=============================================\n", flush=True)
        _print_pose_report(stage, finger_state)
        _write_pose_file(stage, args.pose_output, finger_state)

        if args.exit_after_setup:
            return 0

        last_pose_export = time.monotonic()
        while app.is_running():
            app.update()
            now = time.monotonic()
            if now - last_pose_export >= 0.2:
                _write_pose_file(stage, args.pose_output, finger_state)
                last_pose_export = now
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
