#!/usr/bin/env python3
"""Render one settled object with one selected detached wrist-camera branch.

This is the bridge between support-face validation and the eventual batch
collector.  It deliberately renders one sample per process so a failed GPU or
invalid settled pose cannot contaminate a larger dataset run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
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
    gf_matrix_from_column_transform as _gf_matrix_from_column_transform,
    json_safe as _json_safe,
    set_transform as _set_transform,
    target_instance_ids as _target_instance_ids,
)
from scripts.manual_adjust_left_wrist import (  # noqa: E402
    WORKPIECE_ASSETS,
    _create_detached_wrist,
    _prepare_finger_control,
    _set_finger_angle,
)
from yolo_catchdata.camera_geometry import (  # noqa: E402
    camera_from_object,
)
from yolo_catchdata.approach_quality import (  # noqa: E402
    evaluate_approach_quality,
)
from yolo_catchdata.approach_references import (  # noqa: E402
    ensure_mirrored_left_references,
)
from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402
from yolo_catchdata.hand_randomization import apply_hand_perturbation  # noqa: E402
from yolo_catchdata.hand_sides import (  # noqa: E402
    HAND_SIDES,
    hand_spec,
    reference_filename,
)
from yolo_catchdata.mesh_visibility import (  # noqa: E402
    expanded_fov_degrees,
    rendered_mesh_visibility,
    scene_visible_fraction_from_reference,
)
from yolo_catchdata.pose_policy import (  # noqa: E402
    effective_hand_yaw_delta,
    settled_effective_hand_yaw_delta,
    transport_reference_pose,
)


def _prepare_palm_rig(
    stage: Any, scene_path: Path, hand_side: str
) -> dict[str, Any]:
    from pxr import UsdGeom

    hand = hand_spec(hand_side)
    rig = _create_detached_wrist(stage, scene_path, hand_side)
    cache = UsdGeom.XformCache()
    root_world = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(str(hand["manual_root"]))
    )
    palm_world = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(str(hand["palm"]))
    )
    rig["palm_from_root"] = palm_world * root_world.GetInverse()
    return rig


def _place_palm(rig: dict[str, Any], desired_world_from_palm_column: Any) -> None:
    desired_palm_world = _gf_matrix_from_column_transform(desired_world_from_palm_column)
    desired_root_world = rig["palm_from_root"].GetInverse() * desired_palm_world
    rig["pose_op"].Set(desired_root_world)


def _set_saved_finger_pose(
    stage: Any,
    hand_side: str,
    saved_joint_deg: dict[str, Any],
) -> None:
    """Apply the saved open-hand state using side-correct joint axes."""

    finger_state = _prepare_finger_control(stage, hand_side)
    defaults = {"THUMB_CMC": 90.0, "THUMB_MCP": 1.4}
    for name in finger_state:
        _set_finger_angle(
            finger_state,
            name,
            float(saved_joint_deg.get(name, defaults.get(name, 0.0))),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-shard", type=Path, required=True)
    parser.add_argument("--hand-side", choices=HAND_SIDES, default="right")
    parser.add_argument("--class-name", choices=tuple(WORKPIECE_ASSETS), required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / "output/manual_wrist/references",
    )
    parser.add_argument(
        "--object-yaw-delta-deg",
        type=float,
        default=0.0,
        help="相对该类别 grasp 参考的物体平面 yaw；outhandle 的手不会跟随。",
    )
    parser.add_argument(
        "--distance-from-grasp-m",
        type=float,
        default=0.10,
        help="手掌距 grasp 的接近距离；0=grasp，0.10=保存的 pregrasp。",
    )
    parser.add_argument("--lateral-offset-1-m", type=float, default=0.0)
    parser.add_argument("--lateral-offset-2-m", type=float, default=0.0)
    parser.add_argument("--roll-delta-deg", type=float, default=0.0)
    parser.add_argument("--pitch-delta-deg", type=float, default=0.0)
    parser.add_argument(
        "--palm-yaw-delta-deg",
        type=float,
        default=0.0,
        help="相对 transported palm 参考姿态的局部 yaw 扰动；不是工件 yaw。",
    )
    parser.add_argument("--sample-id", default="smoke")
    parser.add_argument(
        "--distance-stage",
        choices=("grasp", "pre_grasp", "near", "approach", "far"),
        default=None,
        help="采样计划中的距离阶段，写入元数据便于按阶段统计。",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output/dataset/approach_open_smoke",
    )
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera.yaml")
    parser.add_argument(
        "--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml"
    )
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Isaac Sim renderer/PhysX 使用的物理 GPU 编号。",
    )
    return parser.parse_args()


def _quat_wxyz_to_rotation(quaternion: list[float]) -> Any:
    import numpy as np

    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("参考位姿四元数长度为零")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _pose_from_json(entry: dict[str, Any]) -> Any:
    import numpy as np

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _quat_wxyz_to_rotation(entry["quaternion_wxyz"])
    transform[:3, 3] = np.asarray(entry["translation_m"], dtype=float)
    return transform


def _column_transform_from_gf(matrix: Any) -> Any:
    import numpy as np

    return np.asarray(
        [[float(matrix[column][row]) for column in range(4)] for row in range(4)],
        dtype=float,
    )


def _world_matrix(stage: Any, path: str) -> Any:
    from pxr import UsdGeom

    return _column_transform_from_gf(
        UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    )


def _rotation_matrix_to_rotvec(rotation: Any) -> list[float]:
    import numpy as np

    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-10:
        return [0.0, 0.0, 0.0]
    sine = math.sin(angle)
    if abs(sine) <= 1e-8:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis /= np.linalg.norm(axis)
    else:
        axis = np.asarray(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        ) / (2.0 * sine)
    return [float(value * angle) for value in axis]


def _camera_ray_approach(
    grasp_world_from_palm: Any,
    settled_world_from_object: Any,
    reference_world_from_palm: Any,
    reference_world_from_camera: Any,
    distance_m: float,
) -> tuple[Any, list[float]]:
    """Retreat the rigid wrist along its camera-to-object sight line.

    The camera mount is never changed and no look-at rotation is applied.  We
    translate the whole palm so the vector from the settled object center to
    the side-mounted camera keeps the same direction while its length grows.
    This preserves the object's image location instead of letting a straight
    extrapolation of a poorly aligned manual pregrasp sweep it out of frame.
    """

    import numpy as np

    grasp = np.asarray(grasp_world_from_palm, dtype=float)
    settled_object = np.asarray(settled_world_from_object, dtype=float)
    reference_palm = np.asarray(reference_world_from_palm, dtype=float)
    reference_camera = np.asarray(reference_world_from_camera, dtype=float)
    for name, matrix in (
        ("grasp_world_from_palm", grasp),
        ("settled_world_from_object", settled_object),
        ("reference_world_from_palm", reference_palm),
        ("reference_world_from_camera", reference_camera),
    ):
        if matrix.shape != (4, 4):
            raise ValueError(f"{name} 必须是 4x4 位姿")
    palm_from_camera = np.linalg.inv(reference_palm) @ reference_camera
    grasp_world_from_camera = grasp @ palm_from_camera
    ray = grasp_world_from_camera[:3, 3] - settled_object[:3, 3]
    norm = float(np.linalg.norm(ray))
    if norm <= 1e-8:
        raise ValueError("grasp 相机与工件采样中心重合，无法构造接近轨迹")
    retreat_direction = ray / norm
    result = grasp.copy()
    result[:3, 3] += retreat_direction * float(distance_m)
    return result, [float(value) for value in retreat_direction]


def _lift_palm_for_box_wall_clearance(
    desired_world_from_palm: Any,
    settled_world_from_object: Any,
    palm_from_camera: Any,
    inner_min_xy: Any,
    inner_max_xy: Any,
    wall_top_z_m: float,
    clearance_m: float,
    maximum_vertical_lift_m: float,
) -> tuple[Any, dict[str, Any]]:
    """Raise the rigid wrist until its object sight line clears the box rim.

    Starting at an object center inside ``table_box``, find where the segment
    to the camera first exits the inner XY rectangle.  When that crossing is
    below the wall top, translate the whole palm upward just enough to clear
    it.  Camera orientation and the real palm-to-camera mount stay unchanged.
    """

    import numpy as np

    palm = np.asarray(desired_world_from_palm, dtype=float).copy()
    object_pose = np.asarray(settled_world_from_object, dtype=float)
    camera_mount = np.asarray(palm_from_camera, dtype=float)
    minimum = np.asarray(inner_min_xy, dtype=float)
    maximum = np.asarray(inner_max_xy, dtype=float)
    for name, matrix in (
        ("desired_world_from_palm", palm),
        ("settled_world_from_object", object_pose),
        ("palm_from_camera", camera_mount),
    ):
        if matrix.shape != (4, 4):
            raise ValueError(f"{name} 必须是 4x4 位姿")
    if minimum.shape != (2,) or maximum.shape != (2,) or np.any(maximum <= minimum):
        raise ValueError("table_box inner XY bounds 必须是有效的二维范围")
    clearance = max(0.0, float(clearance_m))
    maximum_lift = max(0.0, float(maximum_vertical_lift_m))
    object_position = object_pose[:3, 3]
    camera_before = (palm @ camera_mount)[:3, 3]
    if np.all(camera_before[:2] >= minimum) and np.all(camera_before[:2] <= maximum):
        return palm, {
            "applied": False,
            "reason": "camera_inside_inner_box_xy",
            "vertical_lift_m": 0.0,
            "camera_before_world_m": camera_before.tolist(),
            "camera_after_world_m": camera_before.tolist(),
        }

    delta = camera_before - object_position
    exit_parameters: list[float] = []
    for axis in range(2):
        if delta[axis] > 1e-10:
            value = (maximum[axis] - object_position[axis]) / delta[axis]
        elif delta[axis] < -1e-10:
            value = (minimum[axis] - object_position[axis]) / delta[axis]
        else:
            continue
        if 0.0 <= float(value) <= 1.0:
            exit_parameters.append(float(value))
    if not exit_parameters:
        return palm, {
            "applied": False,
            "reason": "no_inner_wall_crossing",
            "vertical_lift_m": 0.0,
            "camera_before_world_m": camera_before.tolist(),
            "camera_after_world_m": camera_before.tolist(),
        }

    exit_fraction = min(exit_parameters)
    required_crossing_z = float(wall_top_z_m) + clearance
    crossing_z_before = float(object_position[2] + exit_fraction * delta[2])
    requested_lift = max(
        0.0,
        (required_crossing_z - crossing_z_before) / max(exit_fraction, 1e-8),
    )
    applied_lift = min(requested_lift, maximum_lift)
    palm[2, 3] += applied_lift
    camera_after = (palm @ camera_mount)[:3, 3]
    crossing_z_after = crossing_z_before + exit_fraction * applied_lift
    return palm, {
        "applied": bool(applied_lift > 0.0),
        "reason": (
            "cleared_box_wall"
            if requested_lift <= maximum_lift + 1e-12
            else "maximum_vertical_lift_reached"
        ),
        "vertical_lift_m": float(applied_lift),
        "requested_vertical_lift_m": float(requested_lift),
        "maximum_vertical_lift_m": maximum_lift,
        "inner_wall_exit_fraction": exit_fraction,
        "wall_crossing_z_before_m": crossing_z_before,
        "wall_crossing_z_after_m": float(crossing_z_after),
        "required_wall_crossing_z_m": required_crossing_z,
        "camera_before_world_m": camera_before.tolist(),
        "camera_after_world_m": camera_after.tolist(),
    }


def _capture_target_only_visibility(
    rep: Any,
    app: Any,
    stage: Any,
    class_name: str,
    asset_prim_path: str,
    camera_world: Any,
    camera_config: dict[str, Any],
    occluder_paths: list[str],
) -> dict[str, Any]:
    """Render the real target Mesh on a wider-FOV canvas.

    The central crop corresponds to the real camera FOV while the wider render
    provides the complete target silhouette.  This measures frame truncation
    from the rendered Mesh itself and does not use an AABB.
    """

    import numpy as np
    from pxr import Gf, UsdGeom

    visibility_config = camera_config.get("visibility_measurement", {})
    span_factor = float(visibility_config.get("wide_fov_span_factor", 4.0))
    if span_factor <= 1.0:
        raise ValueError("wide_fov_span_factor 必须大于 1")
    width, height = (int(value) for value in camera_config["resolution"])
    intrinsics = camera_config["intrinsics"]
    wide_hfov = expanded_fov_degrees(
        float(intrinsics["horizontal_fov_degrees"]), span_factor
    )
    wide_vfov = expanded_fov_degrees(
        float(intrinsics["vertical_fov_degrees"]), span_factor
    )
    camera_path = "/World/SyntheticData/ApproachOpenVisibilityCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    focal_length = 10.0
    camera.CreateFocalLengthAttr().Set(focal_length)
    camera.CreateHorizontalApertureAttr().Set(
        2.0 * focal_length * math.tan(math.radians(wide_hfov) * 0.5)
    )
    camera.CreateVerticalApertureAttr().Set(
        2.0 * focal_length * math.tan(math.radians(wide_vfov) * 0.5)
    )
    clipping = camera_config["clipping_range_m"]
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*clipping))
    _set_transform(camera.GetPrim(), camera_world)

    hidden: list[tuple[Any, Any]] = []
    for path in occluder_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            continue
        imageable = UsdGeom.Imageable(prim)
        attribute = imageable.GetVisibilityAttr()
        hidden.append((attribute, attribute.Get()))
        imageable.MakeInvisible()

    product = rep.create.render_product(
        camera_path, (width, height), name="ApproachOpenTargetOnlyVisibility"
    )
    annotator = rep.annotators.get(
        "instance_segmentation_fast", init_params={"colorize": False}, device="cpu"
    )
    annotator.attach(product)
    try:
        instance_output: dict[str, Any] = {}
        instance = np.asarray([])
        for attempt in range(6):
            rep.orchestrator.step(rt_subframes=2)
            for _ in range(2):
                app.update()
            instance_output = annotator.get_data()
            instance = np.asarray(_extract_array(instance_output))
            if instance.ndim == 2 and instance.shape == (height, width):
                break
            print(
                f"VISIBILITY_CAPTURE_NOT_READY attempt={attempt + 1} "
                f"shape={instance.shape}",
                flush=True,
            )
        if instance.ndim != 2 or instance.shape != (height, width):
            raise RuntimeError(
                "宽视场实例分割未返回有效二维图像："
                f"expected={(height, width)} actual={instance.shape}"
            )
        target_ids = _target_instance_ids(instance_output, class_name, asset_prim_path)
        mask = np.isin(instance, target_ids)
        result = rendered_mesh_visibility(mask, span_factor)
        result.update(
            {
                "method": "target_only_wide_fov_rendered_mesh",
                "wide_horizontal_fov_degrees": wide_hfov,
                "wide_vertical_fov_degrees": wide_vfov,
                "target_instance_ids": target_ids,
            }
        )
        return result
    finally:
        annotator.detach(product)
        product.destroy()
        for attribute, value in hidden:
            if value is None:
                attribute.Clear()
            else:
                attribute.Set(value)


def _save_sample(
    output_dir: Path,
    class_name: str,
    sample_id: str,
    rgb: Any,
    depth: Any,
    mask: Any,
    pose: dict[str, Any],
) -> dict[str, str]:
    import numpy as np
    from PIL import Image

    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{class_name}_{safe_id}"
    raw_path = output_dir / f"{stem}_raw.png"
    depth_path = output_dir / f"{stem}_depth.npy"
    mask_path = output_dir / f"{stem}_mask.png"
    pose_path = output_dir / f"{stem}_pose.json"
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).save(raw_path)
    np.save(depth_path, np.asarray(depth, dtype=np.float32))
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(mask_path)
    pose_path.write_text(json.dumps(pose, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "raw": str(raw_path),
        "depth": str(depth_path),
        "mask": str(mask_path),
        "pose": str(pose_path),
    }


def main() -> int:
    import numpy as np

    args = parse_args()
    if args.gpu_index < 0:
        raise ValueError("gpu-index 不能为负数")
    shard_path = args.stable_shard.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    if args.hand_side == "left":
        mirror_result = ensure_mirrored_left_references(
            reference_dir, (args.class_name,)
        )
        for path in mirror_result["written"]:
            print(f"MIRRORED_LEFT_REFERENCE={path}", flush=True)
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    if shard.get("class") != args.class_name:
        raise ValueError(
            f"stable shard 类别为 {shard.get('class')!r}，但 --class-name 为 {args.class_name!r}"
        )
    if shard.get("status") != "pass":
        raise ValueError(f"stable shard 未通过：{shard.get('status')}")
    reference_path = reference_dir / reference_filename(
        args.hand_side, args.class_name
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("status") != "complete":
        raise ValueError(f"参考文件未完成：{reference_path}")
    if reference.get("hand_side") != args.hand_side:
        raise ValueError(
            f"参考文件手侧为 {reference.get('hand_side')!r}，"
            f"但本次采集为 {args.hand_side!r}"
        )
    if reference.get("class_name") != args.class_name:
        raise ValueError(
            f"参考文件类别为 {reference.get('class_name')!r}，"
            f"但本次采集为 {args.class_name!r}"
        )
    snapshots = reference["snapshots"]
    ref_object = _pose_from_json(snapshots["grasp"]["world_from_object"])
    ref_grasp = _pose_from_json(snapshots["grasp"]["world_from_palm"])
    ref_pregrasp = _pose_from_json(snapshots["pregrasp"]["world_from_palm"])
    ref_grasp_camera = _pose_from_json(
        snapshots["grasp"]["world_from_camera_rgb"]
    )
    palm_from_camera = np.linalg.inv(
        np.asarray(ref_grasp, dtype=float)
    ) @ np.asarray(ref_grasp_camera, dtype=float)

    assets = load_yaml(args.assets_config)
    camera_config = load_yaml(args.camera_config)["camera"]
    hand = hand_spec(args.hand_side)
    wrist_root_path = str(hand["manual_root"])
    rgb_camera_path = str(hand["rgb_camera"])
    camera_side_config = camera_config.get("hand_sides", {}).get(
        args.hand_side, {}
    )
    camera_name = str(
        camera_side_config.get("name", hand["camera_name"])
    )
    collection_config = load_yaml(args.collection_config)
    scene_path = resolve_from_config(assets, assets["scene"]["usd"])
    asset_path = resolve_from_config(
        assets,
        next(item["usd"] for item in assets["classes"] if item["name"] == args.class_name),
    )
    settled_world_from_asset = shard["pose"]["settled_world_from_asset"]

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": True,
            "multi_gpu": False,
            "active_gpu": args.gpu_index,
            "physics_gpu": args.gpu_index,
            "renderer": args.renderer,
            "width": int(camera_config["resolution"][0]),
            "height": int(camera_config["resolution"][1]),
        }
    )
    annotators: list[Any] = []
    render_product = None
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
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

        from scripts.manual_adjust_left_wrist import _hide_robot

        _hide_robot(stage)
        rig = _prepare_palm_rig(stage, scene_path, args.hand_side)
        grasp_joints = snapshots["grasp"].get("finger_joint_deg", {})
        _set_saved_finger_pose(stage, args.hand_side, grasp_joints)
        wrist_root = stage.GetPrimAtPath(wrist_root_path)
        UsdGeom.Imageable(wrist_root).MakeVisible()

        object_root_path = (
            f"/World/SyntheticData/ApproachOpenSmoke/{args.hand_side}/"
            f"{args.class_name}"
        )
        object_root = UsdGeom.Xform.Define(stage, object_root_path).GetPrim()
        _set_transform(object_root, settled_world_from_asset)
        inspection = inspect_usd(asset_path)
        asset_prim_path = object_root_path + "/Asset"
        asset_prim = add_reference_with_unit_scale(
            stage,
            asset_path,
            asset_prim_path,
            float(inspection["meters_per_unit"]),
            float(UsdGeom.GetStageMetersPerUnit(stage)),
        )
        _add_semantics(asset_prim, args.class_name)
        world_from_object = _world_matrix(stage, object_root_path)
        yaw_delta_rad = math.radians(float(args.object_yaw_delta_deg))
        grasp = transport_reference_pose(
            ref_object,
            ref_grasp,
            world_from_object,
            args.class_name,
            yaw_delta_rad,
            use_settled_heading=True,
        )
        pregrasp = transport_reference_pose(
            ref_object,
            ref_pregrasp,
            world_from_object,
            args.class_name,
            yaw_delta_rad,
            use_settled_heading=True,
        )
        distance = float(args.distance_from_grasp_m)
        if distance < 0.0:
            raise ValueError("distance-from-grasp-m 不能为负数")
        palm_policy = collection_config["approach_open"]["palm_pose"]
        translation_path = str(palm_policy.get("translation_path", ""))
        if translation_path != "box_clearance_camera_ray_from_settled_object":
            raise ValueError(
                "当前只支持箱沿避障的侧置相机射线轨迹 "
                "palm_pose.translation_path="
                "box_clearance_camera_ray_from_settled_object"
            )
        desired_palm, retreat_direction_world = _camera_ray_approach(
            grasp,
            world_from_object,
            ref_grasp,
            ref_grasp_camera,
            distance,
        )
        reference_segment_length = float(
            np.linalg.norm(
                np.asarray(ref_pregrasp, dtype=float)[:3, 3]
                - np.asarray(ref_grasp, dtype=float)[:3, 3]
            )
        )
        if reference_segment_length <= 1e-8:
            reference_segment_length = 0.10
        pregrasp, _ = _camera_ray_approach(
            grasp,
            world_from_object,
            ref_grasp,
            ref_grasp_camera,
            reference_segment_length,
        )
        hand_perturbation = {
            "lateral_offset_1_m": float(args.lateral_offset_1_m),
            "lateral_offset_2_m": float(args.lateral_offset_2_m),
            "roll_delta_deg": float(args.roll_delta_deg),
            "pitch_delta_deg": float(args.pitch_delta_deg),
            "yaw_delta_deg": float(args.palm_yaw_delta_deg),
        }
        desired_palm = apply_hand_perturbation(
            desired_palm, grasp, pregrasp, hand_perturbation
        )
        wall_policy = palm_policy.get("table_box_line_of_sight", {})
        trajectory_wall_clearance: dict[str, Any] = {
            "applied": False,
            "reason": "disabled",
            "vertical_lift_m": 0.0,
        }
        if bool(wall_policy.get("enabled", True)):
            table_box = assets["table_box"]
            desired_palm, trajectory_wall_clearance = _lift_palm_for_box_wall_clearance(
                desired_palm,
                world_from_object,
                palm_from_camera,
                table_box["inner_bounds_world_m"]["min_xy"],
                table_box["inner_bounds_world_m"]["max_xy"],
                float(table_box["outer_bounds_world_m"]["max"][2]),
                float(wall_policy.get("clearance_above_wall_m", 0.02)),
                float(wall_policy.get("maximum_vertical_lift_m", 0.30)),
            )
        _place_palm(rig, desired_palm)
        for _ in range(8):
            app.update()

        width, height = (int(value) for value in camera_config["resolution"])
        render_product = rep.create.render_product(
            rgb_camera_path,
            (width, height),
            name=f"ApproachOpen{args.hand_side.capitalize()}Hand",
        )
        rgb_annotator = rep.annotators.get("rgb", device="cpu")
        instance_annotator = rep.annotators.get(
            "instance_segmentation_fast",
            init_params={"colorize": False},
            device="cpu",
        )
        depth_annotator = rep.annotators.get("distance_to_image_plane", device="cpu")
        annotators = [rgb_annotator, instance_annotator, depth_annotator]
        for annotator in annotators:
            annotator.attach(render_product)
        for _ in range(8):
            app.update()
        rgb = np.asarray([])
        instance = np.asarray([])
        depth = np.asarray([])
        instance_output: dict[str, Any] = {}
        for capture_attempt in range(6):
            rep.orchestrator.step(rt_subframes=4)
            rep.orchestrator.step(rt_subframes=2)
            for _ in range(2):
                app.update()
            rgb = np.asarray(_extract_array(rgb_annotator.get_data())).copy()
            instance_output = instance_annotator.get_data()
            instance = np.asarray(_extract_array(instance_output)).copy()
            depth = np.asarray(
                _extract_array(depth_annotator.get_data()), dtype=np.float32
            ).copy()
            if (
                rgb.ndim >= 3
                and rgb.shape[:2] == (height, width)
                and instance.ndim == 2
                and instance.shape == (height, width)
                and depth.ndim == 2
                and depth.shape == (height, width)
            ):
                break
            print(
                f"MAIN_CAPTURE_NOT_READY attempt={capture_attempt + 1} "
                f"rgb={rgb.shape} instance={instance.shape} depth={depth.shape}",
                flush=True,
            )
        if not (
            rgb.ndim >= 3
            and rgb.shape[:2] == (height, width)
            and instance.ndim == 2
            and instance.shape == (height, width)
            and depth.ndim == 2
            and depth.shape == (height, width)
        ):
            raise RuntimeError(
                f"{args.hand_side} 腕 D405 annotator 在 6 次采集后仍未返回完整帧"
            )
        target_ids = _target_instance_ids(instance_output, args.class_name, asset_prim_path)
        mask = np.isin(instance, target_ids)
        mask_pixels = int(mask.sum())
        clipping = camera_config["clipping_range_m"]
        valid_depth = mask & np.isfinite(depth) & (depth >= float(clipping[0])) & (depth <= float(clipping[1]))
        depth_valid_ratio = float(valid_depth.sum() / mask_pixels) if mask_pixels else 0.0
        mask_bbox = _bbox_from_mask(mask)
        mask_area_fraction_of_image = float(mask_pixels / (width * height))
        touches_image_border = bool(
            mask_bbox
            and (
                mask_bbox[0] <= 0
                or mask_bbox[1] <= 0
                or mask_bbox[2] >= width - 1
                or mask_bbox[3] >= height - 1
            )
        )
        camera_world = _world_matrix(stage, rgb_camera_path)
        camera_from_target = np.asarray(
            camera_from_object(camera_world, world_from_object), dtype=np.float64
        )
        visibility = _capture_target_only_visibility(
            rep,
            app,
            stage,
            args.class_name,
            asset_prim_path,
            camera_world,
            camera_config,
            # 目标单独渲染时隐藏整个环境树，而不只是 table_box：墙体、桌面
            # 或其它环境几何都不能被算进“完整工件投影”参考。
            [wrist_root_path, "/World/Environment"],
        )
        visibility.update(
            scene_visible_fraction_from_reference(mask_pixels, visibility)
        )
        rejection_config = collection_config["approach_open"].get("rejection", {})
        quality = evaluate_approach_quality(
            class_name=args.class_name,
            target_instance_ids=target_ids,
            mask_pixels=mask_pixels,
            depth_valid_ratio=depth_valid_ratio,
            visibility=visibility,
            rejection_config=rejection_config,
            distance_stage=args.distance_stage,
        )
        quality_reasons = quality["quality_reasons"]
        quality_warnings = quality["quality_warnings"]
        thresholds = quality["thresholds"]
        visible_min = float(thresholds["visible_fraction_min"])
        visible_target = float(thresholds["visible_fraction_target"])
        visible_preferred_max = float(thresholds["visible_fraction_preferred_max"])
        depth_min = float(thresholds["depth_valid_ratio_min"])
        min_mask_pixels = int(thresholds["min_mask_pixels"])
        rotation = camera_from_target[:3, :3]
        translation = camera_from_target[:3, 3]
        rotvec = _rotation_matrix_to_rotvec(rotation)
        pose = {
            "camera_id": camera_name,
            "hand_side": args.hand_side,
            "object_class": args.class_name,
            "coordinate_convention": "opencv_x_right_y_down_z_forward",
            "rotation_representation": "Rodrigues_axis_angle_radians",
            "R_vec": {"a": rotvec[0], "b": rotvec[1], "c": rotvec[2]},
            "t_vec": {"x": float(translation[0]), "y": float(translation[1]), "z": float(translation[2])},
            "R_matrix": rotation.tolist(),
            "camera_from_object": camera_from_target.tolist(),
        }
        sample_status = "pass" if not quality_reasons else "fail"
        dataset_root = args.output_root.expanduser().resolve()
        # train/val/test must be directly usable as a training set. Keep rejected
        # RGB/depth/mask/pose artifacts for diagnostics without mixing them in.
        output_dir = dataset_root / args.split
        if sample_status != "pass":
            output_dir = dataset_root / "rejected" / args.split
        files = _save_sample(output_dir, args.class_name, args.sample_id, rgb, depth, mask, pose)
        metadata = {
            "schema_version": 2,
            "artifact_kind": "approach_open_hand_smoke_sample",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": sample_status,
            "hand_side": args.hand_side,
            "class_name": args.class_name,
            "sample_id": args.sample_id,
            "split": args.split,
            "stable_shard": str(shard_path),
            "reference": str(reference_path),
            "requested_object_yaw_delta_deg": float(args.object_yaw_delta_deg),
            "effective_hand_yaw_delta_deg": math.degrees(
                effective_hand_yaw_delta(args.class_name, yaw_delta_rad)
            ),
            "settled_effective_hand_yaw_delta_deg": math.degrees(
                settled_effective_hand_yaw_delta(
                    ref_object,
                    world_from_object,
                    args.class_name,
                    yaw_delta_rad,
                )
            ),
            "trajectory_translation_path": translation_path,
            "trajectory_retreat_direction_world": retreat_direction_world,
            "trajectory_wall_clearance": trajectory_wall_clearance,
            "distance_from_grasp_m": distance,
            "distance_stage": args.distance_stage,
            "hand_perturbation": hand_perturbation,
            "support_face_requested": shard["pose"].get("requested_support_face"),
            "support_face_actual": (shard["pose"].get("actual_support") or {}).get("face"),
            "mask_pixels": mask_pixels,
            "mask_bbox_xyxy": mask_bbox,
            "mask_area_fraction_of_image": mask_area_fraction_of_image,
            "touches_image_border": touches_image_border,
            "depth_valid_ratio": depth_valid_ratio,
            "visibility_reference": visibility,
            "visible_fraction_policy": {
                "min": visible_min,
                "target": visible_target,
                "preferred_max": visible_preferred_max,
                "metric": (
                    "real_scene_mask_pixels_over_target_only_complete_projection"
                ),
                "in_frame_metric": (
                    "fraction_of_target_only_wide_fov_render_inside_real_camera_crop"
                ),
            },
            "depth_valid_ratio_policy": {"min": depth_min},
            "visibility_target_absolute_error": abs(
                float(
                    quality["measurements"][
                        "scene_visible_fraction_of_full_mask"
                    ]
                )
                - visible_target
            )
            if quality["measurements"]["scene_visible_fraction_of_full_mask"]
            is not None
            else None,
            "min_mask_pixels": min_mask_pixels,
            "quality_reasons": quality_reasons,
            "quality_warnings": quality_warnings,
            "target_instance_ids": target_ids,
            "files": files,
        }
        metadata_dir = dataset_root / "_metadata" / args.split
        if sample_status != "pass":
            metadata_dir = dataset_root / "_metadata" / "rejected" / args.split
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / f"{args.class_name}_{args.sample_id}_metadata.json"
        metadata_path.write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"SAMPLE_METADATA={metadata_path}", flush=True)
        print(f"RAW={files['raw']}", flush=True)
        print(f"DEPTH={files['depth']}", flush=True)
        print(f"MASK={files['mask']}", flush=True)
        print(f"POSE={files['pose']}", flush=True)
        print(
            f"STATUS={metadata['status'].upper()} MASK_PIXELS={mask_pixels} "
            f"DEPTH_VALID_RATIO={depth_valid_ratio:.6f}",
            flush=True,
        )
        return 0 if metadata["status"] == "pass" else 1
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
