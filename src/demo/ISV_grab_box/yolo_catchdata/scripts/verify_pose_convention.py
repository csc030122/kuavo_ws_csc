#!/usr/bin/env python3
"""阶段二：使用 Isaac Sim RTX 验证相机、Z-depth、实例 mask 与位姿链路。"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.camera_geometry import (  # noqa: E402
    camera_from_object,
    intrinsics_from_fov,
    look_at_world_from_usd_camera,
    make_transform,
    matrix_multiply,
    project_point,
)
from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402


IDENTITY_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera.yaml")
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output" / "pose_validation"
    )
    parser.add_argument("--renderer", default=None, help="覆盖配置中的 RTX 渲染器")
    return parser.parse_args()


def _as_list(matrix: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _translation(transform: Any) -> list[float]:
    return [float(transform[index][3]) for index in range(3)]


def _check(code: str, passed: bool, actual: Any, expected: str) -> dict[str, Any]:
    return {"code": code, "passed": bool(passed), "actual": actual, "expected": expected}


def _gpu_info() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        rows = []
        for line in completed.stdout.strip().splitlines():
            name, driver, memory = (part.strip() for part in line.split(",", maxsplit=2))
            rows.append(
                {
                    "name": name,
                    "driver_version": driver,
                    "memory_total_mib": int(memory),
                }
            )
        return {"available": bool(rows), "gpus": rows}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "gpus": [], "error": str(exc)}


def _basis_checks() -> list[dict[str, Any]]:
    camera_pose = make_transform(IDENTITY_ROTATION, (0.0, 0.0, 0.0))
    front = camera_from_object(
        camera_pose, make_transform(IDENTITY_ROTATION, (0.0, 0.0, -1.0))
    )
    right = camera_from_object(
        camera_pose, make_transform(IDENTITY_ROTATION, (0.1, 0.0, -1.0))
    )
    down = camera_from_object(
        camera_pose, make_transform(IDENTITY_ROTATION, (0.0, -0.1, -1.0))
    )
    front_translation = _translation(front)
    right_translation = _translation(right)
    down_translation = _translation(down)
    return [
        _check(
            "front_one_meter",
            all(abs(actual - expected) <= 1e-12 for actual, expected in zip(front_translation, [0, 0, 1])),
            front_translation,
            "正前方 1 米时 t=[0, 0, 1]",
        ),
        _check("image_right_positive_x", right_translation[0] > 0.0, right_translation, "图像右侧 x>0"),
        _check("image_down_positive_y", down_translation[1] > 0.0, down_translation, "图像下侧 y>0"),
    ]


def _gf_matrix_from_column_transform(transform: Any) -> Any:
    from pxr import Gf

    matrix = _as_list(transform)
    transposed = [[matrix[column][row] for column in range(4)] for row in range(4)]
    return Gf.Matrix4d(*[value for row in transposed for value in row])


def _column_transform_from_gf(matrix: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(matrix[column][row]) for column in range(4)) for row in range(4)
    )


def _set_transform(prim: Any, transform: Any) -> None:
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        _gf_matrix_from_column_transform(transform)
    )


def _add_semantics(prim: Any, class_name: str) -> None:
    from pxr import Semantics

    semantic = Semantics.SemanticsAPI.Apply(prim, f"class_{class_name}")
    semantic.CreateSemanticTypeAttr().Set("class")
    semantic.CreateSemanticDataAttr().Set(class_name)


def _extract_array(data: Any) -> Any:
    return data["data"] if isinstance(data, dict) and "data" in data else data


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _target_instance_ids(instance_output: dict[str, Any], class_name: str, prim_path: str) -> list[int]:
    info = instance_output.get("info", {})
    semantics = info.get("idToSemantics", {})
    labels = info.get("idToLabels", {})
    result: list[int] = []
    for raw_identifier in set(semantics) | set(labels):
        semantic = semantics.get(raw_identifier, semantics.get(str(raw_identifier), {}))
        label = labels.get(raw_identifier, labels.get(str(raw_identifier), ""))
        semantic_text = json.dumps(_json_safe(semantic), ensure_ascii=False)
        label_text = str(label)
        if class_name in semantic_text or prim_path in label_text:
            result.append(int(raw_identifier))
    return sorted(set(result))


def _bbox_from_mask(mask: Any) -> list[int] | None:
    import numpy as np

    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]


def _project_asset_aabb(
    bounds_m: dict[str, list[float]],
    camera_from_asset: Any,
    intrinsics: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[float], list[list[float]]]:
    from yolo_catchdata.camera_geometry import transform_point

    minimum = bounds_m["min"]
    maximum = bounds_m["max"]
    projected = []
    for x in (minimum[0], maximum[0]):
        for y in (minimum[1], maximum[1]):
            for z in (minimum[2], maximum[2]):
                point_camera = transform_point(camera_from_asset, (x, y, z))
                projected.append(list(project_point(intrinsics, point_camera)))
    x_values = [point[0] for point in projected]
    y_values = [point[1] for point in projected]
    bbox = [
        max(0.0, min(x_values)),
        max(0.0, min(y_values)),
        min(float(width - 1), max(x_values)),
        min(float(height - 1), max(y_values)),
    ]
    return bbox, projected


def _mask_inside_bbox_ratio(mask: Any, bbox: list[float]) -> float:
    import numpy as np

    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return 0.0
    inside = (
        (columns >= math.floor(bbox[0]) - 1)
        & (columns <= math.ceil(bbox[2]) + 1)
        & (rows >= math.floor(bbox[1]) - 1)
        & (rows <= math.ceil(bbox[3]) + 1)
    )
    return float(np.count_nonzero(inside) / len(rows))


def _bbox_iou(first: list[float] | None, second: list[float] | None) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left + 1.0) * max(0.0, bottom - top + 1.0)
    first_area = max(0.0, first[2] - first[0] + 1.0) * max(0.0, first[3] - first[1] + 1.0)
    second_area = max(0.0, second[2] - second[0] + 1.0) * max(0.0, second[3] - second[1] + 1.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


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


def _save_outputs(
    output_dir: Path,
    rgb: Any,
    mask: Any,
    depth: Any,
    projected_bbox: list[float],
    projected_points: list[list[float]],
) -> dict[str, str]:
    import numpy as np
    from PIL import Image, ImageDraw

    preview_dir = output_dir / "preview"
    sample_dir = output_dir / "sample"
    preview_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    rgb_image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3], mode="RGB")
    rgb_path = preview_dir / "rgb.png"
    rgb_image.save(rgb_path)

    mask_image = Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255), mode="L")
    mask_path = preview_dir / "mask.png"
    mask_image.save(mask_path)

    valid = np.isfinite(depth) & (depth > 0.0)
    color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2.0, 98.0])
        span = max(float(high - low), 1e-6)
        normalized = np.clip((depth - low) / span, 0.0, 1.0)
        color[..., 0] = (255.0 * (1.0 - normalized)).astype(np.uint8)
        color[..., 1] = (255.0 * (1.0 - np.abs(normalized - 0.5) * 2.0)).astype(np.uint8)
        color[..., 2] = (255.0 * normalized).astype(np.uint8)
        color[~valid] = 0
    depth_path = preview_dir / "depth.png"
    Image.fromarray(color, mode="RGB").save(depth_path)

    overlay = rgb_image.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(projected_bbox, outline=(255, 40, 40), width=3)
    for x, y in projected_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 220, 0))
    mask_rgba = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    mask_rgba.putalpha(mask_image.point(lambda value: 80 if value else 0))
    green = Image.new("RGBA", overlay.size, (0, 255, 0, 0))
    green.putalpha(mask_rgba.getchannel("A"))
    overlay = Image.alpha_composite(overlay.convert("RGBA"), green).convert("RGB")
    overlay_path = preview_dir / "projection_overlay.png"
    overlay.save(overlay_path)

    depth_npy_path = sample_dir / "depth.npy"
    np.save(depth_npy_path, np.asarray(depth, dtype=np.float32))
    return {
        "rgb": str(rgb_path),
        "mask": str(mask_path),
        "depth_preview": str(depth_path),
        "projection_overlay": str(overlay_path),
        "depth_npy": str(depth_npy_path),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    status = "通过" if report["status"] == "pass" else "失败"
    gpu_text = "；".join(
        f"{item['name']}，驱动 {item['driver_version']}，显存 {item['memory_total_mib']} MiB"
        for item in report["environment"]["gpu"].get("gpus", [])
    ) or "未检测到可用 NVIDIA GPU"
    lines = [
        "# 阶段二：相机、深度与位姿验证报告",
        "",
        f"总体状态：**{status}**",
        "",
        "## 运行环境",
        "",
        f"- GPU：{gpu_text}",
        f"- 渲染器：`{report['environment']['renderer']}`",
        f"- 分辨率：`{report['camera']['resolution'][0]} × {report['camera']['resolution'][1]}`",
        "",
        "## 坐标系约定",
        "",
        "- 输出光学坐标系采用 OpenCV：+X 向图像右侧、+Y 向图像下方、+Z 向前。",
        "- 位姿链为 `T_C_O = T_C_W · T_W_A · T_A_O`。",
        "- Depth 来自 `distance_to_image_plane`，含义为 Z-depth，单位为米。",
        "",
        "## 验收项",
        "",
        "| 验收项 | 状态 | 实际值 | 期望 |",
        "|---|---|---|---|",
    ]
    for check in report["checks"]:
        check_status = "通过" if check["passed"] else "失败"
        actual = json.dumps(check["actual"], ensure_ascii=False)
        lines.append(f"| `{check['code']}` | {check_status} | `{actual}` | {check['expected']} |")
    lines.extend(
        [
            "",
            "## 相机内参说明",
            "",
            f"当前内参来源：`{report['camera']['intrinsics_source']}`。这是用于合成链路验证的 D405 "
            "标称视场角针孔模型，真机采集前必须替换为实际右腕相机的标定结果。",
            "",
            "## 预览与样本文件",
            "",
        ]
    )
    for name, output in report["outputs"].items():
        lines.append(f"- `{name}`：`{output}`")
    if report["issues"]:
        lines.extend(["", "## 问题", ""])
        lines.extend(f"- {issue}" for issue in report["issues"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    camera_config_path = args.camera_config.expanduser().resolve()
    assets_config_path = args.assets_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    camera_config = load_yaml(camera_config_path)["camera"]
    assets_config = load_yaml(assets_config_path)
    validation = camera_config["phase2_validation"]
    renderer = args.renderer or validation["renderer"]
    width, height = (int(value) for value in camera_config["resolution"])
    scene_path = resolve_from_config(assets_config, assets_config["scene"]["usd"])

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": True,
            "renderer": renderer,
            "width": width,
            "height": height,
            "open_usd": str(scene_path),
        }
    )
    annotators = []
    render_product = None
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        from yolo_catchdata.asset_loader import add_reference_with_unit_scale, inspect_usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"无法打开场景：{scene_path}")
        for _ in range(4):
            app.update()

        class_name = str(validation["class_name"])
        class_definition = next(
            item for item in assets_config["classes"] if item["name"] == class_name
        )
        asset_path = resolve_from_config(assets_config, class_definition["usd"])
        asset_inspection = inspect_usd(asset_path)
        scene_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
        table = assets_config["table_box"]
        asset_minimum = asset_inspection["bounds_m"]["min"]
        asset_maximum = asset_inspection["bounds_m"]["max"]
        table_min = table["inner_bounds_world_m"]["min_xy"]
        table_max = table["inner_bounds_world_m"]["max_xy"]
        table_center = [(table_min[index] + table_max[index]) * 0.5 for index in range(2)]
        asset_center = [(asset_minimum[index] + asset_maximum[index]) * 0.5 for index in range(2)]
        asset_translation = (
            table_center[0] - asset_center[0],
            table_center[1] - asset_center[1],
            float(table["floor_z_m"]) - asset_minimum[2],
        )
        world_from_asset = make_transform(IDENTITY_ROTATION, asset_translation)
        asset_from_object = tuple(tuple(float(value) for value in row) for row in class_definition["asset_from_canonical"])
        world_from_object = matrix_multiply(world_from_asset, asset_from_object)

        target_root_path = "/World/SyntheticData/Phase2Target"
        target_root = UsdGeom.Xform.Define(stage, target_root_path).GetPrim()
        _set_transform(target_root, world_from_asset)
        asset_prim_path = f"{target_root_path}/Asset"
        asset_prim = add_reference_with_unit_scale(
            stage,
            asset_path,
            asset_prim_path,
            asset_inspection["meters_per_unit"],
            scene_mpu,
        )
        _add_semantics(asset_prim, class_name)

        sampling_center_object = class_definition["sampling_center_canonical_m"]
        sampling_center_world = [
            sum(world_from_object[row][column] * sampling_center_object[column] for column in range(3))
            + world_from_object[row][3]
            for row in range(3)
        ]
        radius = float(validation["radius_m"])
        elevation = math.radians(float(validation["elevation_degrees"]))
        azimuth = math.radians(float(validation["azimuth_degrees"]))
        rotation_world_from_sampling = assets_config["sampling_frame"]["rotation_world_from_sampling"]
        offset_sampling = (
            radius * math.cos(elevation) * math.cos(azimuth),
            radius * math.cos(elevation) * math.sin(azimuth),
            radius * math.sin(elevation),
        )
        offset_world = [
            sum(rotation_world_from_sampling[row][column] * offset_sampling[column] for column in range(3))
            for row in range(3)
        ]
        camera_position = [sampling_center_world[index] + offset_world[index] for index in range(3)]
        intended_world_from_camera = look_at_world_from_usd_camera(
            camera_position, sampling_center_world
        )

        camera_path = str(camera_config["prim_path"])
        UsdGeom.Xform.Define(stage, str(Path(camera_path).parent))
        camera = UsdGeom.Camera.Define(stage, camera_path)
        focal_length = 10.0
        hfov = float(camera_config["intrinsics"]["horizontal_fov_degrees"])
        vfov = float(camera_config["intrinsics"]["vertical_fov_degrees"])
        horizontal_aperture = 2.0 * focal_length * math.tan(math.radians(hfov) * 0.5)
        vertical_aperture = 2.0 * focal_length * math.tan(math.radians(vfov) * 0.5)
        camera.CreateFocalLengthAttr().Set(focal_length)
        camera.CreateHorizontalApertureAttr().Set(horizontal_aperture)
        camera.CreateVerticalApertureAttr().Set(vertical_aperture)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*camera_config["clipping_range_m"]))
        _set_transform(camera.GetPrim(), intended_world_from_camera)

        dome = UsdLux.DomeLight.Define(stage, "/World/SyntheticData/Phase2DomeLight")
        dome.CreateIntensityAttr().Set(650.0)
        distant = UsdLux.DistantLight.Define(stage, "/World/SyntheticData/Phase2DistantLight")
        distant.CreateIntensityAttr().Set(2200.0)
        distant.CreateAngleAttr().Set(1.0)
        _set_transform(
            distant.GetPrim(),
            look_at_world_from_usd_camera((1.0, 0.5, 3.0), sampling_center_world),
        )

        render_product = rep.create.render_product(camera_path, (width, height), name="Phase2RightWrist")
        rgb_annotator = rep.annotators.get("rgb", device="cpu")
        instance_annotator = rep.annotators.get(
            "instance_segmentation_fast", init_params={"colorize": False}, device="cpu"
        )
        depth_annotator = rep.annotators.get("distance_to_image_plane", device="cpu")
        camera_params_annotator = rep.annotators.get("camera_params", device="cpu")
        annotators = [rgb_annotator, instance_annotator, depth_annotator, camera_params_annotator]
        for annotator in annotators:
            annotator.attach(render_product)

        for _ in range(int(validation["warmup_frames"])):
            app.update()
        rep.orchestrator.step(rt_subframes=int(validation["rt_subframes"]))

        rgb = np.asarray(_extract_array(rgb_annotator.get_data()))
        instance_output = instance_annotator.get_data()
        instance_data = np.asarray(_extract_array(instance_output))
        depth = np.asarray(_extract_array(depth_annotator.get_data()), dtype=np.float32)
        camera_params = _json_safe(camera_params_annotator.get_data())
        target_ids = _target_instance_ids(instance_output, class_name, asset_prim_path)
        mask = np.isin(instance_data, target_ids)

        actual_world_from_camera = _column_transform_from_gf(
            UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())
        )
        camera_from_target = camera_from_object(actual_world_from_camera, world_from_object)
        camera_from_asset = camera_from_object(actual_world_from_camera, world_from_asset)
        intrinsics = intrinsics_from_fov(width, height, hfov, vfov)
        projected_bbox, projected_points = _project_asset_aabb(
            asset_inspection["bounds_m"], camera_from_asset, intrinsics, width, height
        )
        mask_bbox = _bbox_from_mask(mask)
        mask_pixels = int(np.count_nonzero(mask))
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        valid_depth_ratio = float(np.count_nonzero(valid_depth & mask) / mask_pixels) if mask_pixels else 0.0
        inside_ratio = _mask_inside_bbox_ratio(mask, projected_bbox)
        bbox_iou = _bbox_iou(mask_bbox, projected_bbox)
        target_translation = _translation(camera_from_target)
        authored_focal_length = float(camera_params["cameraFocalLength"])
        authored_aperture = camera_params["cameraAperture"]
        authored_intrinsics = {
            "fx": width * authored_focal_length / float(authored_aperture[0]),
            "fy": height * authored_focal_length / float(authored_aperture[1]),
        }
        intrinsics_match = (
            abs(authored_intrinsics["fx"] - float(intrinsics["fx"])) <= 1e-3
            and abs(authored_intrinsics["fy"] - float(intrinsics["fy"])) <= 1e-3
        )

        outputs = _save_outputs(
            output_dir, rgb, mask, depth, projected_bbox, projected_points
        )
        pose_path = output_dir / "sample" / "pose.json"
        rotation = [row[:3] for row in camera_from_target[:3]]
        rotation_vector = _rotation_matrix_to_rotvec(rotation)
        pose = {
            "camera_id": camera_config["name"],
            "object_class": class_name,
            "coordinate_convention": "opencv_x_right_y_down_z_forward",
            "rotation_representation": "Rodrigues_axis_angle_radians",
            "translation_unit": "meter",
            "R_vec": {
                "a": rotation_vector[0],
                "b": rotation_vector[1],
                "c": rotation_vector[2],
            },
            "t_vec": {"x": target_translation[0], "y": target_translation[1], "z": target_translation[2]},
            "R_matrix": rotation,
            "camera_from_object": _as_list(camera_from_target),
        }
        pose_path.write_text(json.dumps(pose, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs["pose_json"] = str(pose_path)

        gpu = _gpu_info()
        checks = _basis_checks()
        expected_shape = [height, width]
        checks.extend(
            [
                _check("nvidia_gpu_available", gpu["available"], gpu, "检测到可用 NVIDIA GPU"),
                _check("rgb_shape", list(rgb.shape[:2]) == expected_shape, list(rgb.shape), f"前两维为 {expected_shape}"),
                _check("mask_shape", list(mask.shape[:2]) == expected_shape, list(mask.shape), f"尺寸为 {expected_shape}"),
                _check("depth_shape", list(depth.shape[:2]) == expected_shape, list(depth.shape), f"尺寸为 {expected_shape}"),
                _check(
                    "camera_intrinsics_authoring",
                    intrinsics_match,
                    authored_intrinsics,
                    "USD Camera 的 fx/fy 与配置计算值误差不超过 1e-3 像素",
                ),
                _check("target_instance_found", bool(target_ids), target_ids, f"实例语义包含 {class_name}"),
                _check("mask_pixels", mask_pixels >= int(validation["min_mask_pixels"]), mask_pixels, f">= {validation['min_mask_pixels']}"),
                _check("valid_depth_ratio", valid_depth_ratio >= float(validation["min_valid_depth_ratio"]), valid_depth_ratio, f">= {validation['min_valid_depth_ratio']}"),
                _check("mask_inside_projected_bbox", inside_ratio >= float(validation["min_mask_inside_projected_bbox_ratio"]), inside_ratio, f">= {validation['min_mask_inside_projected_bbox_ratio']}"),
                _check(
                    "look_at_pose",
                    target_translation[2] > 0.0 and abs(target_translation[0]) <= 1e-6 and abs(target_translation[1]) <= 1e-6,
                    target_translation,
                    "目标中心位于相机前方且在光轴上",
                ),
            ]
        )
        issues = [f"验收项 {item['code']} 未通过" for item in checks if not item["passed"]]
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if not issues else "fail",
            "environment": {"renderer": renderer, "gpu": gpu},
            "scene": {"path": str(scene_path), "meters_per_unit": scene_mpu},
            "camera": {
                "name": camera_config["name"],
                "prim_path": camera_path,
                "resolution": [width, height],
                "intrinsics_source": camera_config["intrinsics"]["source"],
                "horizontal_fov_degrees": hfov,
                "vertical_fov_degrees": vfov,
                "intrinsics": intrinsics,
                "position_world_m": camera_position,
                "world_from_camera_usd": _as_list(actual_world_from_camera),
                "annotator_parameters": camera_params,
            },
            "object": {
                "class_name": class_name,
                "asset_path": str(asset_path),
                "asset_prim_path": asset_prim_path,
                "world_from_asset": _as_list(world_from_asset),
                "asset_from_object": _as_list(asset_from_object),
                "world_from_object": _as_list(world_from_object),
                "camera_from_object": _as_list(camera_from_target),
                "sampling_center_world_m": sampling_center_world,
            },
            "metrics": {
                "target_instance_ids": target_ids,
                "mask_pixels": mask_pixels,
                "valid_depth_ratio": valid_depth_ratio,
                "mask_bbox_xyxy": mask_bbox,
                "projected_asset_aabb_xyxy": projected_bbox,
                "mask_inside_projected_bbox_ratio": inside_ratio,
                "mask_projected_bbox_iou": bbox_iou,
                "instance_info": _json_safe(instance_output.get("info", {})),
            },
            "checks": checks,
            "issues": issues,
            "outputs": outputs,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "pose_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        markdown_path = output_dir / "pose_report.md"
        _write_markdown(report, markdown_path)
        print(f"POSE_REPORT={report_path}", flush=True)
        print(f"POSE_REPORT_MD={markdown_path}", flush=True)
        print(f"STATUS={report['status'].upper()}", flush=True)
        print(
            f"MASK_PIXELS={mask_pixels} DEPTH_VALID_RATIO={valid_depth_ratio:.6f} "
            f"MASK_INSIDE_BBOX={inside_ratio:.6f} BBOX_IOU={bbox_iou:.6f}",
            flush=True,
        )
        return 0 if report["status"] == "pass" else 1
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
