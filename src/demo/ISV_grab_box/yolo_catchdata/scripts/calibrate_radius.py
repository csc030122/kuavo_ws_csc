#!/usr/bin/env python3
"""阶段三：用 RTX 扫描四类工件的距离—投影尺度关系。"""

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
    apply_opencv_roll_to_usd_camera,
    camera_from_object,
    intrinsics_from_fov,
    look_at_world_from_usd_camera,
    matrix_multiply,
    project_point,
    transform_point,
)
from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402
from yolo_catchdata.scale_solver import (  # noqa: E402
    aggregate_viewpoint_solutions,
    bbox_width_scale,
    projected_bbox_truncation_ratio,
    solve_scale_bins,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera.yaml")
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    parser.add_argument("--renderer", default=None)
    parser.add_argument(
        "--class-name",
        default=None,
        help="只标定一个类别并写入分片；正式全量建议分别运行四个类别后再合并。",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="仅用于调试，限制扫描记录数；正式标定不要设置。",
    )
    return parser.parse_args()


def _gpu_info() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = []
        for line in completed.stdout.strip().splitlines():
            name, driver, memory = (part.strip() for part in line.split(",", maxsplit=2))
            rows.append(
                {"name": name, "driver_version": driver, "memory_total_mib": int(memory)}
            )
        return {"available": bool(rows), "gpus": rows}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "gpus": [], "error": str(exc)}


def _texture_status() -> dict[str, Any]:
    texture_dir = PROJECT_ROOT / "isaac_sim" / "materials" / "epoxy_floor"
    names = [
        "epoxy_light_gray_albedo.png",
        "epoxy_light_gray_normal.png",
        "epoxy_light_gray_roughness.png",
        "epoxy_orange_peel_detail_normal.png",
    ]
    files = []
    for name in names:
        path = texture_dir / name
        files.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return {"all_present": all(item["exists"] for item in files), "files": files}


def _as_list(matrix: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _gf_matrix_from_column_transform(transform: Any) -> Any:
    from pxr import Gf

    matrix = _as_list(transform)
    transposed = [[matrix[column][row] for column in range(4)] for row in range(4)]
    return Gf.Matrix4d(*[value for row in transposed for value in row])


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
    result = []
    for raw_identifier in set(semantics) | set(labels):
        semantic = semantics.get(raw_identifier, semantics.get(str(raw_identifier), {}))
        label = labels.get(raw_identifier, labels.get(str(raw_identifier), ""))
        if class_name in json.dumps(_json_safe(semantic), ensure_ascii=False) or prim_path in str(label):
            result.append(int(raw_identifier))
    return sorted(set(result))


def _bbox_from_mask(mask: Any) -> list[int] | None:
    import numpy as np

    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]


def _camera_pose(
    sampling_center_world: list[float],
    rotation_world_from_sampling: list[list[float]],
    radius_m: float,
    theta_deg: float,
    phi_deg: float,
    psi_deg: float,
) -> tuple[list[float], tuple[tuple[float, ...], ...]]:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    offset_sampling = (
        radius_m * math.cos(phi) * math.cos(theta),
        radius_m * math.cos(phi) * math.sin(theta),
        radius_m * math.sin(phi),
    )
    offset_world = [
        sum(rotation_world_from_sampling[row][column] * offset_sampling[column] for column in range(3))
        for row in range(3)
    ]
    position = [sampling_center_world[index] + offset_world[index] for index in range(3)]
    pose = look_at_world_from_usd_camera(position, sampling_center_world)
    return position, apply_opencv_roll_to_usd_camera(pose, psi_deg)


def _project_aabb_unclipped(
    bounds_m: dict[str, list[float]], camera_from_asset: Any, intrinsics: dict[str, Any]
) -> tuple[list[float] | None, list[float]]:
    minimum = bounds_m["min"]
    maximum = bounds_m["max"]
    camera_points = []
    projected = []
    for x in (minimum[0], maximum[0]):
        for y in (minimum[1], maximum[1]):
            for z in (minimum[2], maximum[2]):
                point = transform_point(camera_from_asset, (x, y, z))
                camera_points.append(point)
                if point[2] > 1e-9:
                    projected.append(project_point(intrinsics, point))
    depths = [float(point[2]) for point in camera_points]
    if len(projected) != 8:
        return None, depths
    return [
        min(point[0] for point in projected),
        min(point[1] for point in projected),
        max(point[0] for point in projected),
        max(point[1] for point in projected),
    ], depths


def _save_reference_preview(path: Path, rgb: Any, mask: Any, record: dict[str, Any]) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    rgb_array = np.asarray(rgb, dtype=np.uint8)[..., :3]
    mask_array = np.asarray(mask, dtype=bool)
    image = Image.fromarray(rgb_array, mode="RGB")

    # 原始 RGB 和二值 Mask 必须单独保留。参考图只画轮廓，不能用色块覆盖真实材质。
    raw_path = path.with_name(f"{record['class']}_scale_reference_rgb.png")
    mask_path = path.with_name(f"{record['class']}_scale_reference_mask.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_array, mode="RGB").save(raw_path)
    Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L").save(mask_path)

    interior = mask_array.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    interior[1:-1, 1:-1] &= (
        mask_array[:-2, 1:-1]
        & mask_array[2:, 1:-1]
        & mask_array[1:-1, :-2]
        & mask_array[1:-1, 2:]
    )
    boundary = mask_array & ~interior
    boundary_thick = boundary.copy()
    boundary_thick[:-1, :] |= boundary[1:, :]
    boundary_thick[1:, :] |= boundary[:-1, :]
    boundary_thick[:, :-1] |= boundary[:, 1:]
    boundary_thick[:, 1:] |= boundary[:, :-1]
    annotated = rgb_array.copy()
    annotated[boundary_thick] = (0, 255, 255)
    image = Image.fromarray(annotated, mode="RGB")
    draw = ImageDraw.Draw(image)
    bbox = record.get("mask_bbox_xyxy")
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline=(255, 215, 0), width=2)
    text = (
        f"isolated / unoccluded | {record['class']} | r={record['r_m']:.2f}m | "
        f"s={record['s_pre_occlusion']:.3f} | depth valid={record['depth_valid_ratio']:.3f}"
    )
    draw.rectangle((8, 8, 900, 42), fill=(0, 0, 0))
    draw.text((16, 15), text, fill=(255, 255, 255))
    image.save(path)


def _monotonic_violations(records: list[dict[str, Any]], tolerance: float = 0.01) -> int:
    usable = sorted(
        (item for item in records if item["usable_for_solver"]),
        key=lambda item: item["r_m"],
        reverse=True,
    )
    return sum(
        1
        for previous, current in zip(usable, usable[1:])
        if current["s_pre_occlusion"] + tolerance < previous["s_pre_occlusion"]
    )


def _plot_curves(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    colors = {"frontal_mid": "tab:blue", "left_low_roll": "tab:orange", "right_high_roll": "tab:green"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, (class_name, class_result) in zip(axes.flat, result["classes"].items()):
        for bin_name, limits in result["scale_bins"].items():
            axis.axhspan(limits[0], limits[1], alpha=0.06)
            axis.text(0.945, (limits[0] + limits[1]) * 0.5, bin_name, fontsize=8, va="center")
        for viewpoint_id, viewpoint in class_result["viewpoints"].items():
            records = viewpoint["records"]
            axis.plot(
                [item["r_m"] for item in records],
                [item["s_pre_occlusion"] for item in records],
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=colors.get(viewpoint_id),
                label=viewpoint_id,
            )
            invalid = [item for item in records if not item["usable_for_solver"]]
            axis.scatter(
                [item["r_m"] for item in invalid],
                [item["s_pre_occlusion"] for item in invalid],
                marker="x",
                color="red",
                s=22,
            )
        axis.set_title(class_name)
        axis.set_xlabel("radius r (m)")
        axis.set_ylabel("s_pre_occlusion (bbox width / image width)")
        axis.set_xlim(1.0, 0.05)
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Phase 3 radius-to-projected-scale curves (red x = rejected)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _format_solution(solution: dict[str, Any]) -> str:
    if solution["r0_m"] is None:
        return "不可达"
    bounds = solution["search_bounds_m"]
    return f"r0={solution['r0_m']:.3f}，搜索=[{bounds[0]:.3f}, {bounds[1]:.3f}]"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    gpu_text = "；".join(
        f"{item['name']}，驱动 {item['driver_version']}，显存 {item['memory_total_mib']} MiB"
        for item in result["environment"]["gpu"].get("gpus", [])
    ) or "未检测到 NVIDIA GPU"
    status_text = "通过" if result["status"] == "pass" else "失败"
    lines = [
        "# 阶段三：四类工件半径标定报告",
        "",
        f"总体状态：**{status_text}**",
        "",
        "## 标定定义",
        "",
        "- `r`：相机光心到工件 sampling center 的距离，单位为米。",
        "- `s_pre_occlusion`：无外部遮挡时，完整目标 Mask 的 bbox 宽度除以图像宽度。",
        "- `truncation_ratio`：未裁剪 CAD AABB 投影框落在图像外的面积比例。",
        "- 标定使用隔离米制 Stage，只加载当前工件，不加载支撑面、机器人或料箱；工件位姿来自真实 `table_box` 物理沉降结果。",
        "- 隔离 Stage 只用于获得无遮挡完整 Mask；源 USD 不保存、不修改。",
        "- 红叉扫描点表示 Mask、Depth 或截断质量不足，只留作边界记录，不参与半径插值。",
        "",
        "## 运行环境",
        "",
        f"- GPU：{gpu_text}",
        f"- 渲染器：`{result['environment']['renderer']}`",
        f"- 标定记录数：`{len(result['records'])}`",
        f"- 恢复的地面纹理完整：`{result['environment']['texture_assets']['all_present']}`",
        "",
        "## 类别级闭环初值",
        "",
        "| 类别 | 可用尺度范围 | Far | Mid | Near | Pre-grasp |",
        "|---|---|---|---|---|---|",
    ]
    for class_name, class_result in result["classes"].items():
        observed = class_result["observed_usable_scale_range"]
        observed_text = "无" if observed is None else f"{observed[0]:.3f}–{observed[1]:.3f}"
        aggregate = class_result["aggregate_scale_bins"]
        lines.append(
            f"| {class_name} | {observed_text} | {_format_solution(aggregate['far'])} | "
            f"{_format_solution(aggregate['mid'])} | {_format_solution(aggregate['near'])} | "
            f"{_format_solution(aggregate['pre_grasp'])} |"
        )
    lines.extend(["", "## 代表视角", ""])
    for viewpoint in result["representative_viewpoints"]:
        lines.append(
            f"- `{viewpoint['id']}`：θ={viewpoint['theta_deg']}°，φ={viewpoint['phi_deg']}°，"
            f"ψ={viewpoint['psi_deg']}°"
        )
    if result["warnings"]:
        lines.extend(["", "## 覆盖限制", ""])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    if result["issues"]:
        lines.extend(["", "## 错误", ""])
        lines.extend(f"- {issue}" for issue in result["issues"])
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- 标定配置：`{result['outputs']['calibration_json']}`",
            f"- 曲线图：`{result['outputs']['curves']}`",
        ]
    )
    for preview in result["outputs"]["reference_previews"]:
        lines.append(f"- 参考预览：`{preview}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assets_config = load_yaml(args.assets_config)
    camera_config = load_yaml(args.camera_config)["camera"]
    collection_config = load_yaml(args.collection_config)
    calibration_config = collection_config["radius_calibration"]
    stable_pose_path = CONFIG_DIR / str(calibration_config["stable_pose_config"])
    if not stable_pose_path.is_file():
        raise FileNotFoundError(
            f"缺少物理稳定姿态配置 {stable_pose_path}；请先运行 collect_stable_pose.py 和 "
            "merge_stable_poses.py"
        )
    stable_poses = json.loads(stable_pose_path.read_text(encoding="utf-8"))
    if stable_poses.get("status") != "pass":
        raise RuntimeError(f"物理稳定姿态配置未通过：{stable_pose_path}")
    if stable_poses.get("pose_id") != calibration_config["object_pose_id"]:
        raise RuntimeError("radius_calibration.object_pose_id 与 stable_poses.pose_id 不一致")
    renderer = args.renderer or calibration_config["renderer"]
    width, height = (int(value) for value in camera_config["resolution"])
    scene_path = resolve_from_config(assets_config, assets_config["scene"]["usd"])
    report_dir = resolve_from_config(collection_config, calibration_config["report_dir"])
    if args.class_name is None:
        selected_definitions = list(assets_config["classes"])
    else:
        selected_definitions = [
            item for item in assets_config["classes"] if item["name"] == args.class_name
        ]
        if not selected_definitions:
            names = ", ".join(item["name"] for item in assets_config["classes"])
            raise ValueError(f"未知类别 {args.class_name!r}；可选值：{names}")
    shard_mode = args.class_name is not None
    if shard_mode:
        output_json = report_dir / "shards" / f"{args.class_name}.json"
    else:
        output_json = CONFIG_DIR / str(calibration_config["output"])
    radii = [float(value) for value in calibration_config["radii_m"]]
    viewpoints = calibration_config["representative_viewpoints"]
    expected_records = len(selected_definitions) * len(viewpoints) * len(radii)
    if args.max_records is not None:
        expected_records = min(expected_records, args.max_records)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": True,
            "renderer": renderer,
            "width": width,
            "height": height,
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

        context = omni.usd.get_context()
        context.new_stage()
        for _ in range(2):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("无法创建隔离标定 Stage")
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)
        UsdGeom.SetStageMetersPerUnit(
            stage, float(assets_config["scene"]["expected_meters_per_unit"])
        )
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        for _ in range(4):
            app.update()

        scene_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
        table = assets_config["table_box"]

        table_min = table["inner_bounds_world_m"]["min_xy"]
        table_max = table["inner_bounds_world_m"]["max_xy"]
        table_center = [(table_min[index] + table_max[index]) * 0.5 for index in range(2)]
        object_runtime: dict[str, dict[str, Any]] = {}
        for definition in selected_definitions:
            class_name = definition["name"]
            asset_path = resolve_from_config(assets_config, definition["usd"])
            inspection = inspect_usd(asset_path)
            stable_class = stable_poses.get("classes", {}).get(class_name)
            if stable_class is None:
                raise RuntimeError(f"物理稳定姿态缺少类别：{class_name}")
            if stable_class["asset"]["sha256"] != inspection["sha256"]:
                raise RuntimeError(f"{class_name} 源 USD 已变化，必须重新执行物理沉降")
            world_from_asset = tuple(
                tuple(float(value) for value in row)
                for row in stable_class["pose"]["settled_world_from_asset"]
            )
            asset_from_object = tuple(
                tuple(float(value) for value in row) for row in definition["asset_from_canonical"]
            )
            world_from_object = matrix_multiply(world_from_asset, asset_from_object)
            root_path = f"/World/SyntheticData/CalibrationObjects/{class_name}"
            root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
            _set_transform(root, world_from_asset)
            asset_prim_path = f"{root_path}/Asset"
            asset_prim = add_reference_with_unit_scale(
                stage,
                asset_path,
                asset_prim_path,
                inspection["meters_per_unit"],
                scene_mpu,
            )
            _add_semantics(asset_prim, class_name)
            # 单类别分片无需切换可见性；避免在 SyntheticData 初始化前先隐藏语义实例。
            if len(selected_definitions) > 1:
                UsdGeom.Imageable(root).MakeInvisible()
            sampling_object = definition["sampling_center_canonical_m"]
            sampling_world = [
                sum(world_from_object[row][column] * sampling_object[column] for column in range(3))
                + world_from_object[row][3]
                for row in range(3)
            ]
            object_runtime[class_name] = {
                "definition": definition,
                "inspection": inspection,
                "root": root,
                "asset_prim_path": asset_prim_path,
                "world_from_asset": world_from_asset,
                "world_from_object": world_from_object,
                "sampling_center_world": sampling_world,
                "asset_path": asset_path,
                "stable_pose": stable_class,
            }

        camera_path = str(camera_config["prim_path"])
        UsdGeom.Xform.Define(stage, str(Path(camera_path).parent))
        camera = UsdGeom.Camera.Define(stage, camera_path)
        hfov = float(camera_config["intrinsics"]["horizontal_fov_degrees"])
        vfov = float(camera_config["intrinsics"]["vertical_fov_degrees"])
        focal_length = 10.0
        camera.CreateFocalLengthAttr().Set(focal_length)
        camera.CreateHorizontalApertureAttr().Set(
            2.0 * focal_length * math.tan(math.radians(hfov) * 0.5)
        )
        camera.CreateVerticalApertureAttr().Set(
            2.0 * focal_length * math.tan(math.radians(vfov) * 0.5)
        )
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*camera_config["clipping_range_m"]))

        dome = UsdLux.DomeLight.Define(stage, "/World/SyntheticData/CalibrationDome")
        dome.CreateIntensityAttr().Set(650.0)
        distant = UsdLux.DistantLight.Define(stage, "/World/SyntheticData/CalibrationDistant")
        distant.CreateIntensityAttr().Set(2200.0)
        distant.CreateAngleAttr().Set(1.0)
        _set_transform(
            distant.GetPrim(),
            look_at_world_from_usd_camera((1.0, 0.5, 3.0), [table_center[0], table_center[1], table["floor_z_m"]]),
        )

        render_product = rep.create.render_product(camera_path, (width, height), name="RadiusCalibration")
        rgb_annotator = rep.annotators.get("rgb", device="cpu")
        instance_annotator = rep.annotators.get(
            "instance_segmentation_fast", init_params={"colorize": False}, device="cpu"
        )
        depth_annotator = rep.annotators.get("distance_to_image_plane", device="cpu")
        annotators = [rgb_annotator, instance_annotator, depth_annotator]
        for annotator in annotators:
            annotator.attach(render_product)
        for _ in range(int(calibration_config["warmup_frames"])):
            app.update()

        intrinsics = intrinsics_from_fov(width, height, hfov, vfov)
        sampling_rotation = assets_config["sampling_frame"]["rotation_world_from_sampling"]
        quality = calibration_config["quality"]
        records: list[dict[str, Any]] = []
        preview_best: dict[str, float] = {}
        preview_paths: list[str] = []
        stop = False
        for definition in selected_definitions:
            class_name = definition["name"]
            runtime = object_runtime[class_name]
            if len(selected_definitions) > 1:
                UsdGeom.Imageable(runtime["root"]).MakeVisible()
            for _ in range(2):
                app.update()
            for viewpoint in viewpoints:
                for radius in radii:
                    if args.max_records is not None and len(records) >= args.max_records:
                        stop = True
                        break
                    camera_position, world_from_camera = _camera_pose(
                        runtime["sampling_center_world"],
                        sampling_rotation,
                        radius,
                        float(viewpoint["theta_deg"]),
                        float(viewpoint["phi_deg"]),
                        float(viewpoint["psi_deg"]),
                    )
                    _set_transform(camera.GetPrim(), world_from_camera)
                    rep.orchestrator.step(rt_subframes=int(calibration_config["rt_subframes"]))
                    rgb = np.asarray(_extract_array(rgb_annotator.get_data()))
                    instance_output = instance_annotator.get_data()
                    instance = np.asarray(_extract_array(instance_output))
                    depth = np.asarray(_extract_array(depth_annotator.get_data()), dtype=np.float32)
                    target_ids = _target_instance_ids(
                        instance_output, class_name, runtime["asset_prim_path"]
                    )
                    mask = np.isin(instance, target_ids)
                    bbox = _bbox_from_mask(mask)
                    mask_pixels = int(np.count_nonzero(mask))
                    valid_depth_mask = np.isfinite(depth) & (depth > 0.0)
                    valid_depth_pixels = int(np.count_nonzero(mask & valid_depth_mask))
                    depth_valid_ratio = valid_depth_pixels / mask_pixels if mask_pixels else 0.0
                    camera_from_asset = camera_from_object(
                        world_from_camera, runtime["world_from_asset"]
                    )
                    projected_bbox, corner_depths = _project_aabb_unclipped(
                        runtime["inspection"]["bounds_m"], camera_from_asset, intrinsics
                    )
                    truncation_ratio = (
                        1.0
                        if projected_bbox is None
                        else projected_bbox_truncation_ratio(projected_bbox, width, height)
                    )
                    scale = bbox_width_scale(bbox, width)
                    reasons = []
                    if not target_ids:
                        reasons.append("target_instance_missing")
                    if mask_pixels < int(quality["min_mask_pixels"]):
                        reasons.append("mask_pixels_too_few")
                    if depth_valid_ratio < float(quality["min_valid_depth_ratio"]):
                        reasons.append("depth_valid_ratio_low")
                    if truncation_ratio > float(quality["max_truncation_ratio"]):
                        reasons.append("fov_truncation")
                    near, far = (float(value) for value in camera_config["clipping_range_m"])
                    if min(corner_depths) < near or max(corner_depths) > far:
                        reasons.append("cad_corner_outside_depth_range")
                    record = {
                        "class": class_name,
                        "class_id": int(definition["class_id"]),
                        "object_pose_id": calibration_config["object_pose_id"],
                        "viewpoint_id": viewpoint["id"],
                        "theta_deg": float(viewpoint["theta_deg"]),
                        "phi_deg": float(viewpoint["phi_deg"]),
                        "psi_deg": float(viewpoint["psi_deg"]),
                        "theta": float(viewpoint["theta_deg"]),
                        "phi": float(viewpoint["phi_deg"]),
                        "psi": float(viewpoint["psi_deg"]),
                        "r": radius,
                        "r_m": radius,
                        "s": scale,
                        "s_pre_occlusion": scale,
                        "mask_bbox_xyxy": bbox,
                        "mask_pixels": mask_pixels,
                        "valid_depth_pixels": valid_depth_pixels,
                        "depth_valid_ratio": depth_valid_ratio,
                        "truncation_ratio": truncation_ratio,
                        "projected_cad_aabb_xyxy_unclipped": projected_bbox,
                        "cad_corner_depth_range_m": [min(corner_depths), max(corner_depths)],
                        "camera_position_world_m": camera_position,
                        "usable_for_solver": not reasons,
                        "reject_reasons": reasons,
                    }
                    records.append(record)
                    if len(records) % 10 == 0 or len(records) == expected_records:
                        print(
                            f"CALIBRATION_PROGRESS={len(records)}/{expected_records} "
                            f"class={class_name} viewpoint={viewpoint['id']} r={radius:.2f}",
                            flush=True,
                        )
                    if viewpoint["id"] == "frontal_mid" and record["usable_for_solver"]:
                        difference = abs(scale - 0.35)
                        if difference < preview_best.get(class_name, float("inf")):
                            preview_best[class_name] = difference
                            preview_path = report_dir / "preview" / f"{class_name}_scale_reference.png"
                            _save_reference_preview(preview_path, rgb, mask, record)
                            if str(preview_path) not in preview_paths:
                                preview_paths.append(str(preview_path))
                if stop:
                    break
            if len(selected_definitions) > 1:
                UsdGeom.Imageable(runtime["root"]).MakeInvisible()
            if stop:
                break

        scale_bins = collection_config["sampling"]["scale_bins"]
        classes_result: dict[str, Any] = {}
        issues = []
        warnings = []
        for definition in selected_definitions:
            class_name = definition["name"]
            viewpoint_results = {}
            viewpoint_solutions = []
            class_records = [item for item in records if item["class"] == class_name]
            for viewpoint in viewpoints:
                selected = [
                    item for item in class_records if item["viewpoint_id"] == viewpoint["id"]
                ]
                solved = solve_scale_bins(
                    selected,
                    scale_bins,
                    float(calibration_config["radius_margin_m"]),
                )
                viewpoint_solutions.append(solved)
                usable_scales = [
                    item["s_pre_occlusion"] for item in selected if item["usable_for_solver"]
                ]
                viewpoint_results[viewpoint["id"]] = {
                    "angles_deg": {
                        "theta": float(viewpoint["theta_deg"]),
                        "phi": float(viewpoint["phi_deg"]),
                        "psi": float(viewpoint["psi_deg"]),
                    },
                    "usable_record_count": sum(item["usable_for_solver"] for item in selected),
                    "observed_usable_scale_range": (
                        [min(usable_scales), max(usable_scales)] if usable_scales else None
                    ),
                    "monotonic_violation_count": _monotonic_violations(selected),
                    "scale_bins": solved,
                    "records": selected,
                }
            aggregate = aggregate_viewpoint_solutions(viewpoint_solutions, scale_bins)
            usable_class_records = [item for item in class_records if item["usable_for_solver"]]
            usable_class_scales = [item["s_pre_occlusion"] for item in usable_class_records]
            if len(usable_class_records) < 3:
                issues.append(f"{class_name} 的可用标定点不足 3 个")
            for bin_name, solution in aggregate.items():
                if not solution["target_reachable"]:
                    warnings.append(
                        f"{class_name} 在当前 D405 深度范围和代表视角中无法达到 {bin_name} "
                        f"目标尺度 {solution['target_scale']:.3f}。"
                    )
                elif not solution["full_bin_reachable_in_all_viewpoints"]:
                    warnings.append(
                        f"{class_name} 的 {bin_name} 区间不能在全部代表视角完整覆盖；"
                        "阶段四应按视角使用具体搜索边界。"
                    )
            classes_result[class_name] = {
                "class_id": int(definition["class_id"]),
                "object_pose_id": calibration_config["object_pose_id"],
                "asset_path": str(object_runtime[class_name]["asset_path"]),
                "sampling_center_world_m": object_runtime[class_name]["sampling_center_world"],
                "record_count": len(class_records),
                "usable_record_count": len(usable_class_records),
                "observed_usable_radius_range_m": (
                    [
                        min(item["r_m"] for item in usable_class_records),
                        max(item["r_m"] for item in usable_class_records),
                    ]
                    if usable_class_records
                    else None
                ),
                "observed_usable_scale_range": (
                    [min(usable_class_scales), max(usable_class_scales)]
                    if usable_class_scales
                    else None
                ),
                "aggregate_scale_bins": aggregate,
                "viewpoints": viewpoint_results,
            }

        if len(records) != expected_records:
            issues.append(f"实际记录数 {len(records)} 与期望 {expected_records} 不一致")
        gpu = _gpu_info()
        textures = _texture_status()
        if not gpu["available"]:
            issues.append("未检测到 NVIDIA GPU")
        if not textures["all_present"]:
            warnings.append("实验室场景的地面纹理仍有缺失")
        curves_path = (
            report_dir / "shards" / f"{args.class_name}_curves.png"
            if shard_mode
            else report_dir / "radius_scale_curves.png"
        )
        result = {
            "schema_version": 1,
            "artifact_kind": "radius_calibration_shard" if shard_mode else "radius_calibration",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if not issues else "fail",
            "definition": {
                "r": "camera_optical_center_to_object_sampling_center_m",
                "s": "s_pre_occlusion",
                "s_pre_occlusion": "full_unoccluded_instance_mask_bbox_width_over_image_width",
                "truncation_ratio": "fraction_of_projected_cad_aabb_area_outside_image",
                "external_occluders_hidden": True,
                "isolated_calibration_stage": True,
                "stable_pose_source": str(stable_pose_path),
                "stable_pose_is_physics_settled_in_real_table_box": True,
                "visual_geometry_simplified": False,
                "collision_approximation_used_only_for_settling": stable_poses["physics"][
                    "dynamic_collision_approximation"
                ],
            },
            "environment": {
                "renderer": renderer,
                "gpu": gpu,
                "texture_assets": textures,
            },
            "camera": {
                "name": camera_config["name"],
                "resolution": [width, height],
                "intrinsics_source": camera_config["intrinsics"]["source"],
                "intrinsics": intrinsics,
                "clipping_range_m": [float(value) for value in camera_config["clipping_range_m"]],
            },
            "source_scene_path_for_later_collection": str(scene_path),
            "stable_pose_path": str(stable_pose_path),
            "object_pose_id": calibration_config["object_pose_id"],
            "radii_m": radii,
            "representative_viewpoints": viewpoints,
            "quality_thresholds": quality,
            "scale_bins": scale_bins,
            "classes": classes_result,
            "records": records,
            "warnings": warnings,
            "issues": issues,
            "outputs": {
                "calibration_json": str(output_json),
                "curves": str(curves_path),
                "reference_previews": sorted(preview_paths),
            },
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        _plot_curves(result, curves_path)
        output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path = (
            report_dir / "shards" / f"{args.class_name}_report.md"
            if shard_mode
            else report_dir / "radius_calibration_report.md"
        )
        _write_markdown(result, markdown_path)
        print(
            f"{'RADIUS_SHARD' if shard_mode else 'RADIUS_CALIBRATION'}={output_json}",
            flush=True,
        )
        print(f"RADIUS_REPORT={markdown_path}", flush=True)
        print(f"RECORDS={len(records)} WARNINGS={len(warnings)} STATUS={result['status'].upper()}", flush=True)
        for class_name, class_result in classes_result.items():
            print(
                f"CLASS={class_name} usable={class_result['usable_record_count']}/"
                f"{class_result['record_count']} scale_range={class_result['observed_usable_scale_range']}",
                flush=True,
            )
        return 0 if result["status"] == "pass" else 1
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
