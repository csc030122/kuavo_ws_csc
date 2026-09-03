#!/usr/bin/env python3
"""Phase 0/1 read-only validation for the scene, workpieces and frozen frames.

Run this script with Isaac Sim Python. It never authors or saves a USD layer.
"""

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

from yolo_catchdata.config import (  # noqa: E402
    CONFIG_DIR,
    load_yaml,
    resolve_from_config,
    validate_assets_config,
)
from yolo_catchdata.frame_utils import (  # noqa: E402
    footprint_fits,
    point_inside_aabb,
    sampling_rotation_world_from_sampling,
    validate_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "assets.yaml",
        help="Phase-0 assets configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "asset_validation",
    )
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def close_enough(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _issue(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def validate_one_asset(
    definition: dict[str, Any],
    inspection: dict[str, Any],
    table_inner_size_xy: list[float],
    wall_clearance_m: float,
    scene_meters_per_unit: float,
    table_inner_bounds: dict[str, list[float]],
    table_floor_z_m: float,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    name = definition["name"]
    if inspection["default_prim"] != definition["expected_default_prim"]:
        issues.append(
            _issue(
                "error",
                "default_prim",
                f"{name}: expected {definition['expected_default_prim']}, "
                f"got {inspection['default_prim']}",
            )
        )
    if inspection["meters_per_unit"] <= 0.0:
        issues.append(_issue("error", "units", f"{name}: metersPerUnit must be positive"))
    if inspection["up_axis"].upper() != "Z":
        issues.append(
            _issue("error", "up_axis", f"{name}: expected Z-up, got {inspection['up_axis']}")
        )
    if inspection["root_xform_ops"]:
        issues.append(
            _issue(
                "error",
                "root_transform",
                f"{name}: canonical asset root must remain identity; "
                f"ops={inspection['root_xform_ops']}",
            )
        )
    if inspection["mesh_count"] == 0:
        issues.append(_issue("error", "mesh", f"{name}: asset contains no UsdGeom.Mesh"))

    metadata = inspection["canonical_metadata"]
    expected_version = definition["canonical_frame_version"]
    if metadata.get("kuavo:canonicalFrameVersion") != expected_version:
        issues.append(
            _issue(
                "error",
                "canonical_version",
                f"{name}: expected {expected_version}, "
                f"got {metadata.get('kuavo:canonicalFrameVersion')}",
            )
        )

    try:
        validate_transform(definition["asset_from_canonical"])
    except ValueError as exc:
        issues.append(_issue("error", "canonical_transform", f"{name}: {exc}"))

    size_m = inspection["bounds_m"]["size"]
    size_range = definition["expected_size_m"]
    for axis, actual, minimum, maximum in zip("xyz", size_m, size_range["min"], size_range["max"]):
        if not minimum <= actual <= maximum:
            issues.append(
                _issue(
                    "error",
                    "physical_size",
                    f"{name}: size_{axis}={actual:.6f} m outside [{minimum}, {maximum}]",
                )
            )

    sampling = definition["sampling_center_canonical_m"]
    if not point_inside_aabb(
        sampling,
        inspection["bounds_m"]["min"],
        inspection["bounds_m"]["max"],
        tolerance=0.01,
    ):
        issues.append(
            _issue(
                "error",
                "sampling_center",
                f"{name}: sampling center {sampling} is not inside/near the asset AABB",
            )
        )

    fits = footprint_fits(size_m[:2], table_inner_size_xy, wall_clearance_m)
    if not fits:
        issues.append(
            _issue(
                "error",
                "table_fit",
                f"{name}: footprint {size_m[:2]} m does not fit "
                f"table_box inner size {table_inner_size_xy} m",
            )
        )

    expected_size_units = metadata.get("kuavo:canonicalExpectedSizeUnits")
    if expected_size_units is not None:
        for actual, expected in zip(inspection["bounds_units"]["size"], expected_size_units):
            if not math.isclose(actual, expected, rel_tol=3e-5, abs_tol=3e-3):
                issues.append(
                    _issue(
                        "error",
                        "canonical_size_metadata",
                        f"{name}: USD bounds do not match kuavo:canonicalExpectedSizeUnits",
                    )
                )
                break

    inspection["class_id"] = definition["class_id"]
    inspection["class_name"] = name
    inspection["sampling_center_canonical_m"] = sampling
    inspection["asset_from_canonical"] = definition["asset_from_canonical"]
    inspection["reference_scale_for_scene"] = (
        inspection["meters_per_unit"] / scene_meters_per_unit
    )
    inspection["fits_table_box_default_footprint"] = fits
    asset_min = inspection["bounds_m"]["min"]
    asset_max = inspection["bounds_m"]["max"]
    table_min = table_inner_bounds["min_xy"]
    table_max = table_inner_bounds["max_xy"]
    table_center = [(table_min[index] + table_max[index]) * 0.5 for index in range(2)]
    asset_center = [(asset_min[index] + asset_max[index]) * 0.5 for index in range(2)]
    nominal_translation = [
        table_center[0] - asset_center[0],
        table_center[1] - asset_center[1],
        table_floor_z_m - asset_min[2],
    ]
    placed_min = [asset_min[index] + nominal_translation[index] for index in range(3)]
    placed_max = [asset_max[index] + nominal_translation[index] for index in range(3)]
    nominal_clearance = {
        "x_min": placed_min[0] - table_min[0],
        "x_max": table_max[0] - placed_max[0],
        "y_min": placed_min[1] - table_min[1],
        "y_max": table_max[1] - placed_max[1],
        "floor_error": placed_min[2] - table_floor_z_m,
    }
    no_penetration = (
        all(
            nominal_clearance[key] >= -1e-9
            for key in ("x_min", "x_max", "y_min", "y_max")
        )
        and abs(nominal_clearance["floor_error"]) <= 1e-9
    )
    if not no_penetration:
        issues.append(
            _issue(
                "error",
                "nominal_placement",
                f"{name}: centered floor placement leaves the configured table_box interior",
            )
        )
    inspection["nominal_table_placement"] = {
        "translation_world_m": nominal_translation,
        "aabb_world_m": {"min": placed_min, "max": placed_max},
        "clearance_m": nominal_clearance,
        "no_static_penetration": no_penetration,
        "note": "Static AABB check; physics settle is intentionally deferred.",
    }
    inspection["status"] = (
        "pass" if not any(item["level"] == "error" for item in issues) else "fail"
    )
    return inspection, issues


def save_previews(report: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    figure = plt.figure(figsize=(12, 10))
    for index, asset in enumerate(report["assets"], start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        minimum = np.asarray(asset["bounds_m"]["min"])
        maximum = np.asarray(asset["bounds_m"]["max"])
        center = np.asarray(asset["sampling_center_canonical_m"])
        corners = np.array(
            [
                [x, y, z]
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
                for z in (minimum[2], maximum[2])
            ]
        )
        edges = [
            (a, b)
            for a in range(8)
            for b in range(a + 1, 8)
            if np.count_nonzero(corners[a] != corners[b]) == 1
        ]
        for a, b in edges:
            axis.plot(*zip(corners[a], corners[b]), color="0.35", linewidth=1.0)
        axis.scatter(*center, color="black", marker="o", label="sampling center")
        length = max(asset["bounds_m"]["size"]) * 0.35
        colors = ("red", "green", "blue")
        labels = ("+X_O", "+Y_O", "+Z_O")
        for component, (color, label) in enumerate(zip(colors, labels)):
            endpoint = center.copy()
            endpoint[component] += length
            axis.quiver(*center, *(endpoint - center), color=color, arrow_length_ratio=0.18)
            axis.text(*endpoint, label, color=color)
        axis.set_title(
            f"{asset['class_id']}: {asset['class_name']}\n"
            f"size={tuple(round(v, 4) for v in asset['bounds_m']['size'])} m"
        )
        axis.set_xlabel("X_O [m]")
        axis.set_ylabel("Y_O [m]")
        axis.set_zlabel("Z_O [m]")
        axis.set_box_aspect(np.maximum(asset["bounds_m"]["size"], 0.02))
    figure.suptitle("Phase 1 canonical frames, sampling centers and asset bounds")
    figure.tight_layout()
    canonical_path = preview_dir / "canonical_frames_and_bounds.png"
    figure.savefig(canonical_path, dpi=160)
    plt.close(figure)
    outputs.append(str(canonical_path))

    bounds = report["table_box"]["configured_inner_bounds_world_m"]
    minimum = np.asarray(bounds["min_xy"])
    maximum = np.asarray(bounds["max_xy"])
    size = maximum - minimum
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, asset in zip(axes.flat, report["assets"]):
        asset_size = np.asarray(asset["bounds_m"]["size"][:2])
        if not footprint_fits(asset_size, size, report["table_box"]["wall_clearance_m"]):
            asset_size = asset_size[::-1]
        axis.add_patch(Rectangle(minimum, *size, fill=False, linewidth=2, label="inner box"))
        lower = (minimum + maximum - asset_size) * 0.5
        axis.add_patch(
            Rectangle(lower, *asset_size, alpha=0.45, color="tab:orange", label="asset AABB")
        )
        axis.set_xlim(minimum[0] - 0.05, maximum[0] + 0.05)
        axis.set_ylim(minimum[1] - 0.05, maximum[1] + 0.05)
        axis.set_aspect("equal")
        axis.set_title(f"{asset['class_name']}: default footprint fit")
        axis.set_xlabel("World X [m]")
        axis.set_ylabel("World Y [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
    figure.suptitle("Phase 1 static table_box placement check (top view)")
    figure.tight_layout()
    placement_path = preview_dir / "table_box_static_placement.png"
    figure.savefig(placement_path, dpi=160)
    plt.close(figure)
    outputs.append(str(placement_path))
    return outputs


def write_markdown(report: dict[str, Any], path: Path) -> None:
    overall_status = "通过" if report["status"] == "pass" else "失败"
    lines = [
        "# 阶段 0–1 资产验证报告",
        "",
        f"总体状态：**{overall_status}**",
        "",
        "## 场景",
        "",
        f"- USD: `{report['scene']['path']}`",
        f"- metersPerUnit: `{report['scene']['meters_per_unit']}`",
        f"- upAxis: `{report['scene']['up_axis']}`",
        f"- table_box Prim：`{report['table_box']['scene_prim']['path']}`",
        "",
        "## 工件",
        "",
        "| 编号 | 类别 | 尺寸（米） | metersPerUnit | 场景缩放 | 可放入料箱 | 状态 |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for asset in report["assets"]:
        size = " × ".join(f"{value:.4f}" for value in asset["bounds_m"]["size"])
        fits_text = "是" if asset["fits_table_box_default_footprint"] else "否"
        asset_status = "通过" if asset["status"] == "pass" else "失败"
        lines.append(
            f"| {asset['class_id']} | {asset['class_name']} | {size} | "
            f"{asset['meters_per_unit']:.6g} | {asset['reference_scale_for_scene']:.6g} | "
            f"{fits_text} | {asset_status} |"
        )
    lines.extend(["", "## 采样坐标系", ""])
    lines.append(
        f"`R_W_S = {report['sampling_frame']['rotation_world_from_sampling']}`"
    )
    lines.extend(["", "## 问题", ""])
    if report["issues"]:
        for issue in report["issues"]:
            level = "错误" if issue["level"] == "error" else "警告"
            lines.append(f"- **{level} / `{issue['code']}`**：{issue['message']}")
    else:
        lines.append("- 无")
    lines.extend(["", "## 预览文件", ""])
    for preview in report.get("preview_files", []):
        lines.append(f"- `{preview}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    config = load_yaml(args.config)
    issues = [_issue("error", "config", message) for message in validate_assets_config(config)]

    # SimulationApp must exist before importing pxr/Omniverse modules.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "fast_shutdown": True})
    try:
        from pxr import Usd, UsdGeom

        from yolo_catchdata.asset_loader import (
            add_reference_with_unit_scale,
            inspect_prim,
            inspect_usd,
            scene_name_candidates,
        )

        scene_path = resolve_from_config(config, config["scene"]["usd"])
        table_asset_path = resolve_from_config(config, config["table_box"]["asset_usd"])
        if not scene_path.is_file():
            issues.append(_issue("error", "scene", f"Scene USD does not exist: {scene_path}"))
            raise FileNotFoundError(scene_path)
        if not table_asset_path.is_file():
            issues.append(
                _issue("error", "table_asset", f"table_box USD does not exist: {table_asset_path}")
            )
            raise FileNotFoundError(table_asset_path)

        scene = inspect_usd(scene_path)
        expected_scene_mpu = float(config["scene"]["expected_meters_per_unit"])
        if not close_enough(scene["meters_per_unit"], expected_scene_mpu):
            issues.append(
                _issue(
                    "error",
                    "scene_units",
                    f"Scene metersPerUnit={scene['meters_per_unit']} expected={expected_scene_mpu}",
                )
            )
        if scene["up_axis"].upper() != str(config["scene"]["expected_up_axis"]).upper():
            issues.append(
                _issue("error", "scene_up_axis", f"Scene upAxis={scene['up_axis']} expected=Z")
            )

        table_config = config["table_box"]
        inner = table_config["inner_bounds_world_m"]
        inner_size_xy = [
            float(inner["max_xy"][index]) - float(inner["min_xy"][index])
            for index in range(2)
        ]
        table_asset = inspect_usd(table_asset_path)
        table_scene_prim = inspect_prim(scene_path, table_config["scene_prim_path"])
        configured_outer = table_config["outer_bounds_world_m"]
        actual_outer = table_scene_prim["world_bounds_m"]
        for bound_name in ("min", "max"):
            for axis, actual, expected in zip(
                "xyz", actual_outer[bound_name], configured_outer[bound_name]
            ):
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-3):
                    issues.append(
                        _issue(
                            "error",
                            "table_box_bounds",
                            f"table_box {bound_name}_{axis}={actual:.6f} m "
                            f"expected={expected:.6f} m",
                        )
                    )
        table_report = {
            "asset": table_asset,
            "scene_prim": table_scene_prim,
            "configured_center_world_m": table_config["center_world_m"],
            "configured_outer_bounds_world_m": table_config["outer_bounds_world_m"],
            "configured_inner_bounds_world_m": inner,
            "configured_inner_size_xy_m": inner_size_xy,
            "configured_floor_z_m": table_config["floor_z_m"],
            "wall_clearance_m": table_config["wall_clearance_m"],
            "scene_name_candidates": scene_name_candidates(
                scene_path, terms=("table_box",)
            ),
        }

        actual_rotation = sampling_rotation_world_from_sampling(
            table_config["center_world_m"], table_config["robot_base_world_m"]
        )
        configured_rotation = config["sampling_frame"]["rotation_world_from_sampling"]
        if not all(
            close_enough(
                float(actual_rotation[row][column]),
                float(configured_rotation[row][column]),
            )
            for row in range(3)
            for column in range(3)
        ):
            issues.append(
                _issue(
                    "error",
                    "sampling_frame",
                    "Configured R_W_S does not match table_box-to-robot geometry",
                )
            )

        assets = []
        for definition in config["classes"]:
            asset_path = resolve_from_config(config, definition["usd"])
            inspection = inspect_usd(asset_path)
            inspection, asset_issues = validate_one_asset(
                definition,
                inspection,
                inner_size_xy,
                float(table_config["wall_clearance_m"]),
                scene["meters_per_unit"],
                inner,
                float(table_config["floor_z_m"]),
            )
            assets.append(inspection)
            issues.extend(asset_issues)

        runtime_stage = Usd.Stage.CreateInMemory()
        runtime_world = UsdGeom.Xform.Define(runtime_stage, "/World").GetPrim()
        runtime_stage.SetDefaultPrim(runtime_world)
        UsdGeom.SetStageMetersPerUnit(runtime_stage, scene["meters_per_unit"])
        UsdGeom.SetStageUpAxis(runtime_stage, UsdGeom.Tokens.z)
        runtime_bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        definitions_by_name = {item["name"]: item for item in config["classes"]}
        for asset in assets:
            definition = definitions_by_name[asset["class_name"]]
            prim = add_reference_with_unit_scale(
                runtime_stage,
                resolve_from_config(config, definition["usd"]),
                f"/World/{asset['class_name']}",
                asset["meters_per_unit"],
                scene["meters_per_unit"],
            )
            size_stage_units = (
                runtime_bbox_cache.ComputeWorldBound(prim)
                .ComputeAlignedRange()
                .GetSize()
            )
            size_m = [
                float(value) * scene["meters_per_unit"] for value in size_stage_units
            ]
            matches = all(
                math.isclose(actual, expected, rel_tol=3e-5, abs_tol=3e-6)
                for actual, expected in zip(size_m, asset["bounds_m"]["size"])
            )
            asset["runtime_reference_check"] = {
                "instance_prim_path": str(prim.GetPath()),
                "size_m": size_m,
                "matches_source_physical_size": matches,
            }
            if not matches:
                issues.append(
                    _issue(
                        "error",
                        "runtime_unit_scale",
                        f"{asset['class_name']}: runtime reference size {size_m} "
                        "does not match source physical size",
                    )
                )

        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "read_only": True,
            "config": str(args.config),
            "scene": scene,
            "table_box": table_report,
            "sampling_frame": {
                "rotation_world_from_sampling": [list(row) for row in actual_rotation],
                "x_axis_world": [actual_rotation[row][0] for row in range(3)],
                "y_axis_world": [actual_rotation[row][1] for row in range(3)],
                "z_axis_world": [actual_rotation[row][2] for row in range(3)],
            },
            "assets": assets,
            "issues": issues,
        }
        report["status"] = (
            "fail" if any(issue["level"] == "error" for issue in issues) else "pass"
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_preview:
            report["preview_files"] = save_previews(report, args.output_dir)
        else:
            report["preview_files"] = []
        report_path = args.output_dir / "asset_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path = args.output_dir / "asset_report.md"
        write_markdown(report, markdown_path)

        print(f"ASSET_REPORT={report_path}", flush=True)
        print(f"ASSET_REPORT_MD={markdown_path}", flush=True)
        print(f"STATUS={report['status'].upper()}", flush=True)
        for asset in assets:
            print(
                f"ASSET class={asset['class_name']} size_m={asset['bounds_m']['size']} "
                f"mpu={asset['meters_per_unit']} scene_scale={asset['reference_scale_for_scene']} "
                f"fits={asset['fits_table_box_default_footprint']} status={asset['status']}",
                flush=True,
            )
        return 0 if report["status"] == "pass" else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
