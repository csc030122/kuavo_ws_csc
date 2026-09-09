#!/usr/bin/env python3
"""按仓库源模型修正 S63+D-hand 腕部相机安装关系并导出新场景。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISV_ROOT = PROJECT_ROOT.parent
REPO_ROOT = next(parent for parent in PROJECT_ROOT.parents if (parent / ".git").exists())

DEFAULT_SOURCE_SCENE = PROJECT_ROOT / "yolo_scene/lab_yolo_grab_data.usd"
DEFAULT_OUTPUT_SCENE = PROJECT_ROOT / "yolo_scene/lab_yolo_grab_data_s63_mount_fixed.usd"
EXTENDED_S63_MODEL = ISV_ROOT / "biped_s63/xml/robots/biped_s63_Dhand_fixed.xml"
STANDARD_S63_URDF = REPO_ROOT / "src/kuavo_assets/models/biped_s63/urdf/biped_s63.urdf"
S49_URDF = REPO_ROOT / "src/kuavo_assets/models/biped_s49/urdf/biped_s49.urdf"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "output/wrist_mount_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, default=DEFAULT_SOURCE_SCENE)
    parser.add_argument("--output-scene", type=Path, default=DEFAULT_OUTPUT_SCENE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def _floats(text: str | None, default: tuple[float, ...]) -> list[float]:
    if not text:
        return list(default)
    return [float(value) for value in text.split()]


def _quat_from_rpy(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _same_rotation(left: list[float], right: list[float], tolerance: float = 1e-6) -> bool:
    direct = max(abs(a - b) for a, b in zip(left, right))
    negated = max(abs(a + b) for a, b in zip(left, right))
    return min(direct, negated) <= tolerance


def _body(root: ET.Element, name: str) -> ET.Element:
    matches = root.findall(f".//body[@name='{name}']")
    if len(matches) != 1:
        raise RuntimeError(f"源模型中应当且仅应当存在一个 body：{name}，实际 {len(matches)} 个")
    return matches[0]


def _joint(root: ET.Element, name: str) -> ET.Element:
    matches = root.findall(f".//joint[@name='{name}']")
    if len(matches) != 1:
        raise RuntimeError(f"URDF 中应当且仅应当存在一个 joint：{name}，实际 {len(matches)} 个")
    return matches[0]


def _read_repository_mounts() -> dict[str, Any]:
    extended_root = ET.parse(EXTENDED_S63_MODEL).getroot()
    standard_root = ET.parse(STANDARD_S63_URDF).getroot()
    s49_root = ET.parse(S49_URDF).getroot()

    standard_link_names = {link.get("name") for link in standard_root.findall("link")}
    absent_from_standard = {
        name: name not in standard_link_names
        for name in ("l_palm", "r_palm", "l_hand_camera", "r_hand_camera")
    }
    if not all(absent_from_standard.values()):
        raise RuntimeError("标准 S63 URDF 的末端资产边界发生变化，请重新人工核对")

    sides: dict[str, Any] = {}
    for side in ("left", "right"):
        prefix = "l" if side == "left" else "r"
        tripod = _body(extended_root, f"{prefix}_hand_tripod")
        camera = _body(extended_root, f"{prefix}_hand_camera_link")
        palm = _body(extended_root, f"{prefix}_palm")
        tripod_joint = _joint(s49_root, f"{prefix}_hand_tripod_joint")
        camera_joint = _joint(s49_root, f"{prefix}_hand_camera_joint")
        tripod_origin = tripod_joint.find("origin")
        camera_origin = camera_joint.find("origin")
        if tripod_origin is None or camera_origin is None:
            raise RuntimeError(f"S49 URDF 缺少 {side} 侧安装 origin")

        extended_tripod_xyz = _floats(tripod.get("pos"), (0.0, 0.0, 0.0))
        extended_tripod_quat = _floats(tripod.get("quat"), (1.0, 0.0, 0.0, 0.0))
        urdf_tripod_xyz = _floats(tripod_origin.get("xyz"), (0.0, 0.0, 0.0))
        urdf_tripod_quat = _quat_from_rpy(
            _floats(tripod_origin.get("rpy"), (0.0, 0.0, 0.0))
        )
        extended_camera_xyz = _floats(camera.get("pos"), (0.0, 0.0, 0.0))
        extended_camera_quat = _floats(camera.get("quat"), (1.0, 0.0, 0.0, 0.0))
        urdf_camera_xyz = _floats(camera_origin.get("xyz"), (0.0, 0.0, 0.0))
        urdf_camera_quat = _quat_from_rpy(
            _floats(camera_origin.get("rpy"), (0.0, 0.0, 0.0))
        )

        if max(abs(a - b) for a, b in zip(extended_tripod_xyz, urdf_tripod_xyz)) > 1e-9:
            raise RuntimeError(f"{side} 侧 tripod 平移在两份仓库源模型中不一致")
        if not _same_rotation(extended_tripod_quat, urdf_tripod_quat):
            raise RuntimeError(f"{side} 侧 tripod 旋转在两份仓库源模型中不一致")
        if max(abs(a - b) for a, b in zip(extended_camera_xyz, urdf_camera_xyz)) > 1e-9:
            raise RuntimeError(f"{side} 侧 camera 平移在两份仓库源模型中不一致")
        if not _same_rotation(extended_camera_quat, urdf_camera_quat):
            raise RuntimeError(f"{side} 侧 camera 旋转在两份仓库源模型中不一致")

        sides[side] = {
            "prefix": prefix,
            "r7_to_tripod": {
                "translation_m": extended_tripod_xyz,
                "quaternion_wxyz": extended_tripod_quat,
            },
            "tripod_to_camera_link": {
                "translation_m": extended_camera_xyz,
                "quaternion_wxyz": extended_camera_quat,
            },
            "r7_to_palm": {
                "translation_m": _floats(palm.get("pos"), (0.0, 0.0, 0.0)),
                "quaternion_wxyz": _floats(
                    palm.get("quat"), (1.0, 0.0, 0.0, 0.0)
                ),
            },
        }
    return {
        "standard_s63_has_no_dhand_or_wrist_camera_mount": True,
        "standard_s63_absent_links": absent_from_standard,
        "extended_s63_model": str(EXTENDED_S63_MODEL),
        "s49_mount_reference": str(S49_URDF),
        "sides": sides,
    }


def _find_unique_prim(stage: Any, name: str) -> Any:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == name]
    if len(matches) != 1:
        raise RuntimeError(f"场景中应当且仅应当存在一个 Prim：{name}，实际 {len(matches)} 个")
    return matches[0]


def _quat_list(value: Any) -> list[float]:
    return [
        float(value.GetReal()),
        *(float(component) for component in value.GetImaginary()),
    ]


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S63 腕部相机与灵巧手安装关系核对",
        "",
        f"状态：**{'通过' if result['status'] == 'pass' else '失败'}**",
        "",
        "## 结论",
        "",
        "- 标准 S63 URDF 只提供腕部末端标记坐标系，不包含 D-hand 和腕部相机机械安装定义。",
        "- 当前项目使用的 S63+D-hand 扩展模型以 S49 灵巧手安装数据为依据。两份仓库源模型都规定 `zarm_*7_link → *_hand_tripod` 为零平移、单位旋转。",
        "- 原采集场景在该层额外写入了左右各 90° 的绕 Z 轴旋转，已在新场景中改回单位旋转。",
        "- `r7 → palm`、`tripod → camera_link` 和相机光学轴定义均保持不变。",
        "",
        "## 仓库依据",
        "",
        f"- 标准 S63：`{result['repository']['standard_s63_urdf']}`",
        f"- S63+D-hand 扩展模型：`{result['repository']['extended_s63_model']}`",
        f"- S49 安装参照：`{result['repository']['s49_mount_reference']}`",
        "",
        "## 修正值",
        "",
        "| 侧别 | 变换 | 平移/m | 四元数 wxyz |",
        "|---|---|---|---|",
    ]
    for side in ("left", "right"):
        side_cn = "左" if side == "left" else "右"
        data = result["mounts"][side]
        for key, label in (
            ("r7_to_tripod", "r7 → tripod"),
            ("tripod_to_camera_link", "tripod → camera_link"),
            ("r7_to_palm", "r7 → palm"),
        ):
            transform = data[key]
            lines.append(
                f"| {side_cn} | {label} | `{transform['translation_m']}` | "
                f"`{transform['quaternion_wxyz']}` |"
            )
    lines.extend(
        [
            "",
            "## 场景输出",
            "",
            f"- 原场景保留：`{result['source_scene']}`",
            f"- 修正场景：`{result['output_scene']}`",
            "",
            "注意：手眼安装关系变化后，旧的手—工件最小半径、腕部极值图和后续角度扫描结果都不能继续作为正式标定结果，需要以新场景重新生成。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        mounts = _read_repository_mounts()
        if args.source_scene.resolve() == args.output_scene.resolve():
            raise RuntimeError("输出场景必须使用新文件名，不能覆盖原场景")
        if not args.source_scene.is_file():
            raise FileNotFoundError(args.source_scene)
        args.output_scene.parent.mkdir(parents=True, exist_ok=True)
        args.report_dir.mkdir(parents=True, exist_ok=True)

        from isaacsim import SimulationApp

        app = SimulationApp(
            {
                "headless": True,
                "fast_shutdown": True,
                "multi_gpu": False,
                "disable_viewport_updates": True,
            }
        )
        try:
            from pxr import Gf, Sdf, Usd, UsdGeom

            source_stage = Usd.Stage.Open(str(args.source_scene))
            if source_stage is None:
                raise RuntimeError(f"无法打开原场景：{args.source_scene}")
            if not source_stage.Export(str(args.output_scene)):
                raise RuntimeError(f"无法导出新场景：{args.output_scene}")

            output_stage = Usd.Stage.Open(str(args.output_scene))
            if output_stage is None:
                raise RuntimeError(f"无法重新打开新场景：{args.output_scene}")
            edits: dict[str, Any] = {}
            for side in ("left", "right"):
                prefix = mounts["sides"][side]["prefix"]
                tripod = _find_unique_prim(output_stage, f"{prefix}_hand_tripod")
                orient_ops = [
                    op
                    for op in UsdGeom.Xformable(tripod).GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeOrient
                ]
                if len(orient_ops) != 1:
                    raise RuntimeError(
                        f"{tripod.GetPath()} 应当且仅应当有一个 orient op，实际 {len(orient_ops)} 个"
                    )
                before = _quat_list(orient_ops[0].Get())
                orient_ops[0].Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
                tripod.CreateAttribute(
                    "kuavo:wristMountSource", Sdf.ValueTypeNames.String, custom=True
                ).Set(str(EXTENDED_S63_MODEL))
                tripod.CreateAttribute(
                    "kuavo:wristMountVerified", Sdf.ValueTypeNames.Bool, custom=True
                ).Set(True)
                edits[side] = {
                    "prim_path": str(tripod.GetPath()),
                    "before_quaternion_wxyz": before,
                    "after_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            output_stage.GetRootLayer().Save()

            check_stage = Usd.Stage.Open(str(args.output_scene))
            if check_stage is None:
                raise RuntimeError("新场景保存后无法重新打开")
            for side in ("left", "right"):
                prefix = mounts["sides"][side]["prefix"]
                tripod = _find_unique_prim(check_stage, f"{prefix}_hand_tripod")
                orient = next(
                    op
                    for op in UsdGeom.Xformable(tripod).GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeOrient
                ).Get()
                if not _same_rotation(_quat_list(orient), [1.0, 0.0, 0.0, 0.0]):
                    raise RuntimeError(f"{side} 侧 tripod 修正未持久化")

            result = {
                "schema_version": 1,
                "artifact_kind": "s63_wrist_mount_validation",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "pass",
                "source_scene": str(args.source_scene.resolve()),
                "output_scene": str(args.output_scene.resolve()),
                "repository": {
                    "standard_s63_urdf": str(STANDARD_S63_URDF),
                    "standard_s63_has_no_dhand_or_wrist_camera_mount": True,
                    "extended_s63_model": mounts["extended_s63_model"],
                    "s49_mount_reference": mounts["s49_mount_reference"],
                },
                "mounts": mounts["sides"],
                "scene_edits": edits,
                "unchanged": [
                    "r7_to_palm",
                    "tripod_to_camera_link",
                    "camera_link_to_rgb_optical_camera",
                    "camera_link_to_depth_optical_camera",
                    "all_environment_and_table_box_prims",
                ],
            }
            json_path = args.report_dir / "s63_wrist_mount_validation.json"
            md_path = args.report_dir / "s63_wrist_mount_validation.md"
            json_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _write_markdown(result, md_path)
            print(f"OUTPUT_SCENE={args.output_scene}")
            print(f"REPORT={md_path}")
            print("STATUS=PASS")
            return 0
        finally:
            app.close()
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
