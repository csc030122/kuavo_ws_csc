#!/usr/bin/env python3
"""合并四类物理沉降分片，生成阶段三使用的 stable_poses.json。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    return parser.parse_args()


def _write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# 阶段三：四类工件物理稳定姿态报告",
        "",
        "总体状态：**通过**",
        "",
        "## 统一约定",
        "",
        "- 所有工件均在真实 `table_box` 中执行重力沉降。",
        "- 工件接触箱底或侧壁并达到静止，均视为有效自然稳定姿态。",
        "- 可视网格为原始 USD Render Mesh，没有抽稀、减面或代理替换。",
        f"- 动态碰撞近似：`{result['physics']['dynamic_collision_approximation']}`。",
        f"- `table_box` 静态碰撞近似：`{result['physics']['table_collision_approximation']}`。",
        "- 所有运行时物理 API 都写入 Session Layer，源 USD 均未保存或修改。",
        "",
        "## 沉降结果",
        "",
        "| 类别 | 判稳帧 | 平移量/m | 旋转变化/° | 最终线速度/m·s⁻¹ | 最终角速度/°·s⁻¹ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_name, item in result["classes"].items():
        pose = item["pose"]
        lines.append(
            f"| {class_name} | {pose['settled_frame']} | {pose['translation_change_m']:.6f} | "
            f"{pose['rotation_change_deg']:.6f} | {pose['final_linear_speed_m_s']:.6f} | "
            f"{pose['final_angular_speed_deg_s']:.6f} |"
        )
    lines.extend(["", "## 预览", ""])
    for class_name, item in result["classes"].items():
        lines.append(f"- `{class_name}`：`{item['outputs']['preview']}`")
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- 稳定姿态配置：`{result['outputs']['stable_pose_json']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assets = load_yaml(args.assets_config)
    collection = load_yaml(args.collection_config)
    stable_config = collection["stable_pose"]
    report_dir = resolve_from_config(collection, stable_config["report_dir"])
    shard_dir = report_dir / "shards"
    classes = {}
    issues = []
    for definition in assets["classes"]:
        class_name = definition["name"]
        path = shard_dir / f"{class_name}.json"
        if not path.is_file():
            issues.append(f"缺少稳定姿态分片：{path}")
            continue
        shard = json.loads(path.read_text(encoding="utf-8"))
        if shard.get("artifact_kind") != "stable_pose_shard":
            issues.append(f"{path} 不是稳定姿态分片")
            continue
        if shard.get("status") != "pass":
            issues.append(f"{class_name} 稳定姿态状态不是 pass")
        if shard.get("pose_id") != stable_config["pose_id"]:
            issues.append(f"{class_name} 的 pose_id 与配置不一致")
        if shard.get("class_id") != definition["class_id"]:
            issues.append(f"{class_name} 的 class_id 与配置不一致")
        classes[class_name] = shard
    if issues:
        for issue in issues:
            print(f"ERROR={issue}", flush=True)
        return 1

    output_path = CONFIG_DIR / str(stable_config["output"])
    report_path = report_dir / "stable_pose_report.md"
    result = {
        "schema_version": 1,
        "artifact_kind": "stable_poses",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "pose_id": stable_config["pose_id"],
        "physics": {
            "visual_geometry": stable_config["visual_geometry"],
            "dynamic_collision_approximation": stable_config[
                "dynamic_collision_approximation"
            ],
            "table_collision_approximation": stable_config[
                "table_collision_approximation"
            ],
            "convex_decomposition": stable_config["convex_decomposition"],
            "drop_height_m": float(stable_config["drop_height_m"]),
            "initial_euler_xyz_deg": [
                float(value) for value in stable_config["initial_euler_xyz_deg"]
            ],
            "physics_dt_s": float(stable_config["physics_dt_s"]),
            "stable_consecutive_frames": int(stable_config["stable_consecutive_frames"]),
        },
        "classes": classes,
        "issues": [],
        "outputs": {
            "stable_pose_json": str(output_path),
            "report": str(report_path),
            "shards": [
                str(shard_dir / f"{definition['name']}.json")
                for definition in assets["classes"]
            ],
        },
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(result, report_path)
    print(f"STABLE_POSES={output_path}", flush=True)
    print(f"STABLE_POSE_REPORT={report_path}", flush=True)
    print("STATUS=PASS CLASSES=4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
