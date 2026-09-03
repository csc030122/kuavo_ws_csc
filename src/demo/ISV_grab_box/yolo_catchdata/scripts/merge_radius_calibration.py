#!/usr/bin/env python3
"""合并四个阶段三标定分片，生成最终 radius_calibration.json。"""

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

from scripts.calibrate_radius import _plot_curves, _write_markdown  # noqa: E402
from yolo_catchdata.config import CONFIG_DIR, load_yaml, resolve_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    return parser.parse_args()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> int:
    args = parse_args()
    assets_config = load_yaml(args.assets_config)
    collection_config = load_yaml(args.collection_config)
    calibration_config = collection_config["radius_calibration"]
    report_dir = resolve_from_config(collection_config, calibration_config["report_dir"])
    shard_dir = report_dir / "shards"
    definitions = assets_config["classes"]
    shards: dict[str, dict[str, Any]] = {}
    issues = []
    for definition in definitions:
        class_name = definition["name"]
        path = shard_dir / f"{class_name}.json"
        if not path.is_file():
            issues.append(f"缺少标定分片：{path}")
            continue
        shard = json.loads(path.read_text(encoding="utf-8"))
        if shard.get("artifact_kind") != "radius_calibration_shard":
            issues.append(f"{path} 不是有效的标定分片")
            continue
        if shard.get("status") != "pass":
            issues.append(f"{path} 的状态不是 pass")
        if set(shard.get("classes", {})) != {class_name}:
            issues.append(f"{path} 的类别内容不匹配 {class_name}")
        expected = len(calibration_config["radii_m"]) * len(
            calibration_config["representative_viewpoints"]
        )
        if len(shard.get("records", [])) != expected:
            issues.append(f"{class_name} 分片记录数不是 {expected}")
        shards[class_name] = shard
    if issues:
        for issue in issues:
            print(f"ERROR={issue}", flush=True)
        return 1

    first = shards[definitions[0]["name"]]
    classes = {definition["name"]: shards[definition["name"]]["classes"][definition["name"]] for definition in definitions}
    records = [
        record
        for definition in definitions
        for record in shards[definition["name"]]["records"]
    ]
    warnings = _unique(
        [warning for shard in shards.values() for warning in shard.get("warnings", [])]
    )
    final_path = CONFIG_DIR / str(calibration_config["output"])
    curves_path = report_dir / "radius_scale_curves.png"
    markdown_path = report_dir / "radius_calibration_report.md"
    result = {
        "schema_version": 1,
        "artifact_kind": "radius_calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "definition": first["definition"],
        "environment": {
            **first["environment"],
            "class_shards": {
                name: {
                    "generated_at_utc": shard["generated_at_utc"],
                    "gpu": shard["environment"]["gpu"],
                }
                for name, shard in shards.items()
            },
        },
        "camera": first["camera"],
        "object_pose_id": first["object_pose_id"],
        "radii_m": first["radii_m"],
        "representative_viewpoints": first["representative_viewpoints"],
        "quality_thresholds": first["quality_thresholds"],
        "scale_bins": first["scale_bins"],
        "classes": classes,
        "records": records,
        "warnings": warnings,
        "issues": [],
        "outputs": {
            "calibration_json": str(final_path),
            "curves": str(curves_path),
            "reference_previews": sorted(
                {
                    preview
                    for shard in shards.values()
                    for preview in shard["outputs"]["reference_previews"]
                }
            ),
            "shards": [str(shard_dir / f"{definition['name']}.json") for definition in definitions],
        },
    }
    _plot_curves(result, curves_path)
    final_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result, markdown_path)
    print(f"RADIUS_CALIBRATION={final_path}", flush=True)
    print(f"RADIUS_REPORT={markdown_path}", flush=True)
    print(f"RECORDS={len(records)} WARNINGS={len(warnings)} STATUS=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
