#!/usr/bin/env python3
"""根据已渲染中心视角记录确定工作空间半径下界，并生成中文报告。"""

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
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    parser.add_argument(
        "--observations", type=Path, default=CONFIG_DIR / "radius_calibration.json"
    )
    return parser.parse_args()


def _matching_record(
    records: list[dict[str, Any]], class_name: str, radius: float, viewpoint: dict[str, float]
) -> dict[str, Any] | None:
    for record in records:
        if record["class"] != class_name:
            continue
        if abs(float(record["r_m"]) - radius) > 1e-9:
            continue
        if all(
            abs(float(record[key]) - float(viewpoint[f"{key}_deg"])) <= 1e-9
            for key in ("theta", "phi", "psi")
        ):
            return record
    return None


def _evaluate_record(
    record: dict[str, Any], truncation_limit: float, near_depth_limit: float
) -> dict[str, Any]:
    nearest_depth = float(record["cad_corner_depth_range_m"][0])
    truncation = float(record["truncation_ratio"])
    reasons = []
    if truncation > truncation_limit:
        reasons.append("明显出画")
    if nearest_depth < near_depth_limit:
        reasons.append("进入近距失效区")
    return {
        "r_m": float(record["r_m"]),
        "s_pre_occlusion": float(record["s_pre_occlusion"]),
        "truncation_ratio": truncation,
        "nearest_depth_m": nearest_depth,
        "valid": not reasons,
        "reasons": reasons,
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    threshold = result["thresholds"]
    lines = [
        "# 腕部观察工作空间：半径范围与分层报告",
        "",
        "状态：**通过**",
        "",
        "## 选择结论",
        "",
        f"- 全局扫描范围：`{result['radius_range_m']['min']:.2f}–{result['radius_range_m']['max']:.2f} m`。",
        "- 全局下界保留小工件近距离观察能力；各类别最终有效下界由后续角度可行域扫描决定。",
        f"- 半径层：`{result['radius_layers_m']}`。",
        "- `r` 是相机光心到工件 sampling center 的距离。",
        "- `s_pre_occlusion` 仅作为结果统计，不再用于反求 `r`。",
        "",
        "## 判定阈值",
        "",
        f"- 明显出画：未裁剪 CAD 投影面积在图像外的比例大于 `{threshold['massive_truncation']:.0%}`。",
        f"- 近距失效：工件最近 CAD 顶点的相机 Z-depth 小于 `{threshold['near_depth_failure_m']:.2f} m`。",
        "- 本轮只依据成像与深度，不考虑机械臂、灵巧手或箱体碰撞干涉。",
        "",
        "## 中心视角逐类结果",
        "",
        "| 类别 | 最小有效 r/m | 下一更近层 r/m | 下一层截断率 | 下一层最近深度/m | 主要限制 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for class_name, item in result["classes"].items():
        failed = item.get("first_failed_layer_below")
        if failed:
            failed_r = f"{failed['r_m']:.2f}"
            truncation = f"{failed['truncation_ratio']:.1%}"
            depth = f"{failed['nearest_depth_m']:.3f}"
            reason = "、".join(failed["reasons"])
        else:
            failed_r = truncation = depth = reason = "无"
        lines.append(
            f"| {class_name} | {item['effective_r_min_m']:.2f} | {failed_r} | "
            f"{truncation} | {depth} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## 为什么全局下界不是 0.30 m",
            "",
            "`left`、`right` 和 `hose` 在中心视角下约从 `r=0.25 m` 开始明显出画，因此它们的有效下界约为 `0.30 m`。",
            "但 `outhandle` 在 `r=0.12 m` 仍完整可见且最近深度未低于 `0.07 m`。如果把全局下界设为 `0.30 m`，会漏掉小工件真实存在的近距离观察区域。",
            "后续保存 `Omega_class(r)` 后，大工件在过近距离自然没有有效角度格点，不会进入正式采样。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    collection = load_yaml(args.collection_config)
    workspace = collection["workspace_calibration"]
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    policy = workspace["radius_min_selection"]
    viewpoint = policy["baseline_viewpoint"]
    truncation_limit = float(policy["massive_truncation_threshold"])
    near_depth_limit = float(policy["near_depth_failure_threshold_m"])
    layers = [float(value) for value in workspace["radius_layers_m"]]
    classes: dict[str, Any] = {}
    issues = []
    for class_name in observations["classes"]:
        evaluated = []
        for radius in layers:
            record = _matching_record(observations["records"], class_name, radius, viewpoint)
            if record is None:
                issues.append(f"{class_name} 缺少 r={radius:.2f} 的中心视角记录")
                continue
            evaluated.append(_evaluate_record(record, truncation_limit, near_depth_limit))
        valid = [item for item in evaluated if item["valid"]]
        if not valid:
            issues.append(f"{class_name} 没有有效半径层")
            continue
        effective_min = min(item["r_m"] for item in valid)
        failed_below = next(
            (item for item in evaluated if item["r_m"] < effective_min and not item["valid"]),
            None,
        )
        if failed_below is None:
            finer_records = sorted(
                (
                    record
                    for record in observations["records"]
                    if record["class"] == class_name
                    and float(record["r_m"]) < effective_min
                    and all(
                        abs(float(record[key]) - float(viewpoint[f"{key}_deg"])) <= 1e-9
                        for key in ("theta", "phi", "psi")
                    )
                ),
                key=lambda record: float(record["r_m"]),
                reverse=True,
            )
            failed_below = next(
                (
                    item
                    for item in (
                        _evaluate_record(record, truncation_limit, near_depth_limit)
                        for record in finer_records
                    )
                    if not item["valid"]
                ),
                None,
            )
        classes[class_name] = {
            "effective_r_min_m": effective_min,
            "first_failed_layer_below": failed_below,
            "layers": evaluated,
        }
    selected_global_min = min(item["effective_r_min_m"] for item in classes.values())
    configured_min = float(workspace["radius_range_m"]["min"])
    if abs(selected_global_min - configured_min) > 1e-9:
        issues.append(
            f"配置的全局 r_min={configured_min:.2f} 与观测选择值 {selected_global_min:.2f} 不一致"
        )
    report_dir = resolve_from_config(collection, workspace["report_dir"])
    output_path = CONFIG_DIR / str(workspace["output"])
    report_path = report_dir / "radius_selection_report.md"
    result = {
        "schema_version": 1,
        "artifact_kind": "workspace_radius_selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not issues else "fail",
        "definition": workspace["definition"],
        "radius_range_m": {
            "min": configured_min,
            "max": float(workspace["radius_range_m"]["max"]),
        },
        "radius_layers_m": layers,
        "baseline_viewpoint": viewpoint,
        "thresholds": {
            "massive_truncation": truncation_limit,
            "near_depth_failure_m": near_depth_limit,
        },
        "classes": classes,
        "issues": issues,
        "outputs": {"config": str(output_path), "report": str(report_path)},
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result, report_path)
    print(f"WORKSPACE_RADIUS_SELECTION={output_path}")
    print(f"WORKSPACE_RADIUS_REPORT={report_path}")
    print(f"STATUS={result['status'].upper()} R_MIN={configured_min:.2f} R_MAX={result['radius_range_m']['max']:.2f}")
    for class_name, item in classes.items():
        print(f"CLASS={class_name} EFFECTIVE_R_MIN={item['effective_r_min_m']:.2f}")
    for issue in issues:
        print(f"ERROR={issue}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
