#!/usr/bin/env python3
"""Generate a deterministic pre-physics plan for approach-open samples.

The plan contains the random variables that the Isaac Sim collector will
realize one by one: support face, in-plane object yaw, box XY position, and
distance from grasp.  Physical settling and the final wall check happen later;
this script never declares a candidate valid by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.config import CONFIG_DIR, load_yaml
from yolo_catchdata.hand_randomization import (
    sample_hand_perturbation,
    sample_hand_perturbation_mixture,
    stratified_mixture_mode,
)
from yolo_catchdata.hand_sides import HAND_SIDES
from yolo_catchdata.pose_policy import effective_hand_yaw_delta


CLASS_NAMES = ("left", "right", "hose", "outhandle")
DISTANCE_STAGES = ("grasp", "pre_grasp", "near", "approach", "far")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-side", choices=HAND_SIDES, default="right")
    parser.add_argument("--class-name", choices=CLASS_NAMES, default=None)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="每类 sample_key 的起始编号，用于多轮配额采集时避免重复。",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/dataset/approach_open_smoke/approach_open_plan.json",
    )
    parser.add_argument("--assets-config", type=Path, default=CONFIG_DIR / "assets.yaml")
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    parser.add_argument(
        "--stage-quotas-json",
        type=Path,
        default=None,
        help="可选的 class→stage→候选数 JSON；仅生成尚未收满的配额格。",
    )
    return parser.parse_args()


def _sample_distance(
    rng: random.Random,
    distance_config: dict[str, float],
    class_name: str,
    class_ranges: dict[str, dict[str, dict[str, float]]],
    stage_index: int | None = None,
    forced_stage: str | None = None,
) -> tuple[str, float]:
    configured = class_ranges.get(class_name, class_ranges.get("default"))
    if configured is None:
        configured = {
            "grasp": {"min": 0.0, "max": 0.0},
            "pre_grasp": {"min": 0.02, "max": 0.10},
            "near": {"min": 0.10, "max": 0.20},
            "approach": {"min": 0.20, "max": 0.35},
            "far": {"min": 0.35, "max": float(distance_config["max"])},
        }
    stages = DISTANCE_STAGES
    # 轮换阶段而不是完全随机抽取，保证每类扩容时五个轨迹层都有配额。
    # 同一阶段内部的距离仍保持连续均匀随机。
    if forced_stage is not None and forced_stage not in stages:
        raise ValueError(f"未知距离阶段：{forced_stage}")
    stage = forced_stage or (
        stages[stage_index % len(stages)]
        if stage_index is not None
        else stages[rng.randrange(len(stages))]
    )
    interval = configured[stage]
    lower, upper = float(interval["min"]), float(interval["max"])
    if lower > upper:
        raise ValueError(f"{class_name}/{stage} 距离范围无效：{lower} > {upper}")
    return stage, rng.uniform(lower, upper)


def _sample_yaw_and_xy(
    rng: random.Random,
    inner_min: list[float],
    inner_max: list[float],
    margin: float,
    yaw_deg: float,
    centered_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Sample a candidate in the real table-box interior.

    The plan deliberately does not fit an AABB or a hand-written footprint.
    Exact mesh containment and wall clearance are checked after the USD asset
    is instantiated in Isaac Sim, where transformed Mesh vertices are available.
    """

    lower_x = float(inner_min[0]) + margin
    upper_x = float(inner_max[0]) - margin
    lower_y = float(inner_min[1]) + margin
    upper_y = float(inner_max[1]) - margin
    if lower_x > upper_x or lower_y > upper_y:
        raise ValueError("table_box 内部范围小于 boundary_margin_m")
    centered_scale = float(centered_scale)
    if not 0.0 < centered_scale <= 1.0:
        raise ValueError("centered_scale 必须在 (0, 1] 内")
    center_x = 0.5 * (lower_x + upper_x)
    center_y = 0.5 * (lower_y + upper_y)
    half_x = 0.5 * (upper_x - lower_x) * centered_scale
    half_y = 0.5 * (upper_y - lower_y) * centered_scale
    return (
        float(yaw_deg) % 360.0,
        rng.uniform(center_x - half_x, center_x + half_x),
        rng.uniform(center_y - half_y, center_y + half_y),
    )


def _sample_object_yaw(
    rng: random.Random,
    class_name: str,
    yaw_config: dict[str, object],
    group_index: int,
) -> float:
    """Sample a yaw that is compatible with the current no-wall-contact policy."""

    sampling = str(yaw_config.get("sampling", "uniform_0_to_360_deg"))
    if sampling == "uniform_0_to_360_deg":
        return rng.uniform(0.0, 360.0)
    if sampling == "box_long_axis_bands":
        centers = [float(value) for value in yaw_config.get("band_centers_deg", [])]
        half_width = float(yaw_config.get("band_half_width_deg", 0.0))
        if not centers or not 0.0 <= half_width < 90.0:
            raise ValueError(f"{class_name} 的箱体长轴 yaw 带配置无效")
        center = centers[int(group_index) % len(centers)]
        return (center + rng.uniform(-half_width, half_width)) % 360.0
    raise ValueError(f"{class_name} 不支持的 object yaw 采样方式：{sampling}")


def _yaw_stratum_index(group_index: int, support_face_count: int) -> int:
    """Advance yaw strata independently from the support-face cycle.

    Both support face and yaw band used to index directly by ``group_index``.
    With two faces and two yaw bands that produced only (+z, 0°) and
    (-z, 180°), never the other two combinations.  Hold each yaw stratum for
    one complete support-face cycle so the Cartesian combinations are covered.
    """

    if support_face_count <= 0:
        raise ValueError("support_face_count 必须大于 0")
    return int(group_index) // int(support_face_count)


def _stage_requests(
    samples_per_class: int,
    class_quotas: dict[str, int] | None,
) -> list[tuple[int, str]]:
    """Return ``(group_ordinal, stage)`` rows with stages grouped by scene.

    A scene group shares one settled object pose.  Quota refill can omit stages
    that are already full while the remaining stages still share candidates at
    the same ordinal.
    """

    if class_quotas is None:
        return [
            (index // len(DISTANCE_STAGES), DISTANCE_STAGES[index % len(DISTANCE_STAGES)])
            for index in range(samples_per_class)
        ]
    counts = {stage: int(class_quotas.get(stage, 0)) for stage in DISTANCE_STAGES}
    return [
        (group_ordinal, stage)
        for group_ordinal in range(max(counts.values(), default=0))
        for stage in DISTANCE_STAGES
        if group_ordinal < counts[stage]
    ]


def _split_stage_requests(
    samples_per_class: int,
    class_quotas: dict[str, object] | None,
) -> list[tuple[str | None, int, str]]:
    """Return groupable stage requests, optionally targeted to one split."""

    if class_quotas is None or not set(class_quotas).issubset(set(SPLITS)):
        return [
            (None, group_ordinal, stage)
            for group_ordinal, stage in _stage_requests(
                samples_per_class,
                class_quotas,  # type: ignore[arg-type]
            )
        ]
    requests: list[tuple[str | None, int, str]] = []
    for split in SPLITS:
        split_quotas = class_quotas.get(split, {})
        if not isinstance(split_quotas, dict):
            raise ValueError(f"{split} stage quotas 必须是映射")
        requests.extend(
            (split, group_ordinal, stage)
            for group_ordinal, stage in _stage_requests(
                samples_per_class,
                split_quotas,
            )
        )
    return requests


def main() -> int:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("samples-per-class 必须为正数")
    if args.start_index < 0:
        raise ValueError("start-index 不能为负数")
    assets = load_yaml(args.assets_config)
    collection = load_yaml(args.collection_config)
    approach = collection["approach_open"]
    table = assets["table_box"]
    inner_min = table["inner_bounds_world_m"]["min_xy"]
    inner_max = table["inner_bounds_world_m"]["max_xy"]
    margin = float(approach["object_translation"]["boundary_margin_m"])
    object_xy_mixture = approach["object_translation"].get("mixture", {})
    classes = [args.class_name] if args.class_name else list(CLASS_NAMES)
    stage_quotas = None
    if args.stage_quotas_json is not None:
        stage_quotas = json.loads(
            args.stage_quotas_json.expanduser().resolve().read_text(encoding="utf-8")
        )
        unknown_classes = set(stage_quotas) - set(CLASS_NAMES)
        if unknown_classes:
            raise ValueError(f"stage quotas 包含未知类别：{sorted(unknown_classes)}")
        for class_name, class_quotas in stage_quotas.items():
            quota_keys = set(class_quotas)
            if quota_keys.issubset(set(SPLITS)):
                for split, split_quotas in class_quotas.items():
                    if not isinstance(split_quotas, dict):
                        raise ValueError(f"{class_name}/{split} quotas 必须是映射")
                    unknown_stages = set(split_quotas) - set(DISTANCE_STAGES)
                    if unknown_stages:
                        raise ValueError(
                            f"{class_name}/{split} quotas 包含未知阶段："
                            f"{sorted(unknown_stages)}"
                        )
                    if any(int(value) < 0 for value in split_quotas.values()):
                        raise ValueError(f"{class_name}/{split} quotas 不能为负数")
            else:
                unknown_stages = quota_keys - set(DISTANCE_STAGES)
                if unknown_stages:
                    raise ValueError(
                        f"{class_name} stage quotas 包含未知阶段："
                        f"{sorted(unknown_stages)}"
                    )
                if any(int(value) < 0 for value in class_quotas.values()):
                    raise ValueError(f"{class_name} stage quotas 不能为负数")
        if args.class_name is None:
            classes = [name for name in CLASS_NAMES if name in stage_quotas]
    distance_config = approach["palm_pose"]["distance_from_grasp_m"]
    distance_ranges_by_class = approach["palm_pose"].get("distance_sampling_by_class", {})
    hand_randomization = approach.get("hand_randomization", {})
    hand_randomization_enabled = bool(hand_randomization.get("enabled", False))
    hand_randomization_stages = hand_randomization.get("stages", {})
    hand_distribution = str(hand_randomization.get("distribution", "uniform"))
    hand_mixture = hand_randomization.get("mixture", {})
    # Preserve all existing right-hand plans for a given seed while giving the
    # independent left branch a disjoint, equally reproducible random stream.
    effective_seed = int(args.seed) + (0 if args.hand_side == "right" else 1_000_003)
    rng = random.Random(effective_seed)
    samples: list[dict[str, object]] = []
    stage_mode_ordinals = {stage: 0 for stage in DISTANCE_STAGES}
    class_xy_ordinals = {class_name: 0 for class_name in classes}
    class_ids = {item["name"]: int(item["class_id"]) for item in assets["classes"]}
    for class_name in classes:
        support_faces = list(
            approach["support_pose"]["classes"][class_name]["allowed_support_faces"]
        )
        requests = _split_stage_requests(
            args.samples_per_class,
            stage_quotas.get(class_name, {}) if stage_quotas is not None else None,
        )
        scene_randomization: dict[tuple[str | None, int], dict[str, object]] = {}
        for local_index, (requested_split, group_ordinal, forced_stage) in enumerate(requests):
            index = args.start_index + local_index
            group_key = (requested_split, group_ordinal)
            if group_key not in scene_randomization:
                group_index = args.start_index + len(scene_randomization)
                xy_mode = stratified_mixture_mode(
                    "xy",
                    args.start_index + class_xy_ordinals[class_name],
                    object_xy_mixture,
                )
                class_xy_ordinals[class_name] += 1
                xy_scale = (
                    float(
                        object_xy_mixture.get("centered_scale_by_class", {}).get(
                            class_name,
                            object_xy_mixture.get("centered_scale", 1.0),
                        )
                    )
                    if xy_mode == "centered"
                    else 1.0
                )
                yaw_config = approach["object_yaw"]["classes"][class_name]
                sampled_yaw = _sample_object_yaw(
                    rng,
                    class_name,
                    yaw_config,
                    _yaw_stratum_index(group_index, len(support_faces)),
                )
                yaw_deg, x, y = _sample_yaw_and_xy(
                    rng,
                    inner_min,
                    inner_max,
                    margin,
                    sampled_yaw,
                    centered_scale=xy_scale,
                )
                scene_randomization[group_key] = {
                    "scene_group_index": group_index,
                    "scene_group_id": f"{class_name}_group_{group_index:06d}",
                    "support_face": support_faces[group_index % len(support_faces)],
                    "xy_mode": xy_mode,
                    "yaw_deg": yaw_deg,
                    "yaw_sampling": yaw_config["sampling"],
                    "x": x,
                    "y": y,
                    "split": requested_split,
                }
            scene = scene_randomization[group_key]
            support_face = str(scene["support_face"])
            xy_mode = str(scene["xy_mode"])
            yaw_deg = float(scene["yaw_deg"])
            x, y = float(scene["x"]), float(scene["y"])
            distance_stage, distance = _sample_distance(
                rng,
                distance_config,
                class_name,
                distance_ranges_by_class,
                stage_index=local_index,
                forced_stage=forced_stage,
            )
            if hand_randomization_enabled and hand_distribution == "centered_mixture_with_coverage_tail":
                forced_mode = stratified_mixture_mode(
                    distance_stage,
                    args.start_index + stage_mode_ordinals[distance_stage],
                    hand_mixture,
                )
                stage_mode_ordinals[distance_stage] += 1
                hand_perturbation, hand_sampling_mode = sample_hand_perturbation_mixture(
                    rng,
                    distance_stage,
                    hand_randomization_stages,
                    hand_mixture,
                    forced_mode=forced_mode,
                )
            elif hand_randomization_enabled:
                hand_perturbation = sample_hand_perturbation(
                    rng, distance_stage, hand_randomization_stages
                )
                hand_sampling_mode = "uniform_full_range"
            else:
                hand_perturbation = {
                    "lateral_offset_1_m": 0.0,
                    "lateral_offset_2_m": 0.0,
                    "roll_delta_deg": 0.0,
                    "pitch_delta_deg": 0.0,
                    "yaw_delta_deg": 0.0,
                }
                hand_sampling_mode = "fixed"
            hand_limit_scale = float(
                hand_mixture.get("limit_scale_by_class", {}).get(class_name, 1.0)
            )
            if not 0.0 < hand_limit_scale <= 1.0:
                raise ValueError(f"{class_name} hand limit scale 必须在 (0, 1] 内")
            hand_perturbation = {
                name: float(value) * hand_limit_scale
                for name, value in hand_perturbation.items()
            }
            sample = {
                    "sample_key": f"{class_name}_{index:06d}",
                    "hand_side": args.hand_side,
                    "scene_group_id": scene["scene_group_id"],
                    "scene_group_index": scene["scene_group_index"],
                    "class_name": class_name,
                    "class_id": class_ids[class_name],
                    "support_face": support_face,
                    "object_yaw_delta_deg": yaw_deg,
                    "object_yaw_sampling": scene["yaw_sampling"],
                    "effective_hand_yaw_delta_deg": math.degrees(
                        effective_hand_yaw_delta(class_name, math.radians(yaw_deg))
                    ),
                    "drop_xy_world_m": [x, y],
                    "xy_sampling": "centered_exact_mesh_with_wall_clearance",
                    "object_xy_sampling_mode": xy_mode,
                    "distance_stage": distance_stage,
                    "distance_from_grasp_m": distance,
                    "hand_perturbation": hand_perturbation,
                    "hand_sampling_mode": hand_sampling_mode,
                    "hand_limit_scale": hand_limit_scale,
                    "requires_physics_settle": True,
                    "requires_visual_mesh_boundary_check": True,
                }
            if requested_split is not None:
                sample["split"] = requested_split
            samples.append(sample)
    payload = {
        "schema_version": 2,
        "artifact_kind": "approach_open_randomization_plan",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pre_physics_plan",
        "hand_side": args.hand_side,
        "seed": args.seed,
        "effective_seed": effective_seed,
        "classes": classes,
        "samples_per_class": args.samples_per_class if stage_quotas is None else None,
        "stage_candidate_quotas": stage_quotas,
        "candidate_count_by_class": {
            class_name: sum(item["class_name"] == class_name for item in samples)
            for class_name in classes
        },
        "sampling_frame": "world",
        "xy_strategy": "centered_exact_mesh_with_wall_clearance",
        "samples": samples,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PLAN={output}")
    print(f"STATUS={payload['status'].upper()} SAMPLES={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
