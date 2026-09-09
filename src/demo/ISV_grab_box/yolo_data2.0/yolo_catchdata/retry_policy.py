"""Deterministic retry adjustments for phase-one renders."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping

from yolo_catchdata.hand_randomization import sample_hand_perturbation


def _retry_rng(sample_key: str, attempt: int) -> random.Random:
    digest = hashlib.sha256(f"{sample_key}:retry:{attempt}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def adjust_retry(
    *,
    sample_key: str,
    attempt: int,
    reasons: list[str],
    base_distance_m: float,
    base_perturbation: Mapping[str, float],
    distance_stage: str,
    retry_config: Mapping[str, Any],
    hand_stage_limits: Mapping[str, Mapping[str, float]],
    hand_limit_scale: float = 1.0,
    minimum_distance_m: float = 0.0,
    maximum_distance_m: float = 0.50,
) -> dict[str, Any]:
    """Return the next distance and perturbation from the previous failure."""

    if attempt <= 0:
        return {
            "distance_from_grasp_m": float(base_distance_m),
            "hand_perturbation": dict(base_perturbation),
            "strategy": "initial",
        }
    # grasp 是右手标定/左手镜像的精确终点。失败候选由下一轮换工件落点/yaw 重采，
    # 不能通过后退手掌伪装成 grasp 层样本。
    if distance_stage == "grasp":
        return {
            "distance_from_grasp_m": 0.0,
            "hand_perturbation": {
                name: 0.0 for name in base_perturbation
            },
            "strategy": "keep_exact_grasp_anchor",
        }
    reason_set = set(reasons)
    distance = float(base_distance_m)
    perturbation = dict(base_perturbation)
    strategy = "shrink_perturbation"
    if (
        "target_too_truncated" in reason_set
        or "visibility_reference_clipped" in reason_set
    ):
        distance += float(retry_config.get("truncated_distance_step_m", 0.06)) * attempt
        strategy = "back_off_and_shrink_perturbation"
    elif "target_too_occluded" in reason_set:
        # The box-aware trajectory gains height as it retreats.  A modest
        # back-off plus a smaller perturbation usually exposes a target that
        # was partially hidden by the rim or the hand.
        distance += float(retry_config.get("occluded_distance_step_m", 0.05)) * attempt
        strategy = "raise_sight_line_and_shrink_perturbation"
    elif "empty_mask" in reason_set or "target_instance_missing" in reason_set:
        rng = _retry_rng(sample_key, attempt)
        step = float(retry_config.get("empty_mask_distance_step_m", 0.06))
        distance += rng.uniform(0.75, 1.25) * step * attempt
        strategy = "resample_distance_and_centered_perturbation"
        if bool(retry_config.get("empty_mask_resample_perturbation", True)):
            perturbation = sample_hand_perturbation(
                rng,
                distance_stage,
                hand_stage_limits,
                scale=(
                    float(retry_config.get("empty_mask_resample_scale", 0.55))
                    * float(hand_limit_scale)
                ),
            )
    elif "depth_valid_ratio_low" in reason_set:
        distance += float(retry_config.get("depth_invalid_distance_step_m", 0.03)) * attempt
        strategy = "back_off_for_valid_depth"
    elif "mask_pixels_too_few" in reason_set:
        distance -= float(retry_config.get("small_mask_distance_step_m", 0.03)) * attempt
        strategy = "move_closer_and_shrink_perturbation"

    if strategy != "resample_distance_and_centered_perturbation":
        scale = float(retry_config.get("perturbation_scale_after_retry", 0.80)) ** attempt
        perturbation = {name: float(value) * scale for name, value in base_perturbation.items()}
    minimum_distance = float(minimum_distance_m)
    maximum_distance = float(maximum_distance_m)
    if maximum_distance < minimum_distance:
        raise ValueError("maximum_distance_m 不能小于 minimum_distance_m")
    # A retry must stay inside the requested trajectory stratum.  Otherwise a
    # pre-grasp sample that backs off for visibility can silently become a
    # near sample while retaining the pre-grasp label.
    distance = min(maximum_distance, max(minimum_distance, distance))
    return {
        "distance_from_grasp_m": distance,
        "hand_perturbation": perturbation,
        "strategy": strategy,
    }


__all__ = ["adjust_retry"]
