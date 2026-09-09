"""Quality policy for phase-one approach images."""

from __future__ import annotations

from typing import Any

from yolo_catchdata.mesh_visibility import scene_visible_fraction_from_reference


def class_thresholds(
    rejection_config: dict[str, Any], class_name: str, distance_stage: str | None = None
) -> dict[str, float | int | bool]:
    visible = rejection_config.get("visible_fraction_of_full_mask", {})
    depth = rejection_config.get("depth_valid_ratio", {})
    reference = rejection_config.get("visibility_reference", {})
    return {
        "visible_fraction_min": float(
            visible.get("min_by_class", {}).get(class_name, visible.get("min", 0.0))
        ),
        "visible_fraction_target": float(visible.get("target", 0.50)),
        "visible_fraction_preferred_max": float(visible.get("preferred_max", 1.0)),
        "depth_valid_ratio_min": float(
            depth.get("min_by_stage_class", {})
            .get(distance_stage, {})
            .get(
                class_name,
                depth.get("min_by_class", {}).get(class_name, depth.get("min", 0.95)),
            )
        ),
        "min_mask_pixels": int(
            rejection_config.get("min_mask_pixels_by_class", {}).get(
                class_name, rejection_config.get("min_mask_pixels", 1)
            )
        ),
        "reject_if_wide_mask_touches_border": bool(
            reference.get("reject_if_wide_mask_touches_border", False)
        ),
        "reject_if_visibility_missing": bool(reference.get("reject_if_missing", True)),
    }


def evaluate_approach_quality(
    *,
    class_name: str,
    target_instance_ids: list[int],
    mask_pixels: int,
    depth_valid_ratio: float,
    visibility: dict[str, Any],
    rejection_config: dict[str, Any],
    distance_stage: str | None = None,
) -> dict[str, Any]:
    """Evaluate rendered metrics without an AABB or hand-written footprint."""

    thresholds = class_thresholds(rejection_config, class_name, distance_stage)
    in_frame_fraction = visibility.get("visible_fraction_of_full_mask")
    scene_fraction = visibility.get("scene_visible_fraction_of_full_mask")
    if scene_fraction is None:
        scene_fraction = scene_visible_fraction_from_reference(
            mask_pixels, visibility
        )["scene_visible_fraction_of_full_mask"]
    # Legacy/unit-test metadata may not contain the wide-reference pixel
    # counts.  Keep the old framing metric as a compatibility fallback; every
    # newly rendered sample records the scene-aware measurement explicitly.
    if scene_fraction is None:
        scene_fraction = in_frame_fraction
    reference_clipped = bool(visibility.get("wide_mask_touches_border", False))
    reasons: list[str] = []
    warnings: list[str] = []
    if not target_instance_ids:
        reasons.append("target_instance_missing")
    if mask_pixels <= 0:
        reasons.append("empty_mask")
    elif mask_pixels < int(thresholds["min_mask_pixels"]):
        reasons.append("mask_pixels_too_few")
    if float(depth_valid_ratio) < float(thresholds["depth_valid_ratio_min"]):
        reasons.append("depth_valid_ratio_low")
    # A clipped wide-FOV reference underestimates the complete silhouette in
    # the denominator, so its measured visible fraction is an *upper bound*.
    # It must never bypass the ordinary minimum-visibility gate: if even that
    # upper bound is below the threshold, the real coverage is certainly too
    # small.  The independent clipped-reference policy can additionally reject
    # an otherwise large but untrustworthy measurement.
    if (
        in_frame_fraction is not None
        and float(in_frame_fraction) < float(thresholds["visible_fraction_min"])
    ):
        reasons.append("target_too_truncated")
    elif (
        mask_pixels > 0
        and scene_fraction is not None
        and float(scene_fraction) < float(thresholds["visible_fraction_min"])
    ):
        reasons.append("target_too_occluded")
    if reference_clipped:
        if bool(thresholds["reject_if_wide_mask_touches_border"]):
            reasons.append("visibility_reference_clipped")
        else:
            warnings.append("visibility_reference_clipped")
    if in_frame_fraction is None:
        if bool(thresholds["reject_if_visibility_missing"]):
            reasons.append("visibility_reference_missing")
        else:
            warnings.append("visibility_reference_missing")
    return {
        "status": "pass" if not reasons else "fail",
        "quality_reasons": reasons,
        "quality_warnings": warnings,
        "thresholds": thresholds,
        "measurements": {
            "in_frame_fraction_of_full_mask": in_frame_fraction,
            "scene_visible_fraction_of_full_mask": scene_fraction,
        },
    }
