"""目标投影尺度与半径标定的纯 Python 工具。"""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable


def bbox_width_scale(bbox_xyxy: Iterable[float] | None, image_width: int) -> float:
    """返回 bbox 像素宽度占图像宽度的比例。"""

    if image_width <= 0:
        raise ValueError("图像宽度必须为正数")
    if bbox_xyxy is None:
        return 0.0
    left, _, right, _ = (float(value) for value in bbox_xyxy)
    return max(0.0, right - left + 1.0) / float(image_width)


def projected_bbox_truncation_ratio(
    bbox_xyxy: Iterable[float], image_width: int, image_height: int
) -> float:
    """用未裁剪 CAD AABB 投影框估算 FOV 截断面积比例。"""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("图像尺寸必须为正数")
    left, top, right, bottom = (float(value) for value in bbox_xyxy)
    full_width = max(0.0, right - left)
    full_height = max(0.0, bottom - top)
    full_area = full_width * full_height
    if full_area <= 0.0:
        return 1.0
    clipped_left = max(0.0, left)
    clipped_top = max(0.0, top)
    clipped_right = min(float(image_width - 1), right)
    clipped_bottom = min(float(image_height - 1), bottom)
    clipped_area = max(0.0, clipped_right - clipped_left) * max(
        0.0, clipped_bottom - clipped_top
    )
    return max(0.0, min(1.0, 1.0 - clipped_area / full_area))


def interpolate_radius_for_scale(
    records: Iterable[dict[str, Any]], target_scale: float
) -> dict[str, Any] | None:
    """在可用扫描点之间线性插值得到目标尺度的半径。"""

    usable = sorted(
        (
            {"s": float(record["s_pre_occlusion"]), "r": float(record["r_m"])}
            for record in records
            if record.get("usable_for_solver", False)
        ),
        key=lambda item: item["s"],
    )
    if not usable or target_scale < usable[0]["s"] or target_scale > usable[-1]["s"]:
        return None
    for item in usable:
        if abs(item["s"] - target_scale) <= 1e-12:
            return {
                "r_m": item["r"],
                "bracket_r_m": [item["r"], item["r"]],
                "bracket_s": [item["s"], item["s"]],
            }
    for lower, upper in zip(usable, usable[1:]):
        if lower["s"] <= target_scale <= upper["s"]:
            span = upper["s"] - lower["s"]
            weight = 0.5 if span <= 1e-12 else (target_scale - lower["s"]) / span
            radius = lower["r"] + weight * (upper["r"] - lower["r"])
            return {
                "r_m": radius,
                "bracket_r_m": sorted([lower["r"], upper["r"]]),
                "bracket_s": [lower["s"], upper["s"]],
            }
    return None


def solve_scale_bins(
    records: Iterable[dict[str, Any]],
    scale_bins: dict[str, Iterable[float]],
    radius_margin_m: float,
) -> dict[str, dict[str, Any]]:
    """为一个代表视角求各尺度区间的 r0 和闭环搜索边界。"""

    records = list(records)
    usable_radii = [float(item["r_m"]) for item in records if item.get("usable_for_solver")]
    usable_scales = [
        float(item["s_pre_occlusion"]) for item in records if item.get("usable_for_solver")
    ]
    result: dict[str, dict[str, Any]] = {}
    for name, limits in scale_bins.items():
        lower, upper = (float(value) for value in limits)
        midpoint = (lower + upper) * 0.5
        lower_solution = interpolate_radius_for_scale(records, lower)
        midpoint_solution = interpolate_radius_for_scale(records, midpoint)
        upper_solution = interpolate_radius_for_scale(records, upper)
        boundary_solutions = [item for item in (lower_solution, upper_solution) if item]
        if midpoint_solution is None or not usable_radii:
            search_bounds = None
        else:
            brackets = list(midpoint_solution["bracket_r_m"])
            for solution in boundary_solutions:
                brackets.extend(solution["bracket_r_m"])
            search_bounds = [
                max(min(usable_radii), min(brackets) - radius_margin_m),
                min(max(usable_radii), max(brackets) + radius_margin_m),
            ]
        result[name] = {
            "scale_range": [lower, upper],
            "target_scale": midpoint,
            "target_reachable": midpoint_solution is not None,
            "full_bin_reachable": all(
                item is not None for item in (lower_solution, midpoint_solution, upper_solution)
            ),
            "r0_m": None if midpoint_solution is None else midpoint_solution["r_m"],
            "search_bounds_m": search_bounds,
            "radius_at_scale_lower_m": None if lower_solution is None else lower_solution["r_m"],
            "radius_at_scale_upper_m": None if upper_solution is None else upper_solution["r_m"],
            "observed_usable_scale_range": (
                [min(usable_scales), max(usable_scales)] if usable_scales else None
            ),
        }
    return result


def aggregate_viewpoint_solutions(
    viewpoint_solutions: Iterable[dict[str, dict[str, Any]]],
    scale_bins: dict[str, Iterable[float]],
) -> dict[str, dict[str, Any]]:
    """把多个代表视角的结果聚合为类别级闭环初值。"""

    solutions = list(viewpoint_solutions)
    aggregate: dict[str, dict[str, Any]] = {}
    for name, limits in scale_bins.items():
        entries = [item[name] for item in solutions]
        radii = [float(item["r0_m"]) for item in entries if item["r0_m"] is not None]
        bounds = [item["search_bounds_m"] for item in entries if item["search_bounds_m"]]
        aggregate[name] = {
            "scale_range": [float(value) for value in limits],
            "target_scale": sum(float(value) for value in limits) * 0.5,
            "representative_viewpoints": len(entries),
            "viewpoints_reaching_target": len(radii),
            "target_reachable": bool(radii),
            "full_bin_reachable_in_all_viewpoints": bool(entries)
            and all(item["full_bin_reachable"] for item in entries),
            "r0_m": median(radii) if radii else None,
            "search_bounds_m": (
                [min(item[0] for item in bounds), max(item[1] for item in bounds)]
                if bounds
                else None
            ),
        }
    return aggregate
