"""Visibility measurements derived from rendered object masks."""

from __future__ import annotations

import math
from typing import Any


def expanded_fov_degrees(original_fov_degrees: float, span_factor: float) -> float:
    """Return the wider FOV that preserves focal length on a smaller canvas.

    Rendering the wider FOV at the original resolution shrinks every projected
    length by ``span_factor``.  The central ``1/span_factor`` crop therefore
    represents the real camera frame and can be compared with the full rendered
    silhouette without constructing an AABB.
    """

    fov = float(original_fov_degrees)
    factor = float(span_factor)
    if not 0.0 < fov < 180.0:
        raise ValueError("original_fov_degrees 必须在 (0, 180) 内")
    if factor < 1.0:
        raise ValueError("span_factor 必须大于等于 1")
    tangent = factor * math.tan(math.radians(fov) * 0.5)
    return math.degrees(2.0 * math.atan(tangent))


def rendered_mesh_visibility(wide_mask: Any, span_factor: float) -> dict[str, Any]:
    """Measure original-frame coverage from a target-only wider-FOV mask."""

    import numpy as np

    mask = np.asarray(wide_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("wide_mask 必须是二维数组")
    height, width = mask.shape
    if height <= 0 or width <= 0:
        raise ValueError("wide_mask 不能为空")
    factor = float(span_factor)
    if factor < 1.0:
        raise ValueError("span_factor 必须大于等于 1")
    crop_width = max(1, min(width, int(round(width / factor))))
    crop_height = max(1, min(height, int(round(height / factor))))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    full_pixels = int(mask.sum())
    original_frame_pixels = int(mask[y0:y1, x0:x1].sum())
    touches_wide_border = bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )
    return {
        "full_mask_pixels": full_pixels,
        "original_fov_crop_mask_pixels": original_frame_pixels,
        "visible_fraction_of_full_mask": (
            float(original_frame_pixels / full_pixels) if full_pixels else None
        ),
        "wide_mask_touches_border": touches_wide_border,
        "original_fov_crop_xyxy": [x0, y0, x1 - 1, y1 - 1],
        "span_factor": factor,
    }


def scene_visible_fraction_from_reference(
    scene_mask_pixels: int,
    visibility_reference: dict[str, Any],
) -> dict[str, float | None]:
    """Compare the real scene mask with the target-only complete silhouette.

    The target-only reference is rendered at the same pixel resolution but
    with a focal length reduced by ``span_factor``.  Its area therefore needs
    to be multiplied by ``span_factor ** 2`` before it can be compared with
    the real-camera mask.  Unlike the center-crop fraction, this measurement
    includes both image truncation and occlusion by the hand/table_box.
    """

    pixels = max(0, int(scene_mask_pixels))
    full_pixels = int(visibility_reference.get("full_mask_pixels") or 0)
    span_factor = float(visibility_reference.get("span_factor") or 0.0)
    if full_pixels <= 0 or span_factor < 1.0:
        return {
            "expected_full_mask_pixels_at_real_fov": None,
            "scene_visible_fraction_of_full_mask": None,
        }
    expected_full_pixels = float(full_pixels) * span_factor * span_factor
    # Rasterization at two focal lengths can differ by a few boundary pixels,
    # so a fully visible object may measure slightly above one.
    fraction = min(1.0, float(pixels) / expected_full_pixels)
    return {
        "expected_full_mask_pixels_at_real_fov": expected_full_pixels,
        "scene_visible_fraction_of_full_mask": fraction,
    }


__all__ = [
    "expanded_fov_degrees",
    "rendered_mesh_visibility",
    "scene_visible_fraction_from_reference",
]
