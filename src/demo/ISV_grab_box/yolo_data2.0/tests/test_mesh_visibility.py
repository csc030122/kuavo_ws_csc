from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - Isaac's Python includes NumPy
    np = None

from yolo_catchdata.mesh_visibility import (
    expanded_fov_degrees,
    rendered_mesh_visibility,
    scene_visible_fraction_from_reference,
)


@unittest.skipIf(np is None, "NumPy is provided by the Isaac Sim Python runtime")
class MeshVisibilityTest(unittest.TestCase):
    def test_expanded_fov_preserves_tangent_scale(self) -> None:
        expanded = expanded_fov_degrees(60.0, 3.0)
        self.assertAlmostEqual(
            np.tan(np.radians(expanded) * 0.5),
            3.0 * np.tan(np.radians(60.0) * 0.5),
        )

    def test_visibility_uses_actual_mask_in_center_crop(self) -> None:
        mask = np.zeros((8, 12), dtype=bool)
        mask[2:6, 3:9] = True
        result = rendered_mesh_visibility(mask, 2.0)
        self.assertEqual(result["full_mask_pixels"], 24)
        self.assertEqual(result["original_fov_crop_xyxy"], [3, 2, 8, 5])
        self.assertEqual(result["original_fov_crop_mask_pixels"], 24)
        self.assertEqual(result["visible_fraction_of_full_mask"], 1.0)
        self.assertFalse(result["wide_mask_touches_border"])

    def test_wide_border_is_reported_as_incomplete_reference(self) -> None:
        mask = np.zeros((8, 12), dtype=bool)
        mask[0:4, 0:4] = True
        result = rendered_mesh_visibility(mask, 2.0)
        self.assertTrue(result["wide_mask_touches_border"])

    def test_visibility_fraction_counts_only_rendered_silhouette_pixels(self) -> None:
        mask = np.zeros((8, 12), dtype=bool)
        # Four real silhouette pixels lie in the original-FOV crop and six lie
        # outside it. Empty pixels in the enclosing rectangle are irrelevant.
        mask[3, 4:8] = True
        mask[1, 1:4] = True
        mask[6, 9:12] = True
        result = rendered_mesh_visibility(mask, 2.0)
        self.assertEqual(result["full_mask_pixels"], 10)
        self.assertEqual(result["original_fov_crop_mask_pixels"], 4)
        self.assertAlmostEqual(result["visible_fraction_of_full_mask"], 0.4)

    def test_scene_fraction_includes_occlusion_and_wide_fov_pixel_scale(self) -> None:
        result = scene_visible_fraction_from_reference(
            20,
            {"full_mask_pixels": 10, "span_factor": 2.0},
        )
        self.assertEqual(result["expected_full_mask_pixels_at_real_fov"], 40.0)
        self.assertEqual(result["scene_visible_fraction_of_full_mask"], 0.5)

    def test_scene_fraction_is_clamped_for_rasterization_rounding(self) -> None:
        result = scene_visible_fraction_from_reference(
            45,
            {"full_mask_pixels": 10, "span_factor": 2.0},
        )
        self.assertEqual(result["scene_visible_fraction_of_full_mask"], 1.0)


if __name__ == "__main__":
    unittest.main()
