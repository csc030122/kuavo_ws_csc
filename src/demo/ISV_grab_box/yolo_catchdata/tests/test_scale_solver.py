from __future__ import annotations

import unittest

from yolo_catchdata.scale_solver import (
    aggregate_viewpoint_solutions,
    bbox_width_scale,
    interpolate_radius_for_scale,
    projected_bbox_truncation_ratio,
    solve_scale_bins,
)


class ScaleSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"r_m": 0.8, "s_pre_occlusion": 0.15, "usable_for_solver": True},
            {"r_m": 0.5, "s_pre_occlusion": 0.25, "usable_for_solver": True},
            {"r_m": 0.3, "s_pre_occlusion": 0.45, "usable_for_solver": True},
            {"r_m": 0.2, "s_pre_occlusion": 0.70, "usable_for_solver": False},
        ]

    def test_bbox_width_scale_uses_inclusive_pixel_width(self) -> None:
        self.assertAlmostEqual(bbox_width_scale([10, 2, 109, 80], 1000), 0.1)
        self.assertEqual(bbox_width_scale(None, 1000), 0.0)

    def test_projected_bbox_truncation(self) -> None:
        self.assertEqual(projected_bbox_truncation_ratio([10, 10, 20, 20], 100, 100), 0.0)
        self.assertAlmostEqual(
            projected_bbox_truncation_ratio([-10, 10, 10, 30], 100, 100), 0.5
        )

    def test_radius_interpolation(self) -> None:
        result = interpolate_radius_for_scale(self.records, 0.35)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["r_m"], 0.4)
        self.assertIsNone(interpolate_radius_for_scale(self.records, 0.60))

    def test_scale_bin_solution_and_aggregation(self) -> None:
        bins = {"mid": [0.20, 0.35], "pre_grasp": [0.55, 0.75]}
        solved = solve_scale_bins(self.records, bins, radius_margin_m=0.05)
        self.assertTrue(solved["mid"]["target_reachable"])
        self.assertFalse(solved["pre_grasp"]["target_reachable"])
        aggregate = aggregate_viewpoint_solutions([solved, solved], bins)
        self.assertEqual(aggregate["mid"]["viewpoints_reaching_target"], 2)
        self.assertFalse(aggregate["pre_grasp"]["target_reachable"])


if __name__ == "__main__":
    unittest.main()
