from __future__ import annotations

import unittest

import random

from scripts.generate_approach_open_plan import (
    DISTANCE_STAGES,
    _sample_object_yaw,
    _split_stage_requests,
    _stage_requests,
    _yaw_stratum_index,
)


class ApproachPlanGroupingTest(unittest.TestCase):
    def test_full_plan_groups_five_stages_per_settled_pose(self) -> None:
        requests = _stage_requests(10, None)
        self.assertEqual(requests[:5], [(0, stage) for stage in DISTANCE_STAGES])
        self.assertEqual(requests[5:], [(1, stage) for stage in DISTANCE_STAGES])

    def test_quota_refill_groups_only_stages_that_are_still_missing(self) -> None:
        requests = _stage_requests(
            99,
            {"grasp": 2, "pre_grasp": 1, "near": 0, "approach": 2, "far": 0},
        )
        self.assertEqual(
            requests,
            [(0, "grasp"), (0, "pre_grasp"), (0, "approach"),
             (1, "grasp"), (1, "approach")],
        )

    def test_long_parts_use_balanced_box_axis_yaw_bands(self) -> None:
        config = {
            "sampling": "box_long_axis_bands",
            "band_centers_deg": [0.0, 180.0],
            "band_half_width_deg": 8.0,
        }
        first = _sample_object_yaw(random.Random(1), "left", config, 0)
        second = _sample_object_yaw(random.Random(1), "left", config, 1)
        self.assertTrue(first <= 8.0 or first >= 352.0)
        self.assertGreaterEqual(second, 172.0)
        self.assertLessEqual(second, 188.0)

    def test_support_faces_and_yaw_bands_cover_their_cross_product(self) -> None:
        faces = ["+z", "-z"]
        centers = [0, 180]
        combinations = {
            (
                faces[group_index % len(faces)],
                centers[_yaw_stratum_index(group_index, len(faces)) % len(centers)],
            )
            for group_index in range(4)
        }
        self.assertEqual(
            combinations,
            {("+z", 0), ("-z", 0), ("+z", 180), ("-z", 180)},
        )

    def test_split_quotas_never_mix_splits_in_one_scene_group(self) -> None:
        requests = _split_stage_requests(
            99,
            {
                "train": {"grasp": 2, "pre_grasp": 1},
                "val": {"grasp": 1, "pre_grasp": 1},
                "test": {"grasp": 0, "pre_grasp": 1},
            },
        )
        self.assertEqual(
            requests,
            [
                ("train", 0, "grasp"),
                ("train", 0, "pre_grasp"),
                ("train", 1, "grasp"),
                ("val", 0, "grasp"),
                ("val", 0, "pre_grasp"),
                ("test", 0, "pre_grasp"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
