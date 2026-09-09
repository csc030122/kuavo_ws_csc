from __future__ import annotations

import math
import random
import unittest

from yolo_catchdata.hand_randomization import (
    apply_hand_perturbation,
    sample_hand_perturbation,
    sample_hand_perturbation_mixture,
    stratified_mixture_mode,
)


def pose(x: float, y: float, z: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


class HandRandomizationTest(unittest.TestCase):
    def test_sampler_is_deterministic_and_within_stage_limits(self) -> None:
        first = sample_hand_perturbation(random.Random(7), "near")
        second = sample_hand_perturbation(random.Random(7), "near")
        self.assertEqual(first, second)
        self.assertLessEqual(abs(first["lateral_offset_1_m"]), 0.04)
        self.assertLessEqual(abs(first["lateral_offset_2_m"]), 0.04)
        self.assertLessEqual(abs(first["roll_delta_deg"]), 6.0)
        self.assertLessEqual(abs(first["pitch_delta_deg"]), 6.0)
        self.assertLessEqual(abs(first["yaw_delta_deg"]), 6.0)

    def test_offsets_stay_perpendicular_to_approach_direction(self) -> None:
        grasp = pose(0.0, 0.0, 0.0)
        pregrasp = pose(0.0, 0.0, 1.0)
        base = pose(0.0, 0.0, 0.5)
        actual = apply_hand_perturbation(
            base,
            grasp,
            pregrasp,
            {
                "lateral_offset_1_m": 0.1,
                "lateral_offset_2_m": -0.2,
                "roll_delta_deg": 0.0,
                "pitch_delta_deg": 0.0,
                "yaw_delta_deg": 0.0,
            },
        )
        self.assertAlmostEqual(actual[2][3], 0.5)
        delta = [actual[index][3] - base[index][3] for index in range(3)]
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in delta)), math.sqrt(0.05))

    def test_centered_mixture_reduces_far_limits_and_keeps_tail(self) -> None:
        limits = {"far": {"lateral_offset_m": 0.10, "orientation_delta_deg": 15.0}}
        centered, centered_mode = sample_hand_perturbation_mixture(
            random.Random(1),
            "far",
            limits,
            {"centered_probability": 1.0, "centered_scale": 0.5},
        )
        tail, tail_mode = sample_hand_perturbation_mixture(
            random.Random(1),
            "far",
            limits,
            {"centered_probability": 0.0, "centered_scale": 0.5},
        )
        self.assertEqual(centered_mode, "centered")
        self.assertEqual(tail_mode, "coverage_tail")
        self.assertLessEqual(abs(centered["lateral_offset_1_m"]), 0.05)
        self.assertLessEqual(abs(centered["roll_delta_deg"]), 7.5)
        self.assertLessEqual(abs(tail["lateral_offset_1_m"]), 0.10)

    def test_sampler_scale_must_be_valid(self) -> None:
        with self.assertRaises(ValueError):
            sample_hand_perturbation(random.Random(1), "near", scale=1.1)

    def test_stratified_modes_preserve_small_coverage_tail(self) -> None:
        mixture = {"centered_probability_by_stage": {"far": 0.75}}
        modes = [stratified_mixture_mode("far", index, mixture) for index in range(8)]
        self.assertEqual(modes.count("centered"), 6)
        self.assertEqual(modes.count("coverage_tail"), 2)

    def test_local_yaw_changes_orientation_without_changing_position(self) -> None:
        grasp = pose(0.0, 0.0, 0.0)
        pregrasp = pose(0.0, 0.0, 1.0)
        base = pose(0.0, 0.0, 0.5)
        actual = apply_hand_perturbation(
            base,
            grasp,
            pregrasp,
            {"yaw_delta_deg": 90.0},
        )
        self.assertEqual(tuple(actual[index][3] for index in range(3)), (0.0, 0.0, 0.5))
        for row, expected in enumerate(((0, -1, 0), (1, 0, 0), (0, 0, 1))):
            for column, value in enumerate(expected):
                self.assertAlmostEqual(actual[row][column], value)


if __name__ == "__main__":
    unittest.main()
