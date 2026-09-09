from __future__ import annotations

import unittest

from yolo_catchdata.retry_policy import adjust_retry


BASE = {
    "lateral_offset_1_m": 0.08,
    "lateral_offset_2_m": -0.06,
    "roll_delta_deg": 10.0,
    "pitch_delta_deg": -8.0,
    "yaw_delta_deg": 5.0,
}
LIMITS = {"near": {"lateral_offset_m": 0.04, "orientation_delta_deg": 6.0}}
CONFIG = {
    "truncated_distance_step_m": 0.06,
    "occluded_distance_step_m": 0.05,
    "empty_mask_distance_step_m": 0.06,
    "depth_invalid_distance_step_m": 0.03,
    "small_mask_distance_step_m": 0.03,
    "perturbation_scale_after_retry": 0.8,
    "empty_mask_resample_perturbation": True,
    "empty_mask_resample_scale": 0.55,
}


class RetryPolicyTest(unittest.TestCase):
    def adjust(self, reasons: list[str]) -> dict:
        return adjust_retry(
            sample_key="hose_000001",
            attempt=1,
            reasons=reasons,
            base_distance_m=0.12,
            base_perturbation=BASE,
            distance_stage="near",
            retry_config=CONFIG,
            hand_stage_limits=LIMITS,
        )

    def test_truncated_sample_backs_off_six_centimeters_and_shrinks(self) -> None:
        result = self.adjust(["target_too_truncated"])
        self.assertAlmostEqual(result["distance_from_grasp_m"], 0.18)
        self.assertAlmostEqual(result["hand_perturbation"]["lateral_offset_1_m"], 0.064)
        self.assertEqual(result["strategy"], "back_off_and_shrink_perturbation")

    def test_clipped_visibility_reference_uses_the_same_backoff(self) -> None:
        result = self.adjust(["visibility_reference_clipped"])
        self.assertAlmostEqual(result["distance_from_grasp_m"], 0.18)
        self.assertAlmostEqual(result["hand_perturbation"]["lateral_offset_1_m"], 0.064)
        self.assertEqual(result["strategy"], "back_off_and_shrink_perturbation")

    def test_occlusion_raises_sight_line_and_shrinks(self) -> None:
        result = self.adjust(["target_too_occluded"])
        self.assertAlmostEqual(result["distance_from_grasp_m"], 0.17)
        self.assertAlmostEqual(result["hand_perturbation"]["lateral_offset_1_m"], 0.064)
        self.assertEqual(
            result["strategy"], "raise_sight_line_and_shrink_perturbation"
        )

    def test_empty_mask_resamples_in_centered_subdomain_deterministically(self) -> None:
        first = self.adjust(["target_instance_missing", "empty_mask"])
        second = self.adjust(["target_instance_missing", "empty_mask"])
        self.assertEqual(first, second)
        self.assertGreater(first["distance_from_grasp_m"], 0.12)
        self.assertLessEqual(abs(first["hand_perturbation"]["lateral_offset_1_m"]), 0.022)
        self.assertEqual(first["strategy"], "resample_distance_and_centered_perturbation")

    def test_empty_mask_resample_honors_small_object_limit_scale(self) -> None:
        result = adjust_retry(
            sample_key="outhandle_000001",
            attempt=1,
            reasons=["empty_mask"],
            base_distance_m=0.12,
            base_perturbation=BASE,
            distance_stage="near",
            retry_config=CONFIG,
            hand_stage_limits=LIMITS,
            hand_limit_scale=0.45,
        )
        self.assertLessEqual(
            abs(result["hand_perturbation"]["lateral_offset_1_m"]),
            0.04 * 0.55 * 0.45,
        )

    def test_small_mask_moves_closer(self) -> None:
        result = self.adjust(["mask_pixels_too_few"])
        self.assertAlmostEqual(result["distance_from_grasp_m"], 0.09)

    def test_retry_distance_is_clamped_to_current_stage(self) -> None:
        farther = adjust_retry(
            sample_key="left_000001",
            attempt=1,
            reasons=["target_too_truncated"],
            base_distance_m=0.09,
            base_perturbation=BASE,
            distance_stage="pre_grasp",
            retry_config=CONFIG,
            hand_stage_limits=LIMITS,
            minimum_distance_m=0.02,
            maximum_distance_m=0.10,
        )
        closer = adjust_retry(
            sample_key="left_000002",
            attempt=1,
            reasons=["mask_pixels_too_few"],
            base_distance_m=0.11,
            base_perturbation=BASE,
            distance_stage="near",
            retry_config=CONFIG,
            hand_stage_limits=LIMITS,
            minimum_distance_m=0.10,
            maximum_distance_m=0.20,
        )
        self.assertEqual(farther["distance_from_grasp_m"], 0.10)
        self.assertEqual(closer["distance_from_grasp_m"], 0.10)

    def test_grasp_retry_keeps_exact_reference_anchor(self) -> None:
        result = adjust_retry(
            sample_key="left_000001",
            attempt=1,
            reasons=["target_too_truncated"],
            base_distance_m=0.0,
            base_perturbation=BASE,
            distance_stage="grasp",
            retry_config=CONFIG,
            hand_stage_limits={
                "grasp": {"lateral_offset_m": 0.0, "orientation_delta_deg": 0.0}
            },
        )
        self.assertEqual(result["distance_from_grasp_m"], 0.0)
        self.assertTrue(all(value == 0.0 for value in result["hand_perturbation"].values()))
        self.assertEqual(result["strategy"], "keep_exact_grasp_anchor")


if __name__ == "__main__":
    unittest.main()
