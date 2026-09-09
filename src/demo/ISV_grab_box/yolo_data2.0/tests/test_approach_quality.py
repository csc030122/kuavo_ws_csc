from __future__ import annotations

import unittest

from yolo_catchdata.approach_quality import evaluate_approach_quality


REJECTION = {
    "min_mask_pixels": 1,
    "visible_fraction_of_full_mask": {
        "min": 0.45,
        "target": 0.50,
        "preferred_max": 1.00,
    },
    "depth_valid_ratio": {
        "min": 0.50,
    },
    "visibility_reference": {
        "reject_if_wide_mask_touches_border": False,
        "reject_if_missing": True,
    },
}


class ApproachQualityTest(unittest.TestCase):
    def evaluate(
        self,
        class_name: str,
        visible_fraction: float | None,
        depth_valid_ratio: float = 1.0,
        wide_clipped: bool = False,
        distance_stage: str | None = None,
    ) -> dict:
        return evaluate_approach_quality(
            class_name=class_name,
            target_instance_ids=[1],
            mask_pixels=100_000,
            depth_valid_ratio=depth_valid_ratio,
            visibility={
                "visible_fraction_of_full_mask": visible_fraction,
                "wide_mask_touches_border": wide_clipped,
            },
            rejection_config=REJECTION,
            distance_stage=distance_stage,
        )

    def test_all_classes_accept_roughly_half_visible(self) -> None:
        for class_name in ("left", "right", "hose", "outhandle"):
            self.assertEqual(self.evaluate(class_name, 0.45)["status"], "pass")

    def test_all_classes_reject_below_rough_half_threshold(self) -> None:
        result = self.evaluate("left", 0.449)
        self.assertEqual(result["status"], "fail")
        self.assertIn("target_too_truncated", result["quality_reasons"])

    def test_half_of_visible_mask_must_have_depth(self) -> None:
        self.assertEqual(self.evaluate("hose", 0.50, 0.50)["status"], "pass")
        result = self.evaluate("hose", 0.50, 0.499)
        self.assertEqual(result["status"], "fail")
        self.assertIn("depth_valid_ratio_low", result["quality_reasons"])

    def test_hose_can_use_class_and_grasp_depth_exceptions(self) -> None:
        rejection = {
            **REJECTION,
            "visible_fraction_of_full_mask": {
                **REJECTION["visible_fraction_of_full_mask"],
                "min_by_class": {"hose": 0.40},
            },
            "depth_valid_ratio": {
                **REJECTION["depth_valid_ratio"],
                "min_by_stage_class": {"grasp": {"hose": 0.30}},
            },
        }
        result = evaluate_approach_quality(
            class_name="hose",
            target_instance_ids=[1],
            mask_pixels=200_000,
            depth_valid_ratio=0.33,
            visibility={
                "visible_fraction_of_full_mask": 0.40,
                "wide_mask_touches_border": False,
            },
            rejection_config=rejection,
            distance_stage="grasp",
        )
        self.assertEqual(result["status"], "pass")

    def test_class_specific_mask_floor_is_enforced(self) -> None:
        rejection = {**REJECTION, "min_mask_pixels_by_class": {"outhandle": 500}}
        result = evaluate_approach_quality(
            class_name="outhandle",
            target_instance_ids=[1],
            mask_pixels=499,
            depth_valid_ratio=1.0,
            visibility={
                "visible_fraction_of_full_mask": 1.0,
                "wide_mask_touches_border": False,
            },
            rejection_config=rejection,
        )
        self.assertIn("mask_pixels_too_few", result["quality_reasons"])

    def test_nonempty_mask_has_no_absolute_pixel_floor(self) -> None:
        result = evaluate_approach_quality(
            class_name="outhandle",
            target_instance_ids=[1],
            mask_pixels=1,
            depth_valid_ratio=1.0,
            visibility={
                "visible_fraction_of_full_mask": 0.50,
                "wide_mask_touches_border": False,
            },
            rejection_config=REJECTION,
        )
        self.assertEqual(result["status"], "pass")

    def test_wide_reference_clipping_never_bypasses_visibility_minimum(self) -> None:
        result = self.evaluate("left", 0.10, wide_clipped=True)
        self.assertEqual(result["status"], "fail")
        self.assertIn("target_too_truncated", result["quality_reasons"])
        self.assertEqual(result["quality_warnings"], ["visibility_reference_clipped"])

    def test_clipped_reference_can_be_rejected_independently(self) -> None:
        rejection = {
            **REJECTION,
            "visibility_reference": {
                **REJECTION["visibility_reference"],
                "reject_if_wide_mask_touches_border": True,
            },
        }
        result = evaluate_approach_quality(
            class_name="left",
            target_instance_ids=[1],
            mask_pixels=100_000,
            depth_valid_ratio=1.0,
            visibility={
                "visible_fraction_of_full_mask": 0.80,
                "wide_mask_touches_border": True,
            },
            rejection_config=rejection,
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("visibility_reference_clipped", result["quality_reasons"])

    def test_scene_occlusion_cannot_pass_on_in_frame_fraction_alone(self) -> None:
        result = evaluate_approach_quality(
            class_name="left",
            target_instance_ids=[1],
            mask_pixels=10_000,
            depth_valid_ratio=1.0,
            visibility={
                "full_mask_pixels": 1_000,
                "span_factor": 8.0,
                "visible_fraction_of_full_mask": 1.0,
                "wide_mask_touches_border": False,
            },
            rejection_config=REJECTION,
            distance_stage="far",
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("target_too_occluded", result["quality_reasons"])
        self.assertAlmostEqual(
            result["measurements"]["scene_visible_fraction_of_full_mask"],
            0.15625,
        )


if __name__ == "__main__":
    unittest.main()
