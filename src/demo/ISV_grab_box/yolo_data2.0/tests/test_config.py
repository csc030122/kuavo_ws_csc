from __future__ import annotations

import unittest

from yolo_catchdata.asset_loader import unit_scale_for_scene
from yolo_catchdata.config import CONFIG_DIR, load_yaml, validate_assets_config


class ConfigTest(unittest.TestCase):
    def test_assets_config_is_valid(self) -> None:
        config = load_yaml(CONFIG_DIR / "assets.yaml")
        self.assertEqual(validate_assets_config(config), [])

    def test_stable_pose_uses_original_visual_and_explicit_collision_proxy(self) -> None:
        collection = load_yaml(CONFIG_DIR / "collection.yaml")
        stable = collection["stable_pose"]
        self.assertEqual(stable["visual_geometry"], "original_usd_render_mesh")
        self.assertEqual(stable["dynamic_collision_approximation"], "convexDecomposition")
        self.assertEqual(
            stable["table_collision_approximation"],
            "authored_collision_meshes",
        )
        self.assertGreaterEqual(
            stable["convex_decomposition"]["max_convex_hulls"], 64
        )
        self.assertLessEqual(
            stable["convex_decomposition"]["error_percentage"], 1.0
        )
        self.assertGreater(stable["stable_consecutive_frames"], 0)
        self.assertGreater(stable["maximum_frames"], stable["minimum_frames"])
    def test_millimeter_asset_scale_in_meter_scene(self) -> None:
        self.assertEqual(unit_scale_for_scene(0.001, 1.0), 0.001)
        with self.assertRaises(ValueError):
            unit_scale_for_scene(0.001, 0.0)

    def test_approach_open_uses_class_specific_yaw_transport(self) -> None:
        collection = load_yaml(CONFIG_DIR / "collection.yaml")
        camera = load_yaml(CONFIG_DIR / "camera.yaml")["camera"]
        policy = collection["approach_open"]
        self.assertEqual(
            policy["hand_branches"]["supported"], ["right", "left"]
        )
        self.assertEqual(
            policy["hand_branches"]["mode"],
            "independent_output_roots_and_manifests",
        )
        self.assertEqual(set(camera["hand_sides"]), {"right", "left"})
        self.assertEqual(
            camera["hand_sides"]["left"]["name"], "left_wrist_d405"
        )
        classes = policy["object_yaw"]["classes"]
        self.assertEqual(
            classes["outhandle"]["hand_transport"], "fixed_reference_orientation"
        )
        self.assertEqual(classes["outhandle"]["sampling"], "uniform_0_to_360_deg")
        for class_name in ("left", "right", "hose"):
            self.assertEqual(classes[class_name]["sampling"], "box_long_axis_bands")
            self.assertEqual(
                classes[class_name]["hand_transport"],
                "follow_object_yaw_modulo_180_deg",
            )
            self.assertEqual(classes[class_name]["symmetry_order"], 2)
        self.assertTrue(policy["palm_pose"]["keep_reference_roll_pitch"])
        self.assertEqual(
            policy["palm_pose"]["translation_path"],
            "box_clearance_camera_ray_from_settled_object",
        )
        wall_sight_line = policy["palm_pose"]["table_box_line_of_sight"]
        self.assertTrue(wall_sight_line["enabled"])
        self.assertGreater(wall_sight_line["clearance_above_wall_m"], 0.0)
        self.assertGreater(wall_sight_line["maximum_vertical_lift_m"], 0.0)
        self.assertTrue(policy["palm_pose"]["keep_reference_finger_joints"])
        distance = policy["palm_pose"]["distance_from_grasp_m"]
        self.assertEqual(distance, {"min": 0.0, "max": 0.50})
        randomization = policy["hand_randomization"]
        self.assertTrue(randomization["enabled"])
        self.assertEqual(
            randomization["distribution"],
            "centered_mixture_with_coverage_tail",
        )
        self.assertEqual(
            set(randomization["stages"]),
            {"far", "approach", "near", "pre_grasp", "grasp"},
        )
        self.assertEqual(
            policy["palm_pose"]["distance_sampling_by_class"]["default"]["grasp"],
            {"min": 0.0, "max": 0.0},
        )
        self.assertEqual(randomization["stages"]["grasp"]["lateral_offset_m"], 0.0)
        self.assertEqual(randomization["stages"]["grasp"]["orientation_delta_deg"], 0.0)
        self.assertTrue(randomization["apply_to"]["lateral_offset"])
        self.assertTrue(randomization["apply_to"]["roll"])
        self.assertTrue(randomization["apply_to"]["pitch"])
        self.assertTrue(randomization["apply_to"]["yaw"])
        self.assertEqual(
            camera["visibility_measurement"]["method"],
            "target_only_wide_fov_rendered_mesh",
        )
        self.assertGreater(camera["visibility_measurement"]["wide_fov_span_factor"], 1.0)
        self.assertEqual(
            policy["object_translation"]["strategy"],
            "centered_exact_mesh_with_wall_clearance",
        )
        visible = policy["rejection"]["visible_fraction_of_full_mask"]
        self.assertEqual(visible["min"], 0.45)
        self.assertEqual(visible["min_by_class"], {"hose": 0.40})
        self.assertEqual(policy["rejection"]["depth_valid_ratio"]["min"], 0.50)
        self.assertEqual(
            policy["rejection"]["depth_valid_ratio"]["min_by_stage_class"],
            {"grasp": {"hose": 0.30}},
        )
        self.assertEqual(policy["rejection"]["min_mask_pixels"], 500)
        mixture = randomization["mixture"]
        self.assertGreater(
            mixture["centered_probability_by_stage"]["far"],
            mixture["centered_probability_by_stage"]["pre_grasp"],
        )
        self.assertLess(mixture["centered_probability_by_stage"]["far"], 1.0)
        self.assertEqual(mixture["limit_scale_by_class"]["outhandle"], 0.45)
        self.assertLess(mixture["limit_scale_by_class"]["outhandle"], 1.0)
        self.assertEqual(
            policy["object_translation"]["mixture"]["centered_probability"],
            1.0,
        )
        quota = policy["quota_collection"]
        self.assertGreater(quota["accepted_per_class_stage"], 0)
        self.assertGreater(quota["samples_per_class_per_round"], 0)
        self.assertEqual(quota["stable_process_max_attempts"], 3)
        self.assertGreater(quota["child_process_timeout_s"], 0)

    def test_approach_open_support_faces_match_object_families(self) -> None:
        collection = load_yaml(CONFIG_DIR / "collection.yaml")
        support = collection["approach_open"]["support_pose"]
        self.assertEqual(support["mode"], "physics_settle_then_freeze")
        self.assertEqual(
            support["classes"]["outhandle"]["allowed_support_faces"],
            ["+x", "-x", "+y", "-y", "+z", "-z"],
        )
        for class_name in ("left", "right", "hose"):
            self.assertEqual(
                support["classes"][class_name]["allowed_support_faces"],
                ["+z", "-z"],
            )
        validation = support["validation"]
        self.assertFalse(validation["require_allowed_support_face_after_settle"])
        self.assertFalse(validation["wall_contact_is_allowed"])
        self.assertGreater(validation["wall_clearance_m"], 0.0)
        self.assertGreater(validation["floor_contact_tolerance_m"], 0.0)
        self.assertTrue(validation["use_transformed_visual_mesh_vertices"])
        self.assertTrue(validation["use_table_box_authored_collision"])
        self.assertGreaterEqual(validation["floor_penetration_tolerance_m"], 0.0)
        self.assertGreaterEqual(validation["wall_penetration_tolerance_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
