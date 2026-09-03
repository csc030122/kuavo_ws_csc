from __future__ import annotations

import unittest

from yolo_catchdata.asset_loader import unit_scale_for_scene
from yolo_catchdata.config import CONFIG_DIR, load_yaml, validate_assets_config


class ConfigTest(unittest.TestCase):
    def test_assets_config_is_valid(self) -> None:
        config = load_yaml(CONFIG_DIR / "assets.yaml")
        self.assertEqual(validate_assets_config(config), [])

    def test_collection_smoke_count_is_400(self) -> None:
        assets = load_yaml(CONFIG_DIR / "assets.yaml")
        collection = load_yaml(CONFIG_DIR / "collection.yaml")
        smoke = collection["smoke"]
        total = len(assets["classes"]) * smoke["groups_per_class"] * smoke["viewpoints_per_group"]
        self.assertEqual(total, 400)

    def test_radius_calibration_has_representative_coverage(self) -> None:
        collection = load_yaml(CONFIG_DIR / "collection.yaml")
        self.assertEqual(
            set(collection["sampling"]["scale_bins"]),
            {"far", "mid", "near", "pre_grasp"},
        )
        calibration = collection["radius_calibration"]
        self.assertEqual(len(calibration["representative_viewpoints"]), 3)
        self.assertGreaterEqual(len(calibration["radii_m"]), 15)

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
        self.assertEqual(
            collection["radius_calibration"]["object_pose_id"], stable["pose_id"]
        )

    def test_workspace_sampling_is_radius_first(self) -> None:
        collection = load_yaml(CONFIG_DIR / "collection.yaml")
        sampling = collection["sampling"]
        workspace = collection["workspace_calibration"]
        self.assertEqual(sampling["primary_variable"], "camera_radius_m")
        self.assertEqual(sampling["scale_bins_role"], "diagnostic_only")
        self.assertEqual(collection["radius_calibration"]["role"], "legacy_diagnostic_only")
        self.assertEqual(workspace["radius_range_m"], {"min": 0.12, "max": 0.50})
        layers = workspace["radius_layers_m"]
        self.assertEqual(layers[0], workspace["radius_range_m"]["max"])
        self.assertEqual(layers[-1], workspace["radius_range_m"]["min"])
        self.assertEqual(layers, sorted(layers, reverse=True))
        self.assertEqual(len(layers), len(set(layers)))
        self.assertEqual(workspace["output_domain"], "omega_class_by_radius")

    def test_millimeter_asset_scale_in_meter_scene(self) -> None:
        self.assertEqual(unit_scale_for_scene(0.001, 1.0), 0.001)
        with self.assertRaises(ValueError):
            unit_scale_for_scene(0.001, 0.0)


if __name__ == "__main__":
    unittest.main()
