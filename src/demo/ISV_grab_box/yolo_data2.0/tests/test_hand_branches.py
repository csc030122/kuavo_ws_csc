from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.collect_approach_open_batch import _entry_hand_side
from scripts.collect_approach_until_quota import _quota_counts
from yolo_catchdata.approach_references import (
    MIRROR_METHOD,
    ensure_mirrored_left_references,
    mirror_right_reference_payload,
    validate_reference_set,
)
from yolo_catchdata.hand_sides import HAND_SIDES, hand_spec, reference_filename


class HandBranchTest(unittest.TestCase):
    @staticmethod
    def _right_reference(class_name: str = "hose") -> dict:
        object_pose = {
            "path": f"/World/ManualInspection/Workpieces/{class_name}",
            "translation_m": [1.6, -0.2, 1.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

        def snapshot(stage: str, palm_translation: list[float]) -> dict:
            return {
                "stage": stage,
                "captured_at_utc": "2026-09-08T00:00:00+00:00",
                "world_from_palm": {
                    "path": "/World/ManualInspection/RightWrist/r_palm",
                    "translation_m": palm_translation,
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "world_from_camera_rgb": {
                    "path": (
                        "/World/ManualInspection/RightWrist/r_hand_tripod/"
                        "r_hand_camera_link/right_wrist_d405_rgb"
                    ),
                    "translation_m": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "world_from_object": object_pose,
                "palm_from_object": {},
                "finger_joint_deg": {"THUMB_CMC": 90.0},
            }

        return {
            "schema_version": 1,
            "artifact_kind": "open_hand_approach_reference",
            "status": "complete",
            "generated_at_utc": "2026-09-08T00:00:00+00:00",
            "hand_side": "right",
            "class_name": class_name,
            "reference_output": "source.json",
            "hand_state": {"stage": "approach_open"},
            "snapshots": {
                "grasp": snapshot("grasp", [1.7, -0.1, 1.1]),
                "pregrasp": snapshot("pregrasp", [1.8, 0.0, 1.2]),
            },
            "approach": {
                "direction_world_from_pregrasp_to_grasp": [
                    -1.0 / math.sqrt(3.0),
                    -1.0 / math.sqrt(3.0),
                    -1.0 / math.sqrt(3.0),
                ],
                "backward_direction_world": [
                    1.0 / math.sqrt(3.0),
                    1.0 / math.sqrt(3.0),
                    1.0 / math.sqrt(3.0),
                ],
                "grasp_to_pregrasp_distance_m": math.sqrt(0.03),
            },
        }

    def test_both_side_mounted_wrist_specs_are_distinct(self) -> None:
        self.assertEqual(HAND_SIDES, ("right", "left"))
        right = hand_spec("right")
        left = hand_spec("left")
        self.assertNotEqual(right["manual_root"], left["manual_root"])
        self.assertIn("right_wrist_d405_rgb", right["rgb_camera"])
        self.assertIn("left_wrist_d405_rgb", left["rgb_camera"])
        self.assertEqual(right["finger_joints"]["THUMB_CMC"]["axis"], (0.0, -1.0, 0.0))
        self.assertEqual(left["finger_joints"]["THUMB_CMC"]["axis"], (0.0, 1.0, 0.0))
        self.assertEqual(left["finger_joints"]["THUMB_MCP"]["axis"], (-1.0, 0.0, 0.0))

    def test_reference_validation_rejects_cross_hand_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / reference_filename("left", "hose")
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "hand_side": "right",
                        "class_name": "hose",
                        "snapshots": {"grasp": {}, "pregrasp": {}},
                    }
                ),
                encoding="utf-8",
            )
            errors = validate_reference_set(root, "left", ("hose",))
            self.assertTrue(any("hand-side mismatch" in error for error in errors))

    def test_right_reference_is_mirrored_with_real_left_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / reference_filename("right", "hose")
            target_path = root / reference_filename("left", "hose")
            source = self._right_reference()
            source_path.write_text(json.dumps(source), encoding="utf-8")
            mirrored = mirror_right_reference_payload(
                source, source_path, target_path
            )

            grasp = mirrored["snapshots"]["grasp"]
            pregrasp = mirrored["snapshots"]["pregrasp"]
            self.assertEqual(mirrored["hand_side"], "left")
            self.assertEqual(mirrored["derivation"]["method"], MIRROR_METHOD)
            self.assertAlmostEqual(grasp["world_from_palm"]["translation_m"][0], 1.5)
            self.assertAlmostEqual(pregrasp["world_from_palm"]["translation_m"][0], 1.4)
            self.assertIn("/LeftWrist/l_palm", grasp["world_from_palm"]["path"])
            self.assertIn(
                "left_wrist_d405_rgb", grasp["world_from_camera_rgb"]["path"]
            )
            self.assertAlmostEqual(
                mirrored["approach"]["grasp_to_pregrasp_distance_m"],
                source["approach"]["grasp_to_pregrasp_distance_m"],
            )
            self.assertAlmostEqual(
                mirrored["approach"]["direction_world_from_pregrasp_to_grasp"][0],
                -source["approach"]["direction_world_from_pregrasp_to_grasp"][0],
            )
            quaternion = grasp["world_from_palm"]["quaternion_wxyz"]
            self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0)

    def test_mirror_files_refresh_from_source_and_preserve_manual_left(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / reference_filename("right", "hose")
            target_path = root / reference_filename("left", "hose")
            source = self._right_reference()
            source_path.write_text(json.dumps(source), encoding="utf-8")

            first = ensure_mirrored_left_references(root, ("hose",))
            self.assertEqual(first["written"], [target_path])
            self.assertEqual(validate_reference_set(root, "left", ("hose",)), [])
            old_digest = json.loads(target_path.read_text())["derivation"]["source_sha256"]

            source["revision"] = 2
            source_path.write_text(json.dumps(source), encoding="utf-8")
            refreshed = ensure_mirrored_left_references(root, ("hose",))
            self.assertEqual(refreshed["written"], [target_path])
            new_digest = json.loads(target_path.read_text())["derivation"]["source_sha256"]
            self.assertNotEqual(old_digest, new_digest)

            manual = json.loads(target_path.read_text())
            manual.pop("derivation")
            target_path.write_text(json.dumps(manual), encoding="utf-8")
            preserved = ensure_mirrored_left_references(root, ("hose",))
            self.assertEqual(preserved["preserved_manual"], [target_path])

    def test_quota_counts_only_selected_hand(self) -> None:
        latest = {
            "a": {
                "status": "pass",
                "sample": {
                    "hand_side": "right",
                    "class_name": "hose",
                    "distance_stage": "far",
                },
            },
            "b": {
                "status": "pass",
                "sample": {
                    "hand_side": "left",
                    "class_name": "hose",
                    "distance_stage": "far",
                },
            },
        }
        self.assertEqual(
            _quota_counts(latest, "right")[("hose", "far", "train")], 1
        )
        self.assertEqual(
            _quota_counts(latest, "left")[("hose", "far", "train")], 1
        )

    def test_legacy_manifest_entries_belong_to_right_branch(self) -> None:
        self.assertEqual(_entry_hand_side({"sample": {}}), "right")


if __name__ == "__main__":
    unittest.main()
