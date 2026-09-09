from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.collect_approach_open_batch import (
    _balanced_split_counts,
    _canonical_sample_id,
    _canonicalize_passed_retry,
    _dataset_run_token,
    _expected_stable_path,
    _load_fresh_json,
    _run,
    _scene_group_split_map,
)


class BatchArtifactFreshnessTest(unittest.TestCase):
    def test_twenty_scene_groups_split_exactly_70_15_15(self) -> None:
        samples = [
            {
                "sample_key": f"hose_{group * 5 + stage:06d}",
                "scene_group_id": f"hose_group_{group:06d}",
            }
            for group in range(20)
            for stage in range(5)
        ]
        mapping = _scene_group_split_map(samples, seed=20260908)
        self.assertEqual(_balanced_split_counts(20), {"train": 14, "val": 3, "test": 3})
        self.assertEqual(list(mapping.values()).count("train"), 14)
        self.assertEqual(list(mapping.values()).count("val"), 3)
        self.assertEqual(list(mapping.values()).count("test"), 3)

    def test_explicit_split_is_honored_atomically(self) -> None:
        samples = [
            {
                "sample_key": f"hose_{index:06d}",
                "scene_group_id": "hose_group_000000",
                "split": "val",
            }
            for index in range(5)
        ]
        self.assertEqual(
            _scene_group_split_map(samples, seed=1),
            {"hose_group_000000": "val"},
        )

    def test_explicit_scene_group_cannot_cross_splits(self) -> None:
        samples = [
            {
                "sample_key": "hose_000000",
                "scene_group_id": "hose_group_000000",
                "split": "train",
            },
            {
                "sample_key": "hose_000001",
                "scene_group_id": "hose_group_000000",
                "split": "test",
            },
        ]
        with self.assertRaises(ValueError):
            _scene_group_split_map(samples, seed=1)

    def test_sample_filename_id_does_not_repeat_class_or_stage(self) -> None:
        self.assertEqual(_canonical_sample_id("hose_000123", "hose"), "000123")
        with self.assertRaises(ValueError):
            _canonical_sample_id("left_000123", "hose")

    def test_passed_retry_is_promoted_to_four_canonical_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retry_dir = root / "train"
            metadata_dir = root / "_metadata" / "train"
            retry_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)
            files = {}
            for field, suffix in {
                "raw": "raw.png",
                "depth": "depth.npy",
                "mask": "mask.png",
                "pose": "pose.json",
            }.items():
                path = retry_dir / f"hose_000123_retry1_{suffix}"
                path.write_bytes(field.encode("utf-8"))
                files[field] = str(path)
            metadata_path = metadata_dir / "hose_000123_retry1_metadata.json"
            metadata = {
                "status": "pass",
                "sample_id": "000123_retry1",
                "files": files,
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            canonical_metadata, updated = _canonicalize_passed_retry(
                metadata_path,
                metadata,
                root,
                "train",
                "hose",
                "000123",
            )
            self.assertEqual(updated["sample_id"], "000123")
            self.assertEqual(updated["passed_retry_sample_id"], "000123_retry1")
            self.assertEqual(canonical_metadata.name, "hose_000123_metadata.json")
            self.assertFalse(metadata_path.exists())
            self.assertEqual(
                sorted(path.name for path in retry_dir.iterdir()),
                [
                    "hose_000123_depth.npy",
                    "hose_000123_mask.png",
                    "hose_000123_pose.json",
                    "hose_000123_raw.png",
                ],
            )

    def test_dataset_roots_use_distinct_stable_pose_namespaces(self) -> None:
        first = _dataset_run_token(Path("/tmp/dataset-a"))
        second = _dataset_run_token(Path("/tmp/dataset-b"))
        self.assertNotEqual(first, second)
        sample = {
            "class_name": "left",
            "sample_key": "left_000000",
            "scene_group_id": "left_group_000000",
        }
        name = _expected_stable_path(sample, first).name
        self.assertIn(first, name)
        self.assertIn("left_group_000000", name)

    def test_stale_json_is_not_reused_after_child_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
            written_at = path.stat().st_mtime_ns
            self.assertIsNone(_load_fresh_json(path, written_at + 1))
            self.assertEqual(
                _load_fresh_json(path, max(0, written_at - 1)),
                {"status": "pass"},
            )

    def test_child_process_timeout_is_reported(self) -> None:
        code, log = _run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            os.environ.copy(),
            timeout_s=0.05,
        )
        self.assertEqual(code, 124)
        self.assertIn("PROCESS_TIMEOUT=0.05s", log)


if __name__ == "__main__":
    unittest.main()
