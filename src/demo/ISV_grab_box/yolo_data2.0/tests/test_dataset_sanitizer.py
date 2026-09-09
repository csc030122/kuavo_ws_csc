from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.sanitize_approach_dataset import (
    assign_groups,
    build_cell_targets,
    select_candidates,
    write_dataset,
)


class DatasetSanitizerTest(unittest.TestCase):
    def test_group_assignment_is_split_atomic_and_balances_partial_cell(self) -> None:
        candidates = []
        for index in range(8):
            candidates.append(
                {
                    "class_name": "left",
                    "stage": "approach",
                    "group": f"group_{index}",
                    "source_status": "pass",
                    "source_metadata": f"metadata_{index}",
                    "metadata": {
                        "visibility_reference": {"visible_fraction_of_full_mask": 1.0},
                        "depth_valid_ratio": 1.0,
                        "mask_pixels": 1000,
                    },
                }
            )
        targets = build_cell_targets(
            candidates, 20, {"train": 70, "val": 15, "test": 15}
        )
        self.assertEqual(
            [targets[("left", "approach", split)] for split in ("train", "val", "test")],
            [6, 1, 1],
        )
        assignment = assign_groups(candidates, targets, 7)
        selected = select_candidates(candidates, targets, assignment)
        counts = Counter(item["split"] for item in selected)
        self.assertEqual(counts, {"train": 6, "val": 1, "test": 1})
        self.assertTrue(all(assignment[item["group"]] == item["split"] for item in selected))

    def test_write_dataset_preserves_round_markers_for_safe_resume(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            (source / "plans").mkdir(parents=True)
            (source / "plans" / "round_00.json").write_text("{}\n", encoding="utf-8")
            (source / "plans" / "round_00_candidate_quotas.json").write_text(
                "{}\n", encoding="utf-8"
            )
            files = {}
            for field in ("raw", "depth", "mask", "pose"):
                path = source / f"input_{field}"
                path.write_bytes(field.encode("ascii"))
                files[field] = path
            selected = [
                {
                    "class_name": "hose",
                    "stage": "grasp",
                    "group": "/tmp/hose_group_000000.json",
                    "split": "train",
                    "source_status": "rejected",
                    "source_metadata": source / "input_metadata.json",
                    "files": files,
                    "quality": {
                        "quality_warnings": [],
                        "thresholds": {
                            "visible_fraction_min": 0.4,
                            "visible_fraction_target": 0.5,
                            "visible_fraction_preferred_max": 1.0,
                            "depth_valid_ratio_min": 0.3,
                            "min_mask_pixels": 500,
                        },
                    },
                    "metadata": {
                        "hand_side": "right",
                        "sample_id": "000000_retry1",
                        "distance_from_grasp_m": 0.0,
                        "hand_perturbation": {},
                    },
                }
            ]
            targets = {("hose", "grasp", "train"): 1}
            report = write_dataset(source, output, selected, targets, Counter())
            self.assertTrue((output / "plans" / "round_00.json").is_file())
            self.assertTrue(
                (output / "plans" / "round_00_candidate_quotas.json").is_file()
            )
            self.assertEqual(len(report["imported_source_plans"]), 2)
            self.assertEqual(report["recovered_previously_rejected_samples"], 1)


if __name__ == "__main__":
    unittest.main()
