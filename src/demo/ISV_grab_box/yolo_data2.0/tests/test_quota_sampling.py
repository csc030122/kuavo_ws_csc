from __future__ import annotations

import unittest

from yolo_catchdata.quota_sampling import (
    allocate_missing_cells,
    allocate_missing_split_cells,
    balanced_split_quota,
    next_round_index,
    next_sample_index,
    rotating_cell_split_quotas,
)


class QuotaSamplingTest(unittest.TestCase):
    def test_only_missing_cells_receive_candidates(self) -> None:
        stages = ("grasp", "pre_grasp", "near", "approach", "far")
        missing = {
            ("left", "grasp"): 0,
            ("left", "pre_grasp"): 0,
            ("left", "near"): 3,
            ("left", "approach"): 0,
            ("left", "far"): 2,
        }
        quotas = allocate_missing_cells(missing, ("left",), stages, 8)["left"]
        self.assertEqual(
            quotas,
            {"grasp": 0, "pre_grasp": 0, "near": 3, "approach": 0, "far": 2},
        )

    def test_budget_is_balanced_across_all_active_stages(self) -> None:
        stages = ("grasp", "pre_grasp", "near", "approach", "far")
        missing = {("hose", stage): 5 for stage in stages}
        quotas = allocate_missing_cells(missing, ("hose",), stages, 10)["hose"]
        self.assertEqual(quotas, {stage: 2 for stage in stages})

    def test_candidate_count_never_exceeds_deficit(self) -> None:
        stages = ("grasp", "pre_grasp", "near", "approach", "far")
        missing = {("right", stage): 1 for stage in stages}
        quotas = allocate_missing_cells(missing, ("right",), stages, 8)["right"]
        self.assertEqual(sum(quotas.values()), 5)
        self.assertTrue(all(value == 1 for value in quotas.values()))

    def test_resume_index_is_above_all_existing_sample_keys(self) -> None:
        self.assertEqual(
            next_sample_index(("left_000003", "hose_000018", "invalid")),
            19,
        )

    def test_resume_round_does_not_overwrite_existing_plans(self) -> None:
        self.assertEqual(
            next_round_index(
                ("round_00.json", "round_01.json", "round_01_candidate_quotas.json")
            ),
            2,
        )

    def test_split_quota_is_exact_for_formal_cell_size(self) -> None:
        self.assertEqual(
            balanced_split_quota(100, {"train": 70, "val": 15, "test": 15}),
            {"train": 70, "val": 15, "test": 15},
        )
        self.assertEqual(
            balanced_split_quota(20, {"train": 70, "val": 15, "test": 15}),
            {"train": 14, "val": 3, "test": 3},
        )

    def test_small_split_quota_keeps_validation_and_test_nonempty(self) -> None:
        self.assertEqual(
            balanced_split_quota(5, {"train": 70, "val": 15, "test": 15}),
            {"train": 3, "val": 1, "test": 1},
        )

    def test_split_allocator_targets_only_missing_split_cells(self) -> None:
        stages = ("grasp", "near")
        splits = ("train", "val", "test")
        missing = {
            ("hose", "grasp", "train"): 2,
            ("hose", "grasp", "val"): 0,
            ("hose", "grasp", "test"): 1,
            ("hose", "near", "train"): 3,
            ("hose", "near", "val"): 1,
            ("hose", "near", "test"): 0,
        }
        quotas = allocate_missing_split_cells(
            missing, ("hose",), stages, splits, 10
        )["hose"]
        self.assertEqual(
            quotas,
            {
                "train": {"grasp": 2, "near": 3},
                "val": {"grasp": 0, "near": 1},
                "test": {"grasp": 1, "near": 0},
            },
        )

    def test_small_cells_are_globally_exact_for_twenty_cells(self) -> None:
        quotas = rotating_cell_split_quotas(
            5, 20, {"train": 70, "val": 15, "test": 15}
        )
        self.assertEqual(
            {split: sum(cell[split] for cell in quotas) for split in ("train", "val", "test")},
            {"train": 70, "val": 15, "test": 15},
        )

    def test_formal_cells_are_each_exact(self) -> None:
        quotas = rotating_cell_split_quotas(
            100, 20, {"train": 70, "val": 15, "test": 15}
        )
        self.assertTrue(
            all(cell == {"train": 70, "val": 15, "test": 15} for cell in quotas)
        )


if __name__ == "__main__":
    unittest.main()
