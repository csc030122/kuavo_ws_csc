"""Candidate allocation for class, distance-stage, and split quotas."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def balanced_split_quota(
    total: int,
    split_percent: Mapping[str, float | int],
    splits: Sequence[str] = ("train", "val", "test"),
) -> dict[str, int]:
    """Allocate an integer per-cell quota by largest remainder.

    For the normal 70/15/15 policy, totals of 20 or 100 are exact.  Very small
    pilot quotas are necessarily approximate; whenever possible every
    positive-ratio split receives at least one sample.
    """

    count = int(total)
    if count < 0:
        raise ValueError("total must be non-negative")
    names = tuple(str(name) for name in splits)
    weights = {name: float(split_percent.get(name, 0.0)) for name in names}
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("split percentages must be non-negative")
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        raise ValueError("at least one split percentage must be positive")
    raw = {name: count * weights[name] / weight_sum for name in names}
    result = {name: int(raw[name]) for name in names}
    remaining = count - sum(result.values())
    order = sorted(
        names,
        key=lambda name: (raw[name] - result[name], -names.index(name)),
        reverse=True,
    )
    for index in range(remaining):
        result[order[index % len(order)]] += 1

    positive = [name for name in names if weights[name] > 0.0]
    if count >= len(positive):
        for name in positive:
            if result[name] > 0:
                continue
            donors = [candidate for candidate in positive if result[candidate] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda candidate: result[candidate])
            result[donor] -= 1
            result[name] += 1
    return result


def rotating_cell_split_quotas(
    total_per_cell: int,
    cell_count: int,
    split_percent: Mapping[str, float | int],
    splits: Sequence[str] = ("train", "val", "test"),
) -> list[dict[str, int]]:
    """Distribute rounding across cells while keeping the global ratio exact.

    With 20 class×stage cells and 70/15/15, a five-sample pilot becomes
    exactly 70/15/15 overall even though no individual five-sample cell can
    represent that ratio.  At 20 or 100 samples per cell, every cell is itself
    exact as well.
    """

    total = int(total_per_cell)
    count = int(cell_count)
    if total < 0 or count <= 0:
        raise ValueError("total_per_cell must be non-negative and cell_count positive")
    names = tuple(str(name) for name in splits)
    one_cycle_counts = balanced_split_quota(count, split_percent, names)
    cycle = [
        name
        for name in names
        for _ in range(one_cycle_counts[name])
    ]
    if len(cycle) != count:
        raise AssertionError("split cycle length mismatch")
    stride = 1
    if count > 2:
        stride = next(
            candidate
            for candidate in range(2, count)
            if math.gcd(candidate, count) == 1
        )
    result: list[dict[str, int]] = []
    for cell_index in range(count):
        quotas = {name: 0 for name in names}
        for ordinal in range(total):
            quotas[cycle[(cell_index + ordinal * stride) % count]] += 1
        result.append(quotas)
    return result


def allocate_missing_cells(
    missing: Mapping[tuple[str, str], int],
    classes: Sequence[str],
    stages: Sequence[str],
    samples_per_class: int,
) -> dict[str, dict[str, int]]:
    """Allocate only unfinished cells, without exceeding their current deficit."""

    budget = int(samples_per_class)
    if budget <= 0:
        raise ValueError("samples_per_class must be positive")
    result: dict[str, dict[str, int]] = {}
    for class_name in classes:
        quotas = {stage: 0 for stage in stages}
        active = [stage for stage in stages if int(missing.get((class_name, stage), 0)) > 0]
        remaining_budget = budget
        # One candidate per unfinished stage first, so every trajectory layer
        # continues making progress even when one stage has a larger deficit.
        for stage in active:
            if remaining_budget <= 0:
                break
            quotas[stage] += 1
            remaining_budget -= 1
        while remaining_budget > 0:
            eligible = [
                stage
                for stage in active
                if quotas[stage] < int(missing[(class_name, stage)])
            ]
            if not eligible:
                break
            stage = max(
                eligible,
                key=lambda name: (int(missing[(class_name, name)]) - quotas[name], -stages.index(name)),
            )
            quotas[stage] += 1
            remaining_budget -= 1
        result[class_name] = quotas
    return result


def allocate_missing_split_cells(
    missing: Mapping[tuple[str, str, str], int],
    classes: Sequence[str],
    stages: Sequence[str],
    splits: Sequence[str],
    samples_per_class: int,
) -> dict[str, dict[str, dict[str, int]]]:
    """Allocate candidates only to unfinished class×stage×split cells."""

    budget = int(samples_per_class)
    if budget <= 0:
        raise ValueError("samples_per_class must be positive")
    result: dict[str, dict[str, dict[str, int]]] = {}
    ordered_cells = [(split, stage) for split in splits for stage in stages]
    for class_name in classes:
        quotas = {
            split: {stage: 0 for stage in stages}
            for split in splits
        }
        active = [
            cell
            for cell in ordered_cells
            if int(missing.get((class_name, cell[1], cell[0]), 0)) > 0
        ]
        remaining_budget = budget
        for split, stage in active:
            if remaining_budget <= 0:
                break
            quotas[split][stage] += 1
            remaining_budget -= 1
        while remaining_budget > 0:
            eligible = [
                (split, stage)
                for split, stage in active
                if quotas[split][stage]
                < int(missing[(class_name, stage, split)])
            ]
            if not eligible:
                break
            split, stage = max(
                eligible,
                key=lambda cell: (
                    int(missing[(class_name, cell[1], cell[0])])
                    - quotas[cell[0]][cell[1]],
                    -ordered_cells.index(cell),
                ),
            )
            quotas[split][stage] += 1
            remaining_budget -= 1
        result[class_name] = quotas
    return result


def next_sample_index(sample_keys: Sequence[str]) -> int:
    """Return one index above all valid numeric suffixes in a manifest."""

    indices = []
    for key in sample_keys:
        try:
            indices.append(int(str(key).rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=-1) + 1


def next_round_index(plan_names: Sequence[str]) -> int:
    """Return one round above existing ``round_NN.json`` plan files."""

    indices = []
    for name in plan_names:
        stem = str(name).rsplit("/", 1)[-1]
        if not stem.startswith("round_") or not stem.endswith(".json"):
            continue
        token = stem[len("round_") : -len(".json")]
        if token.isdigit():
            indices.append(int(token))
    return max(indices, default=-1) + 1


__all__ = [
    "allocate_missing_cells",
    "allocate_missing_split_cells",
    "balanced_split_quota",
    "next_round_index",
    "next_sample_index",
    "rotating_cell_split_quotas",
]
