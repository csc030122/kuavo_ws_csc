#!/usr/bin/env python3
"""Build a clean, scene-group-atomic dataset from an interrupted candidate pool.

The source is never modified.  Accepted artifacts and previously rejected
artifacts are re-evaluated with the *current* quality policy, exact trajectory
duplicates are removed, every stable scene is assigned to exactly one split,
and a fresh manifest with collision-free sample ids is written.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.approach_quality import evaluate_approach_quality  # noqa: E402
from yolo_catchdata.config import CONFIG_DIR, load_yaml  # noqa: E402
from yolo_catchdata.mesh_visibility import (  # noqa: E402
    scene_visible_fraction_from_reference,
)
from yolo_catchdata.output_lock import acquire_output_lock  # noqa: E402
from yolo_catchdata.quota_sampling import balanced_split_quota  # noqa: E402


CLASSES = ("left", "right", "hose", "outhandle")
STAGES = ("grasp", "pre_grasp", "near", "approach", "far")
SPLITS = ("train", "val", "test")
FILE_SUFFIXES = {"raw": "raw.png", "depth": "depth.npy", "mask": "mask.png", "pose": "pose.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--accepted-per-class-stage",
        type=int,
        default=20,
        help="清洗输出每个 class×stage 最多保留多少张。",
    )
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _artifact_paths(metadata: dict[str, Any]) -> dict[str, Path] | None:
    files = metadata.get("files") or {}
    result: dict[str, Path] = {}
    for field in FILE_SUFFIXES:
        value = files.get(field)
        if not value:
            return None
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            return None
        result[field] = path
    return result


def _fingerprint(metadata: dict[str, Any]) -> tuple[Any, ...]:
    perturbation = metadata.get("hand_perturbation") or {}
    return (
        str(metadata.get("stable_shard")),
        str(metadata.get("distance_stage")),
        round(float(metadata.get("distance_from_grasp_m", 0.0)), 9),
        tuple(
            round(float(perturbation.get(name, 0.0)), 9)
            for name in (
                "lateral_offset_1_m",
                "lateral_offset_2_m",
                "roll_delta_deg",
                "pitch_delta_deg",
                "yaw_delta_deg",
            )
        ),
    )


def _candidate_score(candidate: dict[str, Any]) -> tuple[float, ...]:
    metadata = candidate["metadata"]
    measurements = candidate.get("quality", {}).get("measurements", {})
    return (
        1.0 if candidate["source_status"] == "pass" else 0.0,
        float(measurements.get("scene_visible_fraction_of_full_mask") or 0.0),
        float(metadata.get("depth_valid_ratio") or 0.0),
        float(metadata.get("mask_pixels") or 0.0),
    )


def load_candidates(
    source_root: Path,
    rejection_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    metadata_paths: list[tuple[str, Path]] = []
    for split in SPLITS:
        metadata_paths.extend(
            ("pass", path)
            for path in sorted((source_root / "_metadata" / split).glob("*_metadata.json"))
        )
        metadata_paths.extend(
            ("rejected", path)
            for path in sorted(
                (source_root / "_metadata" / "rejected" / split).glob("*_metadata.json")
            )
        )
    for source_status, metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            excluded["invalid_metadata"] += 1
            continue
        class_name = str(metadata.get("class_name"))
        stage = str(metadata.get("distance_stage"))
        if class_name not in CLASSES or stage not in STAGES:
            excluded["invalid_class_or_stage"] += 1
            continue
        files = _artifact_paths(metadata)
        if files is None:
            excluded["incomplete_quartet"] += 1
            continue
        quality = evaluate_approach_quality(
            class_name=class_name,
            target_instance_ids=[int(value) for value in metadata.get("target_instance_ids", [])],
            mask_pixels=int(metadata.get("mask_pixels", 0)),
            depth_valid_ratio=float(metadata.get("depth_valid_ratio", 0.0)),
            visibility=metadata.get("visibility_reference") or {},
            rejection_config=rejection_config,
            distance_stage=stage,
        )
        if quality["status"] != "pass":
            for reason in quality["quality_reasons"]:
                excluded[f"quality:{reason}"] += 1
            excluded["quality_rejected_artifact"] += 1
            continue
        candidates.append(
            {
                "class_name": class_name,
                "stage": stage,
                "group": str(metadata.get("stable_shard")),
                "source_status": source_status,
                "source_metadata": metadata_path,
                "metadata": metadata,
                "quality": quality,
                "files": files,
                "fingerprint": _fingerprint(metadata),
            }
        )

    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        fingerprint = candidate["fingerprint"]
        current = deduplicated.get(fingerprint)
        if current is None or _candidate_score(candidate) > _candidate_score(current):
            if current is not None:
                excluded["duplicate_trajectory"] += 1
            deduplicated[fingerprint] = candidate
        else:
            excluded["duplicate_trajectory"] += 1
    return list(deduplicated.values()), excluded


def build_cell_targets(
    candidates: Iterable[dict[str, Any]],
    accepted_per_class_stage: int,
    split_percent: dict[str, int | float],
) -> dict[tuple[str, str, str], int]:
    availability = Counter((item["class_name"], item["stage"]) for item in candidates)
    targets: dict[tuple[str, str, str], int] = {}
    for class_name in CLASSES:
        for stage in STAGES:
            total = min(
                int(accepted_per_class_stage), availability[(class_name, stage)]
            )
            split_targets = balanced_split_quota(total, split_percent, SPLITS)
            for split in SPLITS:
                targets[(class_name, stage, split)] = split_targets[split]
    return targets


def assign_groups(
    candidates: list[dict[str, Any]],
    targets: dict[tuple[str, str, str], int],
    seed: int,
) -> dict[str, str]:
    """Assign every stable shard to one split while minimizing cell deficits."""

    best_assignment: dict[str, str] = {}
    best_missing: int | None = None
    for trial in range(256):
        rng = random.Random(int(seed) + trial * 104729)
        counts: Counter[tuple[str, str, str]] = Counter()
        assignment: dict[str, str] = {}
        for class_name in CLASSES:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in candidates:
                if item["class_name"] == class_name:
                    grouped[item["group"]].append(item)
            availability = Counter(item["stage"] for item in candidates if item["class_name"] == class_name)
            decorated = []
            for group, items in grouped.items():
                stage_counts = Counter(item["stage"] for item in items)
                rarity = sum(
                    min(count, 1) / max(1, availability[stage])
                    for stage, count in stage_counts.items()
                )
                decorated.append((-rarity, rng.random(), group, stage_counts))
            for _, _, group, stage_counts in sorted(decorated):
                choices = []
                for split_index, split in enumerate(SPLITS):
                    benefit = 0.0
                    remaining = 0
                    for stage, amount in stage_counts.items():
                        target = targets[(class_name, stage, split)]
                        deficit = max(0, target - counts[(class_name, stage, split)])
                        benefit += min(amount, deficit) / max(1, target)
                        remaining += deficit
                    choices.append((benefit, remaining, -split_index, split))
                split = max(choices)[3]
                assignment[group] = split
                for stage, amount in stage_counts.items():
                    counts[(class_name, stage, split)] += amount
        missing = sum(
            max(0, target - counts[cell]) for cell, target in targets.items()
        )
        if best_missing is None or missing < best_missing:
            best_missing = missing
            best_assignment = assignment
            if missing == 0:
                break
    return best_assignment


def select_candidates(
    candidates: list[dict[str, Any]],
    targets: dict[tuple[str, str, str], int],
    assignment: dict[str, str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    ordered = sorted(
        candidates,
        key=lambda item: (
            CLASSES.index(item["class_name"]),
            STAGES.index(item["stage"]),
            assignment[item["group"]],
            tuple(-value for value in _candidate_score(item)),
            str(item["source_metadata"]),
        ),
    )
    for item in ordered:
        split = assignment[item["group"]]
        cell = (item["class_name"], item["stage"], split)
        if counts[cell] >= targets[cell]:
            continue
        copy = dict(item)
        copy["split"] = split
        selected.append(copy)
        counts[cell] += 1
    return selected


def _table(counter: Counter[tuple[str, str, str]]) -> dict[str, Any]:
    return {
        class_name: {
            stage: {
                **{split: counter[(class_name, stage, split)] for split in SPLITS},
                "total": sum(counter[(class_name, stage, split)] for split in SPLITS),
            }
            for stage in STAGES
        }
        for class_name in CLASSES
    }


def write_dataset(
    source_root: Path,
    output_root: Path,
    selected: list[dict[str, Any]],
    targets: dict[tuple[str, str, str], int],
    excluded: Counter[str],
) -> dict[str, Any]:
    existing = [path for path in output_root.iterdir()] if output_root.exists() else []
    if any(path.name != ".collection.lock" for path in existing):
        raise FileExistsError(f"清洗输出目录非空，拒绝覆盖：{output_root}")
    lock = acquire_output_lock(output_root, "sanitize_approach_dataset")
    generated = datetime.now(timezone.utc).isoformat()
    # Preserve round markers and candidate plans so a later quota-resume starts
    # at a fresh seed/round instead of accidentally replaying round_00 against
    # the sanitized manifest.  They are provenance only; pass counts always
    # come from the fresh manifest written below.
    imported_plans: list[str] = []
    source_plan_dir = source_root / "plans"
    if source_plan_dir.is_dir():
        destination_plan_dir = output_root / "plans"
        destination_plan_dir.mkdir(parents=True, exist_ok=True)
        for source_plan in sorted(source_plan_dir.glob("round_*.json")):
            destination_plan = destination_plan_dir / source_plan.name
            shutil.copy2(source_plan, destination_plan)
            imported_plans.append(str(destination_plan))
    manifest_entries: list[dict[str, Any]] = []
    next_id = Counter()
    selected_counts: Counter[tuple[str, str, str]] = Counter()
    recovered_rejects = 0
    for item in selected:
        class_name = item["class_name"]
        split = item["split"]
        sample_id = f"{next_id[class_name]:06d}"
        next_id[class_name] += 1
        base = f"{class_name}_{sample_id}"
        destination_dir = output_root / split
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_files: dict[str, str] = {}
        for field, suffix in FILE_SUFFIXES.items():
            destination = destination_dir / f"{base}_{suffix}"
            shutil.copy2(item["files"][field], destination)
            destination_files[field] = str(destination)
        metadata = dict(item["metadata"])
        visibility = dict(metadata.get("visibility_reference") or {})
        visibility.update(
            scene_visible_fraction_from_reference(
                int(metadata.get("mask_pixels", 0)), visibility
            )
        )
        scene_fraction = item["quality"].get("measurements", {}).get(
            "scene_visible_fraction_of_full_mask"
        )
        visible_target = item["quality"]["thresholds"][
            "visible_fraction_target"
        ]
        metadata.update(
            {
                "status": "pass",
                "sample_id": sample_id,
                "split": split,
                "quality_reasons": [],
                "quality_warnings": item["quality"]["quality_warnings"],
                "visible_fraction_policy": {
                    "min": item["quality"]["thresholds"]["visible_fraction_min"],
                    "target": visible_target,
                    "preferred_max": item["quality"]["thresholds"]["visible_fraction_preferred_max"],
                    "metric": (
                        "real_scene_mask_pixels_over_target_only_complete_projection"
                    ),
                    "in_frame_metric": (
                        "fraction_of_target_only_wide_fov_render_inside_real_camera_crop"
                    ),
                },
                "visibility_reference": visibility,
                "visibility_target_absolute_error": (
                    abs(float(scene_fraction) - float(visible_target))
                    if scene_fraction is not None
                    else None
                ),
                "depth_valid_ratio_policy": {
                    "min": item["quality"]["thresholds"]["depth_valid_ratio_min"]
                },
                "min_mask_pixels": item["quality"]["thresholds"]["min_mask_pixels"],
                "files": destination_files,
                "sanitization": {
                    "generated_at_utc": generated,
                    "source_root": str(source_root),
                    "source_metadata": str(item["source_metadata"]),
                    "source_status": item["source_status"],
                    "source_sample_id": item["metadata"].get("sample_id"),
                },
            }
        )
        metadata_path = output_root / "_metadata" / split / f"{base}_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if item["source_status"] == "rejected":
            recovered_rejects += 1
        selected_counts[(class_name, item["stage"], split)] += 1
        sample_key = base
        manifest_entries.append(
            {
                "schema_version": 1,
                "artifact_kind": "approach_open_batch_manifest_entry",
                "generated_at_utc": generated,
                "hand_side": metadata.get("hand_side"),
                "sample": {
                    "sample_key": sample_key,
                    "hand_side": metadata.get("hand_side"),
                    "scene_group_id": Path(item["group"]).stem,
                    "class_name": class_name,
                    "distance_stage": item["stage"],
                    "distance_from_grasp_m": metadata.get("distance_from_grasp_m"),
                    "hand_perturbation": metadata.get("hand_perturbation"),
                    "split": split,
                },
                "split": split,
                "sample_id": sample_id,
                "stable": {"status": "pass", "shard": item["group"]},
                "render": {"status": "pass", "metadata": str(metadata_path)},
                "status": "pass",
                "sanitized_from": str(item["source_metadata"]),
            }
        )
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest_entries),
        encoding="utf-8",
    )
    target_counts = Counter(targets)
    missing = Counter(
        {
            cell: max(0, target - selected_counts[cell])
            for cell, target in targets.items()
            if target > selected_counts[cell]
        }
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "sanitized_approach_dataset_report",
        "generated_at_utc": generated,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "selected_samples": len(selected),
        "recovered_previously_rejected_samples": recovered_rejects,
        "excluded": dict(excluded),
        "selected_table": _table(selected_counts),
        "target_table": _table(target_counts),
        "missing_from_available_pool": _table(missing),
        "imported_source_plans": imported_plans,
        "manifest": str(manifest_path),
    }
    report_path = output_root / "sanitization_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lock.close()
    return report


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("source-root 与 output-root 不能相同")
    if args.accepted_per_class_stage <= 0:
        raise ValueError("accepted-per-class-stage 必须大于 0")
    collection = load_yaml(args.collection_config.expanduser().resolve())
    rejection = collection["approach_open"]["rejection"]
    split_percent = collection["dataset"]["split_percent"]
    candidates, excluded = load_candidates(source_root, rejection)
    targets = build_cell_targets(
        candidates, args.accepted_per_class_stage, split_percent
    )
    assignment = assign_groups(candidates, targets, args.seed)
    selected = select_candidates(candidates, targets, assignment)
    counts = Counter((item["class_name"], item["stage"], item["split"]) for item in selected)
    print(f"VALID_CANDIDATES={len(candidates)} SELECTED={len(selected)}")
    print(json.dumps(_table(counts), ensure_ascii=False, indent=2))
    if args.dry_run:
        print("STATUS=DRY_RUN")
        return 0
    report = write_dataset(source_root, output_root, selected, targets, excluded)
    print(f"REPORT={output_root / 'sanitization_report.json'}")
    print(
        f"STATUS=PASS SELECTED={report['selected_samples']} "
        f"RECOVERED_REJECTS={report['recovered_previously_rejected_samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
