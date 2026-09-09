from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.config import load_yaml  # noqa: E402
from yolo_catchdata.quota_sampling import rotating_cell_split_quotas  # noqa: E402


CLASSES = ("left", "right", "hose", "outhandle")
STAGES = ("grasp", "pre_grasp", "near", "approach", "far")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="严格审计单个腕部相机的完整正式采集结果。"
    )
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--accepted-per-class-stage", type=int, default=250)
    parser.add_argument(
        "--camera-config", type=Path, default=PROJECT_ROOT / "configs/camera.yaml"
    )
    parser.add_argument(
        "--collection-config",
        type=Path,
        default=PROJECT_ROOT / "configs/collection.yaml",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _range_or_none(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def audit(
    root: Path,
    accepted_per_class_stage: int,
    camera_config_path: Path,
    collection_config_path: Path,
):
    if accepted_per_class_stage <= 0:
        raise ValueError("accepted-per-class-stage 必须为正数")
    camera = load_yaml(camera_config_path).get("camera", {})
    collection = load_yaml(collection_config_path)
    width, height = (int(value) for value in camera["resolution"])
    clipping_min, clipping_max = (
        float(value) for value in camera["clipping_range_m"]
    )
    approach_config = collection.get("approach_open", {})
    rejection = approach_config.get("rejection", {})
    visibility_policy = rejection.get("visible_fraction_of_full_mask", {})
    default_visibility_min = float(visibility_policy.get("min", 0.45))
    visibility_min_by_class = {
        str(key): float(value)
        for key, value in visibility_policy.get("min_by_class", {}).items()
    }
    split_percent = collection.get("dataset", {}).get(
        "split_percent", {"train": 70, "val": 15, "test": 15}
    )
    quota_cells = [
        (class_name, stage) for class_name in CLASSES for stage in STAGES
    ]
    rotating_targets = rotating_cell_split_quotas(
        accepted_per_class_stage, len(quota_cells), split_percent, SPLITS
    )
    expected_cell_split_counts = {
        (class_name, stage, split): rotating_targets[cell_index][split]
        for cell_index, (class_name, stage) in enumerate(quota_cells)
        for split in SPLITS
    }
    expected_split_counts = Counter(
        {
            split: sum(
                expected_cell_split_counts[(class_name, stage, split)]
                for class_name, stage in quota_cells
            )
            for split in SPLITS
        }
    )

    errors = []
    warnings = []
    accepted = []
    for split in SPLITS:
        accepted.extend(
            (split, path, load_json(path))
            for path in sorted((root / "_metadata" / split).glob("*_metadata.json"))
        )
    rejected = []
    for split in SPLITS:
        rejected.extend(
            (split, path, load_json(path))
            for path in sorted(
                (root / "_metadata" / "rejected" / split).glob("*_metadata.json")
            )
        )

    split_counts = Counter()
    cell_counts = Counter()
    cell_split_counts = Counter()
    group_splits = defaultdict(set)
    fingerprints = Counter()
    raw_hashes = Counter()
    reason_counts = Counter()
    retry_passes = 0
    wall_lifted = 0
    wall_lift_capped = 0
    touch_border = 0
    visibility = []
    depth_ratios = []
    mask_pixels_values = []
    distance_ranges = defaultdict(list)

    for split, metadata_path, metadata in accepted:
        class_name = str(metadata.get("class_name"))
        stage = str(metadata.get("distance_stage"))
        sample_id = str(metadata.get("sample_id"))
        base = f"{class_name}_{sample_id}"
        if metadata.get("status") != "pass":
            errors.append(f"accepted metadata is not pass: {metadata_path}")
        split_counts[split] += 1
        cell_counts[(class_name, stage)] += 1
        cell_split_counts[(class_name, stage, split)] += 1
        group_splits[str(metadata.get("stable_shard"))].add(split)
        perturb = metadata.get("hand_perturbation") or {}
        fingerprint = (
            str(metadata.get("stable_shard")),
            stage,
            round(float(metadata.get("distance_from_grasp_m", 0.0)), 9),
            tuple(
                round(float(perturb.get(key, 0.0)), 9)
                for key in (
                    "lateral_offset_1_m",
                    "lateral_offset_2_m",
                    "roll_delta_deg",
                    "pitch_delta_deg",
                    "yaw_delta_deg",
                )
            ),
        )
        fingerprints[fingerprint] += 1
        distance_ranges[(class_name, stage)].append(
            float(metadata.get("distance_from_grasp_m", 0.0))
        )

        files = metadata.get("files") or {}
        expected_suffix = {
            "raw": "raw.png",
            "depth": "depth.npy",
            "mask": "mask.png",
            "pose": "pose.json",
        }
        paths = {}
        for field, suffix in expected_suffix.items():
            path = Path(str(files.get(field, "")))
            paths[field] = path
            if not path.is_file():
                errors.append(f"missing {field}: {metadata_path}")
            elif path.name != f"{base}_{suffix}":
                errors.append(f"bad filename {path.name}: {metadata_path}")
            elif path.parent.name != split:
                errors.append(f"file split mismatch {path}: {metadata_path}")
        if any(not path.is_file() for path in paths.values()):
            continue

        raw = np.asarray(Image.open(paths["raw"]))
        mask = np.asarray(Image.open(paths["mask"]))
        depth = np.load(paths["depth"], allow_pickle=False)
        pose = load_json(paths["pose"])
        if raw.shape != (height, width, 3):
            errors.append(f"bad RGB shape {raw.shape}: {paths['raw']}")
        if raw.dtype != np.uint8:
            errors.append(f"bad RGB dtype {raw.dtype}: {paths['raw']}")
        if mask.shape != (height, width):
            errors.append(f"bad mask shape {mask.shape}: {paths['mask']}")
        mask_values = set(int(value) for value in np.unique(mask))
        if not mask_values.issubset({0, 255}) or 255 not in mask_values:
            errors.append(f"mask is not nonempty binary: {paths['mask']} {mask_values}")
        mask_bool = mask == 255
        mask_count = int(mask_bool.sum())
        if mask_count != int(metadata.get("mask_pixels", -1)):
            errors.append(f"mask count mismatch: {metadata_path}")
        if depth.shape != (height, width) or depth.dtype != np.float32:
            errors.append(f"bad depth {depth.shape}/{depth.dtype}: {paths['depth']}")
        valid = (
            mask_bool
            & np.isfinite(depth)
            & (depth >= clipping_min)
            & (depth <= clipping_max)
        )
        measured_depth_ratio = float(valid.sum() / mask_count) if mask_count else 0.0
        if not math.isclose(
            measured_depth_ratio,
            float(metadata.get("depth_valid_ratio", -1.0)),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append(f"depth ratio mismatch: {metadata_path}")

        matrix = np.asarray(pose.get("camera_from_object"), dtype=float)
        rotation = np.asarray(pose.get("R_matrix"), dtype=float)
        t_vec = pose.get("t_vec") or {}
        r_vec = pose.get("R_vec") or {}
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            errors.append(f"bad pose matrix: {paths['pose']}")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            errors.append(f"bad rotation: {paths['pose']}")
        else:
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
                errors.append(f"non-orthonormal rotation: {paths['pose']}")
            if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
                errors.append(f"rotation determinant is not +1: {paths['pose']}")
        if set(t_vec) != {"x", "y", "z"} or set(r_vec) != {"a", "b", "c"}:
            errors.append(f"bad pose vector schema: {paths['pose']}")
        elif not all(math.isfinite(float(value)) for value in (*t_vec.values(), *r_vec.values())):
            errors.append(f"nonfinite pose vector: {paths['pose']}")
        elif float(t_vec["z"]) <= 0.0:
            errors.append(f"target is behind camera: {paths['pose']}")

        reference = metadata.get("visibility_reference") or {}
        scene_fraction = reference.get("scene_visible_fraction_of_full_mask")
        threshold = visibility_min_by_class.get(class_name, default_visibility_min)
        if scene_fraction is None or float(scene_fraction) + 1e-12 < threshold:
            errors.append(f"visibility below threshold: {metadata_path}")
        full_pixels = int(reference.get("full_mask_pixels") or 0)
        span = float(reference.get("span_factor") or 0.0)
        expected = float(full_pixels) * span * span
        recalculated = min(1.0, mask_count / expected) if expected > 0 else None
        if recalculated is None or not math.isclose(
            recalculated, float(scene_fraction), rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"visibility formula mismatch: {metadata_path}")

        raw_hashes[hashlib.sha256(paths["raw"].read_bytes()).hexdigest()] += 1
        if metadata.get("passed_retry_sample_id"):
            retry_passes += 1
        wall = metadata.get("trajectory_wall_clearance") or {}
        if float(wall.get("vertical_lift_m") or 0.0) > 0.0:
            wall_lifted += 1
        if wall.get("reason") == "maximum_vertical_lift_reached":
            wall_lift_capped += 1
        if metadata.get("touches_image_border"):
            touch_border += 1
        visibility.append(float(scene_fraction))
        depth_ratios.append(float(metadata.get("depth_valid_ratio")))
        mask_pixels_values.append(mask_count)

    for _, _, metadata in rejected:
        reason_counts.update(metadata.get("quality_reasons") or [])

    manifest_path = root / "manifest.jsonl"
    if manifest_path.is_file():
        manifest = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        manifest = []
        errors.append(f"missing manifest: {manifest_path}")
    manifest_status = Counter(str(entry.get("status")) for entry in manifest)
    stable_status = Counter(str((entry.get("stable") or {}).get("status")) for entry in manifest)
    render_status = Counter(str((entry.get("render") or {}).get("status")) for entry in manifest)
    attempt_reason_counts = Counter()
    for entry in manifest:
        for attempt in entry.get("render_attempts") or []:
            attempt_reason_counts.update(attempt.get("quality_reasons") or [])

    leaked_groups = {
        group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1
    }
    duplicate_trajectories = sum(count - 1 for count in fingerprints.values() if count > 1)
    duplicate_raws = sum(count - 1 for count in raw_hashes.values() if count > 1)
    if leaked_groups:
        errors.append(f"scene groups cross splits: {len(leaked_groups)}")
    if duplicate_trajectories:
        errors.append(f"duplicate trajectories: {duplicate_trajectories}")
    if duplicate_raws:
        errors.append(f"duplicate RGB files: {duplicate_raws}")
    if split_counts != expected_split_counts:
        errors.append(f"wrong split counts: {dict(split_counts)}")
    expected_cells = {
        (class_name, stage): accepted_per_class_stage
        for class_name in CLASSES
        for stage in STAGES
    }
    if dict(cell_counts) != expected_cells:
        errors.append("class-stage quota mismatch")
    if any(
        cell_split_counts[key] != expected_cell_split_counts.get(key, 0)
        for key in set(cell_split_counts) | set(expected_cell_split_counts)
    ):
        errors.append("class-stage-split quota mismatch")

    return {
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "expected_accepted": accepted_per_class_stage * len(CLASSES) * len(STAGES),
        "accepted": len(accepted),
        "rejected_metadata": len(rejected),
        "expected_split_counts": dict(expected_split_counts),
        "split_counts": dict(split_counts),
        "cell_counts": {f"{key[0]}/{key[1]}": value for key, value in sorted(cell_counts.items())},
        "manifest_entries": len(manifest),
        "manifest_status": dict(manifest_status),
        "stable_status": dict(stable_status),
        "render_status": dict(render_status),
        "attempt_rejection_reasons": dict(attempt_reason_counts),
        "rejected_artifact_reasons": dict(reason_counts),
        "retry_passes": retry_passes,
        "wall_lifted_accepted": wall_lifted,
        "wall_lift_capped_accepted": wall_lift_capped,
        "touch_border_accepted": touch_border,
        "visibility_min_max": _range_or_none(visibility),
        "depth_ratio_min_max": _range_or_none(depth_ratios),
        "mask_pixels_min_max": _range_or_none(mask_pixels_values),
        "distance_ranges": {
            f"{key[0]}/{key[1]}": [min(values), max(values)]
            for key, values in sorted(distance_ranges.items())
        },
        "split_leaks": leaked_groups,
        "duplicate_trajectories": duplicate_trajectories,
        "duplicate_raws": duplicate_raws,
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    arguments = parse_args()
    result = audit(
        arguments.camera_root.expanduser().resolve(),
        arguments.accepted_per_class_stage,
        arguments.camera_config.expanduser().resolve(),
        arguments.collection_config.expanduser().resolve(),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if arguments.report is not None:
        report_path = arguments.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    raise SystemExit(0 if result["status"] == "PASS" else 1)
