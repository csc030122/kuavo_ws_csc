#!/usr/bin/env python3
"""Copy only pass rows from an approach-open manifest into a clean dataset root."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.approach_quality import evaluate_approach_quality  # noqa: E402
from yolo_catchdata.config import CONFIG_DIR, load_yaml  # noqa: E402
from yolo_catchdata.mesh_visibility import (  # noqa: E402
    scene_visible_fraction_from_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reevaluate-current-policy",
        action="store_true",
        help="重新评估 manifest 中所有 render attempts，并物化当前质量策略下最佳的一次。",
    )
    parser.add_argument(
        "--collection-config",
        type=Path,
        default=CONFIG_DIR / "collection.yaml",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _attempt_metadata_paths(entry: dict[str, Any]) -> list[Path]:
    references = [
        attempt.get("metadata") for attempt in entry.get("render_attempts", [])
    ]
    references.append(entry.get("render", {}).get("metadata"))
    result = []
    seen = set()
    for reference in references:
        if not reference:
            continue
        path = Path(reference).expanduser().resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        result.append(path)
    return result


def _reevaluate(
    metadata: dict[str, Any], rejection_config: dict[str, Any]
) -> dict[str, Any]:
    return evaluate_approach_quality(
        class_name=str(metadata["class_name"]),
        target_instance_ids=[int(value) for value in metadata.get("target_instance_ids", [])],
        mask_pixels=int(metadata.get("mask_pixels", 0)),
        depth_valid_ratio=float(metadata.get("depth_valid_ratio", 0.0)),
        visibility=metadata.get("visibility_reference", {}),
        rejection_config=rejection_config,
        distance_stage=metadata.get("distance_stage"),
    )


def _candidate_score(metadata: dict[str, Any], assessment: dict[str, Any]) -> tuple:
    fraction = assessment.get("measurements", {}).get(
        "scene_visible_fraction_of_full_mask"
    )
    target = float(assessment["thresholds"]["visible_fraction_target"])
    target_error = abs(float(fraction) - target) if fraction is not None else float("inf")
    return (
        len(assessment["quality_warnings"]),
        target_error,
        -float(metadata.get("depth_valid_ratio", 0.0)),
        -int(metadata.get("mask_pixels", 0)),
    )


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    rejection_config = (
        load_yaml(args.collection_config.expanduser().resolve())
        .get("approach_open", {})
        .get("rejection", {})
    )
    latest: dict[str, dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        key = entry.get("sample", {}).get("sample_key")
        if key:
            latest[str(key)] = entry

    copied = 0
    reaccepted = 0
    skipped = 0
    for key, entry in sorted(latest.items()):
        assessment = None
        if args.reevaluate_current_policy:
            candidates = []
            for source in _attempt_metadata_paths(entry):
                candidate = _load_json(source)
                candidate_assessment = _reevaluate(candidate, rejection_config)
                if candidate_assessment["status"] == "pass":
                    candidates.append((candidate, candidate_assessment, source))
            if not candidates:
                skipped += 1
                continue
            metadata, assessment, source_metadata = min(
                candidates,
                key=lambda item: _candidate_score(item[0], item[1]),
            )
        else:
            if entry.get("status") != "pass":
                skipped += 1
                continue
            metadata_ref = entry.get("render", {}).get("metadata")
            if not metadata_ref:
                raise FileNotFoundError(f"{key}: manifest 没有 render.metadata")
            source_metadata = Path(metadata_ref).expanduser().resolve()
            metadata = _load_json(source_metadata)
        split = str(metadata.get("split", entry.get("split", "train")))
        if metadata.get("distance_stage") is None:
            metadata["distance_stage"] = entry.get("sample", {}).get("distance_stage")
        destination_dir = output_root / split
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_files: dict[str, str] = {}
        for field, source_ref in metadata.get("files", {}).items():
            source = Path(source_ref).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(f"{key}: 找不到 {field} 文件 {source}")
            destination = destination_dir / source.name
            shutil.copy2(source, destination)
            destination_files[field] = str(destination)
        metadata["files"] = destination_files
        metadata["materialized_from_manifest"] = str(manifest_path)
        if assessment is not None:
            previous_status = metadata.get("status")
            previous_reasons = metadata.get("quality_reasons", [])
            metadata["status"] = assessment["status"]
            metadata["quality_reasons"] = assessment["quality_reasons"]
            metadata["quality_warnings"] = assessment["quality_warnings"]
            visibility = dict(metadata.get("visibility_reference") or {})
            visibility.update(
                scene_visible_fraction_from_reference(
                    int(metadata.get("mask_pixels", 0)), visibility
                )
            )
            metadata["visibility_reference"] = visibility
            scene_fraction = assessment.get("measurements", {}).get(
                "scene_visible_fraction_of_full_mask"
            )
            visible_target = assessment["thresholds"]["visible_fraction_target"]
            metadata["visibility_target_absolute_error"] = (
                abs(float(scene_fraction) - float(visible_target))
                if scene_fraction is not None
                else None
            )
            metadata["visible_fraction_policy"] = {
                "min": assessment["thresholds"]["visible_fraction_min"],
                "target": visible_target,
                "preferred_max": assessment["thresholds"][
                    "visible_fraction_preferred_max"
                ],
                "metric": (
                    "real_scene_mask_pixels_over_target_only_complete_projection"
                ),
                "in_frame_metric": (
                    "fraction_of_target_only_wide_fov_render_inside_real_camera_crop"
                ),
            }
            metadata["quality_reassessment"] = {
                "policy": "current_collection_config",
                "collection_config": str(args.collection_config.expanduser().resolve()),
                "source_metadata": str(source_metadata),
                "previous_status": previous_status,
                "previous_quality_reasons": previous_reasons,
                "thresholds": assessment["thresholds"],
            }
            if previous_status != "pass":
                reaccepted += 1
        destination_metadata = output_root / "_metadata" / split / source_metadata.name
        destination_metadata.parent.mkdir(parents=True, exist_ok=True)
        destination_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        copied += 1

    print(f"OUTPUT_ROOT={output_root}")
    print(f"PASS_SAMPLES={copied}")
    print(f"REACCEPTED_SAMPLES={reaccepted}")
    print(f"SKIPPED_SAMPLES={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
