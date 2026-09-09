#!/usr/bin/env python3
"""Run the approach-open randomization plan through Isaac Sim.

Each plan row is realized as two isolated Isaac Sim processes: one process
settles the object in the real table_box and writes a stable-pose shard; the
next process attaches the detached wrist and renders the RGB/depth/mask/Pose
sample.  The manifest is append-only so an interrupted run can be inspected
and resumed without losing completed rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COLLECT_STABLE = PROJECT_ROOT / "scripts/collect_stable_pose.py"
RENDER_HAND = PROJECT_ROOT / "scripts/render_approach_hand_smoke.py"
COLLECTION_CONFIG = PROJECT_ROOT / "configs/collection.yaml"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "output/manual_wrist/references"

try:
    from yolo_catchdata.approach_references import (
        ensure_mirrored_left_references,
        validate_reference_set,
    )
    from yolo_catchdata.config import load_yaml
    from yolo_catchdata.hand_sides import HAND_SIDES
    from yolo_catchdata.retry_policy import adjust_retry
except ImportError:  # pragma: no cover - direct script invocation fallback
    ensure_mirrored_left_references = None
    validate_reference_set = None
    load_yaml = None
    HAND_SIDES = ("right", "left")
    adjust_retry = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--hand-side",
        choices=HAND_SIDES,
        default=None,
        help="采集手侧；默认读取计划中的 hand_side。",
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path("/home/csc/isaacsim/python.sh"),
        help="Isaac Sim Python wrapper used for each child process.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Isaac Sim renderer/PhysX 使用的物理 GPU 编号。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output/dataset/approach_open",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL manifest; default is <output-root>/manifest.jsonl.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument(
        "--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="若任一候选未通过才返回非零；默认只要批处理完成就返回 0。",
    )
    return parser.parse_args()


def _split_for_local_index(local_index: int) -> str:
    bucket = int(local_index) % 20
    if bucket < 14:
        return "train"
    if bucket < 17:
        return "val"
    return "test"


def _balanced_split_counts(group_count: int) -> dict[str, int]:
    """Allocate whole scene groups as closely as possible to 70/15/15."""

    count = int(group_count)
    if count < 0:
        raise ValueError("group_count 不能为负数")
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    result = {name: int(count * ratio) for name, ratio in ratios.items()}
    remaining = count - sum(result.values())
    order = sorted(
        ratios,
        key=lambda name: (count * ratios[name] - result[name], name == "train"),
        reverse=True,
    )
    for index in range(remaining):
        result[order[index % len(order)]] += 1
    if count >= 3:
        for name in ("val", "test"):
            if result[name] == 0:
                donor = max(result, key=result.get)
                result[donor] -= 1
                result[name] += 1
    return result


def _scene_group_split_map(
    samples: list[dict[str, Any]], seed: int
) -> dict[str, str]:
    """Assign a scene group atomically, preventing cross-split pose leakage."""

    declared = [sample for sample in samples if "split" in sample]
    if declared:
        if len(declared) != len(samples):
            raise ValueError("计划不能混用显式 split 和自动 split")
        result: dict[str, str] = {}
        for sample in samples:
            group_id = str(sample.get("scene_group_id", sample["sample_key"]))
            split = str(sample["split"])
            if split not in ("train", "val", "test"):
                raise ValueError(f"计划包含非法 split：{split!r}")
            previous = result.setdefault(group_id, split)
            if previous != split:
                raise ValueError(
                    f"同一 scene group 不能跨 split：{group_id} "
                    f"包含 {previous!r} 和 {split!r}"
                )
        return result

    group_ids = sorted(
        {
            str(sample.get("scene_group_id", sample["sample_key"]))
            for sample in samples
        },
        key=lambda value: hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).digest(),
    )
    counts = _balanced_split_counts(len(group_ids))
    result: dict[str, str] = {}
    start = 0
    for split in ("train", "val", "test"):
        stop = start + counts[split]
        result.update({group_id: split for group_id in group_ids[start:stop]})
        start = stop
    return result


def _canonical_sample_id(sample_key: str, class_name: str) -> str:
    """Return the unique id portion so filenames are <class>_<id>_* only."""

    prefix = f"{class_name}_"
    if not str(sample_key).startswith(prefix):
        raise ValueError(
            f"sample_key {sample_key!r} 不符合 {class_name!r} 类文件命名约定"
        )
    sample_id = str(sample_key)[len(prefix) :]
    if not sample_id:
        raise ValueError(f"sample_key 缺少 id：{sample_key!r}")
    return sample_id


def _metadata_candidates(
    output_root: Path, split: str, metadata_name: str
) -> tuple[Path, ...]:
    """Return current paths first and legacy in-split paths last."""

    return (
        output_root / "_metadata" / split / metadata_name,
        output_root / "_metadata" / "rejected" / split / metadata_name,
        output_root / split / metadata_name,
        output_root / "rejected" / split / metadata_name,
    )


def _canonicalize_passed_retry(
    metadata_path: Path,
    metadata: dict[str, Any],
    output_root: Path,
    split: str,
    class_name: str,
    canonical_sample_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Promote a passed retry to the canonical four-file dataset basename."""

    attempt_sample_id = str(metadata.get("sample_id", canonical_sample_id))
    canonical_name = f"{class_name}_{canonical_sample_id}"
    destination_dir = output_root / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffixes = {
        "raw": "raw.png",
        "depth": "depth.npy",
        "mask": "mask.png",
        "pose": "pose.json",
    }
    canonical_files: dict[str, str] = {}
    for field, suffix in suffixes.items():
        source_ref = metadata.get("files", {}).get(field)
        if not source_ref:
            raise FileNotFoundError(f"通过样本缺少 {field} 文件引用")
        source = Path(source_ref).expanduser().resolve()
        destination = destination_dir / f"{canonical_name}_{suffix}"
        if source != destination:
            os.replace(source, destination)
        canonical_files[field] = str(destination)
    metadata["files"] = canonical_files
    metadata["sample_id"] = canonical_sample_id
    if attempt_sample_id != canonical_sample_id:
        metadata["passed_retry_sample_id"] = attempt_sample_id
    canonical_metadata = (
        output_root
        / "_metadata"
        / split
        / f"{canonical_name}_metadata.json"
    )
    canonical_metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = canonical_metadata.with_suffix(canonical_metadata.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(canonical_metadata)
    if metadata_path != canonical_metadata:
        metadata_path.unlink(missing_ok=True)
    return canonical_metadata, metadata


def _interesting_log(output: str) -> str:
    interesting = []
    for line in output.splitlines():
        if any(
            token in line
            for token in (
                "STABLE_POSE_SHARD=",
                "CLASS=",
                "SAMPLE_METADATA=",
                "RAW=",
                "DEPTH=",
                "MASK=",
                "POSE=",
                "STATUS=",
                "Traceback",
                "[py stderr]",
                "There was an error running python",
                "Error:",
            )
        ):
            interesting.append(line)
    return "\n".join(interesting[-40:])


def _run(
    command: list[str], env: dict[str, str], timeout_s: float = 180.0
) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        detail = _interesting_log(output or "")
        timeout_line = f"PROCESS_TIMEOUT={float(timeout_s):g}s"
        return 124, f"{detail}\n{timeout_line}".strip()
    return int(process.returncode), _interesting_log(output or "")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_fresh_json(path: Path, started_at_ns: int) -> dict[str, Any] | None:
    """Load only an artifact written by the current child-process attempt."""

    try:
        if path.stat().st_mtime_ns < int(started_at_ns):
            return None
    except FileNotFoundError:
        return None
    return _load_json(path)


def _sample_index(sample_key: str) -> int:
    try:
        return int(sample_key.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"非法 sample_key：{sample_key!r}") from exc


def _dataset_run_token(output_root: Path) -> str:
    """Return a stable namespace for generated stable-pose intermediates."""

    return hashlib.sha256(str(output_root.resolve()).encode("utf-8")).hexdigest()[:10]


def _expected_stable_path(sample: dict[str, Any], run_token: str) -> Path:
    stable_key = str(sample.get("scene_group_id", sample["sample_key"]))
    suffix = f"_approach_{run_token}_{stable_key}"
    return PROJECT_ROOT / "output/stable_pose/shards" / (
        f"{sample['class_name']}{suffix}.json"
    )


def _entry_status(entry: dict[str, Any]) -> str:
    if entry.get("render", {}).get("status") == "pass":
        return "pass"
    if entry.get("stable", {}).get("status") == "pass":
        return "render_fail"
    return "stable_fail"


def _entry_hand_side(entry: dict[str, Any]) -> str:
    return str(entry.get("sample", {}).get("hand_side", "right"))


def _approach_config() -> dict[str, Any]:
    if load_yaml is None:
        return {}
    try:
        return (
            load_yaml(COLLECTION_CONFIG)
            .get("approach_open", {})
        )
    except (OSError, ValueError):
        return {}


def main() -> int:
    args = parse_args()
    if args.gpu_index < 0:
        raise ValueError("gpu-index 不能为负数")
    plan_path = args.plan.expanduser().resolve()
    plan = _load_json(plan_path)
    if not plan or plan.get("artifact_kind") != "approach_open_randomization_plan":
        raise ValueError(f"不是有效的 approach_open 随机计划：{plan_path}")
    samples = list(plan.get("samples", []))
    plan_hand_side = str(plan.get("hand_side", "right"))
    hand_side = args.hand_side or plan_hand_side
    if hand_side not in HAND_SIDES:
        raise ValueError(f"计划中存在非法 hand_side：{hand_side!r}")
    if plan_hand_side != hand_side:
        raise ValueError(
            f"计划手侧为 {plan_hand_side!r}，但 --hand-side 为 {hand_side!r}"
        )
    sample_hand_sides = {
        str(item.get("hand_side", plan_hand_side)) for item in samples
    }
    if sample_hand_sides != {hand_side}:
        raise ValueError(
            f"一个计划只能属于一个手侧分支，实际为 {sorted(sample_hand_sides)}"
        )
    for sample in samples:
        sample["hand_side"] = hand_side
    if args.class_name:
        samples = [item for item in samples if item.get("class_name") == args.class_name]
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples 必须为正数")
        samples = samples[: args.max_samples]
    group_splits = _scene_group_split_map(
        samples,
        int(plan.get("effective_seed", plan.get("seed", 0))),
    )
    output_root = args.output_root.expanduser().resolve()
    run_token = _dataset_run_token(output_root)
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_root / "manifest.jsonl"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                manifest_entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        existing_sides = {_entry_hand_side(entry) for entry in manifest_entries}
        if existing_sides and existing_sides != {hand_side}:
            raise ValueError(
                f"输出根目录已属于 {sorted(existing_sides)} 分支，"
                f"不能写入 {hand_side!r}：{output_root}"
            )

    existing: dict[str, dict[str, Any]] = {}
    if args.resume:
        for entry in manifest_entries:
            key = entry.get("sample", {}).get("sample_key")
            if key:
                existing[key] = entry

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/isaacsim-matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    isaac_python = str(args.isaac_python.expanduser().resolve())
    approach_config = _approach_config()
    if validate_reference_set is None:
        raise RuntimeError("无法加载手侧参考验证器")
    reference_dir = args.reference_dir.expanduser().resolve()
    reference_classes = sorted({str(sample["class_name"]) for sample in samples})
    if hand_side == "left":
        if ensure_mirrored_left_references is None:
            raise RuntimeError("无法加载右手到左手参考镜像器")
        mirror_result = ensure_mirrored_left_references(
            reference_dir, reference_classes
        )
        for path in mirror_result["written"]:
            print(f"MIRRORED_LEFT_REFERENCE={path}", flush=True)
    reference_errors = validate_reference_set(
        reference_dir,
        hand_side,
        reference_classes,
        approach_config.get("required_snapshots", ("grasp", "pregrasp")),
    )
    if reference_errors:
        detail = "\n".join(f"- {error}" for error in reference_errors)
        raise ValueError(f"{hand_side} 手参考文件未就绪：\n{detail}")
    retry_config = approach_config.get("rejection", {}).get("retry", {})
    hand_stage_limits = approach_config.get("hand_randomization", {}).get("stages", {})
    palm_policy = approach_config.get("palm_pose", {})
    global_distance_range = palm_policy.get(
        "distance_from_grasp_m", {"min": 0.0, "max": 0.50}
    )
    distance_ranges_by_class = palm_policy.get("distance_sampling_by_class", {})
    hand_limit_scales = (
        approach_config.get("hand_randomization", {})
        .get("mixture", {})
        .get("limit_scale_by_class", {})
    )
    child_timeout_s = float(
        approach_config.get("quota_collection", {}).get("child_process_timeout_s", 180.0)
    )
    stable_process_max_attempts = max(
        1,
        int(
            approach_config.get("quota_collection", {}).get(
                "stable_process_max_attempts", 2
            )
        ),
    )
    if child_timeout_s <= 0.0:
        raise ValueError("child_process_timeout_s 必须为正数")
    max_retry_attempts = max(0, int(retry_config.get("max_attempts", 0)))
    stable_by_scene_group: dict[str, dict[str, Any]] = {}
    total = len(samples)
    completed_count = passed_count = 0
    manifest_mode = "a" if args.resume else "w"
    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest_file:
        for ordinal, sample in enumerate(samples, start=1):
            key = str(sample["sample_key"])
            if args.resume and key in existing and existing[key].get("status") == "pass":
                passed_count += 1
                completed_count += 1
                print(f"[{ordinal}/{total}] SKIP {key} (already pass)", flush=True)
                continue

            class_name = str(sample["class_name"])
            distance_stage = str(sample["distance_stage"])
            class_distance_ranges = distance_ranges_by_class.get(
                class_name, distance_ranges_by_class.get("default", {})
            )
            stage_distance_range = class_distance_ranges.get(
                distance_stage, global_distance_range
            )
            stage_minimum_distance = float(
                stage_distance_range.get("min", global_distance_range.get("min", 0.0))
            )
            stage_maximum_distance = float(
                stage_distance_range.get("max", global_distance_range.get("max", 0.50))
            )
            support_face = str(sample["support_face"])
            yaw_deg = float(sample["object_yaw_delta_deg"])
            distance = float(sample["distance_from_grasp_m"])
            xy = [float(value) for value in sample["drop_xy_world_m"]]
            scene_group_id = str(sample.get("scene_group_id", key))
            split = group_splits[scene_group_id]
            sample_id = _canonical_sample_id(key, class_name)
            stable_path = _expected_stable_path(sample, run_token)
            stable_command = [
                isaac_python,
                str(COLLECT_STABLE),
                "--class-name",
                class_name,
                f"--support-face={support_face}",
                "--yaw-deg",
                f"{yaw_deg:.9f}",
                "--drop-xy-world-m",
                f"{xy[0]:.9f}",
                f"{xy[1]:.9f}",
                "--output-suffix",
                f"approach_{run_token}_{scene_group_id}",
                "--skip-preview",
                "--gpu-index",
                str(args.gpu_index),
            ]
            base_perturbation = {
                name: float(sample.get("hand_perturbation", {}).get(name, 0.0))
                for name in (
                    "lateral_offset_1_m",
                    "lateral_offset_2_m",
                    "roll_delta_deg",
                    "pitch_delta_deg",
                    "yaw_delta_deg",
                )
            }
            entry: dict[str, Any] = {
                "schema_version": 1,
                "artifact_kind": "approach_open_batch_manifest_entry",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "hand_side": hand_side,
                "sample": sample,
                "split": split,
                "sample_id": sample_id,
                "commands": {
                    "stable": stable_command,
                    "render": [],
                },
                "stable": {},
                "render": {},
                "render_attempts": [],
            }
            print(f"[{ordinal}/{total}] {key} face={support_face} yaw={yaw_deg:.2f} split={split}", flush=True)
            if args.dry_run:
                entry["status"] = "dry_run"
                manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                manifest_file.flush()
                completed_count += 1
                continue

            stable_reused = scene_group_id in stable_by_scene_group
            if stable_reused:
                cached = stable_by_scene_group[scene_group_id]
                stable_code = int(cached["returncode"])
                stable_log = str(cached["log_tail"])
                stable_data = cached["data"]
                stable_attempts = list(cached["attempts"])
            else:
                stable_code, stable_log, stable_data = 1, "", None
                stable_attempts = []
                for stable_attempt in range(stable_process_max_attempts):
                    stable_started_at_ns = time.time_ns()
                    stable_code, stable_log = _run(
                        stable_command, env, child_timeout_s
                    )
                    stable_data = _load_fresh_json(stable_path, stable_started_at_ns)
                    stable_attempts.append(
                        {
                            "attempt": stable_attempt,
                            "returncode": stable_code,
                            "status": (stable_data or {}).get("status", "missing"),
                            "log_tail": stable_log,
                        }
                    )
                    # A real pass or fail shard is authoritative. Retry only a
                    # child process that returned without producing a fresh shard.
                    if stable_data is not None:
                        break
                stable_by_scene_group[scene_group_id] = {
                    "returncode": stable_code,
                    "log_tail": stable_log,
                    "data": stable_data,
                    "attempts": stable_attempts,
                }
            entry["stable"] = {
                "returncode": stable_code,
                "status": (stable_data or {}).get("status", "missing"),
                "shard": str(stable_path) if stable_path.exists() else None,
                "log_tail": stable_log,
                "scene_group_id": scene_group_id,
                "reused_scene_group": stable_reused,
                "process_attempts": stable_attempts,
            }
            if stable_code != 0 or not stable_data or stable_data.get("status") != "pass":
                entry["status"] = _entry_status(entry)
                manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                manifest_file.flush()
                completed_count += 1
                print(f"  STABLE_{entry['stable']['status'].upper()}", flush=True)
                continue

            render_code = 1
            metadata = None
            metadata_path = (
                output_root
                / "_metadata"
                / split
                / f"{class_name}_{sample_id}_metadata.json"
            )
            for attempt in range(max_retry_attempts + 1):
                attempt_distance = distance
                attempt_perturbation = dict(base_perturbation)
                retry_strategy = "initial"
                if attempt:
                    previous = metadata or {}
                    if adjust_retry is None:
                        raise RuntimeError("无法加载 retry policy")
                    adjustment = adjust_retry(
                        sample_key=key,
                        attempt=attempt,
                        reasons=previous.get("quality_reasons", []),
                        base_distance_m=distance,
                        base_perturbation=base_perturbation,
                        distance_stage=distance_stage,
                        retry_config=retry_config,
                        hand_stage_limits=hand_stage_limits,
                        hand_limit_scale=float(
                            sample.get(
                                "hand_limit_scale",
                                hand_limit_scales.get(class_name, 1.0),
                            )
                        ),
                        minimum_distance_m=stage_minimum_distance,
                        maximum_distance_m=stage_maximum_distance,
                    )
                    attempt_distance = float(adjustment["distance_from_grasp_m"])
                    attempt_perturbation = adjustment["hand_perturbation"]
                    retry_strategy = str(adjustment["strategy"])
                attempt_id = sample_id if attempt == 0 else f"{sample_id}_retry{attempt}"
                render_command = [
                    isaac_python,
                    str(RENDER_HAND),
                    "--stable-shard",
                    str(stable_path),
                    "--hand-side",
                    hand_side,
                    "--class-name",
                    class_name,
                    "--reference-dir",
                    str(reference_dir),
                    "--object-yaw-delta-deg",
                    f"{yaw_deg:.9f}",
                    "--distance-from-grasp-m",
                    f"{attempt_distance:.9f}",
                    "--distance-stage",
                    distance_stage,
                    "--lateral-offset-1-m",
                    f"{attempt_perturbation['lateral_offset_1_m']:.9f}",
                    "--lateral-offset-2-m",
                    f"{attempt_perturbation['lateral_offset_2_m']:.9f}",
                    "--roll-delta-deg",
                    f"{attempt_perturbation['roll_delta_deg']:.9f}",
                    "--pitch-delta-deg",
                    f"{attempt_perturbation['pitch_delta_deg']:.9f}",
                    "--palm-yaw-delta-deg",
                    f"{attempt_perturbation['yaw_delta_deg']:.9f}",
                    "--sample-id",
                    attempt_id,
                    "--split",
                    split,
                    "--output-root",
                    str(output_root),
                    "--gpu-index",
                    str(args.gpu_index),
                ]
                render_started_at_ns = time.time_ns()
                render_code, render_log = _run(render_command, env, child_timeout_s)
                metadata_name = f"{class_name}_{attempt_id}_metadata.json"
                metadata_candidates = _metadata_candidates(
                    output_root, split, metadata_name
                )
                fresh_metadata = [
                    (candidate, _load_fresh_json(candidate, render_started_at_ns))
                    for candidate in metadata_candidates
                ]
                metadata_path, metadata = next(
                    ((candidate, data) for candidate, data in fresh_metadata if data is not None),
                    (metadata_candidates[0], None),
                )
                if metadata and metadata.get("status") == "pass":
                    metadata_path, metadata = _canonicalize_passed_retry(
                        metadata_path,
                        metadata,
                        output_root,
                        split,
                        class_name,
                        sample_id,
                    )
                attempt_entry = {
                    "attempt": attempt,
                    "sample_id": attempt_id,
                    "distance_from_grasp_m": attempt_distance,
                    "hand_perturbation": attempt_perturbation,
                    "retry_strategy": retry_strategy,
                    "returncode": render_code,
                    "status": (metadata or {}).get("status", "missing"),
                    "metadata": str(metadata_path) if metadata is not None else None,
                    "log_tail": render_log,
                    "quality_reasons": (metadata or {}).get("quality_reasons", []),
                }
                entry["render_attempts"].append(attempt_entry)
                entry["commands"]["render"].append(render_command)
                if metadata and metadata.get("status") == "pass":
                    break
                # Exact grasp retries cannot change distance or perturbation; a
                # new object placement in the next quota round is the useful retry.
                if distance_stage == "grasp" and metadata is not None:
                    break
            entry["render"] = {
                "returncode": render_code,
                "status": (metadata or {}).get("status", "missing"),
                "metadata": str(metadata_path) if metadata is not None else None,
                "log_tail": entry["render_attempts"][-1].get("log_tail", "")
                if entry["render_attempts"]
                else "",
            }
            entry["status"] = _entry_status(entry)
            manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            manifest_file.flush()
            completed_count += 1
            if entry["status"] == "pass":
                passed_count += 1
            print(
                f"  {entry['status'].upper()} stable={entry['stable']['status']} "
                f"render={entry['render']['status']}",
                flush=True,
            )

    print(f"MANIFEST={manifest_path}")
    print(f"SUMMARY total={total} completed={completed_count} pass={passed_count}")
    return 0 if args.dry_run or not args.strict or passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
