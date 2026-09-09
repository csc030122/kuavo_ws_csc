#!/usr/bin/env python3
"""Collect approach-open samples until every class/stage reaches a pass quota."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.config import CONFIG_DIR, load_yaml  # noqa: E402
from yolo_catchdata.approach_references import (  # noqa: E402
    ensure_mirrored_left_references,
    validate_reference_set,
)
from yolo_catchdata.hand_sides import HAND_SIDES  # noqa: E402
from yolo_catchdata.output_lock import acquire_output_lock  # noqa: E402
from yolo_catchdata.quota_sampling import (  # noqa: E402
    allocate_missing_split_cells,
    next_round_index,
    next_sample_index,
    rotating_cell_split_quotas,
)

GENERATE_PLAN = PROJECT_ROOT / "scripts/generate_approach_open_plan.py"
COLLECT_BATCH = PROJECT_ROOT / "scripts/collect_approach_open_batch.py"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "output/manual_wrist/references"
CLASSES = ("left", "right", "hose", "outhandle")
STAGES = ("grasp", "pre_grasp", "near", "approach", "far")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-side", choices=HAND_SIDES, default="right")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-class-per-round", type=int, default=None)
    parser.add_argument("--accepted-per-class-stage", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--isaac-python", type=Path, default=Path("/home/csc/isaacsim/python.sh"))
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Isaac Sim renderer/PhysX 使用的物理 GPU 编号。",
    )
    parser.add_argument("--class-name", choices=CLASSES, default=None)
    parser.add_argument("--collection-config", type=Path, default=CONFIG_DIR / "collection.yaml")
    parser.add_argument(
        "--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _latest_manifest(output_root: Path) -> dict[str, dict]:
    manifest = output_root / "manifest.jsonl"
    latest: dict[str, dict] = {}
    if not manifest.exists():
        return latest
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        key = entry.get("sample", {}).get("sample_key")
        if key:
            latest[str(key)] = entry
    return latest


def _quota_counts(
    latest: dict[str, dict], hand_side: str = "right"
) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for entry in latest.values():
        if entry.get("status") != "pass":
            continue
        sample = entry.get("sample", {})
        if str(sample.get("hand_side", "right")) != hand_side:
            continue
        split = str(entry.get("split", sample.get("split", "train")))
        if split not in SPLITS:
            continue
        counts[
            (
                str(sample.get("class_name")),
                str(sample.get("distance_stage")),
                split,
            )
        ] += 1
    return counts


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    # Keep this handle alive for the entire function.  Kernel flock cleanup on
    # process exit also makes Ctrl-C and killed Isaac children safe.
    output_lock = acquire_output_lock(
        output_root, f"collect_approach_until_quota:{args.hand_side}"
    )
    collection = load_yaml(args.collection_config.expanduser().resolve())
    quota_config = collection.get("approach_open", {}).get("quota_collection", {})
    split_percent = collection.get("dataset", {}).get(
        "split_percent", {"train": 70, "val": 15, "test": 15}
    )
    samples_per_class_per_round = int(
        args.samples_per_class_per_round
        if args.samples_per_class_per_round is not None
        else quota_config.get("samples_per_class_per_round", 8)
    )
    accepted_per_class_stage = int(
        args.accepted_per_class_stage
        if args.accepted_per_class_stage is not None
        else quota_config.get("accepted_per_class_stage", 5)
    )
    max_rounds = int(
        args.max_rounds if args.max_rounds is not None else quota_config.get("max_rounds", 6)
    )
    if samples_per_class_per_round <= 0 or accepted_per_class_stage <= 0:
        raise ValueError("每轮候选数和通过配额必须为正数")
    if max_rounds <= 0:
        raise ValueError("max-rounds 必须为正数")
    if args.gpu_index < 0:
        raise ValueError("gpu-index 不能为负数")
    classes = (args.class_name,) if args.class_name else CLASSES
    quota_cells = [
        (class_name, stage)
        for class_name in classes
        for stage in STAGES
    ]
    rotating_targets = rotating_cell_split_quotas(
        accepted_per_class_stage,
        len(quota_cells),
        split_percent,
        SPLITS,
    )
    cell_split_target = {
        (class_name, stage, split): rotating_targets[cell_index][split]
        for cell_index, (class_name, stage) in enumerate(quota_cells)
        for split in SPLITS
    }
    reference_dir = args.reference_dir.expanduser().resolve()
    if args.hand_side == "left":
        mirror_result = ensure_mirrored_left_references(reference_dir, classes)
        for path in mirror_result["written"]:
            print(f"MIRRORED_LEFT_REFERENCE={path}", flush=True)
    reference_errors = validate_reference_set(
        reference_dir,
        args.hand_side,
        classes,
        collection.get("approach_open", {}).get(
            "required_snapshots", ("grasp", "pregrasp")
        ),
    )
    if reference_errors and not args.dry_run:
        detail = "\n".join(f"- {error}" for error in reference_errors)
        raise ValueError(
            f"{args.hand_side} 手分支参考文件未就绪：\n{detail}"
        )
    plan_dir = output_root / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    first_round_index = next_round_index(
        tuple(path.name for path in plan_dir.glob("round_*.json"))
    )
    batch_size = samples_per_class_per_round
    for local_round_index in range(max_rounds):
        round_index = first_round_index + local_round_index
        latest = _latest_manifest(output_root)
        existing_sides = {
            str(entry.get("sample", {}).get("hand_side", "right"))
            for entry in latest.values()
        }
        if existing_sides and existing_sides != {args.hand_side}:
            raise ValueError(
                f"输出根目录已属于 {sorted(existing_sides)} 分支，"
                f"不能用于 {args.hand_side!r}：{output_root}"
            )
        counts = _quota_counts(latest, args.hand_side)
        missing = {
            (class_name, stage, split): max(
                0,
                cell_split_target[(class_name, stage, split)]
                - counts[(class_name, stage, split)],
            )
            for class_name in classes
            for stage in STAGES
            for split in SPLITS
        }
        total_missing = sum(missing.values())
        print(f"ROUND={round_index} PASS={sum(counts.values())} MISSING={total_missing}", flush=True)
        if total_missing == 0:
            print("STATUS=QUOTA_REACHED", flush=True)
            return 0

        candidate_quotas = allocate_missing_split_cells(
            missing,
            classes,
            STAGES,
            SPLITS,
            batch_size,
        )

        plan_path = plan_dir / f"round_{round_index:02d}.json"
        quota_path = plan_dir / f"round_{round_index:02d}_candidate_quotas.json"
        quota_path.write_text(
            json.dumps(candidate_quotas, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # Continue above every existing index so resume and partial reruns can
        # never overwrite or shadow an older class sample key.
        start_index = next_sample_index(tuple(latest))
        generate_command = [
            sys.executable,
            str(GENERATE_PLAN),
            "--hand-side",
            args.hand_side,
            "--samples-per-class",
            str(batch_size),
            "--start-index",
            str(start_index),
            "--seed",
            str(args.seed + round_index),
            "--stage-quotas-json",
            str(quota_path),
            "--output",
            str(plan_path),
        ]
        if args.class_name:
            generate_command.extend(["--class-name", args.class_name])
        generated = subprocess.run(generate_command, cwd=PROJECT_ROOT, check=False)
        if generated.returncode != 0:
            raise RuntimeError(f"计划生成失败，退出码 {generated.returncode}")
        if args.dry_run:
            print(f"CANDIDATE_QUOTAS={quota_path}", flush=True)
            print(f"PLAN={plan_path}", flush=True)
            print("STATUS=DRY_RUN", flush=True)
            return 0
        collect_command = [
            str(args.isaac_python.expanduser().resolve()),
            str(COLLECT_BATCH),
            "--hand-side",
            args.hand_side,
            "--plan",
            str(plan_path),
            "--output-root",
            str(output_root),
            "--reference-dir",
            str(reference_dir),
            "--gpu-index",
            str(args.gpu_index),
            "--resume",
        ]
        collected = subprocess.run(collect_command, cwd=PROJECT_ROOT, check=False)
        if collected.returncode != 0:
            print(f"BATCH_EXIT={collected.returncode}（部分候选被门禁拒绝属于正常情况）", flush=True)

    latest = _latest_manifest(output_root)
    counts = _quota_counts(latest, args.hand_side)
    final_missing = sum(
        max(
            0,
            cell_split_target[(class_name, stage, split)]
            - counts[(class_name, stage, split)],
        )
        for class_name in classes
        for stage in STAGES
        for split in SPLITS
    )
    if final_missing == 0:
        print(f"STATUS=QUOTA_REACHED PASS={sum(counts.values())}", flush=True)
        return 0
    print(
        f"STATUS=QUOTA_INCOMPLETE PASS={sum(counts.values())} MISSING={final_missing}",
        flush=True,
    )
    if args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
