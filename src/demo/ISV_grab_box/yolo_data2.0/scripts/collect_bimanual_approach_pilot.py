#!/usr/bin/env python3
"""Collect a right-then-left pilot dataset with one command.

Each hand remains an independent camera branch internally, but this launcher
runs both branches consecutively and places them below one dataset root:

    <output-root>/right_wrist_d405/{train,val,test}
    <output-root>/left_wrist_d405/{train,val,test}

Only RGB/depth/mask/pose files are written inside the three dataset splits.
Manifests, quality metadata, rejected attempts, and plans stay outside them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.hand_sides import hand_spec  # noqa: E402
from yolo_catchdata.output_lock import acquire_output_lock  # noqa: E402


COLLECT_QUOTA = PROJECT_ROOT / "scripts/collect_approach_until_quota.py"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "output/manual_wrist/references"
CLASS_NAMES = ("left", "right", "hose", "outhandle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--accepted-per-class-stage",
        type=int,
        default=5,
        help="每个相机的 class×轨迹层通过配额；默认 5，共 100 张/相机。",
    )
    parser.add_argument(
        "--samples-per-class-per-round",
        type=int,
        default=25,
        help="每轮每类候选上限；默认覆盖五层各 5 个候选。",
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--isaac-python", type=Path, default=Path("/home/csc/isaacsim/python.sh")
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--class-name", choices=CLASS_NAMES, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="达到最大轮数仍有缺额时也返回 0；默认把它视为预采失败。",
    )
    return parser.parse_args()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.accepted_per_class_stage <= 0:
        raise ValueError("accepted-per-class-stage 必须为正数")
    if args.samples_per_class_per_round <= 0:
        raise ValueError("samples-per-class-per-round 必须为正数")
    if args.max_rounds <= 0:
        raise ValueError("max-rounds 必须为正数")

    output_root = args.output_root.expanduser().resolve()
    # The wrapper lock prevents a second bimanual launcher from racing this
    # run.  Each quota child additionally locks its camera-specific root.
    output_lock = acquire_output_lock(output_root, "collect_bimanual_approach")
    reference_dir = args.reference_dir.expanduser().resolve()
    isaac_python = args.isaac_python.expanduser().resolve()
    runs: list[dict[str, Any]] = []
    overall_code = 0
    for hand_side in ("right", "left"):
        camera_name = str(hand_spec(hand_side)["camera_name"])
        camera_root = output_root / camera_name
        command = [
            sys.executable,
            str(COLLECT_QUOTA),
            "--hand-side",
            hand_side,
            "--output-root",
            str(camera_root),
            "--samples-per-class-per-round",
            str(args.samples_per_class_per_round),
            "--accepted-per-class-stage",
            str(args.accepted_per_class_stage),
            "--max-rounds",
            str(args.max_rounds),
            "--seed",
            str(args.seed),
            "--isaac-python",
            str(isaac_python),
            "--reference-dir",
            str(reference_dir),
        ]
        if args.class_name:
            command.extend(["--class-name", args.class_name])
        if args.dry_run:
            command.append("--dry-run")
        if not args.allow_incomplete:
            command.append("--strict")

        print(
            f"\n========== PILOT {hand_side.upper()} / {camera_name} ==========",
            flush=True,
        )
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        runs.append(
            {
                "hand_side": hand_side,
                "camera_id": camera_name,
                "output_root": str(camera_root),
                "returncode": int(completed.returncode),
                "command": command,
            }
        )
        if completed.returncode != 0:
            overall_code = 1
            print(
                f"PILOT_BRANCH_FAILED hand_side={hand_side} "
                f"returncode={completed.returncode}; continuing with the other hand",
                flush=True,
            )

    summary_path = output_root / "pilot_run_summary.json"
    _atomic_write(
        summary_path,
        {
            "schema_version": 1,
            "artifact_kind": "bimanual_approach_pilot_run",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if overall_code == 0 else "incomplete",
            "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
            "runs": runs,
        },
    )
    print(f"\nPILOT_SUMMARY={summary_path}", flush=True)
    print(
        "STATUS=" + ("PASS" if overall_code == 0 else "INCOMPLETE"),
        flush=True,
    )
    return overall_code


if __name__ == "__main__":
    raise SystemExit(main())
