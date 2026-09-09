#!/usr/bin/env python3
"""Generate left-hand approach references from calibrated right-hand poses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_catchdata.approach_references import (  # noqa: E402
    ensure_mirrored_left_references,
    validate_reference_set,
)


CLASS_NAMES = ("left", "right", "hose", "outhandle")
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "output/manual_wrist/references"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--class-name", choices=CLASS_NAMES, action="append")
    parser.add_argument(
        "--force",
        action="store_true",
        help="同时覆盖已有的手工左手参考；默认只刷新自动镜像文件。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_dir = args.reference_dir.expanduser().resolve()
    classes = tuple(args.class_name or CLASS_NAMES)
    result = ensure_mirrored_left_references(
        reference_dir,
        classes,
        force=bool(args.force),
    )
    for path in result["written"]:
        print(f"MIRRORED_LEFT_REFERENCE={path}")
    for path in result["current"]:
        print(f"MIRRORED_LEFT_REFERENCE_CURRENT={path}")
    for path in result["preserved_manual"]:
        print(f"PRESERVED_MANUAL_LEFT_REFERENCE={path}")
    errors = validate_reference_set(reference_dir, "left", classes)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"左手镜像参考校验失败：\n{detail}")
    print(f"STATUS=pass CLASSES={','.join(classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
