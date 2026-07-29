#!/usr/bin/env python3
"""Open an imported lab USD with the Isaac Sim 6 standalone runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


def find_latest_usd(generated_dir: Path) -> Path:
    candidates = [
        path
        for path in generated_dir.rglob("*")
        if path.suffix.lower() in {".usd", ".usda", ".usdc"}
        and "payload" not in {part.lower() for part in path.parts}
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No USD was found below {generated_dir}. Run import_mjcf.sh first."
        )
    preferred = [
        path
        for path in candidates
        if "lab_sence_static_with_box" in path.stem.lower()
    ]
    return max(preferred or candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd",
        type=Path,
        help="USD to open. Defaults to the newest generated lab USD.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Stop after this many simulated wall-clock seconds; 0 means run until closed.",
    )
    parser.add_argument(
        "--renderer",
        default="RaytracedLighting",
        choices=("RaytracedLighting", "PathTracing"),
    )
    args = parser.parse_args()
    if args.usd is None:
        args.usd = find_latest_usd(script_dir / "generated")
    args.usd = args.usd.expanduser().resolve()
    if not args.usd.is_file():
        parser.error(f"USD does not exist: {args.usd}")
    if args.headless and args.seconds <= 0:
        args.seconds = 10.0
    return args


def main() -> int:
    args = parse_args()

    # SimulationApp must be created before importing other Kit/Omniverse modules.
    from isaacsim.simulation_app import SimulationApp

    print(f"Opening USD: {args.usd}")
    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "open_usd": str(args.usd),
            "renderer": args.renderer,
            "multi_gpu": True,
            "max_gpu_count": 2,
        }
    )

    import time

    import omni.timeline

    for _ in range(30):
        simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    start = time.monotonic()
    try:
        while simulation_app.is_running():
            simulation_app.update()
            if args.seconds > 0 and time.monotonic() - start >= args.seconds:
                break
    finally:
        timeline.stop()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

