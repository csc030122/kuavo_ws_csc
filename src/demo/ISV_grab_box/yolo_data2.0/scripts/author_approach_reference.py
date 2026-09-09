#!/usr/bin/env python3
"""Author a two-pose reference for either independent wrist branch.

The script opens the corrected S63 scene, places exactly one workpiece in the
table box, and creates one detached wrist with its real side-mounted D405.
The user adjusts the wrist with the Isaac Sim gizmo and saves two snapshots:

``Save Grasp``
    Palm pose at the final grasp position. The fingers remain open while the
    palm pose is authored; this is the endpoint used by the approach sampler.

``Save Pre-grasp``
    Palm pose after moving backwards along the intended approach direction.

The output JSON contains both world poses, the fixed camera pose, the object
pose, finger angles, and the vector from pre-grasp to grasp. It is deliberately
small and self-contained so a later collector can use it without reading the
Isaac session layer.

All edits are authored in the USD Session Layer. The source USD is never saved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SCENE = PROJECT_ROOT / "yolo_scene/lab_yolo_grab_data_s63_mount_corrected.usd"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "output/manual_wrist/references"
DEFAULT_LIVE_DIR = PROJECT_ROOT / "output/manual_wrist"
CLASS_NAMES = ("left", "right", "hose", "outhandle")

from yolo_catchdata.hand_sides import HAND_SIDES, hand_spec, reference_filename

WORKPIECE_ROOTS = {
    name: f"/World/ManualInspection/Workpieces/{name}"
    for name in CLASS_NAMES
}
WORKPIECE_INSTANCES = {
    name: path + "/Asset" for name, path in WORKPIECE_ROOTS.items()
}
OVERVIEW_CAMERA = "/World/ManualInspection/ApproachReferenceOverview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--hand-side", choices=HAND_SIDES, default="right")
    parser.add_argument("--class-name", choices=CLASS_NAMES, default="right")
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="输出 JSON；默认 output/manual_wrist/references/<side>_<class>.json",
    )
    parser.add_argument("--live-output", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--exit-after-setup", action="store_true")
    args, _ = parser.parse_known_args()
    args.scene = args.scene.expanduser().resolve()
    if args.live_output is None:
        args.live_output = (
            DEFAULT_LIVE_DIR / f"{args.hand_side}_approach_live_pose.json"
        )
    args.live_output = args.live_output.expanduser().resolve()
    if args.reference_output is None:
        args.reference_output = (
            DEFAULT_REFERENCE_DIR
            / reference_filename(args.hand_side, args.class_name)
        )
    else:
        args.reference_output = args.reference_output.expanduser().resolve()
    return args


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _snapshot(
    stage: Any,
    hand_side: str,
    class_name: str,
    finger_state: dict[str, dict[str, Any]],
    stage_name: str,
) -> dict[str, Any]:
    from scripts.manual_adjust_left_wrist import (
        _relative_pose,
        _world_pose,
    )

    object_root = WORKPIECE_ROOTS[class_name]
    hand = hand_spec(hand_side)
    palm = str(hand["palm"])
    camera = str(hand["rgb_camera"])
    return {
        "stage": stage_name,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_from_palm": _world_pose(stage, palm),
        "world_from_camera_rgb": _world_pose(stage, camera),
        "world_from_object": _world_pose(stage, object_root),
        "palm_from_object": _relative_pose(stage, palm, object_root),
        "finger_joint_deg": {
            name: float(joint["angle_deg"])
            for name, joint in finger_state.items()
        },
    }


def _reference_payload(
    hand_side: str,
    class_name: str,
    reference_output: Path,
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "open_hand_approach_reference",
        "status": "complete" if {"grasp", "pregrasp"}.issubset(snapshots) else "incomplete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hand_side": hand_side,
        "class_name": class_name,
        "reference_output": str(reference_output),
        "hand_state": {
            "stage": "approach_open",
            "description_zh": "四指张开，大拇指 CMC=90°；其余关节由用户确认并冻结。",
            "camera_follows_hand": True,
            "camera_look_at_object": False,
        },
        "snapshots": snapshots,
    }
    if "grasp" in snapshots and "pregrasp" in snapshots:
        grasp = [
            float(value)
            for value in snapshots["grasp"]["world_from_palm"]["translation_m"]
        ]
        pregrasp = [
            float(value)
            for value in snapshots["pregrasp"]["world_from_palm"]["translation_m"]
        ]
        delta = [grasp[i] - pregrasp[i] for i in range(3)]
        length = math.sqrt(sum(value * value for value in delta))
        if length <= 1e-6:
            raise ValueError("Grasp 与 Pre-grasp 的手掌位置几乎相同，无法计算接近方向")
        object_grasp = snapshots["grasp"]["world_from_object"]["translation_m"]
        object_pregrasp = snapshots["pregrasp"]["world_from_object"]["translation_m"]
        object_delta = math.sqrt(
            sum((float(object_grasp[i]) - float(object_pregrasp[i])) ** 2 for i in range(3))
        )
        payload["approach"] = {
            "direction_world_from_pregrasp_to_grasp": [value / length for value in delta],
            "backward_direction_world": [-value / length for value in delta],
            "grasp_to_pregrasp_distance_m": length,
            "object_pose_change_between_snapshots_m": object_delta,
            "object_pose_consistent": object_delta <= 0.001,
            "sampling_rule": "palm = grasp_palm - d*a + lateral_offset; orientation = grasp_orientation * local_delta_rotation",
            "suggested_distance_layers_m": {
                "pre_grasp": [0.02, 0.08],
                "near": [0.08, 0.20],
                "approach": [0.20, 0.35],
                "far": [0.35, 0.50],
            },
        }
    return payload


def _write_reference(
    reference_output: Path,
    hand_side: str,
    class_name: str,
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = _reference_payload(
        hand_side, class_name, reference_output, snapshots
    )
    _atomic_write(reference_output, payload)
    print(
        f"REFERENCE_WRITTEN={reference_output} STATUS={payload['status']}",
        flush=True,
    )
    if "approach" in payload:
        approach = payload["approach"]
        print(
            "APPROACH_DIRECTION_WORLD="
            + json.dumps(approach["direction_world_from_pregrasp_to_grasp"]),
            flush=True,
        )
        print(
            f"APPROACH_DISTANCE_M={approach['grasp_to_pregrasp_distance_m']:.6f} "
            f"OBJECT_POSE_CONSISTENT={approach['object_pose_consistent']}",
            flush=True,
        )
    return payload


def _write_live_pose(
    stage: Any,
    hand_side: str,
    class_name: str,
    finger_state: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """Write a compact one-wrist/one-object live pose."""
    snapshot = _snapshot(stage, hand_side, class_name, finger_state, "live")
    payload = {
        "schema_version": 1,
        "artifact_kind": "approach_reference_live_pose",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hand_side": hand_side,
        "class_name": class_name,
        "world_from_palm": snapshot["world_from_palm"],
        "world_from_camera_rgb": snapshot["world_from_camera_rgb"],
        "world_from_object": snapshot["world_from_object"],
        "finger_joint_deg": {
            name: float(joint["angle_deg"])
            for name, joint in finger_state.items()
        },
    }
    _atomic_write(path, payload)
    print(f"LIVE_POSE_WRITTEN={path}", flush=True)


def _move_wrist_by_world_delta(
    stage: Any, hand_side: str, delta_world_m: list[float]
) -> None:
    """Translate the detached wrist root without changing its orientation."""
    from yolo_catchdata.isaac_utils import set_transform as _set_transform
    from scripts.manual_adjust_left_wrist import (
        _column_transform_from_pose,
        _world_pose,
    )

    wrist_root = str(hand_spec(hand_side)["manual_root"])
    current = _world_pose(stage, wrist_root)
    transform = _column_transform_from_pose(current)
    for axis in range(3):
        transform[axis][3] += float(delta_world_m[axis])
    _set_transform(stage.GetPrimAtPath(wrist_root), transform)


def _restore_wrist_pose(
    stage: Any, hand_side: str, snapshot: dict[str, Any]
) -> None:
    """Restore a previously saved world palm pose, preserving the hand-camera mount."""
    from pxr import UsdGeom
    from yolo_catchdata.isaac_utils import set_transform as _set_transform
    from scripts.manual_adjust_left_wrist import _column_transform_from_pose

    cache = UsdGeom.XformCache()
    hand = hand_spec(hand_side)
    root_prim = stage.GetPrimAtPath(str(hand["manual_root"]))
    palm_prim = stage.GetPrimAtPath(str(hand["palm"]))
    palm_from_root = (
        cache.GetLocalToWorldTransform(palm_prim)
        * cache.GetLocalToWorldTransform(root_prim).GetInverse()
    )
    desired_palm = _column_transform_from_pose(snapshot["world_from_palm"])
    from yolo_catchdata.isaac_utils import (
        gf_matrix_from_column_transform as _gf_matrix_from_column_transform,
    )

    desired_palm_gf = _gf_matrix_from_column_transform(desired_palm)
    desired_root_gf = palm_from_root.GetInverse() * desired_palm_gf
    desired_root = [
        [float(desired_root_gf[column][row]) for column in range(4)]
        for row in range(4)
    ]

    _set_transform(
        root_prim,
        desired_root,
    )


def _backward_direction_from_grasp(
    snapshot: dict[str, Any],
) -> tuple[list[float], float]:
    """Return the direction away from the object at the saved grasp pose."""
    palm = snapshot["world_from_palm"]["translation_m"]
    object_origin = snapshot["world_from_object"]["translation_m"]
    vector = [float(palm[i]) - float(object_origin[i]) for i in range(3)]
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-6:
        raise ValueError("Grasp 中手掌原点与工件原点重合，无法定义后退方向")
    return [value / length for value in vector], length


def _set_open_hand(finger_state: dict[str, dict[str, Any]]) -> None:
    from scripts.manual_adjust_left_wrist import _set_finger_angle

    for name in finger_state:
        _set_finger_angle(finger_state, name, 0.0)
    # The current D-hand reference uses a nearly straight thumb MCP while the
    # CMC is at the requested 90 degrees.
    _set_finger_angle(finger_state, "THUMB_CMC", 90.0)
    _set_finger_angle(finger_state, "THUMB_MCP", 1.4)


def _hide_unselected_workpieces(stage: Any, selected_class: str) -> None:
    from pxr import UsdGeom

    for name, path in WORKPIECE_ROOTS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        imageable = UsdGeom.Imageable(prim)
        if name == selected_class:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def _make_control_window(
    stage: Any,
    viewport: Any,
    hand_side: str,
    class_name: str,
    finger_state: dict[str, dict[str, Any]],
    reference_output: Path,
    live_output: Path,
    initial_snapshots: dict[str, dict[str, Any]] | None = None,
) -> Any:
    import omni.ui as ui
    import omni.usd

    from scripts.manual_adjust_left_wrist import _set_finger_angle

    selection = omni.usd.get_context().get_selection()
    hand = hand_spec(hand_side)
    wrist_root = str(hand["manual_root"])
    camera = str(hand["rgb_camera"])
    snapshots: dict[str, dict[str, Any]] = dict(initial_snapshots or {})

    def select(path: str) -> None:
        selection.set_selected_prim_paths([path], True)
        print(f"SELECTED={path}", flush=True)

    def set_view(path: str) -> None:
        from pxr import Sdf
        from omni.kit.viewport.utility import get_active_viewport

        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"VIEWPORT_CAMERA_ERROR=missing_prim:{path}", flush=True)
            return
        active_viewport = get_active_viewport() or viewport
        active_viewport.camera_path = Sdf.Path(path)
        print(f"VIEWPORT_CAMERA={path}", flush=True)

    def save_snapshot(stage_name: str) -> None:
        snapshot = _snapshot(
            stage, hand_side, class_name, finger_state, stage_name
        )
        snapshots[stage_name] = snapshot
        _write_reference(
            reference_output, hand_side, class_name, snapshots
        )
        print(
            f"SAVED_{stage_name.upper()} palm="
            f"{snapshot['world_from_palm']['translation_m']}",
            flush=True,
        )

    def move_back(distance_m: float) -> None:
        grasp = snapshots.get("grasp")
        if grasp is None:
            print("MOVE_BACK_ERROR=save_grasp_first", flush=True)
            return
        direction, grasp_radius = _backward_direction_from_grasp(grasp)
        delta = [float(distance_m) * value for value in direction]
        _move_wrist_by_world_delta(stage, hand_side, delta)
        print(
            f"WRIST_MOVED_BACK_M={distance_m:.3f} "
            f"DIRECTION_WORLD={direction} "
            f"GRASP_OBJECT_DISTANCE_M={grasp_radius:.6f}",
            flush=True,
        )

    def return_to_grasp() -> None:
        grasp = snapshots.get("grasp")
        if grasp is None:
            print("RETURN_TO_GRASP_ERROR=save_grasp_first", flush=True)
            return
        _restore_wrist_pose(stage, hand_side, grasp)
        print("WRIST_RETURNED_TO_SAVED_GRASP=1", flush=True)

    finger_models: dict[str, Any] = {}
    for name, joint in finger_state.items():
        model = ui.SimpleFloatModel(float(joint["angle_deg"]))
        model.add_value_changed_fn(
            lambda value_model, joint_name=name: _set_finger_angle(
                finger_state, joint_name, value_model.as_float
            )
        )
        finger_models[name] = model

    def set_open_hand() -> None:
        _set_open_hand(finger_state)
        for name, model in finger_models.items():
            model.set_value(float(finger_state[name]["angle_deg"]))
        print("HAND_STATE=approach_open CMC=90.0", flush=True)

    def print_current_pose() -> None:
        current = _snapshot(
            stage, hand_side, class_name, finger_state, "live"
        )
        print(
            "CURRENT_POSE=" + json.dumps(current, ensure_ascii=False),
            flush=True,
        )

    def joint_row(label: str, joint_name: str) -> None:
        joint = finger_state[joint_name]
        with ui.HStack(height=28, spacing=5):
            ui.Label(label, width=126)
            ui.FloatSlider(
                model=finger_models[joint_name],
                min=float(joint["min_deg"]),
                max=float(joint["max_deg"]),
            )
            ui.FloatField(model=finger_models[joint_name], width=62)

    window = ui.Window("Approach Reference Authoring", width=480, height=900)
    with window.frame:
        with ui.VStack(spacing=6, height=0):
            ui.Label("Single workpiece: " + class_name, height=25)
            ui.Label(
                f"{hand_side.capitalize()} wrist D405 | W=translate | E=rotate",
                height=22,
            )
            with ui.HStack(height=32, spacing=6):
                ui.Button("Select Wrist", clicked_fn=lambda: select(wrist_root))
                ui.Button(
                    "Select Workpiece",
                    clicked_fn=lambda: select(WORKPIECE_ROOTS[class_name]),
                )
            with ui.HStack(height=32, spacing=6):
                ui.Button("D405 View", clicked_fn=lambda: set_view(camera))
                ui.Button("Overview", clicked_fn=lambda: set_view(OVERVIEW_CAMERA))

            ui.Separator(height=5)
            ui.Label("Approach hand state", height=22)
            ui.Button("Set Open Hand (CMC=90)", clicked_fn=set_open_hand, height=32)

            ui.Label("Optional finger fine tuning", height=22)
            joint_row("Thumb CMC 0-90", "THUMB_CMC")
            joint_row("Thumb MCP 0-50", "THUMB_MCP")
            joint_row("Index MCP 0-75", "INDEX_MCP")
            joint_row("Index PIP 0-120", "INDEX_PIP")
            joint_row("Middle MCP 0-75", "MIDDLE_MCP")
            joint_row("Middle PIP 0-120", "MIDDLE_PIP")
            joint_row("Ring MCP 0-75", "RING_MCP")
            joint_row("Ring PIP 0-120", "RING_PIP")
            joint_row("Little MCP 0-75", "LITTLE_MCP")
            joint_row("Little PIP 0-120", "LITTLE_PIP")

            ui.Separator(height=5)
            ui.Label("Save two palm reference poses", height=22)
            with ui.HStack(height=35, spacing=6):
                ui.Button("Save Grasp", clicked_fn=lambda: save_snapshot("grasp"))
                ui.Button(
                    "Save Pre-grasp",
                    clicked_fn=lambda: save_snapshot("pregrasp"),
                )
            ui.Label(
                "After Save Grasp, move directly away from the object:",
                height=22,
            )
            with ui.HStack(height=34, spacing=5):
                ui.Button("Back 5 cm", clicked_fn=lambda: move_back(0.05))
                ui.Button("Back 10 cm", clicked_fn=lambda: move_back(0.10))
                ui.Button("Back 20 cm", clicked_fn=lambda: move_back(0.20))
                ui.Button("Return Grasp", clicked_fn=return_to_grasp)
            with ui.HStack(height=32, spacing=6):
                ui.Button(
                    "Print Current Pose",
                    clicked_fn=print_current_pose,
                )
                ui.Button(
                    "Export Live Pose",
                    clicked_fn=lambda: _write_live_pose(
                        stage, hand_side, class_name, finger_state, live_output
                    ),
                )
            ui.Label(
                "1) Set Open Hand. 2) Move wrist to final grasp palm pose and save Grasp. "
                "3) Click Back 10 cm (or use W/E for another distance) and save Pre-grasp.",
                word_wrap=True,
                height=58,
            )
            ui.Label(
                "The object must not move between the two saves. Camera stays rigidly attached to the wrist.",
                word_wrap=True,
                height=38,
            )
            if "grasp" in snapshots:
                ui.Label(
                    "Loaded an existing Grasp reference. Back buttons are ready.",
                    word_wrap=True,
                    height=30,
                )
    return window


def main() -> int:
    args = parse_args()
    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)

    reference_payload: dict[str, Any] = {}
    if args.reference_output.is_file():
        reference_payload = json.loads(
            args.reference_output.read_text(encoding="utf-8")
        )
        if reference_payload.get("class_name") not in (None, args.class_name):
            raise ValueError(
                f"参考文件类别为 {reference_payload.get('class_name')!r}，"
                f"但当前选择了 {args.class_name!r}：{args.reference_output}"
            )
        if reference_payload.get("hand_side") not in (None, args.hand_side):
            raise ValueError(
                f"参考文件手侧为 {reference_payload.get('hand_side')!r}，"
                f"但当前选择了 {args.hand_side!r}：{args.reference_output}"
            )
    initial_snapshots = reference_payload.get("snapshots", {})

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": bool(args.headless),
            "fast_shutdown": True,
            "multi_gpu": False,
            "width": 1440,
            "height": 900,
        }
    )
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        from scripts.manual_adjust_left_wrist import (
            _create_detached_wrist,
            _create_overview_camera,
            _hide_robot,
            _prepare_finger_control,
            _set_right_arm_pose,
            _spawn_workpieces,
        )
        from yolo_catchdata.isaac_utils import set_transform as _set_transform
        from scripts.manual_adjust_left_wrist import _column_transform_from_pose

        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene)):
            raise RuntimeError(f"Could not open corrected scene: {args.scene}")
        for _ in range(25):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Could not obtain Isaac Sim Stage")
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))

        if args.hand_side == "right":
            _set_right_arm_pose(stage)
        for _ in range(3):
            app.update()
        centers = _spawn_workpieces(stage)
        _create_detached_wrist(stage, args.scene, args.hand_side)
        finger_state = _prepare_finger_control(stage, args.hand_side)
        _set_open_hand(finger_state)

        # Continue an earlier authoring session. Restore the object and palm
        # from the saved grasp so the Back buttons act on the same geometry the
        # user originally inspected.
        restore_snapshot = initial_snapshots.get("grasp")
        if restore_snapshot is None:
            restore_snapshot = initial_snapshots.get("pregrasp")
        if restore_snapshot is not None:
            _set_transform(
                stage.GetPrimAtPath(WORKPIECE_ROOTS[args.class_name]),
                _column_transform_from_pose(
                    restore_snapshot["world_from_object"]
                ),
            )
            _restore_wrist_pose(stage, args.hand_side, restore_snapshot)
            print(
                f"RESTORED_REFERENCE_STAGE={restore_snapshot.get('stage', 'unknown')}",
                flush=True,
            )
        _hide_robot(stage)
        _hide_unselected_workpieces(stage, args.class_name)
        overview_target = centers[args.class_name]
        if restore_snapshot is not None:
            overview_target = restore_snapshot["world_from_object"]["translation_m"]
        _create_overview_camera(stage, overview_target)
        for _ in range(10):
            app.update()

        print("\n========== APPROACH REFERENCE MODE READY ==========")
        print(f"SCENE={args.scene}")
        print(f"HAND_SIDE={args.hand_side}")
        print(f"TARGET_CLASS={args.class_name}")
        print(f"TARGET_ROOT={WORKPIECE_ROOTS[args.class_name]}")
        selected_hand = hand_spec(args.hand_side)
        print(f"WRIST_ROOT={selected_hand['manual_root']}")
        print(f"D405_CAMERA={selected_hand['rgb_camera']}")
        print(f"REFERENCE_OUTPUT={args.reference_output}")
        print(f"LIVE_OUTPUT={args.live_output}")
        print("HAND_STATE=approach_open FOUR_FINGERS_OPEN THUMB_CMC=90")
        print("SOURCE_USD_SAVED=0")
        print("Use the UI buttons. Close Isaac Sim to exit.")
        print("===================================================\n", flush=True)
        _write_live_pose(
            stage, args.hand_side, args.class_name, finger_state, args.live_output
        )

        if not args.headless:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("No active Isaac Sim viewport")
            from pxr import Sdf

            viewport.camera_path = Sdf.Path(OVERVIEW_CAMERA)
            window = _make_control_window(
                stage,
                viewport,
                args.hand_side,
                args.class_name,
                finger_state,
                args.reference_output,
                args.live_output,
                initial_snapshots,
            )
            _ = window

        if args.exit_after_setup:
            return 0

        last_live = time.monotonic()
        while app.is_running():
            app.update()
            now = time.monotonic()
            if now - last_live >= 0.2:
                _write_live_pose(
                    stage,
                    args.hand_side,
                    args.class_name,
                    finger_state,
                    args.live_output,
                )
                last_live = now
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
