#!/usr/bin/env python3
"""Headless check for physical pickup, loaded turn, and rack transport."""

from __future__ import annotations

import argparse
from collections import Counter
import math
import re
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_USD = SCRIPT_DIR / "lab.usd"
EXTENSION_DIR = SCRIPT_DIR / "exts/kuavo.dexhand_box_carry_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--inspect-palms", action="store_true")
    parser.add_argument("--inspect-grasp-only", action="store_true")
    parser.add_argument("--inspect-scene-only", action="store_true")
    parser.add_argument("--inspect-rack2-only", action="store_true")
    parser.add_argument("--inspect-swap-layout", action="store_true")
    parser.add_argument("--inspect-rack-tags", action="store_true")
    parser.add_argument("--inspect-rack-colliders", action="store_true")
    parser.add_argument("--force-gravity-drop", action="store_true")
    parser.add_argument("--inspect-contacts", action="store_true")
    args, _ = parser.parse_known_args()
    args.usd = args.usd.expanduser().resolve()
    return args


ARGS = parse_args()

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "fast_shutdown": True,
        "headless": True,
        "multi_gpu": False,
        "open_usd": str(ARGS.usd),
    }
)

import omni.usd
from omni.physics.core import ContactEventType, get_physics_simulation_interface
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(EXTENSION_DIR))
import kuavo.dexhand_box_carry_demo.controller as controller_module  # noqa: E402
from kuavo.dexhand_box_carry_demo.controller import (  # noqa: E402
    MAXIMUM_TRANSPORT_TILT,
    MINIMUM_RELEASE_SLIDE_DISTANCE,
    TURN_ANGLE,
    DexhandBoxCarryController,
    unique_named_prim,
    wrap_angle,
    world_pose,
    yaw_from_quaternion,
)

if ARGS.force_gravity_drop:
    controller_module.PLACE_LOWER_TIMEOUT_SECONDS = 4.0


def pitch_from_quaternion(quaternion) -> float:
    import math
    import numpy as np

    w, x, y, z = np.asarray(quaternion, dtype=float)
    value = 2.0 * (w * y - z * x)
    return math.asin(float(np.clip(value, -1.0, 1.0)))


def compact_hand_contacts(paths: set[str]) -> list[str]:
    contacts = set()
    for path in paths:
        match = re.search(r"/(l|r)_(palm|index|middle|ring|little)", path.lower())
        if match:
            contacts.add(f"{match.group(1)}_{match.group(2)}")
    return sorted(contacts)


def main() -> int:
    for _ in range(60):
        simulation_app.update()

    if ARGS.inspect_palms:
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if "palm" in prim.GetName().lower():
                print(
                    f"PALM_PRIM path={prim.GetPath()} type={prim.GetTypeName()} "
                    f"apis={[str(schema) for schema in prim.GetAppliedSchemas()]}"
                )

    if ARGS.inspect_scene_only:
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        tokens = ("rack", "shelf", "storage", "flow")
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            if not prim.IsA(UsdGeom.Xform) or not any(token in name for token in tokens):
                continue
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            print(
                f"SCENE_TARGET path={prim.GetPath()} "
                f"position={world_pose(prim)[0].tolist()} "
                f"min={list(bounds.GetMin())} max={list(bounds.GetMax())}"
            )
        return 0

    if ARGS.inspect_rack2_only:
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        candidates = [
            prim
            for prim in Usd.PrimRange(
                stage.GetPrimAtPath("/World/rack2"), Usd.TraverseInstanceProxies()
            )
            if (
                "level3" in prim.GetName().lower()
                or "rack2" in prim.GetName().lower()
            )
        ] + [unique_named_prim(stage, "grab_box")]
        for prim in candidates:
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            print(
                f"RACK2_GEOMETRY path={prim.GetPath()} type={prim.GetTypeName()} "
                f"min={list(bounds.GetMin())} max={list(bounds.GetMax())}"
            )
        return 0

    if ARGS.inspect_swap_layout:
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        parents = (
            stage.GetPrimAtPath("/World"),
            stage.GetPrimAtPath("/World/Environment/Geometry"),
        )
        for parent in parents:
            for prim in parent.GetChildren():
                if not prim.IsA(UsdGeom.Xform):
                    continue
                bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                print(
                    f"SWAP_CANDIDATE path={prim.GetPath()} "
                    f"position={world_pose(prim)[0].tolist()} "
                    f"min={list(bounds.GetMin())} max={list(bounds.GetMax())} "
                    f"ops={[(op.GetOpName(), str(op.Get())) for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]}"
                )
        return 0

    if ARGS.inspect_rack_tags:
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            if "rack2_tag_" not in name and not (
                "level3" in name and any(token in name for token in ("lane", "roller", "shelf"))
            ):
                continue
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            print(
                f"RACK_DETAIL path={prim.GetPath()} type={prim.GetTypeName()} "
                f"position={world_pose(prim)[0].tolist()} "
                f"min={list(bounds.GetMin())} max={list(bounds.GetMax())} "
                f"ops={[(op.GetOpName(), str(op.Get())) for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]}"
            )
        return 0

    if ARGS.inspect_rack_colliders:
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        for root_path in (
            "/World/rack1",
            "/World/rack2",
            "/World/Environment/Geometry/rack3_flow_rack",
        ):
            root = stage.GetPrimAtPath(root_path)
            summary = Counter()
            collider_paths = []
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
                approximation = (
                    mesh_collision.GetApproximationAttr().Get()
                    if mesh_collision
                    else None
                )
                summary[(prim.GetTypeName(), bool(enabled), str(approximation))] += 1
                collider_paths.append(str(prim.GetPath()))
            print(f"RACK_COLLIDER_SUMMARY root={root_path} counts={dict(summary)}")
            print(f"RACK_COLLIDER_SAMPLE root={root_path} paths={collider_paths[:5]}")
        return 0

    statuses: list[str] = []
    controller = DexhandBoxCarryController(statuses.append)
    active_box_contacts: set[str] = set()
    contact_subscription = None
    if ARGS.inspect_contacts:
        stage = omni.usd.get_context().get_stage()
        box_prim = unique_named_prim(stage, "grab_box")
        PhysxSchema.PhysxContactReportAPI.Apply(
            box_prim
        ).CreateThresholdAttr().Set(0.0)

        def on_contact_event(contact_headers, _contact_data, _friction_anchors):
            for header in contact_headers:
                collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
                collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
                if "grab_box" not in collider0 and "grab_box" not in collider1:
                    continue
                other = collider1 if "grab_box" in collider0 else collider0
                if header.type == ContactEventType.CONTACT_FOUND:
                    active_box_contacts.add(other)
                elif header.type == ContactEventType.CONTACT_LOST:
                    active_box_contacts.discard(other)

        contact_subscription = (
            get_physics_simulation_interface()
            .subscribe_physics_contact_report_events(on_contact_event)
        )
    controller.replay()
    deadline = time.monotonic() + ARGS.timeout
    grasp_position = None
    printed_hand_bounds = False
    grasp_inspection_done = False
    last_phase = None
    try:
        while time.monotonic() < deadline and controller.phase not in {"DEMO_COMPLETE", "ERROR"}:
            simulation_app.update()
            if controller.phase != last_phase:
                last_phase = controller.phase
                stage = omni.usd.get_context().get_stage()
                trace_box_position, trace_box_orientation = world_pose(
                    unique_named_prim(stage, "grab_box")
                )
                print(
                    f"PHASE_TRACE={last_phase} BOX={trace_box_position.tolist()} "
                    f"PITCH_DEG={math.degrees(pitch_from_quaternion(trace_box_orientation)):.3f}"
                )
                if ARGS.inspect_contacts:
                    hand_contacts = compact_hand_contacts(active_box_contacts)
                    print(f"HAND_CONTACTS_{last_phase}={hand_contacts}")
            if controller.phase == "GRASP_ATTACH" and grasp_position is None:
                stage = omni.usd.get_context().get_stage()
                grasp_position, _ = world_pose(unique_named_prim(stage, "grab_box"))
            if (
                controller.phase == "GRASP_ATTACH"
                and (ARGS.inspect_palms or ARGS.inspect_grasp_only)
                and not printed_hand_bounds
            ):
                printed_hand_bounds = True
                stage = omni.usd.get_context().get_stage()
                cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                )
                box_bounds = cache.ComputeWorldBound(
                    unique_named_prim(stage, "grab_box")
                ).ComputeAlignedRange()
                print(
                    f"GRASP_BOX_BOUND min={list(box_bounds.GetMin())} "
                    f"max={list(box_bounds.GetMax())}"
                )
                for prim in stage.Traverse():
                    path = str(prim.GetPath())
                    name = prim.GetName().lower()
                    component_tokens = (
                        "_palm", "_thumb", "_index", "_middle", "_ring", "_little"
                    )
                    if (
                        prim.IsA(UsdGeom.Mesh)
                        and ("/zarm_l7_link/" in path or "/zarm_r7_link/" in path)
                        and any(token in name for token in component_tokens)
                        and not prim.GetParent().GetName().endswith("_1")
                    ):
                        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                        print(
                            f"HAND_MESH_BOUND path={path} min={list(bounds.GetMin())} "
                            f"max={list(bounds.GetMax())}"
                        )
                if ARGS.inspect_grasp_only:
                    grasp_inspection_done = True
                    break
        if grasp_inspection_done:
            return 0
        stage = omni.usd.get_context().get_stage()
        box = unique_named_prim(stage, "grab_box")
        box_position, _ = world_pose(box)
        lift_height = None if grasp_position is None else float(box_position[2] - grasp_position[2])
        _, root_orientation = world_pose(controller._root_prim)
        yaw_error = None
        if hasattr(controller, "_turn_start_yaw"):
            yaw_error = abs(
                wrap_angle(
                    controller._turn_start_yaw
                    + TURN_ANGLE
                    - yaw_from_quaternion(root_orientation)
                )
            )
        is_dynamic = not bool(
            UsdPhysics.RigidBodyAPI(box).GetKinematicEnabledAttr().Get()
        )
        has_scripted_joint = stage.GetPrimAtPath("/DexhandBoxGraspJoint").IsValid()
        root_position = world_pose(controller._root_prim)[0]
        carry_target_error = None
        final_root_target = getattr(
            controller,
            "_retreat_target_position",
            getattr(controller, "_carry_target_position", None),
        )
        if final_root_target is not None:
            carry_target_error = float(
                ((root_position[:2] - final_root_target[:2]) ** 2).sum() ** 0.5
            )
        print(f"FINAL_PHASE={controller.phase}")
        print(f"BOX_POSITION={box_position.tolist()}")
        print(f"LIFT_HEIGHT={lift_height}")
        print(f"TURN_YAW_ERROR={yaw_error}")
        print(f"BOX_IS_DYNAMIC={is_dynamic}")
        print(f"SCRIPTED_GRASP_JOINT_EXISTS={has_scripted_joint}")
        print(f"CARRY_TARGET_ERROR={carry_target_error}")
        print(f"LEVEL3_SURFACE_Z={getattr(controller, '_level3_surface_z', None)}")
        print(f"PLACED_BOX_SPEED={getattr(controller, '_placed_box_speed', None)}")
        print(f"RELEASE_SLIDE_SPEED={getattr(controller, '_release_slide_speed', None)}")
        print(
            f"POST_RELEASE_SLIDE_DISTANCE="
            f"{getattr(controller, '_post_release_slide_distance', None)}"
        )
        print(f"MAX_TRANSPORT_TILT={getattr(controller, '_max_transport_tilt', None)}")
        print(
            f"MAX_TRANSPORT_TILT_PHASE="
            f"{getattr(controller, '_max_transport_tilt_phase', None)}"
        )
        print(
            f"MAX_TRANSPORT_TILT_ELAPSED="
            f"{getattr(controller, '_max_transport_tilt_elapsed', None)}"
        )
        print(f"PLACE_TORSO_ERROR={getattr(controller, '_place_torso_error', None)}")
        print(f"USED_GRAVITY_DROP={getattr(controller, '_used_gravity_drop', False)}")
        print(
            f"POST_PLACE_RETREAT_DISTANCE="
            f"{getattr(controller, '_post_place_retreat_distance', None)}"
        )
        print(
            f"POST_PLACE_WHEEL_ROTATION="
            f"{getattr(controller, '_post_place_wheel_rotation', None)}"
        )
        print(f"FINAL_POSE_ERROR={getattr(controller, '_final_pose_error', None)}")
        print(f"LAST_STATUS={statuses[-1] if statuses else 'none'}")
        if ARGS.inspect_contacts:
            print(f"ACTIVE_BOX_CONTACTS_FINAL={sorted(active_box_contacts)}")
        if (
            controller.phase != "DEMO_COMPLETE"
            or lift_height is None
            or lift_height < 0.12
            or yaw_error is None
            or yaw_error > 0.02
            or not is_dynamic
            or has_scripted_joint
            or carry_target_error is None
            or carry_target_error > 0.02
            or not hasattr(controller, "_placed_box_speed")
            or controller._placed_box_speed > 0.10
            or not hasattr(controller, "_post_release_slide_distance")
            or controller._post_release_slide_distance < MINIMUM_RELEASE_SLIDE_DISTANCE
            or float(box_position[2]) < 1.10
            or float(box_position[2]) > controller._level3_surface_z + 0.05
            or controller._max_transport_tilt > MAXIMUM_TRANSPORT_TILT
            or not hasattr(controller, "_place_torso_error")
            or controller._place_torso_error > math.radians(2.0)
            or not hasattr(controller, "_post_place_retreat_distance")
            or controller._post_place_retreat_distance < 0.50
            or not hasattr(controller, "_post_place_wheel_rotation")
            or controller._post_place_wheel_rotation < 5.0
            or controller._final_pose_error > math.radians(4.0)
            or (
                ARGS.force_gravity_drop
                and not getattr(controller, "_used_gravity_drop", False)
            )
        ):
            return 1
        return 0
    finally:
        contact_subscription = None
        controller.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
