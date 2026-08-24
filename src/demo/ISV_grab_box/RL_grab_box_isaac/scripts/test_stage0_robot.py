#!/usr/bin/env python3
"""Headless stage-0 smoke test for the 13-DOF robot and falling handle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_SCENE = ROOT / "scenes/kuavo_grasp_stage0.usd"
DEFAULT_NOMINAL_JSON = ROOT / "config/stage0_nominal.json"
DEFAULT_VALIDATION_JSON = ROOT / "config/stage0_validation.json"

ARM_JOINTS = tuple(f"zarm_r{i}_joint" for i in range(1, 8))
HAND_JOINTS = (
    "r_thumbCMC",
    "r_thumbMCP",
    "r_indexMCP",
    "r_indexPIP",
    "r_middleMCP",
    "r_middlePIP",
)
ACTIVE_JOINTS = ARM_JOINTS + HAND_JOINTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--nominal-json", type=Path, default=DEFAULT_NOMINAL_JSON)
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION_JSON)
    parser.add_argument("--settle-steps", type=int, default=480)
    parser.add_argument("--joint-steps", type=int, default=90)
    parser.add_argument(
        "--accept-settled-baseline",
        action="store_true",
        help="Replace the nominal settled object pose with this run's measured pose.",
    )
    args = parser.parse_args()
    args.usd = args.usd.expanduser().resolve()
    args.nominal_json = args.nominal_json.expanduser().resolve()
    args.validation_json = args.validation_json.expanduser().resolve()
    if not args.usd.is_file():
        parser.error(f"Stage USD does not exist: {args.usd}")
    if not args.nominal_json.is_file():
        parser.error(f"Nominal JSON does not exist: {args.nominal_json}")
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

import carb
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def unique_named_prim(stage: Usd.Stage, name: str, xform_only: bool = False) -> Usd.Prim:
    matches = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == name
        and (not xform_only or prim.IsA(UsdGeom.Xform))
        and (not xform_only or prim.GetParent().GetName() != name)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one prim named {name}, found "
            f"{[(str(prim.GetPath()), prim.GetTypeName()) for prim in matches]}"
        )
    return matches[0]


def world_pose(prim: Usd.Prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        np.asarray(translation, dtype=float),
        np.asarray(
            [quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
            dtype=float,
        ),
    )


def relative_pose(parent: Usd.Prim, child: Usd.Prim):
    parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    child_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    relative = child_world * parent_world.GetInverse()
    translation = relative.ExtractTranslation()
    quaternion = relative.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return np.asarray(translation, dtype=float), np.asarray(
        [quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=float
    )


def step(frames: int) -> None:
    for _ in range(frames):
        simulation_app.update()


def physics_step(frames: int) -> None:
    """Advance exactly ``frames`` fixed 120 Hz physics steps."""
    SimulationManager.step(steps=frames)


def read_q(robot: Articulation) -> np.ndarray:
    # Warp's NumPy conversion can expose a live shared buffer.  Copy so a
    # baseline snapshot cannot change underneath a later measurement.
    return as_numpy(robot.get_dof_positions()).reshape(-1, len(ACTIVE_JOINTS))[0].copy()


def main() -> int:
    nominal = json.loads(ARGS.nominal_json.read_text(encoding="utf-8"))
    arm_only = bool(nominal.get("removed_complete_robot_body", False))
    initial_q = np.asarray(
        list(nominal["right_arm_initial_joint_positions"].values())
        + list(nominal["right_hand_initial_joint_positions"].values()),
        dtype=float,
    )

    step(60)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not open the stage")

    environment_prim = stage.GetPrimAtPath("/World/Environment")
    if not environment_prim:
        raise RuntimeError("Missing /World/Environment")
    table_top = float(environment_prim.GetAttribute("stage0:tableTop").Get())
    support_top = float(environment_prim.GetAttribute("stage0:supportTop").Get())
    expected_table_top = float(nominal["table"]["top_z"])
    expected_support_top = float(nominal["temporary_support"]["top_z"])
    if not math.isclose(table_top, expected_table_top, abs_tol=1e-9):
        raise RuntimeError(f"Table top is {table_top}, expected {expected_table_top}")
    if not math.isclose(support_top, expected_support_top, abs_tol=1e-9):
        raise RuntimeError(f"Support top is {support_top}, expected {expected_support_top}")
    dome_prim = stage.GetPrimAtPath("/World/EnvironmentLight")
    if not dome_prim or not dome_prim.IsA(UsdLux.DomeLight):
        raise RuntimeError("Missing bright /World/EnvironmentLight dome")
    dome_intensity = float(UsdLux.DomeLight(dome_prim).GetIntensityAttr().Get())
    if dome_intensity < 500.0:
        raise RuntimeError(f"Environment dome is too dim: intensity={dome_intensity}")
    if dome_prim.GetAttribute("visibleInPrimaryRay").Get() is not True:
        raise RuntimeError("Environment dome is not visible as the viewport background")
    table_center = np.asarray(nominal["table"]["center_position"], dtype=float)
    table_size = np.asarray(nominal["table"]["size"], dtype=float)
    support_center = np.asarray(
        nominal["temporary_support"]["center_position"], dtype=float
    )
    object_spawn_nominal = np.asarray(
        nominal["object_spawn_world_pose"]["position"], dtype=float
    )
    if not arm_only and not math.isclose(table_center[1], 0.0, abs_tol=1e-9):
        raise RuntimeError(f"Table is not on the robot forward centerline: {table_center}")
    if np.linalg.norm(support_center[:2] - table_center[:2]) > 1e-9:
        raise RuntimeError("Temporary support is not centered on the table")
    if np.linalg.norm(object_spawn_nominal[:2] - table_center[:2]) > 1e-9:
        raise RuntimeError("Outer handle spawn is not centered on the table")
    table_near_edge_x = float(table_center[0] - table_size[0] * 0.5)
    if not arm_only and table_near_edge_x < 0.40 - 1e-9:
        raise RuntimeError(
            f"Table is too close to the robot front: near_edge_x={table_near_edge_x}"
        )

    revolute_names = [
        prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)
    ]
    if len(revolute_names) != 13 or set(revolute_names) != set(ACTIVE_JOINTS):
        raise RuntimeError(f"Unexpected revolute joints: {revolute_names}")
    pip_limits = {}
    for name in ("r_indexPIP", "r_middlePIP"):
        prim = next(
            prim
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() == name
        )
        joint = UsdPhysics.RevoluteJoint(prim)
        lower_deg = float(joint.GetLowerLimitAttr().Get())
        upper_deg = float(joint.GetUpperLimitAttr().Get())
        if not math.isclose(lower_deg, 0.0, abs_tol=1e-4) or not math.isclose(
            upper_deg, math.degrees(1.25), abs_tol=1e-3
        ):
            raise RuntimeError(
                f"{name} USD limit is [{lower_deg}, {upper_deg}] deg; "
                f"expected [0, {math.degrees(1.25)}] deg"
            )
        pip_limits[name] = {
            "lower_rad": math.radians(lower_deg),
            "upper_rad": math.radians(upper_deg),
            "lower_deg": lower_deg,
            "upper_deg": upper_deg,
        }
    if arm_only:
        body_names = {prim.GetName() for prim in stage.Traverse()}
        forbidden_body_names = {
            "base_link",
            "torso",
            "waist_yaw_link",
            "zarm_l1_link",
            "zhead_1_link",
        }
        leaked = sorted(body_names & forbidden_body_names)
        if leaked:
            raise RuntimeError(f"Complete-robot bodies leaked into arm-only scene: {leaked}")

    articulation_roots = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        and str(prim.GetPath()).startswith("/World/Robot")
    ]
    if len(articulation_roots) != 1:
        raise RuntimeError(
            f"Expected one robot articulation root, found {[str(p.GetPath()) for p in articulation_roots]}"
        )
    root_prim = articulation_roots[0]
    palm_prim = unique_named_prim(stage, "r_palm", xform_only=True)
    arm_base_prim = stage.GetPrimAtPath("/World/Robot/ArmBaseReference")
    if not arm_base_prim or not arm_base_prim.IsA(UsdGeom.Xform):
        raise RuntimeError("Missing fixed /World/Robot/ArmBaseReference frame")
    wrist_prim = unique_named_prim(stage, "zarm_r7_link", xform_only=True)
    object_prim = stage.GetPrimAtPath("/World/OuterHandle")
    if not object_prim or not object_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("OuterHandle is not a rigid body")

    finger_tokens = ("r_thumb", "r_index", "r_middle")
    finger_colliders = [
        prim
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and any(token in str(prim.GetPath()) for token in finger_tokens)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]
    if len(finger_colliders) < 6:
        raise RuntimeError(
            f"Expected colliders on the six moving finger links, found {len(finger_colliders)}"
        )

    ring_little_joints = [
        name for name in revolute_names if "ring" in name.lower() or "little" in name.lower()
    ]
    if ring_little_joints:
        raise RuntimeError(f"Ring/little fingers are unexpectedly articulated: {ring_little_joints}")

    object_start, object_start_quat = world_pose(object_prim)
    if stage.GetEndTimeCode() < 10_000.0:
        raise RuntimeError(
            f"Scene timeline is too short for validation: end={stage.GetEndTimeCode()}"
        )
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_time_codes_per_second(120.0)
    timeline.set_start_time(0.0)
    timeline.set_end_time(10_000.0)
    timeline.set_current_time(0.0)
    timeline.set_looping(False)
    timeline.commit()
    step(2)
    if timeline.get_end_time() < 10_000.0:
        raise RuntimeError(
            f"Isaac timeline did not accept the requested end time: {timeline.get_end_time()}"
        )

    SimulationManager.set_physics_sim_device("cpu")
    SimulationManager.set_physics_dt(1.0 / 120.0)
    SimulationManager.enable_fabric(False)
    carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
    timeline.play()
    step(5)
    sim_time_after_start = SimulationManager.get_simulation_time()
    if sim_time_after_start <= 0.0:
        raise RuntimeError("Kit timeline did not advance after play()")

    robot = Articulation(str(root_prim.GetPath()))
    object_body = RigidPrim("/World/OuterHandle")
    names = list(robot.dof_names)
    if len(names) != 13 or set(names) != set(ACTIVE_JOINTS):
        raise RuntimeError(f"PhysX articulation DOFs do not match the target set: {names}")
    indices = [names.index(name) for name in ACTIVE_JOINTS]
    if indices != list(range(13)):
        initial_q = initial_q[np.argsort(indices)]
    name_to_index = {name: names.index(name) for name in names}

    stiffness = as_numpy(robot.get_dof_gains()[0]).reshape(-1, 13)[0]
    damping = as_numpy(robot.get_dof_gains()[1]).reshape(-1, 13)[0]
    # The stage keeps repository gains.  The smoke test temporarily raises the
    # gains so every unloaded joint produces an unambiguous displacement in a
    # short CPU-only test window; these are not sim-to-real control parameters.
    stiffness[:] = np.asarray([2000] * 7 + [200] * 6)
    damping[:] = np.asarray([120] * 7 + [10] * 6)
    robot.set_dof_gains(stiffness, damping)
    max_effort = as_numpy(robot.get_dof_max_efforts()).reshape(-1, 13)[0]
    max_effort[:] = np.asarray([300] * 7 + [20] * 6)
    robot.set_dof_max_efforts(max_effort)
    armature = as_numpy(robot.get_dof_armatures()).reshape(-1, 13)[0]
    armature[:] = np.asarray([1.0] * 7 + [0.01] * 6)
    robot.set_dof_armatures(armature)
    robot.set_dof_positions(initial_q)
    robot.set_dof_velocities(np.zeros(13))
    robot.set_dof_position_targets(initial_q)
    physics_step(30)
    print(f"SIM_TIME_AFTER_START={SimulationManager.get_simulation_time():.6f}")
    print("PHYSX_DOF_ORDER=" + ",".join(names))
    print("INITIAL_Q_REQUEST=" + np.array2string(initial_q, precision=5))
    print("INITIAL_Q_MEASURED=" + np.array2string(read_q(robot), precision=5))
    print(
        "GAINS_STIFFNESS="
        + np.array2string(as_numpy(robot.get_dof_gains()[0]).reshape(-1, 13)[0], precision=3)
    )
    print(
        "GAINS_DAMPING="
        + np.array2string(as_numpy(robot.get_dof_gains()[1]).reshape(-1, 13)[0], precision=3)
    )
    print(
        "MAX_EFFORT="
        + np.array2string(as_numpy(robot.get_dof_max_efforts()).reshape(-1, 13)[0], precision=3)
    )

    palm_initial, palm_initial_quat = world_pose(palm_prim)
    arm_base_world, arm_base_quat = world_pose(arm_base_prim)
    wrist_to_palm_initial = relative_pose(wrist_prim, palm_prim)
    robot_root_initial, robot_root_quat = world_pose(root_prim)

    motion_results = {}
    for name in ACTIVE_JOINTS:
        index = name_to_index[name]
        robot.set_dof_positions(initial_q)
        robot.set_dof_velocities(np.zeros(13))
        robot.set_dof_position_targets(initial_q)
        physics_step(8)
        baseline = read_q(robot)
        command = initial_q.copy()
        command[index] += 0.08 if name in ARM_JOINTS else 0.15
        robot.set_dof_position_targets(command)
        if name == ACTIVE_JOINTS[0]:
            print(
                "FIRST_TARGET_READBACK="
                + np.array2string(
                    as_numpy(robot.get_dof_position_targets()).reshape(-1, 13)[0],
                    precision=5,
                )
            )
        time_before_motion = SimulationManager.get_simulation_time()
        physics_step(ARGS.joint_steps)
        time_after_motion = SimulationManager.get_simulation_time()
        if time_after_motion <= time_before_motion:
            raise RuntimeError(
                f"Timeline stopped while validating {name}: "
                f"simulation_time={time_after_motion:.6f}"
            )
        measured = read_q(robot)
        signed_motion = float(measured[index] - baseline[index])
        non_target = np.delete(np.abs(measured - baseline), index)
        max_non_target = float(np.max(non_target)) if len(non_target) else 0.0
        passed = signed_motion > (0.008 if name in ARM_JOINTS else 0.015)
        if not passed:
            print(
                f"FIRST_COMMAND_DEBUG baseline={baseline[index]:.8f} "
                f"measured={measured[index]:.8f} "
                f"velocity={as_numpy(robot.get_dof_velocities()).reshape(-1, 13)[0][index]:.8f} "
                f"effort={as_numpy(robot.get_dof_efforts()).reshape(-1, 13)[0][index]:.8f} "
                f"projected_force={as_numpy(robot.get_dof_projected_joint_forces()).reshape(-1, 13)[0][index]:.8f}"
            )
            raise RuntimeError(
                f"{name} did not follow its positive position command: delta={signed_motion:.6f}"
            )
        if max_non_target > 0.10:
            raise RuntimeError(
                f"{name} command disturbed another target joint by {max_non_target:.6f} rad"
            )
        if not np.all(np.isfinite(measured)):
            raise RuntimeError(f"Non-finite joint state after commanding {name}: {measured}")
        motion_results[name] = {
            "command_delta": 0.08 if name in ARM_JOINTS else 0.15,
            "measured_delta": signed_motion,
            "max_other_joint_delta": max_non_target,
            "pass": True,
        }
        print(
            f"JOINT_PASS name={name} measured_delta={signed_motion:.6f} "
            f"max_other_delta={max_non_target:.6f}"
        )

    robot.set_dof_positions(initial_q)
    robot.set_dof_velocities(np.zeros(13))
    robot.set_dof_position_targets(initial_q)
    physics_step(ARGS.settle_steps)

    final_q = read_q(robot)
    if not np.all(np.isfinite(final_q)):
        raise RuntimeError(f"Robot joint state became non-finite: {final_q}")
    object_final, object_final_quat = world_pose(object_prim)
    arm_base_to_object_final = relative_pose(arm_base_prim, object_prim)
    velocities = as_numpy(object_body.get_velocities()).reshape(-1, 6)[0]
    final_speed = float(np.linalg.norm(velocities[:3]))
    final_angular_speed = float(np.linalg.norm(velocities[3:]))
    drop = float(object_start[2] - object_final[2])
    if drop < 0.008:
        raise RuntimeError(
            f"Outer handle did not visibly fall under gravity: drop={drop:.6f} m"
        )
    if object_final[2] < support_top + 0.005:
        raise RuntimeError(
            f"Outer handle penetrated the support/table: center_z={object_final[2]:.6f}"
        )
    if object_final[2] > support_top + 0.10:
        raise RuntimeError(
            f"Outer handle did not settle onto the support: center_z={object_final[2]:.6f}"
        )
    if final_speed > 0.08 or final_angular_speed > 0.8:
        raise RuntimeError(
            "Outer handle is not settled: "
            f"linear_speed={final_speed:.6f}, angular_speed={final_angular_speed:.6f}"
        )
    nominal_object = nominal["object_nominal_settled_world_pose"]
    nominal_object_position = np.asarray(nominal_object["position"], dtype=float)
    nominal_object_quat = np.asarray(nominal_object["orientation"], dtype=float)
    settled_position_error = float(np.linalg.norm(object_final - nominal_object_position))
    quat_dot = float(
        np.clip(
            abs(np.dot(object_final_quat, nominal_object_quat))
            / max(
                np.linalg.norm(object_final_quat) * np.linalg.norm(nominal_object_quat),
                1e-12,
            ),
            -1.0,
            1.0,
        )
    )
    settled_orientation_error = float(2.0 * math.acos(quat_dot))
    if ARGS.accept_settled_baseline:
        nominal["object_nominal_settled_world_pose"] = {
            "position": object_final.tolist(),
            "orientation": object_final_quat.tolist(),
            "note": "Measured after free fall from the authored USD pose; accepted as the grasp-training baseline.",
        }
        nominal["arm_base_to_object_nominal"] = {
            "position": arm_base_to_object_final[0].tolist(),
            "orientation": arm_base_to_object_final[1].tolist(),
        }
        ARGS.nominal_json.write_text(
            json.dumps(nominal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        settled_position_error = 0.0
        settled_orientation_error = 0.0
    elif settled_position_error > 0.005 or settled_orientation_error > math.radians(2.0):
        raise RuntimeError(
            "Outer handle no longer reproduces the saved grasp baseline: "
            f"position_error={settled_position_error:.6f} m, "
            f"orientation_error={math.degrees(settled_orientation_error):.3f} deg"
        )

    palm_final, palm_final_quat = world_pose(palm_prim)
    wrist_to_palm_final = relative_pose(wrist_prim, palm_prim)
    palm_attachment_drift = float(
        np.linalg.norm(wrist_to_palm_final[0] - wrist_to_palm_initial[0])
    )
    if palm_attachment_drift > 1e-4:
        raise RuntimeError(
            f"Palm attachment drifted relative to zarm_r7_link by {palm_attachment_drift:.6f} m"
        )
    palm_object_distance = float(np.linalg.norm(palm_final - object_final))
    if not 0.20 <= palm_object_distance <= 0.40:
        raise RuntimeError(
            f"Nominal palm/object distance {palm_object_distance:.6f} m is outside 0.20-0.40 m"
        )

    robot_root_final, _ = world_pose(root_prim)
    root_drift = float(np.linalg.norm(robot_root_final - robot_root_initial))
    if root_drift > 1e-5:
        raise RuntimeError(f"Fixed robot root moved by {root_drift:.8f} m")

    timeline.stop()
    validation = {
        "scene": str(ARGS.usd),
        "result": "PASS",
        "active_dof_count": len(names),
        "active_dof_names_in_physx_order": names,
        "table_top_z": table_top,
        "support_top_z": support_top,
        "lighting": {
            "dome_path": str(dome_prim.GetPath()),
            "dome_intensity": dome_intensity,
        },
        "layout": {
            "robot_forward_axis": "+X",
            "table_centerline_y": float(table_center[1]),
            "table_near_edge_x": table_near_edge_x,
            "support_xy_offset_from_table_center": float(
                np.linalg.norm(support_center[:2] - table_center[:2])
            ),
            "object_spawn_xy_offset_from_table_center": float(
                np.linalg.norm(object_spawn_nominal[:2] - table_center[:2])
            ),
        },
        "joint_motion": motion_results,
        "pip_joint_limits": pip_limits,
        "robot_root_world_pose": {
            "position": robot_root_initial.tolist(),
            "orientation_wxyz": robot_root_quat.tolist(),
            "position_drift": root_drift,
        },
        "right_arm_base_world_pose": {
            "position": arm_base_world.tolist(),
            "orientation_wxyz": arm_base_quat.tolist(),
        },
        "table": nominal["table"],
        "temporary_support": nominal["temporary_support"],
        "right_palm_initial_world_pose": {
            "position": palm_initial.tolist(),
            "orientation_wxyz": palm_initial_quat.tolist(),
        },
        "right_palm_final_world_pose": {
            "position": palm_final.tolist(),
            "orientation_wxyz": palm_final_quat.tolist(),
        },
        "palm_attachment_drift": palm_attachment_drift,
        "object": {
            "spawn_position": object_start.tolist(),
            "spawn_orientation_wxyz": object_start_quat.tolist(),
            "settled_position": object_final.tolist(),
            "settled_orientation_wxyz": object_final_quat.tolist(),
            "gravity_drop": drop,
            "linear_speed": final_speed,
            "angular_speed": final_angular_speed,
            "settled_baseline_position_error": settled_position_error,
            "settled_baseline_orientation_error_rad": settled_orientation_error,
            "palm_distance": palm_object_distance,
            "arm_base_to_settled_object_pose": {
                "position": arm_base_to_object_final[0].tolist(),
                "orientation_wxyz": arm_base_to_object_final[1].tolist(),
            },
        },
        "finger_collider_count": len(finger_colliders),
        "ring_little_fixed": True,
        "final_joint_positions": dict(zip(names, final_q.tolist())),
    }
    ARGS.validation_json.parent.mkdir(parents=True, exist_ok=True)
    ARGS.validation_json.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"DOF_COUNT={len(names)}")
    print("DOF_NAMES=" + ",".join(names))
    print(f"TABLE_TOP_Z={table_top:.6f}")
    print(f"SUPPORT_TOP_Z={support_top:.6f}")
    print(f"OBJECT_DROP={drop:.6f}")
    print(f"OBJECT_SETTLED_Z={object_final[2]:.6f}")
    print(f"OBJECT_LINEAR_SPEED={final_speed:.6f}")
    print(f"OBJECT_ANGULAR_SPEED={final_angular_speed:.6f}")
    print(f"OBJECT_BASELINE_POSITION_ERROR={settled_position_error:.8f}")
    print(
        f"OBJECT_BASELINE_ORIENTATION_ERROR_DEG={math.degrees(settled_orientation_error):.6f}"
    )
    print(f"DOME_LIGHT_INTENSITY={dome_intensity:.1f}")
    print(f"TABLE_CENTERLINE_Y={table_center[1]:.6f}")
    print(f"TABLE_NEAR_EDGE_X={table_near_edge_x:.6f}")
    print(
        "SUPPORT_OBJECT_CENTER_OFFSET="
        f"{np.linalg.norm(support_center[:2] - table_center[:2]):.6f},"
        f"{np.linalg.norm(object_spawn_nominal[:2] - table_center[:2]):.6f}"
    )
    print(f"PALM_OBJECT_DISTANCE={palm_object_distance:.6f}")
    print(f"PALM_ATTACHMENT_DRIFT={palm_attachment_drift:.8f}")
    print(f"ROBOT_ROOT_DRIFT={root_drift:.8f}")
    print(f"FINGER_COLLIDER_COUNT={len(finger_colliders)}")
    print(
        "PIP_LIMITS_RAD="
        + ",".join(
            f"{name}:[{values['lower_rad']:.6f},{values['upper_rad']:.6f}]"
            for name, values in pip_limits.items()
        )
    )
    print(f"ASSET_SCOPE={'arm-only' if arm_only else 'complete-robot-fixed-chain'}")
    print(f"VALIDATION_JSON={ARGS.validation_json}")
    print("SMOKE_RESULT=PASS")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=exit_code)
    raise SystemExit(exit_code)
