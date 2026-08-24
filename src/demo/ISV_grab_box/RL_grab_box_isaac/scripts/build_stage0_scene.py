#!/usr/bin/env python3
"""Build the independent Kuavo S63 stage-0 grasp scene.

Run this file with Isaac Sim 6's python.sh.  It intentionally does not import
Isaac Lab and does not create an RL task: the output is only the robot asset,
the rigid outer-handle asset, the table/support asset, and one reference USD.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
ISV_DIR = ROOT.parent
REPO_ROOT = ROOT.parents[3]

SOURCE_ROBOT = ISV_DIR / "isaac_sim/biped_s63_Dhand_fixed.xml"
SOURCE_S49_XML = REPO_ROOT / "src/kuavo_assets/models/biped_s49/xml/biped_s49.xml"
SOURCE_S49_URDF = REPO_ROOT / "src/kuavo_assets/models/biped_s49/urdf/biped_s49.urdf"
SOURCE_HANDLE_USD = ROOT / "tpyrced_数模/外拉手.usd"

ROBOT_XML = ROOT / "assets/robots/kuavo_s63_dexhand_rl.xml"
ROBOT_OUTPUT_DIR = ROOT / "assets/robots/generated"
OBJECT_USD = ROOT / "assets/objects/outer_handle_stage0.usd"
ENVIRONMENT_USD = ROOT / "assets/environment/stage0_table_support.usd"
SCENE_USD = ROOT / "scenes/kuavo_grasp_stage0.usd"
NOMINAL_JSON = ROOT / "config/stage0_nominal.json"

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

# Use the S49 URDF hard limits as the single source for both MJCF joint ranges
# and position-actuator control ranges.  The reference S49 MJCF incorrectly
# leaves the two PIP physical ranges at 2.0944 rad while its actuators and URDF
# both limit them to 1.25 rad.
HAND_JOINT_LIMITS = {
    "r_thumbCMC": (0.0, 1.5708),
    "r_thumbMCP": (0.0, 0.87266),
    "r_indexMCP": (0.0, 1.309),
    "r_indexPIP": (0.0, 1.25),
    "r_middleMCP": (0.0, 1.309),
    "r_middlePIP": (0.0, 1.25),
}

# S63 arm gains copied from src/kuavo_assets/config/kuavo_v63/kuavo.json
# (ruiwo_kp/ruiwo_kd, right-arm seven-element block).
ARM_KP = (270.0, 210.0, 256.0, 180.0, 120.0, 120.0, 120.0)
ARM_KD = (4.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0)

# S49 only specifies position actuators and 1.0 joint damping, not a physical
# servo stiffness.  These modest Isaac-only gains are for stage-0 motion
# validation and are deliberately kept out of the source/reference models.
HAND_KP = (20.0,) * 6
HAND_KD = (2.0,) * 6

ARM_Q = (
    -1.04022829,
    -0.61722983,
    1.03403416,
    -2.02895911,
    0.56224872,
    0.88393094,
    0.44813200,
)
HAND_Q = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)

ROBOT_POSITION = (0.0, 0.0, 0.0)
ARM_BASE_POSITION = (0.11075, -0.253, 1.19601)
PALM_POSITION = (0.47986943, -0.15008352, 1.12618281)
# Natural shoulder/elbow IK solution, regularized toward the repository's
# front-of-robot right-arm approach pose.  The palm is 0.30 m from the handle,
# approaches forward/inward/downward, and keeps >= 0.25 rad joint-limit margin.
PALM_ORIENTATION_WXYZ = (0.11631499, 0.43230485, 0.85826107, -0.25094079)
PALM_APPROACH_AXIS = (0.80043720, 0.50028247, -0.33017835)

TABLE_TOP_Z = 1.000
SUPPORT_TOP_Z = 1.010
TABLE_CENTER = (0.72, 0.0, 0.95)
TABLE_SIZE = (0.64, 0.70, 0.10)
SUPPORT_CENTER = (0.72, 0.0, 1.005)
SUPPORT_SIZE = (0.28, 0.28, 0.010)
OBJECT_XY = (0.72, 0.0)
DROP_HEIGHT = 0.020
OBJECT_MASS = 0.18
# Reuse the naturally settled orientation from the first validated scene, but
# recenter it on the front table/support so grasp training has a clean baseline.
OBJECT_SPAWN_Z = 1.04712893
OBJECT_SPAWN_ORIENTATION_WXYZ = (
    0.76987970,
    -0.04931977,
    0.61819910,
    0.15060773,
)
OBJECT_GRASP_POSITION = (
    0.7200040817260742,
    -0.00000016577541828155518,
    1.0271283388137817,
)
OBJECT_GRASP_ORIENTATION_WXYZ = (
    0.7698766589164734,
    -0.04932180068623808,
    0.6182035234807731,
    0.1506044275200183,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-robot-import",
        action="store_true",
        help="Regenerate non-robot assets and scene using the existing imported robot package.",
    )
    parser.add_argument(
        "--robot-usd",
        type=Path,
        help="Use an already imported robot USD instead of importing the derived MJCF.",
    )
    return parser.parse_args()


ARGS = parse_args()

# Isaac Sim 6 exposes pxr and its importer after Kit initialization.
from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {"fast_shutdown": True, "headless": True, "multi_gpu": False}
)

import omni.kit.app
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


def ensure_directories() -> None:
    for directory in (
        ROBOT_XML.parent,
        ROBOT_OUTPUT_DIR,
        OBJECT_USD.parent,
        ENVIRONMENT_USD.parent,
        SCENE_USD.parent,
        NOMINAL_JSON.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def uncomment_active_finger_joints(text: str) -> str:
    for name in HAND_JOINTS:
        pattern = re.compile(
            rf"<!--\s*(<joint\s+name=[\"']{re.escape(name)}[\"'][^>]*?/>)\s*-->"
        )
        text, count = pattern.subn(r"\1", text)
        if count != 1:
            raise RuntimeError(f"Expected one commented S49 joint named {name}, found {count}")
    return text.replace("--&gt;", "")


def remove_element(parent: ET.Element, child: ET.Element) -> None:
    parent.remove(child)


def create_robot_mjcf() -> None:
    if not SOURCE_ROBOT.is_file():
        raise FileNotFoundError(SOURCE_ROBOT)
    if not SOURCE_S49_XML.is_file() or not SOURCE_S49_URDF.is_file():
        raise FileNotFoundError("The S49 XML/URDF reference files are required")

    source = uncomment_active_finger_joints(SOURCE_ROBOT.read_text(encoding="utf-8"))
    root = ET.fromstring(source)
    root.set("model", "kuavo_s63_dexhand_rl")
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("S63 source MJCF has no compiler element")
    compiler.set("meshdir", "../../../biped_s63/meshes")

    # Removing every non-target joint welds all other complete-S63 bodies to
    # their parent.  No geometry or S63->Dhand mounting body transform changes.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "freejoint":
                remove_element(parent, child)
            elif child.tag == "joint" and child.get("name") not in ACTIVE_JOINTS:
                remove_element(parent, child)

    found = [joint.get("name") for joint in root.iter("joint")]
    if len(found) != 13 or set(found) != set(ACTIVE_JOINTS):
        raise RuntimeError(f"Derived MJCF has unexpected joints: {found}")

    for name in ARM_JOINTS:
        joint = next(j for j in root.iter("joint") if j.get("name") == name)
        joint.set("armature", "0.05")
        joint.set("damping", "0.02")
        joint.set("frictionloss", "0.02")

    for name in HAND_JOINTS:
        joint = next(j for j in root.iter("joint") if j.get("name") == name)
        lower, upper = HAND_JOINT_LIMITS[name]
        joint.set("limited", "true")
        joint.set("range", f"{lower:g} {upper:g}")

    old_actuator = root.find("actuator")
    if old_actuator is not None:
        root.remove(old_actuator)
    actuator = ET.SubElement(root, "actuator")
    for name, kp in zip(ARM_JOINTS, ARM_KP):
        ET.SubElement(
            actuator,
            "position",
            name=f"{name}_position",
            joint=name,
            gear="1",
            kp=str(kp),
        )
    for name in HAND_JOINTS:
        lower, upper = HAND_JOINT_LIMITS[name]
        ET.SubElement(
            actuator,
            "position",
            name=f"{name}_position",
            joint=name,
            gear="1",
            ctrllimited="true",
            ctrlrange=f"{lower:g} {upper:g}",
        )

    old_sensor = root.find("sensor")
    if old_sensor is not None:
        root.remove(old_sensor)
    sensor = ET.SubElement(root, "sensor")
    for name in ACTIVE_JOINTS:
        ET.SubElement(sensor, "jointpos", name=f"{name}_pos", joint=name)
        ET.SubElement(sensor, "jointvel", name=f"{name}_vel", joint=name)

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    header = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<!-- Generated by scripts/build_stage0_scene.py. Do not edit the source demo model. -->\n"
    )
    ROBOT_XML.write_text(header + xml + "\n", encoding="utf-8")


def find_existing_robot_usd() -> Path:
    candidates = sorted(
        ROBOT_OUTPUT_DIR.glob("kuavo_s63_dexhand_rl*/kuavo_s63_dexhand_rl.usda")
    )
    if not candidates:
        raise FileNotFoundError(
            "No imported robot package exists; rerun without --skip-robot-import"
        )
    return candidates[0].resolve()


def import_robot() -> Path:
    if ARGS.robot_usd:
        result = ARGS.robot_usd.expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    if ARGS.skip_robot_import:
        return find_existing_robot_usd()

    existing = sorted(
        ROBOT_OUTPUT_DIR.glob("kuavo_s63_dexhand_rl*/kuavo_s63_dexhand_rl.usda")
    )
    if existing:
        return existing[0].resolve()

    config = MJCFImporterConfig(
        mjcf_path=str(ROBOT_XML.resolve()),
        usd_path=str(ROBOT_OUTPUT_DIR.resolve()),
        import_scene=False,
        merge_mesh=False,
        collision_from_visuals=False,
        collision_type="Convex Hull",
        allow_self_collision=False,
        robot_type="Manipulator",
        fix_base=True,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )
    result = Path(MJCFImporter(config).import_mjcf()).resolve()
    if not result.is_file():
        raise RuntimeError(f"MJCF importer did not create its reported output: {result}")
    return result


def world_mesh_data(source_stage: Usd.Stage):
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(source_stage)
    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    for prim in Usd.PrimRange.Stage(source_stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        local_points = mesh.GetPointsAttr().Get() or []
        local_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        local_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if not local_points or not local_indices:
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        base = len(points)
        for point in local_points:
            world = transform.Transform(Gf.Vec3d(point)) * meters_per_unit
            points.append(Gf.Vec3f(world))
        counts.extend(int(value) for value in local_counts)
        indices.extend(base + int(value) for value in local_indices)
    if not points:
        raise RuntimeError(f"No polygon mesh was found in {SOURCE_HANDLE_USD}")
    minimum = [min(float(point[axis]) for point in points) for axis in range(3)]
    maximum = [max(float(point[axis]) for point in points) for axis in range(3)]
    center = [(lo + hi) * 0.5 for lo, hi in zip(minimum, maximum)]
    centered = [
        Gf.Vec3f(
            float(point[0]) - center[0],
            float(point[1]) - center[1],
            float(point[2]) - center[2],
        )
        for point in points
    ]
    extents = [(hi - lo) * 0.5 for lo, hi in zip(minimum, maximum)]
    return centered, counts, indices, extents


def create_object_asset():
    if not SOURCE_HANDLE_USD.is_file():
        raise FileNotFoundError(
            f"Converted CAD USD is missing next to the original STEP: {SOURCE_HANDLE_USD}"
        )
    source = Usd.Stage.Open(str(SOURCE_HANDLE_USD))
    if source is None:
        raise RuntimeError(f"Could not open {SOURCE_HANDLE_USD}")
    points, counts, indices, half_extents = world_mesh_data(source)

    stage = Usd.Stage.CreateNew(str(OBJECT_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/OuterHandle")
    stage.SetDefaultPrim(root.GetPrim())
    root_prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr().Set(OBJECT_MASS)
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    rigid.CreateLinearDampingAttr().Set(0.05)
    rigid.CreateAngularDampingAttr().Set(0.10)
    rigid.CreateEnableCCDAttr().Set(True)
    rigid.CreateSolverPositionIterationCountAttr().Set(16)
    rigid.CreateSolverVelocityIterationCountAttr().Set(4)

    mesh = UsdGeom.Mesh.Define(stage, "/OuterHandle/Geometry")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.38, 0.46, 0.58)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    collision.CreateApproximationAttr().Set("convexDecomposition")
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    physx_collision.CreateContactOffsetAttr().Set(0.002)
    physx_collision.CreateRestOffsetAttr().Set(0.0)

    visual_material = UsdShade.Material.Define(stage, "/OuterHandle/Looks/DarkMetal")
    shader = UsdShade.Shader.Define(stage, "/OuterHandle/Looks/DarkMetal/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.30, 0.38, 0.50)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.60)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.30)
    visual_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual_material)

    physics_material = UsdShade.Material.Define(stage, "/OuterHandle/Looks/Physics")
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(0.80)
    material_api.CreateDynamicFrictionAttr().Set(0.65)
    material_api.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        physics_material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )

    root_prim.CreateAttribute(
        "stage0:sourceCad", Sdf.ValueTypeNames.String, custom=True
    ).Set(str((ROOT / "tpyrced_数模/外拉手.stp").resolve()))
    root_prim.CreateAttribute(
        "stage0:sourceConvertedUsd", Sdf.ValueTypeNames.String, custom=True
    ).Set(str(SOURCE_HANDLE_USD.resolve()))
    root_prim.CreateAttribute(
        "stage0:massStatus", Sdf.ValueTypeNames.String, custom=True
    ).Set("0.18 kg engineering estimate; replace with measured mass")
    root_prim.CreateAttribute(
        "stage0:halfExtents", Sdf.ValueTypeNames.Float3, custom=True
    ).Set(Gf.Vec3f(*half_extents))
    stage.GetRootLayer().Save()
    return half_extents


def define_box(stage, path, center, size, color, opacity=1.0):
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr().Set(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*center))
    box.AddScaleOp().Set(Gf.Vec3d(*size))
    box.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    box.CreateDisplayOpacityAttr().Set([opacity])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    return box


def create_environment_asset() -> None:
    stage = Usd.Stage.CreateNew(str(ENVIRONMENT_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Stage0Environment")
    stage.SetDefaultPrim(root.GetPrim())

    ground = UsdGeom.Plane.Define(stage, "/Stage0Environment/Ground")
    ground.CreateAxisAttr().Set("Z")
    ground.CreateDisplayColorAttr().Set([Gf.Vec3f(0.58, 0.62, 0.68)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    define_box(
        stage,
        "/Stage0Environment/Table/Top",
        TABLE_CENTER,
        TABLE_SIZE,
        (0.58, 0.38, 0.20),
    )
    leg_height = TABLE_CENTER[2] - TABLE_SIZE[2] * 0.5
    leg_dx = TABLE_SIZE[0] * 0.5 - 0.06
    leg_dy = TABLE_SIZE[1] * 0.5 - 0.06
    for index, (dx, dy) in enumerate(
        ((-leg_dx, -leg_dy), (-leg_dx, leg_dy), (leg_dx, -leg_dy), (leg_dx, leg_dy)),
        start=1,
    ):
        define_box(
            stage,
            f"/Stage0Environment/Table/Leg{index}",
            (TABLE_CENTER[0] + dx, TABLE_CENTER[1] + dy, leg_height * 0.5),
            (0.06, 0.06, leg_height),
            (0.38, 0.27, 0.18),
        )

    support = define_box(
        stage,
        "/Stage0Environment/TemporarySupport",
        SUPPORT_CENTER,
        SUPPORT_SIZE,
        (0.20, 0.58, 0.86),
        opacity=0.48,
    )
    support.GetPrim().CreateAttribute(
        "stage0:replacement", Sdf.ValueTypeNames.String, custom=True
    ).Set("Remove when the real 10 mm bin bottom is added")
    root.GetPrim().CreateAttribute(
        "stage0:tableTop", Sdf.ValueTypeNames.Double, custom=True
    ).Set(TABLE_TOP_Z)
    root.GetPrim().CreateAttribute(
        "stage0:supportTop", Sdf.ValueTypeNames.Double, custom=True
    ).Set(SUPPORT_TOP_Z)
    stage.GetRootLayer().Save()


def relative_asset_path(asset: Path, layer: Path) -> str:
    return Path(os.path.relpath(asset, layer.parent)).as_posix()


def create_scene(robot_usd: Path, half_extents) -> None:
    stage = Usd.Stage.CreateNew(str(SCENE_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # Keep headless validation and later RL rollouts from reaching the USD
    # timeline end after only a short settling interval.
    stage.SetTimeCodesPerSecond(120.0)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(1_000_000.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    environment = UsdGeom.Xform.Define(stage, "/World/Environment")
    environment.GetPrim().GetReferences().AddReference(
        relative_asset_path(ENVIRONMENT_USD, SCENE_USD)
    )

    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.GetPrim().GetReferences().AddReference(relative_asset_path(robot_usd, SCENE_USD))
    robot.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_POSITION))
    arm_base_reference = UsdGeom.Xform.Define(stage, "/World/Robot/ArmBaseReference")
    arm_base_reference.AddTranslateOp().Set(Gf.Vec3d(*ARM_BASE_POSITION))
    arm_base_reference.GetPrim().CreateAttribute(
        "stage0:description", Sdf.ValueTypeNames.String, custom=True
    ).Set("Fixed parent frame of zarm_r1_joint; extraction anchor for the later arm-only asset")

    spawn_z = OBJECT_SPAWN_Z
    handle = UsdGeom.Xform.Define(stage, "/World/OuterHandle")
    handle.GetPrim().GetReferences().AddReference(relative_asset_path(OBJECT_USD, SCENE_USD))
    handle.AddTranslateOp().Set(Gf.Vec3d(OBJECT_XY[0], OBJECT_XY[1], spawn_z))
    handle.AddOrientOp().Set(
        Gf.Quatf(
            OBJECT_SPAWN_ORIENTATION_WXYZ[0],
            Gf.Vec3f(*OBJECT_SPAWN_ORIENTATION_WXYZ[1:]),
        )
    )

    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    physx_scene.CreateEnableCCDAttr().Set(True)
    physx_scene.CreateTimeStepsPerSecondAttr().Set(120)

    dome = UsdLux.DomeLight.Define(stage, "/World/EnvironmentLight")
    dome.CreateColorAttr().Set(Gf.Vec3f(0.72, 0.78, 0.90))
    dome.CreateIntensityAttr().Set(850.0)
    dome.CreateExposureAttr().Set(0.5)
    dome.GetPrim().CreateAttribute(
        "visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)
    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.93, 0.82))
    light.CreateIntensityAttr().Set(2200.0)
    light.CreateAngleAttr().Set(2.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 28.0, 24.0))

    target_q = dict(zip(ACTIVE_JOINTS, ARM_Q + HAND_Q))
    gains = dict(zip(ACTIVE_JOINTS, zip(ARM_KP + HAND_KP, ARM_KD + HAND_KD)))
    driven = set()
    all_revolute = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        all_revolute.append(name)
        if name not in target_q:
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(gains[name][0])
        drive.CreateDampingAttr().Set(gains[name][1])
        drive.CreateTargetPositionAttr().Set(math.degrees(target_q[name]))
        state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
        state.CreatePositionAttr().Set(math.degrees(target_q[name]))
        state.CreateVelocityAttr().Set(0.0)
        driven.add(name)

    if set(all_revolute) != set(ACTIVE_JOINTS) or driven != set(ACTIVE_JOINTS):
        raise RuntimeError(
            f"Imported robot joint mismatch: revolute={sorted(all_revolute)}, driven={sorted(driven)}"
        )

    robot_prim = robot.GetPrim()
    robot_prim.CreateAttribute(
        "stage0:activeJointNames", Sdf.ValueTypeNames.StringArray, custom=True
    ).Set(list(ACTIVE_JOINTS))
    robot_prim.CreateAttribute(
        "stage0:armBaseReference", Sdf.ValueTypeNames.String, custom=True
    ).Set("/World/Robot/ArmBaseReference (fixed parent frame of zarm_r1_joint)")
    robot_prim.CreateAttribute(
        "stage0:sourceFixedRobot", Sdf.ValueTypeNames.String, custom=True
    ).Set(str(SOURCE_ROBOT.resolve()))

    world_prim = world.GetPrim()
    world_prim.CreateAttribute(
        "stage0:scope", Sdf.ValueTypeNames.String, custom=True
    ).Set("asset/geometry/physics validation only; no PPO, reward, observations, cameras, or bin")
    stage.GetRootLayer().Save()

    nominal = {
        "schema_version": 1,
        "units": {"length": "m", "angle": "rad", "quaternion": "wxyz"},
        "active_joint_names": list(ACTIVE_JOINTS),
        "arm_base_reference": "/World/Robot/ArmBaseReference (fixed parent frame of zarm_r1_joint)",
        "robot_root_world_pose": {
            "position": list(ROBOT_POSITION),
            "orientation": [1.0, 0.0, 0.0, 0.0],
        },
        "right_arm_base_world_pose": {
            "position": list(ARM_BASE_POSITION),
            "orientation": [1.0, 0.0, 0.0, 0.0],
        },
        "table": {
            "center_position": list(TABLE_CENTER),
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "size": list(TABLE_SIZE),
            "top_z": TABLE_TOP_Z,
        },
        "temporary_support": {
            "center_position": list(SUPPORT_CENTER),
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "size": list(SUPPORT_SIZE),
            "top_z": SUPPORT_TOP_Z,
        },
        "object_spawn_world_pose": {
            "position": [OBJECT_XY[0], OBJECT_XY[1], spawn_z],
            "orientation": list(OBJECT_SPAWN_ORIENTATION_WXYZ),
            "free_fall_gap": DROP_HEIGHT,
        },
        "object_nominal_settled_world_pose": {
            "position": list(OBJECT_GRASP_POSITION),
            "orientation": list(OBJECT_GRASP_ORIENTATION_WXYZ),
            "note": "Natural settled orientation recentered on the front table; this is the grasp-training baseline.",
        },
        "arm_base_to_object_nominal": {
            "position": [
                OBJECT_GRASP_POSITION[0] - ARM_BASE_POSITION[0],
                OBJECT_GRASP_POSITION[1] - ARM_BASE_POSITION[1],
                OBJECT_GRASP_POSITION[2] - ARM_BASE_POSITION[2],
            ],
            "orientation": list(OBJECT_GRASP_ORIENTATION_WXYZ),
        },
        "right_palm_initial_world_pose": {
            "position": list(PALM_POSITION),
            "orientation": list(PALM_ORIENTATION_WXYZ),
            "approach_axis_toward_object": list(PALM_APPROACH_AXIS),
        },
        "arm_base_to_right_palm": {
            "position": [
                PALM_POSITION[0] - ARM_BASE_POSITION[0],
                PALM_POSITION[1] - ARM_BASE_POSITION[1],
                PALM_POSITION[2] - ARM_BASE_POSITION[2],
            ],
            "orientation": list(PALM_ORIENTATION_WXYZ),
        },
        "right_arm_initial_joint_positions": dict(zip(ARM_JOINTS, ARM_Q)),
        "right_hand_initial_joint_positions": dict(zip(HAND_JOINTS, HAND_Q)),
        "right_hand_joint_limits_rad": {
            name: list(HAND_JOINT_LIMITS[name]) for name in HAND_JOINTS
        },
        "object": {
            "half_extents": [float(value) for value in half_extents],
            "mass": OBJECT_MASS,
            "collision": "convexDecomposition",
            "mass_status": "engineering estimate; measure before sim-to-real",
        },
        "control": {
            "arm_gain_source": "src/kuavo_assets/config/kuavo_v63/kuavo.json ruiwo_kp/ruiwo_kd",
            "finger_gain_status": "Isaac stage-0 smoke-test gains; S49 MJCF has position actuators but no servo kp",
        },
    }
    NOMINAL_JSON.write_text(json.dumps(nominal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ensure_directories()
    create_robot_mjcf()
    robot_usd = import_robot()
    half_extents = create_object_asset()
    create_environment_asset()
    create_scene(robot_usd, half_extents)
    print(f"SOURCE_ROBOT={SOURCE_ROBOT}")
    print(f"DERIVED_MJCF={ROBOT_XML}")
    print(f"ROBOT_USD={robot_usd}")
    print(f"OBJECT_USD={OBJECT_USD}")
    print(f"ENVIRONMENT_USD={ENVIRONMENT_USD}")
    print(f"SCENE_USD={SCENE_USD}")
    print(f"NOMINAL_JSON={NOMINAL_JSON}")
    print(f"ACTIVE_DOF_COUNT={len(ACTIVE_JOINTS)}")
    print("ACTIVE_DOFS=" + ",".join(ACTIVE_JOINTS))
    print("BUILD_RESULT=PASS")
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
