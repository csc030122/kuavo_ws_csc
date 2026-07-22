"""Static lab scene with boxes for inspection."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer


SCENE = (
    Path(__file__).resolve().parent
    / "biped_s63/xml/scenes/lab_sence_static_with_box.xml"
)

TARGET_BOX = "lab_target_box"
ARM_JOINTS = (
    [f"zarm_l{i}_joint" for i in range(1, 8)]
    + [f"zarm_r{i}_joint" for i in range(1, 8)]
)


def reset_arm_joints_to_zero(model, data):
    for joint_name in ARM_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            print(f"skip missing joint: {joint_name}")
            continue

        qpos_id = model.jnt_qposadr[joint_id]
        qvel_id = model.jnt_dofadr[joint_id]
        data.qpos[qpos_id] = 0.0
        data.qvel[qvel_id] = 0.0

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{joint_name}_motor",
        )
        if actuator_id >= 0:
            data.ctrl[actuator_id] = 0.0


def main():
    print("scene =", SCENE)
    print("scene exists =", SCENE.exists())

    if not SCENE.exists():
        raise FileNotFoundError(SCENE)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    reset_arm_joints_to_zero(model, data)
    mujoco.mj_forward(model, data)

    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BOX)
    if box_body_id >= 0:
        print(f"target box pos = {data.xpos[box_body_id]}")
    else:
        print(f"target box not found: {TARGET_BOX}")

    print(f"model loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}, ncam={model.ncam}")
    print("opening viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.8, 0.8, 0.8]
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -35

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()