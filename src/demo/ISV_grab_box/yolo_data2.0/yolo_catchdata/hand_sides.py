"""Static paths and joint conventions for the two independent wrist branches."""

from __future__ import annotations

from typing import Any


HAND_SIDES = ("right", "left")


def _finger(
    path: str,
    axis: tuple[float, float, float],
    maximum_deg: float,
) -> dict[str, Any]:
    return {
        "path": path,
        "axis": axis,
        "min_deg": 0.0,
        "max_deg": float(maximum_deg),
    }


RIGHT_ROOT = "/World/ManualInspection/RightWrist"
LEFT_ROOT = "/World/ManualInspection/LeftWrist"

HAND_SPECS: dict[str, dict[str, Any]] = {
    "right": {
        "manual_root": RIGHT_ROOT,
        "palm": RIGHT_ROOT + "/r_palm",
        "camera_link": RIGHT_ROOT + "/r_hand_tripod/r_hand_camera_link",
        "rgb_camera": (
            RIGHT_ROOT
            + "/r_hand_tripod/r_hand_camera_link/right_wrist_d405_rgb"
        ),
        "camera_name": "right_wrist_d405",
        # Fixed transforms from the corrected S63+D-hand source model. The
        # RGB camera origin coincides with *_hand_camera_link, while its local
        # rotation converts the MuJoCo camera axes to the USD camera axes.
        "r7_from_palm": {
            "translation_m": (0.0, 0.0, -0.07),
            "quaternion_wxyz": (0.707388, -0.706825, 0.0, 0.0),
        },
        "r7_from_camera_link": {
            "translation_m": (
                0.115405591590931,
                0.0154312352120435,
                -0.107724128430896,
            ),
            "quaternion_wxyz": (
                0.303629589,
                -0.232925447,
                0.562521384,
                0.732887782,
            ),
        },
        "camera_link_from_rgb": {
            "translation_m": (0.0, 0.0, 0.0),
            "quaternion_wxyz": (0.5, 0.5, -0.5, -0.5),
        },
        "finger_joints": {
            "THUMB_CMC": _finger(RIGHT_ROOT + "/r_thumb_prox", (0.0, -1.0, 0.0), 90.0),
            "THUMB_MCP": _finger(
                RIGHT_ROOT + "/r_thumb_prox/r_thumb_dist", (0.0, 0.0, 1.0), 50.0
            ),
            "INDEX_MCP": _finger(RIGHT_ROOT + "/r_index_prox", (1.0, 0.0, 0.0), 75.0),
            "INDEX_PIP": _finger(
                RIGHT_ROOT + "/r_index_prox/r_index_dist", (1.0, 0.0, 0.0), 120.0
            ),
            "MIDDLE_MCP": _finger(RIGHT_ROOT + "/r_middle_prox", (1.0, 0.0, 0.0), 75.0),
            "MIDDLE_PIP": _finger(
                RIGHT_ROOT + "/r_middle_prox/r_middle_dist", (1.0, 0.0, 0.0), 120.0
            ),
            "RING_MCP": _finger(RIGHT_ROOT + "/r_ring_prox", (1.0, 0.0, 0.0), 75.0),
            "RING_PIP": _finger(
                RIGHT_ROOT + "/r_ring_prox/r_ring_dist", (1.0, 0.0, 0.0), 120.0
            ),
            "LITTLE_MCP": _finger(RIGHT_ROOT + "/r_little_prox", (1.0, 0.0, 0.0), 75.0),
            "LITTLE_PIP": _finger(
                RIGHT_ROOT + "/r_little_prox/r_little_dist", (1.0, 0.0, 0.0), 120.0
            ),
        },
    },
    "left": {
        "manual_root": LEFT_ROOT,
        "palm": LEFT_ROOT + "/l_palm",
        "camera_link": LEFT_ROOT + "/l_hand_tripod/l_hand_camera_link",
        "rgb_camera": (
            LEFT_ROOT
            + "/l_hand_tripod/l_hand_camera_link/left_wrist_d405_rgb"
        ),
        "camera_name": "left_wrist_d405",
        "r7_from_palm": {
            "translation_m": (0.0, 0.0, -0.07),
            "quaternion_wxyz": (
                0.000563312,
                -0.000562864,
                0.706825,
                -0.707388,
            ),
        },
        "r7_from_camera_link": {
            "translation_m": (
                0.11540559159102,
                -0.0145586110669961,
                -0.110123491595499,
            ),
            "quaternion_wxyz": (
                -0.234132096,
                0.302280981,
                -0.730301541,
                -0.566098957,
            ),
        },
        "camera_link_from_rgb": {
            "translation_m": (0.0, 0.0, 0.0),
            "quaternion_wxyz": (0.5, 0.5, -0.5, -0.5),
        },
        # The thumb axes are mirrored in biped_s63_Dhand_fixed.xml.  The four
        # long fingers keep the same local +X flexion axes as the right hand.
        "finger_joints": {
            "THUMB_CMC": _finger(LEFT_ROOT + "/l_thumb_prox", (0.0, 1.0, 0.0), 90.0),
            "THUMB_MCP": _finger(
                LEFT_ROOT + "/l_thumb_prox/l_thumb_dist", (-1.0, 0.0, 0.0), 50.0
            ),
            "INDEX_MCP": _finger(LEFT_ROOT + "/l_Index_prox", (1.0, 0.0, 0.0), 75.0),
            "INDEX_PIP": _finger(
                LEFT_ROOT + "/l_Index_prox/l_index_dist", (1.0, 0.0, 0.0), 120.0
            ),
            "MIDDLE_MCP": _finger(LEFT_ROOT + "/l_middle_prox", (1.0, 0.0, 0.0), 75.0),
            "MIDDLE_PIP": _finger(
                LEFT_ROOT + "/l_middle_prox/l_middle_dist", (1.0, 0.0, 0.0), 120.0
            ),
            "RING_MCP": _finger(LEFT_ROOT + "/l_ring_prox", (1.0, 0.0, 0.0), 75.0),
            "RING_PIP": _finger(
                LEFT_ROOT + "/l_ring_prox/l_ring_dist", (1.0, 0.0, 0.0), 120.0
            ),
            "LITTLE_MCP": _finger(LEFT_ROOT + "/l_little_prox", (1.0, 0.0, 0.0), 75.0),
            "LITTLE_PIP": _finger(
                LEFT_ROOT + "/l_little_prox/l_little_dist", (1.0, 0.0, 0.0), 120.0
            ),
        },
    },
}


def hand_spec(hand_side: str) -> dict[str, Any]:
    """Return the immutable-by-convention specification for one wrist branch."""

    try:
        return HAND_SPECS[str(hand_side)]
    except KeyError as exc:
        raise ValueError(f"Unsupported hand side: {hand_side!r}") from exc


def reference_filename(hand_side: str, class_name: str) -> str:
    hand_spec(hand_side)
    return f"{hand_side}_wrist_{class_name}_approach_reference.json"


__all__ = ["HAND_SIDES", "HAND_SPECS", "hand_spec", "reference_filename"]
