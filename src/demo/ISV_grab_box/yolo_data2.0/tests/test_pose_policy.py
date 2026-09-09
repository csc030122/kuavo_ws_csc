from __future__ import annotations

import math
import unittest

from yolo_catchdata.pose_policy import (
    effective_hand_yaw_delta,
    settled_effective_hand_yaw_delta,
    transport_reference_pose,
    wrap_to_half_turn,
)


def pose(x: float, y: float, z: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def pose_yaw(x: float, y: float, z: float, yaw: float) -> tuple[tuple[float, ...], ...]:
    c, s = math.cos(yaw), math.sin(yaw)
    return (
        (c, -s, 0.0, x),
        (s, c, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


class PosePolicyTest(unittest.TestCase):
    def test_half_turn_symmetry_maps_180_degrees_to_no_hand_turn(self) -> None:
        self.assertAlmostEqual(wrap_to_half_turn(math.pi), 0.0)
        self.assertAlmostEqual(
            effective_hand_yaw_delta("left", math.pi), 0.0
        )
        self.assertAlmostEqual(
            effective_hand_yaw_delta("right", -math.pi), 0.0
        )
        self.assertAlmostEqual(
            effective_hand_yaw_delta("hose", 1.5 * math.pi), -0.5 * math.pi
        )

    def test_outhandle_keeps_reference_hand_orientation_for_any_object_yaw(self) -> None:
        self.assertAlmostEqual(
            effective_hand_yaw_delta("outhandle", math.radians(137.0)), 0.0
        )

    def test_elongated_180_degree_turn_keeps_palm_pose_relative_to_center(self) -> None:
        reference_object = pose(1.0, 2.0, 0.5)
        reference_palm = pose(1.1, 2.2, 0.8)
        new_object = pose(3.0, 4.0, 0.5)
        actual = transport_reference_pose(
            reference_object,
            reference_palm,
            new_object,
            "left",
            math.pi,
        )
        self.assertAlmostEqual(actual[0][3], 3.1)
        self.assertAlmostEqual(actual[1][3], 4.2)
        self.assertAlmostEqual(actual[2][3], 0.8)
        self.assertAlmostEqual(actual[0][0], 1.0)
        self.assertAlmostEqual(actual[1][1], 1.0)

    def test_elongated_90_degree_turn_rotates_reference_offset_and_palm(self) -> None:
        reference_object = pose(0.0, 0.0, 0.0)
        reference_palm = pose(1.0, 0.0, 1.0)
        new_object = pose(2.0, 3.0, 0.0)
        actual = transport_reference_pose(
            reference_object,
            reference_palm,
            new_object,
            "hose",
            0.5 * math.pi,
        )
        # At a +90-degree tie the policy keeps the intuitive +90 representative.
        self.assertAlmostEqual(actual[0][3], 2.0)
        self.assertAlmostEqual(actual[1][3], 4.0)
        self.assertAlmostEqual(actual[2][3], 1.0)
        self.assertAlmostEqual(actual[0][0], 0.0, places=12)
        self.assertAlmostEqual(actual[0][1], -1.0)
        self.assertAlmostEqual(actual[1][0], 1.0)

    def test_outhandle_translation_follows_object_but_offset_does_not_rotate(self) -> None:
        actual = transport_reference_pose(
            pose(1.0, 2.0, 0.0),
            pose(1.1, 2.2, 0.3),
            pose(4.0, 5.0, 0.0),
            "outhandle",
            math.radians(75.0),
        )
        self.assertAlmostEqual(actual[0][3], 4.1)
        self.assertAlmostEqual(actual[1][3], 5.2)
        self.assertAlmostEqual(actual[2][3], 0.3)
        self.assertAlmostEqual(actual[0][0], 1.0)
        self.assertAlmostEqual(actual[1][1], 1.0)

    def test_settled_heading_uses_actual_rotation_instead_of_requested_yaw(self) -> None:
        delta = settled_effective_hand_yaw_delta(
            pose_yaw(0.0, 0.0, 0.0, math.radians(10.0)),
            pose_yaw(1.0, 2.0, 0.0, math.radians(42.0)),
            "hose",
            math.radians(5.0),
        )
        self.assertAlmostEqual(delta, math.radians(32.0))

    def test_transport_can_follow_settled_heading(self) -> None:
        actual = transport_reference_pose(
            pose(0.0, 0.0, 0.0),
            pose(1.0, 0.0, 1.0),
            pose_yaw(2.0, 3.0, 0.0, 0.25 * math.pi),
            "hose",
            0.0,
            use_settled_heading=True,
        )
        root_half = math.sqrt(0.5)
        self.assertAlmostEqual(actual[0][3], 2.0 + root_half)
        self.assertAlmostEqual(actual[1][3], 3.0 + root_half)
        self.assertAlmostEqual(actual[0][0], root_half)
        self.assertAlmostEqual(actual[1][0], root_half)


if __name__ == "__main__":
    unittest.main()
