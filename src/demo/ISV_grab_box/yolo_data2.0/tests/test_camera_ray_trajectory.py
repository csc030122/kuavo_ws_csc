from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # system Python used by lightweight unit tests
    np = None

from scripts.render_approach_hand_smoke import (
    _camera_ray_approach,
    _lift_palm_for_box_wall_clearance,
)


def pose(x: float, y: float, z: float):
    if np is None:  # pragma: no cover - guarded by class skip
        raise RuntimeError("NumPy unavailable")
    result = np.eye(4, dtype=float)
    result[:3, 3] = (x, y, z)
    return result


@unittest.skipUnless(np is not None, "NumPy is provided by the Isaac runtime")
class CameraRayTrajectoryTest(unittest.TestCase):
    def test_whole_palm_retreat_preserves_camera_object_ray(self) -> None:
        reference_palm = pose(0.0, 0.0, 0.0)
        reference_camera = pose(1.0, 0.0, 0.0)
        grasp = pose(2.0, 3.0, 0.0)
        settled_object = pose(2.0, 3.0, 0.0)

        result, direction = _camera_ray_approach(
            grasp,
            settled_object,
            reference_palm,
            reference_camera,
            0.25,
        )

        np.testing.assert_allclose(direction, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(result[:3, 3], [2.25, 3.0, 0.0])
        # The palm/camera orientation is inherited unchanged; this is not a
        # synthetic camera look-at operation.
        np.testing.assert_allclose(result[:3, :3], grasp[:3, :3])

    def test_zero_length_camera_object_ray_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _camera_ray_approach(
                pose(0.0, 0.0, 0.0),
                pose(1.0, 0.0, 0.0),
                pose(0.0, 0.0, 0.0),
                pose(1.0, 0.0, 0.0),
                0.10,
            )

    def test_camera_outside_box_is_lifted_until_sight_line_clears_wall(self) -> None:
        palm, details = _lift_palm_for_box_wall_clearance(
            desired_world_from_palm=pose(2.0, 0.0, 1.0),
            settled_world_from_object=pose(0.0, 0.0, 0.0),
            palm_from_camera=pose(0.0, 0.0, 0.0),
            inner_min_xy=(-1.0, -1.0),
            inner_max_xy=(1.0, 1.0),
            wall_top_z_m=1.0,
            clearance_m=0.1,
            maximum_vertical_lift_m=2.0,
        )
        self.assertTrue(details["applied"])
        self.assertAlmostEqual(details["inner_wall_exit_fraction"], 0.5)
        self.assertAlmostEqual(details["vertical_lift_m"], 1.2)
        self.assertAlmostEqual(details["wall_crossing_z_after_m"], 1.1)
        np.testing.assert_allclose(palm[:3, 3], [2.0, 0.0, 2.2])

    def test_camera_inside_box_needs_no_wall_lift(self) -> None:
        original = pose(0.5, 0.5, 0.4)
        palm, details = _lift_palm_for_box_wall_clearance(
            desired_world_from_palm=original,
            settled_world_from_object=pose(0.0, 0.0, 0.0),
            palm_from_camera=pose(0.0, 0.0, 0.0),
            inner_min_xy=(-1.0, -1.0),
            inner_max_xy=(1.0, 1.0),
            wall_top_z_m=1.0,
            clearance_m=0.1,
            maximum_vertical_lift_m=2.0,
        )
        self.assertFalse(details["applied"])
        self.assertEqual(details["reason"], "camera_inside_inner_box_xy")
        np.testing.assert_allclose(palm, original)


if __name__ == "__main__":
    unittest.main()
