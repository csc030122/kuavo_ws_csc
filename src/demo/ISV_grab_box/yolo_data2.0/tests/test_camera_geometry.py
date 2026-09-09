from __future__ import annotations

import math
import unittest

from yolo_catchdata.camera_geometry import (
    camera_from_object,
    intrinsics_from_fov,
    look_at_world_from_usd_camera,
    make_transform,
    project_point,
)


IDENTITY_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class CameraGeometryTest(unittest.TestCase):
    def test_front_one_meter_is_positive_opencv_z(self) -> None:
        world_from_camera = make_transform(IDENTITY_ROTATION, (0.0, 0.0, 0.0))
        world_from_object = make_transform(IDENTITY_ROTATION, (0.0, 0.0, -1.0))
        camera_from_target = camera_from_object(world_from_camera, world_from_object)
        self.assertEqual(
            tuple(camera_from_target[index][3] for index in range(3)),
            (0.0, 0.0, 1.0),
        )

    def test_image_right_has_positive_x(self) -> None:
        world_from_camera = make_transform(IDENTITY_ROTATION, (0.0, 0.0, 0.0))
        world_from_object = make_transform(IDENTITY_ROTATION, (0.1, 0.0, -1.0))
        camera_from_target = camera_from_object(world_from_camera, world_from_object)
        self.assertGreater(camera_from_target[0][3], 0.0)

    def test_image_down_has_positive_y(self) -> None:
        world_from_camera = make_transform(IDENTITY_ROTATION, (0.0, 0.0, 0.0))
        world_from_object = make_transform(IDENTITY_ROTATION, (0.0, -0.1, -1.0))
        camera_from_target = camera_from_object(world_from_camera, world_from_object)
        self.assertGreater(camera_from_target[1][3], 0.0)

    def test_look_at_places_target_on_optical_axis(self) -> None:
        camera_position = (1.0, 2.0, 3.0)
        target_position = (1.2, 1.6, 2.7)
        world_from_camera = look_at_world_from_usd_camera(camera_position, target_position)
        world_from_object = make_transform(IDENTITY_ROTATION, target_position)
        camera_from_target = camera_from_object(world_from_camera, world_from_object)
        translation = tuple(camera_from_target[index][3] for index in range(3))
        expected_distance = math.sqrt(
            sum((target_position[index] - camera_position[index]) ** 2 for index in range(3))
        )
        self.assertAlmostEqual(translation[0], 0.0, places=12)
        self.assertAlmostEqual(translation[1], 0.0, places=12)
        self.assertAlmostEqual(translation[2], expected_distance, places=12)

    def test_intrinsics_and_projection(self) -> None:
        intrinsics = intrinsics_from_fov(1280, 720, 84.0, 58.0)
        center = project_point(intrinsics, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(center[0], 639.5)
        self.assertAlmostEqual(center[1], 359.5)
        self.assertGreater(float(intrinsics["fx"]), 0.0)
        self.assertGreater(float(intrinsics["fy"]), 0.0)


if __name__ == "__main__":
    unittest.main()
