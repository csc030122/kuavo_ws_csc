from __future__ import annotations

import unittest

from yolo_catchdata.frame_utils import (
    footprint_fits,
    sampling_rotation_world_from_sampling,
    validate_rotation,
    validate_transform,
)


class FrameUtilsTest(unittest.TestCase):
    def test_sampling_frame_matches_frozen_scene(self) -> None:
        rotation = sampling_rotation_world_from_sampling(
            [1.6, -0.26, 1.0], [1.6, 0.30, 0.015]
        )
        expected = (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        self.assertEqual(rotation, expected)
        validate_rotation(rotation)

    def test_invalid_left_handed_rotation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_rotation(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))

    def test_transform_shape_and_bottom_row(self) -> None:
        identity = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        validate_transform(identity)
        invalid = identity[:3] + ((1.0, 0.0, 0.0, 1.0),)
        with self.assertRaises(ValueError):
            validate_transform(invalid)

    def test_footprint_can_rotate_in_table(self) -> None:
        self.assertTrue(footprint_fits([0.426, 0.220], [0.535, 0.335], 0.01))
        self.assertFalse(footprint_fits([0.60, 0.20], [0.535, 0.335], 0.01))


if __name__ == "__main__":
    unittest.main()
