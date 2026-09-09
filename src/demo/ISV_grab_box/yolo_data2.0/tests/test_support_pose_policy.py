from __future__ import annotations

import unittest

from yolo_catchdata.support_pose_policy import (
    SUPPORT_NORMALS_LOCAL,
    support_face_is_down,
    transformed_support_normal,
)


class SupportPosePolicyTest(unittest.TestCase):
    def test_all_declared_faces_map_to_world_down(self) -> None:
        for face in SUPPORT_NORMALS_LOCAL:
            normal = transformed_support_normal(face)
            self.assertAlmostEqual(normal[0], 0.0, places=12)
            self.assertAlmostEqual(normal[1], 0.0, places=12)
            self.assertAlmostEqual(normal[2], -1.0, places=12)
            self.assertTrue(support_face_is_down(face))

    def test_in_plane_yaw_preserves_support_normal(self) -> None:
        for face in SUPPORT_NORMALS_LOCAL:
            normal = transformed_support_normal(face, 1.234)
            self.assertAlmostEqual(normal[0], 0.0, places=12)
            self.assertAlmostEqual(normal[1], 0.0, places=12)
            self.assertAlmostEqual(normal[2], -1.0, places=12)

    def test_unknown_face_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            transformed_support_normal("+q")


if __name__ == "__main__":
    unittest.main()
