"""Allowed support-face orientations for table-box placement."""

from __future__ import annotations

import math
from typing import Iterable


WORLD_DOWN = (0.0, 0.0, -1.0)
SUPPORT_NORMALS_LOCAL = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def _mat_vec(rotation: tuple[tuple[float, ...], ...], vector: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        sum(rotation[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _mat_mul(
    left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def rotation_z(angle_rad: float) -> tuple[tuple[float, ...], ...]:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def base_rotation_for_support_face(face: str) -> tuple[tuple[float, ...], ...]:
    """Return ``R_world_object`` with the named local face normal downward."""

    try:
        SUPPORT_NORMALS_LOCAL[face]
    except KeyError as exc:
        raise ValueError(f"Unknown support face: {face!r}") from exc

    rotations = {
        # local -Z is already world-down; local +Z is flipped over X.
        "-z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "+z": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        # Ry(+/-90) maps +/-X to world -Z.
        "+x": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        "-x": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        # Rx(-/+90) maps +/-Y to world -Z.
        "+y": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        "-y": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    }
    return rotations[face]


def rotation_for_support_face(face: str, yaw_rad: float) -> tuple[tuple[float, ...], ...]:
    """Return a world rotation for a support face plus in-plane world yaw."""

    return _mat_mul(rotation_z(yaw_rad), base_rotation_for_support_face(face))


def transformed_support_normal(face: str, yaw_rad: float = 0.0) -> tuple[float, ...]:
    """Return the selected local support normal expressed in world coordinates."""

    rotation = rotation_for_support_face(face, yaw_rad)
    return _mat_vec(rotation, SUPPORT_NORMALS_LOCAL[face])


def support_face_is_down(face: str, yaw_rad: float = 0.0, tolerance_deg: float = 15.0) -> bool:
    normal = transformed_support_normal(face, yaw_rad)
    cosine = sum(normal[index] * WORLD_DOWN[index] for index in range(3))
    return cosine >= math.cos(math.radians(float(tolerance_deg)))


__all__ = [
    "SUPPORT_NORMALS_LOCAL",
    "WORLD_DOWN",
    "base_rotation_for_support_face",
    "rotation_for_support_face",
    "support_face_is_down",
    "transformed_support_normal",
]
