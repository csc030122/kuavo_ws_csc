"""Coordinate-frame helpers independent of Isaac Sim."""

from __future__ import annotations

import math
from typing import Iterable

def as_vector(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def normalize(vector: Iterable[float]) -> tuple[float, ...]:
    result = as_vector(vector)
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(value / norm for value in result)


def sampling_rotation_world_from_sampling(
    table_center_world: Iterable[float],
    robot_base_world: Iterable[float],
) -> tuple[tuple[float, ...], ...]:
    """Return R_W_S with +X_S toward the robot and +Z_S vertical."""

    table = as_vector(table_center_world)
    robot = as_vector(robot_base_world)
    x_axis = normalize((robot[0] - table[0], robot[1] - table[1], 0.0))
    z_axis = (0.0, 0.0, 1.0)
    y_axis = normalize(
        (
            z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
            z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
            z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
        )
    )
    rotation = tuple(
        tuple(axis[row] for axis in (x_axis, y_axis, z_axis)) for row in range(3)
    )
    validate_rotation(rotation)
    return rotation


def validate_rotation(rotation: Iterable[Iterable[float]], atol: float = 1e-8) -> None:
    matrix = tuple(as_vector(row) for row in rotation)
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Rotation must be 3x3")
    columns = tuple(tuple(matrix[row][column] for row in range(3)) for column in range(3))
    for first in range(3):
        for second in range(3):
            dot = sum(columns[first][index] * columns[second][index] for index in range(3))
            expected = 1.0 if first == second else 0.0
            if not math.isclose(dot, expected, abs_tol=atol):
                raise ValueError("Rotation columns are not orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=atol):
        raise ValueError(f"Rotation is not right-handed: det={determinant}")


def validate_transform(transform: Iterable[Iterable[float]], atol: float = 1e-8) -> None:
    matrix = tuple(as_vector(row) for row in transform)
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Transform must be 4x4")
    validate_rotation(
        tuple(tuple(row[column] for column in range(3)) for row in matrix[:3]),
        atol=atol,
    )
    if any(
        not math.isclose(actual, expected, abs_tol=atol)
        for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0))
    ):
        raise ValueError("Transform bottom row must be [0, 0, 0, 1]")


def point_inside_aabb(
    point: Iterable[float],
    minimum: Iterable[float],
    maximum: Iterable[float],
    tolerance: float = 0.0,
) -> bool:
    p = as_vector(point)
    lo = as_vector(minimum)
    hi = as_vector(maximum)
    return all(
        minimum_value - tolerance <= value <= maximum_value + tolerance
        for value, minimum_value, maximum_value in zip(p, lo, hi)
    )


def footprint_fits(
    asset_size_xy: Iterable[float],
    container_size_xy: Iterable[float],
    clearance: float,
) -> bool:
    asset = sorted(float(value) for value in asset_size_xy)
    container = sorted(float(value) - 2.0 * clearance for value in container_size_xy)
    return asset[0] <= container[0] and asset[1] <= container[1]
