"""相机内参与 USD/OpenCV 坐标系转换的纯 Python 几何函数。"""

from __future__ import annotations

import math
from typing import Iterable


Matrix = tuple[tuple[float, ...], ...]
Vector = tuple[float, ...]


def _vector(values: Iterable[float]) -> Vector:
    return tuple(float(value) for value in values)


def _cross(first: Vector, second: Vector) -> Vector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _normalize(values: Iterable[float]) -> Vector:
    vector = _vector(values)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("不能归一化零向量")
    return tuple(value / norm for value in vector)


def matrix_multiply(first: Iterable[Iterable[float]], second: Iterable[Iterable[float]]) -> Matrix:
    """矩阵相乘，使用列向量约定。"""

    left = tuple(_vector(row) for row in first)
    right = tuple(_vector(row) for row in second)
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("矩阵形状不匹配")
    if any(len(row) != len(left[0]) for row in left) or any(
        len(row) != len(right[0]) for row in right
    ):
        raise ValueError("矩阵必须是规则二维数组")
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def rigid_inverse(transform: Iterable[Iterable[float]]) -> Matrix:
    """求刚体齐次变换的逆。"""

    matrix = tuple(_vector(row) for row in transform)
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("刚体变换必须是 4×4")
    rotation_transpose = tuple(
        tuple(matrix[column][row] for column in range(3)) for row in range(3)
    )
    translation = tuple(matrix[row][3] for row in range(3))
    inverse_translation = tuple(
        -sum(rotation_transpose[row][column] * translation[column] for column in range(3))
        for row in range(3)
    )
    return tuple(
        rotation_transpose[row] + (inverse_translation[row],) for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def transform_point(transform: Iterable[Iterable[float]], point: Iterable[float]) -> Vector:
    """用 4×4 齐次矩阵变换三维点。"""

    matrix = tuple(_vector(row) for row in transform)
    xyz = _vector(point)
    if len(xyz) != 3:
        raise ValueError("三维点必须包含 3 个数")
    homogeneous = xyz + (1.0,)
    result = tuple(
        sum(matrix[row][column] * homogeneous[column] for column in range(4))
        for row in range(4)
    )
    if abs(result[3]) <= 1e-12:
        raise ValueError("齐次坐标 w 为 0")
    return tuple(result[index] / result[3] for index in range(3))


def make_transform(rotation: Iterable[Iterable[float]], translation: Iterable[float]) -> Matrix:
    """由旋转矩阵和位移生成列向量约定的齐次变换。"""

    rows = tuple(_vector(row) for row in rotation)
    offset = _vector(translation)
    if len(rows) != 3 or any(len(row) != 3 for row in rows) or len(offset) != 3:
        raise ValueError("旋转必须为 3×3，位移必须为三维")
    return tuple(rows[row] + (offset[row],) for row in range(3)) + (
        (0.0, 0.0, 0.0, 1.0),
    )


def look_at_world_from_usd_camera(
    position_world: Iterable[float],
    target_world: Iterable[float],
    up_world: Iterable[float] = (0.0, 0.0, 1.0),
) -> Matrix:
    """构造 USD 相机位姿 ``T_W_Cusd``。

    USD 相机局部 +X 向右、+Y 向上、-Z 朝前。
    """

    position = _vector(position_world)
    target = _vector(target_world)
    forward = _normalize(tuple(target[index] - position[index] for index in range(3)))
    right = _normalize(_cross(forward, _normalize(up_world)))
    z_axis = tuple(-value for value in forward)
    y_axis = _normalize(_cross(z_axis, right))
    rotation = tuple(
        tuple(axis[row] for axis in (right, y_axis, z_axis)) for row in range(3)
    )
    return make_transform(rotation, position)


def apply_opencv_roll_to_usd_camera(
    world_from_camera_usd: Iterable[Iterable[float]], roll_degrees: float
) -> Matrix:
    """绕 OpenCV 光学轴施加 roll，并返回新的 USD 相机位姿。"""

    angle = math.radians(-float(roll_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    usd_local_roll = (
        (cosine, -sine, 0.0, 0.0),
        (sine, cosine, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return matrix_multiply(world_from_camera_usd, usd_local_roll)


USD_CAMERA_TO_OPENCV: Matrix = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def camera_from_object(
    world_from_camera_usd: Iterable[Iterable[float]],
    world_from_object: Iterable[Iterable[float]],
) -> Matrix:
    """计算 OpenCV 光学系下的 ``T_C_O``。

    实现的坐标链为 ``T_C_O = T_C_W · T_W_O``；当物体位姿由资产
    局部变换给出时，调用者应先组成 ``T_W_O = T_W_A · T_A_O``。
    """

    camera_usd_from_world = rigid_inverse(world_from_camera_usd)
    return matrix_multiply(
        USD_CAMERA_TO_OPENCV,
        matrix_multiply(camera_usd_from_world, world_from_object),
    )


def intrinsics_from_fov(
    width: int,
    height: int,
    horizontal_fov_degrees: float,
    vertical_fov_degrees: float,
) -> dict[str, float | list[list[float]]]:
    """由分辨率和水平/垂直视场角计算中心主点的针孔内参。"""

    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸必须为正数")
    if not 0.0 < horizontal_fov_degrees < 180.0 or not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError("视场角必须在 0° 与 180° 之间")
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_degrees) * 0.5))
    fy = height / (2.0 * math.tan(math.radians(vertical_fov_degrees) * 0.5))
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }


def project_point(
    intrinsics: dict[str, float | list[list[float]]], point_camera: Iterable[float]
) -> tuple[float, float]:
    """把 OpenCV 相机坐标点投影到像素坐标。"""

    x, y, z = _vector(point_camera)
    if z <= 0.0:
        raise ValueError("待投影点必须位于相机前方")
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    return fx * x / z + cx, fy * y / z + cy
