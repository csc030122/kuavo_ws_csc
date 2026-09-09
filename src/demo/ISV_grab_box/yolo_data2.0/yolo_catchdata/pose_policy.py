"""Pose transport rules for the two-stage hand-mounted-camera dataset.

The functions in this module are intentionally independent of Isaac Sim.  They
describe how a saved palm reference is transported when the object is moved or
rotated in the table frame.  Isaac Sim adapters can convert the returned 4x4
column-vector matrices to USD transforms.
"""

from __future__ import annotations

import math
from typing import Iterable


ELONGATED_CLASSES = frozenset(("left", "right", "hose"))
FIXED_YAW_CLASSES = frozenset(("outhandle",))


def _as_matrix(values: Iterable[Iterable[float]]) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(float(value) for value in row) for row in values)
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Pose must be a 4x4 matrix")
    return matrix


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def wrap_to_half_turn(angle_rad: float) -> float:
    """Wrap an angle to the nearest representative in ``[-pi/2, pi/2]``.

    This is the quotient representation for a grasp that is invariant under a
    180-degree object turn.  Consequently an object yaw delta of ``pi`` gives
    zero palm rotation.
    """

    angle = float(angle_rad)
    wrapped = (angle + 0.5 * math.pi) % math.pi - 0.5 * math.pi
    # The two endpoints represent the same C2 class.  Keep the sign of an
    # exact quarter-turn so a +90-degree input remains the intuitive +90-degree
    # hand turn (and -90 remains -90).
    if math.isclose(wrapped, -0.5 * math.pi, abs_tol=1e-12) and math.sin(angle) > 0.0:
        return 0.5 * math.pi
    return wrapped


def effective_hand_yaw_delta(class_name: str, object_yaw_delta_rad: float) -> float:
    """Return the palm yaw change prescribed by the class policy."""

    if class_name in FIXED_YAW_CLASSES:
        return 0.0
    if class_name in ELONGATED_CLASSES:
        return wrap_to_half_turn(object_yaw_delta_rad)
    raise ValueError(f"Unknown workpiece class: {class_name!r}")


def rotation_z(angle_rad: float) -> tuple[tuple[float, ...], ...]:
    """Return a right-handed world-Z rotation as a 3x3 tuple."""

    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


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


def _settled_planar_heading_delta(
    reference_object_world: tuple[tuple[float, ...], ...],
    new_object_world: tuple[tuple[float, ...], ...],
) -> float | None:
    """Measure the object's actual table-plane heading change.

    Physics settling can rotate a part tens of degrees away from its requested
    release yaw.  Pick the same local object axis in both poses whose XY
    projection is best conditioned, then compare its world-plane headings.
    This also remains usable when a part rolls onto another support face.
    """

    best: tuple[float, int] | None = None
    for column in range(3):
        reference_norm = math.hypot(
            reference_object_world[0][column],
            reference_object_world[1][column],
        )
        new_norm = math.hypot(
            new_object_world[0][column],
            new_object_world[1][column],
        )
        score = min(reference_norm, new_norm)
        if best is None or score > best[0]:
            best = (score, column)
    if best is None or best[0] <= 1e-6:
        return None
    column = best[1]
    reference_heading = math.atan2(
        reference_object_world[1][column],
        reference_object_world[0][column],
    )
    new_heading = math.atan2(
        new_object_world[1][column],
        new_object_world[0][column],
    )
    return wrap_to_pi(new_heading - reference_heading)


def settled_effective_hand_yaw_delta(
    reference_object_world: Iterable[Iterable[float]],
    new_object_world: Iterable[Iterable[float]],
    class_name: str,
    requested_object_yaw_delta_rad: float,
) -> float:
    """Return the hand yaw dictated by the *settled* object heading.

    The small ``outhandle`` intentionally keeps its calibrated palm yaw.  The
    elongated C2-symmetric parts follow the measured settled heading modulo
    180 degrees, falling back to the requested yaw only for a degenerate
    projection.
    """

    if class_name in FIXED_YAW_CLASSES:
        return 0.0
    if class_name not in ELONGATED_CLASSES:
        raise ValueError(f"Unknown workpiece class: {class_name!r}")
    measured = _settled_planar_heading_delta(
        _as_matrix(reference_object_world),
        _as_matrix(new_object_world),
    )
    return wrap_to_half_turn(
        requested_object_yaw_delta_rad if measured is None else measured
    )


def transport_reference_pose(
    reference_object_world: Iterable[Iterable[float]],
    reference_palm_world: Iterable[Iterable[float]],
    new_object_world: Iterable[Iterable[float]],
    class_name: str,
    object_yaw_delta_rad: float,
    *,
    use_settled_heading: bool = False,
) -> tuple[tuple[float, ...], ...]:
    """Transport a saved palm pose to a new planar object pose.

    The object is treated as rotating about its sampling center.  The saved
    palm-to-center translation is rotated by the effective hand yaw, while the
    palm orientation receives the same world-Z yaw delta.  For ``outhandle``
    the effective delta is zero; for the three elongated parts it is folded
    modulo 180 degrees.  Roll, pitch, finger joints, and the camera mount are
    inherited from the saved reference.
    """

    object_ref = _as_matrix(reference_object_world)
    palm_ref = _as_matrix(reference_palm_world)
    object_new = _as_matrix(new_object_world)
    effective_delta = (
        settled_effective_hand_yaw_delta(
            object_ref,
            object_new,
            class_name,
            object_yaw_delta_rad,
        )
        if use_settled_heading
        else effective_hand_yaw_delta(class_name, object_yaw_delta_rad)
    )
    rz = rotation_z(effective_delta)

    object_ref_translation = tuple(object_ref[row][3] for row in range(3))
    palm_ref_translation = tuple(palm_ref[row][3] for row in range(3))
    object_new_translation = tuple(object_new[row][3] for row in range(3))
    reference_offset = tuple(
        palm_ref_translation[index] - object_ref_translation[index]
        for index in range(3)
    )
    transported_offset = _mat_vec(rz, reference_offset)
    palm_translation = tuple(
        object_new_translation[index] + transported_offset[index]
        for index in range(3)
    )
    reference_rotation = tuple(tuple(palm_ref[row][column] for column in range(3)) for row in range(3))
    palm_rotation = _mat_mul(rz, reference_rotation)
    return tuple(
        tuple(
            palm_rotation[row][column] if column < 3 else palm_translation[row]
            for column in range(4)
        )
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


__all__ = [
    "ELONGATED_CLASSES",
    "FIXED_YAW_CLASSES",
    "effective_hand_yaw_delta",
    "rotation_z",
    "settled_effective_hand_yaw_delta",
    "transport_reference_pose",
    "wrap_to_half_turn",
    "wrap_to_pi",
]
