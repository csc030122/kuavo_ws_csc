"""Pure-numpy hand pose randomization for the approach-open collector.

The sampler is deliberately independent of Isaac Sim.  A plan records the
random variables, and the Isaac adapter applies them to the transported
grasp/pregrasp pose.  The lateral offsets live in the plane perpendicular to
the saved approach direction; roll/pitch/yaw are applied in the palm frame.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


DEFAULT_STAGE_LIMITS: dict[str, dict[str, float]] = {
    "grasp": {"lateral_offset_m": 0.0, "orientation_delta_deg": 0.0},
    "far": {"lateral_offset_m": 0.10, "orientation_delta_deg": 15.0},
    "approach": {"lateral_offset_m": 0.07, "orientation_delta_deg": 10.0},
    "near": {"lateral_offset_m": 0.04, "orientation_delta_deg": 6.0},
    "pre_grasp": {"lateral_offset_m": 0.02, "orientation_delta_deg": 3.0},
}


def _unit(vector: Sequence[float], name: str) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm <= 1e-10:
        raise ValueError(f"{name} must be non-zero")
    return tuple(float(value) / norm for value in vector)  # type: ignore[return-value]


def _rotation_x(angle_rad: float) -> tuple[tuple[float, ...], ...]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def _rotation_y(angle_rad: float) -> tuple[tuple[float, ...], ...]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _rotation_z(angle_rad: float) -> tuple[tuple[float, ...], ...]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _mat_mul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(sum(float(left[row][k]) * float(right[k][column]) for k in range(3)) for column in range(3))
        for row in range(3)
    )


def sample_hand_perturbation(
    rng: Any,
    stage: str,
    limits: Mapping[str, Mapping[str, float]] | None = None,
    scale: float = 1.0,
) -> dict[str, float]:
    """Sample one deterministic hand perturbation for a distance stage.

    ``rng`` only needs a ``uniform(low, high)`` method, so ``random.Random``
    can be used by the plan generator without coupling this module to a
    particular random-number implementation.
    """

    configured = DEFAULT_STAGE_LIMITS if limits is None else limits
    if stage not in configured:
        raise ValueError(f"Unknown approach distance stage: {stage!r}")
    stage_limits = configured[stage]
    scale = float(scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be in [0, 1]")
    lateral = abs(float(stage_limits["lateral_offset_m"])) * scale
    orientation = abs(float(stage_limits["orientation_delta_deg"])) * scale
    return {
        "lateral_offset_1_m": float(rng.uniform(-lateral, lateral)),
        "lateral_offset_2_m": float(rng.uniform(-lateral, lateral)),
        "roll_delta_deg": float(rng.uniform(-orientation, orientation)),
        "pitch_delta_deg": float(rng.uniform(-orientation, orientation)),
        "yaw_delta_deg": float(rng.uniform(-orientation, orientation)),
    }


def sample_hand_perturbation_mixture(
    rng: Any,
    stage: str,
    limits: Mapping[str, Mapping[str, float]],
    mixture: Mapping[str, Any],
    forced_mode: str | None = None,
) -> tuple[dict[str, float], str]:
    """Mix centered candidates with a smaller full-range coverage tail."""

    centered_probability = float(
        mixture.get("centered_probability_by_stage", {}).get(
            stage, mixture.get("centered_probability", 0.0)
        )
    )
    centered_scale = float(
        mixture.get("centered_scale_by_stage", {}).get(
            stage, mixture.get("centered_scale", 1.0)
        )
    )
    if not 0.0 <= centered_probability <= 1.0:
        raise ValueError("centered_probability must be in [0, 1]")
    if not 0.0 <= centered_scale <= 1.0:
        raise ValueError("centered_scale must be in [0, 1]")
    if forced_mode is not None and forced_mode not in ("centered", "coverage_tail"):
        raise ValueError(f"Unknown forced mixture mode: {forced_mode!r}")
    mode = forced_mode or (
        "centered" if float(rng.random()) < centered_probability else "coverage_tail"
    )
    centered = mode == "centered"
    perturbation = sample_hand_perturbation(
        rng,
        stage,
        limits,
        scale=centered_scale if centered else 1.0,
    )
    return perturbation, mode


def stratified_mixture_mode(
    stage: str,
    ordinal: int,
    mixture: Mapping[str, Any],
) -> str:
    """Choose a deterministic mode while matching the configured proportion."""

    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    probability = float(
        mixture.get("centered_probability_by_stage", {}).get(
            stage, mixture.get("centered_probability", 0.0)
        )
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError("centered_probability must be in [0, 1]")
    centered_before = round(ordinal * probability)
    centered_after = round((ordinal + 1) * probability)
    return "centered" if centered_after > centered_before else "coverage_tail"


def apply_hand_perturbation(
    base_world_from_palm: Any,
    grasp_world_from_palm: Any,
    pregrasp_world_from_palm: Any,
    perturbation: Mapping[str, float] | None,
) -> tuple[tuple[float, ...], ...]:
    """Apply lateral and local RPY perturbations to a transported palm pose.

    ``grasp_world_from_palm`` and ``pregrasp_world_from_palm`` define the
    approach line.  The returned transform is a fresh 4x4 matrix and does not
    mutate the input arrays.
    """

    base = [[float(value) for value in row] for row in base_world_from_palm]
    grasp = [[float(value) for value in row] for row in grasp_world_from_palm]
    pregrasp = [[float(value) for value in row] for row in pregrasp_world_from_palm]
    if any(len(matrix) != 4 for matrix in (base, grasp, pregrasp)) or any(
        len(row) != 4 for matrix in (base, grasp, pregrasp) for row in matrix
    ):
        raise ValueError("Palm poses must be 4x4 matrices")
    perturbation = {} if perturbation is None else perturbation

    approach_vector = tuple(pregrasp[index][3] - grasp[index][3] for index in range(3))
    approach_axis = _unit(approach_vector, "approach direction")
    # Use the palm local X axis projected into the approach-normal plane.  The
    # fallback keeps the basis well-defined if a reference happens to align
    # its X axis with the approach direction.
    e1 = tuple(base[index][0] for index in range(3))
    projection = _dot(e1, approach_axis)
    e1 = tuple(e1[index] - projection * approach_axis[index] for index in range(3))
    if math.sqrt(_dot(e1, e1)) <= 1e-10:
        e1 = tuple(base[index][1] for index in range(3))
        projection = _dot(e1, approach_axis)
        e1 = tuple(e1[index] - projection * approach_axis[index] for index in range(3))
    e1 = _unit(e1, "lateral basis e1")
    e2 = _unit(_cross(approach_axis, e1), "lateral basis e2")

    delta_1 = float(perturbation.get("lateral_offset_1_m", 0.0))
    delta_2 = float(perturbation.get("lateral_offset_2_m", 0.0))
    for index in range(3):
        base[index][3] += delta_1 * e1[index] + delta_2 * e2[index]

    roll = math.radians(float(perturbation.get("roll_delta_deg", 0.0)))
    pitch = math.radians(float(perturbation.get("pitch_delta_deg", 0.0)))
    yaw = math.radians(float(perturbation.get("yaw_delta_deg", 0.0)))
    local_delta = _mat_mul(_mat_mul(_rotation_x(roll), _rotation_y(pitch)), _rotation_z(yaw))
    base_rotation = tuple(tuple(base[row][column] for column in range(3)) for row in range(3))
    rotated = _mat_mul(base_rotation, local_delta)
    for row in range(3):
        for column in range(3):
            base[row][column] = rotated[row][column]
    return tuple(tuple(row) for row in base)


__all__ = [
    "DEFAULT_STAGE_LIMITS",
    "apply_hand_perturbation",
    "sample_hand_perturbation",
    "sample_hand_perturbation_mixture",
    "stratified_mixture_mode",
]
