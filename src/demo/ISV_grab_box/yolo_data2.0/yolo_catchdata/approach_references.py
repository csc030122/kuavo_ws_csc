"""Validation and right-to-left mirroring for approach references."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .hand_sides import hand_spec, reference_filename


MIRROR_METHOD = "right_to_left_sagittal_mirror_v1"
_WORLD_LATERAL_REFLECTION = (
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
# In the S63 r7 frames, left/right CAD geometry is reflected across local Y.
_R7_LOCAL_REFLECTION = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _mat_mul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            sum(float(left[row][k]) * float(right[k][column]) for k in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _transpose(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(matrix[column][row]) for column in range(3))
        for row in range(3)
    )


def _mat_vec(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("Quaternion must contain four wxyz values")
    quaternion = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("Quaternion must be non-zero")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _quat_to_rotation(
    values: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    w, x, y, z = _normalize_quaternion(values)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _rotation_to_quat(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation to a normalized wxyz quaternion."""

    m = tuple(tuple(float(value) for value in row) for row in matrix)
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (m[2][1] - m[1][2]) / scale,
            (m[0][2] - m[2][0]) / scale,
            (m[1][0] - m[0][1]) / scale,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        values = (
            (m[2][1] - m[1][2]) / scale,
            0.25 * scale,
            (m[0][1] + m[1][0]) / scale,
            (m[0][2] + m[2][0]) / scale,
        )
    elif m[1][1] > m[2][2]:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        values = (
            (m[0][2] - m[2][0]) / scale,
            (m[0][1] + m[1][0]) / scale,
            0.25 * scale,
            (m[1][2] + m[2][1]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        values = (
            (m[1][0] - m[0][1]) / scale,
            (m[0][2] + m[2][0]) / scale,
            (m[1][2] + m[2][1]) / scale,
            0.25 * scale,
        )
    quaternion = _normalize_quaternion(values)
    # A stable sign makes generated JSON deterministic across platforms.
    if quaternion[0] < 0.0:
        quaternion = tuple(-value for value in quaternion)  # type: ignore[assignment]
    return quaternion


def _pose_transform(pose: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
    rotation = _quat_to_rotation(pose["quaternion_wxyz"])
    translation = tuple(float(value) for value in pose["translation_m"])
    if len(translation) != 3:
        raise ValueError("Pose translation must contain three values")
    return tuple(
        tuple(rotation[row][column] for column in range(3)) + (translation[row],)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def _compose(
    parent_from_middle: Sequence[Sequence[float]],
    middle_from_child: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    rotation_parent = tuple(
        tuple(float(parent_from_middle[row][column]) for column in range(3))
        for row in range(3)
    )
    rotation_child = tuple(
        tuple(float(middle_from_child[row][column]) for column in range(3))
        for row in range(3)
    )
    rotation = _mat_mul(rotation_parent, rotation_child)
    child_translation = tuple(float(middle_from_child[row][3]) for row in range(3))
    rotated_translation = _mat_vec(rotation_parent, child_translation)
    translation = tuple(
        float(parent_from_middle[row][3]) + rotated_translation[row]
        for row in range(3)
    )
    return tuple(rotation[row] + (translation[row],) for row in range(3)) + (
        (0.0, 0.0, 0.0, 1.0),
    )


def _inverse(transform: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    rotation = tuple(
        tuple(float(transform[row][column]) for column in range(3))
        for row in range(3)
    )
    inverse_rotation = _transpose(rotation)
    translation = tuple(float(transform[row][3]) for row in range(3))
    inverse_translation = _mat_vec(inverse_rotation, tuple(-value for value in translation))
    return tuple(
        inverse_rotation[row] + (inverse_translation[row],) for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def _transform_pose(
    transform: Sequence[Sequence[float]], path: str
) -> dict[str, Any]:
    rotation = tuple(
        tuple(float(transform[row][column]) for column in range(3))
        for row in range(3)
    )
    return {
        "path": path,
        "translation_m": [float(transform[row][3]) for row in range(3)],
        "quaternion_wxyz": list(_rotation_to_quat(rotation)),
    }


def _fixed_transform(spec: dict[str, Any], key: str) -> tuple[tuple[float, ...], ...]:
    entry = spec[key]
    return _pose_transform(
        {
            "translation_m": entry["translation_m"],
            "quaternion_wxyz": entry["quaternion_wxyz"],
        }
    )


def _left_palm_local_reflection() -> tuple[tuple[float, ...], ...]:
    """Map the left palm axes to the reflected right palm axes from CAD."""

    right_rotation = _quat_to_rotation(
        hand_spec("right")["r7_from_palm"]["quaternion_wxyz"]
    )
    left_rotation = _quat_to_rotation(
        hand_spec("left")["r7_from_palm"]["quaternion_wxyz"]
    )
    return _mat_mul(
        _mat_mul(_transpose(right_rotation), _R7_LOCAL_REFLECTION),
        left_rotation,
    )


def _mirror_palm_pose(
    source_pose: dict[str, Any], object_pose: dict[str, Any]
) -> dict[str, Any]:
    source_rotation = _quat_to_rotation(source_pose["quaternion_wxyz"])
    mirrored_rotation = _mat_mul(
        _mat_mul(_WORLD_LATERAL_REFLECTION, source_rotation),
        _left_palm_local_reflection(),
    )
    source_translation = [float(value) for value in source_pose["translation_m"]]
    object_translation = [float(value) for value in object_pose["translation_m"]]
    mirrored_translation = [
        2.0 * object_translation[0] - source_translation[0],
        source_translation[1],
        source_translation[2],
    ]
    return {
        "path": str(hand_spec("left")["palm"]),
        "translation_m": mirrored_translation,
        "quaternion_wxyz": list(_rotation_to_quat(mirrored_rotation)),
    }


def _left_camera_from_palm(
    world_from_left_palm: dict[str, Any]
) -> dict[str, Any]:
    left = hand_spec("left")
    r7_from_camera_rgb = _compose(
        _fixed_transform(left, "r7_from_camera_link"),
        _fixed_transform(left, "camera_link_from_rgb"),
    )
    palm_from_camera = _compose(
        _inverse(_fixed_transform(left, "r7_from_palm")),
        r7_from_camera_rgb,
    )
    world_from_camera = _compose(
        _pose_transform(world_from_left_palm), palm_from_camera
    )
    return _transform_pose(world_from_camera, str(left["rgb_camera"]))


def _relative_pose(
    world_from_parent: dict[str, Any],
    world_from_child: dict[str, Any],
    parent_path: str,
    child_path: str,
) -> dict[str, Any]:
    parent_from_child = _compose(
        _inverse(_pose_transform(world_from_parent)),
        _pose_transform(world_from_child),
    )
    pose = _transform_pose(parent_from_child, child_path)
    pose.pop("path")
    return {
        "parent_path": parent_path,
        "child_path": child_path,
        **pose,
    }


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror_right_reference_payload(
    source: dict[str, Any],
    source_path: Path,
    target_path: Path,
) -> dict[str, Any]:
    """Derive one left-hand reference from a calibrated right-hand reference.

    Position is reflected across the world-X plane through the workpiece
    origin in each snapshot. Orientation uses both the world reflection and
    the real S63 left/right palm frame conventions, so the result is a proper
    rotation rather than an invalid reflected rotation matrix. The left D405
    pose is rebuilt from the real left palm-to-camera fixed transform.
    """

    if source.get("status") != "complete":
        raise ValueError(f"Right reference is incomplete: {source_path}")
    if source.get("hand_side") != "right":
        raise ValueError(f"Mirror source is not a right-hand reference: {source_path}")
    class_name = str(source.get("class_name", ""))
    if not class_name:
        raise ValueError(f"Mirror source has no class_name: {source_path}")
    snapshots = source.get("snapshots", {})
    if not {"grasp", "pregrasp"}.issubset(snapshots):
        raise ValueError(f"Mirror source lacks grasp/pregrasp: {source_path}")

    generated_at = datetime.now(timezone.utc).isoformat()
    target = copy.deepcopy(source)
    target["generated_at_utc"] = generated_at
    target["hand_side"] = "left"
    target["reference_output"] = str(target_path.resolve())
    target["derivation"] = {
        "method": MIRROR_METHOD,
        "source_reference": str(source_path.resolve()),
        "source_sha256": _source_digest(source_path),
        "derived_at_utc": generated_at,
        "mirror_plane": "world X normal through each reference object origin",
        "palm_frame_mapping": "S63 CAD right/left r7-to-palm transforms",
        "camera_mapping": "real S63 left palm-to-side-D405 fixed transform",
    }

    left_hand = hand_spec("left")
    mirrored_snapshots: dict[str, dict[str, Any]] = {}
    for stage_name, source_snapshot in snapshots.items():
        snapshot = copy.deepcopy(source_snapshot)
        object_pose = copy.deepcopy(source_snapshot["world_from_object"])
        palm_pose = _mirror_palm_pose(
            source_snapshot["world_from_palm"], object_pose
        )
        camera_pose = _left_camera_from_palm(palm_pose)
        snapshot["source_captured_at_utc"] = source_snapshot.get("captured_at_utc")
        snapshot["derived_at_utc"] = generated_at
        snapshot["world_from_palm"] = palm_pose
        snapshot["world_from_camera_rgb"] = camera_pose
        snapshot["world_from_object"] = object_pose
        snapshot["palm_from_object"] = _relative_pose(
            palm_pose,
            object_pose,
            str(left_hand["palm"]),
            str(object_pose.get("path", "")),
        )
        mirrored_snapshots[stage_name] = snapshot
    target["snapshots"] = mirrored_snapshots

    grasp = mirrored_snapshots["grasp"]["world_from_palm"]["translation_m"]
    pregrasp = mirrored_snapshots["pregrasp"]["world_from_palm"]["translation_m"]
    delta = [float(grasp[index]) - float(pregrasp[index]) for index in range(3)]
    length = math.sqrt(sum(value * value for value in delta))
    if length <= 1e-6:
        raise ValueError(f"Mirrored grasp/pregrasp distance is zero: {source_path}")
    approach = copy.deepcopy(source.get("approach", {}))
    approach["direction_world_from_pregrasp_to_grasp"] = [
        value / length for value in delta
    ]
    approach["backward_direction_world"] = [-value / length for value in delta]
    approach["grasp_to_pregrasp_distance_m"] = length
    target["approach"] = approach
    return target


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_mirrored_left_references(
    reference_dir: Path,
    class_names: Iterable[str],
    force: bool = False,
) -> dict[str, list[Path]]:
    """Create or refresh derived left references while preserving manual ones."""

    reference_dir = Path(reference_dir)
    result: dict[str, list[Path]] = {
        "written": [],
        "current": [],
        "preserved_manual": [],
    }
    for class_name in class_names:
        source_path = reference_dir / reference_filename("right", class_name)
        target_path = reference_dir / reference_filename("left", class_name)
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Cannot mirror missing right reference: {source_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot mirror invalid right reference: {source_path}: {exc}") from exc
        source_digest = _source_digest(source_path)

        existing: dict[str, Any] | None = None
        try:
            existing = json.loads(target_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        derivation = (existing or {}).get("derivation", {})
        is_derived = derivation.get("method") == MIRROR_METHOD
        is_valid_left = (
            (existing or {}).get("status") == "complete"
            and (existing or {}).get("hand_side") == "left"
            and (existing or {}).get("class_name") == class_name
        )
        if existing is not None and is_valid_left and not is_derived and not force:
            result["preserved_manual"].append(target_path)
            continue
        if (
            is_valid_left
            and is_derived
            and derivation.get("source_sha256") == source_digest
            and not force
        ):
            result["current"].append(target_path)
            continue

        payload = mirror_right_reference_payload(source, source_path, target_path)
        _atomic_write(target_path, payload)
        result["written"].append(target_path)
    return result


def validate_reference_set(
    reference_dir: Path,
    hand_side: str,
    class_names: Iterable[str],
    required_snapshots: Iterable[str] = ("grasp", "pregrasp"),
) -> list[str]:
    """Return human-readable errors without accepting cross-hand references."""

    hand_spec(hand_side)
    errors: list[str] = []
    required = set(required_snapshots)
    for class_name in class_names:
        path = Path(reference_dir) / reference_filename(hand_side, class_name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            errors.append(f"missing reference: {path}")
            continue
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON reference: {path}: {exc}")
            continue
        if payload.get("status") != "complete":
            errors.append(f"incomplete reference: {path}")
        if payload.get("hand_side") != hand_side:
            errors.append(
                f"hand-side mismatch: {path}: {payload.get('hand_side')!r} != {hand_side!r}"
            )
        if payload.get("class_name") != class_name:
            errors.append(
                f"class mismatch: {path}: {payload.get('class_name')!r} != {class_name!r}"
            )
        missing = required - set(payload.get("snapshots", {}))
        if missing:
            errors.append(f"missing snapshots {sorted(missing)}: {path}")

        derivation = payload.get("derivation", {})
        if hand_side == "left" and derivation.get("method") == MIRROR_METHOD:
            source_path = Path(reference_dir) / reference_filename("right", class_name)
            try:
                actual_digest = _source_digest(source_path)
            except FileNotFoundError:
                errors.append(f"missing mirror source: {source_path}")
            else:
                if derivation.get("source_sha256") != actual_digest:
                    errors.append(f"stale mirrored reference: {path}")
    return errors


__all__ = [
    "MIRROR_METHOD",
    "ensure_mirrored_left_references",
    "mirror_right_reference_payload",
    "validate_reference_set",
]
