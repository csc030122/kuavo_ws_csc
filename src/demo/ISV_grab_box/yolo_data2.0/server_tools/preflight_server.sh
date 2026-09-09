#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/csc/isaacsim/python.sh}"

for command_name in nvidia-smi python3 setsid sha256sum; do
  command -v "${command_name}" >/dev/null || {
    echo "缺少命令：${command_name}" >&2
    exit 2
  }
done
[[ -x "${ISAAC_PYTHON}" ]] || {
  echo "Isaac Sim Python 不可执行：${ISAAC_PYTHON}" >&2
  exit 2
}

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${gpu_count}" -lt 2 ]]; then
  echo "至少需要两张 NVIDIA GPU，当前只检测到 ${gpu_count} 张。" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python3 -c 'import yaml' || {
  echo "系统 Python 缺少 PyYAML；请执行 python3 -m pip install PyYAML。" >&2
  exit 2
}

required_files=(
  configs/assets.yaml
  configs/camera.yaml
  configs/collection.yaml
  yolo_scene/lab_yolo_grab_data_s63_mount_corrected.usd
  yolo_scene/lab_yolo_grab_data.usd
  yolo_scene/box\(2\).usd
  yolo_scene/left.usd
  yolo_scene/right.usd
  yolo_scene/hose.usd
  yolo_scene/outhandle.usd
)
for relative_path in "${required_files[@]}"; do
  [[ -f "${PROJECT_ROOT}/${relative_path}" ]] || {
    echo "缺少必要文件：${relative_path}" >&2
    exit 2
  }
done
for side in right left; do
  for class_name in left right hose outhandle; do
    reference="${PROJECT_ROOT}/references/${side}_wrist_${class_name}_approach_reference.json"
    [[ -f "${reference}" ]] || {
      echo "缺少参考标定：${reference}" >&2
      exit 2
    }
  done
done

if [[ -f "${PROJECT_ROOT}/MANIFEST.sha256" ]]; then
  (
    cd "${PROJECT_ROOT}"
    sha256sum -c MANIFEST.sha256
  )
fi

available_kb="$(df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')"
required_kb=$((100 * 1024 * 1024))
if [[ "${available_kb}" -lt "${required_kb}" ]]; then
  echo "建议至少准备 100 GiB 可用空间；当前不足。" >&2
  exit 2
fi

(
  cd "${PROJECT_ROOT}"
  PYTHONPATH=. "${ISAAC_PYTHON}" -m unittest discover -s tests
)

temporary_root="$(mktemp -d)"
trap 'rm -rf "${temporary_root}"' EXIT
for side in right left; do
  gpu_index=0
  [[ "${side}" == "left" ]] && gpu_index=1
  python3 "${PROJECT_ROOT}/scripts/collect_approach_until_quota.py" \
    --hand-side "${side}" \
    --gpu-index "${gpu_index}" \
    --output-root "${temporary_root}/${side}" \
    --accepted-per-class-stage 250 \
    --samples-per-class-per-round 125 \
    --max-rounds 20 \
    --seed 20260911 \
    --isaac-python "${ISAAC_PYTHON}" \
    --reference-dir "${PROJECT_ROOT}/references" \
    --strict \
    --dry-run
done

echo "PREFLIGHT=PASS"
