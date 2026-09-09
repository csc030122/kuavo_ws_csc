#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/csc/isaacsim/python.sh}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/dataset/approach_open_formal_5000_each}"
REFERENCE_DIR="${REFERENCE_DIR:-${PROJECT_ROOT}/references}"
RIGHT_GPU="${RIGHT_GPU:-0}"
LEFT_GPU="${LEFT_GPU:-1}"
DATASET_SEED="${DATASET_SEED:-20260911}"
ACCEPTED_PER_CLASS_STAGE=250
SAMPLES_PER_CLASS_PER_ROUND="${SAMPLES_PER_CLASS_PER_ROUND:-125}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "Isaac Sim Python 不可执行：${ISAAC_PYTHON}" >&2
  exit 2
fi
if [[ "${RIGHT_GPU}" == "${LEFT_GPU}" ]]; then
  echo "RIGHT_GPU 与 LEFT_GPU 必须是不同的物理 GPU 编号" >&2
  exit 2
fi

LOG_ROOT="${OUTPUT_ROOT}/logs"
STATE_ROOT="${OUTPUT_ROOT}/run_state"
mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"

for side in right left; do
  pid_file="${STATE_ROOT}/${side}.pid"
  if [[ -s "${pid_file}" ]]; then
    pid="$(<"${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "${side} 分支已经运行，PID=${pid}；拒绝重复启动。" >&2
      exit 3
    fi
  fi
done

common_args=(
  --accepted-per-class-stage "${ACCEPTED_PER_CLASS_STAGE}"
  --samples-per-class-per-round "${SAMPLES_PER_CLASS_PER_ROUND}"
  --max-rounds "${MAX_ROUNDS}"
  --seed "${DATASET_SEED}"
  --isaac-python "${ISAAC_PYTHON}"
  --reference-dir "${REFERENCE_DIR}"
  --strict
)

right_command=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/collect_approach_until_quota.py"
  --hand-side right
  --gpu-index "${RIGHT_GPU}"
  --output-root "${OUTPUT_ROOT}/right_wrist_d405"
  "${common_args[@]}"
)
left_command=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/collect_approach_until_quota.py"
  --hand-side left
  --gpu-index "${LEFT_GPU}"
  --output-root "${OUTPUT_ROOT}/left_wrist_d405"
  "${common_args[@]}"
)

echo "启动右手：GPU ${RIGHT_GPU}，目标 5000 张"
setsid "${right_command[@]}" >"${LOG_ROOT}/right_gpu${RIGHT_GPU}.log" 2>&1 < /dev/null &
right_pid=$!
printf '%s\n' "${right_pid}" >"${STATE_ROOT}/right.pid"

echo "启动左手：GPU ${LEFT_GPU}，目标 5000 张"
setsid "${left_command[@]}" >"${LOG_ROOT}/left_gpu${LEFT_GPU}.log" 2>&1 < /dev/null &
left_pid=$!
printf '%s\n' "${left_pid}" >"${STATE_ROOT}/left.pid"

stop_children() {
  kill -TERM -- "-${right_pid}" 2>/dev/null || true
  kill -TERM -- "-${left_pid}" 2>/dev/null || true
}
trap 'stop_children; exit 130' INT TERM

echo "RIGHT_PID=${right_pid}"
echo "LEFT_PID=${left_pid}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

set +e
wait "${right_pid}"
right_code=$?
wait "${left_pid}"
left_code=$?
set -e

rm -f "${STATE_ROOT}/right.pid" "${STATE_ROOT}/left.pid"
printf '%s\n' "${right_code}" >"${STATE_ROOT}/right.exit_code"
printf '%s\n' "${left_code}" >"${STATE_ROOT}/left.exit_code"

if [[ "${right_code}" -eq 0 && "${left_code}" -eq 0 ]]; then
  echo "STATUS=PASS：左右手均达到 5000 张严格配额"
  exit 0
fi
echo "STATUS=INCOMPLETE：right=${right_code} left=${left_code}" >&2
exit 1
