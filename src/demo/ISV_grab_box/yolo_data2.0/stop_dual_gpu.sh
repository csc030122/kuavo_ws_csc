#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/dataset/approach_open_formal_5000_each}"
STATE_ROOT="${OUTPUT_ROOT}/run_state"

found=0
for side in right left; do
  pid_file="${STATE_ROOT}/${side}.pid"
  if [[ ! -s "${pid_file}" ]]; then
    continue
  fi
  pid="$(<"${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    found=1
    echo "停止 ${side} 进程组，PID=${pid}"
    kill -TERM -- "-${pid}" 2>/dev/null || true
  fi
done

if [[ "${found}" -eq 0 ]]; then
  echo "没有发现正在运行的左右手采集进程。"
  exit 0
fi

for _ in {1..20}; do
  alive=0
  for side in right left; do
    pid_file="${STATE_ROOT}/${side}.pid"
    [[ -s "${pid_file}" ]] || continue
    pid="$(<"${pid_file}")"
    kill -0 "${pid}" 2>/dev/null && alive=1
  done
  [[ "${alive}" -eq 0 ]] && break
  sleep 0.5
done

for side in right left; do
  pid_file="${STATE_ROOT}/${side}.pid"
  [[ -s "${pid_file}" ]] || continue
  pid="$(<"${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "${side} 未及时退出，发送 SIGKILL：PID=${pid}" >&2
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
done
echo "采集已停止；再次运行启动命令会从现有 manifest 安全续采。"
