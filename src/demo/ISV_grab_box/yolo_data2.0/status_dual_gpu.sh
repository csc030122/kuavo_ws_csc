#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/dataset/approach_open_formal_5000_each}"
STATE_ROOT="${OUTPUT_ROOT}/run_state"

nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || true
for side in right left; do
  camera_root="${OUTPUT_ROOT}/${side}_wrist_d405"
  pid_file="${STATE_ROOT}/${side}.pid"
  state="stopped"
  pid="-"
  if [[ -s "${pid_file}" ]]; then
    pid="$(<"${pid_file}")"
    kill -0 "${pid}" 2>/dev/null && state="running"
  fi
  accepted=0
  if [[ -d "${camera_root}/_metadata" ]]; then
    accepted="$(find "${camera_root}/_metadata/train" "${camera_root}/_metadata/val" "${camera_root}/_metadata/test" -maxdepth 1 -type f -name '*_metadata.json' 2>/dev/null | wc -l)"
  fi
  echo "${side}: state=${state} pid=${pid} accepted=${accepted}/5000"
  log_file="$(find "${OUTPUT_ROOT}/logs" -maxdepth 1 -type f -name "${side}_gpu*.log" 2>/dev/null | head -1 || true)"
  if [[ -n "${log_file}" ]]; then
    tail -5 "${log_file}"
  fi
done
