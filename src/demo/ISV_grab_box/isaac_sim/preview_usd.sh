#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_root="${ISAAC_SIM_ROOT:-/isaac-sim}"
isaac_python="${isaac_root}/python.sh"

if [[ ! -x "${isaac_python}" ]]; then
  echo "Isaac Sim Python was not found: ${isaac_python}" >&2
  echo "Run this script inside the isaac-sim container, or set ISAAC_SIM_ROOT." >&2
  exit 2
fi

"${isaac_python}" "${script_dir}/preview_usd.py" "$@"

