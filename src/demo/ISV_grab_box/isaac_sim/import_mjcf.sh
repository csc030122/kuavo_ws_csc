#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
demo_dir="$(cd -- "${script_dir}/.." && pwd)"
isaac_root="${ISAAC_SIM_ROOT:-/isaac-sim}"
isaac_python="${isaac_root}/python.sh"
importer="${isaac_root}/standalone_examples/api/isaacsim.asset.importer.mjcf/mjcf_import.py"
mjcf_scene="${MJCF_SCENE:-${demo_dir}/biped_s63/xml/scenes/lab_sence_static_with_box.xml}"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${USD_OUTPUT_DIR:-${script_dir}/generated/mjcf_baseline_${timestamp}}"

if [[ ! -x "${isaac_python}" ]]; then
  echo "Isaac Sim Python was not found: ${isaac_python}" >&2
  echo "Run this script inside the isaac-sim container, or set ISAAC_SIM_ROOT." >&2
  exit 2
fi

if [[ ! -f "${importer}" ]]; then
  echo "Isaac Sim 6 MJCF importer example was not found: ${importer}" >&2
  exit 2
fi

python3 "${script_dir}/inspect_mjcf.py" "${mjcf_scene}"
mkdir -p "${output_dir}"

echo "Importing S63 lab MJCF..."
echo "  source: ${mjcf_scene}"
echo "  output: ${output_dir}"

"${isaac_python}" "${importer}" \
  --mjcf "${mjcf_scene}" \
  --usd-path "${output_dir}" \
  --robot-type "Mobile Manipulators" \
  --import-scene \
  --debug-mode \
  --no-fix-base \
  "$@"

echo "Import complete. Generated USD files:"
find "${output_dir}" -type f \( -name '*.usd' -o -name '*.usda' -o -name '*.usdc' \) -print

