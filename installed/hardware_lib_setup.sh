#!/usr/bin/env sh

script_dir=$(cd "$(dirname "$0")" && pwd)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$script_dir/lib
export PATH=$PATH:$script_dir/bin
export CMAKE_PREFIX_PATH=$script_dir:$CMAKE_PREFIX_PATH
export KUAVO_ASSETS_PATH=$script_dir/share/kuavo_assets
export KUAVO_COMMON_PATH=$script_dir/share/kuavo_common
export HARDWARE_PLANT_PATH=$script_dir/share/hardware_plant