#!/bin/bash

# Simple mono-GPU launch script for RLT
# Usage: ./launch_mono.sh cfgs/run_cfg/your_config.yaml [extra_hydra_args]

# resolve yaml path
if [[ "$1" == cfgs/run_cfg/* ]]; then
  yaml_file="$1"
else
  yaml_file="cfgs/run_cfg/$1"
fi

# collect extra args
extra_args=()
for arg in "${@:2}"; do
  extra_args+=("$arg")
done

echo "Running mono-GPU launch script..."
echo "Config file: $yaml_file"
echo "Extra args: ${extra_args[@]}"

# Check if config file exists
if [[ ! -f "$yaml_file" ]]; then
  echo "Error: Config file $yaml_file not found!"
  exit 1
fi

# Set CUDA device to use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Launch training
echo "Starting mono-GPU training..."
python train.py --config-path="$(dirname "$yaml_file")" --config-name="$(basename "$yaml_file" .yaml)" "${extra_args[@]}"

echo "Training completed!" 