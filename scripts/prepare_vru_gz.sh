#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
  echo "Usage: bash scripts/prepare_vru_gz.sh {all|colmap|weights|config} --data-root PATH --output-root PATH"
  exit 1
fi
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IFVC_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${IFVC_DATA_ROOT:-}"
OUTPUT_ROOT="${IFVC_OUTPUT_ROOT:-}"
SCENE_DIR="${VRU_SCENE_DIR:-}"
CHECKPOINT_DIR="${TORCH_CHECKPOINT_DIR:-${HOME}/.cache/torch/hub/checkpoints}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --scene-dir)
      SCENE_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$SCENE_DIR" && -n "$DATA_ROOT" ]]; then
  SCENE_DIR="$DATA_ROOT/prepared/VRU/Dataset/VRU_gz"
fi

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Missing $name"
    exit 1
  fi
}

prepare_colmap() {
  require_value "--data-root or --scene-dir" "$SCENE_DIR"

  local frame0="$SCENE_DIR/frame000000"
  mkdir -p "$frame0/distorted/sparse"

  if [[ ! -f "$frame0/sparse/0/cameras.bin" ]]; then
    colmap feature_extractor \
      --database_path "$frame0/distorted/database.db" \
      --image_path "$frame0/inputs" \
      --ImageReader.single_camera 1 \
      --ImageReader.camera_model OPENCV \
      --SiftExtraction.use_gpu 0

    colmap exhaustive_matcher \
      --database_path "$frame0/distorted/database.db" \
      --SiftMatching.use_gpu 0

    colmap mapper \
      --database_path "$frame0/distorted/database.db" \
      --image_path "$frame0/inputs" \
      --output_path "$frame0/distorted/sparse"

    colmap image_undistorter \
      --image_path "$frame0/inputs" \
      --input_path "$frame0/distorted/sparse/0" \
      --output_path "$frame0" \
      --output_type COLMAP

    mkdir -p "$frame0/sparse/0"
    for file in cameras.bin images.bin points3D.bin; do
      if [[ -f "$frame0/sparse/$file" ]]; then
        mv "$frame0/sparse/$file" "$frame0/sparse/0/$file"
      fi
    done
  fi

  for frame_id in $(seq 1 249); do
    printf -v frame_name 'frame%06d' "$frame_id"
    local frame_dir="$SCENE_DIR/$frame_name"

    if [[ -f "$frame_dir/sparse/0/cameras.bin" && -d "$frame_dir/images" ]]; then
      continue
    fi

    colmap image_undistorter \
      --image_path "$frame_dir" \
      --input_path "$frame0/distorted/sparse/0" \
      --output_path "$frame_dir" \
      --output_type COLMAP

    mkdir -p "$frame_dir/sparse/0"
    for file in cameras.bin images.bin points3D.bin; do
      if [[ -f "$frame_dir/sparse/$file" ]]; then
        mv "$frame_dir/sparse/$file" "$frame_dir/sparse/0/$file"
      fi
    done
  done
}

prepare_weights() {
  mkdir -p "$CHECKPOINT_DIR"

  if [[ ! -f "$CHECKPOINT_DIR/vgg16-397923af.pth" ]]; then
    aria2c --continue=true --max-connection-per-server=16 --split=16 \
      --min-split-size=1M --file-allocation=none \
      --dir="$CHECKPOINT_DIR" \
      --out='vgg16-397923af.pth' \
      'https://hf-mirror.com/ByteDance/LatentSync-1.5/resolve/main/auxiliary/vgg16-397923af.pth'
  fi

  if [[ ! -f "$CHECKPOINT_DIR/vgg.pth" ]]; then
    curl -fL --retry 5 \
      -o "$CHECKPOINT_DIR/vgg.pth" \
      'https://ghfast.top/https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth'
  fi
}

prepare_config() {
  require_value "--data-root" "$DATA_ROOT"
  require_value "--output-root" "$OUTPUT_ROOT"

  cd "$IFVC_ROOT"
  mkdir -p .cache/configs

  python - "$DATA_ROOT" "$OUTPUT_ROOT" <<'PY'
import json
import os
import sys

data_root, output_root = sys.argv[1:]
source = "configs/VRU/VRU_gz/VRU_gz_0.004_1.0.json"
with open(source, "r") as f:
    config = json.load(f)

config["base_source_path"] = os.path.join(data_root, "prepared", "VRU", "Dataset")
config["base_model_path"] = output_root
config["frame_start"] = 0
config["frame_end"] = 249
config["Init_frame_params"]["iterations"] = 30000
config["Init_frame_params"]["test_iterations"] = [30000]
config["Init_frame_params"]["save_iterations"] = [30000]
config["I_frame_params"]["iterations"] = 30000
config["I_frame_params"]["test_iterations"] = [30000]
config["I_frame_params"]["save_iterations"] = [30000]
config["P_frame_params"]["iterations"] = 500
config["P_frame_params"]["first_ref_iteration"] = 30000
config["P_frame_params"]["ref_iter"] = 500
config["P_frame_params"]["test_iterations"] = [500]
config["P_frame_params"]["save_iterations"] = [500]

with open(".cache/configs/VRU_gz_full.json", "w") as f:
    json.dump(config, f, indent=2)
PY
}

case "$ACTION" in
  all)
    prepare_colmap
    prepare_weights
    prepare_config
    ;;
  colmap)
    prepare_colmap
    ;;
  weights)
    prepare_weights
    ;;
  config)
    prepare_config
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Use one of: all, colmap, weights, config"
    exit 1
    ;;
esac

