#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

export NANO_INFER_DEVICE="${NANO_INFER_DEVICE:-0}"
MODEL="${EDGE_MODEL_CAR_DETECT:-${NANO_MODEL_CAR:-$(pwd)/algorithms/car/weights/car.pt}}"

if [[ ! -f "$MODEL" ]]; then
  echo "警告: 车辆模型不存在 $MODEL — 服务仍可启动" >&2
fi

python3 main.py --host 0.0.0.0 --port "${EDGE_PORT:-8802}"
