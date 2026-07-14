#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地单帧推理自检（Car_Detection / algorithms/car/weights/car.pt）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from config import INFER_DEVICE  # noqa: E402
from inference.shared_engine import SharedDetectEngine  # noqa: E402
from registry import get_registry  # noqa: E402


def main():
    reg = get_registry()
    path, imgsz = reg.resolve_weight_path("car_detect", "low", ROOT, "Car_Detection")
    engine = SharedDetectEngine.try_get(path, imgsz, 0.25, 0.7, INFER_DEVICE)
    if engine is None:
        print(f"模型未加载: {path}")
        return 1
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets, _ = engine.infer_bgr(frame)
    print(f"model={path} device={INFER_DEVICE} classes={engine.class_names}")
    print(f"detections={len(dets)}")
    for d in dets[:5]:
        print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
