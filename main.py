#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano 边缘视觉服务 — 目录与 API 与 edge_bm1688 / core_code_0526 对齐。

  main.py / Manager.py / inference_service.py / report.py / registry.py
  config.py / utils.py / algorithm_registry.json / init_status.json

推理：Ultralytics YOLO，权重路径见 algorithm_registry.json（algorithms/car/weights/car.pt）
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

EDGE_ROOT = Path(__file__).resolve().parent
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

from config import EDGE_ROOT, GST_CODEC, GST_DECODER, INIT_STATUS_FILE, STREAM_BACKEND, UPLOAD_URL  # noqa: E402
from core.api_models import BatchCameraRunning  # noqa: E402
from core.runtime import EdgeRuntime  # noqa: E402
from registry import get_registry, registered_algorithm_names  # noqa: E402
from utils import create_camera_id, create_camera_ip  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

runtime: Optional[EdgeRuntime] = None


def _normalize_resolution_mode(mode: Optional[str]) -> str:
    if isinstance(mode, str) and mode.strip().lower() in ("low", "high"):
        return mode.strip().lower()
    return "low"


def _model_path(algo_type: str, algo_name: str, *env_keys: str) -> str:
    reg = get_registry()
    path, _ = reg.resolve_weight_path(algo_type, "low", EDGE_ROOT, algo_name)
    for key in env_keys:
        override = os.getenv(key)
        if override:
            return override
    return path or ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    for algo_type, algo_name, env_keys, label in (
        ("car_detect", "Car_Detection", ("EDGE_MODEL_CAR_DETECT", "NANO_MODEL_CAR"), "车辆"),
        ("bag_detect", "Bag_Detection", ("EDGE_MODEL_BAG_DETECT", "NANO_MODEL_BAG"), "背包"),
    ):
        model_path = _model_path(algo_type, algo_name, *env_keys)
        if not model_path or not os.path.isfile(model_path):
            logger.warning(
                "%s模型不存在: %s — 服务可启动，检测任务将在模型就绪后加载",
                label,
                model_path,
            )
    runtime = EdgeRuntime(upload_url=UPLOAD_URL)
    await runtime.ensure_started()
    if os.path.exists(INIT_STATUS_FILE):
        logger.info("配置文件: %s", INIT_STATUS_FILE)
    else:
        logger.info(
            "未找到 %s，可由 batch_running 自动生成或复制 init_status.example.json",
            INIT_STATUS_FILE,
        )
    logger.info("vision 已就绪，算法: %s", registered_algorithm_names())
    yield
    if runtime:
        await runtime.shutdown()


app = FastAPI(title="Edge Vision Jetson Nano", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    car_path = _model_path("car_detect", "Car_Detection", "EDGE_MODEL_CAR_DETECT", "NANO_MODEL_CAR")
    bag_path = _model_path("bag_detect", "Bag_Detection", "EDGE_MODEL_BAG_DETECT", "NANO_MODEL_BAG")
    engines = runtime.list_loaded_engines() if runtime else {}
    return {
        "status": "ok",
        "platform": "jetson-nano",
        "stream_backend": STREAM_BACKEND,
        "gst_codec": GST_CODEC,
        "gst_decoder": GST_DECODER,
        "model_car": car_path,
        "model_car_exists": bool(car_path and os.path.isfile(car_path)),
        "model_bag": bag_path,
        "model_bag_exists": bool(bag_path and os.path.isfile(bag_path)),
        "loaded_engines": engines,
        "algorithms": registered_algorithm_names(),
    }


@app.post("/cameras/batch_running")
async def batch_running_cameras(batch: BatchCameraRunning):
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")

    operations = []
    for item in batch.option:
        if not item.algorithms:
            continue
        nvr_ip = item.nvr_ip
        nvr_poe_num = item.camera_id
        resolution_mode = _normalize_resolution_mode(getattr(item, "resolution_mode", None))

        if getattr(item, "camera_ip", None):
            camera_ip = item.camera_ip
        else:
            camera_ip = create_camera_ip(nvr_ip=nvr_ip, nvr_poe_num=str(nvr_poe_num))

        op_map = {0: "remove", 1: "add", 2: "update"}
        for opt in item.algorithms:
            op_type = op_map.get(opt.enabled)
            if op_type is None:
                continue
            operations.append(
                (op_type, nvr_ip, nvr_poe_num, camera_ip, opt.type, opt.params, resolution_mode)
            )

    if not operations:
        return {"total_operations": 0, "results": [], "message": "No valid operations to perform"}

    results = await runtime.batch_operation(operations)
    return {"total_operations": len(operations), "results": results}


@app.get("/cameras/all_running_config")
async def get_all_running_config():
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return runtime.get_all_camera_configs()


@app.post("/cameras/reload_from_disk")
async def reload_from_disk():
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    if not os.path.exists(INIT_STATUS_FILE):
        raise HTTPException(status_code=404, detail="init_status.json not found")
    await runtime.reload_from_disk()
    return {
        "message": "Configuration reloaded from init_status.json successfully",
        "status": "success",
    }


@app.post("/cameras/clear_config")
async def clear_config():
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    await runtime.clear_config_async()
    return {"message": "Configuration cleared successfully", "status": "success"}


@app.get("/cameras/{nvr_ip}/{nvr_poe_num}/algorithms")
async def get_camera_algorithms(nvr_ip: str, nvr_poe_num: int):
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    camera_id = create_camera_id(nvr_poe_num, nvr_ip)
    cfg = runtime.get_all_camera_configs().get(camera_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return {
        "camera_id": camera_id,
        "algorithms": [
            {"name": name, "parameter": params}
            for name, params in cfg.get("algorithms", {}).items()
        ],
    }


@app.get("/cameras/running_status")
async def running_status():
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return {"cameras": runtime.list_running()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8802)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
