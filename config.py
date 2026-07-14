# -*- coding: utf-8 -*-
"""Jetson Nano 边缘配置。"""
import json
import os
from pathlib import Path

# OpenCV FFmpeg 拉流：单线程解码，降低 libavcodec pthread_frame 崩溃概率
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    os.getenv(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|threads;1|stimeout;5000000",
    ),
)

EDGE_ROOT = Path(__file__).resolve().parent

# NVR IP -> {"username": "...", "password": "..."}；未配置时使用匿名 RTSP
_ALL_DEVICE_RAW = os.getenv("EDGE_ALL_DEVICE", "{}")
try:
    ALL_DEVICE = json.loads(_ALL_DEVICE_RAW)
except json.JSONDecodeError:
    ALL_DEVICE = {}

SAVE_DIR = os.getenv("EDGE_SAVE_DIR", "save_video")
SERVER_IP = os.getenv("EDGE_SERVER_IP", "10.7.5.234")
UPLOAD_URL = os.getenv(
    "EDGE_UPLOAD_URL",
    f"http://{SERVER_IP}:19091/openApi/ai/alarm/push",
)


INIT_STATUS_FILE = os.getenv(
    "EDGE_INIT_STATUS_FILE",
    str(EDGE_ROOT / "init_status.json"),
)
CAMERAS_STATE_FILE = os.getenv(
    "EDGE_CAMERAS_STATE_FILE",
    str(EDGE_ROOT / "edge_settings" / "cameras_state.json"),
)

INFER_DEVICE = os.getenv("NANO_INFER_DEVICE", "0")
IMG_SIZE_LOW = int(os.getenv("NANO_IMG_SIZE_LOW", "640"))
IMG_SIZE_HIGH = int(os.getenv("NANO_IMG_SIZE_HIGH", "1088"))

DEFAULT_CONF = float(os.getenv("EDGE_DEFAULT_CONF", "0.25"))
DEFAULT_NMS = float(os.getenv("EDGE_DEFAULT_NMS", "0.7"))
# 仅作 algorithm_types 未配置 interval 时的 fallback
DEFAULT_DETECT_INTERVAL = float(os.getenv("EDGE_DETECT_INTERVAL", "1.0"))

# 取流后端: opencv（FFmpeg，默认）| gstreamer（Jetson 硬解 pipeline）
#   export EDGE_STREAM_BACKEND=gstreamer
#   export EDGE_GST_CODEC=h264          # 或 h265
#   export EDGE_GST_DECODER=nvv4l2decoder  # Jetson；开发机可设 avdec_h264
STREAM_BACKEND = os.getenv("EDGE_STREAM_BACKEND", "opencv").strip().lower()
GST_CODEC = os.getenv("EDGE_GST_CODEC", "h264").strip().lower()
GST_DECODER = os.getenv("EDGE_GST_DECODER", "nvv4l2decoder").strip()

# 模型路径见 algorithm_registry.json；可选环境变量:
#   EDGE_MODEL_CAR_DETECT 或 NANO_MODEL_CAR（兼容）
#   EDGE_MODEL_BAG_DETECT 或 NANO_MODEL_BAG（兼容）
