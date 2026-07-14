# -*- coding: utf-8 -*-
"""RTSP 取流 — 支持 OpenCV(FFmpeg) 与 GStreamer(Jetson 硬解) 两种后端。"""
from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol, Tuple, runtime_checkable

import cv2
import numpy as np

from config import GST_CODEC, GST_DECODER, STREAM_BACKEND

logger = logging.getLogger(__name__)


@runtime_checkable
class RtspCapture(Protocol):
    def open(self) -> bool: ...

    def read(self) -> Tuple[bool, Optional[np.ndarray]]: ...

    def release(self) -> None: ...


def _escape_gst_url(url: str) -> str:
    return url.replace("\\", "\\\\").replace('"', '\\"')


def build_gstreamer_pipeline(
    rtsp_url: str,
    *,
    codec: str = GST_CODEC,
    decoder: str = GST_DECODER,
) -> str:
    """构建 Jetson 友好的 GStreamer pipeline（供 OpenCV CAP_GSTREAMER 使用）。"""
    url = _escape_gst_url(rtsp_url)
    codec_norm = (codec or "h264").strip().lower()
    if codec_norm in ("h265", "hevc"):
        depay = "rtph265depay ! h265parse !"
    else:
        depay = "rtph264depay ! h264parse !"

    dec = (decoder or "nvv4l2decoder").strip()
    if dec == "nvv4l2decoder":
        convert = "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR !"
    else:
        convert = "videoconvert ! video/x-raw,format=BGR !"

    return (
        f'rtspsrc location="{url}" latency=0 drop-on-latency=true protocols=tcp '
        f"timeout=10000000 retry=3 ! "
        f"{depay} {dec} ! {convert} "
        f"appsink drop=1 max-buffers=2 sync=false"
    )


class OpenCVFFmpegCapture:
    """OpenCV + FFmpeg RTSP（原有方式）。"""

    backend_name = "opencv"

    def __init__(self, url: str):
        self._url = url
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        with self._lock:
            self._release_unlocked()
            self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            ok = self._cap is not None and self._cap.isOpened()
            if ok:
                logger.info("OpenCV(FFmpeg) 已连接: %s", self._url)
            return ok

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            return self._cap.read()

    def release(self) -> None:
        with self._lock:
            self._release_unlocked()

    def _release_unlocked(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class OpenCVGStreamerCapture:
    """OpenCV + GStreamer pipeline（Jetson 推荐 nvv4l2decoder 硬解）。"""

    backend_name = "gstreamer"

    def __init__(self, url: str, *, codec: str = GST_CODEC, decoder: str = GST_DECODER):
        self._url = url
        self._codec = codec
        self._decoder = decoder
        self._pipeline = build_gstreamer_pipeline(url, codec=codec, decoder=decoder)
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        with self._lock:
            self._release_unlocked()
            self._cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
            ok = self._cap is not None and self._cap.isOpened()
            if ok:
                logger.info(
                    "GStreamer 已连接 codec=%s decoder=%s url=%s",
                    self._codec,
                    self._decoder,
                    self._url,
                )
            else:
                logger.warning(
                    "GStreamer 打开失败（请确认 OpenCV 已启用 GStreamer 且插件可用）: %s",
                    self._url,
                )
            return ok

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            return self._cap.read()

    def release(self) -> None:
        with self._lock:
            self._release_unlocked()

    def _release_unlocked(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


def normalize_stream_backend(value: Optional[str]) -> str:
    v = (value or STREAM_BACKEND or "opencv").strip().lower()
    if v in ("gstreamer", "gst", "gs"):
        return "gstreamer"
    return "opencv"


def create_rtsp_capture(
    url: str,
    backend: Optional[str] = None,
    *,
    codec: Optional[str] = None,
    decoder: Optional[str] = None,
) -> RtspCapture:
    """按配置创建 RTSP 捕获实例；backend 未指定时使用 config.STREAM_BACKEND。"""
    name = normalize_stream_backend(backend)
    if name == "gstreamer":
        return OpenCVGStreamerCapture(
            url,
            codec=codec or GST_CODEC,
            decoder=decoder or GST_DECODER,
        )
    return OpenCVFFmpegCapture(url)
