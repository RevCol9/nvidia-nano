# -*- coding: utf-8 -*-
"""
报警基础策略与异步上报 — 与总项目 core_code_0526/report.py 对齐。

- Policy / Async_Alarm：根目录公共能力
- SmokeReport / FlameReport / …：各自 algorithms/<name>/report.py
"""
from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import UPLOAD_URL
from core.alarm_upload import AlarmUploadService
from utils import parse_camera_id

logger = logging.getLogger(__name__)


class Policy:
    """与总项目 report.Policy 一致。"""

    def __init__(
        self,
        algo_type: str = "",
        description: str = "",
        label: str = "",
        times: int = 300,
        required_frames: int = 10,
        continue_time_delta: float = 2.0,
    ):
        self.type = algo_type
        self.description = description
        self.label = label
        self.delta_time = times
        self.required_frames = required_frames
        self.continue_time_delta = continue_time_delta
        self.pre_report_time = datetime.now() - timedelta(seconds=self.delta_time + 1)
        self.pre_detect_time = datetime.now() - timedelta(seconds=self.delta_time + 1)
        self.continue_frequence = 0
        self.track = False

    def tick_alarm(self, event_time: datetime, detect_flag: bool) -> bool:
        """连续帧计数 + 冷却；detect_flag 为本帧是否命中业务条件。"""
        if not detect_flag:
            self.continue_frequence = max(0, self.continue_frequence - 1)
            return False

        delta = (event_time - self.pre_detect_time).total_seconds()
        self.pre_detect_time = event_time
        if delta <= self.continue_time_delta:
            self.continue_frequence += 1
        else:
            self.continue_frequence = 1

        if (
            self.continue_frequence >= self.required_frames
            and (event_time - self.pre_report_time).total_seconds() >= self.delta_time
        ):
            self.continue_frequence = 0
            self.pre_report_time = event_time
            return True
        return False

    def need_report(self, event_time: datetime, boxes, parameter: dict, names: dict) -> bool:
        if not isinstance(boxes, np.ndarray):
            boxes = getattr(boxes, "data", boxes)
        if boxes is None or len(boxes) == 0:
            detect_flag = False
        else:
            detect_flag = False
            for index in range(boxes.shape[0]):
                name = names[int(boxes[index][-1])]
                if name == self.label:
                    if "conf" in parameter and parameter["conf"] < boxes[index][-2]:
                        detect_flag = True
                    elif "conf" not in parameter:
                        detect_flag = True

        return self.tick_alarm(event_time, detect_flag)

    def clean_stale_tracks(self, max_age=30):
        pass


class Async_Alarm:
    """与总项目 report.Async_Alarm 一致：队列缓存 + 异步 HTTP。"""

    def __init__(
        self,
        alarm_policy: Policy,
        upload_url: str = UPLOAD_URL,
        cache_len: int = 150,
        cache_flush_interval: float = 10.0,
    ):
        self.upload_url = upload_url
        self.policy = alarm_policy
        self.cache_len = cache_len
        self.cache_flush_interval = max(0.0, float(cache_flush_interval))
        self.lock: Optional[asyncio.Lock] = None
        self.alarm_data: Optional[asyncio.Queue] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._cache_flush_task: Optional[asyncio.Task] = None

    async def initialize(self):
        self.lock = asyncio.Lock()
        self.alarm_data = asyncio.Queue(maxsize=self.cache_len)
        self._stop_event = asyncio.Event()
        if self.cache_flush_interval > 0:
            self._cache_flush_task = asyncio.create_task(self._cache_flush_loop())

    async def close(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._cache_flush_task is not None:
            self._cache_flush_task.cancel()
            try:
                await self._cache_flush_task
            except asyncio.CancelledError:
                pass
            self._cache_flush_task = None

    @staticmethod
    def numpy_to_base64(image_array: np.ndarray, fmt: str = ".jpg") -> str:
        if not isinstance(image_array, np.ndarray) or image_array.size == 0:
            return ""
        if fmt == ".jpg":
            ok, buf = cv2.imencode(".jpg", image_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            ok, buf = cv2.imencode(".png", image_array)
        return base64.b64encode(buf).decode("utf-8") if ok else ""

    async def _cache_flush_loop(self):
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.cache_flush_interval)
                if self.lock is None:
                    continue
                async with self.lock:
                    await self.process_cache()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Cache flush loop error: %s", exc)

    async def report(
        self,
        data_time: datetime,
        image: np.ndarray,
        orig_image: np.ndarray,
        camera_id: str,
        camera_ip: str,
    ):
        report_image = await self.numpy_to_base64_async(image)
        orig_image_base64 = await self.numpy_to_base64_async(orig_image)
        payload = self.prepare_payload(
            report_image, orig_image_base64, camera_id, camera_ip, data_time
        )
        await self.upload_photo_cache(payload, camera_id)

    async def numpy_to_base64_async(self, image_array: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.numpy_to_base64, image_array)

    async def upload_photo_cache(self, payload, camera_index):
        asyncio.create_task(self._upload_photo_cache_background(payload, camera_index))

    async def _upload_photo_cache_background(self, payload, camera_index):
        try:
            success = await self._upload_payload(payload, camera_index)
            if success:
                await self.process_cache()
            else:
                await self.add_to_cache(payload, camera_index)
        except Exception as exc:
            logger.error("Upload task failed for Camera %s: %s", camera_index, exc)
            try:
                await self.add_to_cache(payload, camera_index)
            except Exception:
                logger.exception("Failed to enqueue payload for Camera %s", camera_index)

    async def _upload_payload(self, payload, camera_index) -> bool:
        uploader = AlarmUploadService.get(self.upload_url)
        return await uploader.upload(payload, camera_index)

    async def add_to_cache(self, payload, camera_index):
        try:
            self.alarm_data.put_nowait((payload, camera_index))
            logger.info("Added to cache: Camera %s", camera_index)
        except asyncio.QueueFull:
            logger.warning("Cache full, replacing oldest entry")
            try:
                await asyncio.wait_for(self.alarm_data.get(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            try:
                self.alarm_data.put_nowait((payload, camera_index))
            except asyncio.QueueFull:
                logger.error("Failed to add Camera %s to cache", camera_index)

    async def process_cache(self):
        if self.alarm_data is None:
            return
        while not self.alarm_data.empty():
            try:
                payload, camera_index = self.alarm_data.get_nowait()
            except asyncio.QueueEmpty:
                break
            if await self._upload_payload(payload, camera_index):
                continue
            self.alarm_data.put_nowait((payload, camera_index))
            break

    def prepare_payload(
        self,
        image_base64: str,
        orig_image_base64: str,
        camera_index: str,
        camera_ip: str,
        data_time: datetime,
    ):
        timestamp_str = data_time.strftime("%Y%m%d_%H%M%S")
        nvr_poe_num, nvr_ip = parse_camera_id(camera_index)
        video_path = str(
            Path(f"save_video/{nvr_ip}/camera_{nvr_poe_num}/{timestamp_str}_{self.policy.label}.mp4").resolve()
        )
        return {
            "alarmName": f"Camera {camera_index} Alarm",
            "description": f"This is a alarm triggered by camera {self.policy.description}.",
            "area": f"{camera_index}",
            "grade": 1,
            "priority": 1,
            "cameraIp": camera_ip,
            "traindata": orig_image_base64,
            "equipmentCode": f"Camera {camera_index}",
            "alarmPicture": image_base64,
            "alarmVideo": video_path,
            "alarmDate": data_time.isoformat(),
            "typeName": self.policy.description,
            "alarmInfo": {},
        }
