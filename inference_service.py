# -*- coding: utf-8 -*-
"""
边缘推理服务 — 对应总项目 inference_service.py 中的拉流、检测、报警流程。

非跟踪类报警: process_non_tracking_alarms
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from functools import partial
from typing import Dict, Optional, Union

import config  # noqa: F401 — 尽早设置 OPENCV_FFMPEG_CAPTURE_OPTIONS

from config import STREAM_BACKEND, UPLOAD_URL
from core.stream_capture import create_rtsp_capture, normalize_stream_backend
from draw_picture import draw_alarm_boxes, draw_bag_alarm_boxes
from inference.shared_engine import SharedDetectEngine
from registry import create_report_policy, get_registry
from report import Async_Alarm
from utils import detections_to_boxes

try:
    from algorithms.bag.report import BagReport
except ImportError:
    BagReport = None  # type: ignore

logger = logging.getLogger(__name__)


class AlgorithmSlot:
    """单路摄像头上的单个算法：Policy + Async_Alarm（对齐总项目 inference_service 报警对象）。"""

    def __init__(self, algo_name: str, params: Optional[Dict] = None, upload_url: str = UPLOAD_URL):
        self.algo_name = algo_name
        self.params = params or {}
        policy = create_report_policy(algo_name)
        if policy is None:
            raise ValueError(f"未注册算法: {algo_name}")
        self.policy = policy
        self.alarm = Async_Alarm(policy, upload_url=upload_url)

    async def ensure_initialized(self):
        if self.alarm.alarm_data is None:
            await self.alarm.initialize()

    def update_params(self, params: Dict):
        self.params = params or {}


class CameraStreamTask:
    """单路 RTSP 检测任务（边缘版 AlgorithmInstance 简化）。

    与 core_code_0526 一致：同一路可绑定多个算法；按 algorithm_type 分引擎推理，
    同 type 多算法共享一次推理结果。
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        algorithms: Dict[str, Dict],
        engines: Union[Dict[str, SharedDetectEngine], SharedDetectEngine],
        upload_url: str = UPLOAD_URL,
        stream_backend: Optional[str] = None,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.stream_backend = normalize_stream_backend(stream_backend or STREAM_BACKEND)
        self.algorithms_cfg = algorithms
        if isinstance(engines, SharedDetectEngine):
            self.engines = {"_legacy": engines}
        else:
            self.engines = dict(engines)
        self.upload_url = upload_url
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self.status = "stopped"
        self.start_time: Optional[str] = None
        self.last_frame_time: Optional[str] = None
        self.last_error: Optional[str] = None
        self.fps: float = 0.0
        self._slots: Dict[str, AlgorithmSlot] = {}
        self._slot_algo_type: Dict[str, str] = {}
        self._last_infer_by_type: Dict[str, float] = {}
        self._rebuild_slots()

    def _rebuild_slot_types(self):
        reg = get_registry()
        self._slot_algo_type = {}
        for name in self._slots:
            algo_type = reg.get_algorithm_type(name)
            if algo_type:
                self._slot_algo_type[name] = algo_type

    def _rebuild_slots(self):
        self._slots.clear()
        for name, params in self.algorithms_cfg.items():
            try:
                self._slots[name] = AlgorithmSlot(name, params, self.upload_url)
            except ValueError as exc:
                logger.warning("[%s] %s", self.camera_id, exc)
        self._rebuild_slot_types()

    def _slots_for_algo_type(self, algo_type: str) -> Dict[str, AlgorithmSlot]:
        return {
            name: slot
            for name, slot in self._slots.items()
            if self._slot_algo_type.get(name) == algo_type
        }

    def _interval_for_algo_type(self, algo_type: str) -> float:
        """同 algorithm_type 共享一次推理；间隔仅来自 registry algorithm_types。"""
        return get_registry().get_interval_for_type(algo_type)

    def get_detect_interval_by_type(self) -> Dict[str, float]:
        """各 algorithm_type → 推理间隔（秒），只读 registry 配置。"""
        return {algo_type: self._interval_for_algo_type(algo_type) for algo_type in self.engines}

    def update_algorithms(self, algorithms: Dict[str, Dict]):
        self.algorithms_cfg = algorithms
        old = set(self._slots.keys())
        new = set(algorithms.keys())
        for name in old - new:
            slot = self._slots.pop(name, None)
            if slot and slot.alarm.alarm_data:
                asyncio.create_task(slot.alarm.close())
        for name, params in algorithms.items():
            if name in self._slots:
                self._slots[name].update_params(params)
            else:
                try:
                    self._slots[name] = AlgorithmSlot(name, params, self.upload_url)
                except ValueError as exc:
                    logger.warning("[%s] %s", self.camera_id, exc)
        self._rebuild_slot_types()

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self.status = "running"
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._task = asyncio.create_task(self._run(), name=f"cam-{self.camera_id}")

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for slot in self._slots.values():
            await slot.alarm.close()
        self.status = "stopped"

    async def process_non_tracking_alarms(
        self,
        detections,
        camera_id: str,
        orig_bgr,
        event_time: datetime,
        *,
        algo_type: str,
        engine: SharedDetectEngine,
    ):
        """与总项目 inference_service.process_non_tracking_alarms 对齐。"""
        slots = self._slots_for_algo_type(algo_type)
        if not slots:
            return

        detections = detections or []
        names = dict(engine.class_names) if engine else {}
        boxes = detections_to_boxes(detections, names)

        loop = asyncio.get_running_loop()
        for task_name, slot in slots.items():
            alarm = slot.alarm
            if not alarm:
                continue
            await slot.ensure_initialized()
            policy = alarm.policy
            if policy.track:
                continue
            parameter = slot.params
            if BagReport is not None and isinstance(policy, BagReport):
                flag = await loop.run_in_executor(
                    None,
                    partial(
                        policy.need_report_from_detections,
                        event_time,
                        detections,
                        parameter,
                        orig_bgr,
                    ),
                )
                if flag:
                    annotated = await loop.run_in_executor(
                        None,
                        partial(draw_bag_alarm_boxes, orig_bgr, detections, parameter),
                    )
                    await alarm.report(event_time, annotated, orig_bgr, camera_id, self.rtsp_url)
                    logger.info("[%s] 触发报警 %s (%s)", camera_id, task_name, algo_type)
                continue

            flag = policy.need_report(event_time, boxes, parameter, names)
            if flag:
                # 与 GPU 版一致：只画触发报警的 label（draw_custom_boxes 风格）
                annotated = await loop.run_in_executor(
                    None,
                    partial(draw_alarm_boxes, orig_bgr, detections, policy.label),
                )
                await alarm.report(event_time, annotated, orig_bgr, camera_id, self.rtsp_url)
                logger.info("[%s] 触发报警 %s (%s)", camera_id, task_name, algo_type)

    async def _run(self):
        capture = create_rtsp_capture(self.rtsp_url, self.stream_backend)
        loop = asyncio.get_running_loop()
        logger.info("[%s] 取流 backend=%s", self.camera_id, self.stream_backend)
        if not await loop.run_in_executor(None, capture.open):
            self.status = "error"
            self.last_error = (
                f"无法打开 RTSP ({self.stream_backend}): {self.rtsp_url}"
            )
            return

        frame_count = 0
        t0 = time.monotonic()
        self._last_infer_by_type.clear()

        try:
            while not self._stop.is_set():
                ok, frame = await loop.run_in_executor(None, capture.read)
                if not ok or frame is None:
                    await asyncio.sleep(3)
                    if self._stop.is_set():
                        break
                    await loop.run_in_executor(None, capture.open)
                    continue

                self.last_frame_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                frame_count += 1
                elapsed = time.monotonic() - t0
                if elapsed >= 5.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    t0 = time.monotonic()

                now = time.monotonic()
                event_time = datetime.now()
                for algo_type, engine in self.engines.items():
                    interval = self._interval_for_algo_type(algo_type)
                    last = self._last_infer_by_type.get(algo_type, 0.0)
                    if interval > 0 and (now - last) < interval:
                        continue
                    self._last_infer_by_type[algo_type] = now

                    detections, _ = await loop.run_in_executor(None, engine.infer_bgr, frame)
                    await self.process_non_tracking_alarms(
                        detections,
                        self.camera_id,
                        frame,
                        event_time,
                        algo_type=algo_type,
                        engine=engine,
                    )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.status = "error"
            self.last_error = str(exc)
            logger.error("[%s] %s", self.camera_id, exc, exc_info=True)
        finally:
            await loop.run_in_executor(None, capture.release)
