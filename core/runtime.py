# -*- coding: utf-8 -*-
"""Jetson 边缘运行时 — 按算法类型加载 YOLO；权重缺失时不报错。"""
import json
import logging
import os
from typing import Dict, List, Optional

from config import (
    CAMERAS_STATE_FILE,
    DEFAULT_CONF,
    DEFAULT_NMS,
    EDGE_ROOT,
    GST_CODEC,
    GST_DECODER,
    INFER_DEVICE,
    INIT_STATUS_FILE,
    STREAM_BACKEND,
    UPLOAD_URL,
)
from core.state_io import load_init_status, save_init_status
from core.alarm_upload import AlarmUploadService
from inference.shared_engine import SharedDetectEngine
from inference_service import CameraStreamTask
from registry import get_registry

logger = logging.getLogger(__name__)


def _normalize_resolution_mode(value: Optional[str]) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("low", "high"):
            return v
    return "low"


class EdgeRuntime:
    """对应总项目 Manager.AlgorithmManager + 进程内推理。"""

    def __init__(self, upload_url: str = UPLOAD_URL):
        self.upload_url = upload_url
        self._cameras: Dict[str, Dict] = {}
        self._tasks: Dict[str, CameraStreamTask] = {}
        self._engines: Dict[str, SharedDetectEngine] = {}
        self._started = False

    def _resolve_model_path(
        self, algo_type: str, mode: str, algo_name: Optional[str] = None
    ) -> Optional[str]:
        reg = get_registry()
        path, _ = reg.resolve_weight_path(algo_type, mode, EDGE_ROOT, algo_name)
        env_overrides = {
            "car_detect": ("EDGE_MODEL_CAR_DETECT", "NANO_MODEL_CAR"),
            "bag_detect": ("EDGE_MODEL_BAG_DETECT", "NANO_MODEL_BAG"),
        }
        for key, (primary, legacy) in env_overrides.items():
            if algo_type == key:
                override = os.getenv(primary) or os.getenv(legacy)
                if override:
                    return override
                break
        return path

    def _engine_cache_key(self, algo_type: str, mode: str) -> str:
        return f"{algo_type}:{_normalize_resolution_mode(mode)}"

    def _get_engine_for_type(
        self,
        algo_type: str,
        resolution_mode: Optional[str],
        algo_name: Optional[str] = None,
    ) -> Optional[SharedDetectEngine]:
        mode = _normalize_resolution_mode(resolution_mode)
        key = self._engine_cache_key(algo_type, mode)
        if key in self._engines:
            return self._engines[key]

        reg = get_registry()
        _, imgsz = reg.resolve_weight_path(algo_type, mode, EDGE_ROOT, algo_name)
        path = self._resolve_model_path(algo_type, mode, algo_name)
        if not path:
            logger.warning("算法类型 %s 未配置权重路径", algo_type)
            return None

        engine = SharedDetectEngine.try_get(
            path, imgsz, DEFAULT_CONF, DEFAULT_NMS, INFER_DEVICE
        )
        if engine is not None:
            self._engines[key] = engine
            logger.info("已加载引擎 type=%s path=%s imgsz=%s", algo_type, path, imgsz)
        return engine

    def _collect_algo_types(self, algorithms: Dict[str, Dict]) -> List[str]:
        """与 core_code_0526 _organize_algorithm_configs 一致：按 algorithm_type 去重保序。"""
        reg = get_registry()
        types: List[str] = []
        seen = set()
        for name in algorithms:
            algo_type = reg.get_algorithm_type(name)
            if algo_type and algo_type not in seen:
                seen.add(algo_type)
                types.append(algo_type)
        return types

    def _pick_primary_algo_name(self, algorithms: Dict[str, Dict], algo_type: str) -> Optional[str]:
        reg = get_registry()
        for name in algorithms:
            if reg.get_algorithm_type(name) == algo_type:
                return name
        return None

    def _resolve_engines_for_algorithms(
        self,
        algorithms: Dict[str, Dict],
        resolution_mode: str,
    ) -> Dict[str, SharedDetectEngine]:
        """为一路摄像头上的各 algorithm_type 解析引擎（同 type 多算法共享一个引擎）。"""
        mode = _normalize_resolution_mode(resolution_mode)
        engines: Dict[str, SharedDetectEngine] = {}
        for algo_type in self._collect_algo_types(algorithms):
            primary_name = self._pick_primary_algo_name(algorithms, algo_type)
            engine = self._get_engine_for_type(algo_type, mode, primary_name)
            if engine is not None:
                engines[algo_type] = engine
        return engines

    @staticmethod
    def _engine_map_signature(engines: Dict[str, SharedDetectEngine]) -> Dict[str, str]:
        return {algo_type: engine.weights_path for algo_type, engine in engines.items()}

    def list_loaded_engines(self) -> Dict[str, str]:
        return {k: v.weights_path for k, v in self._engines.items()}

    async def ensure_started(self):
        if self._started:
            return
        await AlarmUploadService.get(self.upload_url).start()
        self._load_state()
        await self._restore_tasks()
        self._started = True

    async def shutdown(self):
        for t in list(self._tasks.values()):
            await t.stop()
        self._tasks.clear()
        self._engines.clear()
        await AlarmUploadService.get(self.upload_url).stop()
        self._started = False

    def _load_state(self):
        if os.path.exists(INIT_STATUS_FILE):
            self._cameras = load_init_status(INIT_STATUS_FILE)
            logger.info("已从 init_status.json 加载 %d 路", len(self._cameras))
            return
        if os.path.exists(CAMERAS_STATE_FILE):
            try:
                with open(CAMERAS_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                self._cameras = data.get("cameras", data)
            except Exception as exc:
                logger.error("加载 cameras_state.json 失败: %s", exc)

    def _save_state(self):
        save_init_status(self._cameras, INIT_STATUS_FILE)

    async def _restore_tasks(self):
        for cid in list(self._cameras.keys()):
            try:
                await self._sync_camera_task(cid)
            except Exception as exc:
                logger.error("恢复 %s 失败: %s", cid, exc)

    async def reload_from_disk(self):
        cameras = load_init_status(INIT_STATUS_FILE)
        for t in list(self._tasks.values()):
            await t.stop()
        self._tasks.clear()
        self._cameras = cameras
        await self._restore_tasks()

    def get_all_camera_configs(self) -> Dict:
        return dict(self._cameras)

    async def clear_config_async(self):
        for t in list(self._tasks.values()):
            await t.stop()
        self._tasks.clear()
        self._cameras.clear()
        self._save_state()

    def _ensure_camera_entry(
        self,
        camera_id: str,
        camera_ip: str,
        resolution_mode: str,
        nvr_ip: str = "",
        channel: int = 0,
    ):
        if camera_id not in self._cameras:
            self._cameras[camera_id] = {
                "camera_ip": camera_ip,
                "nvr_ip": nvr_ip,
                "channel": channel,
                "resolution_mode": resolution_mode,
                "algorithms": {},
            }

    async def batch_operation(self, operations: List[tuple]) -> List[Dict]:
        await self.ensure_started()
        results = []
        for op_type, nvr_ip, channel, camera_ip, algo_name, params, resolution_mode in operations:
            from utils import create_camera_id

            camera_id = create_camera_id(channel, nvr_ip)
            mode = _normalize_resolution_mode(resolution_mode)
            try:
                if op_type == "remove":
                    await self._remove_algo(camera_id, algo_name)
                    results.append({"camera_id": camera_id, "algorithm": algo_name, "status": "removed"})
                elif op_type in ("add", "update"):
                    reg = get_registry()
                    if reg.get_algorithm_type(algo_name) is None:
                        logger.warning("未注册算法 %s，跳过", algo_name)
                        results.append(
                            {
                                "camera_id": camera_id,
                                "algorithm": algo_name,
                                "status": "skipped",
                                "message": "algorithm not registered",
                            }
                        )
                        continue
                    self._ensure_camera_entry(
                        camera_id,
                        camera_ip,
                        mode,
                        nvr_ip=nvr_ip,
                        channel=int(channel),
                    )
                    self._cameras[camera_id]["camera_ip"] = camera_ip
                    self._cameras[camera_id]["nvr_ip"] = nvr_ip
                    self._cameras[camera_id]["channel"] = int(channel)
                    self._cameras[camera_id]["resolution_mode"] = mode
                    self._cameras[camera_id]["algorithms"][algo_name] = params
                    await self._sync_camera_task(camera_id)
                    results.append({"camera_id": camera_id, "algorithm": algo_name, "status": op_type})
            except Exception as exc:
                results.append(
                    {
                        "camera_id": camera_id,
                        "algorithm": algo_name,
                        "status": "error",
                        "message": str(exc),
                    }
                )
        self._save_state()
        return results

    async def _remove_algo(self, camera_id: str, algo_name: str):
        if camera_id not in self._cameras:
            return
        self._cameras[camera_id].get("algorithms", {}).pop(algo_name, None)
        if not self._cameras[camera_id].get("algorithms"):
            self._cameras.pop(camera_id, None)
            if camera_id in self._tasks:
                await self._tasks[camera_id].stop()
                self._tasks.pop(camera_id, None)
        else:
            await self._sync_camera_task(camera_id)

    async def _sync_camera_task(self, camera_id: str):
        cfg = self._cameras.get(camera_id)
        if not cfg or not cfg.get("algorithms"):
            if camera_id in self._tasks:
                await self._tasks[camera_id].stop()
                del self._tasks[camera_id]
            return

        rtsp = cfg["camera_ip"]
        algos = cfg["algorithms"]
        mode = cfg.get("resolution_mode", "low")
        algo_types = self._collect_algo_types(algos)
        if not algo_types:
            logger.warning("[%s] 无已注册算法，跳过检测任务", camera_id)
            return

        engines = self._resolve_engines_for_algorithms(algos, mode)
        missing_types = [t for t in algo_types if t not in engines]
        if missing_types:
            logger.warning(
                "[%s] 部分 algorithm_type 模型未就绪: %s",
                camera_id,
                missing_types,
            )
        if not engines:
            logger.warning("[%s] 无可用推理引擎，跳过检测任务", camera_id)
            if camera_id in self._tasks:
                await self._tasks[camera_id].stop()
                del self._tasks[camera_id]
            return

        engine_sig = self._engine_map_signature(engines)
        if camera_id in self._tasks:
            task = self._tasks[camera_id]
            need_recreate = (
                task.rtsp_url != rtsp
                or self._engine_map_signature(task.engines) != engine_sig
            )
            if need_recreate:
                await task.stop()
                del self._tasks[camera_id]
            else:
                task.update_algorithms(algos)
                if task.status != "running":
                    await task.start()
                return

        task = CameraStreamTask(
            camera_id=camera_id,
            rtsp_url=rtsp,
            algorithms=algos,
            engines=engines,
            upload_url=self.upload_url,
        )
        self._tasks[camera_id] = task
        await task.start()

    def get_status(self, camera_id: str) -> Optional[Dict]:
        t = self._tasks.get(camera_id)
        cfg = self._cameras.get(camera_id, {})
        if not t:
            return {"camera_id": camera_id, "status": "stopped"}
        return {
            "camera_id": camera_id,
            "status": t.status,
            "rtsp_url": t.rtsp_url,
            "stream_backend": t.stream_backend,
            "resolution_mode": cfg.get("resolution_mode", "low"),
            "algorithms": list(t._slots.keys()),
            "algorithm_types": list(t.engines.keys()),
            "models": {algo_type: eng.weights_path for algo_type, eng in t.engines.items()},
            "model": next(iter(t.engines.values())).weights_path if t.engines else None,
            "detect_interval_by_type": t.get_detect_interval_by_type(),
            "start_time": t.start_time,
            "last_frame_time": t.last_frame_time,
            "last_error": t.last_error,
            "fps": round(t.fps, 2),
        }

    def list_running(self) -> List[Dict]:
        return [self.get_status(cid) for cid in self._tasks.keys()]

    def get_stream_config(self) -> Dict:
        return {
            "stream_backend": STREAM_BACKEND,
            "gst_codec": GST_CODEC,
            "gst_decoder": GST_DECODER,
        }
