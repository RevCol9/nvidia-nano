# -*- coding: utf-8 -*-
"""全进程共享 multi_detect YOLO 引擎（Jetson / CUDA），接口与 edge_bm1688 SharedDetectEngine 一致。"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SharedDetectEngine:
    """按 (权重路径, imgsz, device) 缓存单例；infer_bgr 返回与 BM1688 相同的 detection 列表。"""

    _instances: Dict[tuple, "SharedDetectEngine"] = {}
    _lock_init = threading.Lock()

    @classmethod
    def try_get(
        cls,
        weights_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        nms: float = 0.7,
        device: str = "0",
    ) -> Optional["SharedDetectEngine"]:
        """权重不存在时返回 None，不抛异常。"""
        if not Path(weights_path).is_file():
            logger.warning("权重不存在，跳过加载: %s", weights_path)
            return None
        return cls.get(weights_path, imgsz, conf, nms, device)

    @classmethod
    def get(
        cls,
        weights_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        nms: float = 0.7,
        device: str = "0",
    ) -> "SharedDetectEngine":
        key = (str(weights_path), int(imgsz), str(device), float(conf), float(nms))
        with cls._lock_init:
            if key not in cls._instances:
                cls._instances[key] = cls(weights_path, imgsz, conf, nms, device)
            return cls._instances[key]

    def __init__(
        self,
        weights_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        nms: float = 0.7,
        device: str = "0",
    ):
        from ultralytics import YOLO

        self.weights_path = weights_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(nms)
        self.device = device
        self._infer_lock = threading.Lock()
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"权重不存在: {weights_path}")
        self.model = YOLO(weights_path)
        self.class_names: Dict[int, str] = dict(self.model.names or {})
        logger.info(
            "SharedDetectEngine 已加载 %s imgsz=%s device=%s",
            weights_path,
            self.imgsz,
            self.device,
        )

    def infer_bgr(self, frame_bgr: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        with self._infer_lock:
            results = self.model.predict(
                source=frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        r = results[0]
        detections: List[Dict] = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            names = r.names
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].detach().cpu().numpy()
                cls_id = int(boxes.cls[i].item())
                score = float(boxes.conf[i].item())
                name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": name,
                        "score": score,
                        "bbox": [
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ],
                    }
                )
        return detections, frame_bgr.copy()
