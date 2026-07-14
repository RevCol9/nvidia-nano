# -*- coding: utf-8 -*-
"""Depth Anything V2 Small — 仅在有行李检测时按需推理（单例、懒加载）。"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from config import EDGE_ROOT, INFER_DEVICE

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SUBDIR = Path("algorithms/bag/weights/depth/Depth-Anything-V2-Small-hf")
HF_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def resolve_depth_model_source() -> Optional[str]:
    for key in ("BAG_DEPTH_MODEL_DIR", "EDGE_BAG_DEPTH_MODEL"):
        override = os.getenv(key)
        if override:
            p = Path(override)
            if p.is_dir() and (p / "config.json").is_file():
                return str(p.resolve())
            logger.warning("深度模型路径无效: %s", override)
    local = EDGE_ROOT / DEFAULT_MODEL_SUBDIR
    if local.is_dir() and (local / "config.json").is_file():
        return str(local.resolve())
    return None


def _pick_torch_device(requested: str | None = None) -> torch.device:
    req = (requested or INFER_DEVICE or "auto").strip().lower()
    if req in ("cpu", "-1"):
        return torch.device("cpu")
    if req in ("auto", "cuda", "gpu", "0", "cuda:0") and torch.cuda.is_available():
        return torch.device("cuda")
    if req.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{req}")
    return torch.device("cpu")


def _max_side_from_params(parameter: dict | None, default: int) -> int:
    if parameter and "depth_max_side" in parameter:
        return max(32, int(parameter["depth_max_side"]))
    return default


def _half_from_params(parameter: dict | None) -> bool:
    if parameter and "depth_half" in parameter:
        return bool(parameter["depth_half"])
    return False


def _resize_for_infer(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    orig_h, orig_w = bgr.shape[:2]
    if max_side <= 0 or max(orig_h, orig_w) <= max_side:
        return bgr, (orig_h, orig_w)
    scale = max_side / float(max(orig_h, orig_w))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), (orig_h, orig_w)


class BagDepthEngine:
    """全进程单例；模型目录不存在时 try_get 返回 None。"""

    _instance: Optional["BagDepthEngine"] = None
    _lock = threading.Lock()

    @classmethod
    def try_get(cls, parameter: dict | None = None) -> Optional["BagDepthEngine"]:
        source = resolve_depth_model_source()
        if not source:
            return None
        with cls._lock:
            if cls._instance is None:
                try:
                    cls._instance = cls(source)
                except Exception as exc:
                    logger.error("加载深度模型失败: %s", exc)
                    return None
            return cls._instance

    def __init__(self, model_source: str):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.model_source = model_source
        self.device = _pick_torch_device()
        self._infer_lock = threading.Lock()
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_source, use_fast=True)
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(model_source)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()
        logger.info("BagDepthEngine 已加载 %s device=%s", model_source, self.device)

    @torch.inference_mode()
    def infer_bgr(self, frame_bgr: np.ndarray, parameter: dict | None = None) -> np.ndarray:
        from algorithms.bag import config as C

        max_side = _max_side_from_params(parameter, C.DEFAULT_DEPTH_MAX_SIDE)
        use_half = _half_from_params(parameter) and self.device.type == "cuda"

        infer_bgr, (orig_h, orig_w) = _resize_for_infer(frame_bgr, max_side)
        rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if use_half:
            inputs = {
                k: v.half() if v.is_floating_point() else v
                for k, v in inputs.items()
            }

        with self._infer_lock:
            outputs = self.model(**inputs)

        depth = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        return depth.float().cpu().numpy().astype(np.float32)
