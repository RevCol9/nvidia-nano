# -*- coding: utf-8 -*-
"""无人行李后处理：有行李时才跑深度 → 同深度的人 → 扩框 IOU。"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from algorithms.bag import config as C

logger = logging.getLogger(__name__)

LUGGAGE_CLASSES = frozenset(C.LUGGAGE_CLASSES)
PERSON_CLASS = C.PERSON_CLASS
DRAW_CLASSES = frozenset(C.DRAW_CLASSES)

_depth_missing_logged = False


def _conf_threshold(parameter: dict | None) -> float:
    if parameter and "conf" in parameter:
        return float(parameter["conf"])
    return C.DEFAULT_CONF


def _iou_threshold(parameter: dict | None) -> float:
    if parameter and "iou_thresh" in parameter:
        return float(parameter["iou_thresh"])
    return C.DEFAULT_IOU_THRESHOLD


def _expand_scale(parameter: dict | None) -> float:
    if parameter and "expand_scale" in parameter:
        return float(parameter["expand_scale"])
    return C.BOX_EXPAND_SCALE


def _depth_diff_threshold(parameter: dict | None) -> float:
    if parameter and "depth_diff_thresh" in parameter:
        return float(parameter["depth_diff_thresh"])
    return C.DEFAULT_DEPTH_DIFF_THRESHOLD


def _use_depth(parameter: dict | None) -> bool:
    if parameter and "use_depth" in parameter:
        return bool(parameter["use_depth"])
    for key in ("EDGE_BAG_USE_DEPTH", "BAG_USE_DEPTH"):
        env = os.getenv(key, "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
    return C.DEFAULT_USE_DEPTH


def expand_bbox(bbox: Sequence[float], scale: float = C.BOX_EXPAND_SCALE) -> List[float]:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _passes_conf(score: float, conf: float) -> bool:
    return score > conf


def split_detections(
    detections: Sequence[Dict],
    conf: float,
) -> tuple[List[Dict], List[Dict]]:
    luggage: List[Dict] = []
    persons: List[Dict] = []
    for det in detections or []:
        name = det.get("class_name", "")
        score = float(det.get("score", 0.0))
        if not _passes_conf(score, conf):
            continue
        if name in LUGGAGE_CLASSES:
            luggage.append(det)
        elif name == PERSON_CLASS:
            persons.append(det)
    return luggage, persons


def has_luggage_candidates(
    detections: Sequence[Dict],
    parameter: dict | None = None,
) -> bool:
    """是否存在需关注的行李（用于跳过深度推理的快速判断）。"""
    conf = _conf_threshold(parameter)
    luggage, _ = split_detections(detections, conf)
    return bool(luggage)


def sample_bbox_depth(depth_map: np.ndarray, bbox: Sequence[float]) -> float:
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx = int(max(0, min(w - 1, round((x1 + x2) * 0.5))))
    cy = int(max(0, min(h - 1, round((y1 + y2) * 0.5))))
    return float(depth_map[cy, cx])


def filter_persons_same_depth(
    luggage: Dict,
    persons: Sequence[Dict],
    depth_map: np.ndarray,
    depth_diff_thresh: float,
) -> List[Dict]:
    """只保留与行李处于相近相对深度的人。"""
    lug_depth = sample_bbox_depth(depth_map, luggage["bbox"])
    matched: List[Dict] = []
    for person in persons:
        p_depth = sample_bbox_depth(depth_map, person["bbox"])
        if abs(p_depth - lug_depth) <= depth_diff_thresh:
            matched.append(person)
    return matched


def _infer_depth_map(frame_bgr: np.ndarray, parameter: dict | None) -> Optional[np.ndarray]:
    global _depth_missing_logged
    from algorithms.bag.depth_engine import BagDepthEngine

    engine = BagDepthEngine.try_get(parameter)
    if engine is None:
        if not _depth_missing_logged:
            logger.warning(
                "未找到深度模型目录，Bag 将退回纯 2D IOU。"
                "请放置到 algorithms/bag/weights/depth/Depth-Anything-V2-Small-hf "
                "或设置 BAG_DEPTH_MODEL_DIR"
            )
            _depth_missing_logged = True
        return None
    try:
        return engine.infer_bgr(frame_bgr, parameter)
    except Exception as exc:
        logger.error("深度推理失败: %s", exc)
        return None


def find_unattended_luggage(
    detections: Sequence[Dict],
    *,
    conf: float | None = None,
    iou_threshold: float | None = None,
    expand_scale: float | None = None,
    parameter: dict | None = None,
    frame_bgr: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    无人行李判定：
    1. 无 backpack/suitcase → 不跑深度，直接返回 []
    2. 有行李 → （可选）深度图，仅保留同深度的人
    3. 扩框后 IOU 仍低于阈值 → 无人看管
    """
    conf = C.DEFAULT_CONF if conf is None else conf
    iou_threshold = C.DEFAULT_IOU_THRESHOLD if iou_threshold is None else iou_threshold
    expand_scale = C.BOX_EXPAND_SCALE if expand_scale is None else expand_scale
    if parameter is not None:
        conf = _conf_threshold(parameter)
        iou_threshold = _iou_threshold(parameter)
        expand_scale = _expand_scale(parameter)

    luggage, persons = split_detections(detections, conf)
    if not luggage:
        return []

    use_depth = _use_depth(parameter)
    if depth_map is None and use_depth and frame_bgr is not None and frame_bgr.size > 0:
        depth_map = _infer_depth_map(frame_bgr, parameter)

    depth_diff = _depth_diff_threshold(parameter)
    unattended: List[Dict] = []
    for lug in luggage:
        relevant_persons = list(persons)
        if depth_map is not None:
            relevant_persons = filter_persons_same_depth(
                lug, persons, depth_map, depth_diff
            )
        expanded_persons = [expand_bbox(p["bbox"], expand_scale) for p in relevant_persons]
        exp_lug = expand_bbox(lug["bbox"], expand_scale)
        max_iou = 0.0
        for person_box in expanded_persons:
            max_iou = max(max_iou, bbox_iou(exp_lug, person_box))
        if max_iou < iou_threshold:
            unattended.append(lug)
    return unattended


def frame_has_unattended_luggage(
    detections: Sequence[Dict],
    parameter: dict | None = None,
    frame_bgr: Optional[np.ndarray] = None,
) -> bool:
    return bool(find_unattended_luggage(detections, parameter=parameter, frame_bgr=frame_bgr))


def filter_alarm_draw_detections(
    detections: Sequence[Dict],
    parameter: dict | None = None,
) -> List[Dict]:
    conf = _conf_threshold(parameter)
    out: List[Dict] = []
    for det in detections or []:
        name = det.get("class_name", "")
        if name not in DRAW_CLASSES:
            continue
        if not _passes_conf(float(det.get("score", 0.0)), conf):
            continue
        out.append(det)
    return out
