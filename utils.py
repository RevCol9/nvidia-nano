# -*- coding: utf-8 -*-
"""与总项目 utils.py 对齐的摄像头工具 + 检测框转换。"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ALL_DEVICE

# 默认类别表（实际推理以 engine.class_names 为准）
CLASS_NAMES = ["car"]
NAMES: Dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES)}


def create_camera_id(num: int, nvr_ip: str) -> str:
    return f"{nvr_ip}_{num}"


def parse_camera_id(camera_id: str) -> Tuple[int, str]:
    if not camera_id or "_" not in camera_id:
        raise ValueError(f"invalid camera_id: {camera_id}")
    nvr_ip, channel = camera_id.rsplit("_", 1)
    return int(channel), nvr_ip


def create_camera_ip(
    nvr_ip: str,
    nvr_poe_num: str,
    mid_fix: str = "/Streaming/Channels/",
) -> str:
    if nvr_ip not in ALL_DEVICE:
        return f"rtsp://{nvr_ip}:554{mid_fix}{nvr_poe_num}01"
    cred = ALL_DEVICE[nvr_ip]
    return (
        f"rtsp://{cred['username']}:{cred['password']}@{nvr_ip}:554"
        f"{mid_fix}{nvr_poe_num}01"
    )


def detections_to_boxes(
    detections: List[Dict], names: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """将检测列表转为 report.Policy.need_report 所需的 boxes 数组。"""
    if not detections:
        return np.zeros((0, 6))
    name_to_id = {v: k for k, v in (names or NAMES).items()}
    rows = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls_id = d.get("class_id")
        if cls_id is None:
            cls_id = name_to_id.get(d.get("class_name", ""), -1)
        if cls_id is None or int(cls_id) < 0:
            continue
        rows.append([x1, y1, x2, y2, float(d["score"]), int(cls_id)])
    if not rows:
        return np.zeros((0, 6))
    return np.array(rows, dtype=np.float32)
