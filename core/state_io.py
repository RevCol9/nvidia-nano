# -*- coding: utf-8 -*-
"""init_status.json 读写 — 对应总项目 sequence.py Serialize/Deserialization。"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from config import INIT_STATUS_FILE

logger = logging.getLogger(__name__)


def Serialize(camera_algorithms: dict, path: str = None):
    """与总项目 sequence.Serialize 一致。"""
    save_init_status(camera_algorithms, path or INIT_STATUS_FILE)


def Deserialization(path: str = None) -> dict:
    """与总项目 sequence.Deserialization 一致。"""
    return load_init_status(path or INIT_STATUS_FILE)


def load_init_status(path: str = None) -> Dict[str, Dict]:
    file_path = Path(path or INIT_STATUS_FILE)
    if not file_path.is_file():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception as exc:
        logger.error("读取 %s 失败: %s", file_path, exc)
        return {}
    if isinstance(data, dict) and "cameras" in data and len(data) == 1:
        return dict(data["cameras"])
    return dict(data) if isinstance(data, dict) else {}


def save_init_status(cameras: Dict[str, Dict], path: str = None) -> None:
    file_path = Path(path or INIT_STATUS_FILE)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cameras, f, ensure_ascii=False, indent=4)


def deserialization_status(camera_status: Dict[str, Dict]) -> List[Tuple]:
    from utils import parse_camera_id

    operations = []
    for camera_id, cfg in camera_status.items():
        if not isinstance(cfg, dict):
            continue
        try:
            poe_num, nvr_ip = parse_camera_id(camera_id)
        except ValueError:
            nvr_ip = cfg.get("nvr_ip")
            channel = cfg.get("channel")
            if not nvr_ip or channel is None:
                continue
            poe_num = int(channel)
        camera_ip = cfg.get("camera_ip", "")
        resolution_mode = cfg.get("resolution_mode", "low")
        if isinstance(resolution_mode, str):
            resolution_mode = resolution_mode.strip().lower()
            if resolution_mode not in ("low", "high"):
                resolution_mode = "low"
        else:
            resolution_mode = "low"
        for algo_name, params in (cfg.get("algorithms") or {}).items():
            operations.append(
                ("add", nvr_ip, poe_num, camera_ip, algo_name, params or {}, resolution_mode)
            )
    return operations
