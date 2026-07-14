# -*- coding: utf-8 -*-
"""
报警/检测可视化 — 与总项目 draw_picture.py 的 draw_custom_boxes 对齐。

- 半透明填充 + 四角描边
- 中文标签（label_map）
- 自适应字号与标签背景
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EDGE_ROOT = Path(__file__).resolve().parent

# 与总项目 draw_picture.label_map 一致
label_map = {
    "smoke": "烟雾",
    "flame": "火焰",
    "fire_extinguisher": "灭火器",
    "person": "行人",
    "head_without_helmet": "未戴安全帽",
    "head_with_helmet": "穿戴安全帽",
    "car": "车辆",
    "scooter": "车辆",
    "bag": "背包",
    "backpack": "背包",
    "suitcase": "行李箱",
    "unattended_luggage": "无人行李",
    "fall": "人员跌倒",
    "sleep": "睡觉",
}
LABEL_MAP = label_map

DEFAULT_FONT_CANDIDATES = [
    str(EDGE_ROOT / "fonts" / "simkai.ttf"),
    str(EDGE_ROOT / "simkai.ttf"),
    "simkai.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

_FONT_CACHE: Dict = {}
_COLOR_MAP_CACHE: Dict = {}


def _load_font(font_size: int, font_candidates=None):
    font_candidates = font_candidates or DEFAULT_FONT_CANDIDATES
    cache_key = (font_size, tuple(font_candidates))
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]
    for path in font_candidates:
        if path and os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                _FONT_CACHE[cache_key] = font
                return font
            except OSError:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[cache_key] = font
    return font


def _build_color_map(class_names: Dict[int, str]):
    cache_key = tuple(sorted(class_names.items()))
    if cache_key in _COLOR_MAP_CACHE:
        return _COLOR_MAP_CACHE[cache_key]
    rng = np.random.default_rng(42)
    color_map = {}
    for cls_id in sorted(class_names.keys()):
        rgb = rng.integers(50, 256, size=3, dtype=np.int64)
        color_map[cls_id] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
    _COLOR_MAP_CACHE[cache_key] = color_map
    return color_map


def _normalize_detections(
    detections: Sequence[Dict],
    class_names: Dict[int, str],
    filter_class_names: Optional[Sequence[str]] = None,
) -> List[Dict]:
    out = []
    filters = set(filter_class_names) if filter_class_names else None
    for d in detections:
        name = d.get("class_name", "")
        if filters and name not in filters:
            continue
        cls_id = d.get("class_id")
        if cls_id is None:
            rev = {v: k for k, v in class_names.items()}
            cls_id = rev.get(name, -1)
        if cls_id < 0:
            continue
        x1, y1, x2, y2 = d["bbox"]
        out.append(
            {
                "cls_id": int(cls_id),
                "cls_name": name,
                "conf": float(d.get("score", 0)),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )
    return out


def _draw_boxes_on_frame(
    orig_img: np.ndarray,
    items: List[Dict],
    class_color_map_bgr: Dict[int, tuple],
    *,
    alpha: float = 0.25,
    font_candidates=None,
) -> np.ndarray:
    """与总项目 draw_custom_boxes 相同的绘制逻辑。"""
    if not items:
        return orig_img.copy()

    height, width = orig_img.shape[:2]
    # 与 ultralytics orig_img 一致：直接 fromarray，不做 BGR/RGB 转换
    pil_img = Image.fromarray(orig_img)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    font_cache = {}
    text_jobs = []

    for item in items:
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cls_id = item["cls_id"]
        conf = item["conf"]
        cls_name = item["cls_name"]
        readable_name = label_map.get(cls_name, cls_name)
        label = f"{readable_name} {conf * 100:.0f}%"

        box_color_bgr = class_color_map_bgr.get(cls_id, (0, 255, 0))
        box_color_rgb = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])

        overlay_draw.rectangle([x1, y1, x2, y2], fill=(*box_color_rgb, int(255 * alpha)))

        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        min_side = max(1, min(box_width, box_height))
        corner_length = int(min_side * 0.2)
        edge_width = max(4, int(min_side * 0.02))
        base_font_size = int(min_side * 0.10)
        font_size = max(12, min(base_font_size, 40))

        if font_size not in font_cache:
            font_cache[font_size] = _load_font(font_size, font_candidates=font_candidates)

        text_jobs.append(
            {
                "label": label,
                "box": (x1, y1, x2, y2),
                "font_size": font_size,
                "color_rgb": box_color_rgb,
                "corner_length": corner_length,
                "edge_width": edge_width,
            }
        )

    if text_jobs:
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(pil_img)
    for job in text_jobs:
        x1, y1, x2, y2 = job["box"]
        box_color_rgb = job["color_rgb"]
        font = font_cache[job["font_size"]]
        corner_length = job["corner_length"]
        edge_width = job["edge_width"]

        corners = [
            ((x1, y1), (x1 + corner_length, y1)),
            ((x1, y1), (x1, y1 + corner_length)),
            ((x2, y1), (x2 - corner_length, y1)),
            ((x2, y1), (x2, y1 + corner_length)),
            ((x1, y2), (x1 + corner_length, y2)),
            ((x1, y2), (x1, y2 - corner_length)),
            ((x2, y2), (x2 - corner_length, y2)),
            ((x2, y2), (x2, y2 - corner_length)),
        ]
        for start, end in corners:
            draw.line([start, end], fill=box_color_rgb, width=edge_width)

        text_bbox = draw.textbbox((0, 0), job["label"], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        pad = 5
        text_x = min(x1, pil_img.width - text_width - pad * 2)
        text_x = max(0, text_x)
        preferred_y = y1 - text_height - pad * 2
        if preferred_y < 0:
            preferred_y = min(pil_img.height - text_height - pad * 2, y2 + pad)
        text_y = max(0, preferred_y)
        bg_coords = [
            text_x - pad,
            text_y - pad,
            text_x + text_width + pad,
            text_y + text_height + pad,
        ]
        draw.rectangle(bg_coords, fill=box_color_rgb)
        draw.text((text_x, text_y), job["label"], fill="white", font=font)

    return np.array(pil_img)


def draw_custom_boxes_from_detections(
    orig_img: np.ndarray,
    detections: Sequence[Dict],
    class_names: Optional[Dict[int, str]] = None,
    *,
    filter_class_names: Optional[Sequence[str]] = None,
    alpha: float = 0.25,
    font_candidates=None,
) -> np.ndarray:
    """将边缘检测列表画成与总项目 draw_custom_boxes 相同风格。"""
    if orig_img is None or orig_img.size == 0:
        return orig_img
    if class_names is None:
        from utils import NAMES

        class_names = NAMES

    items = _normalize_detections(detections, class_names, filter_class_names)
    if not items:
        return orig_img.copy()

    class_color_map_bgr = _build_color_map(class_names)
    return _draw_boxes_on_frame(
        orig_img,
        items,
        class_color_map_bgr,
        alpha=alpha,
        font_candidates=font_candidates,
    )


def draw_bag_alarm_boxes(
    orig_img: np.ndarray,
    detections: Sequence[Dict],
    parameter: dict | None = None,
) -> np.ndarray:
    """无人行李报警：绘制背包、行李箱、行人。"""
    from algorithms.bag.postprocess import filter_alarm_draw_detections

    items = filter_alarm_draw_detections(detections, parameter=parameter)
    if not items:
        return orig_img.copy() if orig_img is not None else orig_img
    class_names = {}
    for det in items:
        cls_id = det.get("class_id")
        name = det.get("class_name", "")
        if cls_id is not None and name:
            class_names[int(cls_id)] = name
    return draw_custom_boxes_from_detections(orig_img, items, class_names)


def draw_alarm_boxes(orig_img: np.ndarray, detections: Sequence[Dict], alarm_label: str) -> np.ndarray:
    """报警专用：只画触发 label，供 run_in_executor 调用。"""
    return draw_custom_boxes_from_detections(
        orig_img,
        detections,
        filter_class_names=[alarm_label],
    )


def draw_custom_boxes(results, *, alpha=0.25, font_candidates=None):
    """兼容总项目签名；results 可为 ultralytics Results 或 (frame, detections)。"""
    if isinstance(results, (list, tuple)) and len(results) == 2:
        frame, dets = results
        return draw_custom_boxes_from_detections(frame, dets, alpha=alpha, font_candidates=font_candidates)
    if not results or len(results) == 0:
        return None
    orig_img = getattr(results[0], "orig_img", None)
    if orig_img is None:
        return None
    boxes = getattr(results[0], "boxes", [])
    if not boxes:
        return orig_img
    class_names = results[0].names
    dets = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].detach().cpu().numpy()
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        name = class_names.get(cls_id, str(cls_id))
        dets.append(
            {
                "class_id": cls_id,
                "class_name": name,
                "score": conf,
                "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
            }
        )
    return draw_custom_boxes_from_detections(orig_img, dets, class_names, alpha=alpha, font_candidates=font_candidates)
