#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单张图片推理测试 — 使用 algorithms/car/weights/car.pt，保存画框结果图。

用法（在 vision 项目根目录）:
  python algorithms/car/test_image.py test.jpg
  python algorithms/car/test_image.py test.jpg -o result.jpg --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ALGO_DIR = Path(__file__).resolve().parent
ROOT = ALGO_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from draw_picture import draw_custom_boxes_from_detections  # noqa: E402
from registry import get_registry  # noqa: E402


def resolve_device(requested: str) -> str:
    import torch

    req = (requested or "auto").strip().lower()
    if req in ("auto", ""):
        dev = "0" if torch.cuda.is_available() else "cpu"
        print(f"device={dev} (cuda_available={torch.cuda.is_available()})")
        return dev
    if req in ("0", "cuda", "cuda:0") and not torch.cuda.is_available():
        print("警告: CUDA 不可用，已改用 cpu")
        return "cpu"
    return requested


def default_model_path() -> Path:
    reg = get_registry()
    path, _ = reg.resolve_weight_path("car_detect", "low", ROOT, "Car_Detection")
    return Path(path) if path else ALGO_DIR / "weights" / "car.pt"


def run_inference(
    image_path: Path,
    output_path: Path,
    model_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> int:
    if not model_path.is_file():
        print(f"模型不存在: {model_path}")
        return 1
    if not image_path.is_file():
        print(f"图片不存在: {image_path}")
        return 1

    from ultralytics import YOLO

    bgr = cv2.imread(str(image_path))
    if bgr is None or bgr.size == 0:
        print(f"无法读取图片: {image_path}")
        return 1

    model = YOLO(str(model_path))
    results = model.predict(
        source=bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    r = results[0]
    class_names = dict(r.names or {})
    detections = []

    boxes = getattr(r, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().numpy()
            cls_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            name = class_names.get(cls_id, str(cls_id))
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": name,
                    "score": score,
                    "bbox": [float(v) for v in xyxy],
                }
            )

    print(f"model={model_path}")
    print(f"device={device}")
    print(f"classes={class_names}")
    print(f"detections={len(detections)}")
    for d in detections:
        print(
            f"  {d['class_name']} {d['score']:.2f} "
            f"bbox=[{d['bbox'][0]:.0f},{d['bbox'][1]:.0f},{d['bbox'][2]:.0f},{d['bbox'][3]:.0f}]"
        )

    annotated = draw_custom_boxes_from_detections(bgr, detections, class_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        print(f"保存失败: {output_path}")
        return 1

    print(f"结果已保存: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Car_Detection 单图推理测试")
    parser.add_argument("image", type=Path, help="输入图片路径")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出图片路径")
    parser.add_argument("-m", "--model", type=Path, default=None, help="权重路径")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto / 0 / cpu")
    args = parser.parse_args()

    image_path = args.image.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else image_path.with_name(f"{image_path.stem}_result{image_path.suffix or '.jpg'}")
    )
    model_path = (args.model or default_model_path()).resolve()
    device = resolve_device(args.device)

    return run_inference(
        image_path=image_path,
        output_path=output_path,
        model_path=model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
