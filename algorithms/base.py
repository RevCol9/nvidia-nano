# -*- coding: utf-8 -*-
"""
multi_detect 算法说明。

判警策略: algorithms/<name>/report.py（各类别独立）
注册表:   registry.py + algorithm_registry.json
推理报警: inference_service.py → process_non_tracking_alarms
"""
from inference_service import AlgorithmSlot

MultiDetectAlgo = AlgorithmSlot

__all__ = ["MultiDetectAlgo", "AlgorithmSlot"]
