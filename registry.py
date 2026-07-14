# -*- coding: utf-8 -*-
"""
算法注册表 — 与总项目 registry.py 中 create_report_policy 对齐。
各类别 Report 从 algorithms/<name>/report.py 加载。
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Type

from report import Policy

from config import IMG_SIZE_HIGH, IMG_SIZE_LOW

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parent / "algorithm_registry.json"
_instance: Optional["EdgeAlgorithmRegistry"] = None

_REPORT_META_KEYS = frozenset({"policy_type", "module", "class_name"})


class EdgeAlgorithmRegistry:
    def __init__(self, path: Path = _REGISTRY_PATH):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.algorithm_types: Dict = data.get("algorithm_types", {})
        self.algorithms: Dict = data.get("algorithms", {})

    @staticmethod
    def _import_class(module_path: str, class_name: str) -> Type[Policy]:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        if not issubclass(cls, Policy):
            raise TypeError(f"{module_path}.{class_name} 必须继承 report.Policy")
        return cls

    def get_label(self, algo_name: str) -> Optional[str]:
        algo = self.algorithms.get(algo_name)
        return algo.get("label") if algo else None

    def get_algorithm_config(self, algo_name: str) -> Optional[Dict]:
        return self.algorithms.get(algo_name)

    def get_algorithm_type(self, algo_name: str) -> Optional[str]:
        algo = self.algorithms.get(algo_name)
        return algo.get("type") if algo else None

    def get_interval_for_type(self, algo_type: str) -> float:
        """检测间隔（秒）— 仅 algorithm_registry.json → algorithm_types[type].interval。"""
        from config import DEFAULT_DETECT_INTERVAL

        tc = self.algorithm_types.get(algo_type) or {}
        if "interval" in tc:
            return max(0.0, float(tc["interval"]))
        return DEFAULT_DETECT_INTERVAL

    def resolve_weight_path(
        self,
        algo_type: str,
        resolution_mode: str = "low",
        edge_root: Optional[Path] = None,
        algo_name: Optional[str] = None,
    ) -> tuple[Optional[str], int]:
        """优先 algorithms[algo_name].weight_paths，否则 algorithm_types[algo_type].weight_paths。"""
        root = edge_root or Path(__file__).resolve().parent
        mode = (resolution_mode or "low").strip().lower()
        size_key = "1088" if mode == "high" else "640"
        default_imgsz = IMG_SIZE_HIGH if size_key == "1088" else IMG_SIZE_LOW

        candidates = []
        if algo_name:
            algo = self.algorithms.get(algo_name) or {}
            if algo.get("weight_paths"):
                candidates.append((algo.get("weight_paths"), algo.get("image_size")))
        tc = self.algorithm_types.get(algo_type) or {}
        candidates.append((tc.get("weight_paths"), tc.get("image_size")))

        for weight_paths, custom_imgsz in candidates:
            if not weight_paths:
                continue
            rel = weight_paths.get(size_key) or weight_paths.get("640")
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = root / path
            imgsz = int(custom_imgsz or default_imgsz)
            if mode == "high" and size_key == "1088":
                imgsz = max(imgsz, IMG_SIZE_HIGH)
            return str(path), imgsz
        return None, default_imgsz

    def create_report_policy(self, algo_name: str) -> Optional[Policy]:
        """与总项目 registry.create_report_policy 相同入口。"""
        algo = self.algorithms.get(algo_name)
        if not algo or "report" not in algo:
            logger.warning("算法 %s 未定义 report 配置", algo_name)
            return None

        rc = algo["report"]
        pt = rc.get("policy_type", "custom")

        if pt == "custom":
            module_path = rc.get("module")
            class_name = rc.get("class_name")
            if not module_path or not class_name:
                logger.error("算法 %s report 缺少 module/class_name", algo_name)
                return None
            try:
                cls = self._import_class(module_path, class_name)
            except Exception as exc:
                logger.error("加载 %s report 失败: %s", algo_name, exc)
                return None
            kwargs = {k: v for k, v in rc.items() if k not in _REPORT_META_KEYS}
            if "label" not in kwargs:
                kwargs.setdefault("label", algo.get("label", ""))
            if "algo_type" not in kwargs:
                kwargs.setdefault("algo_type", algo.get("type", ""))
            return cls(**kwargs) if kwargs else cls()

        if pt == "simple":
            return Policy(
                algo_type=algo.get("type", ""),
                description=rc.get("description", algo_name),
                label=algo.get("label", ""),
                times=rc.get("times", 600),
                required_frames=rc.get("required_frames", 10),
                continue_time_delta=rc.get("continue_time_delta", 2),
            )

        logger.warning("未知 policy_type=%s，算法 %s", pt, algo_name)
        return None


def get_registry() -> EdgeAlgorithmRegistry:
    global _instance
    if _instance is None:
        _instance = EdgeAlgorithmRegistry()
    return _instance


def create_report_policy(algo_name: str) -> Optional[Policy]:
    return get_registry().create_report_policy(algo_name)


def registered_algorithm_names():
    reg = get_registry()
    return list(reg.algorithms.keys())
