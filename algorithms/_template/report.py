# -*- coding: utf-8 -*-
"""新算法 report 模板 — 复制到 algorithms/<name>/report.py 并修改。"""
from report import Policy

from algorithms._template import config as C


class TemplateReport(Policy):
    def __init__(
        self,
        algo_type: str = "multi_detect",
        description: str | None = None,
        label: str | None = None,
        times: int | None = None,
        required_frames: int | None = None,
        continue_time_delta: float | None = None,
    ):
        super().__init__(
            algo_type=algo_type,
            description=description or C.DESCRIPTION,
            label=label or C.LABEL,
            times=times if times is not None else C.DEFAULT_TIMES,
            required_frames=required_frames if required_frames is not None else C.DEFAULT_REQUIRED_FRAMES,
            continue_time_delta=continue_time_delta if continue_time_delta is not None else C.DEFAULT_CONTINUE_DELTA,
        )


__all__ = ["TemplateReport"]
