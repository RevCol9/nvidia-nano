# -*- coding: utf-8 -*-
"""无人行李报警策略。"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Sequence

from report import Policy

from algorithms.bag import config as C
from algorithms.bag.postprocess import frame_has_unattended_luggage


class BagReport(Policy):
    def __init__(
        self,
        algo_type: str = "bag_detect",
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

    def need_report_from_detections(
        self,
        event_time: datetime,
        detections: Sequence[Dict],
        parameter: dict | None = None,
        frame_bgr=None,
    ) -> bool:
        """无人行李：有行李时才跑深度 → 同深度 + 扩框 IOU。"""
        hit = frame_has_unattended_luggage(
            detections,
            parameter=parameter,
            frame_bgr=frame_bgr,
        )
        return self.tick_alarm(event_time, hit)


__all__ = ["BagReport"]
