# -*- coding: utf-8 -*-
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RunningOptionItem(BaseModel):
    type: str
    enabled: int  # 0=remove, 1=add, 2=update
    params: Dict = Field(default_factory=dict)


class BatchCameraRunningItem(BaseModel):
    camera_id: int
    algorithms: List[RunningOptionItem]
    nvr_ip: str
    nvr_channel_num: str = ""
    camera_ip: str = ""
    resolution_mode: Optional[Literal["low", "high"]] = None


class BatchCameraRunning(BaseModel):
    option: List[BatchCameraRunningItem]
