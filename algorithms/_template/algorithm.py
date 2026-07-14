# -*- coding: utf-8 -*-
from typing import Dict, Optional

from algorithms.base import MultiDetectAlgo
from algorithms._template import config as C


class TemplateAlgorithm(MultiDetectAlgo):
    api_name = C.API_NAME
    label = C.LABEL
    description = C.DESCRIPTION
    default_times = C.DEFAULT_TIMES
    default_required_frames = C.DEFAULT_REQUIRED_FRAMES
    default_continue_delta = C.DEFAULT_CONTINUE_DELTA

    def __init__(self, params: Optional[Dict] = None, upload_url: str = ""):
        super().__init__(params, upload_url)
