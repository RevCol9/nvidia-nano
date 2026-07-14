# -*- coding: utf-8 -*-
from typing import Dict, Optional

from inference_service import AlgorithmSlot


class SmokeAlgorithm(AlgorithmSlot):
    def __init__(self, params: Optional[Dict] = None, upload_url: str = ""):
        super().__init__("Smoke", params, upload_url)
