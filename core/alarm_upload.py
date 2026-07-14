# -*- coding: utf-8 -*-
"""
全进程共享 aiohttp 上传队列 — 串行 POST，避免多路报警并发抢连接导致 Bad file descriptor。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class AlarmUploadService:
    """单 session + 单 worker 串行上传。"""

    _instances: Dict[str, "AlarmUploadService"] = {}

    @classmethod
    def get(cls, upload_url: str) -> "AlarmUploadService":
        if upload_url not in cls._instances:
            cls._instances[upload_url] = cls(upload_url)
        return cls._instances[upload_url]

    def __init__(self, upload_url: str):
        self.upload_url = upload_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None
        self._started = False

    async def start(self):
        if self._started:
            return
        self._queue = asyncio.Queue(maxsize=200)
        self._worker = asyncio.create_task(self._worker_loop(), name="alarm-upload")
        self._started = True
        logger.info("AlarmUploadService started: %s", self.upload_url)

    async def stop(self):
        if not self._started:
            return
        self._started = False
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        await self._close_session()
        self._queue = None

    async def upload(self, payload: Dict[str, Any], camera_index: str, timeout: float = 30.0) -> bool:
        if not self._started or self._queue is None:
            raise RuntimeError("AlarmUploadService not started")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((self.upload_url, payload, camera_index, timeout, future))
        return await future

    async def _worker_loop(self):
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                break
            url, payload, camera_index, timeout, future = item
            try:
                ok = await self._post_once(url, payload, camera_index, timeout)
            except Exception as exc:
                logger.error("Upload worker error: %s", exc)
                ok = False
            if not future.done():
                future.set_result(ok)

    async def _post_once(
        self,
        url: str,
        payload: Dict[str, Any],
        camera_index: str,
        timeout: float,
    ) -> bool:
        session = await self._ensure_session()
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout, connect=10, sock_read=timeout),
            ) as response:
                await response.read()
                if response.status == 200:
                    logger.info("Camera %s upload success", camera_index)
                    return True
                logger.warning("Camera %s upload failed: HTTP %s", camera_index, response.status)
                return False
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            logger.error("Camera %s upload error: %s", camera_index, exc)
            await self._close_session()
            return False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=2,
                force_close=True,
                enable_cleanup_closed=False,
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _close_session(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        await asyncio.sleep(0)
