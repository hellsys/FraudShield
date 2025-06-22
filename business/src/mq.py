"""Асинхронный RPC-клиент RabbitMQ (aio-pika)."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Final

import aio_pika

from .config import get_settings

settings = get_settings()

REQ_QUEUE: Final = "predict.request"
RES_QUEUE: Final = "predict.response"


class MQClient:
    """Лёгкий RPC-клиент поверх aio-pika (single channel)."""

    def __init__(self) -> None:
        self._conn: aio_pika.RobustConnection | None = None
        self._chan: aio_pika.Channel | None = None
        self._futures: dict[str, asyncio.Future] = {}

    # ------------------------------------------------------------------ #
    async def _ensure_conn(self) -> None:
        if self._conn:
            return
        self._conn = await aio_pika.connect_robust(settings.rabbit_url)
        self._chan = await self._conn.channel()
        await self._chan.set_qos(prefetch_count=settings.rabbit_prefetch)

        # Ответная очередь + consumer единый
        res_q = await self._chan.declare_queue(RES_QUEUE, durable=True)
        await res_q.consume(self._on_response, no_ack=False)

    # ------------------------------------------------------------------ #
    async def rpc_call(
        self, payload: dict[str, Any], timeout: float = 30.0
    ) -> dict[str, Any]:
        """Отправить сообщение и ждать ответ (RPC)."""
        await self._ensure_conn()

        corr_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self._futures[corr_id] = future

        await self._chan.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload, default=str).encode(),
                correlation_id=corr_id,
                reply_to=RES_QUEUE,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=REQ_QUEUE,
        )
        return await asyncio.wait_for(future, timeout)

    # alias для старого названия
    predict = rpc_call

    # ------------------------------------------------------------------ #
    async def _on_response(self, msg: aio_pika.IncomingMessage) -> None:  # noqa: D401
        async with msg.process(ignore_processed=True):
            if fut := self._futures.pop(msg.correlation_id, None):
                fut.set_result(json.loads(msg.body))
