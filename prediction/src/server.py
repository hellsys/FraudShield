import asyncio
import json
from contextlib import asynccontextmanager
from logging import getLogger
from typing import AsyncIterator

import aio_pika
import uvloop

from .config import get_settings
from .model_registry import ModelRegistry
from .predictor import Predictor

logger = getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def rabbit_connection(
    url: str, retries: int = 10, delay: int = 3
) -> AsyncIterator[aio_pika.RobustConnection]:
    for attempt in range(1, retries + 1):
        try:
            yield await aio_pika.connect_robust(url)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rabbit attempt %d/%d failed: %s", attempt, retries, exc)
            await asyncio.sleep(delay)
    raise RuntimeError("RabbitMQ not reachable")


async def main() -> None:  # noqa: D401
    registry = await ModelRegistry.create()
    predictor = Predictor(registry)

    async with rabbit_connection(settings.rabbit_url) as conn:
        chan = await conn.channel()
        await chan.set_qos(prefetch_count=settings.prefetch)

        req_q = await chan.declare_queue("predict.request", durable=True)

        async with req_q.iterator() as queue_iter:
            async for msg in queue_iter:
                async with msg.process():
                    result = await predictor.predict(msg.body)
                    await chan.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(result).encode(),
                            correlation_id=msg.correlation_id,
                        ),
                        routing_key=msg.reply_to or "predict.response",
                    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())
