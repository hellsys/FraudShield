"""
Интеграционный тест: кладём транзакцию в RabbitMQ → ждём ответ от prediction.

Требуется работающий prediction-service и RabbitMQ, адрес берётся
из переменной RABBIT_URL   (amqp://user:pass@host:5672/)
"""

import asyncio
import json
import os
import uuid

import aio_pika
import pytest

RABBIT_URL = os.getenv("RABBIT_URL", "amqp://guest:guest@localhost:5673/")
REQ_QUEUE = "predict.request"
RES_QUEUE = "predict.response"


@pytest.mark.asyncio
async def test_prediction_end_to_end():
    conn = await aio_pika.connect_robust(RABBIT_URL)
    async with conn:
        channel = await conn.channel()

        await channel.declare_queue(REQ_QUEUE, durable=True)

        reply_q = await channel.declare_queue("", exclusive=True, auto_delete=True)
        reply_queue_name = reply_q.name

        fut = asyncio.get_running_loop().create_future()
        corr_id = str(uuid.uuid4())

        async def on_response(msg: aio_pika.IncomingMessage) -> None:
            if msg.correlation_id == corr_id:
                fut.set_result(json.loads(msg.body))
                await msg.ack()

        consume_tag = await reply_q.consume(on_response, no_ack=False)
        payload = {
            "transaction_id": 999,
            "trans_date_trans_time": "2025-06-08T12:00:00Z",
            "cc_num": "0000",
            "merchant": "store",
            "category": "shopping_pos",
            "amt": 10.0,
            "first": "A",
            "last": "B",
            "gender": "M",
            "street": "x",
            "city": "y",
            "state": "NY",
            "zip": 1,
            "lat": 0,
            "long": 0,
            "city_pop": 1,
            "job": "dev",
            "dob": "1990-01-01",
            "trans_num": "TST-1",
            "unix_time": 1,
            "merch_lat": 0,
            "merch_long": 0,
            "time_diff_h": 0,
            "prev_amount": 0,
            "amount_diff": 0,
            "amount_ratio": 0,
            "roll_mean_amt_5": 0,
            "roll_std_amt_5": 0,
            "unique_merch_last_30d": 0,
        }

        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode(),
                correlation_id=corr_id,
                reply_to=reply_queue_name,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=REQ_QUEUE,
        )

        result = await asyncio.wait_for(fut, timeout=15)

        assert result["transaction_id"] == 999
        assert 0.0 <= result["probability"] <= 1.0

        await reply_q.cancel(consume_tag)
        await channel.close()
