"""
Требует запущенный business-service и Postgres (DATABASE_URL).
Клиент делает реальные HTTP-запросы через uvicorn-сервис.
"""

import os
import random
import string
from datetime import datetime

import httpx
import pytest

BUSINESS_URL = os.getenv("BUSINESS_BASE_URL", "http://localhost:8080")

print(f"Using business service URL: {BUSINESS_URL}")
def _rnd(n=6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


@pytest.mark.asyncio
async def test_create_and_get_transaction():
    print(BUSINESS_URL)
    async with httpx.AsyncClient(base_url=BUSINESS_URL, timeout=10.0) as client:
        tx_json = {
            "trans_date_trans_time": f"{datetime.utcnow().isoformat()}Z",
            "cc_num": _rnd(),
            "merchant": f"shop-{_rnd()}",
            "category": "shopping_pos",
            "amt": 17.55,
            "first": "Jane",
            "last": "Roe",
            "gender": "F",
            "street": "7th Ave",
            "city": "NY",
            "state": "NY",
            "zip": 10001,
            "lat": 0,
            "long": 0,
            "city_pop": 1,
            "job": "QA",
            "dob": "1990-01-01",
            "trans_num": _rnd(12),
            "unix_time": int(datetime.utcnow().timestamp()),
            "merch_lat": 0,
            "merch_long": 0,
            "is_fraud": None,
        }

        # 1. POST /transactions
        resp = await client.post("/transactions", json=tx_json)
        assert resp.status_code == 200
        tx_id = resp.json()["id"]

        # 2. GET /transactions/{id}
        resp2 = await client.get(f"/transactions/{tx_id}")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["id"] == tx_id
        assert 0.0 <= (data["fraud_prob"] or 0) <= 1.0
