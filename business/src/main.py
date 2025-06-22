"""Точка входа FastAPI-приложения."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from .api.v1 import router as api_router
from .config import get_settings
from .db import _engine
from .models import Base
from .mq import MQClient
from .services.transactions import TransactionService

settings = get_settings()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # миграция схемы
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # создаём синглтоны и кладём в app.state
    app.state.mq_client = MQClient()
    app.state.service = TransactionService(app.state.mq_client)

    yield

    # при остановке, если нужно, закрываем подключения
    await app.state.mq_client._conn.close()  # noqa: SLF001


app = FastAPI(lifespan=lifespan, title="Business Service")
app.include_router(api_router)
