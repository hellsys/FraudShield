"""SQLAlchemy engine и фабрика AsyncSession."""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Final

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from .config import get_settings

settings = get_settings()

_engine: Final[AsyncEngine] = create_async_engine(
    settings.database_url,
    pool_size=20,
    pool_pre_ping=True,
)
AsyncSessionLocal: Final = async_sessionmaker(_engine, expire_on_commit=False)


async def get_session() -> AsyncIterator:
    """Dependency-генератор для FastAPI."""
    async with AsyncSessionLocal() as session:
        yield session
