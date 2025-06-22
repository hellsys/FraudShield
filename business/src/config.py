import os
from functools import lru_cache
from logging import getLogger
from typing import Final

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = getLogger(__name__)


class Settings(BaseSettings):
    # PostgreSQL
    database_url: str = Field(validation_alias="DATABASE_URL")

    # RabbitMQ
    rabbit_url: str = Field(validation_alias="RABBIT_URL")
    rabbit_prefetch: int = Field(4, validation_alias="PREFETCH")

    # Misc
    service_name: str = "business"
    history_window: int = Field(
        100,
        validation_alias="HIST_WINDOW",
        description="N последних транзакций",
    )
    tz_utc: Final[str] = "UTC"

    model_config = SettingsConfigDict(
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Singleton-доступ."""
    logger.warning("Loading settings...")
    logger.warning(os.environ.get("DATABASE_URL", "Not set"))
    logger.warning(os.environ.get("RABBIT_URL", "Not set"))
    logger.warning("Loading settings...")
    return Settings()
