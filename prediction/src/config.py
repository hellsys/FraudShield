from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Глобальная конфигурация приложения, собирается из ENV-переменных.

    Используем pydantic → строгая валидация и автодокументация.
    """

    # ───────── Infrastructure ──────────────────────────────────────────────
    rabbit_url: str = Field(..., validation_alias="RABBIT_URL")
    prefetch: int = Field(4, validation_alias="PREFETCH")

    # ───────── S3 ──────────────────────────────────────────────────────────
    s3_endpoint: HttpUrl = Field(..., validation_alias="S3_ENDPOINT")
    s3_access_key: str = Field(..., validation_alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(..., validation_alias="S3_SECRET_KEY")
    s3_bucket: str = Field(..., validation_alias="S3_BUCKET")

    # ───────── Model artefacts ─────────────────────────────────────────────
    gnn_model_key: str = Field(..., validation_alias="GNN_MODEL_KEY")
    catboost_model_key: str = Field(..., validation_alias="CATBOOST_MODEL_KEY")
    lr_model_key: str = Field(..., validation_alias="LR_MODEL_KEY")
    cc2idx_key: str = Field(..., validation_alias="CC2IDX_KEY")
    node_scaler_key: str = Field(..., validation_alias="NODE_SCALER_KEY")
    edge_scaler_key: str = Field(..., validation_alias="EDGE_SCALER_KEY")
    preproc_key: str = Field(..., validation_alias="PREPROC_KEY")
    node_embeddings_key: str = Field(..., validation_alias="NODE_EMBEDDINGS_KEY")

    # ───────── Misc ────────────────────────────────────────────────────────
    tmp_dir: Path = Path("/tmp/models")

    model_config = SettingsConfigDict(
        case_sensitive=True,
    )


@lru_cache  # «один раз на процесс»
def get_settings() -> Settings:  # noqa: D401
    """Singleton-доступ к конфигурации."""
    settings = Settings()
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    return settings
