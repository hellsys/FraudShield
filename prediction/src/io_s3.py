import asyncio
from pathlib import Path
from typing import Final

import aioboto3

from .config import get_settings

settings = get_settings()


async def download_if_needed(key: str) -> Path:
    """Скачать файл из S3, если он ещё не валяется локально.

    Возвращает локальный путь, чтобы остальные модули не знали о S3 вовсе.
    """
    local = settings.tmp_dir / Path(key).name
    if local.exists():
        return local

    session: Final = aioboto3.Session()

    async with session.client(
        "s3",
        endpoint_url=str(settings.s3_endpoint),
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    ) as s3:
        await s3.download_file(settings.s3_bucket, key, str(local))

    return local


async def batch_download(keys: list[str]) -> dict[str, Path]:
    """Асинхронно скачать пачку файлов (S3 → /tmp)."""
    paths = await asyncio.gather(*(download_if_needed(k) for k in keys))
    return dict(zip(keys, paths))
