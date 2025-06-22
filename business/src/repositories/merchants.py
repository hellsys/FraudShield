from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Merchant


class MerchantRepository:
    """Работа c таблицей merchants."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ────────────────────────────────────────────────────────────────
    async def get_by_name(self, name: str) -> Optional[Merchant]:
        stmt = select(Merchant).where(Merchant.name == name)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def create(
        self,
        *,
        name: str,
        category: str | None,
        merch_lat: float | None,
        merch_long: float | None,
    ) -> Merchant:
        merchant = Merchant(
            name=name,
            category=category,
            merch_lat=merch_lat,
            merch_long=merch_long,
        )
        self._session.add(merchant)
        await self._session.flush()
        return merchant
