from __future__ import annotations

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Transaction


class TransactionRepository:
    """Создание и дополнительные операции с transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ────────────────────────────────────────────────────────────────
    async def create(
        self,
        *,
        trans_num: str,
        user_id: int,
        merchant_id: int,
        amt: float,
        trans_time: datetime,
        unix_time: int,
    ) -> Transaction:
        tx = Transaction(
            trans_num=trans_num,
            user_id=user_id,
            merchant_id=merchant_id,
            amt=amt,
            trans_time=trans_time,
            unix_time=unix_time,
        )
        self._session.add(tx)
        await self._session.flush()
        return tx
