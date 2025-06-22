"""Высокоуровневая бизнес-логика транзакций."""

from __future__ import annotations

from datetime import datetime

from dateutil import parser as dtparser
from dateutil import tz
from sqlalchemy.ext.asyncio import AsyncSession

from ..history_feats import history_feats
from ..mq import MQClient
from ..repositories.merchants import MerchantRepository
from ..repositories.transactions import TransactionRepository
from ..repositories.users import UserRepository
from ..schemas import TransactionIn


class TransactionService:  # noqa: D101
    def __init__(self, mq_client: MQClient) -> None:
        self._mq = mq_client

    # ------------------------------------------------------------------ #
    async def create(
        self,
        tx_in: TransactionIn,
        session: AsyncSession,
    ) -> dict[str, float]:
        """Создать запись в БД, обогатить историей, получить скор."""
        # ---- upsert user / merchant -----------------------------------
        users = UserRepository(session)
        merchants = MerchantRepository(session)
        tx_repo = TransactionRepository(session)

        user = await users.get_by_cc(tx_in.cc_num) or await users.create(
            cc_num=tx_in.cc_num,
            first=tx_in.first,
            last=tx_in.last,
            gender=tx_in.gender,
            street=tx_in.street,
            city=tx_in.city,
            state=tx_in.state,
            zip=tx_in.zip,
            lat=tx_in.lat,
            long=tx_in.long,
            city_pop=tx_in.city_pop,
            job=tx_in.job,
            dob=dtparser.isoparse(tx_in.dob).date(),
        )
        merchant = await merchants.get_by_name(
            tx_in.merchant
        ) or await merchants.create(
            name=tx_in.merchant,
            category=tx_in.category,
            merch_lat=tx_in.merch_lat,
            merch_long=tx_in.merch_long,
        )

        trans_time_utc: datetime = dtparser.isoparse(
            tx_in.trans_date_trans_time
        ).astimezone(tz.UTC)

        tx = await tx_repo.create(
            trans_num=tx_in.trans_num,
            user_id=user.id,
            merchant_id=merchant.id,
            amt=tx_in.amt,
            trans_time=trans_time_utc,
            unix_time=tx_in.unix_time,
        )

        # ---- history-features ----------------------------------------
        hist = await history_feats(tx_in.model_dump() | {"user_id": user.id}, session)

        # ---- RPC в prediction-сервис ---------------------------------
        payload = tx_in.model_dump() | hist | {"transaction_id": tx.id}
        pred = await self._mq.predict(payload)
        tx.fraud_prob = pred["probability"]

        await session.commit()
        return {"id": tx.id, "fraud_prob": tx.fraud_prob}
