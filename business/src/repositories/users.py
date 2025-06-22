from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import User


class UserRepository:  # noqa: D101
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    async def get_by_cc(self, cc_num: str) -> Optional[User]:
        stmt = select(User).where(User.cc_num == cc_num)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def create(
        self,
        *,
        cc_num: str,
        first: str,
        last: str,
        gender: str,
        street: str,
        city: str,
        state: str,
        zip: int,
        lat: float,
        long: float,
        city_pop: int,
        job: str,
        dob: dt.date,
    ) -> User:
        user = User(
            cc_num=cc_num,
            first=first,
            last=last,
            gender=gender,
            street=street,
            city=city,
            state=state,
            zip=zip,
            lat=lat,
            long=long,
            city_pop=city_pop,
            job=job,
            dob=dob,
        )
        self._session.add(user)
        await self._session.flush()
        return user
