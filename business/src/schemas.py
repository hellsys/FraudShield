from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    trans_date_trans_time: str
    cc_num: str
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float


class UserOut(BaseModel):
    id: int
    cc_num: str
    first: str | None
    last: str | None
    gender: str | None
    city: str | None
    state: str | None
    job: str | None
    age: int | None = Field(
        None,
    )
    lat: float | None
    long: float | None

    transactions: Optional[list["TransactionOut"]] = Field(None)


class MerchantOut(BaseModel):
    id: int
    name: str
    category: str | None
    merch_lat: float | None
    merch_long: float | None

    transactions: Optional[list["TransactionOut"]] = Field(None)


class ThinUser(BaseModel):
    id: int
    first: str | None
    last: str | None
    lat: float | None
    long: float | None

    class Config:
        orm_mode = True


class ThinMerchant(BaseModel):
    id: int
    name: str
    merch_lat: float | None
    merch_long: float | None


class TransactionOut(BaseModel):
    id: int
    amt: float
    trans_time: datetime
    fraud_prob: float | None
    user: ThinUser
    merchant: ThinMerchant
