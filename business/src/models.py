import datetime as dt
from datetime import date

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cc_num: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    first: Mapped[str | None] = mapped_column(String(50), nullable=True)
    last: Mapped[str | None] = mapped_column(String(50), nullable=True)
    gender: Mapped[str | None] = mapped_column(String(1), nullable=True)
    street: Mapped[str | None] = mapped_column(String(120), nullable=True)
    city: Mapped[str | None] = mapped_column(String(60), nullable=True)
    state: Mapped[str | None] = mapped_column(String(2), nullable=True)
    zip: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    long: Mapped[float | None] = mapped_column(Float, nullable=True)
    city_pop: Mapped[int | None] = mapped_column(Integer, nullable=True)
    job: Mapped[str | None] = mapped_column(String(100), nullable=True)
    dob: Mapped[dt.date | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    @property
    def age(self) -> int | None:
        if not self.dob:
            return None
        born = self.dob.date() if hasattr(self.dob, "date") else self.dob
        today = date.today()
        return (
            today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        )

    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction", back_populates="user"
    )


class Merchant(Base):
    __tablename__ = "merchants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    merch_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    merch_long: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction", back_populates="merchant"
    )


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = ({"postgresql_partition_by": None},)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trans_num: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE")
    )
    merchant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("merchants.id", ondelete="CASCADE")
    )
    amt: Mapped[float] = mapped_column(Numeric(12, 2))
    trans_time: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    unix_time: Mapped[int] = mapped_column(Integer, nullable=False)
    fraud_prob: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="transactions")
    merchant: Mapped["Merchant"] = relationship(
        "Merchant", back_populates="transactions"
    )
