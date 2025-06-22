from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from ..db import get_session
from ..models import Merchant, Transaction, User
from ..schemas import MerchantOut, TransactionIn, TransactionOut, UserOut
from ..services.transactions import TransactionService

router = APIRouter()
service: TransactionService | None = None  # будет инициализирован в lifespan


def get_service(request: Request) -> TransactionService:
    return request.app.state.service


# ───────── 1. POST /transactions ────────────────────────────────────────
@router.post("/transactions")
async def create_tx(
    tx: TransactionIn,
    db: AsyncSession = Depends(get_session),
    svc: TransactionService = Depends(get_service),
):
    return await svc.create(tx, db)


# ───────── 2. GET /transactions ----------------------------------------
@router.get("/transactions", response_model=list[TransactionOut])
async def list_transactions(
    fraud_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_session),
):
    stmt = (
        select(Transaction)
        .options(joinedload(Transaction.user), joinedload(Transaction.merchant))
        .order_by(Transaction.id.desc())
        .limit(limit)
    )
    if fraud_only:
        stmt = stmt.where(Transaction.fraud_prob >= 0.5)
    return (await db.execute(stmt)).scalars().all()


# ───────── 3. GET /transactions/{id} ------------------------------------
@router.get("/transactions/{tx_id}", response_model=TransactionOut)
async def get_tx(tx_id: int, db: AsyncSession = Depends(get_session)):
    stmt = (
        select(Transaction)
        .options(joinedload(Transaction.user), joinedload(Transaction.merchant))
        .where(Transaction.id == tx_id)
    )
    tr = (await db.execute(stmt)).scalar_one_or_none()
    if tr is None:
        raise HTTPException(404)
    return tr


# ───────── 4. GET /users/{id} ------------------------------------------
@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int, db: AsyncSession = Depends(get_session)):
    stmt = (
        select(User)
        .options(
            selectinload(User.transactions).selectinload(Transaction.merchant),
        )
        .where(User.id == user_id)
    )
    u = (await db.execute(stmt)).scalar_one_or_none()
    if not u:
        raise HTTPException(404)
    return u


# ───────── 5. GET /merchants/{id} --------------------------------------
@router.get("/merchants/{merchant_id}", response_model=MerchantOut)
async def get_merchant(merchant_id: int, db: AsyncSession = Depends(get_session)):
    stmt = (
        select(Merchant)
        .options(
            selectinload(Merchant.transactions).selectinload(Transaction.user),
        )
        .where(Merchant.id == merchant_id)
    )
    m = (await db.execute(stmt)).scalar_one_or_none()
    if not m:
        raise HTTPException(404)
    return m
