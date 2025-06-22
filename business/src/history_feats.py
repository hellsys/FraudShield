import datetime as dt
from typing import Any, Dict

import numpy as np
from dateutil import parser as dtparser
from sqlalchemy import text


def _ts(x: str | dt.datetime) -> dt.datetime:
    return x if isinstance(x, dt.datetime) else dtparser.isoparse(x)


async def history_feats(tx: Dict[str, Any], db) -> Dict[str, float]:
    uid = tx["user_id"]
    curr_ts = _ts(tx["trans_date_trans_time"])
    if curr_ts.tzinfo is None:
        curr_ts = curr_ts.replace(tzinfo=dt.timezone.utc)

    q = """
    SELECT amt, merchant_id, trans_time
    FROM transactions
    WHERE user_id = :uid AND trans_time < :curr
    ORDER BY trans_time DESC
    LIMIT 100
    """
    rows = (await db.execute(text(q), {"uid": uid, "curr": curr_ts})).all()

    if not rows:
        return {
            "prev_amount": 0.0,
            "amount_diff": 0.0,
            "amount_ratio": 0.0,
            "roll_mean_amt_5": 0.0,
            "roll_std_amt_5": 0.0,
            "unique_merch_last_30d": 0,
            "time_diff_h": 4.36,  # медиана из обучения
        }

    amts = [float(r[0]) for r in rows]
    prev_amt = amts[0]

    if prev_amt == 0:
        time_diff_h = 4.36
    else:
        last_db_time = rows[0][2]
        time_diff_h = (curr_ts - last_db_time).total_seconds() / 3600

    amt_diff = float(tx["amt"]) - prev_amt
    amt_ratio = float(tx["amt"]) / (prev_amt or 1.0)

    window_vals = amts[:5]
    roll_mean = float(np.mean(window_vals))
    roll_std = float(np.std(window_vals, ddof=0)) if len(window_vals) > 1 else 0.0

    border = curr_ts - dt.timedelta(days=30)
    uniq_merch = len({r[1] for r in rows if r[2] >= border})

    return {
        "prev_amount": prev_amt,
        "amount_diff": amt_diff,
        "amount_ratio": amt_ratio,
        "roll_mean_amt_5": roll_mean,
        "roll_std_amt_5": roll_std,
        "unique_merch_last_30d": uniq_merch,
        "time_diff_h": time_diff_h,
    }
