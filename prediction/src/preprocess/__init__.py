import datetime as dt
import math
from typing import Any, Dict

from dateutil import parser as dtparser
from dateutil import tz

from .feature_names import NUMERIC_EDGE_ORDER


def _ts(x: str | dt.datetime) -> dt.datetime:
    return x if isinstance(x, dt.datetime) else dtparser.isoparse(x)


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def calc_age(dob: str, ref: dt.datetime) -> int:
    d = dtparser.isoparse(dob).date()
    y = ref.year - d.year
    if (ref.month, ref.day) < (d.month, d.day):
        y -= 1
    return y


def make_edge_features(tx: Dict[str, Any]) -> list[float]:
    return [tx[f] for f in NUMERIC_EDGE_ORDER]


def initial_feats(tx: Dict[str, Any]) -> Dict[str, Any]:
    t = _ts(tx["trans_date_trans_time"]).astimezone(tz.UTC)

    return {
        "hour": t.hour,
        "dayofweek": t.strftime("%A"),
        "month": t.month,
        "distance_km": haversine(
            tx["lat"], tx["long"], tx["merch_lat"], tx["merch_long"]
        ),
        "age": calc_age(tx["dob"], t),
        "is_night": int(t.hour < 6 or t.hour > 20),
        "is_business_hour": int(9 <= t.hour <= 18),
        "is_weekend": int(t.weekday() >= 5),
        "gender": 1 if tx["gender"] == "F" else 0,
    }
