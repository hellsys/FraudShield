import os
import urllib.parse
from datetime import date, datetime

import httpx
import pandas as pd
import pydeck as pdk
import streamlit as st

# общие настройки
API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="Fraud Monitor", layout="wide")
PAGES = ["Транзакции", "Профиль пользователя", "Профиль магазина"]

# URL-параметры
raw_page = st.query_params.get("page", "Транзакции")
page = urllib.parse.unquote(raw_page)
if page not in PAGES:
    page = "Транзакции"

user_id_param = st.query_params.get("user_id")
merch_id_param = st.query_params.get("merch_id")
try:
    user_id_param = int(user_id_param) if user_id_param is not None else None
except ValueError:
    user_id_param = None
try:
    merch_id_param = int(merch_id_param) if merch_id_param is not None else None
except ValueError:
    merch_id_param = None

# навигация
page = st.sidebar.radio("Раздел", PAGES, index=PAGES.index(page))


# helpers
@st.cache_data(ttl=60)
def fetch_json(endpoint: str):
    r = httpx.get(f"{API}{endpoint}", timeout=10.0)
    r.raise_for_status()
    return r.json()


def compute_age(dob_str: str) -> int | None:
    try:
        dob = datetime.fromisoformat(dob_str).date()
        t = date.today()
        return t.year - dob.year - ((t.month, t.day) < (dob.month, dob.day))
    except Exception:
        return None


def scatter_text_layers(df, color, text_color=(0, 0, 0, 255)):
    return [
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lon, lat]",
            get_radius=200,
            get_color=color,
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            data=df,
            get_position="[lon, lat]",
            get_text="label",
            get_size=16,
            get_color=text_color,
            get_alignment_baseline='"bottom"',
        ),
    ]


# 1. Транзакции
if page == "Транзакции":
    st.header("Все транзакции")
    c1, c2 = st.columns(2)
    with c1:
        fraud_only = st.checkbox("Только подозрительные", value=True)
    with c2:
        limit = st.slider("Кол-во записей", 10, 500, 100, 10)

    txs = fetch_json(f"/transactions?fraud_only={fraud_only}&limit={limit}")
    df = pd.json_normalize(txs)
    st.dataframe(df, height=400, use_container_width=True)

    if sel := st.selectbox("Выберите transaction_id", [None] + df.id.tolist()):
        tx = next(t for t in txs if t["id"] == sel)
        st.subheader(f"Транзакция #{sel}")
        st.json(tx)

        coords = pd.DataFrame(
            [
                {"lat": tx["user"]["lat"], "lon": tx["user"]["long"], "label": "User"},
                {
                    "lat": tx["merchant"]["merch_lat"],
                    "lon": tx["merchant"]["merch_long"],
                    "label": "Merchant",
                },
            ]
        )
        center = coords[["lat", "lon"]].mean()
        deck = pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=center.lat, longitude=center.lon, zoom=11
            ),
            layers=scatter_text_layers(coords, color=[100, 100, 255]),
            map_style="mapbox://styles/mapbox/light-v11",
        )
        st.subheader("Расположение")
        st.pydeck_chart(deck)

        st.markdown(
            f"[👤 Профиль пользователя](/?page={urllib.parse.quote('Профиль пользователя')}"
            f"&user_id={tx['user']['id']}) &nbsp;|&nbsp; "
            f"[🏬 Профиль магазина](/?page={urllib.parse.quote('Профиль магазина')}"
            f"&merch_id={tx['merchant']['id']})"
        )

# 2. Профиль пользователя
elif page == "Профиль пользователя":
    st.header("Профиль пользователя")
    user_id = st.number_input("User ID", 1, value=user_id_param or 1, step=1)
    if st.button("Показать", key="show_user"):
        u = fetch_json(f"/users/{user_id}")
        u["age"] = u.get("age") or compute_age(u.get("dob", ""))
        st.markdown(f"**{u.get('first', '')} {u.get('last', '')}**  (ID {u['id']})")
        st.write(
            {
                "cc_num": u.get("cc_num"),
                "Пол": u.get("gender"),
                "Город, штат": f"{u.get('city')}, {u.get('state')}",
                "Профессия": u.get("job"),
                "Возраст": u.get("age"),
            }
        )

        if u.get("lat") and u.get("long"):
            df_loc = pd.DataFrame(
                [{"lat": u["lat"], "lon": u["long"], "label": "User"}]
            )
            deck = pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=u["lat"], longitude=u["long"], zoom=12
                ),
                layers=scatter_text_layers(df_loc, color=[255, 0, 0]),
                map_style="mapbox://styles/mapbox/light-v11",
            )
            st.subheader("Локация")
            st.pydeck_chart(deck)

        all_txs = fetch_json("/transactions?fraud_only=false")
        user_txs = u.get("transactions") or []
        st.subheader("Последние транзакции")
        if user_txs:
            st.dataframe(pd.json_normalize(user_txs), use_container_width=True)
        else:
            st.info("Транзакции не найдены")

# 3. Профиль магазина
elif page == "Профиль магазина":
    st.header("Профиль магазина")
    merch_id = st.number_input("Merchant ID", 1, value=merch_id_param or 1, step=1)
    if st.button("Показать", key="show_merch"):
        m = fetch_json(f"/merchants/{merch_id}")
        st.markdown(f"**{m.get('name')}**  (ID {m['id']})")
        st.write({"Категория": m.get("category")})

        if m.get("merch_lat") and m.get("merch_long"):
            df_loc = pd.DataFrame(
                [{"lat": m["merch_lat"], "lon": m["merch_long"], "label": "Merchant"}]
            )
            deck = pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=m["merch_lat"], longitude=m["merch_long"], zoom=12
                ),
                layers=scatter_text_layers(df_loc, color=[0, 200, 100]),
                map_style="mapbox://styles/mapbox/light-v11",
            )
            st.subheader("Локация")
            st.pydeck_chart(deck)

        all_txs = fetch_json("/transactions?fraud_only=false")
        merch_txs = m.get("transactions") or []
        st.subheader("Последние транзакции")
        if merch_txs:
            st.dataframe(pd.json_normalize(merch_txs), use_container_width=True)
        else:
            st.info("Транзакции не найдены")
