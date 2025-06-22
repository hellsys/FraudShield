from logging import getLogger
from typing import List, Tuple, Union

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = getLogger(__name__)
TARGET_FRAC = 0.001


class FraudDataPreprocessor:
    def __init__(
        self,
        graph_features: bool = False,
        random_state: int = 42,
        gnn_prepare: bool = False,
    ):
        self.graph_features = graph_features
        self.random_state = random_state
        self.gnn_prepare = gnn_prepare

        self.numeric_node_feats: List[str] = []
        self.numeric_edge_feats: List[str] = []
        self.low_cardinal_feats: List[str] = []
        self.high_cardinal_feats: List[str] = []
        self.as_is_cols: List[str] = []
        self.low_ohe_cols: List[str] = []

        self.node_scaler: StandardScaler = None
        self.edge_scaler: StandardScaler = None

        self.ohe: OneHotEncoder = None
        self.target_encoder: ce.TargetEncoder = None

    # ---------- FIT ----------
    def fit(self, df: pd.DataFrame, y: pd.Series, n_splits: int = 5):
        # --- 1. определяем списки фич ---
        all_numeric = [
            "amt",
            "city_pop",
            "distance_km",
            "age",
            "time_diff_h",
            "prev_amount",
            "amount_diff",
            "amount_ratio",
            "roll_mean_amt_5",
            "roll_std_amt_5",
            "unique_merch_last_30d",
        ]

        #  узловые фичи (city_pop & age + граф-метрики)
        self.numeric_node_feats = ["city_pop", "age"]
        if self.graph_features:
            self.numeric_node_feats += ["c_deg", "c_comm_size", "m_deg", "m_comm_size"]

        # всё остальное → edge-numeric
        self.numeric_edge_feats = [
            f for f in all_numeric if f not in self.numeric_node_feats
        ]

        self.as_is_cols = [
            "gender",
            "is_weekend",
            "is_business_hour",
            "is_night",
        ]

        # категориальные
        self.low_cardinal_feats = ["category", "dayofweek", "hour", "month"]
        self.high_cardinal_feats = ["job", "state"]
        if self.graph_features:
            self.high_cardinal_feats += ["c_comm", "m_comm"]
            self.as_is_cols += [c for c in df.columns if c.startswith("emb_")]
        if self.gnn_prepare:
            self.as_is_cols += ["cc_num", "merchant"]

        # --- 2. обучаем скейлеры ---
        self.node_scaler = StandardScaler().fit(df[self.numeric_node_feats])
        self.edge_scaler = StandardScaler().fit(df[self.numeric_edge_feats])

        # --- 3. кодеры для категорий ---
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(df[self.low_cardinal_feats])
        self.low_ohe_cols = self.ohe.get_feature_names_out(
            self.low_cardinal_feats
        ).tolist()

        self.target_encoder = ce.TargetEncoder(
            cols=self.high_cardinal_feats, smoothing=0.4
        ).fit(df[self.high_cardinal_feats], y)

        return self

    # ---------- TRANSFORM ----------
    def transform(
        self, df: pd.DataFrame
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if any(
            obj is None
            for obj in (
                self.node_scaler,
                self.edge_scaler,
                self.ohe,
                self.target_encoder,
            )
        ):
            raise RuntimeError("Run .fit() first")

        # узловые и рёберные численные отдельно
        X_node = self.node_scaler.transform(df[self.numeric_node_feats])
        X_edge = self.edge_scaler.transform(df[self.numeric_edge_feats])

        X_low = self.ohe.transform(df[self.low_cardinal_feats])
        X_high = self.target_encoder.transform(df[self.high_cardinal_feats]).values
        X_as = df[self.as_is_cols].to_numpy() if self.as_is_cols else None

        arrays = [X_node, X_edge, X_low, X_high]

        if X_as is not None:
            arrays.append(X_as)

        return np.hstack(arrays)

    # ---------- FIT-TRANSFORM (не менялся) ----------
    def fit_transform(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        y = df["is_fraud"]
        X = df.drop(columns=["is_fraud"])
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        self.fit(X_train, y_train)

        # кросс-валидационный target-encoding для train-части (как было)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        X_train_high_oof = pd.DataFrame(
            index=X_train.index, columns=self.high_cardinal_feats, dtype=float
        )
        for col in self.high_cardinal_feats:
            for tr_idx, val_idx in kf.split(X_train):
                te = ce.TargetEncoder(cols=[col], smoothing=0.4)
                te.fit(X_train.iloc[tr_idx][[col]], y_train.iloc[tr_idx])
                X_train_high_oof.iloc[
                    val_idx, X_train_high_oof.columns.get_loc(col)
                ] = te.transform(X_train.iloc[val_idx][[col]])[col]

        # формируем окончательные матрицы
        X_train_proc = np.hstack(
            [
                self.node_scaler.transform(X_train[self.numeric_node_feats]),
                self.edge_scaler.transform(X_train[self.numeric_edge_feats]),
                self.ohe.transform(X_train[self.low_cardinal_feats]),
                X_train_high_oof.values,
                (
                    X_train[self.as_is_cols].to_numpy()
                    if self.as_is_cols
                    else np.empty((len(X_train), 0))
                ),
            ]
        )

        X_test_proc = self.transform(X_val)

        # балансировка ↓ — логика без изменений
        pos_mask = y_train.values == 1
        n_pos = pos_mask.sum()
        n_neg_needed = int(np.floor(n_pos * (1 - TARGET_FRAC) / TARGET_FRAC))
        neg_mask = ~pos_mask

        if neg_mask.sum() > n_neg_needed:
            rng = np.random.RandomState(self.random_state)
            neg_indices = np.where(neg_mask)[0]
            sampled_neg = rng.choice(neg_indices, size=n_neg_needed, replace=False)
            keep = np.concatenate([np.where(pos_mask)[0], sampled_neg])
            X_train_proc = X_train_proc[keep]
            y_train = y_train.iloc[keep]

        return X_train_proc, X_test_proc, y_train, y_val
