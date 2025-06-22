from __future__ import annotations

import json
import math
from logging import getLogger
from time import perf_counter
from typing import Any, Final

import pandas as pd
import torch

from .model_registry import ModelRegistry
from .preprocess import initial_feats, make_edge_features

logger = getLogger(__name__)


class Predictor:
    def __init__(self, registry: ModelRegistry) -> None:
        self._models = registry

    # ------------------------------------------------------------------ #
    async def predict(self, raw_tx: bytes) -> dict[str, Any]:
        """JSON-байты → вероятность фрода."""
        tx: Final = json.loads(raw_tx)
        tx_id: str = tx["transaction_id"]
        tic = perf_counter()

        # -------- 1. instant-фичи (быстрые) -----------------------------
        tx |= initial_feats(tx)  # city_pop, age, distance, …

        # -------- 2. табличные фичи через FraudDataPreprocessor --------
        df_row = pd.DataFrame([tx])
        X_full = self._models["preproc"].transform(df_row)[0]
        col_order = (
            self._models["preproc"].numeric_node_feats
            + self._models["preproc"].numeric_edge_feats
            + self._models["preproc"].low_ohe_cols
            + self._models["preproc"].high_cardinal_feats
            + self._models["preproc"].as_is_cols
        )
        X_full_dict = pd.Series(X_full, index=col_order).to_dict()

        # -------- 3. node-embedding либо OOV ---------------------------
        idx = self._models["cc2idx"].get(tx["cc_num"])
        h_u = (
            self._models["node_embeddings"][idx]
            if idx is not None
            else self._models["oov_vector"]
        )
        h_u = torch.from_numpy(h_u)

        # -------- 4. GNN-ветка + CatBoost + LR-бленд -------------------
        edge_attr = torch.tensor(make_edge_features(X_full_dict), dtype=torch.float64)

        with torch.no_grad():
            edge_vec = torch.cat([h_u, edge_attr]).unsqueeze(0).double()
            logit = self._models["gnn"](edge_vec)[0]
            gnn_p = torch.sigmoid(logit).item()

            cat_p = self._models["catboost"].predict_proba([X_full])[0, 1]
            final_p = self._models["lr"].predict_proba([[gnn_p, cat_p]])[0, 1]

        logger.info(
            "Tx %s | GNN: %.4f | Cat: %.4f | Final: %.4f | %.2f ms",
            tx_id,
            gnn_p,
            cat_p,
            final_p,
            (perf_counter() - tic) * 1_000,
        )

        return {"transaction_id": tx_id, "probability": final_p}
