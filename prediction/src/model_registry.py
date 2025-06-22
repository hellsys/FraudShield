"""Единая точка загрузки и хранения ML-артефактов.

Позволяет:
* лениво скачивать и кешировать файлы;
* использовать DI в тестах (можно подменить ModelRegistry двойником).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import torch

from .config import get_settings
from .gnn import EdgeGNNClassifier
from .io_s3 import batch_download
from .preprocess.preprocessor import FraudDataPreprocessor

settings = get_settings()

sys.modules["__main__"].FraudDataPreprocessor = FraudDataPreprocessor


class ModelRegistry:  # noqa: D101
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    @classmethod
    async def create(cls) -> "ModelRegistry":
        """Фабричный метод (async-ctor) — загружает все модели единым батчем."""
        if cls._instance:
            return cls._instance

        self = cls()
        keys = [
            settings.gnn_model_key,
            settings.catboost_model_key,
            settings.lr_model_key,
            settings.cc2idx_key,
            settings.node_scaler_key,
            settings.edge_scaler_key,
            settings.preproc_key,
            settings.node_embeddings_key,
        ]
        files: Final[dict[str, Path]] = await batch_download(keys)

        # ───────── GNN ──────────────────────────────────────────────────
        state_dict = torch.load(files[settings.gnn_model_key], map_location="cpu")
        gnn = EdgeGNNClassifier(in_feats=5, edge_feats=69, hidden=64)
        gnn.load_state_dict(state_dict)
        gnn = gnn.edge_mlp.double().eval()

        # ───────── Сторонние модели ─────────────────────────────────────
        catboost_model = joblib.load(files[settings.catboost_model_key])
        lr_model = joblib.load(files[settings.lr_model_key])

        # ───────── Препроцессор ─────────────────────────────────────────
        preproc: FraudDataPreprocessor = joblib.load(files[settings.preproc_key])
        preproc.as_is_cols = [
            c for c in preproc.as_is_cols if c not in ("cc_num", "merchant")
        ]

        # ───────── Сопутствующие файлы ──────────────────────────────────
        node_embeddings: np.ndarray = np.load(files[settings.node_embeddings_key])
        cc2idx: dict[str, int] = json.load(open(files[settings.cc2idx_key]))

        # ───────── Сохраняем в словарь ──────────────────────────────────
        self._models = {
            "gnn": gnn,
            "catboost": catboost_model,
            "lr": lr_model,
            "node_embeddings": node_embeddings,
            "cc2idx": cc2idx,
            "node_scaler": joblib.load(files[settings.node_scaler_key]),
            "edge_scaler": joblib.load(files[settings.edge_scaler_key]),
            "preproc": preproc,
            "oov_vector": node_embeddings.mean(axis=0),
        }

        cls._instance = self
        return self

    def __getitem__(self, item: str) -> Any:  # noqa: D401
        return self._models[item]

    async def reload(self) -> None:
        """Скачать модели заново — пригодится при hot-reload в dev-режиме."""
        new_instance = await ModelRegistry.create()
        self._models = new_instance._models
