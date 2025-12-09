import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator

# ---------------------------------------------------------------------------
#  Global constants – defined once, passed explicitly into every task
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent
TMP_ROOT = Path(tempfile.gettempdir()) / "user_embeddings"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

from prediction.src.config import get_settings
from prediction.src.gnn import EdgeGNNClassifier
from prediction.src.io_s3 import download_if_needed


def extract_transactions(tmp_root: str, **_):
    import os
    import uuid
    import warnings
    from pathlib import Path

    import pandas as pd
    import sqlalchemy as sa

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is required")

    engine = sa.create_engine(dsn.replace("+asyncpg", "+psycopg2"), future=True)

    df = pd.read_sql_query(
        """
        SELECT  u.cc_num      AS cc_num,
                t.amt         AS amt,
                t.merchant_id AS merchant_id,
                t.trans_time  AS trans_time
        FROM    transactions t
        JOIN    users u ON u.id = t.user_id
        ORDER   BY t.trans_time
        """,
        engine,
    )

    if df.empty:
        warnings.warn("transactions table is empty – embeddings will be zeroed")

    out = Path(tmp_root) / f"transactions_{uuid.uuid4().hex}.parquet"
    df.to_parquet(out)
    return str(out)


def build_graph_task(transactions_path: str, tmp_root: str):
    import uuid
    from pathlib import Path

    import orjson
    import pandas as pd

    df = pd.read_parquet(transactions_path, engine="pyarrow")

    def _build_graph(frame: pd.DataFrame):
        g = {}
        for _, row in frame.iterrows():
            bucket = g.setdefault(row.cc_num, {"neighbors": set(), "amounts": []})
            bucket["neighbors"].add(row.merchant_id)
            bucket["amounts"].append(float(row.amt))
        return g

    graph_jsonable = {
        k: {"neighbors": list(v["neighbors"]), "amounts": v["amounts"]}
        for k, v in _build_graph(df).items()
    }

    out = Path(tmp_root) / f"graph_{uuid.uuid4().hex}.json"
    out.write_bytes(orjson.dumps(graph_jsonable))
    return str(out)


def compute_embeddings_task(
    graph_path: str,
    tmp_root: str,
    get_settings,
    EdgeGNNClassifier,
    download_if_needed,
):
    import asyncio
    import json
    import uuid
    from pathlib import Path

    import numpy as np
    import orjson
    import torch

    g_json = orjson.loads(Path(graph_path).read_bytes())

    node_order = list(g_json.keys())
    n_nodes = len(node_order)
    if not n_nodes:
        print("Graph is empty, returning empty embeddings.")
        return json.dumps({"emb_path": "", "idx_path": ""})

    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(n_nodes, 5, device=device)

    edge_index = torch.arange(n_nodes, device=device).repeat(2, 1)
    edge_attr = torch.zeros(n_nodes, 69, device=device)

    settings = get_settings()
    state_file = asyncio.run(download_if_needed(settings.gnn_model_key))
    state_dict = torch.load(state_file, map_location=device)

    gnn = EdgeGNNClassifier(in_feats=5, edge_feats=69, hidden=64).to(device)
    gnn.load_state_dict(state_dict)
    gnn.eval()

    with torch.no_grad():
        h = torch.relu(gnn.conv1(x, edge_index, edge_attr))
        h = torch.relu(gnn.conv2(h, edge_index, edge_attr))

    matrix = h.cpu().numpy().astype(np.float32)
    emb_file = Path(tmp_root) / f"node_embeddings_{uuid.uuid4().hex}.npy"
    idx_file = Path(tmp_root) / f"cc2idx_{uuid.uuid4().hex}.json"

    np.save(emb_file, matrix)
    idx_file.write_text(
        orjson.dumps({uid: i for i, uid in enumerate(node_order)}).decode("UTF-8")
    )

    return json.dumps({"emb_path": str(emb_file), "idx_path": str(idx_file)})


def upload_embeddings_task(get_settings, artifacts, **ctx):
    import asyncio
    import json
    import sys
    from pathlib import Path

    import aioboto3

    if isinstance(artifacts, str):
        artifacts = json.loads(artifacts)
    if not artifacts or not artifacts["emb_path"]:
        return "Nothing to upload – graph was empty."

    settings = get_settings()
    session = aioboto3.Session()

    async def _upload(key: str, local: Path):
        async with session.client(
            "s3",
            endpoint_url=str(settings.s3_endpoint),
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        ) as s3:
            await s3.upload_file(str(local), settings.s3_bucket, key)

    asyncio.run(_upload(settings.node_embeddings_key, Path(artifacts["emb_path"])))
    asyncio.run(_upload(settings.cc2idx_key, Path(artifacts["idx_path"])))

    return (
        f"Uploaded {Path(artifacts['emb_path']).name} & "
        f"{Path(artifacts['idx_path']).name} → s3://{settings.s3_bucket}/"
    )


def _make_dag():
    default_args = {
        "owner": "fraud_shield",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    }

    with DAG(
        dag_id="recalculate_user_embeddings",
        description="Daily rebuild of user graph embeddings for the online predictor.",
        schedule_interval="@daily",
        start_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
        catchup=False,
        default_args=default_args,
        tags=["features", "embeddings"],
    ) as _dag:
        # 1) Extract
        t_extract = PythonVirtualenvOperator(
            task_id="extract_transactions",
            python_callable=extract_transactions,
            op_kwargs={"tmp_root": str(TMP_ROOT)},
            requirements=[
                "pandas>=2.2.2",
                "sqlalchemy>=2.0",
                "psycopg2-binary>=2.9",
                "pyarrow>=16.1",
            ],
            system_site_packages=False,
            do_xcom_push=True,
            provide_context=True,
        )

        # 2) Graph
        t_graph = PythonVirtualenvOperator(
            task_id="build_graph_task",
            python_callable=build_graph_task,
            op_kwargs={
                "transactions_path": "{{ ti.xcom_pull(task_ids='extract_transactions') }}",
                "tmp_root": str(TMP_ROOT),
            },
            requirements=[
                "pandas>=2.2.2",
                "orjson>=3.10",
                "pyarrow>=16.1",
                "pydantic-settings",
            ],
            system_site_packages=False,
            do_xcom_push=True,
        )

        # 3) Embedding
        t_embed = PythonVirtualenvOperator(
            task_id="compute_embeddings_task",
            python_callable=compute_embeddings_task,
            op_kwargs={
                "graph_path": "{{ ti.xcom_pull(task_ids='build_graph_task') }}",
                "tmp_root": str(TMP_ROOT),
                "get_settings": get_settings,
                "EdgeGNNClassifier": EdgeGNNClassifier,
                "download_if_needed": download_if_needed,
            },
            requirements=[
                "torch==2.2.0",
                "numpy==1.26.4",
                "orjson>=3.10",
                "aioboto3>=12.0.0",
                "pydantic-settings",
                "fastapi[all]",
                "torch_geometric",
                "catboost",
                "scikit-learn==1.6.1",
                "scipy<=1.15.3",
                "joblib",
                "pydantic",
                "pydantic-settings",
                "aioboto3",
                "sqlalchemy[asyncio]",
                "asyncpg",
                "python-dateutil",
                "category_encoders",
                "aio-pika",
                "pymon",
            ],
            system_site_packages=False,
            do_xcom_push=True,
            provide_context=True,
        )

        # 4) Upload
        t_upload = PythonVirtualenvOperator(
            task_id="upload_embeddings_task",
            python_callable=upload_embeddings_task,
            op_kwargs={
                "get_settings": get_settings,
                "artifacts": "{{ ti.xcom_pull(task_ids='compute_embeddings_task') }}",
            },
            requirements=[
                "aioboto3>=12.0.0",
                "pydantic",
                "pydantic-settings",
            ],
            system_site_packages=False,
            provide_context=True,
        )

        t_extract >> t_graph >> t_embed >> t_upload

    return _dag


dag = _make_dag()
