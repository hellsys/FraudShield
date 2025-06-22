# Система обнаружения мошеннических транзакций

Этот репозиторий содержит **демо-проект** из двух микросервисов, показывающий полный
цикл онлайн-скоринга финансовых транзакций — от приёма события до выдачи
вероятности мошенничества (fraud probability) и визуализации результата.

* **business-service** — REST API для приёма транзакций, хранения истории и RPC-вызова
  скоринга;
* **prediction-service** — воркер, подготавливающий признаки и вычисляющий
  вероятность мошенничества с помощью ансамбля (GNN + CatBoost + LogReg);
* **PostgreSQL**, **RabbitMQ** и **MinIO** — поддерживающая инфраструктура;
* `ui.py` — небольшой интерфейс на Streamlit для анализа результатов.

Архитектура (упрощённо):

```
┌──────────────┐       HTTP            ┌────────────────┐      RPC (RabbitMQ)      ┌──────────────────┐
│   Клиенты    │ ───────────────────▶  │  Business API  │ ───────────────────────▶ │ Prediction-worker│
└──────────────┘  (FastAPI :8080)      │   + Postgres   │   (predict.request)      │   + ML-модели    │
                                       └────────────────┘  ◀────────────────────── │  + artefacts S3  │
                                               ▲                                   └──────────────────┘
                                               │
                                       Streamlit UI
```

## Быстрый старт

Требуется установленный **Docker + Docker Compose v2**.

```bash
cp .env.example .env      # при необходимости отредактируйте
docker compose up --build # запускаем всю систему
```

После старта сервисы будут доступны:

* Business API          — http://localhost:8080/docs
* RabbitMQ UI           — http://localhost:15673 (guest / guest)
* MinIO Console         — http://localhost:9001  (minioadmin / minioadmin)
* Streamlit UI          — http://localhost:8501 (отдельно `streamlit run ui.py`)

> Файлы моделей можно положить в каталог `bootstrap/` — при поднятии контейнера
> MinIO они автоматически загрузятся в бакет, указанный в `.env` (`S3_BUCKET`).

## Business-service

Каталог: `business/`

* Стэк: FastAPI, SQLAlchemy 2 (async), aio-pika.
* Основные эндпоинты:
  * `POST /transactions` — приём транзакции, расчёт history-features, RPC-скоринг;
  * `GET  /transactions` — лента последних транзакций;
  * `GET  /transactions/{id}`;
  * `GET  /users/{id}`;
  * `GET  /merchants/{id}`.
* Логика создания транзакции:
  1. UPSERT пользователя и магазина;
  2. расчёт агрегатов по истории (`history_feats.py`);
  3. RPC-вызов `predict.request` и ожидание `predict.response`;
  4. сохранение `fraud_prob` и ответ клиенту.

## Prediction-service

Каталог: `prediction/`

* Воркер читает очередь `predict.request` с префетчем (`PREFETCH`).
* Формирование признаков:
  * instant-фичи (`initial_feats`),
  * history-фичи (передаются из business-service),
  * табличные фичи через сериализованный `FraudDataPreprocessor`,
  * node-embedding `cc_num` (или OOV-вектор).
* Модель: `EdgeGNN → CatBoost → LogisticRegression` (blending).
* Артефакты скачиваются из MinIO и кэшируются в `/tmp/models`.
* Результат вида `{ "transaction_id": 123, "probability": 0.87 }` кладётся в
  `predict.response`.

## Тесты


### Покрытие тестами

Каталог `tests/` содержит **интеграционные** сценарии «чёрного ящика»,
которые эмулируют реальную работу микросервисов поверх поднятого окружения.

| Файл                         | Что проверяем | Зависимости |
|------------------------------|---------------|-------------|
| `test_api_system.py`         | 1) создание транзакции через `POST /transactions`; 2) получение этой же записи через `GET /transactions/{id}`; 3) корректность поля `fraud_prob` (0‒1). | Business-service (+ Postgres & RabbitMQ & Prediction-service) |
| `test_predict_system.py`     | Отправка сообщения напрямую в очередь `predict.request` и получение ответа из временной `reply_to` — проверяем, что воркер отдаёт `probability` в диапазоне 0‒1. | Prediction-service + RabbitMQ |

Системные тесты используют `pytest` и **асинхронные клиенты** (`httpx`, `aio-pika`).
В `tests/.env.pytest` определены значения `RABBIT_URL` и `API_BASE_URL`, которые
загружаются через `python-dotenv` в `conftest.py`.

### Как запустить

1. Запустите инфраструктуру (Postgres, RabbitMQ, MinIO) и оба сервиса командой:

   ```bash
   docker compose up --build -d  # либо без -d, если нужно видеть логи
   ```

2. Убедитесь, что контейнеры здоровы (`docker compose ps`).
3. Выполните тесты из корня проекта:

   ```bash
   pytest -q
   ```

При необходимости можно запустить **часть** тестов:

```bash
# только API-тест
pytest tests/test_api_system.py -q

# только prediction-worker
pytest tests/test_predict_system.py -q
```

> Обратите внимание: для успешного прохождения `test_api_system.py` сервис
> `prediction` должен быть запущен, так как Business-service ждёт ответа из
> RabbitMQ прежде чем вернуть результат HTTP-клиенту.

## Структура репозитория

```
.
├── business/        # FastAPI-сервис и бизнес-логика
│   └── src/
├── prediction/      # ML-воркер
│   └── src/
├── bootstrap/       # (опц.) артефакты для MinIO
├── tests/           # pytest-тесты
├── ui.py            # Streamlit-дашборд
├── docker-compose.yml
└── .env.example     # пример переменных окружения
```

## Переменные окружения

| Группа     | Переменная                  | Описание                           |
|------------|-----------------------------|------------------------------------|
| Postgres   | POSTGRES_DB/USER/PASSWORD   | учётные данные и имя БД            |
| RabbitMQ   | RABBIT_URL, PREFETCH        | AMQP-URL и размер prefetch         |
| MinIO/S3   | S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET | S3-доступ |
| Модели     | GNN_MODEL_KEY, CATBOOST_MODEL_KEY, … | ключи файлов моделей |

Полный список — в `.env.example`.

## Локальный запуск без Docker

```bash
# 1. Установить зависимости
python -m venv venv && . venv/bin/activate
pip install -r business/requirements.txt -r prediction/requirements.txt

# 2. Запустить Postgres и RabbitMQ локально
export DATABASE_URL=postgresql+asyncpg://frauduser:fraudpass@localhost:5432/fraud
export RABBIT_URL=amqp://guest:guest@localhost:5672/

# 3. Business-service (FastAPI)
uvicorn business.src.main:app --reload --port 8000

# 4. Prediction-service (воркер)
python -m prediction.src.server
```
