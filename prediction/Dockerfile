FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Сборка wheel-пакета не нужна — копируем исходники
COPY ./src ./prediction
CMD ["python", "-m", "prediction.server"]