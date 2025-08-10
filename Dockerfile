FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# ensure artifacts dirs exist
RUN mkdir -p artifacts/models artifacts/logs artifacts/db

EXPOSE 5000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5000"]
