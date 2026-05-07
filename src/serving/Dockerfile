FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[ml,serving,agent,monitoring,security]" && \
    pip install --no-cache-dir boto3 s3fs langchain-ollama

COPY src/ src/
COPY configs/ configs/

ENV PYTHONPATH=/app
ENV APP_ENV=production

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
