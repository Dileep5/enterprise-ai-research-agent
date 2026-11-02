FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA wheels.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Bake the embedding and reranker models into the image so containers start offline.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"

COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY eval ./eval

RUN useradd --create-home appuser && mkdir -p /app/storage && chown -R appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
