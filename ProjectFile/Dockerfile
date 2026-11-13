FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_METAL=OFF" \
    CMAKE_BUILD_PARALLEL_LEVEL=2
RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RAG_DATA_DIR=/app/data \
    RAG_MODEL_DIR=/app/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2
WORKDIR /app
COPY --chown=app:app app.py ./
COPY --chown=app:app rag ./rag
COPY --chown=app:app assignments ./assignments
COPY --chown=app:app .streamlit ./.streamlit
RUN mkdir -p /app/data /app/models && chown -R app:app /app
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "rag.server", "--server.address=0.0.0.0", "--server.port=8501"]
