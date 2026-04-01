# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ---- Labels ---------------------------------------------------------------
LABEL maintainer="vishweshreddy007"
LABEL description="Kitchen Env — OpenEnv RL Environment for Restaurant Kitchen Management"
LABEL version="1.0.0"

# ---- Default environment variables ----------------------------------------
# Users can override these via HF Space secrets.
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---- System dependencies ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory -----------------------------------------------------
WORKDIR /app

# ---- Install Python dependencies -------------------------------------------
# Copy requirements first to leverage Docker layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----------------------------------------------------
COPY . .

# ---- Port ------------------------------------------------------------------
# Required by Hugging Face Spaces.
EXPOSE 7860

# ---- Health check ----------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ---- Start server ----------------------------------------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
