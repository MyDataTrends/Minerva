# Stage 1: builder — install all dependencies
FROM python:3.12-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git gcc g++ cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid the 3 GB CUDA download
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install all remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: runtime — lean image with pre-installed packages
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Create runtime directories and non-root user
RUN useradd -m -u 1000 assay \
    && mkdir -p local_data logs User_Data models output_files metadata mcp_data mcp_temp \
    && chown -R assay:assay /app

USER assay

EXPOSE 8000 8501 8766

# Default: run FastAPI + embedded MCP server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
