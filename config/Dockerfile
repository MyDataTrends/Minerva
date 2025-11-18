# Base image for builder
FROM python:3.12-bullseye AS builder

# Set environment variables
ENV DOCKER_BUILDKIT=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/usr/local/bin:/root/.local/bin:$PATH" \
    PYTHONUSERBASE="/root/.local"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    curl \
    make \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && $HOME/.cargo/bin/rustup default stable \
    && $HOME/.cargo/bin/rustup update stable \
    && rm -rf /var/lib/apt/lists/* \
    && gcc --version

# Install conda and mamba
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && ~/miniconda/bin/conda init \
    && ~/miniconda/bin/conda config --set always_yes yes --set changeps1 no \
    && ~/miniconda/bin/conda update -q conda \
    && ~/miniconda/bin/conda install -c conda-forge mamba \
    && ~/miniconda/bin/conda info -a

# Set environment variables for conda
ENV PATH=~/miniconda/bin:$PATH

# Copy environment.yml for caching
COPY environment.yml .

# Install dependencies using mamba
RUN mamba env create -f environment.yml

# Set working directory
WORKDIR /app

# Copy application code
COPY . .
RUN pip install requests flask

# Final stage for production
FROM python:3.12-bullseye

# Set environment variables for production
ENV PYTHONPATH=/app \
    AWS_DEFAULT_REGION=us-east-1 \
    PATH="/usr/local/bin:/root/.local/bin:/usr/local/lib/python3.9/site-packages:$PATH" \
    PYTHONUSERBASE="/root/.local"

# Install build dependencies again for the production image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && gcc --version

# Install conda and mamba
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && ~/miniconda/bin/conda init \
    && ~/miniconda/bin/conda config --set always_yes yes --set changeps1 no \
    && ~/miniconda/bin/conda update -q conda \
    && ~/miniconda/bin/conda install -c conda-forge mamba \
    && ~/miniconda/bin/conda info -a

# Set environment variables for conda
ENV PATH=~/miniconda/bin:$PATH

# Set working directory
WORKDIR /app

# Copy the app code and the installed dependencies from the builder stage
COPY --from=builder /app /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/miniconda /root/miniconda

# Expose the port for Streamlit and metrics/health endpoints
EXPOSE 8501 8000

# Add healthcheck for /healthz
HEALTHCHECK CMD curl -f http://localhost:8000/healthz || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "ui/dashboard.py", "--server.port=8501", "--server.headless=true"]