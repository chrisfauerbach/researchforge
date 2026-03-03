FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies for document parsing (pymupdf needs libmupdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source and config
COPY pyproject.toml ./
COPY src/ src/
COPY scripts/ scripts/
COPY eval/ eval/

# Install package with all extras
RUN pip install --no-cache-dir ".[all]"

# Create data and log directories
RUN mkdir -p data logs eval/datasets eval/results

EXPOSE 8000

CMD ["python", "-m", "researchforge", "serve"]
