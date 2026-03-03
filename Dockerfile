FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies for document parsing (pymupdf needs libmupdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application source
COPY src/ src/
COPY scripts/ scripts/

# Re-install in editable mode now that source is present
RUN pip install --no-cache-dir -e ".[all]"

# Create data and log directories
RUN mkdir -p data logs eval/datasets eval/results

EXPOSE 8000

CMD ["python", "-m", "researchforge", "serve"]
