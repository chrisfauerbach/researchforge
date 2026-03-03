#!/bin/bash
# Pull all required Ollama models.
# Run with: docker compose exec ollama bash /scripts/setup_models.sh
set -e

echo "Pulling ResearchForge models..."

MODELS=(
    "deepseek-r1:14b"
    "qwen2.5:14b"
    "qwen2.5:7b"
    "deepseek-r1:7b"
    "mistral-nemo:12b"
    "nomic-embed-text"
)

for model in "${MODELS[@]}"; do
    echo "--- Pulling $model ---"
    ollama pull "$model"
done

echo ""
echo "All models pulled successfully."
ollama list
