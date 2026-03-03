#!/bin/bash
# Wrapper for Claude Desktop MCP integration.
# Set the path in Claude Desktop's MCP config to this script.
# The -T flag disables pseudo-TTY (required for stdio transport).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Use local venv if available, otherwise fall back to Docker
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    exec "$SCRIPT_DIR/.venv/bin/python" -m researchforge mcp
else
    exec docker compose -f "$SCRIPT_DIR/docker-compose.yml" exec -T app python -m researchforge mcp
fi
